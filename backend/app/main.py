import json
import os
import wave
from dataclasses import asdict
from io import BytesIO
from typing import Annotated, Any

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import httpx

from .metrics import evaluate
from .providers import MODEL_SPECS, get_model, transcribe_audio

app = FastAPI(
    title="VahanAI API",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "https://vahan-ai-assignment.vercel.app",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def read_root() -> dict[str, str]:
    return {"message": "VahanAI backend is running"}


@app.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/models")
def list_models() -> dict[str, list[dict]]:
    return {"models": [asdict(model) for model in MODEL_SPECS]}


@app.post("/benchmark")
async def benchmark(
    files: Annotated[list[UploadFile], File()],
    references: Annotated[str, Form()],
    model_ids: Annotated[str, Form()],
    api_keys: Annotated[str, Form()] = "{}",
    entities: Annotated[str, Form()] = "[]",
    conditions: Annotated[str, Form()] = "[]",
    durations: Annotated[str, Form()] = "[]",
) -> dict:
    parsed_references: list[str] = json.loads(references)
    parsed_entities: list[str] = json.loads(entities)
    parsed_conditions: list[str] = json.loads(conditions) if conditions else []
    parsed_durations: list[float | None] = json.loads(durations) if durations else []
    selected_models: list[str] = json.loads(model_ids)
    keys: dict[str, str] = json.loads(api_keys) if api_keys else {}
    selected_models = list(dict.fromkeys(selected_models))

    if len(parsed_references) != len(files):
        return {"ok": False, "error": "Each audio file needs one reference transcript."}
    if parsed_conditions and len(parsed_conditions) != len(files):
        return {"ok": False, "error": "Condition labels must match the uploaded files."}
    if parsed_durations and len(parsed_durations) != len(files):
        return {"ok": False, "error": "Audio durations must match the uploaded files."}
    if not selected_models:
        return {"ok": False, "error": "Select at least one ASR model."}

    items = []
    for index, file in enumerate(files):
        content = await file.read()
        items.append(
            {
                "content": content,
                "file_name": file.filename or f"recording-{index + 1}.wav",
                "content_type": file.content_type or "application/octet-stream",
                "reference": parsed_references[index],
                "entities": parsed_entities[index],
                "condition": parsed_conditions[index] if parsed_conditions else "Unlabeled",
                "duration_seconds": parsed_durations[index] if parsed_durations else None,
            }
        )

    return await _run_benchmark(items, selected_models, keys)


@app.post("/benchmark-urls")
async def benchmark_urls(payload: dict[str, Any]) -> dict:
    selected_models = list(dict.fromkeys(payload.get("model_ids") or []))
    keys: dict[str, str] = payload.get("api_keys") or {}
    samples = payload.get("samples") or []
    if not selected_models:
        return {"ok": False, "error": "Select at least one ASR model."}
    if not samples:
        return {"ok": False, "error": "Select at least one sample URL."}

    items = []
    async with httpx.AsyncClient(timeout=180, follow_redirects=True) as client:
        for index, sample in enumerate(samples):
            url = sample.get("audio_url", "")
            if not url:
                return {"ok": False, "error": "Each sample needs audio_url."}
            response = await client.get(url)
            try:
                response.raise_for_status()
            except httpx.HTTPStatusError as exc:
                return {"ok": False, "error": f"Could not fetch {url}: {response.status_code} {response.text[:200]}"}
            items.append(
                {
                    "content": response.content,
                    "file_name": sample.get("file_name") or f"url-sample-{index + 1}.wav",
                    "content_type": response.headers.get("content-type", "audio/wav").split(";", 1)[0],
                    "reference": sample.get("reference", ""),
                    "entities": sample.get("entities", ""),
                    "condition": sample.get("condition", "External sample"),
                    "duration_seconds": sample.get("duration_seconds"),
                    "metadata": sample.get("metadata", {}),
                }
            )

    return await _run_benchmark(items, selected_models, keys)


async def _run_benchmark(items: list[dict], selected_models: list[str], keys: dict[str, str]) -> dict:
    runs = []
    for item in items:
        recording_results = []
        duration_seconds = item.get("duration_seconds") or _audio_duration_seconds(
            item["content"],
            item["content_type"],
            item["file_name"],
        )
        for model_id in selected_models:
            spec = get_model(model_id)
            transcription = await transcribe_audio(
                spec,
                item["content"],
                item["file_name"],
                item["content_type"],
                keys,
            )
            reference = item.get("reference", "")
            metrics = (
                asdict(evaluate(reference, transcription["transcript"], item.get("entities", "")))
                if transcription["ok"] and reference.strip()
                else None
            )
            rtf = _real_time_factor(transcription["latency_ms"], duration_seconds) if transcription["ok"] else None
            recording_results.append(
                {
                    "model_id": spec.id,
                    "provider": spec.provider,
                    "label": spec.label,
                    "transcript": transcription["transcript"],
                    "latency_ms": transcription["latency_ms"],
                    "rtf": rtf,
                    "realtime_status": _realtime_status(rtf),
                    "ok": transcription["ok"],
                    "error": transcription.get("error"),
                    "metrics": metrics,
                }
            )
        runs.append(
            {
                "file_name": item["file_name"],
                "reference": item.get("reference", ""),
                "ground_truth": item.get("reference", ""),
                "entities": item.get("entities", ""),
                "condition": item.get("condition", "Unlabeled"),
                "duration_seconds": duration_seconds,
                "metadata": item.get("metadata", {}),
                "results": recording_results,
            }
        )

    aggregate = []
    for model_id in selected_models:
        ok_results = [
            result
            for run in runs
            for result in run["results"]
            if result["model_id"] == model_id and result["ok"]
        ]
        values = [result for result in ok_results if result["metrics"]]
        spec = get_model(model_id)
        count = len(ok_results)
        aggregate.append(
            {
                "model_id": spec.id,
                "provider": spec.provider,
                "label": spec.label,
                "recordings": count,
                "wer": _avg(values, "wer"),
                "cer": _avg(values, "cer"),
                "entity_f1": _avg(values, "entity_f1"),
                "entity_recall": _avg(values, "entity_recall"),
                "hallucination_rate": _avg(values, "hallucination_rate"),
                "similarity": _avg(values, "similarity"),
                "latency_ms": round(sum(item["latency_ms"] for item in ok_results) / count) if count else None,
                "rtf": _avg_result(ok_results, "rtf"),
                "failures": len(items) - count,
            }
        )

    return {
        "ok": True,
        "runs": runs,
        "aggregate": sorted(aggregate, key=lambda item: (item["wer"] is None, item["wer"] or 999)),
        "failure_analysis": _failure_analysis(runs),
        "methodology": {
            "primary": "WER ranks broad transcription accuracy; CER protects against spelling drift in locality names and short utterances.",
            "entity": "Entity recall/F1 isolates the phrases that matter most in address, place, and operational workflows.",
            "latency": "Latency is reported because an ASR that is slightly less accurate may still be better for interactive use.",
            "rtf": "Real-time factor is processing time divided by audio duration: below 1 is faster than real time, near 1 is real time, above 1 is slower.",
            "hallucination": "Hallucination rate tracks inserted model words that were not supported by the ground truth.",
            "semantic_warning": "WER is lexical, not semantic: translation, transliteration, script changes, or equivalent wording can make a correct model output look wrong.",
            "limitations": "Scores depend on only the uploaded sample, exact references, audio quality, and provider normalization choices.",
        },
    }


def _avg(values: list[dict], key: str) -> float | None:
    if not values:
        return None
    return round(sum(item["metrics"][key] for item in values) / len(values), 4)


def _avg_result(values: list[dict], key: str) -> float | None:
    present = [item[key] for item in values if item.get(key) is not None]
    if not present:
        return None
    return round(sum(present) / len(present), 3)


def _real_time_factor(latency_ms: int | float | None, duration_seconds: float | None) -> float | None:
    if not latency_ms or not duration_seconds or duration_seconds <= 0:
        return None
    return round((latency_ms / 1000) / duration_seconds, 3)


def _realtime_status(rtf: float | None) -> str | None:
    if rtf is None:
        return None
    if rtf < 0.95:
        return "faster-than-real-time"
    if rtf <= 1.05:
        return "real-time"
    return "slower-than-real-time"


def _audio_duration_seconds(content: bytes, content_type: str, file_name: str) -> float | None:
    normalized = (content_type or "").split(";", 1)[0].lower()
    if normalized in {"audio/wav", "audio/x-wav", "audio/wave"} or file_name.lower().endswith(".wav"):
        try:
            with wave.open(BytesIO(content), "rb") as audio:
                frames = audio.getnframes()
                rate = audio.getframerate()
                return round(frames / rate, 3) if rate else None
        except wave.Error:
            return None
    return None


def _failure_analysis(runs: list[dict]) -> dict:
    by_condition: dict[str, dict] = {}
    by_model: dict[str, dict] = {}
    by_entity: dict[str, int] = {}

    for run in runs:
        condition = run.get("condition") or "Unlabeled"
        condition_bucket = by_condition.setdefault(
            condition,
            {"total": 0, "failed_models": 0, "missed_entities": 0},
        )
        condition_bucket["total"] += 1

        for result in run["results"]:
            label = result["label"]
            model_bucket = by_model.setdefault(
                label,
                {"total": 0, "failures": 0, "missed_entities": 0, "wer_values": []},
            )
            model_bucket["total"] += 1
            if not result["ok"]:
                model_bucket["failures"] += 1
                condition_bucket["failed_models"] += 1
                continue

            metrics = result.get("metrics") or {}
            if metrics.get("wer") is not None:
                model_bucket["wer_values"].append(metrics["wer"])
            missed = metrics.get("missed_entities") or []
            model_bucket["missed_entities"] += len(missed)
            condition_bucket["missed_entities"] += len(missed)
            for entity in missed:
                by_entity[entity] = by_entity.get(entity, 0) + 1

    for bucket in by_model.values():
        values = bucket.pop("wer_values")
        bucket["avg_wer"] = round(sum(values) / len(values), 4) if values else None

    return {
        "by_condition": by_condition,
        "by_model": by_model,
        "by_entity": dict(sorted(by_entity.items(), key=lambda item: item[1], reverse=True)),
    }


@app.post("/analysis")
async def analyze_results(payload: dict[str, Any]) -> dict:
    fallback = _heuristic_analysis(payload)
    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        return {"ok": True, "source": "heuristic", "summary": fallback}

    prompt = (
        "You are evaluating ASR benchmark results for Indian conversational speech. "
        "Write a concise, human-readable Markdown summary. Keep it minimal. "
        "Use these exact sections: ## Recommendation, ## What the metrics mean, ## Key result, "
        "## Failure modes, ## Limitations. Explain WER, CER, hallucination rate, entity recall, latency, and real-time factor in simple terms. "
        "Be careful: WER is only lexical overlap between ground truth and model output. If the ground truth is in one language/script "
        "and the model output is a correct translation, transliteration, or semantically equivalent phrase in another language/script, "
        "say that WER may overstate the failure and that the transcript evidence should be inspected manually. "
        "Do not call a model worse solely from WER when this mismatch is visible. "
        "Avoid fluff and avoid long paragraphs.\n\n"
        f"Results JSON:\n{json.dumps(payload)[:12000]}"
    )
    try:
        async with httpx.AsyncClient(timeout=60) as client:
            response = await client.post(
                "https://api.openai.com/v1/chat/completions",
                headers={"Authorization": f"Bearer {api_key}"},
                json={
                    "model": os.getenv("OPENAI_ANALYSIS_MODEL", "gpt-4o-mini"),
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.2,
                },
            )
        response.raise_for_status()
        data = response.json()
        return {"ok": True, "source": "openai", "summary": data["choices"][0]["message"]["content"]}
    except Exception as exc:
        return {"ok": True, "source": "heuristic", "summary": f"{fallback}\n\nLLM analysis unavailable: {exc}"}


def _heuristic_analysis(payload: dict[str, Any]) -> str:
    aggregate = payload.get("aggregate") or []
    failures = payload.get("failure_analysis") or {}
    best = next((item for item in aggregate if not item.get("failures")), aggregate[0] if aggregate else None)
    worst_entities = list((failures.get("by_entity") or {}).items())[:5]
    condition_items = failures.get("by_condition") or {}
    risky_conditions = sorted(
        condition_items.items(),
        key=lambda item: item[1].get("missed_entities", 0) + item[1].get("failed_models", 0),
        reverse=True,
    )[:3]
    lines = [
        "## Recommendation",
        f"Use **{best['label']}** on this sample, but validate on all 20 recordings before finalizing." if best else "No successful model result yet.",
        "",
        "## What the metrics mean",
        "- **WER**: word error rate. Lower is better.",
        "- **CER**: character error rate. Lower catches spelling mistakes in names.",
        "- **Hallucination rate**: inserted words in model output that are not in the ground truth. Lower is better.",
        "- **Entity recall**: how often locality names were captured. Higher is better.",
        "- **Latency**: response time. Lower is better for calls.",
        "- **Real-time factor**: processing time divided by audio duration. Below 1 is faster than real time.",
        "- **Important**: WER can be unfair when the model output is a correct translation/transliteration or uses another script.",
        "",
        "## Key result",
        "Prioritize entity recall and transcript evidence over WER alone. If language/script differs but meaning is right, mark it as a WER limitation rather than a true ASR failure.",
    ]
    if worst_entities:
        lines.extend(["", "## Failure modes", "- Most missed entities: " + ", ".join(f"{name} ({count})" for name, count in worst_entities)])
    if risky_conditions:
        if "## Failure modes" not in lines:
            lines.extend(["", "## Failure modes"])
        lines.append("- Riskiest conditions: " + ", ".join(name for name, _ in risky_conditions))
    lines.extend(["", "## Limitations", "This summary is directional until the full 20-recording dataset is run with condition labels."])
    return "\n".join(lines)
