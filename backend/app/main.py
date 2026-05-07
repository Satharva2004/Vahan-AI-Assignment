import json
import os
from dataclasses import asdict
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
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
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
) -> dict:
    parsed_references: list[str] = json.loads(references)
    parsed_entities: list[str] = json.loads(entities)
    parsed_conditions: list[str] = json.loads(conditions) if conditions else []
    selected_models: list[str] = json.loads(model_ids)
    keys: dict[str, str] = json.loads(api_keys) if api_keys else {}
    selected_models = list(dict.fromkeys(selected_models))

    if len(parsed_references) != len(files):
        return {"ok": False, "error": "Each audio file needs one reference transcript."}
    if parsed_conditions and len(parsed_conditions) != len(files):
        return {"ok": False, "error": "Condition labels must match the uploaded files."}
    if not selected_models:
        return {"ok": False, "error": "Select at least one ASR model."}

    runs = []
    for index, file in enumerate(files):
        content = await file.read()
        recording_results = []
        for model_id in selected_models:
            spec = get_model(model_id)
            transcription = await transcribe_audio(
                spec,
                content,
                file.filename or f"recording-{index + 1}.wav",
                file.content_type or "application/octet-stream",
                keys,
            )
            metrics = (
                asdict(evaluate(parsed_references[index], transcription["transcript"], parsed_entities[index]))
                if transcription["ok"]
                else None
            )
            recording_results.append(
                {
                    "model_id": spec.id,
                    "provider": spec.provider,
                    "label": spec.label,
                    "transcript": transcription["transcript"],
                    "latency_ms": transcription["latency_ms"],
                    "ok": transcription["ok"],
                    "error": transcription.get("error"),
                    "metrics": metrics,
                }
            )
        runs.append(
            {
                "file_name": file.filename,
                "reference": parsed_references[index],
                "entities": parsed_entities[index],
                "condition": parsed_conditions[index] if parsed_conditions else "Unlabeled",
                "results": recording_results,
            }
        )

    aggregate = []
    for model_id in selected_models:
        values = [
            result
            for run in runs
            for result in run["results"]
            if result["model_id"] == model_id and result["ok"] and result["metrics"]
        ]
        spec = get_model(model_id)
        count = len(values)
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
                "similarity": _avg(values, "similarity"),
                "latency_ms": round(sum(item["latency_ms"] for item in values) / count) if count else None,
                "failures": len(files) - count,
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
            "limitations": "Scores depend on only the uploaded sample, exact references, audio quality, and provider normalization choices.",
        },
    }


def _avg(values: list[dict], key: str) -> float | None:
    if not values:
        return None
    return round(sum(item["metrics"][key] for item in values) / len(values), 4)


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
        "Write a concise, opinionated summary with recommendation, surprising insight candidates, "
        "failure modes, and limitations. Avoid fluff.\n\n"
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
        f"Recommendation: use {best['label']} on this sample, but validate on all 20 recordings before finalizing." if best else "Recommendation: no successful model result yet.",
        "Why: prioritize entity recall and transcript evidence over WER alone because locality names are the operational risk.",
    ]
    if worst_entities:
        lines.append("Most missed entities: " + ", ".join(f"{name} ({count})" for name, count in worst_entities))
    if risky_conditions:
        lines.append("Riskiest conditions: " + ", ".join(name for name, _ in risky_conditions))
    lines.append("Limitation: this summary is directional until the full 20-recording dataset is run with condition labels.")
    return "\n".join(lines)
