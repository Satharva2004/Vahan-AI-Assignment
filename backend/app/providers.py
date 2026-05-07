from __future__ import annotations

import base64
import os
import time
from dataclasses import dataclass
from typing import Any

import httpx
from dotenv import load_dotenv


load_dotenv()


@dataclass(frozen=True)
class ModelSpec:
    id: str
    provider: str
    label: str
    model: str
    language: str | None = None
    mode: str | None = None
    rationale: str = ""


MODEL_SPECS = [
    ModelSpec(
        id="deepgram-nova-3",
        provider="deepgram",
        label="Deepgram Nova-3",
        model="nova-3",
        rationale="Current Deepgram default recommendation for high-accuracy prerecorded ASR.",
    ),
    ModelSpec(
        id="deepgram-nova-2",
        provider="deepgram",
        label="Deepgram Nova-2",
        model="nova-2",
        rationale="Useful older Deepgram baseline to see whether Nova-3 improves your domain.",
    ),
    ModelSpec(
        id="deepgram-base",
        provider="deepgram",
        label="Deepgram Base",
        model="base",
        rationale="Low-cost baseline; helps quantify accuracy/cost tradeoffs.",
    ),
    ModelSpec(
        id="sarvam-saaras-v3-transcribe",
        provider="sarvam",
        label="Sarvam Saaras v3",
        model="saaras:v3",
        mode="transcribe",
        rationale="Sarvam's newer Indic-focused ASR with automatic language detection.",
    ),
    ModelSpec(
        id="sarvam-saaras-v3-codemix",
        provider="sarvam",
        label="Sarvam Saaras v3 Codemix",
        model="saaras:v3",
        mode="codemix",
        rationale="Important for Indian speech where locality names and English words are code-mixed.",
    ),
    ModelSpec(
        id="sarvam-saarika-v25",
        provider="sarvam",
        label="Sarvam Saarika v2.5",
        model="saarika:v2.5",
        rationale="Legacy Sarvam baseline to validate migration impact.",
    ),
    ModelSpec(
        id="openai-gpt-4o-transcribe",
        provider="openai",
        label="OpenAI GPT-4o Transcribe",
        model="gpt-4o-transcribe",
        rationale="Optional third-party baseline with strong general ASR performance.",
    ),
    ModelSpec(
        id="openai-gpt-4o-mini-transcribe",
        provider="openai",
        label="OpenAI GPT-4o Mini Transcribe",
        model="gpt-4o-mini-transcribe",
        rationale="Optional lower-cost general ASR baseline.",
    ),
    ModelSpec(
        id="assemblyai-best",
        provider="assemblyai",
        label="AssemblyAI Best",
        model="best",
        rationale="API-based general ASR baseline with hosted inference.",
    ),
    ModelSpec(
        id="assemblyai-slam-1",
        provider="assemblyai",
        label="AssemblyAI Slam-1",
        model="slam-1",
        rationale="AssemblyAI's newer speech model when enabled for the account.",
    ),
    ModelSpec(
        id="google-stt-long",
        provider="google",
        label="Google STT Long",
        model="long",
        language="hi-IN",
        rationale="Google Cloud Speech baseline for Hindi/Indian speech.",
    ),
    ModelSpec(
        id="google-stt-telephony",
        provider="google",
        label="Google STT Telephony",
        model="telephony",
        language="hi-IN",
        rationale="Google Cloud Speech model for phone-call style audio.",
    ),
]


class TranscriptionError(Exception):
    pass


def get_model(model_id: str) -> ModelSpec:
    for spec in MODEL_SPECS:
        if spec.id == model_id:
            return spec
    raise TranscriptionError(f"Unknown model id: {model_id}")


async def transcribe_audio(
    spec: ModelSpec,
    content: bytes,
    filename: str,
    content_type: str,
    api_keys: dict[str, str],
) -> dict[str, Any]:
    started = time.perf_counter()
    normalized_content_type = _normalize_content_type(content_type, filename)
    resolved_keys = {
        "deepgram": api_keys.get("deepgram") or os.getenv("DEEPGRAM_API_KEY", ""),
        "sarvam": api_keys.get("sarvam") or os.getenv("SARVAM_API_KEY", ""),
        "openai": api_keys.get("openai") or os.getenv("OPENAI_API_KEY", ""),
        "assemblyai": api_keys.get("assemblyai") or os.getenv("ASSEMBLYAI_API_KEY", ""),
        "google": api_keys.get("google") or os.getenv("GOOGLE_API_KEY", ""),
    }
    try:
        if spec.provider == "deepgram":
            transcript, raw = await _deepgram(spec, content, normalized_content_type, resolved_keys["deepgram"])
        elif spec.provider == "sarvam":
            transcript, raw = await _sarvam(spec, content, filename, normalized_content_type, resolved_keys["sarvam"])
        elif spec.provider == "openai":
            transcript, raw = await _openai(spec, content, filename, normalized_content_type, resolved_keys["openai"])
        elif spec.provider == "assemblyai":
            transcript, raw = await _assemblyai(spec, content, normalized_content_type, resolved_keys["assemblyai"])
        elif spec.provider == "google":
            transcript, raw = await _google_stt(spec, content, normalized_content_type, resolved_keys["google"])
        else:
            raise TranscriptionError(f"Unsupported provider: {spec.provider}")
        return {
            "ok": True,
            "transcript": transcript,
            "latency_ms": round((time.perf_counter() - started) * 1000),
            "raw": raw,
        }
    except Exception as exc:
        return {
            "ok": False,
            "transcript": "",
            "latency_ms": round((time.perf_counter() - started) * 1000),
            "error": str(exc),
        }


async def _deepgram(
    spec: ModelSpec,
    content: bytes,
    content_type: str,
    api_key: str,
) -> tuple[str, dict[str, Any]]:
    if not api_key:
        raise TranscriptionError("Missing Deepgram API key")
    params = {"model": spec.model, "smart_format": "true", "punctuate": "true"}
    headers = {"Authorization": f"Token {api_key}", "Content-Type": content_type or "application/octet-stream"}
    async with httpx.AsyncClient(timeout=180) as client:
        response = await client.post("https://api.deepgram.com/v1/listen", params=params, headers=headers, content=content)
    _raise_provider_error(response, "Deepgram")
    data = response.json()
    transcript = (
        data.get("results", {})
        .get("channels", [{}])[0]
        .get("alternatives", [{}])[0]
        .get("transcript", "")
    )
    return transcript, data


async def _sarvam(
    spec: ModelSpec,
    content: bytes,
    filename: str,
    content_type: str,
    api_key: str,
) -> tuple[str, dict[str, Any]]:
    if not api_key:
        raise TranscriptionError("Missing Sarvam API key")
    files = {"file": (filename, content, content_type or "application/octet-stream")}
    data: dict[str, str] = {"model": spec.model}
    if spec.language:
        data["language_code"] = spec.language
    if spec.mode:
        data["mode"] = spec.mode
    headers = {"api-subscription-key": api_key}
    async with httpx.AsyncClient(timeout=180) as client:
        response = await client.post("https://api.sarvam.ai/speech-to-text", headers=headers, data=data, files=files)
    _raise_provider_error(response, "Sarvam")
    payload = response.json()
    return payload.get("transcript", ""), payload


async def _openai(
    spec: ModelSpec,
    content: bytes,
    filename: str,
    content_type: str,
    api_key: str,
) -> tuple[str, dict[str, Any] | str]:
    if not api_key:
        raise TranscriptionError("Missing OpenAI API key")
    files = {"file": (filename, content, content_type or "application/octet-stream")}
    data = {"model": spec.model, "response_format": "json"}
    headers = {"Authorization": f"Bearer {api_key}"}
    async with httpx.AsyncClient(timeout=180) as client:
        response = await client.post("https://api.openai.com/v1/audio/transcriptions", headers=headers, data=data, files=files)
    _raise_provider_error(response, "OpenAI")
    payload = response.json()
    return payload.get("text", ""), payload


async def _assemblyai(
    spec: ModelSpec,
    content: bytes,
    content_type: str,
    api_key: str,
) -> tuple[str, dict[str, Any]]:
    if not api_key:
        raise TranscriptionError("Missing AssemblyAI API key")
    headers = {"Authorization": api_key}
    async with httpx.AsyncClient(timeout=240) as client:
        upload = await client.post(
            "https://api.assemblyai.com/v2/upload",
            headers={**headers, "Content-Type": content_type or "application/octet-stream"},
            content=content,
        )
        _raise_provider_error(upload, "AssemblyAI upload")
        audio_url = upload.json().get("upload_url")
        if not audio_url:
            raise TranscriptionError("AssemblyAI upload did not return upload_url")

        payload: dict[str, Any] = {"audio_url": audio_url, "language_detection": True, "format_text": True}
        if spec.model:
            payload["speech_model"] = spec.model
        submitted = await client.post("https://api.assemblyai.com/v2/transcript", headers=headers, json=payload)
        _raise_provider_error(submitted, "AssemblyAI transcript")
        transcript_id = submitted.json()["id"]

        for _ in range(90):
            poll = await client.get(f"https://api.assemblyai.com/v2/transcript/{transcript_id}", headers=headers)
            _raise_provider_error(poll, "AssemblyAI transcript")
            data = poll.json()
            if data.get("status") == "completed":
                return data.get("text", ""), data
            if data.get("status") == "error":
                raise TranscriptionError(f"AssemblyAI error: {data.get('error')}")
            await _sleep(2)
    raise TranscriptionError("AssemblyAI transcription timed out")


async def _google_stt(
    spec: ModelSpec,
    content: bytes,
    content_type: str,
    api_key: str,
) -> tuple[str, dict[str, Any]]:
    if not api_key:
        raise TranscriptionError("Missing Google API key")
    config: dict[str, Any] = {
        "languageCode": spec.language or "hi-IN",
        "model": spec.model,
        "enableAutomaticPunctuation": True,
        "alternativeLanguageCodes": ["en-IN", "kn-IN"],
    }
    encoding = _google_encoding(content_type)
    if encoding:
        config["encoding"] = encoding
    if content_type in {"audio/webm", "video/webm"}:
        config["sampleRateHertz"] = 48000

    payload = {
        "config": config,
        "audio": {"content": base64.b64encode(content).decode("ascii")},
    }
    async with httpx.AsyncClient(timeout=180) as client:
        response = await client.post(
            "https://speech.googleapis.com/v1/speech:recognize",
            params={"key": api_key},
            json=payload,
        )
    _raise_provider_error(response, "Google STT")
    data = response.json()
    transcript = " ".join(
        result.get("alternatives", [{}])[0].get("transcript", "")
        for result in data.get("results", [])
    ).strip()
    return transcript, data


def _raise_provider_error(response: httpx.Response, provider: str) -> None:
    try:
        response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        detail = response.text[:500]
        raise TranscriptionError(f"{provider} {response.status_code}: {detail}") from exc


def _normalize_content_type(content_type: str, filename: str) -> str:
    base = (content_type or "").split(";", 1)[0].strip().lower()
    if base:
        return base

    lower_name = filename.lower()
    if lower_name.endswith(".webm"):
        return "audio/webm"
    if lower_name.endswith(".wav"):
        return "audio/wav"
    if lower_name.endswith(".mp3"):
        return "audio/mpeg"
    if lower_name.endswith(".m4a"):
        return "audio/x-m4a"
    if lower_name.endswith(".mp4"):
        return "audio/mp4"
    if lower_name.endswith(".ogg") or lower_name.endswith(".opus"):
        return "audio/ogg"
    if lower_name.endswith(".flac"):
        return "audio/flac"
    return "application/octet-stream"


def _google_encoding(content_type: str) -> str | None:
    if content_type in {"audio/webm", "video/webm"}:
        return "WEBM_OPUS"
    if content_type in {"audio/wav", "audio/x-wav", "audio/wave"}:
        return "LINEAR16"
    if content_type in {"audio/mpeg", "audio/mp3", "audio/x-mp3"}:
        return "MP3"
    if content_type in {"audio/flac", "audio/x-flac"}:
        return "FLAC"
    if content_type in {"audio/ogg", "audio/opus"}:
        return "OGG_OPUS"
    return None


async def _sleep(seconds: float) -> None:
    import asyncio

    await asyncio.sleep(seconds)
