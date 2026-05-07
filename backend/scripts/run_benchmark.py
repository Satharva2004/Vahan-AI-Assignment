from __future__ import annotations

import argparse
import asyncio
import csv
import json
import mimetypes
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.metrics import evaluate
from app.providers import get_model, transcribe_audio


DEFAULT_MODELS = ["deepgram-nova-3", "sarvam-saaras-v3-transcribe", "assemblyai-best"]


async def run(manifest_path: Path, output_dir: Path, models: list[str]) -> None:
    rows = list(csv.DictReader(manifest_path.open(encoding="utf-8")))
    base_dir = manifest_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    detailed = []
    summary_rows = []
    for row in rows:
        audio_path = (base_dir / row["file"]).resolve()
        content = audio_path.read_bytes()
        content_type = mimetypes.guess_type(audio_path.name)[0] or "application/octet-stream"
        reference = row["reference"]
        entities = row.get("entities", "")

        for model_id in models:
            spec = get_model(model_id)
            transcription = await transcribe_audio(spec, content, audio_path.name, content_type, {})
            metrics = asdict(evaluate(reference, transcription["transcript"], entities)) if transcription["ok"] else None
            record = {
                "file": row["file"],
                "locality": row.get("locality", ""),
                "language": row.get("language", ""),
                "condition": row.get("condition", "Unlabeled"),
                "reference": reference,
                "model_id": spec.id,
                "model": spec.label,
                "ok": transcription["ok"],
                "latency_ms": transcription["latency_ms"],
                "transcript": transcription["transcript"],
                "error": transcription.get("error", ""),
                "metrics": metrics,
            }
            detailed.append(record)
            summary_rows.append(
                {
                    "file": record["file"],
                    "condition": record["condition"],
                    "locality": record["locality"],
                    "model": record["model"],
                    "ok": record["ok"],
                    "wer": metrics["wer"] if metrics else "",
                    "cer": metrics["cer"] if metrics else "",
                    "entity_recall": metrics["entity_recall"] if metrics else "",
                    "entity_f1": metrics["entity_f1"] if metrics else "",
                    "latency_ms": record["latency_ms"],
                    "missed_entities": "; ".join(metrics["missed_entities"]) if metrics else "",
                    "transcript": record["transcript"],
                    "error": record["error"],
                }
            )

    (output_dir / "results.json").write_text(json.dumps(detailed, indent=2, ensure_ascii=False), encoding="utf-8")
    with (output_dir / "results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    aggregates = []
    for model_id in models:
        model_records = [item for item in detailed if item["model_id"] == model_id and item["metrics"]]
        spec = get_model(model_id)
        aggregates.append(
            {
                "model": spec.label,
                "recordings": len(model_records),
                "wer": avg(model_records, "wer"),
                "cer": avg(model_records, "cer"),
                "entity_recall": avg(model_records, "entity_recall"),
                "entity_f1": avg(model_records, "entity_f1"),
                "latency_ms": round(sum(item["latency_ms"] for item in model_records) / len(model_records)) if model_records else None,
                "failures": len(rows) - len(model_records),
            }
        )
    (output_dir / "summary.json").write_text(json.dumps(aggregates, indent=2, ensure_ascii=False), encoding="utf-8")


def avg(records: list[dict], key: str) -> float | None:
    if not records:
        return None
    return round(sum(item["metrics"][key] for item in records) / len(records), 4)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ASR benchmark from a manifest CSV.")
    parser.add_argument("--manifest", default="../data/manifest.csv", help="Path to manifest CSV.")
    parser.add_argument("--output", default="../data/results", help="Output directory.")
    parser.add_argument("--models", nargs="*", default=DEFAULT_MODELS, help="Model ids to run.")
    args = parser.parse_args()
    asyncio.run(run(Path(args.manifest).resolve(), Path(args.output).resolve(), args.models))


if __name__ == "__main__":
    main()
