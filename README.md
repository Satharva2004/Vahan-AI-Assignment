# VahanAI ASR Benchmark

Tooling for the ASR Shootout intern assignment: record/upload Indian conversational speech, compare ASR providers against Deepgram Nova-3, and generate evidence for a concise report.

## What This Covers

- Deepgram Nova-3 baseline
- Sarvam models
- AssemblyAI models
- OpenAI transcription models
- Google Speech-to-Text models
- Browser recording and upload
- Public sample URLs from Voice of India/Josh Talks while the 20-recording set is still being collected
- Condition labels such as quiet, traffic, phone call, rushed, whispered
- WER, CER, entity recall/F1, latency, failure analysis
- Live per-model response timing
- CLI benchmark pipeline from a manifest CSV

## Dataset Workflow

Record 20 natural Bangalore locality utterances. Use varied conditions and clear filenames.

Example:

```text
data/
  manifest.csv
  recordings/
    01_koramangala_quiet.webm
    02_indiranagar_traffic.webm
```

Manifest format:

```csv
file,reference,locality,language,condition,entities
recordings/01_koramangala_quiet.webm,"haan main koramangala mein rehta hoon",Koramangala,Hinglish,Quiet,Koramangala
```

Use [data/manifest.example.csv](data/manifest.example.csv) as the template.

The web app also includes starter public samples:

- Hindi speaker stems from Voice of India
- Kannada speaker stems from Voice of India/Josh Talks

These WAV files are fetched by the backend from public Google Cloud Storage URLs. Add the real transcript in the UI when available; without a reference transcript, the app can still show provider output and latency, but WER/CER/entity scores will be unavailable for that sample.

## API Keys

Put keys in `backend/.env`:

```env
DEEPGRAM_API_KEY=...
SARVAM_API_KEY=...
ASSEMBLYAI_API_KEY=...
OPENAI_API_KEY=...
GOOGLE_SERVICE_ACCOUNT_JSON_B64=...
```

`backend/.env` is git-ignored.

For Google STT, keep the service account JSON in the env only. This project supports a base64-encoded JSON value through `GOOGLE_SERVICE_ACCOUNT_JSON_B64`.

## Backend

```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Backend runs at http://localhost:8000.

If port `8000` is blocked on Windows, use another port:

```powershell
uvicorn app.main:app --reload --host 127.0.0.1 --port 8010
```

Then set `NEXT_PUBLIC_API_BASE_URL=http://localhost:8010` for the frontend.

## Frontend

```powershell
cd frontend
npm install
npm run dev
```

Frontend runs at http://localhost:3000.

## CLI Benchmark

Run the reproducible benchmark:

```powershell
cd backend
.\.venv\Scripts\Activate.ps1
python scripts\run_benchmark.py --manifest ..\data\manifest.csv --output ..\data\results --models deepgram-nova-3 sarvam-saaras-v3-transcribe assemblyai-best
```

Outputs:

- `data/results/results.csv`
- `data/results/results.json`
- `data/results/summary.json`

## Report Notes

The final report should be max 3 pages and include:

- Model selection rationale
- Dataset and condition breakdown
- Metrics and why they matter
- Failure analysis by condition, model, and locality
- Recommendation against the Deepgram baseline
- Limitations and one surprising insight
