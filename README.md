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

Then update `frontend/.env` to `NEXT_PUBLIC_API_BASE_URL=http://localhost:8010`.

## Frontend

```powershell
cd frontend
npm install
npm run dev
```

Frontend runs at http://localhost:3000.

Frontend API target is controlled from one file: `frontend/.env`.

```env
NEXT_PUBLIC_API_BASE_URL=https://vahan-ai-assignment-74ig.vercel.app
NEXT_PUBLIC_API_ROUTE_PREFIX=
```

For local backend testing, change `NEXT_PUBLIC_API_BASE_URL` to `http://localhost:8000`. If a backend is mounted under a route prefix such as `/_/backend`, set `NEXT_PUBLIC_API_ROUTE_PREFIX=/_/backend`.

## CLI Benchmark

Run the reproducible benchmark:

```powershell
cd backend
.\.venv\Scripts\Activate.ps1
python scripts\run_benchmark.py --manifest ..\data\manifest.csv --output ..\data\results --models deepgram-nova-3 sarvam-saaras-v3-transcribe assemblyai-best
```

Run the full Voice Notes Excel benchmark:

```powershell
cd backend
.\.venv\Scripts\Activate.ps1
python scripts\run_benchmark.py --manifest "..\data\Voice Notes.xlsx" --output ..\data\results --frontend-public ..\frontend\public
```

Outputs:

- `data/results/results.csv`
- `data/results/results.json`
- `data/results/summary.json`
- `frontend/public/benchmark-results.json`

## Voice Notes Benchmark

The current saved benchmark uses `data/Voice Notes.xlsx`: 32 Cloudinary OGG recordings with ground truth, language, condition, and entity labels. It benchmarks 11 ASR configurations across Deepgram, Sarvam, OpenAI, AssemblyAI, and Google STT.

Top aggregate results by WER:

| Rank | Model | WER | Accuracy | Exact Match | Entity Recall | Hallucination | RTF | Failures |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | Sarvam Saarika v2.5 | 16.44% | 83.56% | 46.67% | 76.67% | 2.83% | 0.172x | 2 |
| 2 | Sarvam Saaras v3 Codemix | 17.86% | 82.14% | 36.67% | 65.00% | 2.30% | 0.175x | 2 |
| 3 | OpenAI GPT-4o Mini Transcribe | 18.35% | 81.65% | 34.38% | 60.44% | 2.61% | 0.248x | 0 |
| 4 | Sarvam Saaras v3 | 18.69% | 81.31% | 33.33% | 65.00% | 2.67% | 0.183x | 2 |
| 5 | Deepgram Nova-3 | 24.16% | 75.84% | 15.62% | 35.66% | 4.24% | 0.334x | 0 |

Interpretation: Sarvam leads on WER and entity recall on the short recordings, but its synchronous API rejects two files longer than 30 seconds. OpenAI GPT-4o Mini is the strongest zero-failure challenger. Deepgram Nova-3 remains reliable but misses more locality entities in this dataset.

## Report Notes

The final report should be max 3 pages and include:

- Model selection rationale
- Dataset and condition breakdown
- Metrics and why they matter
- Failure analysis by condition, model, and locality
- Recommendation against the Deepgram baseline
- Limitations and one surprising insight
