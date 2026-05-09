# VahanAI Backend

FastAPI backend for VahanAI.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Run

```powershell
uvicorn app.main:app --reload
```

API docs will be available at http://localhost:8000/docs.

If Windows blocks port `8000`, run on a different local port:

```powershell
uvicorn app.main:app --reload --host 127.0.0.1 --port 8010
```

## Public URL Samples

`POST /benchmark-urls` accepts public audio URLs, downloads them server-side, and benchmarks them with the selected models. This is used by the frontend sample library so the app does not need to store large WAV files locally.
