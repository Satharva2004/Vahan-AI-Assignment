# VahanAI

This repository contains separate backend and frontend applications.

## Structure

```text
VahanAI/
  backend/   FastAPI API service
  frontend/  Next.js web app
```

## Backend

```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Backend runs at http://localhost:8000.

## Frontend

```powershell
cd frontend
npm run dev
```

Frontend runs at http://localhost:3000.
