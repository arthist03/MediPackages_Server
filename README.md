# MediPackages OCR Server (Gemini Vision v4)

Fast, accurate medical document OCR powered by Google Gemini 2.0 Flash.

## Architecture

```
Image Upload
    ↓
Image Validation                   — size check, format validation
    ↓
Gemini Vision Extraction           — single API call → structured JSON
    ↓
Validation Agent (Pydantic)        — confidence scoring
    ↓
Human Review                       — POST /feedback to approve/reject
    ↓
Package Matching Agent             — keyword search against MAA YOJANA database
    ↓
FastAPI Response → Flutter App
```

**LLM:** Google Gemini 2.0 Flash (fast, accurate, free tier available)
**Framework:** LangGraph (SQLite checkpointing, interrupt/resume HitL)
**Memory:** SQLite — rejection history + approved examples

---

## Benefits of Gemini

| Metric   | Old (Local Qwen 3B)       | New (Gemini 2.0 Flash) |
| -------- | ------------------------- | ---------------------- |
| Speed    | 30-60 seconds             | **2-5 seconds**        |
| Accuracy | ~70% handwriting          | **~95% handwriting**   |
| Setup    | Complex (llama.cpp, VRAM) | **Simple (API key)**   |
| Cost     | Free (local GPU)          | Free tier (15 req/min) |

---

## Setup

## Production Deployment

Use these defaults for production hosting (including PythonAnywhere):

- `APP_ENV=production`
- `ENABLE_DOCS=false`
- `CORS_ORIGINS=https://your-app-domain.com`
- `TRUSTED_HOSTS=your-api-domain.com`
- `API_AUTH_TOKEN=<long-random-secret>`
- At least one LLM key: `GEMINI_API_KEY` or `GROQ_API_KEY`

Recommended process command:

```bash
gunicorn -k uvicorn.workers.UvicornWorker -w 2 -b 0.0.0.0:8000 main:app
```

Notes:

- Keep `/docs` and `/openapi.json` disabled in production unless needed temporarily.
- Restrict CORS to trusted frontends; avoid `*` in production.
- Set `TRUSTED_HOSTS` to real domains to block host-header abuse.
- Keep `SERVER_WORKERS` aligned with available CPU/RAM on your host.

### 1. Get Gemini API Key (FREE)

1. Go to https://aistudio.google.com/apikey
2. Click "Create API key"
3. Copy the key

### 2. Install dependencies

```powershell
cd server
pip install -r requirements.txt
```

### 3. Configure environment

Edit `.env` and add your API key:

```
GEMINI_API_KEY=your_api_key_here
```

### 4. Start server

```powershell
cd server
python -m uvicorn main:app --host 0.0.0.0 --port 8000
```

Or use the start script:

```powershell
python start.py
```

---

## API Reference

### `POST /extract`

Upload a medical document image. Returns structured JSON.

**Request:** `multipart/form-data` with field `image` (JPEG/PNG, max 20MB)

**Response (pending_review):**

```json
{
  "success": true,
  "status": "pending_review",
  "session_id": "abc-123",
  "message": "AI Doctor summary ready — please verify.",
  "preview": {
    "patient_name": "...",
    "diagnosis": "...",
    "medications": [...]
  },
  "validation": {
    "confidence": 0.85,
    "issues": []
  },
  "processing_time_seconds": 3.2
}
```

### `POST /feedback`

Resume a paused pipeline with human decision.

```json
{
  "session_id": "abc-123",
  "decision": "approved",
  "corrections": { "patient_name": "Corrected Name" }
}
```

**Response (after approval):**

```json
{
  "success": true,
  "status": "completed",
  "data": {
    "patient_name": "...",
    "diagnosis": "...",
    "best_packages": [
      {
        "package_code": "2847-BM001B",
        "package_name": "...",
        "rate": 40000,
        "alignment_score": 85
      }
    ]
  }
}
```

### `GET /health`

Returns server and Gemini API status.

```json
{
  "status": "healthy",
  "mode": "Gemini Vision OCR v4",
  "pipeline_ready": true,
  "gemini_reachable": true
}
```

### `GET /stats`

Returns rejection patterns and approval history from long-term memory.

---

## Human-in-the-Loop Flow

1. `/extract` ALWAYS returns `pending_review` with extracted data preview
2. Flutter shows the preview to the operator for verification
3. Operator calls `POST /feedback` with `approved` or `rejected`
4. If approved → package matching runs → final result returned
5. If rejected → stored in memory for future learning

---

## Configuration (`.env`)

| Key              | Default            | Description            |
| ---------------- | ------------------ | ---------------------- |
| `GEMINI_API_KEY` | (required)         | Your Gemini API key    |
| `GEMINI_MODEL`   | `gemini-2.0-flash` | Model to use           |
| `LLM_TIMEOUT`    | `60`               | API timeout (seconds)  |
| `TOP_K_PACKAGES` | `5`                | Package search results |
| `SERVER_PORT`    | `8000`             | Server port            |

---

## Extracted Data Schema

```json
{
  "patient_name": "Full name",
  "patient_age": 45,
  "patient_gender": "Male/Female",
  "date": "DD-MM-YYYY",
  "doctor_name": "Dr. ...",
  "clinic_name": "Hospital name",
  "department": "Cardiology/Orthopedics/etc.",
  "chief_complaints": ["symptoms..."],
  "diagnosis": "Primary diagnosis (expanded)",
  "secondary_diagnoses": ["comorbidities"],
  "surgery_required": true/false,
  "surgery_name": "if required",
  "lab_tests_ordered": true/false,
  "lab_tests": ["CBC", "HbA1c", ...],
  "medications": [
    {"name": "Drug", "dose": "10mg", "frequency": "twice daily", "duration": "7 days"}
  ],
  "vitals": {"bp": "120/80", "pulse": "72"},
  "follow_up_date": "next review date"
}
```

---

## Rate Limits (Gemini Free Tier)

- 15 requests per minute
- 1 million tokens per day
- 1,500 requests per day

For higher limits, upgrade to Gemini API paid tier.
