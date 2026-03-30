"""
config/settings.py  —  Central configuration for the agentic OCR pipeline.
All values are read from environment / .env at import time.
"""
from __future__ import annotations

import os
from pathlib import Path
from dotenv import load_dotenv

# ── Environment Detection ─────────────────────────────────────────────
IS_VERCEL = "VERCEL" in os.environ or "VERCEL_ENV" in os.environ

if not IS_VERCEL:
    # On Replit/Local, load from .env file
    load_dotenv(Path(__file__).parents[1] / ".env", override=True)
else:
    # On Vercel, environment variables are managed via the dashboard
    print("🚀 Running in Vercel Serverless environment")



def _parse_bool(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _parse_csv(value: str | None, default: list[str]) -> list[str]:
    if not value:
        return default
    items = [item.strip() for item in value.split(",") if item.strip()]
    return items or default


# ── Paths ─────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parents[1].resolve()
PACKAGES_JSON = Path(os.getenv("PACKAGES_JSON", str(
    BASE_DIR.parent / "assets" / "maa_packages.json")))

if IS_VERCEL:
    # Vercel filesystem is read-only; use /tmp for all write operations
    # Note: SQLite data will be lost when the serverless function spins down
    OUTPUT_DIR = Path("/tmp/output")
    MEMORY_DB = Path("/tmp/agent_memory.db")
else:
    OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", str(BASE_DIR / "output")))
    MEMORY_DB = Path(os.getenv("MEMORY_DB", str(
        BASE_DIR / "memory" / "agent_memory.db")))

# ── Gemini API (Primary LLM) ──────────────────────────────────────────
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
LLM_TIMEOUT = int(os.getenv("LLM_TIMEOUT", "60"))
LLM_MAX_RETRIES = int(os.getenv("LLM_MAX_RETRIES", "2"))
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.1"))
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "4096"))

# ── Groq API (Free fallback) ─────────────────────────────────────────
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv(
    "GROQ_MODEL", "meta-llama/llama-4-scout-17b-16e-instruct")

# ── Fallback: Local LLM (llama.cpp) ───────────────────────────────────
USE_LOCAL_LLM = os.getenv("USE_LOCAL_LLM", "false").lower() == "true"
LLAMA_SERVER_URL = os.getenv("LLAMA_SERVER_URL", "http://localhost:8080/v1")
LLAMA_MODEL_NAME = os.getenv("LLAMA_MODEL_NAME", "local-model")

# ── OCR (EasyOCR) - Optional fallback ─────────────────────────────────
USE_EASYOCR = os.getenv("USE_EASYOCR", "false").lower() == "true"
OCR_LANGUAGES = os.getenv("OCR_LANGUAGES", "en").split(",")
OCR_GPU = os.getenv("OCR_GPU", "false").lower() == "true"

# ── Package matching ──────────────────────────────────────────────────
TOP_K_PACKAGES = int(os.getenv("TOP_K_PACKAGES", "5"))

# ── Server ────────────────────────────────────────────────────────────
SERVER_HOST = os.getenv("SERVER_HOST", "0.0.0.0")
# Replit sets PORT automatically; fall back to SERVER_PORT then 8000
SERVER_PORT = int(os.getenv("PORT", os.getenv("SERVER_PORT", "8000")))
SERVER_WORKERS = int(os.getenv("SERVER_WORKERS", "1"))
RATE_LIMIT_RPM = int(os.getenv("RATE_LIMIT_RPM", "60"))
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "20"))
CORS_ORIGINS = _parse_csv(os.getenv("CORS_ORIGINS"), ["*"])
CORS_ALLOW_CREDENTIALS = _parse_bool(
    os.getenv("CORS_ALLOW_CREDENTIALS"), default=False)
TRUSTED_HOSTS = _parse_csv(
    os.getenv("TRUSTED_HOSTS"), ["*"])
ENABLE_DOCS = _parse_bool(os.getenv("ENABLE_DOCS"), default=False)
APP_ENV = os.getenv("APP_ENV", "production").strip().lower()
# ── Ensure dirs exist ─────────────────────────────────────────────────
try:
    if IS_VERCEL:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    else:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        MEMORY_DB.parent.mkdir(parents=True, exist_ok=True)
except Exception as e:
    if not IS_VERCEL:
        print(f"⚠️ Could not create directories: {e}")

# ── Security validation ────────────────────────────────────────────────
# API_AUTH_TOKEN MUST be set in .env or Replit Secrets
API_AUTH_TOKEN = os.getenv("API_AUTH_TOKEN", "")
if not API_AUTH_TOKEN:
    import warnings
    warnings.warn(
        "⚠️ API_AUTH_TOKEN is not set! "
        "Set it in .env or Replit Secrets before going live.",
        RuntimeWarning,
        stacklevel=2,
    )
    # Generate a temporary token so the server can start for setup
    import secrets as _sec
    API_AUTH_TOKEN = _sec.token_hex(32)
    print(f"🔑 Temporary API_AUTH_TOKEN generated: {API_AUTH_TOKEN}")
    print("   Set this in Replit Secrets → API_AUTH_TOKEN for persistence.")
