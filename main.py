"""
main.py  —  FastAPI production server for the LangGraph OCR pipeline.

Endpoints:
  POST /extract        — Main OCR endpoint (called by Flutter app)
  POST /feedback       — Human-in-the-loop response (approve/reject with corrections)
  GET  /health         — Health check (LLM + system status)
  GET  /stats          — Pipeline statistics from long-term memory

Usage:
  uvicorn main:app --host 0.0.0.0 --port 8000
"""
from __future__ import annotations

import base64
import gc
import json
import logging
import secrets
import time
import uuid
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, File, HTTPException, UploadFile, Depends, Security, Request
from fastapi.security.api_key import APIKeyHeader
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.middleware.gzip import GZipMiddleware
from starlette.middleware.trustedhost import TrustedHostMiddleware
import uvicorn

from config.settings import (
    MAX_UPLOAD_MB,
    SERVER_HOST,
    SERVER_PORT,
    SERVER_WORKERS,
    LOG_LEVEL,
    GROQ_API_KEY,
    API_AUTH_TOKEN,
    CORS_ORIGINS,
    CORS_ALLOW_CREDENTIALS,
    TRUSTED_HOSTS,
    ENABLE_DOCS,
    APP_ENV,
)

# ── Logging ────────────────────────────────────────────────────────────
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s │ %(name)s │ %(levelname)s │ %(message)s",
)
logger = logging.getLogger("server")

# ── Valid MIME types ───────────────────────────────────────────────────
VALID_MIME_TYPES = frozenset(
    {"image/jpeg", "image/jpg", "image/png", "image/webp", "image/bmp"})
MAX_FILE_BYTES = MAX_UPLOAD_MB * 1024 * 1024

# ── Global pipeline (compiled once at startup) ─────────────────────────
_pipeline = None
_pending_sessions: dict[str, dict] = {}  # session_id → session data
SESSION_TTL_SECONDS = 3600  # sessions older than 1h are pruned


def _prune_stale_sessions() -> None:
    """Remove sessions older than SESSION_TTL_SECONDS to prevent memory leaks."""
    now = time.time()
    stale = [
        sid for sid, data in _pending_sessions.items()
        if now - data.get("created_at", now) > SESSION_TTL_SECONDS
    ]
    for sid in stale:
        _pending_sessions.pop(sid, None)
    if stale:
        logger.info(f"Pruned {len(stale)} stale session(s)")


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _pipeline
    gc.collect()

    # Check Groq API key
    if not GROQ_API_KEY:
        logger.error(
            "⚠️ GROQ_API_KEY not set! Get one at https://console.groq.com/keys")
    else:
        logger.info("✅ Groq API key configured")

    logger.info("🚀 Compiling LangGraph pipeline…")
    try:
        from graph.pipeline import get_compiled_graph
        _pipeline = get_compiled_graph()
        logger.info("✅ LangGraph pipeline ready")
    except Exception as e:
        logger.error(f"Failed to compile pipeline: {e}")
        _pipeline = None

    yield

    gc.collect()
    logger.info("Server shutdown complete")


# ── App ────────────────────────────────────────────────────────────────
app = FastAPI(
    title="MediPackages OCR Server (Groq Vision)",
    version="4.0.0",
    description="Groq-powered medical OCR: Image → Extraction → Validation → Package Matching",
    lifespan=lifespan,
    docs_url="/docs" if ENABLE_DOCS else None,
    redoc_url="/redoc" if ENABLE_DOCS else None,
    openapi_url="/openapi.json" if ENABLE_DOCS else None,
)


class RequestContextMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
        request.state.request_id = request_id
        start = time.perf_counter()

        response = await call_next(request)

        elapsed_ms = (time.perf_counter() - start) * 1000
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["X-Response-Time-Ms"] = f"{elapsed_ms:.2f}"
        logger.info(
            "%s %s -> %s %.2fms (%s)",
            request.method,
            request.url.path,
            response.status_code,
            elapsed_ms,
            request_id,
        )
        return response


app.add_middleware(RequestContextMiddleware)
app.add_middleware(GZipMiddleware, minimum_size=1024)
app.add_middleware(TrustedHostMiddleware, allowed_hosts=TRUSTED_HOSTS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=CORS_ALLOW_CREDENTIALS,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    request_id = getattr(request.state, "request_id", "unknown")
    logger.warning("Validation error (%s): %s", request_id, exc.errors())
    return JSONResponse(
        status_code=422,
        content={
            "success": False,
            "error": "Validation error",
            "details": exc.errors(),
            "request_id": request_id,
        },
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    request_id = getattr(request.state, "request_id", "unknown")
    logger.error("Unhandled exception (%s): %s", request_id, exc, exc_info=exc)
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "error": "Internal server error",
            "request_id": request_id,
        },
    )

# ── Security ───────────────────────────────────────────────────────────
api_key_scheme = APIKeyHeader(name="X-API-Key", auto_error=False)


def get_api_key(api_key_value: str = Security(api_key_scheme)):
    if api_key_value and secrets.compare_digest(api_key_value, API_AUTH_TOKEN):
        return api_key_value
    logger.warning(
        "Unauthorized access attempt with invalid or missing API Key")
    raise HTTPException(
        status_code=403,
        detail="Could not validate credentials",
    )


# ── Request models ─────────────────────────────────────────────────────
class FeedbackRequest(BaseModel):
    session_id: str
    decision: str          # "approved" or "rejected"
    reason: Optional[str] = ""
    corrections: Optional[dict] = None   # {field_name: corrected_value}


# ── Smart Search Request Models ───────────────────────────────────────
class SmartSearchRequest(BaseModel):
    query: str
    mode: str = "normal"  # normal, smart (combined AI selector)
    procedure: str = ""   # specific procedure name
    disease: str = ""     # specific disease/condition
    symptoms: list[str] = []
    patient_age: int = 0
    patient_gender: str = ""
    limit: int = 50  # Default to 50 to show all matching packages


class PackageResultModel(BaseModel):
    package_code: str
    package_name: str
    rate: float
    speciality: str
    category: str
    is_main: bool = False
    is_addon: bool = False
    is_implant: bool = False
    medical_reason: Optional[str] = None


class SmartSearchResponse(BaseModel):
    main_package: Optional[PackageResultModel] = None
    auto_implant: Optional[PackageResultModel] = None
    suggested_addons: list[PackageResultModel] = []
    blocked_rules: list[str] = []
    doctor_reasoning: str = ""
    raw_packages: list[dict] = []
    approval_likelihood: str = ""  # HIGH, MEDIUM, LOW, REJECTED


# ── Helper ─────────────────────────────────────────────────────────────
def _detect_mime(upload: UploadFile) -> str:
    mime = getattr(upload, "content_type", None)
    if not mime or mime in ("application/octet-stream", "application/x-www-form-urlencoded"):
        filename = (upload.filename or "").lower()
        if filename.endswith(".png"):
            return "image/png"
        if filename.endswith(".webp"):
            return "image/webp"
        return "image/jpeg"
    return mime


async def _run_pipeline_async(state: dict | None, config: dict) -> dict:
    """Run the pipeline asynchronously and return final state."""
    final_state = state or {}
    async for event in _pipeline.astream(state, config=config, stream_mode="values"):
        final_state = event
    return final_state


# ── Routes ─────────────────────────────────────────────────────────────
@app.post("/extract")
async def extract_ocr(image: UploadFile = File(...), api_key: str = Depends(get_api_key)):
    """
    Extract structured medical data from an uploaded document image.
    Returns the same JSON schema as before for full Flutter compatibility.
    """
    if _pipeline is None:
        raise HTTPException(503, "Pipeline not initialized")

    # Validate MIME type
    mime_type = _detect_mime(image)
    if mime_type not in VALID_MIME_TYPES:
        raise HTTPException(415, f"Unsupported file type: {mime_type}")

    # Read & size-check iteratively to avoid memory bombs
    try:
        contents = bytearray()
        while chunk := await image.read(1024 * 1024):  # 1MB chunks
            contents.extend(chunk)
            if len(contents) > MAX_FILE_BYTES:
                await image.close()
                raise HTTPException(
                    413, f"File too large. Max: {MAX_UPLOAD_MB}MB")
        contents = bytes(contents)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"File upload error: {e}")
        raise HTTPException(400, "Failed to read upload")
    finally:
        await image.close()

    logger.info(
        f"Processing: {image.filename} ({len(contents)/1024:.0f}KB, {mime_type})")

    # Resize image to reduce LLM token count for vision model
    import io
    from PIL import Image
    try:
        with Image.open(io.BytesIO(contents)) as pil_img:
            if pil_img.mode in ("RGBA", "P"):
                pil_img = pil_img.convert("RGB")

            max_dim = 1024
            if max(pil_img.size) > max_dim:
                ratio = max_dim / max(pil_img.size)
                new_size = (int(pil_img.size[0] * ratio),
                            int(pil_img.size[1] * ratio))
                pil_img = pil_img.resize(new_size, Image.Resampling.LANCZOS)

                buffer = io.BytesIO()
                pil_img.save(buffer, format="JPEG", quality=85)
                contents = buffer.getvalue()
                mime_type = "image/jpeg"
                logger.info(
                    f"Image resized to {new_size} ({len(contents)/1024:.0f}KB) to optimize API tokens")
    except Exception as e:
        logger.warning(
            f"Failed to resize image, continuing with original: {e}")

    session_id = str(uuid.uuid4())
    image_b64 = base64.b64encode(contents).decode("utf-8")
    del contents  # free early

    # Initial state
    initial_state = {
        "session_id": session_id,
        "image_b64": image_b64,
        "mime_type": mime_type,
        "retry_count": 0,
        "supervisor_notes": [],
    }

    config = {"configurable": {"thread_id": session_id}}

    async def event_generator():
        start_t = time.time()
        yield f"data: {json.dumps({'status': 'progress', 'message': 'Image pre-processing completed.'})}\n\n"

        try:
            final_state = initial_state

            # Phase A: OCR → Extraction → Validation → INTERRUPT (always)
            async for event in _pipeline.astream(initial_state, config=config, stream_mode="values"):
                final_state = event

                notes = event.get("supervisor_notes", [])
                msg = notes[-1] if notes else "Pipeline step completed..."
                yield f"data: {json.dumps({'status': 'progress', 'message': msg})}\n\n"

                # Check interruption (human review point)
                if event.get("__interrupt__") or "__interrupt__" in event:
                    final_state["__interrupted__"] = True
                    break

            elapsed = time.time() - start_t

            # Phase A complete → always send pending_review with AI Doctor summary
            extracted = final_state.get("extracted", {})
            validation = final_state.get("validation", {})

            # Store session for resume
            _pending_sessions[session_id] = {
                "thread_id": session_id,
                "extracted": extracted,
                "validation": validation,
                "retry_count": 0,
                "image_b64": image_b64,
                "mime_type": mime_type,
                "raw_text": final_state.get("raw_text", ""),
                "vision_text": final_state.get("vision_text", ""),
                "created_at": time.time(),
            }

            # Prune old sessions on each new request to avoid unbounded growth
            _prune_stale_sessions()

            resp = {
                "success": True,
                "status": "pending_review",
                "session_id": session_id,
                "message": "AI Doctor summary ready — please verify.",
                "preview": extracted,
                "validation": validation,
                "processing_time_seconds": round(elapsed, 2),
            }
            logger.info(
                f"✅ Phase A complete in {elapsed:.2f}s — awaiting user verification")
            yield f"data: {json.dumps(resp)}\n\n"

        except Exception as e:
            logger.exception("Pipeline streaming error")
            yield f"data: {json.dumps({'success': False, 'status': 'error', 'error': str(e)})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/feedback")
async def submit_feedback(req: FeedbackRequest, api_key: str = Depends(get_api_key)):
    """
    Human-in-the-loop feedback endpoint.
    - Approved: resumes pipeline → package matching → final result
    - Rejected: stores rejection in memory, returns retry_available
    """
    session_id = req.session_id
    if session_id not in _pending_sessions:
        raise HTTPException(404, f"No pending session: {session_id}")

    if req.decision not in ("approved", "rejected"):
        raise HTTPException(400, "decision must be 'approved' or 'rejected'")

    session = _pending_sessions[session_id]

    if req.decision == "rejected":
        # Store rejection in memory for learning
        try:
            from memory.sqlite_store import AgentMemory
            mem = AgentMemory()
            request_id = mem.create_request(
                session_id, session.get("raw_text", ""))
            mem.update_request(request_id, session.get(
                "extracted", {}), status="rejected")
            mem.store_feedback(
                request_id=request_id,
                decision="rejected",
                reason=req.reason or "User rejected AI Doctor summary",
            )
            logger.info(
                f"❌ Rejection stored in memory (request {request_id}): {req.reason}")
        except Exception as e:
            logger.warning(f"Failed to store rejection: {e}")

        retry_count = session.get("retry_count", 0)
        can_retry = retry_count < 3

        return {
            "success": True,
            "status": "rejected",
            "message": "Rejection stored. AI will learn from this.",
            "retry_available": can_retry,
            "retry_count": retry_count,
        }

    # User approved → Phase B: resume pipeline (package matching → supervisor)
    config = {"configurable": {"thread_id": session_id}}

    # Merge session data with user feedback
    resume_state = {
        "session_id": session_id,
        "extracted": session.get("extracted", {}),
        "validation": session.get("validation", {}),
        "image_b64": session.get("image_b64", ""),
        "mime_type": session.get("mime_type", "image/jpeg"),
        "human_decision": "approved",
        "human_reason": req.reason or "",
        "human_correction": req.corrections or {},
    }

    # Apply corrections if provided
    if req.corrections:
        for field, value in req.corrections.items():
            resume_state["extracted"][field] = value

    try:
        # Update the state with user feedback
        _pipeline.update_state(config, resume_state)
        # Resume the pipeline
        result = await _run_pipeline_async(None, config)
        _pending_sessions.pop(session_id, None)

        final = result.get("final_response", {})
        if not final:
            logger.warning(
                "Pipeline returned no final_response. Constructing fallback.")
            final = {
                "success": True,
                "data": {
                    **resume_state.get("extracted", {}),
                    "_agentic_data": {
                        "best_packages": result.get("best_packages", [])
                    }
                }
            }

        # Store approval in memory
        try:
            from memory.sqlite_store import AgentMemory
            mem = AgentMemory()
            request_id = mem.create_request(
                session_id, session.get("raw_text", ""))
            mem.update_request(request_id, session.get(
                "extracted", {}), status="completed")
            mem.store_feedback(request_id=request_id, decision="approved")
        except Exception as e:
            logger.warning(f"Failed to store approval: {e}")

        return final
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Feedback processing error")
        raise HTTPException(
            500, "Internal Server Error during feedback processing")


@app.post("/retry")
async def retry_extraction(req: FeedbackRequest, api_key: str = Depends(get_api_key)):
    """
    Re-run extraction with rejection context from memory.
    Uses the same image but with improved prompts.
    """
    session_id = req.session_id
    if session_id not in _pending_sessions:
        raise HTTPException(404, f"No pending session: {session_id}")

    session = _pending_sessions[session_id]
    retry_count = session.get("retry_count", 0) + 1

    if retry_count > 3:
        _pending_sessions.pop(session_id, None)
        return {
            "success": False,
            "status": "max_retries",
            "message": "Maximum retry attempts reached. Please try scanning again.",
        }

    # Re-run Phase A with the same image
    image_b64 = session.get("image_b64", "")
    mime_type = session.get("mime_type", "image/jpeg")

    if not image_b64:
        raise HTTPException(400, "No image data available for retry")

    new_session_id = str(uuid.uuid4())
    initial_state = {
        "session_id": new_session_id,
        "image_b64": image_b64,
        "mime_type": mime_type,
        "retry_count": retry_count,
        "supervisor_notes": [],
    }
    config = {"configurable": {"thread_id": new_session_id}}

    async def event_generator():
        start_t = time.time()
        yield f"data: {json.dumps({'status': 'progress', 'message': f'Retry {retry_count}/3 — re-analyzing with improved prompts...'})}\n\n"

        try:
            final_state = initial_state
            async for event in _pipeline.astream(initial_state, config=config, stream_mode="values"):
                final_state = event
                notes = event.get("supervisor_notes", [])
                msg = notes[-1] if notes else "Pipeline step completed..."
                yield f"data: {json.dumps({'status': 'progress', 'message': msg})}\n\n"
                if event.get("__interrupt__") or "__interrupt__" in event:
                    final_state["__interrupted__"] = True
                    break

            elapsed = time.time() - start_t
            extracted = final_state.get("extracted", {})
            validation = final_state.get("validation", {})

            # Store new session
            _pending_sessions.pop(session_id, None)  # remove old
            _pending_sessions[new_session_id] = {
                "thread_id": new_session_id,
                "extracted": extracted,
                "validation": validation,
                "retry_count": retry_count,
                "image_b64": image_b64,
                "mime_type": mime_type,
                "raw_text": final_state.get("raw_text", ""),
                "vision_text": final_state.get("vision_text", ""),
                "created_at": time.time(),
            }

            resp = {
                "success": True,
                "status": "pending_review",
                "session_id": new_session_id,
                "message": f"Retry {retry_count}/3 — AI Doctor summary re-generated.",
                "preview": extracted,
                "validation": validation,
                "processing_time_seconds": round(elapsed, 2),
            }
            logger.info(f"✅ Retry {retry_count} complete in {elapsed:.2f}s")
            yield f"data: {json.dumps(resp)}\n\n"

        except Exception as e:
            logger.exception("Retry pipeline error")
            yield f"data: {json.dumps({'success': False, 'status': 'error', 'error': str(e)})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/health")
async def health():
    """Health check: Groq API reachability + pipeline status."""
    llm_ok = False
    try:
        from tools.llm_tool import check_llm_connection
        llm_ok = check_llm_connection()
    except Exception:
        pass

    status = "healthy" if (llm_ok and _pipeline is not None) else "degraded"
    code = 200 if status == "healthy" else 503

    return JSONResponse(status_code=code, content={
        "status": status,
        "environment": APP_ENV,
        "mode": "Groq Vision OCR v4",
        "pipeline_ready": _pipeline is not None,
        "groq_reachable": llm_ok,
        "pending_sessions": len(_pending_sessions),
    })


@app.get("/stats")
async def stats(api_key: str = Depends(get_api_key)):
    """Return pipeline statistics from long-term memory."""
    try:
        from memory.sqlite_store import AgentMemory
        mem = AgentMemory()
        return {
            "top_rejection_patterns": mem.get_top_rejection_patterns(limit=10),
            "recent_approvals_count": len(mem.get_recent_approvals(limit=50)),
        }
    except Exception as e:
        logger.exception("Stats endpoint failed")
        raise HTTPException(500, "Failed to fetch statistics")


# ── Smart Package Search ──────────────────────────────────────────────
_packages_cache: list[dict] = []
_robotic_cache: list[dict] = []


def _load_packages_cache():
    """Load packages into memory for fast search."""
    global _packages_cache, _robotic_cache
    if _packages_cache and _robotic_cache:
        return

    from config.settings import BASE_DIR
    import json

    try:
        pkg_path = BASE_DIR.parent / "assets" / "maa_packages.json"
        with open(pkg_path, "r", encoding="utf-8") as f:
            _packages_cache = json.load(f)
        logger.info(f"Loaded {len(_packages_cache)} standard packages")
    except Exception as e:
        logger.warning(f"Failed to load packages: {e}")
        _packages_cache = []

    try:
        rob_path = BASE_DIR.parent / "assets" / "maa_robotic_surgeries.json"
        with open(rob_path, "r", encoding="utf-8") as f:
            _robotic_cache = json.load(f)
        logger.info(f"Loaded {len(_robotic_cache)} robotic packages")
    except Exception as e:
        logger.warning(f"Failed to load robotic packages: {e}")
        _robotic_cache = []


def _search_packages_basic(query: str, limit: int = 50) -> list[dict]:
    """Basic keyword search to pre-filter packages with medical synonym expansion."""
    _load_packages_cache()
    query_lower = query.lower()
    terms = query_lower.split()

    # Expand search terms with medical synonyms and related terms
    expanded_terms = set(terms)
    SYMPTOM_EXPANSIONS = {
        "chest pain": ["coronary", "angiography", "cardiac", "heart", "ptca", "cabg", "thrombolysis", "mi"],
        "heart attack": ["mi", "myocardial", "thrombolysis", "ptca", "coronary", "stemi", "nstemi", "cabg"],
        "breathlessness": ["heart failure", "cardiac", "pulmonary", "respiratory", "chf"],
        "stomach pain": ["appendix", "gallbladder", "cholecyst", "pancreat", "intestin"],
        "abdominal pain": ["appendix", "gallbladder", "cholecyst", "pancreat", "intestin", "hernia"],
        "eye": ["cataract", "glaucoma", "retina", "phaco", "iol"],
        "knee": ["arthroplasty", "tkr", "replacement", "arthroscopy", "ligament"],
        "hip": ["arthroplasty", "thr", "replacement", "hemiarthroplasty", "fracture"],
        "kidney stone": ["urolithiasis", "pcnl", "ursl", "lithotripsy", "renal"],
        "fracture": ["orif", "fixation", "plate", "nail", "reduction"],
        "burn": ["burns", "graft", "debridement", "dressing", "skin", "eschar", "tbsa"],
        "gastric": ["gastric", "gastrectomy", "gastrojejunostomy", "ulcer", "stomach"],
        "blood": ["transfusion", "blood", "component", "packed", "ffp", "platelet"],
    }

    for symptom, related in SYMPTOM_EXPANSIONS.items():
        if symptom in query_lower or any(t in symptom for t in terms):
            expanded_terms.update(related)

    all_packages = _packages_cache + _robotic_cache
    scored = []

    for pkg in all_packages:
        name = str(pkg.get("PACKAGE NAME", pkg.get("Package Name", ""))).lower()
        code = str(pkg.get("PACKAGE CODE", "")).lower()
        spec = str(pkg.get("SPECIALITY", pkg.get("Speciality", ""))).lower()

        score = 0
        # Score original terms higher
        for term in terms:
            if term in code:
                score += 15
            if term in name:
                score += 10
            if term in spec:
                score += 5

        # Score expanded terms
        for term in expanded_terms:
            if term not in terms:  # Don't double count
                if term in code:
                    score += 8
                if term in name:
                    score += 5
                if term in spec:
                    score += 3

        if score > 0:
            scored.append((score, pkg))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [p for _, p in scored[:limit]]


def _format_packages_for_ai(packages: list[dict]) -> str:
    """Format packages for AI context."""
    result = []
    for i, pkg in enumerate(packages[:20]):
        name = pkg.get("PACKAGE NAME", pkg.get("Package Name", ""))
        code = pkg.get("PACKAGE CODE", "")
        rate = pkg.get("RATE", pkg.get("Rate", 0))
        spec = pkg.get("SPECIALITY", pkg.get("Speciality", ""))
        cat = pkg.get("PACKAGE CATEGORY", pkg.get("PACKAGE TYPE", ""))
        implant = pkg.get("IMPLANT PACKAGE", pkg.get("IMPLANT", "NO IMPLANT"))
        result.append(
            f"{i+1}. [{code}] {name[:100]}... | Rate: {rate} | {spec} | {cat} | Implant: {implant[:50] if implant else 'NO'}")
    return "\n".join(result)


def _identify_package_type(name: str, rate: float, implant_field: str) -> dict[str, bool]:
    """Identify package type based on package name, rate, and implant field."""
    name_upper = name.upper()
    implant_upper = implant_field.upper()
    is_implant = 'IMPLANT' in implant_upper and 'NO IMPLANT' not in implant_upper

    return {
        'is_surgical': rate > 0 and '[REGULAR PROCEDURE]' in name_upper,
        'is_medical_management': rate == 0 and '[ADD' not in name_upper,
        'is_standalone': 'STAND-ALONE' in name_upper or 'STAND ALONE' in name_upper,
        'is_addon': '[ADD-ON' in name_upper or '[ADD ON' in name_upper or 'ADDON' in name_upper,
        'is_implant': is_implant,
        'is_extended_los': 'EXTENDED LOS' in name_upper,
    }


def _validate_package_combination(main_type: dict[str, bool], candidate_type: dict[str, bool], candidate_code: str) -> str | None:
    """Validate if two packages can be combined based on MAA Yojana rules."""
    # Rule 2: Stand-alone packages cannot be booked with any other package
    if candidate_type['is_standalone']:
        return f"Rule 2 VIOLATION: {candidate_code} is Stand-alone - cannot combine with any other package"

    if main_type['is_standalone']:
        return f"Rule 2 VIOLATION: Main package is Stand-alone - cannot add {candidate_code}"

    # Rule 1: Surgical and medical management packages cannot be booked together
    if main_type['is_surgical'] and candidate_type['is_medical_management']:
        return f"Rule 1 VIOLATION: Cannot combine surgical package with medical management ({candidate_code}, rate=₹0)"

    if main_type['is_medical_management'] and candidate_type['is_surgical']:
        return f"Rule 1 VIOLATION: Cannot combine medical management package with surgical ({candidate_code})"

    # Rule 5: Extended length of stay can ONLY be booked with surgery packages
    if candidate_type['is_extended_los'] and not main_type['is_surgical']:
        return f"Rule 5 VIOLATION: Extended LOS ({candidate_code}) can only be booked with surgery packages"

    return None  # Valid combination


def _get_ai_prompt(mode: str) -> str:
    """Get mode-specific AI system prompt."""
    base_rules = """═══════════════════════════════════════════════════════════════
STRICT MAA YOJANA/PMJAY PACKAGE COMBINATION RULES (MUST FOLLOW):
═══════════════════════════════════════════════════════════════

RULE 1: SURGICAL + MEDICAL MANAGEMENT = BLOCKED
- Surgical procedure packages CANNOT be booked with medical management packages
- Medical management packages typically have ZERO (₹0) rate
- Example: If patient needs surgery (e.g., appendectomy), you CANNOT also add a general disease package with ₹0 rate
- VIOLATION triggers blocked_rules warning

RULE 2: STAND-ALONE PACKAGES = EXCLUSIVE
- Stand-alone packages CANNOT be booked with ANY other package
- They are complete treatment packages covering everything
- Example: A comprehensive cancer treatment package is stand-alone
- VIOLATION triggers blocked_rules warning

RULE 3: ADD-ON PACKAGES = ALLOWED WITH REGULAR
- Add-on packages (ICU, anesthesia, extended stay) CAN combine with regular procedure packages
- Examples of valid add-ons:
  • ICU charges (per day)
  • General anesthesia charges
  • Extended length of stay
  • Post-operative care
- Only suggest add-ons that are medically necessary

RULE 4: IMPLANT PACKAGES = AUTO-POPUP
- Implant packages AUTOMATICALLY apply with procedures requiring implants
- Examples:
  • Hip replacement → Hip implant auto-added
  • Cardiac stent → Stent implant auto-added
  • Intraocular lens → IOL implant auto-added
- Check if the procedure needs an implant, then find matching implant code

RULE 5: EXTENDED LOS = ALLOWED WITH SURGERY
- Extended Length of Stay packages CAN combine with surgery packages
- Only suggest if the surgery typically requires extended recovery

RULE 6: ONE MAIN PACKAGE PER CLAIM
- Only ONE primary procedure/disease package per claim
- If multiple procedures needed, patient needs separate claims

═══════════════════════════════════════════════════════════════

APPROVAL CRITERIA:
- Package must match the EXACT procedure/disease being treated
- Patient must be eligible (Ayushman/MAA Yojana card holder)
- Hospital must be empaneled for this specific package
- Pre-authorization required for most surgeries"""

    if mode == "smart":
        mode_text = """
MODE: MAA YOJANA SMART PACKAGE SELECTOR (Dr. Arth - Clinical Expert)

YOUR ROLE: You are Dr. Arth, a senior consultant with 15+ years experience in PMJAY/Ayushman Bharat.
You THINK LIKE A DOCTOR - understanding symptoms, diagnosis pathways, and treatment options.

═══════════════════════════════════════════════════════════════
CLINICAL REASONING PATHWAYS (Think step-by-step like a doctor):
═══════════════════════════════════════════════════════════════

🫀 CHEST PAIN / CARDIAC SYMPTOMS:
   Step 1: Coronary Angiography (diagnosis) - ALWAYS FIRST for chest pain
   Step 2: If MI/Heart Attack confirmed → Thrombolysis OR Primary PTCA
   Step 3: If blockages found:
           - 1-2 vessels → PTCA (Angioplasty) with stent
           - 3 vessels or Left Main → CABG (Bypass surgery)
   Step 4: If Heart Failure → CHF management package

🫁 BREATHLESSNESS:
   - If cardiac cause → Heart Failure package
   - If pulmonary → Pulmonology packages
   - May need valve surgery if valvular disease

🤕 ABDOMINAL PAIN (by location):
   - Right Lower Quadrant → Appendicitis → Appendectomy
   - Right Upper Quadrant → Gallstones → Cholecystectomy
   - Epigastric → Pancreatitis or Peptic ulcer
   - Generalized → Obstruction or Peritonitis

🦴 ORTHOPEDIC:
   - Fracture → ORIF with implant (plates/screws)
   - Knee OA in elderly → Total Knee Replacement
   - Hip fracture/OA → Total Hip Replacement or Hemiarthroplasty

👁️ EYE:
   - Gradual vision loss + lens opacity → Cataract surgery (Phaco + IOL)
   - Pressure/field loss → Glaucoma surgery

═══════════════════════════════════════════════════════════════
YOUR ANALYSIS WORKFLOW:
═══════════════════════════════════════════════════════════════

1. 🔍 IDENTIFY: What symptom/condition is presented?
2. 🧠 DIAGNOSE: What's the likely diagnosis? What workup is needed?
3. 💊 TREAT: What's the standard treatment pathway?
4. 📦 PACKAGE: Find the EXACT MAA Yojana package for this treatment
5. 🔗 ADDONS: Does it need implants? ICU? Extended stay?
6. ✅ VALIDATE: Check business rules - no violations

RESPONSE REQUIREMENTS:
- ALWAYS provide a main_package_code - NEVER return null or "not specified"
- If symptom is vague, suggest the DIAGNOSTIC package first
- Explain the clinical reasoning in doctor_summary
- For chest pain → ALWAYS suggest Coronary Angiography first
- For heart attack → Thrombolysis or PTCA as primary"""
    elif mode == "procedure":
        mode_text = """
MODE: PROCEDURE SEARCH
Finding the exact surgical/medical procedure package.
- Match procedure name to exact package code
- Suggest required implants if procedure needs them
- Add relevant add-ons only if medically necessary"""
    elif mode == "disease":
        mode_text = """
MODE: DISEASE/CONDITION SEARCH
Finding packages that TREAT this specific condition.
- Consider which procedures are typically used to treat this disease
- Select the most common/appropriate treatment approach
- Consider: Surgical vs Conservative (medical management) approach"""
    else:
        mode_text = """
MODE: GENERAL SEARCH
Find the single most relevant package for the query."""

    return f"""You are Dr. Arth, an expert MAA Yojana/PMJAY package consultant with 15+ years experience in Ayushman Bharat scheme in Gujarat, India.

{mode_text}

{base_rules}

CRITICAL OUTPUT RULES:
- Return ONLY 1-3 most relevant packages
- Do NOT include packages that don't match the query
- If blocked_rules has violations, set approval_likelihood to "REJECTED"
- Be specific in doctor_summary - mention package code and why it was selected

Return ONLY valid JSON:
{{
    "main_package_code": "EXACT_CODE_FROM_LIST or null if no match",
    "main_package_reason": "Why this package is appropriate for this case",
    "implant_code": "IMPLANT_CODE or null if no implant needed",
    "addons": [{{"code": "ADDON_CODE", "reason": "Medical justification for this add-on"}}],
    "alternative_codes": ["Max 2 alternative codes if diagnosis is uncertain"],
    "blocked_rules": ["List any MAA Yojana rule violations here"],
    "approval_likelihood": "HIGH / MEDIUM / LOW / REJECTED",
    "doctor_summary": "Dr. Arth's professional assessment: Explain selection, any concerns, and approval guidance"
}}"""""


@app.post("/smart-search", response_model=SmartSearchResponse)
async def smart_search(request: SmartSearchRequest):
    """AI-powered smart package search with doctor reasoning."""
    # Build search query from all inputs
    search_terms = []
    if request.query:
        search_terms.append(request.query)
    if request.procedure:
        search_terms.append(request.procedure)
    if request.disease:
        search_terms.append(request.disease)

    combined_query = " ".join(search_terms).strip()
    if not combined_query:
        return SmartSearchResponse(
            doctor_reasoning="Please provide a procedure name, disease, or search query.",
            raw_packages=[]
        )

    # Parse comma-separated query: first = main package, rest = add-ons
    main_search_term = combined_query
    addon_search_terms = []
    if "," in combined_query:
        parts = [p.strip() for p in combined_query.split(",") if p.strip()]
        if parts:
            main_search_term = parts[0]
            addon_search_terms = parts[1:] if len(parts) > 1 else []

    # Get clinical pathway hints
    clinical_hint = ""
    try:
        from tools.medical_knowledge import get_clinical_pathway, get_packages_for_symptom
        pathway = get_clinical_pathway(main_search_term)
        if pathway:
            clinical_hint = f"\n\nCLINICAL PATHWAY HINT:\n{pathway.get('doctor_reasoning', '')}"
        symptom_packages = get_packages_for_symptom(main_search_term)
        if symptom_packages:
            clinical_hint += "\n\nRECOMMENDED PACKAGES FOR THIS SYMPTOM:"
            for pkg in symptom_packages[:3]:
                clinical_hint += f"\n- {pkg['code']}: {pkg['name']} ({pkg['reason']})"
    except Exception as e:
        logger.warning(f"Could not get clinical pathway: {e}")

    # Search for main packages with higher limit
    relevant_packages = _search_packages_basic(
        main_search_term, limit=request.limit)

    # Also search for add-on packages if specified
    addon_packages = []
    for addon_term in addon_search_terms:
        addon_results = _search_packages_basic(addon_term, limit=20)
        addon_packages.extend(addon_results)

    # Combine but keep main packages first
    all_relevant_packages = relevant_packages.copy()
    seen_codes = {pkg.get("PACKAGE CODE", "") for pkg in relevant_packages}
    for pkg in addon_packages:
        code = pkg.get("PACKAGE CODE", "")
        if code not in seen_codes:
            all_relevant_packages.append(pkg)
            seen_codes.add(code)

    if not relevant_packages:
        return SmartSearchResponse(
            doctor_reasoning=f"No packages found matching '{main_search_term}'. Please try different keywords or check spelling.",
            raw_packages=[]
        )

    try:
        from groq import Groq

        if not GROQ_API_KEY:
            raise ValueError("GROQ_API_KEY not configured")

        from config.settings import GROQ_MODEL

        client = Groq(api_key=GROQ_API_KEY)
        packages_context = _format_packages_for_ai(
            all_relevant_packages[:25])  # Increase context for AI

        symptoms_str = ', '.join(
            request.symptoms) if request.symptoms else 'None specified'

        # Indicate main vs addon terms to AI
        addon_hint = ""
        if addon_search_terms:
            addon_hint = f"\n- Add-on procedures requested: {', '.join(addon_search_terms)}"

        # Build detailed user prompt with clinical pathway
        user_prompt = f"""PATIENT CASE:
- Main Procedure Search: {main_search_term}
- Search Query: {request.query or 'Not specified'}
- Procedure Needed: {request.procedure or 'Not specified'}
- Disease/Condition: {request.disease or 'Not specified'}
- Symptoms: {symptoms_str}
- Patient Age: {request.patient_age if request.patient_age > 0 else 'Not specified'}
- Patient Gender: {request.patient_gender or 'Not specified'}{addon_hint}
{clinical_hint}

AVAILABLE PACKAGES (select ONLY from these):
{packages_context}

TASK: As Dr. Arth, select the BEST matching package(s) for MAA Yojana approval.
Think like a doctor: What is the diagnosis? What treatment is needed? Which package covers that treatment?
IMPORTANT: You MUST return a main_package_code - do NOT return null or empty.
The FIRST search term "{main_search_term}" is the MAIN package. Any additional comma-separated terms are ADD-ONS.
Return ONLY packages that will likely be APPROVED for this specific case."""

        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": _get_ai_prompt(
                    request.mode or "smart")},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.2,  # Lower for more consistent results
            max_tokens=1500,
            response_format={"type": "json_object"}
        )

        ai_result = json.loads(response.choices[0].message.content)

        # Build result with ONLY AI-selected packages + validation
        selected_codes = set()
        # Start with AI-detected violations
        validation_errors = list(ai_result.get("blocked_rules", []))
        result = SmartSearchResponse(
            doctor_reasoning=ai_result.get("doctor_summary", ""),
            blocked_rules=[],  # Will be populated with validated rules
            approval_likelihood=ai_result.get("approval_likelihood", ""),
            raw_packages=[]  # Will populate with only selected packages
        )

        main_pkg_data = None
        main_pkg_type = None

        # Find main package
        main_code = ai_result.get("main_package_code")
        if main_code and main_code != "null":
            selected_codes.add(main_code)
            for pkg in relevant_packages:
                if pkg.get("PACKAGE CODE", "") == main_code:
                    pkg_name = pkg.get(
                        "PACKAGE NAME", pkg.get("Package Name", ""))
                    pkg_rate = float(pkg.get("RATE", pkg.get("Rate", 0)))
                    pkg_implant = pkg.get("IMPLANT PACKAGE", "")

                    main_pkg_type = _identify_package_type(
                        pkg_name, pkg_rate, pkg_implant)
                    main_pkg_data = pkg

                    result.main_package = PackageResultModel(
                        package_code=main_code,
                        package_name=pkg_name,
                        rate=pkg_rate,
                        speciality=pkg.get(
                            "SPECIALITY", pkg.get("Speciality", "")),
                        category=pkg.get("PACKAGE CATEGORY",
                                         pkg.get("PACKAGE TYPE", "")),
                        is_main=True,
                        medical_reason=ai_result.get("main_package_reason", "")
                    )
                    break

        # Find implant (with validation if main package exists)
        implant_code = ai_result.get("implant_code")
        if implant_code and implant_code != "null":
            selected_codes.add(implant_code)
            for pkg in _packages_cache + _robotic_cache:
                if pkg.get("PACKAGE CODE", "") == implant_code:
                    pkg_name = pkg.get(
                        "PACKAGE NAME", pkg.get("Package Name", ""))
                    pkg_rate = float(pkg.get("RATE", pkg.get("Rate", 0)))
                    pkg_implant = pkg.get("IMPLANT PACKAGE", "")

                    # Validate if main package exists
                    if main_pkg_type:
                        implant_type = _identify_package_type(
                            pkg_name, pkg_rate, pkg_implant)
                        validation_error = _validate_package_combination(
                            main_pkg_type, implant_type, implant_code)
                        if validation_error:
                            validation_errors.append(validation_error)
                            break  # Skip this implant

                    result.auto_implant = PackageResultModel(
                        package_code=implant_code,
                        package_name=pkg_name,
                        rate=pkg_rate,
                        speciality=pkg.get(
                            "SPECIALITY", pkg.get("Speciality", "")),
                        category=pkg.get("PACKAGE CATEGORY",
                                         pkg.get("PACKAGE TYPE", "")),
                        is_implant=True,
                        medical_reason="Rule 4: Auto-suggested implant"
                    )
                    break

        # Find add-ons (with validation)
        for addon in ai_result.get("addons", [])[:5]:  # Max 5 addons
            addon_code = addon.get("code")
            if addon_code and main_pkg_type:
                for pkg in _packages_cache + _robotic_cache:
                    if pkg.get("PACKAGE CODE", "") == addon_code:
                        pkg_name = pkg.get(
                            "PACKAGE NAME", pkg.get("Package Name", ""))
                        pkg_rate = float(pkg.get("RATE", pkg.get("Rate", 0)))
                        pkg_implant = pkg.get("IMPLANT PACKAGE", "")

                        addon_type = _identify_package_type(
                            pkg_name, pkg_rate, pkg_implant)
                        validation_error = _validate_package_combination(
                            main_pkg_type, addon_type, addon_code)

                        if validation_error:
                            validation_errors.append(validation_error)
                        else:
                            selected_codes.add(addon_code)
                            addon_reason = addon.get("reason", "")
                            if addon_type['is_extended_los']:
                                addon_reason = f"Rule 5: Extended LOS with surgery. {addon_reason}"
                            elif addon_type['is_addon']:
                                addon_reason = f"Rule 3: Compatible add-on. {addon_reason}"

                            result.suggested_addons.append(PackageResultModel(
                                package_code=addon_code,
                                package_name=pkg_name,
                                rate=pkg_rate,
                                speciality=pkg.get(
                                    "SPECIALITY", pkg.get("Speciality", "")),
                                category=pkg.get(
                                    "PACKAGE CATEGORY", pkg.get("PACKAGE TYPE", "")),
                                is_addon=True,
                                medical_reason=addon_reason
                            ))
                        break

        # Add alternatives to raw_packages (max 2)
        for alt_code in ai_result.get("alternative_codes", [])[:2]:
            if alt_code and alt_code not in selected_codes:
                selected_codes.add(alt_code)

        # Populate raw_packages with ALL matching packages (AI-selected first)
        # First add AI-selected packages
        for pkg in all_relevant_packages:
            code = pkg.get("PACKAGE CODE", "")
            if code in selected_codes:
                result.raw_packages.append({
                    "code": code,
                    "name": pkg.get("PACKAGE NAME", pkg.get("Package Name", "")),
                    "rate": pkg.get("RATE", pkg.get("Rate", 0)),
                    "speciality": pkg.get("SPECIALITY", pkg.get("Speciality", "")),
                    "ai_selected": True
                })

        # Then add remaining packages (not AI-selected)
        for pkg in all_relevant_packages:
            code = pkg.get("PACKAGE CODE", "")
            if code not in selected_codes:
                result.raw_packages.append({
                    "code": code,
                    "name": pkg.get("PACKAGE NAME", pkg.get("Package Name", "")),
                    "rate": pkg.get("RATE", pkg.get("Rate", 0)),
                    "speciality": pkg.get("SPECIALITY", pkg.get("Speciality", "")),
                    "ai_selected": False
                })

        # Set blocked rules and adjust approval likelihood if violations found
        result.blocked_rules = validation_errors
        if validation_errors:
            if result.approval_likelihood not in ["REJECTED", "LOW"]:
                result.approval_likelihood = "LOW"
            # Append validation summary to doctor reasoning
            result.doctor_reasoning += f"\n\n⚠️ RULE VIOLATIONS DETECTED:\n" + \
                "\n".join(f"• {err}" for err in validation_errors[:5])

        logger.info(
            f"Smart search completed: '{main_search_term}' -> {len(selected_codes)} packages selected, {len(validation_errors)} violations")
        return result

    except Exception as e:
        logger.warning(f"AI search failed, falling back to basic search: {e}")
        # Fallback: return ALL keyword matches
        return SmartSearchResponse(
            doctor_reasoning=f"AI analysis unavailable. Showing all {len(all_relevant_packages)} matches for: {main_search_term}",
            raw_packages=[{
                "code": p.get("PACKAGE CODE", ""),
                "name": p.get("PACKAGE NAME", p.get("Package Name", "")),
                "rate": p.get("RATE", p.get("Rate", 0)),
                "speciality": p.get("SPECIALITY", p.get("Speciality", "")),
                "ai_selected": False
            } for p in all_relevant_packages]
        )


# ── Dev entry point ────────────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=SERVER_HOST,
        port=SERVER_PORT,
        workers=max(1, SERVER_WORKERS),
        reload=False,
    )
