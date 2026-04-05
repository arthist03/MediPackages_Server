"""
main.py  —  FastAPI production server for the LangGraph OCR pipeline.
Now optimized for Vercel Serverless Functions.

Endpoints:
  POST /extract        — Main OCR endpoint (called by Flutter app)
  POST /feedback       — Human-in-the-loop response (approve/reject with corrections)
  GET  /health         — Health check (LLM + system status)
  GET  /stats          — Pipeline statistics from long-term memory

Usage:
  Vercel: auto-deployed via vercel.json
  Local:  uvicorn main:app --host 0.0.0.0 --port 8000
"""
from __future__ import annotations

import base64
import difflib
import gc
import json
import logging
from pathlib import Path
import re
import secrets
import time
import uuid
from contextlib import asynccontextmanager
from typing import Optional, List, Dict, Any

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

# ── Interactive search flow sessions ────────────────────────────────────
_interactive_flows: dict[str, Any] = {}  # session_id → FlowState


def _normalize_interactive_step_title(step_name: str) -> str:
    """Force legacy supportive heading variants to the new Add-ons label."""
    normalized = (step_name or "").strip().lower()
    if "supportive" in normalized and "suggest" in normalized:
        return "Add Ons (If Applicable):"
    if normalized in {
        "supportive suggestion for:",
        "supportive suggestions for:",
        "supportive suggestion",
        "supportive suggestions",
    }:
        return "Add Ons (If Applicable):"
    return step_name


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

    from config.settings import GROQ_API_KEY, FIREBASE_SERVICE_ACCOUNT
    # Check Groq API key
    if not GROQ_API_KEY:
        logger.error(
            "⚠️ GROQ_API_KEY not set! Get one at https://console.groq.com/keys")
    else:
        logger.info("✅ Groq API key configured")

    if not FIREBASE_SERVICE_ACCOUNT:
        logger.warning("⚠️ FIREBASE_SERVICE_ACCOUNT not set! Push notifications will be disabled.")
    else:
        logger.info("✅ Firebase Service Account configured")

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


# ── Push Notification Models ──────────────────────────────────────────
class PushMessage(BaseModel):
    token: str
    title: str
    body: str
    icon: Optional[str] = "not_icon"
    color: Optional[str] = "#0052D4"


class PushNotificationRequest(BaseModel):
    messages: List[PushMessage]


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


# ── Interactive Search Flow Models ───────────────────────────────────────
class SearchOption(BaseModel):
    id: str
    label: str
    description: str
    specialty: Optional[str] = None
    code: Optional[str] = None
    rate: Optional[float] = None
    reasoning: Optional[str] = None
    rank: Optional[int] = None


class SearchStepResponse(BaseModel):
    step_number: int
    step_name: str
    description: str
    options: list[SearchOption]
    requires_user_selection: bool
    context: Optional[dict] = None


class AnalyzeQueryRequest(BaseModel):
    query: str


class AnalyzeQueryResponse(BaseModel):
    summary: str
    keywords: list[str]


class InteractiveSearchStartRequest(BaseModel):
    query: str
    procedure: str = ""
    disease: str = ""
    symptoms: list[str] = []
    patient_age: int = 0
    patient_gender: str = ""


class InteractiveSearchStartResponse(BaseModel):
    session_id: str
    query: str
    parsed_terms: list[str]
    current_step: SearchStepResponse
    message: str


class SelectionRequest(BaseModel):
    option_id: str
    notes: Optional[str] = None
    manual_package: Optional[dict] = None


class SelectionResponse(BaseModel):
    success: bool
    message: str
    next_step: Optional[SearchStepResponse] = None
    flow_complete: bool = False
    final_recommendation: Optional[dict] = None


class FlowStatusResponse(BaseModel):
    session_id: str
    query: str
    current_step_number: int
    total_steps: int
    selections_made: dict
    violations: list[str] = []
    flow_complete: bool


def _auto_advance_single_option_steps(flow: Any, packages: List[Dict]) -> None:
    """Auto-select steps that have only one non-manual option to reduce unnecessary taps."""
    from tools.smart_search_flow import process_step_selection

    safety_counter: int = 0
    while not flow.flow_complete and flow.current_step < len(flow.steps):
        if safety_counter > 20:
            break
        safety_counter = safety_counter + 1

        step = flow.steps[flow.current_step]
        options = step.options or []
        if len(options) != 1:
            break

        only_opt = options[0]
        opt_id = str(only_opt.get("id", ""))
        if opt_id.startswith("manual_add"):
            break

        success, _ = process_step_selection(flow, {"id": opt_id}, packages)
        if not success:
            break


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
            
            # Persist to DB for Vercel statelessness
            try:
                from memory.sqlite_store import AgentMemory
                mem = AgentMemory()
                mem.save_session(session_id, "ocr_pending", _pending_sessions[session_id])
            except Exception as e:
                logger.error(f"Failed to persist OCR session to DB: {e}")

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
        # Try to recover from DB
        try:
            from memory.sqlite_store import AgentMemory
            mem = AgentMemory()
            db_session = mem.get_session(session_id)
            if db_session:
                _pending_sessions[session_id] = db_session
            else:
                raise HTTPException(404, f"No pending session: {session_id}")
        except HTTPException:
            raise
        except Exception:
            raise HTTPException(404, f"No pending session found: {session_id}")

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
        
        # Remove from DB persistence
        try:
            from memory.sqlite_store import AgentMemory
            mem = AgentMemory()
            mem.delete_session(session_id)
        except:
            pass

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
        # Try to recover from DB
        try:
            from memory.sqlite_store import AgentMemory
            mem = AgentMemory()
            db_session = mem.get_session(session_id)
            if db_session:
                _pending_sessions[session_id] = db_session
            else:
                raise HTTPException(404, f"No pending session was found: {session_id}")
        except HTTPException:
            raise
        except Exception:
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
            
            # Persist new session to DB, remove old one
            try:
                from memory.sqlite_store import AgentMemory
                mem = AgentMemory()
                mem.delete_session(session_id)
                mem.save_session(new_session_id, "ocr_pending", _pending_sessions[new_session_id])
            except:
                pass

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


@app.post("/send-push")
async def send_push_notification(req: PushNotificationRequest, api_key: str = Depends(get_api_key)):
    """
    Secure proxy for sending FCM HTTP v1 notifications.
    Uses the server-side Service Account to negotiate tokens.
    """
    from config.settings import FIREBASE_SERVICE_ACCOUNT
    if not FIREBASE_SERVICE_ACCOUNT:
        raise HTTPException(
            500, "Push notifications not configured: Missing FIREBASE_SERVICE_ACCOUNT")

    try:
        sa_info = json.loads(FIREBASE_SERVICE_ACCOUNT)
        project_id = sa_info.get("project_id")

        # 1. Negotiate Access Token
        from google.oauth2 import service_account
        import google.auth.transport.requests as auth_requests

        scopes = ["https://www.googleapis.com/auth/firebase.messaging"]
        creds = service_account.Credentials.from_service_account_info(
            sa_info, scopes=scopes)

        request = auth_requests.Request()
        creds.refresh(request)
        access_token = creds.token

        # 2. Batch Send
        import requests
        results = []
        for msg in req.messages:
            fcm_url = f"https://fcm.googleapis.com/v1/projects/{project_id}/messages:send"
            payload = {
                "message": {
                    "token": msg.token,
                    "notification": {
                        "title": msg.title,
                        "body": msg.body,
                    },
                    "android": {
                        "notification": {
                            "icon": msg.icon,
                            "color": msg.color
                        }
                    },
                    "apns": {
                        "payload": {
                            "aps": {
                                "sound": "default"
                            }
                        }
                    },
                    "data": {
                        "click_action": "FLUTTER_NOTIFICATION_CLICK"
                    }
                }
            }
            res = requests.post(
                fcm_url,
                headers={
                    "Authorization": f"Bearer {access_token}",
                    "Content-Type": "application/json"
                },
                json=payload
            )
            results.append(
                {"token": msg.token, "ok": res.ok, "status": res.status_code})

        success_count = len([r for r in results if r["ok"]])
        return {"success": True, "sent": success_count, "total": len(req.messages), "details": results}

    except Exception as e:
        logger.exception("FCM Proxy Error")
        raise HTTPException(500, f"Failed to send notifications: {str(e)}")


# ── Smart Package Search ──────────────────────────────────────────────
_packages_cache: list[dict] = []
_robotic_cache: list[dict] = []
_spelling_vocab_cache: set[str] = set()


def _load_packages_cache():
    """Load packages into memory for fast search."""
    global _packages_cache, _robotic_cache
    if _packages_cache and _robotic_cache:
        return

    from config.settings import BASE_DIR
    import json

    def _load_json_from_candidates(filename: str) -> list[dict]:
        candidates = [
            BASE_DIR.parent / "assets" / filename,
            BASE_DIR / "assets" / filename,
            Path.cwd() / "assets" / filename,
        ]

        for candidate in candidates:
            try:
                if candidate.exists():
                    with open(candidate, "r", encoding="utf-8-sig") as f:
                        data = json.load(f)
                    logger.info("Loaded %s entries from %s",
                                len(data), candidate)
                    return data
            except Exception as exc:
                logger.warning("Failed loading %s from %s: %s",
                               filename, candidate, exc)

        logger.warning("Could not locate %s in any known asset path", filename)
        return []

    try:
        _packages_cache = _load_json_from_candidates("maa_packages.json")
        logger.info(f"Loaded {len(_packages_cache)} standard packages")
    except Exception as e:
        logger.warning(f"Failed to load packages: {e}")
        _packages_cache = []

    try:
        _robotic_cache = _load_json_from_candidates(
            "maa_robotic_surgeries.json")
        logger.info(f"Loaded {len(_robotic_cache)} robotic packages")
    except Exception as e:
        logger.warning(f"Failed to load robotic packages: {e}")
        _robotic_cache = []


def _normalize_search_text(value: str) -> str:
    """Normalize free-text terms for stable package matching."""
    text = (value or "").lower()
    text = text.replace("appendectomy", "appendicectomy")
    text = text.replace("gall bladder", "gallbladder")
    text = text.replace("lap chole", "laparoscopic cholecystectomy")
    text = text.replace("kidney transplant", "renal transplant")
    text = text.replace("liver tx", "liver transplant")
    return re.sub(r"\s+", " ", text).strip()


def _build_spelling_vocab() -> set[str]:
    """Build a cached medical vocabulary from package metadata for typo correction."""
    global _spelling_vocab_cache
    if _spelling_vocab_cache:
        return _spelling_vocab_cache

    _load_packages_cache()
    vocab: set[str] = set()
    for pkg in (_packages_cache + _robotic_cache):
        blob = " ".join([
            str(pkg.get("PACKAGE NAME", pkg.get("Package Name", ""))),
            str(pkg.get("SPECIALITY", pkg.get("Speciality", ""))),
            str(pkg.get("PACKAGE CATEGORY", pkg.get("PACKAGE TYPE", ""))),
            str(pkg.get("Procedure Sub Category", "")),
        ]).lower()
        for tok in re.findall(r"[a-z]{4,}", blob):
            vocab.add(tok)

    _spelling_vocab_cache = vocab
    return _spelling_vocab_cache


def _correct_query_terms_spelling(raw_terms: list[str]) -> tuple[list[str], dict[str, str]]:
    """Conservative spell correction. Only correct when confidence is high."""
    vocab = _build_spelling_vocab()
    if not vocab:
        return raw_terms, {}

    corrections: dict[str, str] = {}
    corrected_terms: list[str] = []

    for term in raw_terms:
        original = (term or "").strip()
        if not original:
            continue

        tokens = re.findall(r"[a-z0-9]+", original.lower())
        if not tokens:
            corrected_terms.append(original)
            continue

        corrected_tokens: list[str] = []
        changed = False

        for token in tokens:
            if len(token) < 4 or token.isdigit() or token in vocab:
                corrected_tokens.append(token)
                continue

            candidates = difflib.get_close_matches(
                token, list(vocab), n=3, cutoff=0.84)
            replacement = token
            for cand in candidates:
                # Guardrails: same first letter + small length delta to avoid wrong suggestions.
                if cand and cand[0] == token[0] and abs(len(cand) - len(token)) <= 2:
                    replacement = cand
                    break

            corrected_tokens.append(replacement)
            if replacement != token:
                changed = True

        corrected_term = " ".join(corrected_tokens).strip()
        if changed and corrected_term:
            corrections[original] = corrected_term
            corrected_terms.append(corrected_term)
        else:
            corrected_terms.append(original)

    return corrected_terms, corrections


def _search_packages_basic(query: str, limit: int = 50) -> List[Dict[str, Any]]:
    """Basic keyword search to pre-filter packages with medical synonym expansion."""
    _load_packages_cache()
    query_lower = _normalize_search_text(query)
    terms = re.findall(r"[a-z0-9]+", query_lower)
    generic_terms = {
        "surgery", "surgical", "procedure", "management", "treatment", "operation", "package",
        "pain", "ache", "discomfort", "symptom", "disease",
    }
    filtered_terms = [t for t in terms if t not in generic_terms]
    if not filtered_terms:
        filtered_terms = terms
    normalized_query = " ".join(filtered_terms).strip()

    def _tokens(value: str) -> list[str]:
        return re.findall(r"[a-z0-9]+", value)

    def _has_term(term: str, text: str, tokens: list[str]) -> bool:
        if not term:
            return False
        if " " in term:
            return term in text
        return any(tok == term or tok.startswith(term) for tok in tokens)

    def _has_keyword(keyword: str, text: str, tokens: list[str]) -> bool:
        if not keyword:
            return False
        if " " in keyword:
            return keyword in text
        return any(tok == keyword or tok.startswith(keyword) for tok in tokens)

    def _normalize_words(value: str) -> str:
        cleaned = " ".join((value or "").lower().replace(
            "/", " ").replace("-", " ").split())
        return f" {cleaned} "

    from tools.medical_knowledge import get_specialties_for_term, get_clinical_pathway
    intent_specialties = {s.lower().strip()
                          for s in get_specialties_for_term(query_lower)}
    pathway = get_clinical_pathway(query_lower)
    if pathway:
        pathway_specialty = pathway.get("specialty")
        if isinstance(pathway_specialty, list):
            for spec in pathway_specialty:
                spec_norm = str(spec).strip().lower()
                if spec_norm:
                    intent_specialties.add(spec_norm)
        elif isinstance(pathway_specialty, str):
            spec_norm = pathway_specialty.strip().lower()
            if spec_norm:
                intent_specialties.add(spec_norm)

        for step in pathway.get("steps", []):
            step_spec = str(step.get("specialty", "")).strip().lower()
            if step_spec:
                intent_specialties.add(step_spec)

    symptom_query_indicators = {
        "pain", "fever", "breath", "cough", "bleeding", "swelling", "weakness", "dizziness", "attack",
    }
    strict_intent_query = bool(intent_specialties) and any(
        ind in query_lower for ind in symptom_query_indicators)

    # Expand search terms with medical synonyms and related terms
    expanded_terms = set(filtered_terms)
    active_related_terms = set()
    SYMPTOM_EXPANSIONS = {
        "chest pain": ["coronary", "angiography", "cardiac", "heart", "ptca", "cabg", "thrombolysis", "mi"],
        "heart attack": ["mi", "myocardial", "thrombolysis", "ptca", "coronary", "stemi", "nstemi", "cabg"],
        "angioplasty": ["ptca", "coronary", "stent", "cardiology", "pci"],
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
        "appendix": ["appendicitis", "appendicectomy", "appendicular"],
        "appendicitis": ["appendix", "appendicectomy", "appendicular"],
        "hernia": ["inguinal", "ventral", "umbilical", "hernia repair"],
        "cholecystectomy": ["gallbladder", "cholecystitis", "laparoscopic"],
        "thyroid": ["thyroidectomy", "endocrine", "ent"],
        "thyroid surgery": ["thyroidectomy", "endocrine", "ent"],
        "renal transplant": ["kidney transplant", "transplant", "nephrology", "urology"],
        "kidney transplant": ["renal transplant", "transplant", "nephrology", "urology"],
        "liver transplant": ["hepatic transplant", "transplant", "surgical gastroenterology", "gastroenterology"],
        "blood transfusion": ["transfusion", "platelet", "packed", "whole blood", "component"],
    }

    for symptom, related in SYMPTOM_EXPANSIONS.items():
        if symptom in query_lower or any(t in symptom for t in filtered_terms):
            expanded_terms.update(related)
            active_related_terms.update(related)

    PHRASE_PRIORITY = {
        "angioplasty": ["ptca", "coronary angioplasty", "coronary", "angioplasty", "pci"],
        "appendix": ["appendicectomy", "appendicitis", "appendicular", "appendix"],
        "cholecystectomy": ["cholecystectomy", "gallbladder", "cholecyst"],
        "hernia": ["hernia", "inguinal", "umbilical", "ventral"],
        "thyroid": ["thyroid", "thyroidectomy"],
        "renal transplant": ["renal transplant", "kidney transplant", "transplant"],
        "liver transplant": ["liver transplant", "hepatic transplant", "transplant"],
        "blood transfusion": ["blood transfusion", "platelet transfusion", "whole blood", "component"],
    }

    CONDITION_SPECIALTY_HINTS = {
        "angioplasty": ["cardiology", "interventional cardiology", "cath lab"],
        "appendix": ["general surgery", "surgical gastroenterology", "laparoscopic"],
    }

    all_packages = _packages_cache + _robotic_cache
    scored: list[Any] = []

    for pkg in all_packages:
        name = _normalize_search_text(
            str(pkg.get("PACKAGE NAME", pkg.get("Package Name", ""))))
        code = _normalize_search_text(str(pkg.get("PACKAGE CODE", "")))
        spec = _normalize_search_text(
            str(pkg.get("SPECIALITY", pkg.get("Speciality", ""))))
        name_tokens = _tokens(name)
        code_tokens = _tokens(code)
        spec_tokens = _tokens(spec)

        score: int = 0
        term_hits_name: int = 0
        term_hits_code: int = 0
        term_hits_spec: int = 0

        # Score original terms higher
        for term in filtered_terms:
            if _has_term(term, code, code_tokens):
                score = score + 15
                term_hits_code = term_hits_code + 1
            if _has_term(term, name, name_tokens):
                score = score + 10
                term_hits_name = term_hits_name + 1
            if _has_term(term, spec, spec_tokens):
                score = score + 5
                term_hits_spec = term_hits_spec + 1

        # Generic direct-match priority: most exact textual matches should always rank first.
        if normalized_query:
            if normalized_query in code:
                score = score + 90
            if normalized_query in name:
                score = score + 75
            if normalized_query in spec:
                score = score + 30

        exact_priority = 0
        if normalized_query:
            if normalized_query == code:
                exact_priority = 5
            elif normalized_query == name:
                exact_priority = 4
            elif f" {normalized_query} " in f" {name} ":
                exact_priority = 3
            elif f" {normalized_query} " in f" {spec} ":
                exact_priority = 2

        if len(filtered_terms) == 1:
            token = filtered_terms[0]
            if any(tok == token for tok in name_tokens):
                exact_priority = max(exact_priority, 4)
            elif any(tok == token for tok in code_tokens):
                exact_priority = max(exact_priority, 4)
            elif any(tok == token for tok in spec_tokens):
                exact_priority = max(exact_priority, 3)

        total_terms = len(filtered_terms)
        if total_terms > 0:
            if term_hits_name == total_terms:
                score = score + 40
            elif term_hits_name > 0:
                score = score + (term_hits_name * 8)

            if term_hits_code == total_terms:
                score = score + 45
            elif term_hits_code > 0:
                score = score + (term_hits_code * 10)

            # For one-word searches, elevate exact name matches more aggressively.
            if total_terms == 1 and _has_term(filtered_terms[0], name, name_tokens):
                score = score + 25

            # Reduce expansion-only matches so direct textual matches stay above them.
            if (term_hits_name + term_hits_code + term_hits_spec) == 0:
                score = score - 12

        # Score expanded terms
        for term in expanded_terms:
            if term not in filtered_terms:  # Don't double count
                if _has_term(term, code, code_tokens):
                    score = score + 8
                if _has_term(term, name, name_tokens):
                    score = score + 5
                if _has_term(term, spec, spec_tokens):
                    score = score + 3

        # Strong phrase-level boosts for high-precision terms.
        for trigger, keywords in PHRASE_PRIORITY.items():
            if trigger in query_lower:
                if any(_has_keyword(k, name, name_tokens) for k in keywords):
                    score = score + 40

        # Mild penalty for clearly off-specialty matches on high-precision triggers.
        for trigger, spec_hints in CONDITION_SPECIALTY_HINTS.items():
            if trigger in query_lower and not any(_has_keyword(h, spec, spec_tokens) for h in spec_hints):
                score = score - 8

        # Generic symptom-intent anchoring: for symptom-like queries, prefer packages matching inferred specialties.
        spec_norm = _normalize_words(spec)
        spec_matches_intent = any(_normalize_words(
            s) in spec_norm for s in intent_specialties)
        related_match = any(
            _has_term(rel, name, name_tokens)
            or _has_term(rel, spec, spec_tokens)
            or _has_term(rel, code, code_tokens)
            for rel in active_related_terms
        )
        if spec_matches_intent:
            score = score + 10
        elif strict_intent_query:
            score = score - 18
            if not related_match:
                continue

        # Keep angioplasty refinement as secondary to generic direct-match ordering.
        if "angioplasty" in query_lower and "peripheral" not in query_lower:
            if _has_keyword("ptca", name, name_tokens) or _has_keyword("coronary", name, name_tokens):
                score = score + 16
            if _has_keyword("peripheral", name, name_tokens) or _has_keyword("peripheral", spec, spec_tokens):
                score = score - 10

        if score > 0:
            scored.append((exact_priority, score, pkg))

    scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
    return [p for _, _, p in scored[:limit]]


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


def _identify_package_type(pkg: dict) -> dict[str, bool]:
    """Identify package type based on package data."""
    name = pkg.get("PACKAGE NAME", pkg.get("Package Name", "")).upper()
    rate_val = pkg.get("RATE", pkg.get("Rate", 0))
    try:
        rate = float(str(rate_val).replace(",", "").strip()) if rate_val else 0.0
    except Exception:
        rate = 0.0
    implant_field = str(pkg.get("IMPLANT PACKAGE", pkg.get("IMPLANT", "NO IMPLANT")) or "NO IMPLANT").upper()
    cat = str(pkg.get("PACKAGE CATEGORY", pkg.get("PACKAGE TYPE", pkg.get("Procedure Type", ""))) or "").upper()

    is_implant = 'IMPLANT' in implant_field and 'NO IMPLANT' not in implant_field
    is_standalone = 'STAND-ALONE' in name or 'STAND ALONE' in name or 'STAND ALONE' in cat or 'STAND-ALONE' in cat
    is_addon = '[ADD-ON' in name or '[ADD ON' in name or 'ADDON' in name or 'ADD-ON' in cat or 'ADD ON' in cat or 'ADDON' in cat

    return {
        'is_surgical': rate > 0 and ('[REGULAR PROCEDURE]' in name or 'REGULAR PKG' in cat),
        'is_medical_management': rate == 0 and not is_addon,
        'is_standalone': is_standalone,
        'is_addon': is_addon,
        'is_implant': is_implant or cat == 'IMP',
        'is_extended_los': 'EXTENDED LOS' in name,
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
- For heart attack → Thrombolysis or PTCA as primary

MANDATORY DOCTOR MINDSET:
- Think as consulting physician, not keyword matcher.
- Use differential diagnosis before final package selection.
- Prioritize patient safety and standard-of-care pathway.
- Reject unsafe combinations even if keyword match exists.
- If add-ons are requested after commas, treat first term as chief diagnosis/procedure and remaining terms as supportive add-ons only."""
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
- In doctor_summary, include a brief 3-step clinical chain:
    1) likely diagnosis
    2) treatment pathway
    3) why selected package/add-ons are medically justified

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


def _split_query_terms(raw_query: str) -> list[str]:
    normalized = (raw_query or "").replace(";", ",").replace("|", ",")
    return [part.strip() for part in normalized.split(",") if part.strip()]


def _append_unique_term(target: list[str], value: str) -> None:
    val = (value or "").strip()
    if not val:
        return

    val_norm = val.lower()
    if any(existing.lower() == val_norm for existing in target):
        return
    target.append(val)


def _expand_implicit_addon_terms(main_term: str) -> list[str]:
    term = (main_term or "").lower()
    implied: list[str] = []

    # Clinical mapping for common disease-to-supportive-package add-ons.
    implicit_addons_map = {
        "anemia": ["blood transfusion"],
        "anaemia": ["blood transfusion"],
        "heart attack": ["blood transfusion"],
        "myocardial infarction": ["blood transfusion"],
        "mi": ["blood transfusion"],
        "hemorrhage": ["blood transfusion"],
        "haemorrhage": ["blood transfusion"],
    }

    for key, addon_terms in implicit_addons_map.items():
        if key in term:
            for addon in addon_terms:
                _append_unique_term(implied, addon)

    return implied


def _is_transfusion_term(term: str) -> bool:
    t = (term or "").lower().strip()
    return "transfusion" in t or "blood" in t


def _build_raw_package_row(pkg: dict, ai_selected: bool = False) -> dict:
    return {
        "code": pkg.get("PACKAGE CODE", ""),
        "name": pkg.get("PACKAGE NAME", pkg.get("Package Name", "")),
        "rate": pkg.get("RATE", pkg.get("Rate", 0)),
        "speciality": pkg.get("SPECIALITY", pkg.get("Speciality", "")),
        "ai_selected": ai_selected,
    }


def _prioritize_exact_main_term_first(packages: list[dict], main_term: str) -> list[dict]:
    """Keep exact main-term matches at the top while preserving existing order for ties."""
    normalized_term = _normalize_search_text(main_term or "")
    if not normalized_term:
        return packages

    term_tokens = re.findall(r"[a-z0-9]+", normalized_term)
    if not term_tokens:
        return packages

    padded_exact_phrase = f" {normalized_term} "

    def _rank(pkg: dict) -> int:
        full_name = str(pkg.get("PACKAGE NAME", pkg.get("Package Name", "")))
        primary_name = _normalize_search_text(full_name.split("|")[0] if "|" in full_name else full_name)
        full_name_norm = _normalize_search_text(full_name)
        
        primary_padded = f" {primary_name} "
        full_padded = f" {full_name_norm} "

        name_tokens = re.findall(r"[a-z0-9]+", full_name_norm)
        primary_tokens = re.findall(r"[a-z0-9]+", primary_name)

        # Rank 0: Exact phrase in the PRIMARY name (e.g. searching "blood" gets a package named "Blood...")
        if padded_exact_phrase in primary_padded:
            return 0
        if len(term_tokens) == 1 and any(tok == term_tokens[0] for tok in primary_tokens):
            return 1
        
        # Rank 2: Exact phrase anywhere in the full description (e.g. "Severe Anemia | blood transfusion")
        if padded_exact_phrase in full_padded:
            return 2
        
        if all(tok in name_tokens for tok in term_tokens):
            return 3
        if any(tok in name_tokens for tok in term_tokens):
            return 4
        if normalized_term in full_name_norm:
            return 5
        return 6

    return sorted(packages, key=_rank)

def _classify_input_intent(term_specialties_map: dict[str, list[str]]) -> dict[str, str]:
    """
    Classify terms as 'Surgical', 'Medical', or 'Unknown'.
    First uses internal deterministic logic, then queries Groq AI.
    """
    from groq import Groq
    from config.settings import GROQ_MODEL, GROQ_API_KEY
    from tools.medical_knowledge import is_medical_management_term, is_surgical_term
    
    if not GROQ_API_KEY or not term_specialties_map:
        return {t: "Unknown" for t in term_specialties_map}

    intents = {}
    terms_for_ai = {}

    for term, specs in term_specialties_map.items():
        term_lower = term.lower()
        if is_medical_management_term(term) or term_lower in ["sepsis", "anemia", "fever", "infection", "shock"]:
            intents[term] = "Medical"
        elif is_surgical_term(term):
            intents[term] = "Surgical"
        else:
            terms_for_ai[term] = specs
            
    if not terms_for_ai:
        return intents

    input_details = []
    for term, specs in terms_for_ai.items():
        spec_str = ", ".join(specs) if specs else "No specific specialty"
        input_details.append(f"- Term: '{term}' (Matching Package Specialties: {spec_str})")
        
    inputs_str = "\n".join(input_details)

    prompt = f"""You are a medical categorization assistant for the MAA Yojana health insurance scheme.
Classify each of the following medical terms/inputs as either "Surgical" (requires an operation or procedure) or "Medical" (medical management, conservative treatment, general disease with no procedure).

Use the provided "Matching Package Specialties" to guide your decision.
- Surgical Specialties: General Surgery, Orthopaedics, Cardiac Surgery, Neurosurgery, Paediatric Surgery.
- Medical Specialties: General Medicine, Medical Oncology, Cardiology, Paediatric Medical Management, Nephrology, Neurology, Pulmonology.

IMPORTANT MAA YOJANA RULE: "Blood", "Blood Transfusion", "ICU", "Extended LOS", implants, and supportive care MUST always be classified as "Medical".

Inputs to classify:
{inputs_str}

Respond ONLY with a valid JSON object where keys are the exact input terms and values are either "Surgical" or "Medical". Do not include any other text or explanation.

Example:
{{"lap chole": "Surgical", "dengue": "Medical", "fever": "Medical", "appendectomy": "Surgical"}}
"""
    try:
        client = Groq(api_key=GROQ_API_KEY)
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            response_format={"type": "json_object"}
        )
        content = response.choices[0].message.content
        import json
        llm_intents = json.loads(content)
        intents.update(llm_intents)
        return intents
    except Exception as e:
        logger.warning(f"AI classification failed: {e}")
        for t in terms_for_ai:
            intents[t] = "Unknown"
        return intents

def _check_intent_rule_violation(intents: dict[str, str]) -> str | None:
    """Check if the intents violate Rule 1 (mix of Surgical and Medical)."""
    if not intents:
        return None
    has_surgical = any(v.lower() == "surgical" for v in intents.values())
    has_medical = any(v.lower() == "medical" for v in intents.values())
    
    if has_surgical and has_medical:
        surgical_terms = [k for k, v in intents.items() if v.lower() == "surgical"]
        medical_terms = [k for k, v in intents.items() if v.lower() == "medical"]
        return f"Rule 1 VIOLATION: Cannot combine surgical procedures ({', '.join(surgical_terms)}) with medical management conditions ({', '.join(medical_terms)})."
    return None

def _get_ai_related_terms(terms: list[str]) -> list[str]:
    """
    Use Groq AI to suggest 3-5 related medical conditions or treatments for the input terms.
    E.g. if terms = ['blood transfusion'], it might suggest ['anemia', 'hemorrhage'].
    """
    from groq import Groq
    if not GROQ_API_KEY or not terms:
        return []
    
    prompt = f"""You are a senior reporting doctor.
Given the patient's search inputs: {", ".join(terms)}

What are 3 to 5 related medical conditions or supportive treatments you would suggest exploring packages for?
Return a simple JSON array of strings. Do not include the original terms.

Example for 'blood transfusion': ["anemia", "hemorrhage", "thalassemia"]
"""
    try:
        client = Groq(api_key=GROQ_API_KEY)
        response = client.chat.completions.create(
            model="llama3-8b-8192",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            response_format={"type": "json_object"}  # Workaround: instruct might be better as array, but using json_object needs an object.
        )
        content = response.choices[0].message.content
        import json
        data = json.loads(content)
        # Handle if the LLM wraps it in a dict like {"suggestions": [...]}
        if isinstance(data, dict):
            for v in data.values():
                if isinstance(v, list):
                    return [str(item) for item in v]
        elif isinstance(data, list):
            return [str(item) for item in data]
        return []
    except Exception as e:
        logger.warning(f"AI related terms suggestion failed: {e}")
        return []

@app.post("/smart-search", response_model=SmartSearchResponse)
async def smart_search(request: SmartSearchRequest):
    """AI-powered smart package search with doctor reasoning."""
    query_terms = _split_query_terms(request.query)
    procedure_term = (request.procedure or "").strip()
    disease_term = (request.disease or "").strip()

    if procedure_term:
        _append_unique_term(query_terms, procedure_term)
    if disease_term:
        _append_unique_term(query_terms, disease_term)

    query_terms, spelling_corrections = _correct_query_terms_spelling(
        query_terms)

    if query_terms:
        main_search_term = query_terms[0]
        addon_search_terms = query_terms[1:]
    else:
        main_search_term = ""
        addon_search_terms = []

    for implied_addon in _expand_implicit_addon_terms(main_search_term):
        _append_unique_term(addon_search_terms, implied_addon)

    combined_query = ", ".join(
        [main_search_term, *addon_search_terms]).strip(", ")
    if not combined_query:
        return SmartSearchResponse(
            doctor_reasoning="Please provide a procedure name, disease, or search query.",
            raw_packages=[]
        )

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

    effective_limit = max(25, min(100, request.limit))

    # Search for main packages with higher limit and preserve ordering.
    relevant_packages = _search_packages_basic(
        main_search_term, limit=effective_limit)
    relevant_packages = _prioritize_exact_main_term_first(
        relevant_packages, main_search_term)

    # Also search add-on packages per add-on term.
    addon_packages = []
    addon_candidates_by_term: dict[str, list[dict]] = {}
    for addon_term in addon_search_terms:
        addon_results = _search_packages_basic(addon_term, limit=30)
        addon_candidates_by_term[addon_term] = addon_results
        addon_packages.extend(addon_results)

    # ----- AI Intent Classification Check (with Package Specialties) -----
    if len(query_terms) > 1:
        term_specialties = {}
        # Main term specs
        specs = set()
        for p in relevant_packages[:15]:
            spec = str(p.get("SPECIALITY", "")).strip()
            if spec: specs.add(spec)
        term_specialties[main_search_term] = list(specs)[:3]
        
        # Addon terms specs
        for t in addon_search_terms:
            pkgs = addon_candidates_by_term.get(t, [])
            specs = set()
            for p in pkgs[:15]:
                spec = str(p.get("SPECIALITY", "")).strip()
                if spec: specs.add(spec)
            term_specialties[t] = list(specs)[:3]
            
        input_intents = _classify_input_intent(term_specialties)
        violation = _check_intent_rule_violation(input_intents)
        if violation:
            return SmartSearchResponse(
                doctor_reasoning=f"⚠️ {violation}\n\nPlease revise your search. Under MAA Yojana rules, surgical procedures and medical management packages with ₹0 rate cannot be booked together.",
                raw_packages=[],
                blocked_rules=[violation],
                approval_likelihood="REJECTED"
            )
    # ---------------------------------------------------------------------

    # Combine but keep main packages first
    all_relevant_packages = relevant_packages.copy()
    seen_codes = {pkg.get("PACKAGE CODE", "") for pkg in relevant_packages}
    for pkg in addon_packages:
        code = pkg.get("PACKAGE CODE", "")
        if code not in seen_codes:
            all_relevant_packages.append(pkg)
            seen_codes.add(code)

    if not relevant_packages:
        correction_hint = ""
        if spelling_corrections:
            correction_hint = " Did you mean: " + \
                ", ".join(spelling_corrections.values()) + "?"
        return SmartSearchResponse(
            doctor_reasoning=f"No packages found matching '{main_search_term}'. Please try different keywords or check spelling.{correction_hint}",
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
You MUST prioritize add-ons from the requested add-on terms when clinically safe.
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

        main_pkg_type = None
        standalone_main_selected = False

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
                    standalone_main_selected = bool(
                        main_pkg_type.get("is_standalone"))

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

        # Fallback main package when AI response has no valid code in candidate list.
        if result.main_package is None and relevant_packages:
            fallback_pkg = relevant_packages[0]
            main_code = fallback_pkg.get("PACKAGE CODE", "")
            selected_codes.add(main_code)
            pkg_name = fallback_pkg.get(
                "PACKAGE NAME", fallback_pkg.get("Package Name", ""))
            pkg_rate = float(fallback_pkg.get(
                "RATE", fallback_pkg.get("Rate", 0)))
            pkg_implant = fallback_pkg.get("IMPLANT PACKAGE", "")
            main_pkg_type = _identify_package_type(
                pkg_name, pkg_rate, pkg_implant)
            standalone_main_selected = bool(main_pkg_type.get("is_standalone"))
            result.main_package = PackageResultModel(
                package_code=main_code,
                package_name=pkg_name,
                rate=pkg_rate,
                speciality=fallback_pkg.get(
                    "SPECIALITY", fallback_pkg.get("Speciality", "")),
                category=fallback_pkg.get(
                    "PACKAGE CATEGORY", fallback_pkg.get("PACKAGE TYPE", "")),
                is_main=True,
                medical_reason="Best clinical match from prioritized main query results",
            )

        # Stand-alone package must remain exclusive in smart-search too.
        if standalone_main_selected:
            result.auto_implant = None
            result.suggested_addons = []
        # Find implant (with validation if main package exists)
        implant_code = ai_result.get("implant_code")
        if not standalone_main_selected and implant_code and implant_code != "null":
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
        for addon in ([] if standalone_main_selected else ai_result.get("addons", [])[:5]):  # Max 5 addons
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

        # Deterministic add-on fallback: enforce user-requested add-on terms in order.
        # This ensures "main, addon1, addon2" behavior is visible in app results.
        for addon_term in ([] if standalone_main_selected else addon_search_terms):
            candidates = addon_candidates_by_term.get(addon_term, [])
            for pkg in candidates:
                addon_code = pkg.get("PACKAGE CODE", "")
                if not addon_code or addon_code in selected_codes:
                    continue

                pkg_name = pkg.get("PACKAGE NAME", pkg.get("Package Name", ""))
                pkg_rate = pkg.get("RATE", pkg.get("Rate", 0))
                addon_type = _identify_package_type(pkg)

                validation_error = None
                if main_pkg_type:
                    validation_error = _validate_package_combination(
                        main_pkg_type, addon_type, addon_code)

                if validation_error:
                    # Clinical override for transfusion requests (e.g., anemia support).
                    if _is_transfusion_term(addon_term):
                        validation_error = None
                    else:
                        validation_errors.append(validation_error)
                        continue

                selected_codes.add(addon_code)
                reason_prefix = "Requested add-on"
                if addon_term.lower() in {"blood transfusion", "transfusion"}:
                    reason_prefix = "Clinical add-on"

                result.suggested_addons.append(PackageResultModel(
                    package_code=addon_code,
                    package_name=pkg_name,
                    rate=pkg_rate,
                    speciality=pkg.get(
                        "SPECIALITY", pkg.get("Speciality", "")),
                    category=pkg.get("PACKAGE CATEGORY",
                                     pkg.get("PACKAGE TYPE", "")),
                    is_addon=True,
                    medical_reason=f"{reason_prefix}: {addon_term}",
                ))
                break

        # Hard fallback: if transfusion was requested but still no add-on picked, force first safe visible candidate.
        if (not standalone_main_selected) and any(_is_transfusion_term(t) for t in addon_search_terms) and not result.suggested_addons:
            for transfusion_term in addon_search_terms:
                if not _is_transfusion_term(transfusion_term):
                    continue
                candidates = addon_candidates_by_term.get(transfusion_term, [])
                if not candidates:
                    continue
                pkg = candidates[0]
                addon_code = pkg.get("PACKAGE CODE", "")
                if not addon_code:
                    continue
                selected_codes.add(addon_code)
                result.suggested_addons.append(PackageResultModel(
                    package_code=addon_code,
                    package_name=pkg.get(
                        "PACKAGE NAME", pkg.get("Package Name", "")),
                    rate=float(pkg.get("RATE", pkg.get("Rate", 0))),
                    speciality=pkg.get(
                        "SPECIALITY", pkg.get("Speciality", "")),
                    category=pkg.get("PACKAGE CATEGORY",
                                     pkg.get("PACKAGE TYPE", "")),
                    is_addon=True,
                    medical_reason="Clinical add-on fallback for transfusion support",
                ))
                break

        # Add alternatives to raw_packages (max 2)
        for alt_code in ai_result.get("alternative_codes", [])[:2]:
            if alt_code and alt_code not in selected_codes:
                selected_codes.add(alt_code)

        # Curated output ordering: main -> implant -> add-ons -> limited alternatives.
        ordered_curated_codes: list[str] = []
        if result.main_package and result.main_package.package_code:
            ordered_curated_codes.append(result.main_package.package_code)
        if result.auto_implant and result.auto_implant.package_code:
            ordered_curated_codes.append(result.auto_implant.package_code)
        ordered_curated_codes.extend(
            [a.package_code for a in result.suggested_addons if a.package_code]
        )

        # Include at most 6 alternatives after primary and add-ons.
        alt_limit = 6
        alternatives_added: int = 0
        for pkg in all_relevant_packages:
            code = pkg.get("PACKAGE CODE", "")
            if not code or code in ordered_curated_codes:
                continue
            if alternatives_added >= alt_limit:
                break
            ordered_curated_codes.append(code)
            alternatives_added = alternatives_added + 1

        package_by_code = {
            pkg.get("PACKAGE CODE", ""): pkg
            for pkg in all_relevant_packages
            if pkg.get("PACKAGE CODE", "")
        }

        result.raw_packages = []
        for code in ordered_curated_codes:
            pkg = package_by_code.get(code)
            if not pkg:
                continue
            result.raw_packages.append(
                _build_raw_package_row(pkg, ai_selected=code in selected_codes)
            )

        # Set blocked rules and adjust approval likelihood if violations found
        result.blocked_rules = validation_errors
        if validation_errors:
            if result.approval_likelihood not in ["REJECTED", "LOW"]:
                result.approval_likelihood = "LOW"
            # Append validation summary to doctor reasoning
            result.doctor_reasoning += "\n\n⚠️ RULE VIOLATIONS DETECTED:\n" + \
                "\n".join(f"• {err}" for err in validation_errors[:5])

        if spelling_corrections:
            correction_text = ", ".join(
                f"{src} -> {dst}" for src, dst in spelling_corrections.items()
            )
            result.doctor_reasoning = f"Spelling suggestion applied: {correction_text}\n\n{result.doctor_reasoning}"

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


# ════════════════════════════════════════════════════════════════════════════════
# INTERACTIVE MULTI-STEP SMART SEARCH ENDPOINTS
# ════════════════════════════════════════════════════════════════════════════════

async def _get_or_reconstruct_flow(session_id: str) -> Tuple[Any, List[Dict], Dict]:
    """Helper to load flow from memory OR reconstruct from SQLite on Vercel."""
    from memory.sqlite_store import AgentMemory
    from tools.smart_search_flow import reconstruct_flow_from_state
    
    mem = AgentMemory()
    
    # Try in-memory first (fastest)
    if session_id in _interactive_flows:
        data = _interactive_flows[session_id]
        return data["flow"], data["all_packages"], data.get("per_term_packages", {})
    
    # Try reconstruction from DB
    session_data = mem.get_session(session_id)
    if not session_data:
        logger.warning(f"Session {session_id} not found in memory or DB")
        raise HTTPException(404, "Session not found or expired. Please start a new search.")
    
    query = session_data.get("query", "")
    addon_terms = session_data.get("addon_terms", [])
    selections = session_data.get("selections_list", []) # List of selection dicts
    
    # Re-run search (fast due to _packages_cache)
    _load_packages_cache()
    all_packages = _packages_cache + _robotic_cache
    
    main_term = query
    matching_packages = _search_packages_basic(main_term, limit=200)
    matching_packages = _prioritize_exact_main_term_first(matching_packages, main_term)
    
    per_term_packages: dict[str, list] = {main_term: matching_packages}
    for term in addon_terms:
        term_pkgs = _search_packages_basic(term, limit=200)
        term_pkgs = _prioritize_exact_main_term_first(term_pkgs, term)
        per_term_packages[term] = term_pkgs
        
    flow = reconstruct_flow_from_state(
        query=query,
        addon_terms=addon_terms,
        selections=selections,
        matching_packages=matching_packages,
        all_packages=all_packages,
        per_term_packages=per_term_packages
    )
    
    # Sync in-memory for future requests on this instance
    _interactive_flows[session_id] = {
        "flow": flow,
        "packages": matching_packages,
        "all_packages": all_packages,
        "per_term_packages": per_term_packages,
        "created_at": time.time(),
        "selections_list": selections
    }
    
    return flow, all_packages, per_term_packages


def _update_session_database(session_id: str, flow: Any, selections_list: List[Dict]):
    """Sync the in-memory flow state back to the persistent SQLite DB."""
    try:
        from memory.sqlite_store import AgentMemory
        mem = AgentMemory()
        
        # We only store query, terms, and the order of selections
        # We re-run the search on load to keep DB size small
        session_record = {
            "query": flow.query,
            "addon_terms": [t for t in flow.parsed_terms if t != flow.query],
            "selections_list": selections_list,
            "flow_complete": flow.flow_complete
        }
        mem.save_session(session_id, "interactive_search", session_record)
    except Exception as e:
        logger.error(f"Failed to sync session to DB: {e}")


@app.post("/interactive-search/analyze-query", response_model=AnalyzeQueryResponse)
async def analyze_interactive_query(request: AnalyzeQueryRequest):
    """
    Use Groq AI to summarize a free-text patient history and extract 1-3 highly relevant clinical keywords
    that match potential medical packages or procedures.
    """
    from groq import Groq
    from config.settings import GROQ_API_KEY
    import json
    
    if not GROQ_API_KEY:
        raise HTTPException(500, "Groq API key not configured")
        
    client = Groq(api_key=GROQ_API_KEY)
    
    prompt = f"""
You are an expert PMJAY medical AI assistant.
The user has provided an unstructured medical history, symptoms, procedures, abbreviations, or diagnoses.
Your job is to:
1. Write a concise, professional summary of the case.
2. Extract exactly 1-3 highly relevant clinical keywords (e.g., precise procedure names, diagnoses) that will be used to search for medical packages.
3. IMPORTANT DEDUPLICATION: Do NOT include redundant terms, synonyms, or full names alongside their acronyms (e.g. if returning "CABG", do NOT also return "Coronary Artery Bypass Grafting"). Also do not extract a symptom if you also extract the definitive procedure treating it (e.g., if extracting "CABG", drop "Chest Pain"). Deduplicate into the core 1-2 root conditions/procedures.

User input: "{request.query}"

Return ONLY a valid JSON object with the following schema:
{{
  "summary": "String containing the clinical summary",
  "keywords": ["Keyword1", "Keyword2"]
}}
"""
    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            response_format={"type": "json_object"}
        )
        
        content = response.choices[0].message.content
        parsed = json.loads(content)
        
        summary = parsed.get("summary", "No summary generated.")
        keywords = parsed.get("keywords", [])
        
        if not keywords:
            # Fallback if AI fails to extract
            keywords = [request.query[:50]]
            
        return AnalyzeQueryResponse(summary=summary, keywords=keywords)
    except Exception as e:
        logger.error(f"Error analyzing query with Groq: {e}")
        # Fallback
        return AnalyzeQueryResponse(
            summary=f"Query analysis failed but you can proceed. Original query: {request.query}",
            keywords=[request.query]
        )


@app.post("/interactive-search/start", response_model=InteractiveSearchStartResponse)
async def start_interactive_search(request: InteractiveSearchStartRequest):
    """
    Start an interactive multi-step package search flow.

    Returns:
    - session_id: Use this for subsequent steps
    - current_step: First step (usually procedure/approach selection)
    """
    from tools.smart_search_flow import (
        build_search_flow,
        _split_query_terms,
        advance_past_empty_optional_steps,
    )
    from tools.medical_knowledge import get_specialties_for_term

    # Parse query terms
    query_terms = _split_query_terms(request.query)
    if request.procedure:
        query_terms.insert(0, request.procedure)
    if request.disease and request.disease not in query_terms:
        query_terms.append(request.disease)

    query_terms, spelling_corrections = _correct_query_terms_spelling(
        query_terms)

    if not query_terms:
        raise HTTPException(
            400, "Please provide a query, procedure, or disease")

    # AI intent check moved after package search natively uses specialties

    main_term = query_terms[0]
    addon_terms = query_terms[1:]

    # Search for matching packages for the MAIN term
    matching_packages = _search_packages_basic(main_term, limit=200)
    matching_packages = _prioritize_exact_main_term_first(matching_packages, main_term)
    if request.disease:
        disease_packages = _search_packages_basic(request.disease, limit=120)
        seen = {p.get("PACKAGE CODE", "") for p in matching_packages}
        for pkg in disease_packages:
            if pkg.get("PACKAGE CODE", "") not in seen:
                matching_packages.append(pkg)

    # Specialty fallback for main term
    if not matching_packages:
        _load_packages_cache()
        all_pkgs = _packages_cache + _robotic_cache
        if not all_pkgs:
            raise HTTPException(
                503,
                "Package datasets are not loaded on server. Ensure assets/maa_packages.json and assets/maa_robotic_surgeries.json are present.",
            )
        mapped_specs = get_specialties_for_term(main_term)
        if mapped_specs:
            spec_matches: list[Any] = []
            seen_codes = set()
            for pkg in all_pkgs:
                spec = str(
                    pkg.get("SPECIALITY", pkg.get("Speciality", ""))).lower()
                code = pkg.get("PACKAGE CODE", "")
                if not code or code in seen_codes:
                    continue
                if any(ms.lower() in spec for ms in mapped_specs):
                    spec_matches.append(pkg)
                    seen_codes.add(code)
            matching_packages = spec_matches[:250]

    if not matching_packages:
        raise HTTPException(404, f"No packages found for: {main_term}")

    # Search for matching packages per addon term
    _load_packages_cache()
    all_packages = _packages_cache + _robotic_cache

    per_term_packages: dict[str, list] = {main_term: matching_packages}
    for term in addon_terms:
        term_pkgs = _search_packages_basic(term, limit=200)
        term_pkgs = _prioritize_exact_main_term_first(term_pkgs, term)
        if not term_pkgs:
            # Fallback: search from all packages by keyword
            term_lower = term.lower().strip()
            import re as _re
            term_tokens = [t for t in _re.findall(r"[a-z0-9]+", term_lower) if len(t) > 2]
            if term_tokens:
                for pkg in all_packages:
                    name = str(pkg.get("PACKAGE NAME", "")).lower()
                    spec = str(pkg.get("SPECIALITY", "")).lower()
                    if any(tok in f"{name} {spec}" for tok in term_tokens):
                        term_pkgs.append(pkg)
                term_pkgs = term_pkgs[:200]
        per_term_packages[term] = term_pkgs

    # ----- AI Intent Classification Check (with Package Specialties) -----
    start_violation = None
    if len(query_terms) > 1:
        term_specialties = {}
        for t in query_terms:
            pkgs = per_term_packages.get(t, [])
            if t == main_term and not pkgs:
                pkgs = matching_packages
            specs = set()
            for p in pkgs[:15]:
                spec = str(p.get("SPECIALITY", "")).strip()
                if spec:
                    specs.add(spec)
            term_specialties[t] = list(specs)[:3]
            
        input_intents = _classify_input_intent(term_specialties)
        start_violation = _check_intent_rule_violation(input_intents)
        if start_violation:
            logger.warning(f"Initial intent violation detected: {start_violation}")
    # ---------------------------------------------------------------------

    # Build the flow with per-term package lists
    flow = build_search_flow(
        main_term,
        addon_terms,
        matching_packages,
        all_packages_for_addons=all_packages,
        per_term_packages=per_term_packages,
    )
    advance_past_empty_optional_steps(flow)
    _auto_advance_single_option_steps(flow, all_packages)

    # Store flow session
    session_id = str(uuid.uuid4())
    _interactive_flows[session_id] = {
        "flow": flow,
        "packages": matching_packages,
        "all_packages": all_packages,
        "per_term_packages": per_term_packages,
        "created_at": time.time(),
        "selections_list": [],
        "request": {
            "query": request.query,
            "procedure": request.procedure,
            "disease": request.disease,
            "symptoms": request.symptoms,
            "patient_age": request.patient_age,
            "patient_gender": request.patient_gender,
        }
    }
    
    # Persist to DB for Vercel statelessness
    _update_session_database(session_id, flow, [])

    # Get first step
    first_step = flow.steps[flow.current_step] if flow.steps and flow.current_step < len(
        flow.steps) else None
    if not first_step:
        raise HTTPException(500, "Failed to build search flow")

    first_step_response = SearchStepResponse(
        step_number=first_step.step_number,
        step_name=_normalize_interactive_step_title(first_step.step_name),
        description=first_step.description,
        options=[
            SearchOption(
                id=opt.get("id", ""),
                label=opt.get("label", ""),
                description=opt.get("description", ""),
                specialty=opt.get("specialty"),
                code=opt.get("code"),
                rate=opt.get("rate"),
                reasoning=opt.get("reasoning"),
                rank=opt.get("rank"),
            )
            for opt in first_step.options
        ],
        requires_user_selection=first_step.requires_user_selection,
        context=first_step.context,
    )

    return InteractiveSearchStartResponse(
        session_id=session_id,
        query=request.query,
        parsed_terms=query_terms,
        current_step=first_step_response,
        message=(
            (f"{start_violation} | " if start_violation else "") +
            (f"Did you mean: {', '.join(spelling_corrections.values())}? " if spelling_corrections else "") +
            f"Starting interactive search for: {main_term}. Found {len(matching_packages)} related packages."
        ),
    )


@app.get("/interactive-search/{session_id}/step")
async def get_current_step(session_id: str):
    """Get the current step for an ongoing flow."""
    if session_id not in _interactive_flows:
        raise HTTPException(404, "Session not found")

    flow, packages, per_term_packages = await _get_or_reconstruct_flow(session_id)
    
    from tools.smart_search_flow import advance_past_empty_optional_steps
    advance_past_empty_optional_steps(flow)
    _auto_advance_single_option_steps(flow, packages)

    if flow.flow_complete:
        return {
            "status": "complete",
            "final_recommendation": flow.final_recommendation,
            "selections": flow.selections,
        }

    current_step = flow.steps[flow.current_step] if flow.current_step < len(
        flow.steps) else None
    if not current_step:
        raise HTTPException(500, "Invalid flow state")

    return {
        "step_number": current_step.step_number,
        "step_name": _normalize_interactive_step_title(current_step.step_name),
        "description": current_step.description,
        "options": [
            {
                "id": opt.get("id", ""),
                "label": opt.get("label", ""),
                "description": opt.get("description", ""),
                "specialty": opt.get("specialty"),
                "code": opt.get("code"),
                "rate": opt.get("rate"),
                "reasoning": opt.get("reasoning") or opt.get("reason"),
                "rank": opt.get("rank"),
            }
            for opt in current_step.options
        ],
        "requires_user_selection": current_step.requires_user_selection,
        "context": current_step.context,
    }


@app.post("/interactive-search/{session_id}/select")
async def submit_step_selection(session_id: str, selection: SelectionRequest):
    """
    Submit user's selection for the current step and move to next step.

    Returns:
    - success: Whether selection was valid
    - next_step: The next step to present (or None if flow complete)
    - flow_complete: Whether the flow finished
    - final_recommendation: Final packages if flow is complete
    """
    if session_id not in _interactive_flows:
        raise HTTPException(404, "Session not found")

    flow, packages, per_term_packages = await _get_or_reconstruct_flow(session_id)

    # Process the selection
    from tools.smart_search_flow import process_step_selection

    success, error = process_step_selection(
        flow,
        {
            "id": selection.option_id,
            "notes": selection.notes,
            "manual_package": selection.manual_package,
        },
        packages,
    )

    if not success:
        return SelectionResponse(
            success=False,
            message=f"Error: {error}",
            next_step=None,
            flow_complete=False,
        )

    # Success: Update our selection history and Sync to DB
    selections_list = _interactive_flows[session_id].get("selections_list", [])
    selections_list.append({
        "id": selection.option_id,
        "notes": selection.notes,
        "manual_package": selection.manual_package,
    })
    _interactive_flows[session_id]["selections_list"] = selections_list
    _update_session_database(session_id, flow, selections_list)

    # Auto-progress any subsequent single-option steps.
    _auto_advance_single_option_steps(flow, packages)

    # If flow is complete, build final recommendation
    if flow.flow_complete:
        # Compile final recommendation from selections
        final_recommendation = await _build_final_recommendation(flow, packages)
        flow.final_recommendation = final_recommendation

        return SelectionResponse(
            success=True,
            message="Search flow completed! Here are your recommendations.",
            next_step=None,
            flow_complete=True,
            final_recommendation=final_recommendation,
        )

    # Get next step
    next_step = flow.steps[flow.current_step] if flow.current_step < len(
        flow.steps) else None
    if not next_step:
        return SelectionResponse(
            success=True,
            message="Flow completed.",
            next_step=None,
            flow_complete=True,
        )

    next_step_response = SearchStepResponse(
        step_number=next_step.step_number,
        step_name=_normalize_interactive_step_title(next_step.step_name),
        description=next_step.description,
        options=[
            SearchOption(
                id=opt.get("id", ""),
                label=opt.get("label", ""),
                description=opt.get("description", ""),
                specialty=opt.get("specialty"),
                code=opt.get("code"),
                rate=opt.get("rate"),
                reasoning=opt.get("reasoning") or opt.get("reason"),
                rank=opt.get("rank"),
            )
            for opt in next_step.options
        ],
        requires_user_selection=next_step.requires_user_selection,
        context=next_step.context,
    )

    return SelectionResponse(
        success=True,
        message="Selection received. Moving to next step.",
        next_step=next_step_response,
        flow_complete=False,
    )


@app.post("/interactive-search/{session_id}/undo")
async def undo_step_selection(session_id: str):
    """
    Undo the last selection made in the flow.
    Returns the step to which the flow has reverted.
    """
    if session_id not in _interactive_flows:
        raise HTTPException(404, "Session not found")

    flow, packages, per_term_packages = await _get_or_reconstruct_flow(session_id)

    from tools.smart_search_flow import undo_last_selection
    success, message = undo_last_selection(flow)

    if success:
        # Update history
        selections_list = _interactive_flows[session_id].get("selections_list", [])
        if selections_list:
            selections_list.pop()
        _interactive_flows[session_id]["selections_list"] = selections_list
        _update_session_database(session_id, flow, selections_list)

    if not success:
        return JSONResponse(
            status_code=400,
            content={"success": False, "message": message}
        )

    # Get the "new" current step after undo
    current_step = flow.steps[flow.current_step] if flow.current_step < len(flow.steps) else None
    
    if not current_step:
        return JSONResponse(
            status_code=500,
            content={"success": False, "message": "Reverted, but no current step found."}
        )

    step_response = SearchStepResponse(
        step_number=current_step.step_number,
        step_name=_normalize_interactive_step_title(current_step.step_name),
        description=current_step.description,
        options=[
            SearchOption(
                id=opt.get("id", ""),
                label=opt.get("label", ""),
                description=opt.get("description", ""),
                specialty=opt.get("specialty"),
                code=opt.get("code"),
                rate=opt.get("rate"),
                reasoning=opt.get("reasoning") or opt.get("reason"),
                rank=opt.get("rank"),
            )
            for opt in current_step.options
        ],
        requires_user_selection=current_step.requires_user_selection,
        context=current_step.context,
    )

    return {
        "success": True,
        "message": message,
        "current_step": step_response
    }


@app.get("/interactive-search/{session_id}/status")
async def get_flow_status(session_id: str):
    """Get the current status of a flow session."""
    if session_id not in _interactive_flows:
        raise HTTPException(404, "Session not found")

    session_data = _interactive_flows[session_id]
    flow = session_data["flow"]

    return FlowStatusResponse(
        session_id=session_id,
        query=flow.query,
        current_step_number=flow.current_step,
        total_steps=len(flow.steps),
        selections_made=flow.selections,
        violations=flow.violations,
        flow_complete=flow.flow_complete,
    )


async def _build_final_recommendation(flow: Any, packages: List[Dict]) -> Dict:
    """Build final package recommendation from flow selections.

    Produces both:
    - Flat lists (selected_packages, implant_packages, etc.) for backward compat
    - term_groups: ordered list of per-term objects for grouped rendering
    """
    from tools.smart_search_flow import validate_package_combination

    result: Dict[str, Any] = {
        "main_package": None,
        "selected_packages": [],
        "implant_package": None,
        "implant_packages": [],
        "stratification_package": None,
        "stratification_packages": [],
        "addon_packages": [],
        "term_groups": [],           # NEW: per-term grouped packages
        "blocked_rules": [],
        "approval_likelihood": "MEDIUM",
        "doctor_reasoning": "Packages selected through interactive flow.",
    }

    package_by_code = {pkg.get("PACKAGE CODE", ""): pkg for pkg in packages}

    def _make_entry(pkg: Dict, max_name: int = 100) -> Dict:
        name = pkg.get("PACKAGE NAME", pkg.get("Package Name", ""))
        rate = pkg.get("RATE", pkg.get("Rate", 0))
        spec = pkg.get("SPECIALITY", pkg.get("Speciality", pkg.get("SPECIALITY NAME", "")))
        cat = pkg.get("PACKAGE CATEGORY", pkg.get("PACKAGE TYPE", pkg.get("Procedure Type", "")))
        return {
            "code": pkg.get("PACKAGE CODE", ""),
            "name": name[:max_name],
            "rate": rate,
            "specialty": spec,
            "package_category": cat,
            "pre_auth_document": pkg.get("PRE AUTH DOCUMENT", pkg.get("Mandatory Documents", "")),
            "claim_document": pkg.get("CLAIM DOCUMENT", pkg.get("Mandatory Documents - Claim Processing", "")),
        }

    def _safe_rate(entry: Dict) -> float:
        raw = entry.get("rate", 0)
        if isinstance(raw, (int, float)):
            return float(raw)
        try:
            return float(str(raw).replace(",", "").strip())
        except Exception:
            return 0.0

    # ── Walk steps in order to build term_groups ─────────────────────────
    # Each step stores context.intent_term; we group by that.
    # term_groups_map: term -> {main_package, implants, strats, subtotal}
    from collections import OrderedDict
    term_groups_map: OrderedDict = OrderedDict()

    current_term = ""
    for step_idx, step in enumerate(flow.steps):
        sel_key = f"step_{step_idx}"
        selection = flow.selections.get(sel_key)
        if not isinstance(selection, dict):
            continue

        sel_id = str(selection.get("id", ""))
        sel_code = str(selection.get("code", "")).strip()

        # Determine which term this step belongs to
        step_term = (step.context.get("intent_term", "") or "").strip()
        if step_term:
            current_term = step_term

        # Skip non-package selections (skip, manual without code, clarification)
        if not sel_code or "skip" in sel_id.lower():
            continue

        pkg = package_by_code.get(sel_code)
        if not pkg:
            continue

        entry = _make_entry(pkg)

        # Ensure term group exists
        term_key = current_term or "main"
        if term_key not in term_groups_map:
            term_groups_map[term_key] = {
                "term": term_key,
                "main_package": None,
                "implant_packages": [],
                "stratification_packages": [],
                "subtotal": 0.0,
            }
        group = term_groups_map[term_key]

        if sel_id.startswith("package_"):
            group["main_package"] = entry
            group["subtotal"] += _safe_rate(entry)
            # Flat compat
            result["selected_packages"].append(entry)
            if result["main_package"] is None:
                result["main_package"] = entry

        elif sel_id.startswith("implant_"):
            if sel_code != "NO_IMPLANT":
                group["implant_packages"].append(entry)
                group["subtotal"] += _safe_rate(entry)
                # Flat compat
                result["implant_packages"].append(entry)
                if result["implant_package"] is None:
                    result["implant_package"] = entry

        elif sel_id.startswith("strat_"):
            group["stratification_packages"].append(entry)
            group["subtotal"] += _safe_rate(entry)
            # Flat compat
            result["stratification_packages"].append(entry)
            if result["stratification_package"] is None:
                result["stratification_package"] = entry

        elif sel_id.startswith("addon_"):
            # Add-ons go to flat list only (not per-term)
            addon_entry = {**entry, "reason": selection.get("reason", "")}
            result["addon_packages"].append(addon_entry)

    # Convert term_groups_map to ordered list
    result["term_groups"] = list(term_groups_map.values())

    # Validate combinations
    main_package_result = result.get("main_package")
    if isinstance(main_package_result, dict):
        main_pkg = package_by_code.get(main_package_result.get("code", ""))
        if not main_pkg:
            return result

        addon_pkgs = []
        # Include other selected packages as combination targets
        for sp in result["selected_packages"][1:]:
            p = package_by_code.get(sp.get("code", ""))
            if p:
                addon_pkgs.append(p)

        stratification_result = result.get("stratification_package")
        strat_pkg = None
        if isinstance(stratification_result, dict):
            strat_pkg = package_by_code.get(
                stratification_result.get("code", ""))

        implant_result = result.get("implant_package")
        implant_pkg = None
        if isinstance(implant_result, dict):
            implant_pkg = package_by_code.get(
                implant_result.get("code", ""))

        for addon in result["addon_packages"]:
            pkg = package_by_code.get(addon["code"])
            if pkg:
                addon_pkgs.append(pkg)

        is_valid, violations = validate_package_combination(
            main_pkg, implant_pkg, strat_pkg, addon_pkgs)
        result["blocked_rules"] = violations
        if not is_valid:
            result["approval_likelihood"] = "LOW"

    return result


# ── Dev entry point ────────────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=SERVER_HOST,
        port=SERVER_PORT,
        workers=max(1, SERVER_WORKERS),
        reload=False,
    )
