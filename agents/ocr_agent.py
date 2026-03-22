"""
agents/ocr_agent.py  —  Node 1: Image validation and preprocessing.

With Groq Vision, we don't need EasyOCR for text extraction.
This node now just validates the image and prepares it for the extraction agent.
"""
from __future__ import annotations

import logging
import uuid

from graph.state import OcrPipelineState
from memory.sqlite_store import AgentMemory

logger = logging.getLogger("ocr_agent")
_memory = AgentMemory()


def ocr_agent(state: OcrPipelineState) -> dict:
    """
    LangGraph node: Validate image and prepare for Groq extraction.
    No longer uses EasyOCR - Groq Vision handles everything.
    """
    logger.info("🔍 OCR Agent: Validating image...")

    image_b64 = state.get("image_b64", "")
    session_id = state.get("session_id", str(uuid.uuid4()))

    if not image_b64:
        return {"error": "No image data provided to OCR agent"}

    # Create a DB record for this request
    try:
        request_id = _memory.create_request(session_id)
    except Exception:
        request_id = 0

    # Basic validation - check image is not too small (likely empty/invalid)
    try:
        import base64
        raw_bytes = base64.b64decode(image_b64)
        if len(raw_bytes) < 1000:  # Less than 1KB is likely invalid
            return {
                "error": "Image too small - may be blank or corrupted",
                "request_id": request_id,
                "session_id": session_id,
                "supervisor_notes": ["OCR: Image validation failed (too small)"],
            }

        logger.info(f"✅ Image validated: {len(raw_bytes) / 1024:.1f} KB")
        return {
            "raw_text": "",  # No EasyOCR - Groq will extract directly
            "request_id": request_id,
            "session_id": session_id,
            "supervisor_notes": [f"Image validated: {len(raw_bytes) / 1024:.1f} KB"],
        }

    except Exception as e:
        logger.exception("Image validation error")
        return {
            "error": f"Image validation failed: {e}",
            "request_id": request_id,
            "session_id": session_id,
            "supervisor_notes": [f"OCR error: {e}"],
        }
