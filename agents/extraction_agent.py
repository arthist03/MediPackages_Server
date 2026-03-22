"""
agents/extraction_agent.py  —  Node 2: Single-pass Groq Vision extraction.

STRATEGY: One Groq Vision API call → Structured JSON
Fast (2-5 seconds) and accurate for handwritten prescriptions.
"""
from __future__ import annotations

import json
import logging

from graph.state import OcrPipelineState
from memory.sqlite_store import AgentMemory
from tools.llm_tool import extract_medical_data

logger = logging.getLogger("extraction_agent")
_memory = AgentMemory()


def extraction_agent(state: OcrPipelineState) -> dict:
    """
    LangGraph node: Extract medical data from prescription image using Groq Vision.
    Single-pass extraction - no EasyOCR needed.
    """
    image_b64 = state.get("image_b64", "")
    mime_type = state.get("mime_type", "image/jpeg")

    if not image_b64:
        return {
            "error": "No image data provided",
            "extracted": {},
            "supervisor_notes": ["Extraction: no image data"],
        }

    logger.info(
        "🔍 Extraction Agent: Analyzing prescription with Groq Vision...")

    try:
        # Single Groq Vision call for extraction
        extracted = extract_medical_data(image_b64, mime_type)

        # Check for parse errors
        if extracted.get("parse_error"):
            logger.warning("Groq returned unparseable response")
            return {
                "extracted": {},
                "error": "Failed to parse extraction response",
                "supervisor_notes": ["Extraction: JSON parse error"],
            }

        # Count non-empty fields for quality check
        fields_found = sum(1 for v in extracted.values()
                           if v and v != [] and v != {})

        logger.info(f"✅ Extraction complete: {fields_found} fields extracted")

        return {
            "extracted": extracted,
            "extraction_raw_response": json.dumps(extracted, ensure_ascii=False),
            "supervisor_notes": [f"Groq extraction: {fields_found} fields found"],
        }

    except Exception as e:
        logger.exception("Extraction failed")
        return {
            "error": f"Extraction failed: {e}",
            "extracted": {},
            "supervisor_notes": [f"Extraction error: {e}"],
        }
