"""
agents/validation_agent.py  —  Node 3: Pydantic validation + confidence scoring.

Validates the extraction, computes confidence, and decides whether to route to
human review or directly to package matching.
"""
from __future__ import annotations

import logging

from graph.state import OcrPipelineState
from tools.validation_tool import validate_extraction

logger = logging.getLogger("validation_agent")

# Minimum confidence to skip human review
AUTO_APPROVE_THRESHOLD = 0.75


def validation_agent(state: OcrPipelineState) -> dict:
    """LangGraph node: validate extracted data and compute confidence score."""
    logger.info("✅ Validation Agent: checking extraction quality")

    extracted = state.get("extracted", {})
    raw_text = state.get("raw_text", "")

    if not extracted:
        return {
            "validation_passed": False,
            "validation": {"confidence": 0.0, "issues": ["No extracted data"], "drugs_verified": 0, "drugs_total": 0},
            "supervisor_notes": ["Validation: no data to validate"],
        }

    try:
        medical_model, val_result = validate_extraction(extracted, raw_text)
        val_dict = val_result.model_dump()
        confidence = val_result.confidence
        passed = confidence >= AUTO_APPROVE_THRESHOLD and len(val_result.issues) == 0

        logger.info(
            f"Validation: confidence={confidence:.2f}, issues={val_result.issues}, passed={passed}"
        )

        return {
            "validation": val_dict,
            "validation_passed": passed,
            "supervisor_notes": [f"Validation: confidence={confidence:.2f}, issues={val_result.issues}"],
        }

    except Exception as e:
        logger.exception("Validation agent error")
        return {
            "validation_passed": False,
            "validation": {"confidence": 0.0, "issues": [str(e)], "drugs_verified": 0, "drugs_total": 0},
            "supervisor_notes": [f"Validation error: {e}"],
        }
