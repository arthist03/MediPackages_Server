"""
agents/supervisor_agent.py  —  Node 5: Supervisor / orchestrator.

Inspects current state, applies learning from past rejections,
and decides final routing. Also assembles the final response.
"""
from __future__ import annotations

import logging
from datetime import datetime

from graph.state import OcrPipelineState
from memory.sqlite_store import AgentMemory
from tools.validation_tool import validate_extraction, extraction_to_response_dict

logger = logging.getLogger("supervisor_agent")
_memory = AgentMemory()


def supervisor_agent(state: OcrPipelineState) -> dict:
    """
    LangGraph node: final supervisor step.
    - Applies human corrections if present
    - Assembles final response dict
    - Updates SQLite memory with result
    """
    logger.info("🎯 Supervisor Agent: finalising pipeline result")

    extracted = state.get("extracted", {})
    human_correction = state.get("human_correction", {})
    raw_text = state.get("raw_text", "")
    best_packages = state.get("best_packages", [])
    request_id = state.get("request_id", 0)
    human_decision = state.get("human_decision", "approved")
    human_reason = state.get("human_reason", "")

    # Apply any field-level corrections from human
    if human_correction and isinstance(human_correction, dict):
        for field, value in human_correction.items():
            if value is not None:
                extracted[field] = value
        logger.info(f"Applied {len(human_correction)} human corrections")

    # Store feedback in long-term memory
    if request_id and human_decision:
        try:
            _memory.store_feedback(
                request_id=request_id,
                decision=human_decision,
                reason=human_reason,
            )
        except Exception as e:
            logger.error(f"Memory store error: {e}")

    # Get top rejection patterns (for logging/monitoring)
    try:
        top_rejections = _memory.get_top_rejection_patterns(limit=3)
        if top_rejections:
            logger.info(f"Known rejection patterns: {top_rejections}")
    except Exception:
        pass

    # Final Pydantic validation + response construction
    try:
        medical_model, val_result = validate_extraction(extracted, raw_text)
        final_response = extraction_to_response_dict(medical_model, val_result, best_packages)
    except Exception as e:
        logger.exception("Response assembly error")
        final_response = {
            "success": False,
            "error": str(e),
            "data": {},
        }

    # Persist final result in memory
    if request_id:
        try:
            _memory.update_request(request_id, extracted, status="completed")
        except Exception:
            pass

    notes = state.get("supervisor_notes", [])
    notes.append("Supervisor: pipeline complete")
    logger.info("✅ Supervisor: final response assembled")

    return {
        "final_response": final_response,
        "supervisor_notes": ["Supervisor: completed"],
    }


def error_handler_node(state: OcrPipelineState) -> dict:
    """Fallback node: called when any agent sets state['error']."""
    error = state.get("error", "Unknown pipeline error")
    logger.error(f"Pipeline error: {error}")

    request_id = state.get("request_id", 0)
    if request_id:
        try:
            _memory.update_request(request_id, {}, status="error")
        except Exception:
            pass

    return {
        "final_response": {
            "success": False,
            "error": error,
            "data": {},
        },
        "supervisor_notes": [f"Error handler: {error}"],
    }
