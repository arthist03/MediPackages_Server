"""
graph/pipeline.py  —  LangGraph StateGraph: the full OCR agentic pipeline.

Graph flow:
  START
    → ocr_node        (EasyOCR text extraction)
    → extraction_node (Two-pass: Vision read + LLM structure)
    → validation_node (Pydantic schema + confidence)
    → human_review    (LangGraph interrupt — ALWAYS pauses for user verification)
    → package_node    (MAA YOJANA package matching — only after user approval)
    → supervisor_node (apply corrections, assemble response, store memory)
    → END

If any node sets state["error"], routing jumps to error_node → END.
Human review ALWAYS runs — user must approve the AI Doctor summary.
"""
from __future__ import annotations

import logging
from typing import Literal

from langgraph.graph import StateGraph, END, START

from config.settings import MEMORY_DB
from graph.state import OcrPipelineState
from agents.ocr_agent import ocr_agent
from agents.extraction_agent import extraction_agent
from agents.validation_agent import validation_agent
from agents.package_matching_agent import package_matching_agent
from agents.supervisor_agent import supervisor_agent, error_handler_node

logger = logging.getLogger("pipeline")


# ── Routing functions ──────────────────────────────────────────────────

def route_after_ocr(state: OcrPipelineState) -> Literal["extraction", "error"]:
    return "error" if state.get("error") else "extraction"


def route_after_extraction(state: OcrPipelineState) -> Literal["validation", "error"]:
    return "error" if state.get("error") else "validation"


def route_after_validation(
    state: OcrPipelineState,
) -> Literal["human_review", "error"]:
    if state.get("error"):
        return "error"
    # ALWAYS send to human review — user must verify AI Doctor summary
    return "human_review"


def route_after_human_review(
    state: OcrPipelineState,
) -> Literal["package_matching", "extraction", "error"]:
    decision = state.get("human_decision", "approved")
    if decision == "rejected":
        # Re-run extraction with corrections applied
        retry = state.get("retry_count", 0)
        if retry < 3:
            logger.info(
                f"Human rejected (retry {retry + 1}/3) → re-extraction")
            return "extraction"
        logger.warning("Max retries reached after human rejection → error")
        return "error"
    # User approved → proceed to package matching
    return "package_matching"


def route_after_packages(state: OcrPipelineState) -> Literal["supervisor", "error"]:
    return "error" if state.get("error") else "supervisor"


# ── Graph builder ──────────────────────────────────────────────────────

def build_graph() -> StateGraph:
    """Build and compile the LangGraph pipeline."""
    graph = StateGraph(OcrPipelineState)

    # Register all nodes
    graph.add_node("ocr", ocr_agent)
    graph.add_node("extraction", extraction_agent)
    graph.add_node("validation", validation_agent)
    graph.add_node("human_review", _human_review_node)  # interrupt node
    graph.add_node("package_matching", package_matching_agent)
    graph.add_node("supervisor", supervisor_agent)
    graph.add_node("error", error_handler_node)

    # Edges
    graph.add_edge(START, "ocr")
    graph.add_conditional_edges("ocr", route_after_ocr, {
                                "extraction": "extraction", "error": "error"})
    graph.add_conditional_edges("extraction", route_after_extraction, {
                                "validation": "validation", "error": "error"})
    graph.add_conditional_edges(
        "validation",
        route_after_validation,
        {"human_review": "human_review", "error": "error"},
    )
    graph.add_conditional_edges(
        "human_review",
        route_after_human_review,
        {"extraction": "extraction",
            "package_matching": "package_matching", "error": "error"},
    )
    graph.add_conditional_edges(
        "package_matching",
        route_after_packages,
        {"supervisor": "supervisor", "error": "error"},
    )
    graph.add_edge("supervisor", END)
    graph.add_edge("error", END)

    return graph


def _human_review_node(state: OcrPipelineState) -> dict:
    """
    Human-in-the-Loop interrupt node.
    LangGraph pauses here via interrupt_before.
    """
    logger.info("⏸ Sending AI Doctor summary to user for verification...")
    # No need to call dynamic interrupt() if we use interrupt_before
    return {}


def get_compiled_graph():
    """Return a compiled graph with in-memory checkpointing."""
    from langgraph.checkpoint.memory import MemorySaver

    checkpointer = MemorySaver()
    graph = build_graph()
    return graph.compile(checkpointer=checkpointer, interrupt_before=["human_review"])
