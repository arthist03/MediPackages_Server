"""
graph/state.py  —  LangGraph shared state for the OCR pipeline.

TypedDict is required by LangGraph for the state schema.
All agents read/write to this shared mutable state dict.
"""
from __future__ import annotations

from typing import Any, Optional
from typing_extensions import TypedDict, Annotated
import operator


class OcrPipelineState(TypedDict, total=False):
    # ── Input ─────────────────────────────────────────────────────────
    session_id: str
    image_b64: str           # base64 image from Flutter
    mime_type: str           # e.g. "image/jpeg"

    # ── OCR Agent output ──────────────────────────────────────────────
    raw_text: str            # raw text extracted by EasyOCR
    vision_text: str         # raw text from vision model (pass 1)

    # ── Extraction Agent output ───────────────────────────────────────
    extracted: dict[str, Any]      # LLM-parsed structured fields
    extraction_raw_response: str   # raw LLM response (for debugging)

    # ── AI Doctor Summary fields ──────────────────────────────────────
    surgery_required: bool
    surgery_name: str
    procedure_required: bool
    procedure_name: str
    lab_tests_ordered: bool
    department: str
    history: str

    # ── Validation Agent output ───────────────────────────────────────
    validation: dict[str, Any]     # ValidationResult as dict
    validation_passed: bool

    # ── Human review ─────────────────────────────────────────────────
    human_decision: str            # "approved" | "rejected"
    human_reason: str              # rejection reason
    human_correction: dict         # field-level corrections

    # ── Package matching output ───────────────────────────────────────
    best_packages: list[dict]

    # ── Final output ──────────────────────────────────────────────────
    final_response: dict[str, Any] # response body sent to Flutter

    # ── Pipeline control ─────────────────────────────────────────────
    error: str                     # set if any step fails
    retry_count: int               # number of retries attempted
    request_id: int                # SQLite request ID for memory tracking

    # ── Supervisor metadata ───────────────────────────────────────────
    supervisor_notes: Annotated[list[str], operator.add]  # append-only log
