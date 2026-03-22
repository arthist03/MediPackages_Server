"""
tools/llm_tool.py  —  Groq Vision API for medical document OCR extraction.

Uses Groq's Llama 3.2 Vision for fast, accurate extraction from prescription images.
Single-pass extraction: Image → Structured JSON (no EasyOCR needed).
"""
from __future__ import annotations

import json
import logging
import time
from typing import Any

from config.settings import (
    GROQ_API_KEY,
    GROQ_MODEL,
    LLM_MAX_RETRIES,
    LLM_TEMPERATURE,
    LLM_MAX_TOKENS,
)

logger = logging.getLogger("llm_tool")

# ── Configure Groq ───────────────────────────────────────────────────
_groq_client = None


def _get_groq_client():
    """Lazy-load Groq client."""
    global _groq_client
    if _groq_client is not None:
        return _groq_client

    if not GROQ_API_KEY:
        raise ValueError(
            "GROQ_API_KEY not set in environment. Get one at https://console.groq.com/keys")

    from groq import Groq
    _groq_client = Groq(api_key=GROQ_API_KEY)
    logger.info(f"Groq client initialized: {GROQ_MODEL}")
    return _groq_client


# ── Optimized Medical Extraction Prompt ───────────────────────────────
EXTRACTION_PROMPT = """\
You are an expert Indian medical AI analyzing a prescription/OPD slip image.

TASK: Extract ALL visible information into the JSON schema below.

CRITICAL RULES:
1. READ the image carefully - include ALL handwritten and printed text
2. EXPAND medical shorthand:
   - c/o, C/O = chief complaints (symptoms patient is complaining about)
   - k/c/o, K/C/O = known case of (existing diagnosis)
   - h/o = history of
   - Rx = prescription/treatment
   - DM/T2DM = Diabetes Mellitus, HTN = Hypertension
   - CAD = Coronary Artery Disease, CKD = Chronic Kidney Disease
   - OD = once daily, BD = twice daily, TDS = three times daily
   - Tab = Tablet, Cap = Capsule, Inj = Injection, Syr = Syrup
   - AC = before food, PC = after food, HS = at bedtime
3. Use "" for missing text fields, null for missing numbers, [] for empty lists
4. DO NOT hallucinate - only include what you can clearly see

JSON SCHEMA:
{
  "patient_name": "full name as written",
  "patient_age": integer or null,
  "patient_gender": "Male/Female/Other or empty string",
  "date": "DD-MM-YYYY format",
  "doctor_name": "with title Dr./Prof.",
  "qualifications": "MBBS, MD, etc.",
  "clinic_name": "hospital or clinic name",
  "clinic_address": "full address if visible",
  "department": "Cardiology/Orthopedics/General Medicine/etc.",
  "chief_complaints": ["expanded list of c/o entries"],
  "history": "past medical history (k/c/o entries)",
  "diagnosis": "PRIMARY diagnosis fully expanded",
  "secondary_diagnoses": ["other conditions/comorbidities"],
  "surgery_required": true or false,
  "surgery_name": "name if surgery mentioned, else empty",
  "procedure_required": true or false,
  "procedure_name": "name if procedure mentioned, else empty",
  "lab_tests_ordered": true or false,
  "lab_tests": ["list of tests ordered"],
  "medications": [
    {"name": "drug name", "dose": "strength", "frequency": "expanded (e.g., twice daily)", "duration": "days/weeks"}
  ],
  "vitals": {"bp": "", "pulse": "", "spo2": "", "weight": "", "rbs": ""},
  "follow_up_date": "next review date",
  "confidence_notes": "any unclear or ambiguous readings"
}

Analyze the image and return ONLY valid JSON:"""


def extract_medical_data(image_b64: str, mime_type: str = "image/jpeg") -> dict[str, Any]:
    """
    Extract structured medical data from a prescription image using Groq Vision.

    Args:
        image_b64: Base64 encoded image
        mime_type: MIME type of the image

    Returns:
        dict with extracted medical data
    """
    client = _get_groq_client()
    last_error = None

    for attempt in range(1, LLM_MAX_RETRIES + 2):
        try:
            start_time = time.time()

            # Groq expects base64 data URL format
            data_url = f"data:{mime_type};base64,{image_b64}"

            response = client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": EXTRACTION_PROMPT},
                            {"type": "image_url", "image_url": {"url": data_url}},
                        ],
                    }
                ],
                temperature=LLM_TEMPERATURE,
                max_completion_tokens=LLM_MAX_TOKENS,
                response_format={"type": "json_object"},
            )

            elapsed = time.time() - start_time
            text = response.choices[0].message.content

            if not text:
                raise ValueError("Groq returned empty response")

            result = _parse_json(text)
            logger.info(f"Groq extraction complete in {elapsed:.2f}s ({len(result)} fields)")
            return result

        except Exception as e:
            last_error = e
            wait = min(2 ** attempt, 8)
            logger.warning(f"Groq attempt {attempt} failed: {e}. Retrying in {wait}s...")
            if attempt <= LLM_MAX_RETRIES:
                time.sleep(wait)

    raise RuntimeError(f"Groq extraction failed after {LLM_MAX_RETRIES + 1} attempts: {last_error}")


def call_llm_json(prompt: str, system: str = "") -> dict[str, Any]:
    """
    Call Groq with a text-only prompt for JSON response.
    Used for validation and package matching refinement.
    """
    client = _get_groq_client()

    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    try:
        response = client.chat.completions.create(
            model=GROQ_MODEL,  # Llama 4 Scout handles text too
            messages=messages,
            temperature=LLM_TEMPERATURE,
            max_completion_tokens=LLM_MAX_TOKENS,
            response_format={"type": "json_object"},
        )

        text = response.choices[0].message.content
        if not text:
            return {"error": "Empty response"}

        return _parse_json(text)

    except Exception as e:
        logger.error(f"Groq text call failed: {e}")
        return {"error": str(e)}


def _parse_json(raw: str) -> dict[str, Any]:
    """Parse JSON from LLM response, handling markdown fences."""
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        cleaned = raw.strip()

        # Remove markdown code fences
        if cleaned.startswith("```"):
            lines = cleaned.split("\n")
            cleaned = "\n".join(
                lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
            cleaned = cleaned.strip()

        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass

        # Find JSON object in response
        start = cleaned.find("{")
        end = cleaned.rfind("}") + 1
        if start != -1 and end > start:
            try:
                return json.loads(cleaned[start:end])
            except json.JSONDecodeError:
                pass

        logger.error(f"Failed to parse JSON: {raw[:500]}")
        return {"_raw": raw, "parse_error": True}


def check_llm_connection() -> bool:
    """Check if Groq API is accessible."""
    try:
        client = _get_groq_client()
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": "Say OK"}],
            max_completion_tokens=10,
        )
        return bool(response.choices[0].message.content)
    except Exception as e:
        logger.error(f"Groq connection check failed: {e}")
        return False


# ── Legacy compatibility ──────────────────────────────────────────────
def call_llm_vision(image_b64: str, prompt: str, **kwargs) -> dict:
    """Legacy wrapper - redirects to extract_medical_data."""
    return extract_medical_data(image_b64, kwargs.get("mime_type", "image/jpeg"))


def call_llm(prompt: str, **kwargs) -> str:
    """Legacy wrapper - redirects to call_llm_json."""
    result = call_llm_json(prompt, kwargs.get("system", ""))
    if "_raw" in result:
        return result["_raw"]
    return json.dumps(result)


_VISION_SYSTEM = EXTRACTION_PROMPT  # Legacy export
