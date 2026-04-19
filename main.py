"""
main.py  —  FastAPI production server for the LangGraph OCR pipeline.
Optimized for Vercel Serverless Functions.

Endpoints:
  POST /extract                              — Medical document OCR extraction
  POST /feedback                             — Human-in-the-loop verification (approve/reject)
  POST /retry                                — Re-run extraction with rejection context
  GET  /health                               — Health check (LLM + system status)
  GET  /stats                                — Pipeline statistics from long-term memory
  POST /send-push                            — FCM push notification proxy
  POST /smart-search                         — AI-powered MAA Yojana package search
  POST /interactive-search/analyze-query     — NLP query analysis
  POST /interactive-search/start             — Start multi-step package selection
  POST /interactive-search/{id}/select       — Submit step selection
  POST /interactive-search/{id}/undo         — Undo last selection
  GET  /interactive-search/{id}/step         — Current step state
  GET  /interactive-search/{id}/status       — Flow session status

Usage:
  Vercel: auto-deployed via vercel.json
  Local:  uvicorn main:app --host 0.0.0.0 --port 8000
"""
from __future__ import annotations

import base64
import difflib
import gc
import io
import json
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import re
import secrets
import time
import uuid

from fastapi import FastAPI, File, HTTPException, UploadFile, Depends, Security, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.security.api_key import APIKeyHeader
from groq import Groq, AsyncGroq
import functools
from PIL import Image
from pydantic import BaseModel
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.middleware.gzip import GZipMiddleware
from starlette.middleware.trustedhost import TrustedHostMiddleware
import uvicorn

from config.settings import (
    MAX_UPLOAD_MB, SERVER_HOST, SERVER_PORT, SERVER_WORKERS, LOG_LEVEL,
    GROQ_API_KEY, GROQ_MODEL, API_AUTH_TOKEN, CORS_ORIGINS, CORS_ALLOW_CREDENTIALS,
    TRUSTED_HOSTS, ENABLE_DOCS, APP_ENV,
)
from tools.medical_knowledge import (
    get_specialties_for_term, get_clinical_pathway, get_packages_for_symptom,
    is_medical_management_term, is_surgical_term,
)

# ═══════════════════════════════════════════════════════════════════════
# LOGGING
# ═══════════════════════════════════════════════════════════════════════
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s │ %(name)s │ %(levelname)s │ %(message)s",
)
logger = logging.getLogger("server")

# ═══════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════════
VALID_MIME_TYPES = frozenset({"image/jpeg", "image/jpg", "image/png", "image/webp", "image/bmp"})
MAX_FILE_BYTES = MAX_UPLOAD_MB * 1024 * 1024
SESSION_TTL_SECONDS = 3600

# ═══════════════════════════════════════════════════════════════════════
# GLOBAL STATE
# ═══════════════════════════════════════════════════════════════════════
_pipeline = None
_groq_client: Groq | None = None
_async_groq_client: AsyncGroq | None = None
_pending_sessions: dict[str, dict] = {}
_interactive_flows: dict[str, Any] = {}
_packages_cache: list[dict] = []
_robotic_cache: list[dict] = []
_all_packages_cache: list[dict] = []
_search_index: list[dict] = []
_spelling_vocab_cache: set[str] = set()
_spelling_vocab_list_cache: list[str] = []
_abbrev_expansion_cache: dict[str, str] = {}



# ═══════════════════════════════════════════════════════════════════════
# CLINICAL KNOWLEDGE MAPS
# Centralised medical synonym & pathway expansions used by the
# keyword-based package scorer and the AI prompt builder.
# ═══════════════════════════════════════════════════════════════════════

# Maps symptom/condition phrases → related clinical terms for search expansion
CLINICAL_SEARCH_EXPANSIONS: dict[str, list[str]] = {
    # Cardiac
    "chest pain":       ["coronary", "angiography", "cardiac", "heart", "ptca", "cabg", "thrombolysis", "mi"],
    "heart attack":     ["mi", "myocardial", "thrombolysis", "ptca", "coronary", "stemi", "nstemi", "cabg"],
    "angioplasty":      ["ptca", "coronary", "stent", "cardiology", "pci"],
    "breathlessness":   ["heart failure", "cardiac", "pulmonary", "respiratory", "chf"],
    "rheumatic fever":  ["rheumatic fever", "valvular heart", "acute rheumatic"],
    "acute rheumatic fever": ["rheumatic fever", "valvular heart", "acute rheumatic"],
    # Gastrointestinal
    "stomach pain":     ["appendix", "gallbladder", "cholecyst", "pancreat", "intestin"],
    "abdominal pain":   ["appendix", "gallbladder", "cholecyst", "pancreat", "intestin", "hernia"],
    "appendix":         ["appendicitis", "appendicectomy", "appendicular"],
    "appendicitis":     ["appendix", "appendicectomy", "appendicular"],
    "cholecystectomy":  ["gallbladder", "cholecystitis", "laparoscopic"],
    "gastric":          ["gastric", "gastrectomy", "gastrojejunostomy", "ulcer", "stomach"],
    "hernia":           ["inguinal", "ventral", "umbilical", "hernia repair"],
    # Ophthalmology
    "eye":              ["cataract", "glaucoma", "retina", "phaco", "iol"],
    # Orthopaedics
    "knee":             ["arthroplasty", "tkr", "replacement", "arthroscopy", "ligament"],
    "hip":              ["arthroplasty", "thr", "replacement", "hemiarthroplasty", "fracture"],
    "fracture":         ["orif", "fixation", "plate", "nail", "reduction"],
    # Urology / Nephrology
    "kidney stone":     ["urolithiasis", "pcnl", "ursl", "lithotripsy", "renal"],
    "renal transplant": ["kidney transplant", "transplant", "nephrology", "urology"],
    "kidney transplant":["renal transplant", "transplant", "nephrology", "urology"],
    # Hepatology
    "liver transplant": ["hepatic transplant", "transplant", "surgical gastroenterology", "gastroenterology"],
    # Burns
    "burn":             ["burns", "graft", "debridement", "dressing", "skin", "eschar", "tbsa", "thermal", "electrical", "chemical", "flame"],
    "tbsa":             ["burns", "thermal", "flame", "electrical", "tbsa"],
    "electrical burn":  ["electrical", "contact", "burns", "tbsa"],
    "electrical burns": ["electrical", "contact", "burns", "tbsa"],
    "thermal burn":     ["thermal", "flame", "burns", "tbsa"],
    "thermal burns":    ["thermal", "flame", "burns", "tbsa"],
    # Endocrine
    "thyroid":          ["thyroidectomy", "endocrine", "ent"],
    "thyroid surgery":  ["thyroidectomy", "endocrine", "ent"],
    # Critical care / Haematology
    "icu":              ["icu", "intensive", "care"],
    "blood":            ["transfusion", "blood", "component", "packed", "ffp", "platelet"],
    "blood transfusion":["transfusion", "platelet", "packed", "whole blood", "component"],
}

# High-precision phrase → keyword boosters (exact match scoring)
PHRASE_PRIORITY_KEYWORDS: dict[str, list[str]] = {
    "angioplasty":      ["ptca", "coronary angioplasty", "coronary", "angioplasty", "pci"],
    "ptca":             ["ptca"],
    "tbsa":             ["tbsa", "thermal", "electrical", "flame"],
    "tbsa burns":       ["tbsa", "thermal", "electrical", "flame"],
    "burn":             ["tbsa", "burns", "thermal", "flame", "chemical", "electrical"],
    "electrical burn":  ["electrical contact burns", "electrical"],
    "electrical burns": ["electrical contact burns", "electrical"],
    "thermal burn":     ["thermal burns", "tbsa"],
    "thermal burns":    ["thermal burns", "tbsa"],
    "rheumatic fever":  ["rheumatic fever"],
    "acute rheumatic fever": ["rheumatic fever"],
    "icu":              ["icu", "intensive care unit"],
    "appendix":         ["appendicectomy", "appendicitis", "appendicular", "appendix"],
    "cholecystectomy":  ["cholecystectomy", "gallbladder", "cholecyst"],
    "hernia":           ["hernia", "inguinal", "umbilical", "ventral"],
    "thyroid":          ["thyroid", "thyroidectomy"],
    "renal transplant": ["renal transplant", "kidney transplant", "transplant"],
    "liver transplant": ["liver transplant", "hepatic transplant", "transplant"],
    "blood transfusion":["blood transfusion", "platelet transfusion", "whole blood", "component"],
}

# Condition → expected specialties (for mild off-specialty penalty)
CONDITION_SPECIALTY_HINTS: dict[str, list[str]] = {
    "angioplasty": ["cardiology", "interventional cardiology", "cath lab"],
    "appendix":    ["general surgery", "surgical gastroenterology", "laparoscopic"],
}

# Terms too generic to use alone in scoring
GENERIC_MEDICAL_TERMS = frozenset({
    "surgery", "surgical", "procedure", "management", "treatment", "operation",
    "package", "pain", "ache", "discomfort", "symptom", "disease",
})

# Symptom indicators that trigger strict specialty anchoring
SYMPTOM_QUERY_INDICATORS = frozenset({
    "pain", "fever", "breath", "cough", "bleeding", "swelling",
    "weakness", "dizziness", "attack",
})

# Implicit supportive add-on expansions (disease → supportive package)
IMPLICIT_ADDON_MAP: dict[str, list[str]] = {
    "anemia":                ["blood transfusion"],
    "anaemia":               ["blood transfusion"],
    "heart attack":          ["blood transfusion"],
    "myocardial infarction": ["blood transfusion"],
    "mi":                    ["blood transfusion"],
    "hemorrhage":            ["blood transfusion"],
    "haemorrhage":           ["blood transfusion"],
}

# Procedure name normalisations
PROCEDURE_ALIASES: dict[str, str] = {
    "appendectomy":    "appendicectomy",
    "gall bladder":    "gallbladder",
    "lap chole":       "laparoscopic cholecystectomy",
    "kidney transplant": "renal transplant",
    "liver tx":        "liver transplant",
}


# ═══════════════════════════════════════════════════════════════════════
# PACKAGE FIELD HELPERS  (DRY access to heterogeneous JSON keys)
# ═══════════════════════════════════════════════════════════════════════

def pkg_name(pkg: dict) -> str:
    return str(pkg.get("PACKAGE NAME", pkg.get("Package Name", "")))


def pkg_code(pkg: dict) -> str:
    return str(pkg.get("PACKAGE CODE", ""))


def pkg_rate(pkg: dict) -> float:
    raw = pkg.get("RATE", pkg.get("Rate", 0))
    try:
        return float(str(raw).replace(",", "").strip()) if raw else 0.0
    except Exception:
        return 0.0


def pkg_specialty(pkg: dict) -> str:
    return str(pkg.get("SPECIALITY", pkg.get("Speciality", "")))


def pkg_category(pkg: dict) -> str:
    return str(pkg.get("PACKAGE CATEGORY", pkg.get("PACKAGE TYPE", pkg.get("Procedure Type", ""))))


def pkg_implant_field(pkg: dict) -> str:
    return str(pkg.get("IMPLANT PACKAGE", pkg.get("IMPLANT", "NO IMPLANT")) or "NO IMPLANT")


def pkg_strat(pkg: dict) -> str:
    return str(pkg.get("STRATIFICATION PACKAGE", ""))


# ═══════════════════════════════════════════════════════════════════════
# TEXT NORMALISATION & SPELLING
# ═══════════════════════════════════════════════════════════════════════

def _normalize_search_text(value: str) -> str:
    """Apply procedure alias normalisation and whitespace cleanup."""
    text = (value or "").lower()
    for alias, canonical in PROCEDURE_ALIASES.items():
        text = text.replace(alias, canonical)
    return re.sub(r"\s+", " ", text).strip()


def _tokenize(value: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", value)


def _normalize_padded(value: str) -> str:
    """Return ` lowered text ` for substring phrase matching."""
    return f" {' '.join((value or '').lower().replace('/', ' ').replace('-', ' ').split())} "


async def _expand_abbreviations_llm(text: str) -> str:
    """Uses LLM to dynamically expand medical abbreviations in Indian clinical context."""
    if not text or len(text.strip()) > 100 or not _async_groq_client:
        return text

    # Check cache first
    cached = _abbrev_expansion_cache.get(text.lower())
    if cached:
        return cached

    words = text.split()
    possible_abbrev = False
    for w in words:
        clean = re.sub(r'[^a-zA-Z]', '', w)
        if 2 <= len(clean) <= 5 and clean.isupper():
            possible_abbrev = True
            break
        if 2 <= len(clean) <= 4 and clean.lower() not in {"and", "the", "for", "with", "pain", "leg", "arm", "eye", "ear", "cut", "burn"}:
            possible_abbrev = True
            break
            
    if not possible_abbrev:
        return text

    prompt = f"""Expand medical abbreviations in this query to their full PMJAY/Indian clinical terminology.
CRITICAL: DO NOT write any explanations. DO NOT use conversational language. ONLY output the final translated query.

Input: LAP CHOLE
Output: Laparoscopic Cholecystectomy

Input: PTCA
Output: Percutaneous Transluminal Coronary Angioplasty

Input: TKR
Output: Total Knee Replacement

Input: {text}
Output:"""
    try:
        resp = await _async_groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=60,
        )
        # Parse only the first line to strip trailing explanations if the LLM ignores instructions
        expanded = resp.choices[0].message.content.strip().split('\n')[0].strip('"').strip("'")
        
        # Hard safety bounds: If the output is suspiciously long, conversational, or contains weird characters, revert.
        if len(expanded) > len(text) + 50 or "since " in expanded.lower() or "assume " in expanded.lower() or "however" in expanded.lower():
            logger.warning("Rejected hallucinatory abbreviation expansion: '%s'", expanded)
            return text
            
        if expanded and expanded.lower() != text.lower() and len(expanded) > len(text):
            logger.info("LLM Expanded Abbreviation: '%s' -> '%s'", text, expanded)
            _abbrev_expansion_cache[text.lower()] = expanded
            return expanded
        
        # Cache failed or unexpanded results to avoid retrying
        _abbrev_expansion_cache[text.lower()] = text
    except Exception as e:
        logger.warning("LLM abbreviation expansion failed: %s", e)
    return text


def _build_spelling_vocab() -> set[str]:
    global _spelling_vocab_cache, _spelling_vocab_list_cache
    if _spelling_vocab_cache:
        return _spelling_vocab_cache
    _load_packages_cache()
    vocab: set[str] = set()
    for pkg in (_packages_cache + _robotic_cache):
        blob = " ".join([
            pkg_name(pkg), pkg_specialty(pkg), pkg_category(pkg),
            str(pkg.get("Procedure Sub Category", "")),
        ]).lower()
        for tok in re.findall(r"[a-z]{4,}", blob):
            vocab.add(tok)
    _spelling_vocab_cache = vocab
    _spelling_vocab_list_cache = list(vocab)
    return vocab


def _correct_query_terms_spelling(raw_terms: list[str]) -> tuple[list[str], dict[str, str]]:
    """Conservative spell-correction against medical vocabulary."""
    vocab = _build_spelling_vocab()
    if not vocab:
        return raw_terms, {}

    corrections: dict[str, str] = {}
    corrected: list[str] = []

    for term in raw_terms:
        original = (term or "").strip()
        if not original:
            continue
        tokens = re.findall(r"[a-z0-9]+", original.lower())
        if not tokens:
            corrected.append(original)
            continue

        new_tokens: list[str] = []
        changed = False
        for tok in tokens:
            if len(tok) < 4 or tok.isdigit() or tok in vocab:
                new_tokens.append(tok)
                continue
            candidates = difflib.get_close_matches(tok, _spelling_vocab_list_cache, n=3, cutoff=0.84)
            replacement = tok
            for c in candidates:
                if c and c[0] == tok[0] and abs(len(c) - len(tok)) <= 2:
                    replacement = c
                    break
            new_tokens.append(replacement)
            if replacement != tok:
                changed = True

        corrected_term = " ".join(new_tokens).strip()
        if changed and corrected_term:
            corrections[original] = corrected_term
            corrected.append(corrected_term)
        else:
            corrected.append(original)

    return corrected, corrections


# ═══════════════════════════════════════════════════════════════════════
# PACKAGE DATA LOADING
# ═══════════════════════════════════════════════════════════════════════

def _load_packages_cache():
    global _packages_cache, _robotic_cache, _all_packages_cache, _search_index
    if _packages_cache and _robotic_cache:
        return

    from config.settings import BASE_DIR

    def _load_json(filename: str) -> list[dict]:
        for candidate in [
            BASE_DIR.parent / "assets" / filename,
            BASE_DIR / "assets" / filename,
            Path.cwd() / "assets" / filename,
        ]:
            try:
                if candidate.exists():
                    with open(candidate, "r", encoding="utf-8-sig") as f:
                        data = json.load(f)
                    logger.info("Loaded %d entries from %s", len(data), candidate)
                    return data
            except Exception as exc:
                logger.warning("Failed loading %s from %s: %s", filename, candidate, exc)
        logger.warning("Could not locate %s", filename)
        return []

    _packages_cache = _load_json("maa_packages.json")
    _robotic_cache = _load_json("maa_robotic_surgeries.json")
    _all_packages_cache = _packages_cache + _robotic_cache
    logger.info("Package cache: %d standard + %d robotic", len(_packages_cache), len(_robotic_cache))

    # Pre-compute search index — normalise text/tokens ONCE for all packages
    _search_index = []
    for pkg in _all_packages_cache:
        name = _normalize_search_text(pkg_name(pkg))
        code = _normalize_search_text(pkg_code(pkg))
        spec = _normalize_search_text(pkg_specialty(pkg))
        strat = _normalize_search_text(pkg_strat(pkg))
        _search_index.append({
            "pkg": pkg,
            "name": name, "name_tok": _tokenize(name),
            "code": code, "code_tok": _tokenize(code),
            "spec": spec, "spec_tok": _tokenize(spec),
            "strat": strat, "strat_tok": _tokenize(strat),
        })


def _all_packages() -> list[dict]:
    _load_packages_cache()
    return _all_packages_cache


# ═══════════════════════════════════════════════════════════════════════
# PACKAGE TYPE IDENTIFICATION & COMBINATION VALIDATION
# (MAA Yojana / PMJAY business rules)
# ═══════════════════════════════════════════════════════════════════════

def _identify_package_type(pkg: dict) -> dict[str, bool]:
    name_upper = pkg_name(pkg).upper()
    rate = pkg_rate(pkg)
    implant_upper = pkg_implant_field(pkg).upper()
    cat_upper = pkg_category(pkg).upper()

    return {
        "is_surgical":           rate > 0 and ("[REGULAR PROCEDURE]" in name_upper or "REGULAR PKG" in cat_upper),
        "is_medical_management": rate == 0 and "[ADD-ON" not in name_upper and "[ADD ON" not in name_upper,
        "is_standalone":         any(s in name_upper or s in cat_upper for s in ("STAND-ALONE", "STAND ALONE")),
        "is_addon":              any(s in name_upper or s in cat_upper for s in ("[ADD-ON", "[ADD ON", "ADDON", "ADD-ON", "ADD ON")),
        "is_implant":            ("IMPLANT" in implant_upper and "NO IMPLANT" not in implant_upper) or cat_upper == "IMP",
        "is_extended_los":       "EXTENDED LOS" in name_upper,
    }


def _validate_package_combination(main_type: dict[str, bool], candidate_type: dict[str, bool], candidate_code: str) -> str | None:
    """Return violation message or None if combination is valid."""
    if candidate_type["is_standalone"]:
        return f"Rule 2 VIOLATION: {candidate_code} is Stand-alone – cannot combine"
    if main_type["is_standalone"]:
        return f"Rule 2 VIOLATION: Main package is Stand-alone – cannot add {candidate_code}"
    if main_type["is_surgical"] and candidate_type["is_medical_management"]:
        return f"Rule 1 VIOLATION: Cannot combine surgical + medical management ({candidate_code}, rate=₹0)"
    if main_type["is_medical_management"] and candidate_type["is_surgical"]:
        return f"Rule 1 VIOLATION: Cannot combine medical management + surgical ({candidate_code})"
    if candidate_type["is_extended_los"] and not main_type["is_surgical"]:
        return f"Rule 5 VIOLATION: Extended LOS ({candidate_code}) only with surgery"
    return None


# ═══════════════════════════════════════════════════════════════════════
# KEYWORD PACKAGE SEARCH  (pre-filter + medical synonym expansion)
# ═══════════════════════════════════════════════════════════════════════

def _has_term(term: str, text: str, tokens: list[str]) -> bool:
    if not term:
        return False
    if " " in term:
        return term in text
    return any(tok == term or tok.startswith(term) for tok in tokens)


_PEDIA_KEYWORDS = (
    "pediatric", "paediatric", "neonatal", "neonate", "neonat",
    "children", "child", "infant", "infantile", "newborn",
    "juvenile", "toddler", "baby",
)

def _is_pediatric_package(name: str, spec: str) -> bool:
    """Return True if the package is clearly pediatric / child-specific."""
    combined = f"{name} {spec}"
    return any(kw in combined for kw in _PEDIA_KEYWORDS)


def _passes_patient_type(pkg: dict, pt_type: str) -> bool:
    """Filter packages based on patient demographic (Adult vs Pediatric)."""
    if not pt_type:
        return True
    name = str(pkg.get("PACKAGE NAME", "")).lower()
    spec = str(pkg.get("SPECIALITY", "")).lower()
    is_pedia = _is_pediatric_package(name, spec)

    if pt_type.lower() == "adult":
        # Exclude packages that are clearly pediatric/child-specific
        return not is_pedia
    elif pt_type.lower() == "pediatric":
        # Include pediatric packages + general packages; exclude explicit "adult" only packages
        if is_pedia:
            return True
        # Exclude packages that explicitly say "adult" in their name
        if "adult" in name:
            return False
        # Allow general packages (not explicitly adult, not explicitly pediatric)
        return True
    return True

@functools.lru_cache(maxsize=1024)
def _cached_search_packages_basic(query: str, limit: int = 50, patient_type: str = "") -> list[dict]:
    """Score and rank packages against a clinical query string using pre-computed index."""
    _load_packages_cache()
    query_lower = _normalize_search_text(query)
    terms = _tokenize(query_lower)
    filtered_terms = [t for t in terms if t not in GENERIC_MEDICAL_TERMS] or terms
    normalized_query = " ".join(filtered_terms).strip()

    # Expand search terms via clinical synonym map
    expanded_terms: set[str] = set(filtered_terms)
    active_related: set[str] = set()
    for trigger, related in CLINICAL_SEARCH_EXPANSIONS.items():
        if trigger in query_lower or any(t in trigger for t in filtered_terms):
            expanded_terms.update(related)
            active_related.update(related)

    # Determine intent specialties (imports already at module level)
    intent_specialties: set[str] = {s.lower().strip() for s in get_specialties_for_term(query_lower)}
    pathway = get_clinical_pathway(query_lower)
    if pathway:
        ps = pathway.get("specialty")
        if isinstance(ps, list):
            intent_specialties.update(str(s).strip().lower() for s in ps if str(s).strip())
        elif isinstance(ps, str) and ps.strip():
            intent_specialties.add(ps.strip().lower())
        for step in pathway.get("steps", []):
            ss = str(step.get("specialty", "")).strip().lower()
            if ss:
                intent_specialties.add(ss)

    strict_intent = bool(intent_specialties) and any(ind in query_lower for ind in SYMPTOM_QUERY_INDICATORS)

    # Pre-compute expansion-only terms once
    expansion_only = expanded_terms - set(filtered_terms)
    scored: list[tuple[int, int, dict]] = []

    for idx_entry in _search_index:
        pkg = idx_entry["pkg"]
        if not _passes_patient_type(pkg, patient_type):
            continue

        name = idx_entry["name"];      name_tok = idx_entry["name_tok"]
        code = idx_entry["code"];      code_tok = idx_entry["code_tok"]
        spec = idx_entry["spec"];      spec_tok = idx_entry["spec_tok"]
        strat = idx_entry["strat"];    strat_tok = idx_entry["strat_tok"]

        score = 0
        hits_name = hits_code = hits_spec = hits_strat = 0

        for t in filtered_terms:
            if _has_term(t, code, code_tok):  score += 15; hits_code += 1
            if _has_term(t, name, name_tok):  score += 10; hits_name += 1
            if _has_term(t, spec, spec_tok):  score += 5;  hits_spec += 1
            if _has_term(t, strat, strat_tok): score += 15; hits_strat += 1

        # Full-phrase direct match bonuses
        if normalized_query:
            if normalized_query in code: score += 90
            if normalized_query in name: score += 75
            if normalized_query in spec: score += 30
            if normalized_query in strat: score += 40

        # Exact-match priority tier
        exact_p = 0
        if normalized_query:
            if normalized_query == code:                        exact_p = 5
            elif normalized_query == name:                      exact_p = 4
            elif f" {normalized_query} " in f" {name} ":       exact_p = 3
            elif f" {normalized_query} " in f" {spec} ":       exact_p = 2

        if len(filtered_terms) == 1:
            tok = filtered_terms[0]
            if any(t == tok for t in name_tok):   exact_p = max(exact_p, 4)
            elif any(t == tok for t in code_tok): exact_p = max(exact_p, 4)
            elif any(t == tok for t in spec_tok): exact_p = max(exact_p, 3)

        total = len(filtered_terms)
        if total:
            if hits_name == total: score += 40
            elif hits_name: score += hits_name * 8
            if hits_code == total: score += 45
            elif hits_code: score += hits_code * 10
            if total == 1 and _has_term(filtered_terms[0], name, name_tok):
                score += 25
            if (hits_name + hits_code + hits_spec) == 0:
                score -= 12

        # Expanded-term scoring (lower weight to avoid drowning direct matches)
        for t in expansion_only:
            if _has_term(t, code, code_tok): score += 8
            if _has_term(t, name, name_tok): score += 5
            if _has_term(t, spec, spec_tok): score += 3

        # Phrase-priority boosts
        for trigger, kws in PHRASE_PRIORITY_KEYWORDS.items():
            if trigger in query_lower and any(_has_term(k, name, name_tok) for k in kws):
                score += 40

        # Off-specialty penalty
        for trigger, hints in CONDITION_SPECIALTY_HINTS.items():
            if trigger in query_lower and not any(_has_term(h, spec, spec_tok) for h in hints):
                score -= 8

        # Intent-specialty anchoring
        spec_padded = _normalize_padded(spec)
        matches_intent = any(_normalize_padded(s) in spec_padded for s in intent_specialties)
        related_match = any(
            _has_term(r, name, name_tok) or _has_term(r, spec, spec_tok) or _has_term(r, code, code_tok)
            for r in active_related
        )
        if matches_intent:
            score += 10
        elif strict_intent:
            score -= 18
            if not related_match:
                continue

        # Coronary angioplasty refinement
        if "angioplasty" in query_lower and "peripheral" not in query_lower:
            if _has_term("ptca", name, name_tok) or _has_term("coronary", name, name_tok):
                score += 16
            if _has_term("peripheral", name, name_tok) or _has_term("peripheral", spec, spec_tok):
                score -= 10

        if score > 0:
            scored.append((exact_p, score, pkg))

    scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
    return [p for _, _, p in scored[:limit]]


def _search_packages_basic(query: str, limit: int = 50, patient_type: str = "") -> list[dict]:
    """Wrapper to retrieve cached search results safely."""
    # Normalize before caching to maximize cache hits
    normalized_query = _normalize_search_text(query)
    # We return a shallow copy to protect the internal LRU cache from mutation
    return list(_cached_search_packages_basic(normalized_query, limit, patient_type))


def _prioritize_exact_main_term_first(packages: list[dict], main_term: str) -> list[dict]:
    """Stable re-sort: exact main-term hits bubble to top."""
    norm = _normalize_search_text(main_term or "")
    if not norm:
        return packages
    term_tokens = _tokenize(norm)
    if not term_tokens:
        return packages
    padded = f" {norm} "

    def _rank(pkg: dict) -> int:
        full = _normalize_search_text(pkg_name(pkg))
        primary = _normalize_search_text(full.split("|")[0] if "|" in full else full)
        primary_tok = _tokenize(primary)
        full_tok = _tokenize(full)
        if padded in f" {primary} ":                          return 0
        if len(term_tokens) == 1 and term_tokens[0] in primary_tok: return 1
        if padded in f" {full} ":                             return 2
        if all(t in full_tok for t in term_tokens):           return 3
        if any(t in full_tok for t in term_tokens):           return 4
        if norm in full:                                      return 5
        return 6

    return sorted(packages, key=_rank)


# ═══════════════════════════════════════════════════════════════════════
# AI / LLM UTILITIES
# ═══════════════════════════════════════════════════════════════════════

def _format_packages_for_ai(packages: list[dict], n: int = 20) -> str:
    lines = []
    for i, pkg in enumerate(packages[:n]):
        imp = pkg_implant_field(pkg)
        lines.append(
            f"{i+1}. [{pkg_code(pkg)}] {pkg_name(pkg)[:100]}… | ₹{pkg_rate(pkg)} "
            f"| {pkg_specialty(pkg)} | {pkg_category(pkg)} | Implant: {imp[:50] if imp else 'NO'}"
        )
    return "\n".join(lines)


def _get_ai_system_prompt(mode: str) -> str:
    """Build mode-specific system prompt with MAA Yojana rules for Groq."""
    base_rules = """═══════════════════════════════════════════════════════════════
STRICT MAA YOJANA/PMJAY PACKAGE COMBINATION RULES:
═══════════════════════════════════════════════════════════════
RULE 1: Surgical + Medical Management = BLOCKED (₹0 rate = medical mgmt)
RULE 2: Stand-Alone packages = EXCLUSIVE (no combinations)
RULE 3: Add-On packages (ICU, anaesthesia, extended stay) = ALLOWED with regular procedures
RULE 4: Implant packages = AUTO-POPUP with procedures requiring implants
RULE 5: Extended LOS = ALLOWED only with surgery packages
RULE 6: ONE main procedure/disease package per claim

APPROVAL CRITERIA:
- Package must match the EXACT procedure/disease being treated
- Patient must be eligible (Ayushman/MAA Yojana card holder)
- Hospital must be empaneled for this specific package
- Pre-authorisation required for most surgeries"""

    mode_sections = {
        "smart": """
MODE: MAA YOJANA SMART PACKAGE SELECTOR (Dr. Arth – Clinical Expert)

ROLE: Senior consultant with 15+ years PMJAY/Ayushman Bharat experience.
Think like a doctor: symptoms → differential → treatment → package → validation.

CLINICAL REASONING PATHWAYS:
🫀 Chest Pain: Angiography → MI confirmed? → Thrombolysis/PTCA/CABG
🫁 Breathlessness: Cardiac → HF pkg | Pulmonary → Pulmonology
🤕 Abdominal: RLQ→Appendectomy | RUQ→Cholecystectomy | Epigastric→Pancreatitis/PUD
🦴 Ortho: Fracture→ORIF+implant | Knee OA→TKR | Hip→THR/Hemiarthroplasty
👁  Eye: Lens opacity→Phaco+IOL | Pressure→Glaucoma surgery

WORKFLOW: 1)Identify 2)Diagnose 3)Treat 4)Package 5)Add-ons 6)Validate

MANDATORY:
- ALWAYS provide main_package_code – NEVER null
- For vague symptoms suggest the DIAGNOSTIC package first
- For chest pain → Coronary Angiography first
- Think as consulting physician, not keyword matcher
- First comma-separated term = MAIN package, rest = ADD-ONS""",
        "procedure": """
MODE: PROCEDURE SEARCH – Find exact surgical/medical procedure package.
Match procedure name to exact code, suggest required implants, add relevant add-ons.""",
        "disease": """
MODE: DISEASE/CONDITION SEARCH – Find packages that TREAT this condition.
Consider surgical vs conservative approach, select most appropriate treatment.""",
    }
    mode_text = mode_sections.get(mode, "\nMODE: GENERAL SEARCH – Find the single most relevant package.")

    return f"""You are Dr. Arth, expert MAA Yojana/PMJAY package consultant (Gujarat, India).
{mode_text}
{base_rules}

CRITICAL OUTPUT RULES:
- Return ONLY exact, highly related packages. Give PRECISE, DEFINITIVE, and non-vague reasoning. Do not use broad categories.
- Translate layman/simple terms (e.g. "kidney", "heart", "eye") to accurate medical terminology (e.g. "renal", "cardiac", "ophthalmic") before determining matches.
- Include 3-step clinical chain in doctor_summary: layman term mapping → exact clinical diagnosis → treatment → justification.
- ALWAYS proactively suggest 1-3 highly relevant clinical ADD-ON packages (e.g. blood transfusion, ICU, anaesthesia, biopsy, extended LOS, implants) in 'addons' if clinically appropriate and allowed by rules, even if not explicitly requested.
- If blocked_rules has violations set approval_likelihood to "REJECTED"

Return ONLY valid JSON:
{{
    "main_package_code": "EXACT_CODE or null",
    "main_package_reason": "Provide precise medical justification why this package fits",
    "implant_code": "IMPLANT_CODE or null",
    "addons": [{{"code": "CODE", "reason": "Specific medical justification for this add-on"}}],
    "alternative_codes": ["Max 2 precise alternatives"],
    "blocked_rules": ["Any rule violations"],
    "approval_likelihood": "HIGH / MEDIUM / LOW / REJECTED",
    "doctor_summary": "Precise, step-by-step clinical assessment and terminology mapping without vague statements."
}}"""


async def _classify_input_intent(term_specialties_map: dict[str, list[str]]) -> dict[str, str]:
    """Classify terms as Surgical / Medical using deterministic rules then Groq fallback."""
    if not _async_groq_client or not term_specialties_map:
        return {t: "Unknown" for t in term_specialties_map}

    intents: dict[str, str] = {}
    need_ai: dict[str, list[str]] = {}

    for term, specs in term_specialties_map.items():
        tl = term.lower()
        if is_medical_management_term(term) or tl in {"sepsis", "anemia", "fever", "infection", "shock"}:
            intents[term] = "Medical"
        elif is_surgical_term(term):
            intents[term] = "Surgical"
        else:
            need_ai[term] = specs

    if not need_ai:
        return intents

    details = "\n".join(
        f"- Term: '{t}' (Specialties: {', '.join(s) if s else 'None'})"
        for t, s in need_ai.items()
    )
    prompt = f"""Classify each medical term as "Surgical" or "Medical".
Surgical = requires operation. Medical = conservative/management.
"Blood", "Blood Transfusion", "ICU", "Extended LOS", implants → always "Medical".

{details}

Return ONLY JSON: {{"term": "Surgical"|"Medical", ...}}"""

    try:
        resp = await _async_groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            response_format={"type": "json_object"},
        )
        intents.update(json.loads(resp.choices[0].message.content))
    except Exception as e:
        logger.warning("AI intent classification failed: %s", e)
        for t in need_ai:
            intents[t] = "Unknown"
    return intents


def _check_intent_rule_violation(intents: dict[str, str]) -> str | None:
    has_surg = any(v.lower() == "surgical" for v in intents.values())
    has_med = any(v.lower() == "medical" for v in intents.values())
    if has_surg and has_med:
        surg = [k for k, v in intents.items() if v.lower() == "surgical"]
        med = [k for k, v in intents.items() if v.lower() == "medical"]
        return f"Rule 1 VIOLATION: Cannot combine surgical ({', '.join(surg)}) with medical management ({', '.join(med)})."
    return None


# ═══════════════════════════════════════════════════════════════════════
# QUERY TERM UTILITIES
# ═══════════════════════════════════════════════════════════════════════

def _split_query_terms(raw_query: str) -> list[str]:
    return [p.strip() for p in (raw_query or "").replace(";", ",").replace("|", ",").split(",") if p.strip()]


def _append_unique_term(target: list[str], value: str) -> None:
    val = (value or "").strip()
    if val and not any(e.lower() == val.lower() for e in target):
        target.append(val)


def _expand_implicit_addon_terms(main_term: str) -> list[str]:
    term = (main_term or "").lower()
    implied: list[str] = []
    for key, addons in IMPLICIT_ADDON_MAP.items():
        if key in term:
            for a in addons:
                _append_unique_term(implied, a)
    return implied


def _is_transfusion_term(term: str) -> bool:
    t = (term or "").lower().strip()
    return "transfusion" in t or "blood" in t


def _build_raw_package_row(pkg: dict, ai_selected: bool = False) -> dict:
    return {"code": pkg_code(pkg), "name": pkg_name(pkg), "rate": pkg_rate(pkg),
            "speciality": pkg_specialty(pkg), "ai_selected": ai_selected}


# ═══════════════════════════════════════════════════════════════════════
# SESSION MANAGEMENT
# ═══════════════════════════════════════════════════════════════════════

def _prune_stale_sessions() -> None:
    now = time.time()
    stale = [sid for sid, d in _pending_sessions.items()
             if now - d.get("created_at", now) > SESSION_TTL_SECONDS]
    for sid in stale:
        _pending_sessions.pop(sid, None)
    if stale:
        logger.info("Pruned %d stale session(s)", len(stale))


def _normalize_interactive_step_title(step_name: str) -> str:
    n = (step_name or "").strip().lower()
    if "supportive" in n and "suggest" in n:
        return "Add Ons (If Applicable):"
    if n in {"supportive suggestion for:", "supportive suggestions for:",
             "supportive suggestion", "supportive suggestions"}:
        return "Add Ons (If Applicable):"
    return step_name


def _auto_advance_single_option_steps(flow: Any, packages: list[dict]) -> None:
    from tools.smart_search_flow import process_step_selection
    safety = 0
    while not flow.flow_complete and flow.current_step < len(flow.steps) and safety < 20:
        safety += 1
        step = flow.steps[flow.current_step]
        opts = step.options or []
        if len(opts) != 1:
            break
        opt_id = str(opts[0].get("id", ""))
        if opt_id.startswith("manual_add"):
            break
        ok, _ = process_step_selection(flow, {"id": opt_id}, packages)
        if not ok:
            break


# ═══════════════════════════════════════════════════════════════════════
# FILE / MIME HELPERS
# ═══════════════════════════════════════════════════════════════════════

def _detect_mime(upload: UploadFile) -> str:
    mime = getattr(upload, "content_type", None)
    if not mime or mime in ("application/octet-stream", "application/x-www-form-urlencoded"):
        fn = (upload.filename or "").lower()
        if fn.endswith(".png"):  return "image/png"
        if fn.endswith(".webp"): return "image/webp"
        return "image/jpeg"
    return mime


async def _run_pipeline_async(state: dict | None, config: dict) -> dict:
    final = state or {}
    async for event in _pipeline.astream(state, config=config, stream_mode="values"):
        final = event
    return final


# ═══════════════════════════════════════════════════════════════════════
# APP LIFESPAN
# ═══════════════════════════════════════════════════════════════════════

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _pipeline, _groq_client, _async_groq_client
    gc.collect()

    from config.settings import FIREBASE_SERVICE_ACCOUNT
    if not GROQ_API_KEY:
        logger.error("⚠️ GROQ_API_KEY not set! https://console.groq.com/keys")
    else:
        _groq_client = Groq(api_key=GROQ_API_KEY)
        _async_groq_client = AsyncGroq(api_key=GROQ_API_KEY)
        logger.info("✅ Groq API keys configured — clients initialised")

    if not FIREBASE_SERVICE_ACCOUNT:
        logger.warning("⚠️ FIREBASE_SERVICE_ACCOUNT not set – push notifications disabled")
    else:
        logger.info("✅ Firebase Service Account configured")

    logger.info("🚀 Compiling LangGraph pipeline…")
    try:
        from graph.pipeline import get_compiled_graph
        _pipeline = get_compiled_graph()
        logger.info("✅ LangGraph pipeline ready")
    except Exception as e:
        logger.error("Failed to compile pipeline: %s", e)
        _pipeline = None

    yield
    gc.collect()
    logger.info("Server shutdown complete")


# ═══════════════════════════════════════════════════════════════════════
# FASTAPI APP & MIDDLEWARE
# ═══════════════════════════════════════════════════════════════════════

app = FastAPI(
    title="MediPackages OCR Server (Groq Vision)",
    version="4.0.0",
    description="Groq-powered medical OCR: Image → Extraction → Validation → MAA Yojana Package Matching",
    lifespan=lifespan,
    docs_url="/docs" if ENABLE_DOCS else None,
    redoc_url="/redoc" if ENABLE_DOCS else None,
    openapi_url="/openapi.json" if ENABLE_DOCS else None,
)


class RequestContextMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        rid = request.headers.get("X-Request-ID", str(uuid.uuid4()))
        request.state.request_id = rid
        start = time.perf_counter()
        response = await call_next(request)
        ms = (time.perf_counter() - start) * 1000
        response.headers.update({
            "X-Request-ID": rid,
            "X-Content-Type-Options": "nosniff",
            "X-Frame-Options": "DENY",
            "Referrer-Policy": "strict-origin-when-cross-origin",
            "X-Response-Time-Ms": f"{ms:.2f}",
        })
        logger.info("%s %s -> %s %.2fms (%s)", request.method, request.url.path, response.status_code, ms, rid)
        return response


app.add_middleware(RequestContextMiddleware)
app.add_middleware(GZipMiddleware, minimum_size=1024)
app.add_middleware(TrustedHostMiddleware, allowed_hosts=TRUSTED_HOSTS)
app.add_middleware(CORSMiddleware, allow_origins=CORS_ORIGINS, allow_credentials=CORS_ALLOW_CREDENTIALS,
                   allow_methods=["*"], allow_headers=["*"])


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    rid = getattr(request.state, "request_id", "unknown")
    logger.warning("Validation error (%s): %s", rid, exc.errors())
    return JSONResponse(status_code=422, content={"success": False, "error": "Validation error",
                                                   "details": exc.errors(), "request_id": rid})


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    rid = getattr(request.state, "request_id", "unknown")
    logger.error("Unhandled exception (%s): %s", rid, exc, exc_info=exc)
    return JSONResponse(status_code=500, content={"success": False, "error": "Internal server error", "request_id": rid})


# ═══════════════════════════════════════════════════════════════════════
# AUTH
# ═══════════════════════════════════════════════════════════════════════
api_key_scheme = APIKeyHeader(name="X-API-Key", auto_error=False)


def get_api_key(api_key_value: str = Security(api_key_scheme)):
    if api_key_value and secrets.compare_digest(api_key_value, API_AUTH_TOKEN):
        return api_key_value
    logger.warning("Unauthorized access attempt")
    raise HTTPException(403, "Could not validate credentials")


# ═══════════════════════════════════════════════════════════════════════
# REQUEST / RESPONSE MODELS
# ═══════════════════════════════════════════════════════════════════════

class PushMessage(BaseModel):
    token: str
    title: str
    body: str
    icon: Optional[str] = "not_icon"
    color: Optional[str] = "#0052D4"


class PushNotificationRequest(BaseModel):
    messages: List[PushMessage]


class FeedbackRequest(BaseModel):
    session_id: str
    decision: str
    reason: Optional[str] = ""
    corrections: Optional[dict] = None


class SmartSearchRequest(BaseModel):
    query: str
    mode: str = "normal"
    procedure: str = ""
    disease: str = ""
    symptoms: list[str] = []
    patient_age: int = 0
    patient_gender: str = ""
    limit: int = 50


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
    approval_likelihood: str = ""


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
    patient_type: str = ""


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


# ═══════════════════════════════════════════════════════════════════════
# STEP RESPONSE BUILDER (DRY)
# ═══════════════════════════════════════════════════════════════════════

def _step_to_response(step) -> SearchStepResponse:
    return SearchStepResponse(
        step_number=step.step_number,
        step_name=_normalize_interactive_step_title(step.step_name),
        description=step.description,
        options=[SearchOption(
            id=o.get("id", ""), label=o.get("label", ""), description=o.get("description", ""),
            specialty=o.get("specialty"), code=o.get("code"), rate=o.get("rate"),
            reasoning=o.get("reasoning") or o.get("reason"), rank=o.get("rank"),
        ) for o in step.options],
        requires_user_selection=step.requires_user_selection,
        context=step.context,
    )


# ═══════════════════════════════════════════════════════════════════════
# ROUTES: OCR EXTRACTION
# ═══════════════════════════════════════════════════════════════════════

@app.post("/extract")
async def extract_ocr(image: UploadFile = File(...), api_key: str = Depends(get_api_key)):
    """Extract structured medical data from an uploaded document image."""
    if _pipeline is None:
        raise HTTPException(503, "Pipeline not initialized")

    mime_type = _detect_mime(image)
    if mime_type not in VALID_MIME_TYPES:
        raise HTTPException(415, f"Unsupported file type: {mime_type}")

    # Read with size guard
    try:
        contents = bytearray()
        while chunk := await image.read(1024 * 1024):
            contents.extend(chunk)
            if len(contents) > MAX_FILE_BYTES:
                await image.close()
                raise HTTPException(413, f"File too large. Max: {MAX_UPLOAD_MB}MB")
        contents = bytes(contents)
    except HTTPException:
        raise
    except Exception as e:
        logger.error("File upload error: %s", e)
        raise HTTPException(400, "Failed to read upload")
    finally:
        await image.close()

    logger.info("Processing: %s (%0.fKB, %s)", image.filename, len(contents) / 1024, mime_type)

    # Resize to reduce vision-model token cost
    try:
        with Image.open(io.BytesIO(contents)) as img:
            if img.mode in ("RGBA", "P"):
                img = img.convert("RGB")
            if max(img.size) > 1024:
                ratio = 1024 / max(img.size)
                new_sz = (int(img.size[0] * ratio), int(img.size[1] * ratio))
                img = img.resize(new_sz, Image.Resampling.LANCZOS)
                buf = io.BytesIO()
                img.save(buf, format="JPEG", quality=85)
                contents = buf.getvalue()
                mime_type = "image/jpeg"
                logger.info("Resized to %s (%0.fKB)", new_sz, len(contents) / 1024)
    except Exception as e:
        logger.warning("Resize failed, using original: %s", e)

    session_id = str(uuid.uuid4())
    image_b64 = base64.b64encode(contents).decode()
    del contents

    initial_state = {"session_id": session_id, "image_b64": image_b64,
                     "mime_type": mime_type, "retry_count": 0, "supervisor_notes": []}
    config = {"configurable": {"thread_id": session_id}}

    async def event_generator():
        start_t = time.time()
        yield f"data: {json.dumps({'status': 'progress', 'message': 'Image pre-processing completed.'})}\n\n"
        try:
            final = initial_state
            async for ev in _pipeline.astream(initial_state, config=config, stream_mode="values"):
                final = ev
                notes = ev.get("supervisor_notes", [])
                yield f"data: {json.dumps({'status': 'progress', 'message': notes[-1] if notes else 'Pipeline step completed...'})}\n\n"
                if ev.get("__interrupt__") or "__interrupt__" in ev:
                    final["__interrupted__"] = True
                    break

            elapsed = time.time() - start_t
            extracted = final.get("extracted", {})
            validation = final.get("validation", {})

            session_data = {
                "thread_id": session_id, "extracted": extracted, "validation": validation,
                "retry_count": 0, "image_b64": image_b64, "mime_type": mime_type,
                "raw_text": final.get("raw_text", ""), "vision_text": final.get("vision_text", ""),
                "created_at": time.time(),
            }
            _pending_sessions[session_id] = session_data
            try:
                from memory.sqlite_store import AgentMemory
                AgentMemory().save_session(session_id, "ocr_pending", session_data)
            except Exception as e:
                logger.error("Failed to persist OCR session: %s", e)

            _prune_stale_sessions()

            yield f"data: {json.dumps({'success': True, 'status': 'pending_review', 'session_id': session_id, 'message': 'AI Doctor summary ready — please verify.', 'preview': extracted, 'validation': validation, 'processing_time_seconds': round(elapsed, 2)})}\n\n"
            logger.info("✅ Phase A complete in %.2fs — awaiting verification", elapsed)
        except Exception as e:
            logger.exception("Pipeline streaming error")
            yield f"data: {json.dumps({'success': False, 'status': 'error', 'error': str(e)})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"})


# ═══════════════════════════════════════════════════════════════════════
# ROUTES: FEEDBACK & RETRY
# ═══════════════════════════════════════════════════════════════════════

def _recover_session(session_id: str) -> dict:
    """Return pending session from memory or DB; raise 404 if missing."""
    if session_id in _pending_sessions:
        return _pending_sessions[session_id]
    try:
        from memory.sqlite_store import AgentMemory
        data = AgentMemory().get_session(session_id)
        if data:
            _pending_sessions[session_id] = data
            return data
    except Exception:
        pass
    raise HTTPException(404, f"No pending session: {session_id}")


@app.post("/feedback")
async def submit_feedback(req: FeedbackRequest, api_key: str = Depends(get_api_key)):
    """Human-in-the-loop: approve → resume pipeline; reject → store for learning."""
    session = _recover_session(req.session_id)
    if req.decision not in ("approved", "rejected"):
        raise HTTPException(400, "decision must be 'approved' or 'rejected'")

    if req.decision == "rejected":
        try:
            from memory.sqlite_store import AgentMemory
            mem = AgentMemory()
            rid = mem.create_request(req.session_id, session.get("raw_text", ""))
            mem.update_request(rid, session.get("extracted", {}), status="rejected")
            mem.store_feedback(request_id=rid, decision="rejected", reason=req.reason or "User rejected")
        except Exception as e:
            logger.warning("Failed to store rejection: %s", e)

        rc = session.get("retry_count", 0)
        return {"success": True, "status": "rejected", "message": "Rejection stored. AI will learn.",
                "retry_available": rc < 3, "retry_count": rc}

    # Approved → Phase B
    config = {"configurable": {"thread_id": req.session_id}}
    resume = {
        "session_id": req.session_id, "extracted": session.get("extracted", {}),
        "validation": session.get("validation", {}), "image_b64": session.get("image_b64", ""),
        "mime_type": session.get("mime_type", "image/jpeg"),
        "human_decision": "approved", "human_reason": req.reason or "",
        "human_correction": req.corrections or {},
    }
    if req.corrections:
        for field, val in req.corrections.items():
            resume["extracted"][field] = val

    try:
        _pipeline.update_state(config, resume)
        result = await _run_pipeline_async(None, config)
        _pending_sessions.pop(req.session_id, None)
        try:
            from memory.sqlite_store import AgentMemory
            AgentMemory().delete_session(req.session_id)
        except Exception:
            pass

        final = result.get("final_response") or {
            "success": True,
            "data": {**resume.get("extracted", {}), "_agentic_data": {"best_packages": result.get("best_packages", [])}}
        }

        try:
            from memory.sqlite_store import AgentMemory
            mem = AgentMemory()
            rid = mem.create_request(req.session_id, session.get("raw_text", ""))
            mem.update_request(rid, session.get("extracted", {}), status="completed")
            mem.store_feedback(request_id=rid, decision="approved")
        except Exception as e:
            logger.warning("Failed to store approval: %s", e)

        return final
    except HTTPException:
        raise
    except Exception:
        logger.exception("Feedback processing error")
        raise HTTPException(500, "Internal Server Error during feedback processing")


@app.post("/retry")
async def retry_extraction(req: FeedbackRequest, api_key: str = Depends(get_api_key)):
    """Re-run extraction with rejection context."""
    session = _recover_session(req.session_id)
    retry_count = session.get("retry_count", 0) + 1
    if retry_count > 3:
        _pending_sessions.pop(req.session_id, None)
        return {"success": False, "status": "max_retries", "message": "Maximum retry attempts reached."}

    image_b64 = session.get("image_b64", "")
    if not image_b64:
        raise HTTPException(400, "No image data for retry")

    new_sid = str(uuid.uuid4())
    initial = {"session_id": new_sid, "image_b64": image_b64,
               "mime_type": session.get("mime_type", "image/jpeg"),
               "retry_count": retry_count, "supervisor_notes": []}
    config = {"configurable": {"thread_id": new_sid}}

    async def gen():
        start_t = time.time()
        yield f"data: {json.dumps({'status': 'progress', 'message': f'Retry {retry_count}/3 — re-analysing…'})}\n\n"
        try:
            final = initial
            async for ev in _pipeline.astream(initial, config=config, stream_mode="values"):
                final = ev
                notes = ev.get("supervisor_notes", [])
                yield f"data: {json.dumps({'status': 'progress', 'message': notes[-1] if notes else 'Pipeline step completed...'})}\n\n"
                if ev.get("__interrupt__") or "__interrupt__" in ev:
                    final["__interrupted__"] = True
                    break

            elapsed = time.time() - start_t
            extracted = final.get("extracted", {})
            validation = final.get("validation", {})
            _pending_sessions.pop(req.session_id, None)
            new_data = {"thread_id": new_sid, "extracted": extracted, "validation": validation,
                        "retry_count": retry_count, "image_b64": image_b64,
                        "mime_type": session.get("mime_type", "image/jpeg"),
                        "raw_text": final.get("raw_text", ""), "vision_text": final.get("vision_text", ""),
                        "created_at": time.time()}
            _pending_sessions[new_sid] = new_data
            try:
                from memory.sqlite_store import AgentMemory
                mem = AgentMemory()
                mem.delete_session(req.session_id)
                mem.save_session(new_sid, "ocr_pending", new_data)
            except Exception:
                pass

            yield f"data: {json.dumps({'success': True, 'status': 'pending_review', 'session_id': new_sid, 'message': f'Retry {retry_count}/3 — summary re-generated.', 'preview': extracted, 'validation': validation, 'processing_time_seconds': round(elapsed, 2)})}\n\n"
        except Exception as e:
            logger.exception("Retry pipeline error")
            yield f"data: {json.dumps({'success': False, 'status': 'error', 'error': str(e)})}\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"})


# ═══════════════════════════════════════════════════════════════════════
# ROUTES: HEALTH / STATS
# ═══════════════════════════════════════════════════════════════════════

@app.get("/health")
async def health():
    llm_ok = False
    try:
        from tools.llm_tool import check_llm_connection
        llm_ok = check_llm_connection()
    except Exception:
        pass
    ok = llm_ok and _pipeline is not None
    return JSONResponse(status_code=200 if ok else 503, content={
        "status": "healthy" if ok else "degraded", "environment": APP_ENV,
        "mode": "Groq Vision OCR v4", "pipeline_ready": _pipeline is not None,
        "groq_reachable": llm_ok, "pending_sessions": len(_pending_sessions),
    })


@app.get("/stats")
async def stats(api_key: str = Depends(get_api_key)):
    try:
        from memory.sqlite_store import AgentMemory
        mem = AgentMemory()
        return {"top_rejection_patterns": mem.get_top_rejection_patterns(limit=10),
                "recent_approvals_count": len(mem.get_recent_approvals(limit=50))}
    except Exception:
        logger.exception("Stats endpoint failed")
        raise HTTPException(500, "Failed to fetch statistics")


# ═══════════════════════════════════════════════════════════════════════
# ROUTES: PUSH NOTIFICATIONS
# ═══════════════════════════════════════════════════════════════════════

@app.post("/send-push")
async def send_push_notification(req: PushNotificationRequest, api_key: str = Depends(get_api_key)):
    from config.settings import FIREBASE_SERVICE_ACCOUNT
    if not FIREBASE_SERVICE_ACCOUNT:
        raise HTTPException(500, "Push not configured: missing FIREBASE_SERVICE_ACCOUNT")

    try:
        sa_info = json.loads(FIREBASE_SERVICE_ACCOUNT)
        project_id = sa_info.get("project_id")

        from google.oauth2 import service_account
        import google.auth.transport.requests as auth_requests
        creds = service_account.Credentials.from_service_account_info(
            sa_info, scopes=["https://www.googleapis.com/auth/firebase.messaging"])
        creds.refresh(auth_requests.Request())

        import requests
        results = []
        for msg in req.messages:
            res = requests.post(
                f"https://fcm.googleapis.com/v1/projects/{project_id}/messages:send",
                headers={"Authorization": f"Bearer {creds.token}", "Content-Type": "application/json"},
                json={"message": {
                    "token": msg.token,
                    "notification": {"title": msg.title, "body": msg.body},
                    "android": {"notification": {"icon": msg.icon, "color": msg.color,
                                                  "channel_id": "admin_push_channel", "notification_priority": "PRIORITY_HIGH"}},
                    "apns": {"payload": {"aps": {"sound": "default"}}},
                    "data": {"title": msg.title, "body": msg.body, "click_action": "FLUTTER_NOTIFICATION_CLICK"},
                }},
            )
            results.append({"token": msg.token, "ok": res.ok, "status": res.status_code})

        ok_count = sum(1 for r in results if r["ok"])
        return {"success": True, "sent": ok_count, "total": len(req.messages), "details": results}
    except Exception as e:
        logger.exception("FCM Proxy Error")
        raise HTTPException(500, f"Failed to send notifications: {e}")


# ═══════════════════════════════════════════════════════════════════════
# ROUTES: SMART PACKAGE SEARCH
# ═══════════════════════════════════════════════════════════════════════

@app.post("/smart-search", response_model=SmartSearchResponse)
async def smart_search(request: SmartSearchRequest):
    """AI-powered MAA Yojana package search with clinical reasoning."""
    import asyncio
    expanded_query, expanded_proc, expanded_dis = await asyncio.gather(
        _expand_abbreviations_llm(request.query),
        _expand_abbreviations_llm(request.procedure),
        _expand_abbreviations_llm(request.disease)
    )
    
    query_terms = _split_query_terms(expanded_query)
    if expanded_proc:
        _append_unique_term(query_terms, expanded_proc)
    if expanded_dis:
        _append_unique_term(query_terms, expanded_dis)
        
    query_terms, spelling_corrections = _correct_query_terms_spelling(query_terms)

    main_term = query_terms[0] if query_terms else ""
    addon_terms = query_terms[1:] if len(query_terms) > 1 else []

    for imp in _expand_implicit_addon_terms(main_term):
        _append_unique_term(addon_terms, imp)

    combined = ", ".join([main_term, *addon_terms]).strip(", ")
    if not combined:
        return SmartSearchResponse(doctor_reasoning="Please provide a procedure, disease, or query.", raw_packages=[])

    # Clinical pathway hints
    clinical_hint = ""
    try:
        pathway = get_clinical_pathway(main_term)
        if pathway:
            clinical_hint = f"\n\nCLINICAL PATHWAY HINT:\n{pathway.get('doctor_reasoning', '')}"
        for sp in (get_packages_for_symptom(main_term) or [])[:3]:
            clinical_hint += f"\n- {sp['code']}: {sp['name']} ({sp['reason']})"
    except Exception as e:
        logger.warning("Clinical pathway lookup failed: %s", e)

    limit = max(25, min(100, request.limit))
    relevant = _search_packages_basic(main_term, limit=limit, patient_type=request.patient_type if hasattr(request, 'patient_type') else "")
    relevant = _prioritize_exact_main_term_first(relevant, main_term)

    addon_pkgs: list[dict] = []
    addon_by_term: dict[str, list[dict]] = {}
    for at in addon_terms:
        res = _search_packages_basic(at, limit=30, patient_type=request.patient_type if hasattr(request, 'patient_type') else "")
        addon_by_term[at] = res
        addon_pkgs.extend(res)

    # Intent classification for multi-term queries
    # Removed blocking await _classify_input_intent(ts) as it was mostly ignored and caused delays
    # Validation is already handled by the Groq prompt later for overall combination rules.

    # Combine keeping main first
    all_relevant = relevant.copy()
    seen = {pkg_code(p) for p in relevant}
    for p in addon_pkgs:
        c = pkg_code(p)
        if c not in seen:
            all_relevant.append(p)
            seen.add(c)

    # For AI, we need to balance showing main packages vs add-on packages to ensure all search terms are considered
    ai_context_pkgs = relevant[:12]
    seen_ctx = {pkg_code(p) for p in ai_context_pkgs}
    for p in addon_pkgs:
        c = pkg_code(p)
        if c not in seen_ctx:
            ai_context_pkgs.append(p)
            seen_ctx.add(c)
        if len(ai_context_pkgs) >= 30:
            break

    if not relevant:
        hint = f" Did you mean: {', '.join(spelling_corrections.values())}?" if spelling_corrections else ""
        return SmartSearchResponse(
            doctor_reasoning=f"No packages found for '{main_term}'. Try different keywords.{hint}", raw_packages=[])

    # AI package selection via Groq
    try:
        if not _async_groq_client:
            raise ValueError("Groq client not initialised")

        ctx = _format_packages_for_ai(ai_context_pkgs, n=30)
        symptoms_str = ", ".join(request.symptoms) if request.symptoms else "None"
        addon_hint = f"\n- Add-on procedures: {', '.join(addon_terms)}" if addon_terms else ""

        user_prompt = f"""PATIENT CASE:
- Main Procedure: {main_term}
- Query: {request.query or 'N/A'}
- Procedure: {request.procedure or 'N/A'}
- Disease: {request.disease or 'N/A'}
- Symptoms: {symptoms_str}
- Age: {request.patient_age if request.patient_age > 0 else 'N/A'}
- Gender: {request.patient_gender or 'N/A'}{addon_hint}
{clinical_hint}

AVAILABLE PACKAGES:
{ctx}

Select BEST matching package(s). First term "{main_term}" = MAIN. Additional terms = ADD-ONS.
Return ONLY approved-likely packages."""

        resp = await _async_groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "system", "content": _get_ai_system_prompt(request.mode or "smart")},
                      {"role": "user", "content": user_prompt}],
            temperature=0.2, max_tokens=1500,
            response_format={"type": "json_object"},
        )
        ai = json.loads(resp.choices[0].message.content)

        # Build validated result
        selected_codes: set[str] = set()
        violations: list[str] = list(ai.get("blocked_rules", []))
        result = SmartSearchResponse(
            doctor_reasoning=ai.get("doctor_summary", ""),
            approval_likelihood=ai.get("approval_likelihood", ""),
        )

        pkg_lookup = {pkg_code(p): p for p in all_relevant}
        main_type = None
        standalone = False

        # Main package
        mc = ai.get("main_package_code")
        if mc and mc != "null" and mc in pkg_lookup:
            p = pkg_lookup[mc]
            main_type = _identify_package_type(p)
            standalone = main_type.get("is_standalone", False)
            selected_codes.add(mc)
            result.main_package = PackageResultModel(
                package_code=mc, package_name=pkg_name(p), rate=pkg_rate(p),
                speciality=pkg_specialty(p), category=pkg_category(p),
                is_main=True, medical_reason=ai.get("main_package_reason", ""),
            )

        # Fallback main
        if not result.main_package and relevant:
            fb = relevant[0]
            mc = pkg_code(fb)
            selected_codes.add(mc)
            main_type = _identify_package_type(fb)
            standalone = main_type.get("is_standalone", False)
            result.main_package = PackageResultModel(
                package_code=mc, package_name=pkg_name(fb), rate=pkg_rate(fb),
                speciality=pkg_specialty(fb), category=pkg_category(fb),
                is_main=True, medical_reason="Best clinical match from prioritised results",
            )

        if standalone:
            result.auto_implant = None
            result.suggested_addons = []
        else:
            # Implant
            ic = ai.get("implant_code")
            if ic and ic != "null":
                for p in _all_packages():
                    if pkg_code(p) == ic:
                        if main_type:
                            err = _validate_package_combination(main_type, _identify_package_type(p), ic)
                            if err:
                                violations.append(err)
                                break
                        selected_codes.add(ic)
                        result.auto_implant = PackageResultModel(
                            package_code=ic, package_name=pkg_name(p), rate=pkg_rate(p),
                            speciality=pkg_specialty(p), category=pkg_category(p),
                            is_implant=True, medical_reason="Rule 4: Auto-suggested implant",
                        )
                        break

            # AI add-ons
            for addon in ai.get("addons", [])[:5]:
                ac = addon.get("code")
                if not ac or not main_type:
                    continue
                for p in _all_packages():
                    if pkg_code(p) == ac:
                        at = _identify_package_type(p)
                        err = _validate_package_combination(main_type, at, ac)
                        if err:
                            violations.append(err)
                        else:
                            selected_codes.add(ac)
                            reason = addon.get("reason", "")
                            if at["is_extended_los"]: reason = f"Rule 5: Extended LOS. {reason}"
                            elif at["is_addon"]: reason = f"Rule 3: Compatible add-on. {reason}"
                            result.suggested_addons.append(PackageResultModel(
                                package_code=ac, package_name=pkg_name(p), rate=pkg_rate(p),
                                speciality=pkg_specialty(p), category=pkg_category(p),
                                is_addon=True, medical_reason=reason,
                            ))
                        break

            # Deterministic add-on fallback for user-requested terms
            for at_term in addon_terms:
                for p in addon_by_term.get(at_term, []):
                    ac = pkg_code(p)
                    if not ac or ac in selected_codes:
                        continue
                    err = _validate_package_combination(main_type, _identify_package_type(p), ac) if main_type else None
                    if err:
                        if _is_transfusion_term(at_term):
                            err = None
                        else:
                            violations.append(err)
                            continue
                    selected_codes.add(ac)
                    prefix = "Clinical add-on" if at_term.lower() in {"blood transfusion", "transfusion"} else "Requested add-on"
                    result.suggested_addons.append(PackageResultModel(
                        package_code=ac, package_name=pkg_name(p), rate=pkg_rate(p),
                        speciality=pkg_specialty(p), category=pkg_category(p),
                        is_addon=True, medical_reason=f"{prefix}: {at_term}",
                    ))
                    break

            # Hard transfusion fallback
            if any(_is_transfusion_term(t) for t in addon_terms) and not result.suggested_addons:
                for tt in addon_terms:
                    if not _is_transfusion_term(tt):
                        continue
                    cands = addon_by_term.get(tt, [])
                    if cands:
                        p = cands[0]
                        ac = pkg_code(p)
                        if ac:
                            selected_codes.add(ac)
                            result.suggested_addons.append(PackageResultModel(
                                package_code=ac, package_name=pkg_name(p), rate=pkg_rate(p),
                                speciality=pkg_specialty(p), category=pkg_category(p),
                                is_addon=True, medical_reason="Clinical add-on fallback: transfusion support",
                            ))
                            break

        # Build curated raw_packages
        ordered: list[str] = []
        if result.main_package: ordered.append(result.main_package.package_code)
        if result.auto_implant: ordered.append(result.auto_implant.package_code)
        ordered.extend(a.package_code for a in result.suggested_addons)

        code_to_pkg = {pkg_code(p): p for p in all_relevant}
        result.raw_packages = [_build_raw_package_row(code_to_pkg[c], c in selected_codes)
                               for c in ordered if c in code_to_pkg]

        result.blocked_rules = violations
        if violations and result.approval_likelihood not in ("REJECTED", "LOW"):
            result.approval_likelihood = "LOW"
            result.doctor_reasoning += "\n\n⚠️ RULE VIOLATIONS:\n" + "\n".join(f"• {e}" for e in violations[:5])

        if spelling_corrections:
            result.doctor_reasoning = (
                f"Spelling correction: {', '.join(f'{s}→{d}' for s, d in spelling_corrections.items())}\n\n"
                + result.doctor_reasoning
            )

        logger.info("Smart search '%s' → %d packages, %d violations", main_term, len(selected_codes), len(violations))
        return result

    except Exception as e:
        logger.warning("AI search failed, basic fallback: %s", e)
        return SmartSearchResponse(
            doctor_reasoning=f"AI unavailable. Showing {len(all_relevant)} matches for: {main_term}",
            raw_packages=[_build_raw_package_row(p) for p in all_relevant],
        )


# ═══════════════════════════════════════════════════════════════════════
# ROUTES: INTERACTIVE MULTI-STEP SEARCH
# ═══════════════════════════════════════════════════════════════════════

async def _get_or_reconstruct_flow(session_id: str) -> tuple[Any, list[dict], dict]:
    from memory.sqlite_store import AgentMemory
    from tools.smart_search_flow import reconstruct_flow_from_state

    if session_id in _interactive_flows:
        d = _interactive_flows[session_id]
        return d["flow"], d["all_packages"], d.get("per_term_packages", {})

    data = AgentMemory().get_session(session_id)
    if not data:
        raise HTTPException(404, "Session not found or expired. Please start a new search.")

    query = data.get("query", "")
    addon_terms = data.get("addon_terms", [])
    sels = data.get("selections_list", [])
    pt_type = data.get("patient_type", "")

    _load_packages_cache()
    all_pkgs = _packages_cache + _robotic_cache
    if pt_type:
        all_pkgs = [p for p in all_pkgs if _passes_patient_type(p, pt_type)]
        
    matching = _prioritize_exact_main_term_first(_search_packages_basic(query, 200, patient_type=pt_type), query)
    per_term: dict[str, list] = {query: matching}
    for t in addon_terms:
        per_term[t] = _prioritize_exact_main_term_first(_search_packages_basic(t, 200, patient_type=pt_type), t)

    flow = reconstruct_flow_from_state(query=query, addon_terms=addon_terms, selections=sels,
                                       matching_packages=matching, all_packages=all_pkgs, per_term_packages=per_term)
    _interactive_flows[session_id] = {
        "flow": flow, "packages": matching, "all_packages": all_pkgs,
        "per_term_packages": per_term, "created_at": time.time(), "selections_list": sels,
        "request": {"query": query, "patient_type": pt_type},
    }
    return flow, all_pkgs, per_term


def _sync_session_db(session_id: str, flow: Any, sels: list[dict], pt_type: str = ""):
    try:
        from memory.sqlite_store import AgentMemory
        AgentMemory().save_session(session_id, "interactive_search", {
            "query": flow.query, "addon_terms": [t for t in flow.parsed_terms if t != flow.query],
            "selections_list": sels, "flow_complete": flow.flow_complete,
            "patient_type": pt_type,
        })
    except Exception as e:
        logger.error("Failed to sync session: %s", e)


@app.post("/interactive-search/analyze-query", response_model=AnalyzeQueryResponse)
async def analyze_interactive_query(request: AnalyzeQueryRequest):
    """NLP analysis of free-text patient history → clinical keywords."""
    if not _async_groq_client:
        raise HTTPException(500, "Groq client not initialised")

    prompt = f"""You are an expert PMJAY/Ayushman Bharat medical AI.
Given an unstructured medical history, extract a comprehensive set of clinical keywords to find exact medical packages.

1. Concise professional summary (1-2 sentences)
2. Extract ALL highly relevant clinical keywords into a single list, ensuring you capture:
   - Primary Diagnosis / Main Disease
   - Likely Surgical Procedures or Medical Treatments required (e.g., Skin Grafting, Amputation, Chemotherapy)
   - Important Comorbidities or Complications (e.g., Sepsis, Malnutrition, Diabetes Mellitus)
   - Necessary Supportive Care / Add-ons (e.g., ICU Care, Blood Transfusion, Mechanical Ventilation)
3. IMPORTANT RULES:
   - PREDICT PROCEDURES: If the clinical history strongly implies a specific procedure or treatment (e.g., "limb ischemia + deep burns" -> Amputation/Debridement; "oral cancer + weakness" -> Chemotherapy/Radiotherapy/Supportive Care), include the procedure as a keyword!
   - TRANSLATE colloquial/simple layman terms to EXACT medical equivalents (e.g. kidney → renal/nephrology, heart → cardiac, stomach → gastric). CRITICAL for PMJAY accuracy.
   - EXPAND abbreviations (e.g. CABG → Coronary Artery Bypass Grafting, TKR → Total Knee Replacement).
   - DEDUPLICATE: No synonyms, no acronym+full-name pairs.
   - Do NOT artificially limit yourself to 1-3 keywords; provide as many as are clinically significant (typically 3-8 keywords).

Input: "{request.query}"

Return ONLY JSON: {{"summary": "...", "keywords": ["..."]}}"""

    try:
        resp = await _async_groq_client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0, response_format={"type": "json_object"},
        )
        parsed = json.loads(resp.choices[0].message.content)
        return AnalyzeQueryResponse(
            summary=parsed.get("summary", "No summary."),
            keywords=parsed.get("keywords") or [request.query[:50]],
        )
    except Exception as e:
        logger.error("Query analysis failed: %s", e)
        return AnalyzeQueryResponse(summary=f"Analysis failed. Original: {request.query}", keywords=[request.query])


@app.post("/interactive-search/start", response_model=InteractiveSearchStartResponse)
async def start_interactive_search(request: InteractiveSearchStartRequest):
    """Start multi-step interactive package selection flow."""
    try:
        from tools.smart_search_flow import build_search_flow, _split_query_terms as flow_split, advance_past_empty_optional_steps
        import asyncio

        expanded_query, expanded_proc, expanded_dis = await asyncio.gather(
            _expand_abbreviations_llm(request.query),
            _expand_abbreviations_llm(request.procedure),
            _expand_abbreviations_llm(request.disease)
        )

        terms = flow_split(expanded_query)
        if expanded_proc:
            terms.insert(0, expanded_proc)
        if expanded_dis and expanded_dis not in terms:
            terms.append(expanded_dis)
        terms, corrections = _correct_query_terms_spelling(terms)
        if not terms:
            raise HTTPException(400, "Please provide a query, procedure, or disease")

        main_term = terms[0]
        addon_terms = terms[1:]
        pt_type = request.patient_type

        matching = _prioritize_exact_main_term_first(_search_packages_basic(main_term, 200, patient_type=pt_type), main_term)
        if request.disease:
            seen = {pkg_code(p) for p in matching}
            for p in _search_packages_basic(request.disease, 120, patient_type=pt_type):
                if pkg_code(p) not in seen:
                    matching.append(p)

        if not matching:
            _load_packages_cache()
            all_pkgs = _packages_cache + _robotic_cache
            if not all_pkgs:
                raise HTTPException(503, "Package datasets not loaded.")
            specs = get_specialties_for_term(main_term)
            if specs:
                seen_c: set[str] = set()
                for p in all_pkgs:
                    c = pkg_code(p)
                    if c and c not in seen_c and any(s.lower() in pkg_specialty(p).lower() for s in specs):
                        if _passes_patient_type(p, pt_type):
                            matching.append(p)
                            seen_c.add(c)
                matching = matching[:250]

        if not matching:
            raise HTTPException(404, f"No packages found for: {main_term}")

        _load_packages_cache()
        all_pkgs = _packages_cache + _robotic_cache
        # ── Apply patient-type filter to the entire pool ──
        if pt_type:
            all_pkgs = [p for p in all_pkgs if _passes_patient_type(p, pt_type)]

        per_term: dict[str, list] = {main_term: matching}
        for t in addon_terms:
            tp = _prioritize_exact_main_term_first(_search_packages_basic(t, 200, patient_type=pt_type), t)
            if not tp:
                tl = t.lower()
                toks = [tok for tok in _tokenize(tl) if len(tok) > 2]
                if toks:
                    tp = [p for p in all_pkgs if any(tok in f"{pkg_name(p).lower()} {pkg_specialty(p).lower()}" for tok in toks)][:200]
            per_term[t] = tp

        # Intent check
        violation_msg = None
        if len(terms) > 1:
            ts = {}
            for t in terms:
                pkgs = per_term.get(t, matching[:15] if t == main_term else [])
                ts[t] = list({pkg_specialty(p).strip() for p in pkgs[:15] if pkg_specialty(p).strip()})[:3]
            violation_msg = _check_intent_rule_violation(await _classify_input_intent(ts))

        flow = build_search_flow(main_term, addon_terms, matching,
                                 all_packages_for_addons=all_pkgs, per_term_packages=per_term)
        advance_past_empty_optional_steps(flow)
        _auto_advance_single_option_steps(flow, all_pkgs)

        sid = str(uuid.uuid4())
        _interactive_flows[sid] = {
            "flow": flow, "packages": matching, "all_packages": all_pkgs,
            "per_term_packages": per_term, "created_at": time.time(), "selections_list": [],
            "request": {"query": request.query, "procedure": request.procedure,
                         "disease": request.disease, "symptoms": request.symptoms,
                         "patient_age": request.patient_age, "patient_gender": request.patient_gender,
                         "patient_type": pt_type},
        }
        _sync_session_db(sid, flow, [], pt_type=pt_type)

        first = flow.steps[flow.current_step] if flow.steps and flow.current_step < len(flow.steps) else None
        if not first:
            raise HTTPException(500, "Failed to build search flow")

        msg_parts = []
        if violation_msg: msg_parts.append(violation_msg)
        if corrections: msg_parts.append(f"Did you mean: {', '.join(corrections.values())}?")
        msg_parts.append(f"Starting search for: {main_term}. Found {len(matching)} packages.")

        return InteractiveSearchStartResponse(
            session_id=sid, query=request.query, parsed_terms=terms,
            current_step=_step_to_response(first), message=" | ".join(msg_parts),
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error("start_interactive_search crashed: %s", e, exc_info=True)
        raise HTTPException(500, f"Internal error: {str(e)}")


@app.get("/interactive-search/{session_id}/step")
async def get_current_step(session_id: str):
    flow, pkgs, _ = await _get_or_reconstruct_flow(session_id)
    from tools.smart_search_flow import advance_past_empty_optional_steps
    advance_past_empty_optional_steps(flow)
    _auto_advance_single_option_steps(flow, pkgs)

    if flow.flow_complete:
        return {"status": "complete", "final_recommendation": flow.final_recommendation, "selections": flow.selections}

    step = flow.steps[flow.current_step] if flow.current_step < len(flow.steps) else None
    if not step:
        raise HTTPException(500, "Invalid flow state")
    return _step_to_response(step).dict()


@app.post("/interactive-search/{session_id}/select")
async def submit_step_selection(session_id: str, selection: SelectionRequest):
    flow, pkgs, _ = await _get_or_reconstruct_flow(session_id)
    from tools.smart_search_flow import process_step_selection

    ok, err = process_step_selection(flow, {"id": selection.option_id, "notes": selection.notes,
                                             "manual_package": selection.manual_package}, pkgs)
    if not ok:
        return SelectionResponse(success=False, message=f"Error: {err}")

    sels = _interactive_flows[session_id].get("selections_list", [])
    sels.append({"id": selection.option_id, "notes": selection.notes, "manual_package": selection.manual_package})
    _interactive_flows[session_id]["selections_list"] = sels
    _pt = _interactive_flows.get(session_id, {}).get("request", {}).get("patient_type", "")
    _sync_session_db(session_id, flow, sels, pt_type=_pt)
    _auto_advance_single_option_steps(flow, pkgs)

    if flow.flow_complete:
        final = await _build_final_recommendation(flow, pkgs)
        flow.final_recommendation = final
        return SelectionResponse(success=True, message="Search complete!", flow_complete=True, final_recommendation=final)

    step = flow.steps[flow.current_step] if flow.current_step < len(flow.steps) else None
    if not step:
        return SelectionResponse(success=True, message="Flow completed.", flow_complete=True)
    return SelectionResponse(success=True, message="Selection received.", next_step=_step_to_response(step))


@app.post("/interactive-search/{session_id}/undo")
async def undo_step_selection(session_id: str):
    flow, _, _ = await _get_or_reconstruct_flow(session_id)
    from tools.smart_search_flow import undo_last_selection

    ok, msg = undo_last_selection(flow)
    if ok:
        sels = _interactive_flows[session_id].get("selections_list", [])
        if sels: sels.pop()
        _interactive_flows[session_id]["selections_list"] = sels
        _pt = _interactive_flows.get(session_id, {}).get("request", {}).get("patient_type", "")
        _sync_session_db(session_id, flow, sels, pt_type=_pt)

    if not ok:
        return JSONResponse(status_code=400, content={"success": False, "message": msg})

    step = flow.steps[flow.current_step] if flow.current_step < len(flow.steps) else None
    if not step:
        return JSONResponse(status_code=500, content={"success": False, "message": "No current step after undo."})
    return {"success": True, "message": msg, "current_step": _step_to_response(step).dict()}


@app.get("/interactive-search/{session_id}/status")
async def get_flow_status(session_id: str):
    if session_id not in _interactive_flows:
        raise HTTPException(404, "Session not found")
    flow = _interactive_flows[session_id]["flow"]
    return FlowStatusResponse(
        session_id=session_id, query=flow.query, current_step_number=flow.current_step,
        total_steps=len(flow.steps), selections_made=flow.selections,
        violations=flow.violations, flow_complete=flow.flow_complete,
    )


# ═══════════════════════════════════════════════════════════════════════
# FINAL RECOMMENDATION BUILDER
# ═══════════════════════════════════════════════════════════════════════

async def _build_final_recommendation(flow: Any, packages: list[dict]) -> dict:
    from tools.smart_search_flow import validate_package_combination

    result: dict[str, Any] = {
        "main_package": None, "selected_packages": [], "implant_package": None,
        "implant_packages": [], "stratification_package": None, "stratification_packages": [],
        "addon_packages": [], "term_groups": [], "blocked_rules": [],
        "approval_likelihood": "MEDIUM", "doctor_reasoning": "Packages selected through interactive flow.",
    }

    code_to_pkg = {pkg_code(p): p for p in packages}

    def _entry(p: dict) -> dict:
        return {
            "code": pkg_code(p), "name": pkg_name(p)[:100], "rate": pkg_rate(p),
            "specialty": pkg_specialty(p), "package_category": pkg_category(p),
            "pre_auth_document": p.get("PRE AUTH DOCUMENT", p.get("Mandatory Documents", "")),
            "claim_document": p.get("CLAIM DOCUMENT", p.get("Mandatory Documents - Claim Processing", "")),
        }

    groups: dict = {}
    current_term = ""

    for idx, step in enumerate(flow.steps):
        sel = flow.selections.get(f"step_{idx}")
        if not isinstance(sel, dict):
            continue
        sel_id = str(sel.get("id", ""))
        sel_code = str(sel.get("code", "")).strip()
        step_term = (step.context.get("intent_term", "") or "").strip()
        if step_term:
            current_term = step_term
        if not sel_code or "skip" in sel_id.lower():
            continue

        pkg = code_to_pkg.get(sel_code)
        if not pkg:
            continue
        e = _entry(pkg)
        rate = pkg_rate(pkg)
        tk = current_term or "main"
        if tk not in groups:
            groups[tk] = {"term": tk, "main_package": None, "implant_packages": [],
                          "stratification_packages": [], "subtotal": 0.0}
        g = groups[tk]

        if sel_id.startswith("package_"):
            g["main_package"] = e; g["subtotal"] += rate
            result["selected_packages"].append(e)
            if not result["main_package"]: result["main_package"] = e
        elif sel_id.startswith("implant_") and sel_code != "NO_IMPLANT":
            g["implant_packages"].append(e); g["subtotal"] += rate
            result["implant_packages"].append(e)
            if not result["implant_package"]: result["implant_package"] = e
        elif sel_id.startswith("strat_"):
            g["stratification_packages"].append(e); g["subtotal"] += rate
            result["stratification_packages"].append(e)
            if not result["stratification_package"]: result["stratification_package"] = e
        elif sel_id.startswith("addon_"):
            result["addon_packages"].append({**e, "reason": sel.get("reason", "")})

    result["term_groups"] = list(groups.values())

    # Validate combinations
    mp = result.get("main_package")
    if isinstance(mp, dict):
        main_pkg = code_to_pkg.get(mp.get("code", ""))
        if main_pkg:
            combo_pkgs = [code_to_pkg[sp["code"]] for sp in result["selected_packages"][1:] if sp["code"] in code_to_pkg]
            strat = code_to_pkg.get((result.get("stratification_package") or {}).get("code", ""))
            imp = code_to_pkg.get((result.get("implant_package") or {}).get("code", ""))
            combo_pkgs.extend(code_to_pkg[a["code"]] for a in result["addon_packages"] if a["code"] in code_to_pkg)
            valid, viols = validate_package_combination(main_pkg, imp, strat, combo_pkgs)
            result["blocked_rules"] = viols
            if not valid:
                result["approval_likelihood"] = "LOW"

    return result


# ═══════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    uvicorn.run("main:app", host=SERVER_HOST, port=SERVER_PORT,
                workers=max(1, SERVER_WORKERS), reload=False)