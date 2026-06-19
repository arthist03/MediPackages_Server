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
    DEEPSEEK_API_KEY, DEEPSEEK_MODEL,
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
_async_deepseek_client = None
_pending_sessions: dict[str, dict] = {}
_interactive_flows: dict[str, Any] = {}
_packages_cache: list[dict] = []
_robotic_cache: list[dict] = []
_pmjay_cache: list[dict] = []
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
    "cabg":             ["coronary artery bypass", "bypass", "grafting", "cabg", "cardiac", "cardiothoracic"],
    "coronary artery bypass grafting": ["cabg", "bypass", "grafting", "coronary", "cardiothoracic"],
    "breathlessness":   ["heart failure", "cardiac", "pulmonary", "respiratory", "chf"],
    "rheumatic fever":  ["rheumatic fever", "valvular heart", "acute rheumatic"],
    "acute rheumatic fever": ["rheumatic fever", "valvular heart", "acute rheumatic"],
    # Gastrointestinal
    "stomach pain":     ["appendix", "gallbladder", "cholecyst", "pancreat", "intestin"],
    "abdominal pain":   ["appendix", "gallbladder", "cholecyst", "pancreat", "intestin", "hernia"],
    "appendix":         ["appendicitis", "appendicectomy", "appendicular"],
    "appendicitis":     ["appendix", "appendicectomy", "appendicular"],
    "cholecystectomy":  ["gallbladder", "cholecystitis", "laparoscopic"],
    "colostomy":        ["colostomy", "stoma", "intestin", "colon"],
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
    # Direct disease/condition terms
    "sepsis":           ["sepsis", "septicemia", "septic", "infection", "icu"],
    "anemia":           ["anemia", "anaemia", "transfusion", "blood"],
    "anaemia":          ["anemia", "anaemia", "transfusion", "blood"],
}

# High-precision phrase → keyword boosters (exact match scoring)
PHRASE_PRIORITY_KEYWORDS: dict[str, list[str]] = {
    "angioplasty":      ["ptca", "coronary angioplasty", "coronary", "angioplasty", "pci"],
    "ptca":             ["ptca"],
    "cabg":             ["coronary artery bypass", "cabg", "bypass grafting"],
    "coronary artery bypass grafting": ["coronary artery bypass", "cabg", "bypass grafting"],
    "colostomy":        ["colostomy"],
    "sepsis":           ["sepsis", "septicemia"],
    "anemia":           ["anemia", "anaemia"],
    "anaemia":          ["anemia", "anaemia"],
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
    "cabg":            "coronary artery bypass grafting",
}


# ═══════════════════════════════════════════════════════════════════════
# PACKAGE FIELD HELPERS  (DRY access to heterogeneous JSON keys)
# ═══════════════════════════════════════════════════════════════════════

def pkg_name(pkg: dict) -> str:
    return str(pkg.get("PACKAGE NAME") or pkg.get("Package Name") or pkg.get("package_name") or "")


def pkg_code(pkg: dict) -> str:
    return str(pkg.get("PACKAGE CODE") or pkg.get("package_code") or "")


def pkg_rate(pkg: dict) -> float:
    raw = pkg.get("RATE") or pkg.get("Rate") or pkg.get("package_amount") or 0
    try:
        return float(str(raw).replace(",", "").strip()) if raw else 0.0
    except Exception:
        return 0.0


def pkg_specialty(pkg: dict) -> str:
    return str(pkg.get("SPECIALITY") or pkg.get("Speciality") or pkg.get("speciality") or "")


def pkg_category(pkg: dict) -> str:
    return str(pkg.get("PACKAGE CATEGORY") or pkg.get("PACKAGE TYPE") or pkg.get("Procedure Type") or pkg.get("procedure_type") or "")


def pkg_implant_field(pkg: dict) -> str:
    return str(pkg.get("IMPLANT PACKAGE", pkg.get("IMPLANT", "NO IMPLANT")) or "NO IMPLANT")


def pkg_strat(pkg: dict) -> str:
    return str(pkg.get("STRATIFICATION PACKAGE", ""))


def parse_embedded_package(parent_pkg: dict, ref_str: str) -> dict | None:
    ref_str = ref_str.strip()
    if not ref_str:
        return None
    
    parts = [p.strip() for p in ref_str.split('|')]
    if not parts:
        return None
        
    first_part = parts[0]
    parent_code = pkg_code(parent_pkg).strip()
    pattern = rf"({re.escape(parent_code)})\s*-\s*(STR\d+|IMP\d+|[A-Z0-9]+)"
    match = re.search(pattern, first_part, re.IGNORECASE)
    
    if match:
        extracted_code = f"{match.group(1)}-{match.group(2)}".replace(" ", "").upper()
    else:
        match_any = re.search(r"([A-Z0-9\-]+(?:STR|IMP)\d+)", first_part, re.IGNORECASE)
        if match_any:
            extracted_code = match_any.group(1).upper()
        else:
            extracted_code = first_part.split('|')[0].strip().split()[0]
            
    extracted_code = extracted_code.strip("- ").replace(" ", "")
    
    clean_first = first_part
    if match:
        clean_first = first_part[match.end():].strip("- ")
    elif extracted_code in first_part:
        clean_first = first_part.replace(extracted_code, "").strip("- ")
        
    names = []
    if clean_first:
        names.append(clean_first)
    if len(parts) > 1 and parts[1]:
        names.append(parts[1])
        
    name = " - ".join(names) if names else first_part
    
    rate = 0.0
    rate_match = re.search(r"\(RATE\s*:\s*([\d\.]+)\)", ref_str, re.IGNORECASE)
    if rate_match:
        rate = float(rate_match.group(1))
    else:
        rate_match2 = re.search(r"RATE\s*:\s*([\d\.]+)", ref_str, re.IGNORECASE)
        if rate_match2:
            rate = float(rate_match2.group(1))
            
    category = "Regular"
    for part in parts:
        if part.startswith('[') and part.endswith(']'):
            cat_name = part[1:-1].strip()
            if "ADD" in cat_name.upper():
                category = "Add On"
            elif "REGULAR" in cat_name.upper():
                category = "Regular"
            elif "IMPLANT" in cat_name.upper():
                category = "Implant"
            elif "STRATIFICATION" in cat_name.upper():
                category = "Stratification"
                
    if "STR" in extracted_code:
        category = "Stratification"
    elif "IMP" in extracted_code:
        category = "Implant"
        
    return {
        "PACKAGE CODE": extracted_code,
        "PACKAGE NAME": name,
        "RATE": rate,
        "PACKAGE CATEGORY": category,
        "SPECIALITY": parent_pkg.get("SPECIALITY", ""),
        "PRE AUTH DOCUMENT": parent_pkg.get("PRE AUTH DOCUMENT", parent_pkg.get("Mandatory Documents", "")),
        "CLAIM DOCUMENT": parent_pkg.get("CLAIM DOCUMENT", parent_pkg.get("Mandatory Documents - Claim Processing", "")),
        "GOVT RESERVE": parent_pkg.get("GOVT RESERVE", "NO"),
        "IMPLANT": "NO",
        "STRATIFICATION PACKAGE": "NO",
        "_source": parent_pkg.get("_source", "maa"),
        "parent_code": parent_code
    }


def _get_package_by_code(code: str, all_packages: list[dict]) -> dict:
    if not code:
        return {}
    code = code.strip().upper()
    
    for p in all_packages:
        if pkg_code(p).strip().upper() == code:
            return p
            
    parent_code = ""
    if "-STR" in code:
        parent_code = code.split("-STR")[0].strip()
    elif "-IMP" in code:
        parent_code = code.split("-IMP")[0].strip()
        
    if parent_code:
        parent_pkg = None
        for p in all_packages:
            if pkg_code(p).strip().upper() == parent_code:
                parent_pkg = p
                break
                
        if parent_pkg:
            strat_field = pkg_strat(parent_pkg)
            implant_field = pkg_implant_field(parent_pkg)
            
            sub_strs = []
            if strat_field and strat_field.upper() not in ["NO", "NO STRATIFICATION", "N"]:
                sub_strs.extend(strat_field.split(';'))
            if implant_field and implant_field.upper() not in ["NO", "NO IMPLANT", "N"]:
                sub_strs.extend(implant_field.split(';'))
                
            for sub_str in sub_strs:
                sub_str = sub_str.replace('\r', ' ').replace('\n', ' ').strip()
                if not sub_str:
                    continue
                parsed = parse_embedded_package(parent_pkg, sub_str)
                if parsed and parsed.get("PACKAGE CODE") == code:
                    return parsed

    return {}



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


CLINICAL_ABBREVIATIONS = {
    "tkr": "Total Knee Replacement",
    "thr": "Total Hip Replacement",
    "ptca": "Percutaneous Transluminal Coronary Angioplasty",
    "pci": "Percutaneous Coronary Intervention",
    "cabg": "Coronary Artery Bypass Grafting",
    "lap chole": "Laparoscopic Cholecystectomy",
    "lap": "Laparoscopic",
    "icu": "Intensive Care Unit",
    "los": "Length of Stay",
    "dvt": "Deep Vein Thrombosis",
    "lmwh": "Low Molecular Weight Heparin",
    "mi": "Myocardial Infarction",
    "stemi": "ST-Elevation Myocardial Infarction",
    "nstemi": "Non-ST-Elevation Myocardial Infarction",
    "pcnl": "Percutaneous Nephrolithotomy",
    "sdp": "Single Donor Platelet",
    "ffp": "Fresh Frozen Plasma",
    "turp": "Transurethral Resection of the Prostate",
    "vvf": "Vesicovaginal Fistula",
    "avd": "Atrioventricular Block",
}


async def _expand_abbreviations_llm(text: str) -> str:
    """Uses local clinical maps or LLM to expand medical abbreviations in Indian clinical context."""
    if not text:
        return text

    # Try local abbreviation expansions first (instant, saves tokens, avoids rate limits)
    t = text.lower()
    has_local_match = False
    for ab, full in CLINICAL_ABBREVIATIONS.items():
        pattern = r'\b' + re.escape(ab) + r'\b'
        if re.search(pattern, t, flags=re.IGNORECASE):
            text = re.sub(pattern, full, text, flags=re.IGNORECASE)
            has_local_match = True

    if has_local_match:
        return text

    if len(text.strip()) > 100 or not _async_groq_client:
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
            timeout=2.0,
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
    for pkg in _all_packages_cache:
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

    synonyms_map = {
        "appendectomy": "appendicectomy",
        "laproscpic": "lap",
        "laproscopic": "lap",
        "laparoscopy": "lap",
        "laparoscopic": "lap",
        "cholesistectomy": "cholecystectomy",
        "cholecystectemy": "cholecystectomy",
        "cholecistectomy": "cholecystectomy",
        "hernea": "hernia",
        "harnial": "hernia",
        "hysterectemy": "hysterectomy",
        "thyroidectemy": "thyroidectomy",
        "mastectemy": "mastectomy",
    }

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
            if tok in synonyms_map:
                new_tokens.append(synonyms_map[tok])
                changed = True
                continue
            if len(tok) < 4 or tok.isdigit() or tok in vocab:
                new_tokens.append(tok)
                continue
            filtered_vocab = [w for w in _spelling_vocab_list_cache if w.startswith(tok[0])]
            candidates = difflib.get_close_matches(tok, filtered_vocab, n=3, cutoff=0.84)
            replacement = tok
            for c in candidates:
                if abs(len(c) - len(tok)) <= 2:
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
    global _packages_cache, _robotic_cache, _pmjay_cache, _all_packages_cache, _search_index
    if _all_packages_cache:
        return

    from config.settings import BASE_DIR

    def _load_json(filename: str, source_label: str) -> list[dict]:
        for candidate in [
            BASE_DIR.parent / "assets" / filename,
            BASE_DIR / "assets" / filename,
            Path.cwd() / "assets" / filename,
        ]:
            try:
                if candidate.exists():
                    with open(candidate, "r", encoding="utf-8-sig") as f:
                        data = json.load(f)
                    for pkg in data:
                        pkg["_source"] = source_label
                    logger.info("Loaded %d entries from %s", len(data), candidate)
                    return data
            except Exception as exc:
                logger.warning("Failed loading %s from %s: %s", filename, candidate, exc)
        logger.warning("Could not locate %s", filename)
        return []

    _packages_cache = _load_json("maa_packages.json", "maa")
    _robotic_cache = _load_json("maa_robotic_surgeries.json", "maa")
    _pmjay_cache = _load_json("PMJAY_flattened.json", "pmjay")
    _all_packages_cache = _packages_cache + _robotic_cache + _pmjay_cache
    logger.info("Package cache: %d standard + %d robotic + %d PMJAY", len(_packages_cache), len(_robotic_cache), len(_pmjay_cache))

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

    is_implant = ("IMPLANT" in implant_upper and "NO IMPLANT" not in implant_upper) or cat_upper == "IMP" or "IMPLANT" in name_upper
    is_addon = any(s in name_upper or s in cat_upper for s in ("[ADD-ON", "[ADD ON", "ADDON", "ADD-ON", "ADD ON"))
    is_extended_los = "EXTENDED LOS" in name_upper
    is_standalone = any(s in name_upper or s in cat_upper for s in ("STAND-ALONE", "STAND ALONE"))
    
    is_surgical = rate > 0 and not is_addon and not is_implant and not is_extended_los
    is_medical_management = rate == 0 and not is_addon and not is_implant

    return {
        "is_surgical":           is_surgical,
        "is_medical_management": is_medical_management,
        "is_standalone":         is_standalone,
        "is_addon":              is_addon,
        "is_implant":            is_implant,
        "is_extended_los":       is_extended_los,
        "source":                pkg.get("_source", "maa"),
    }


def _validate_package_combination(main_type: dict[str, bool], candidate_type: dict[str, bool], candidate_code: str) -> str | None:
    """Return violation message or None if combination is valid."""
    main_src = main_type.get("source", "maa")
    cand_src = candidate_type.get("source", "maa")
    if main_src != cand_src:
        src_map = {"maa": "MAA Yojana", "pmjay": "PMJAY"}
        return f"Rule 6 VIOLATION: Cannot combine {src_map.get(main_src, main_src)} package with {src_map.get(cand_src, cand_src)} package ({candidate_code})"
    
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
    name = pkg_name(pkg).lower()
    spec = pkg_specialty(pkg).lower()
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

        # Laparoscopic vs Open logic
        has_lap_in_query = "lap" in filtered_terms or "laparoscopic" in query_lower or "lap" in query_lower
        is_lap_package = "lap" in name_tok or "laparoscopic" in name.lower() or "lap." in name.lower() or "lap-" in name.lower()
        is_open_package = "open" in name_tok or "open" in name.lower()
        
        if has_lap_in_query:
            if is_lap_package:
                score += 25
            if is_open_package:
                score -= 15
        else:
            # Open is preferred if lap is not mentioned
            if is_lap_package:
                score -= 25
            if is_open_package:
                score += 15

        # Coronary PTCA vs CABG logic
        has_ptca_in_query = "ptca" in query_lower or "angioplasty" in query_lower
        has_cabg_in_query = "cabg" in query_lower or "bypass" in query_lower
        is_ptca_package = "ptca" in name.lower() or "ptca" in code.lower()
        is_cabg_package = "cabg" in name.lower() or "bypass" in name.lower()
        
        if has_ptca_in_query:
            if is_ptca_package:
                score += 40
            if is_cabg_package:
                score -= 60
        
        if has_cabg_in_query:
            if is_cabg_package:
                score += 40
        # Surgical vs Medical logic based on query indicators
        has_surgical_in_query = any(tok in query_lower for tok in {
            "replacement", "surgery", "surgical", "operative", "repair", "ectomy", "plasty", "otomy", 
            "resection", "ptca", "cabg", "fixation", "orif", "cholecystectomy", "appendicectomy", "hysterectomy"
        })
        has_medical_in_query = any(tok in query_lower for tok in {
            "anemia", "anaemia", "thalassemia", "sepsis", "septicemia", "fever", "diarrhoea", "diarrhea", "medical"
        })
        
        pkg_types = _identify_package_type(pkg)
        is_surgical_package = pkg_types["is_surgical"]
        is_medical_package = pkg_types["is_medical_management"]
        
        if has_surgical_in_query and not has_medical_in_query:
            if is_surgical_package:
                score += 150
            if is_medical_package:
                score -= 150
        elif has_medical_in_query and not has_surgical_in_query:
            if is_medical_package:
                score += 150
            if is_surgical_package:
                score -= 150

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

def _clean_query_terms(terms: list[str]) -> list[str]:
    import re
    cleaned = []
    patterns_to_remove = [
        r'\b\d+\s*(?:yr|year|years|yo|yoa|m|months|d|days)\b',
        r'\b(?:old|age)\b',
        r'\b(?:female|male|boy|girl|man|woman|child|infant|neonate|baby)\b',
        r'\b(?:patient|pt)\b',
        r'\b(?:icu|ward|stay|days|hospital|private|public|nabh|non-nabh|room|rent|care\s*unit|underwent|followed\s*by|for|with)\b',
    ]
    for term in terms:
        t_cleaned = term.lower()
        for p in patterns_to_remove:
            t_cleaned = re.sub(p, ' ', t_cleaned, flags=re.IGNORECASE)
        t_cleaned = re.sub(r'\s+', ' ', t_cleaned).strip()
        # Keep if there's at least one word > 2 characters
        if any(len(w) > 2 for w in re.findall(r'[a-zA-Z0-9]+', t_cleaned)):
            cleaned.append(t_cleaned)
    return cleaned


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
            "speciality": pkg_specialty(pkg), "ai_selected": ai_selected, "source": pkg.get("_source", "maa")}


# ═══════════════════════════════════════════════════════════════════════
# SESSION MANAGEMENT
# ═══════════════════════════════════════════════════════════════════════

def _prune_stale_sessions() -> None:
    now = time.time()
    stale_pending = [sid for sid, d in _pending_sessions.items()
             if now - d.get("created_at", now) > SESSION_TTL_SECONDS]
    for sid in stale_pending:
        _pending_sessions.pop(sid, None)
        
    stale_flows = [sid for sid, d in _interactive_flows.items()
             if now - d.get("created_at", now) > SESSION_TTL_SECONDS]
    for sid in stale_flows:
        _interactive_flows.pop(sid, None)

    if stale_pending or stale_flows:
        logger.info("Pruned %d stale pending session(s), %d stale interactive flow(s)", len(stale_pending), len(stale_flows))


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
        _groq_client = Groq(api_key=GROQ_API_KEY, max_retries=0)
        _async_groq_client = AsyncGroq(api_key=GROQ_API_KEY, max_retries=0)
        logger.info("✅ Groq API keys configured — clients initialised (max_retries=0)")

    if not FIREBASE_SERVICE_ACCOUNT:
        logger.warning("⚠️ FIREBASE_SERVICE_ACCOUNT not set – push notifications disabled")
    else:
        logger.info("✅ Firebase Service Account configured")
        
    from config.settings import DEEPSEEK_API_KEY
    if DEEPSEEK_API_KEY:
        try:
            from openai import AsyncOpenAI
            global _async_deepseek_client
            _async_deepseek_client = AsyncOpenAI(
                api_key=DEEPSEEK_API_KEY,
                base_url="https://api.deepseek.com",
                max_retries=0
            )
            logger.info("✅ DeepSeek API configured")
        except ImportError:
            logger.warning("⚠️ openai package not installed, DeepSeek disabled. run `pip install openai`")

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

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=CORS_ALLOW_CREDENTIALS,
    allow_methods=["*"],
    allow_headers=["*"],
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
    patient_type: str = ""
    scheme: str = ""
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
    usage: Optional[dict] = None


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
    scheme: str = ""  # 'maa', 'pmjay', or '' for all


class InteractiveSearchStartResponse(BaseModel):
    session_id: str
    query: str
    parsed_terms: list[str]
    current_step: Optional[SearchStepResponse] = None
    message: str
    status: Optional[str] = "running"
    final_recommendation: Optional[dict] = None


class SelectionRequest(BaseModel):
    option_id: str
    notes: Optional[str] = None
    manual_package: Optional[dict] = None

class RecalculateRequest(BaseModel):
    session_id: str
    package_codes: list[str]
    custom_rates: Optional[dict[str, float]] = None
    package_types: Optional[dict[str, str]] = None


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
    
    extracted_history_summary = ""
    spelling_corrections = {}
    
    # NLP History Extraction (Triggered if query is a long sentence/paragraph)
    if len((request.query or "").split()) > 3:
        hist_res = await _extract_keywords_from_history(request.query)
        query_terms = hist_res.get("keywords", [])
        if not hasattr(request, 'patient_type') or not request.patient_type:
            request.patient_type = hist_res.get("patient_type", "")
        extracted_history_summary = hist_res.get("summary", "")
        
        expanded_proc, expanded_dis = await asyncio.gather(
            _expand_abbreviations_llm(request.procedure),
            _expand_abbreviations_llm(request.disease)
        )
    else:
        expanded_query, expanded_proc, expanded_dis = await asyncio.gather(
            _expand_abbreviations_llm(request.query),
            _expand_abbreviations_llm(request.procedure),
            _expand_abbreviations_llm(request.disease)
        )
        query_terms = _split_query_terms(expanded_query)
        query_terms, spelling_corrections = _correct_query_terms_spelling(query_terms)

    query_terms = _clean_query_terms(query_terms)
    main_term = query_terms[0] if query_terms else ""
    addon_terms = query_terms[1:] if len(query_terms) > 1 else []

    for imp in _expand_implicit_addon_terms(main_term):
        _append_unique_term(addon_terms, imp)

    combined = ", ".join([main_term, *addon_terms]).strip(", ")
    if not combined:
        _load_packages_cache()
        scheme_pkgs = _filter_by_scheme(_all_packages_cache, getattr(request, "scheme", ""))
        pt_type = getattr(request, "patient_type", "")
        if pt_type:
            scheme_pkgs = [p for p in scheme_pkgs if _passes_patient_type(p, pt_type)]
        
        limit = max(25, min(100, request.limit))
        raw = [_build_raw_package_row(p) for p in scheme_pkgs[:limit]]
        return SmartSearchResponse(doctor_reasoning="Showing available packages.", raw_packages=raw)

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
    relevant = _filter_by_scheme(relevant, getattr(request, "scheme", ""))
    relevant = _prioritize_exact_main_term_first(relevant, main_term)

    addon_pkgs: list[dict] = []
    addon_by_term: dict[str, list[dict]] = {}
    for at in addon_terms:
        res = _search_packages_basic(at, limit=30, patient_type=request.patient_type if hasattr(request, 'patient_type') else "")
        res = _filter_by_scheme(res, getattr(request, "scheme", ""))
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
        if hasattr(resp, 'usage') and resp.usage:
            result.usage = {
                "prompt_tokens": resp.usage.prompt_tokens,
                "completion_tokens": resp.usage.completion_tokens,
                "total_tokens": resp.usage.total_tokens,
            }

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
                            reason = f"[RULE VIOLATION] {addon.get('reason', '')} (Warning: {err})"
                        else:
                            reason = addon.get("reason", "")
                            if at["is_extended_los"]: reason = f"Rule 5: Extended LOS. {reason}"
                            elif at["is_addon"]: reason = f"Rule 3: Compatible add-on. {reason}"
                        
                        selected_codes.add(ac)
                        if ac not in seen:
                            all_relevant.append(p)
                            seen.add(ac)
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
                            reason = f"Clinical add-on: {at_term}"
                        else:
                            violations.append(err)
                            reason = f"[RULE VIOLATION] Requested add-on: {at_term} (Warning: {err})"
                    else:
                        prefix = "Clinical add-on" if at_term.lower() in {"blood transfusion", "transfusion"} else "Requested add-on"
                        reason = f"{prefix}: {at_term}"

                    selected_codes.add(ac)
                    if ac not in seen:
                        all_relevant.append(p)
                        seen.add(ac)
                    result.suggested_addons.append(PackageResultModel(
                        package_code=ac, package_name=pkg_name(p), rate=pkg_rate(p),
                        speciality=pkg_specialty(p), category=pkg_category(p),
                        is_addon=True, medical_reason=reason,
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
                            if ac not in seen:
                                all_relevant.append(p)
                                seen.add(ac)
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
            
        if extracted_history_summary:
            result.doctor_reasoning = (
                f"Extracted clinical keywords from history: {extracted_history_summary}\n\n"
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

def _filter_by_scheme(pkgs: list[dict], scheme: str) -> list[dict]:
    """Filter packages by scheme: 'maa', 'pmjay', or '' for all."""
    if not scheme:
        return pkgs
    scheme_lower = scheme.lower()
    return [p for p in pkgs if p.get("_source", "maa").lower() == scheme_lower]


async def _get_or_reconstruct_flow(session_id: str) -> tuple[Any, list[dict], list[dict], dict]:
    from memory.sqlite_store import AgentMemory
    from tools.smart_search_flow import reconstruct_flow_from_state

    if session_id in _interactive_flows:
        d = _interactive_flows[session_id]
        return d["flow"], d["all_packages"], d["packages"], d.get("per_term_packages", {})

    data = AgentMemory().get_session(session_id)
    if not data:
        raise HTTPException(404, "Session not found or expired. Please start a new search.")

    query = data.get("query", "")
    addon_terms = data.get("addon_terms", [])
    sels = data.get("selections_list", [])
    pt_type = data.get("patient_type", "")
    scheme = data.get("scheme", "")

    _load_packages_cache()
    all_pkgs = _filter_by_scheme(_all_packages_cache, scheme)
    if pt_type:
        all_pkgs = [p for p in all_pkgs if _passes_patient_type(p, pt_type)]
        
    matching = _prioritize_exact_main_term_first(_search_packages_basic(query, 200, patient_type=pt_type), query)
    matching = _filter_by_scheme(matching, scheme)
    per_term: dict[str, list] = {query: matching}
    for t in addon_terms:
        tp = _prioritize_exact_main_term_first(_search_packages_basic(t, 200, patient_type=pt_type), t)
        per_term[t] = _filter_by_scheme(tp, scheme)

    flow = reconstruct_flow_from_state(query=query, addon_terms=addon_terms, selections=sels,
                                       matching_packages=matching, all_packages=all_pkgs, per_term_packages=per_term)
    _interactive_flows[session_id] = {
        "flow": flow, "packages": matching, "all_packages": all_pkgs,
        "per_term_packages": per_term, "created_at": time.time(), "selections_list": sels,
        "request": {"query": query, "patient_type": pt_type, "scheme": scheme},
    }
    return flow, all_pkgs, matching, per_term


def _sync_session_db(session_id: str, flow: Any, sels: list[dict], pt_type: str = "", scheme: str = ""):
    try:
        from memory.sqlite_store import AgentMemory
        AgentMemory().save_session(session_id, "interactive_search", {
            "query": flow.query, "addon_terms": [t for t in flow.parsed_terms if t != flow.query],
            "selections_list": sels, "flow_complete": flow.flow_complete,
            "patient_type": pt_type, "scheme": scheme,
        })
    except Exception as e:
        logger.error("Failed to sync session: %s", e)


def _deterministic_keyword_extractor(query: str) -> dict:
    t = (query or "").lower()
    keywords = []
    
    # ── Phase 1: Compound clinical term detection (MUST run first) ──────────
    # These are multi-word patterns that map to a SINGLE package keyword.
    # Order matters: check longer/more-specific patterns first.
    compound_clinical_map = [
        # Electrical burns — compound conditions (voltage + limb status)
        (r'high\s+voltage.*(?:limb\s*loss|amputat)', "Electrical contact burns High voltage with limb loss"),
        (r'(?:limb\s*loss|amputat).*high\s+voltage', "Electrical contact burns High voltage with limb loss"),
        (r'high\s+voltage.*(?:electr|burn|contact)', "Electrical contact burns High voltage"),
        (r'(?:electr|contact).*burn.*high\s+voltage', "Electrical contact burns High voltage"),
        (r'low\s+voltage.*(?:limb\s*loss|amputat)', "Electrical contact burns Low voltage with limb loss"),
        (r'(?:limb\s*loss|amputat).*low\s+voltage', "Electrical contact burns Low voltage with limb loss"),
        (r'low\s+voltage.*(?:electr|burn|contact)', "Electrical contact burns Low voltage"),
        (r'(?:electr|contact).*burn.*low\s+voltage', "Electrical contact burns Low voltage"),
        # Ortho compound
        (r'total\s+knee\s+replacement', "Total Knee Replacement"),
        (r'total\s+hip\s+replacement', "Total Hip Replacement"),
        (r'total\s+shoulder\s+replacement', "Total Shoulder Replacement"),
        (r'coronary\s+artery\s+bypass', "CABG"),
        (r'severe\s+sepsis', "Severe Sepsis"),
        (r'septic\s+shock', "Septic Shock"),
    ]
    
    compound_matched = set()
    for pattern, standard in compound_clinical_map:
        if re.search(pattern, t, flags=re.IGNORECASE):
            keywords.append(standard)
            compound_matched.add(standard.lower())
    
    # ── Phase 2: TBSA percentage extraction ─────────────────────────────────
    tbsa_match = re.search(r'(\d+)\s*%\s*(?:tbsa|total\s+body\s+surface|body\s+surface)', t)
    if not tbsa_match:
        tbsa_match = re.search(r'(?:tbsa|total\s+body\s+surface)\s*(?:area)?\s*(?:of|:)?\s*(\d+)\s*%', t)
    tbsa_pct = int(tbsa_match.group(1)) if tbsa_match else None
    
    # ── Phase 3: Simple keyword mapping (skip if compound already matched) ──
    clinical_keywords_map = {
        # Orthopedics
        "tkr": "Total Knee Replacement",
        "thr": "Total Hip Replacement",
        "fracture": "Fracture",
        "arthroplasty": "Arthroplasty",
        
        # Cardiology / Cardiothoracic
        "ptca": "PTCA",
        "angioplasty": "PTCA",
        "cabg": "CABG",
        "angiography": "Coronary Angiography",
        "myocardial infarction": "Myocardial Infarction",
        "heart attack": "Myocardial Infarction",
        "chest pain": "Myocardial Infarction",
        
        # Gastrointestinal / General Surgery
        "appendicectomy": "Appendicectomy",
        "appendectomy": "Appendicectomy",
        "appendicitis": "Appendicectomy",
        "cholecystectomy": "Cholecystectomy",
        "cholecystitis": "Cholecystectomy",
        "lap chole": "Cholecystectomy",
        "hernia": "Hernia",
        "inguinal hernia": "Hernia",
        
        # Urology / Nephrology
        "pcnl": "PCNL",
        "renal transplant": "Renal Transplant",
        "kidney transplant": "Renal Transplant",
        "kidney stone": "Renal Calculi",
        
        # Burns (simple — compound already handled above)
        "flame burns": "Flame burns",
        "thermal burns": "Thermal burns",
        "chemical burns": "Chemical burns",
        "scald burns": "Scald burns",
        "burns": "Burns",
        
        # Gynecology / Obstetrics
        "hysterectomy": "Hysterectomy",
        "c-section": "Caesarean Delivery",
        "cesarean": "Caesarean Delivery",
        "caesarean": "Caesarean Delivery",
        
        # Medical / Supportive (these are always SECONDARY to primary disease)
        "sepsis": "Sepsis",
        "septicemia": "Sepsis",
        "anemia": "Anemia",
        "anaemia": "Anemia",
        "thalassemia": "Thalassemia",
    }
    
    for term, standard in clinical_keywords_map.items():
        # Skip if compound already matched this concept
        if standard.lower() in compound_matched:
            continue
        # For burns: if TBSA was matched and compound already added a burn keyword, skip
        if "burn" in standard.lower() and any("burn" in c for c in compound_matched):
            continue
        pattern = r'\b' + re.escape(term) + r'\b'
        if re.search(pattern, t, flags=re.IGNORECASE):
            # Append TBSA% to burn keywords if available
            if "burn" in standard.lower() and tbsa_pct is not None:
                standard = f"{standard} {tbsa_pct}% TBSA"
            keywords.append(standard)
            
    # Deduplicate while preserving order
    seen = set()
    deduped = []
    for k in keywords:
        if k.lower() not in seen:
            seen.add(k.lower())
            deduped.append(k)
    
    # ── Phase 4: Enforce primary-disease-first ordering ─────────────────────
    # Supportive/add-on terms must never be first
    _SUPPORTIVE_TERMS = {"blood transfusion", "transfusion", "icu stay", "ventilator support", 
                          "extended los", "blood component"}
    if len(deduped) > 1:
        first_is_supportive = deduped[0].lower() in _SUPPORTIVE_TERMS
        if first_is_supportive:
            # Move first supportive to end, promote first non-supportive
            supportive = deduped[0]
            non_supportive = [k for k in deduped if k.lower() not in _SUPPORTIVE_TERMS]
            supportive_all = [k for k in deduped if k.lower() in _SUPPORTIVE_TERMS]
            deduped = non_supportive + supportive_all
            
    # Fallback to query if empty
    if not deduped:
        deduped = [query]
        
    detected_pt = _detect_patient_type_from_text(query)
    
    return {
        "summary": "Clinical Case Summary (Deterministic Fallback)",
        "msso_instructions": "Verify matched packages.",
        "keywords": deduped,
        "patient_type": detected_pt,
    }


async def _extract_keywords_from_history(query: str) -> dict:
    if not _async_groq_client:
        return _deterministic_keyword_extractor(query)
        
    # Build a dynamic list of valid specialties + procedure keywords from loaded packages
    _load_packages_cache()
    specialty_names = sorted({pkg_specialty(p).strip() for p in _all_packages_cache if pkg_specialty(p).strip()})
    # Extract unique short procedure keywords from package names (first 60 chars) for reference
    procedure_sample = set()
    for p in _all_packages_cache:
        name = pkg_name(p).strip()
        if name:
            # Take up to first pipe or newline as the procedure label
            short = name.split("|")[0].split("\n")[0].strip()[:80]
            if short:
                procedure_sample.add(short)
    # Limit to ~120 most common procedure names to keep prompt manageable
    procedure_names_list = sorted(procedure_sample)[:120]
    specialties_str = ", ".join(specialty_names)
    procedures_str = ", ".join(procedure_names_list)

    prompt = f"""You are a PMJAY/MAA Yojana package keyword extractor. Think like a treating doctor.
The system contains packages from BOTH MAA Yojana (Gujarat state) and PMJAY (Ayushman Bharat national) schemes.

TASK: 
1. Generate a brief 2-5 word clinical summary of the patient's condition.
2. Convert user input into EXACT medical package name keywords that match packages in our database.
3. Write brief instructions for the PMJAY MSSO operator (e.g., "Admit under Cardiology, book PTCA").

CRITICAL RULES:
1. ONLY return ONE package keyword per distinct medical condition. Do NOT return multiple packages for the same condition (e.g., for heart attack, pick ONLY the most appropriate primary intervention, like "PTCA" OR "Coronary Angiography", NOT both).
2. If the user mentions multiple conditions (e.g., "heart attack and kidney stone"), return ONE package keyword for the first, and ONE for the second.
3. If the user's input is a layman term (e.g. "heart attack"), translate it to medical terminology. If the user's input is ALREADY a valid medical diagnosis or procedure (e.g. "Anemia", "Sepsis", "Appendectomy"), keep it exactly as-is. NEVER force a diagnosis to become a treatment (e.g., do NOT translate "Anemia" into "Blood Transfusion").
4. EXCLUDE hospital stay details, ward types, ICU stays/days, or facility tiers (e.g., "private hospital", "2 days ICU", "ward stay") from the "keywords" list. These are supportive accommodations, not primary procedures/treatments.
5. PRESERVE CLINICAL QUANTIFIERS in keywords — TBSA percentages, voltage levels (high/low), laterality (bilateral/unilateral), severity grades, and body regions MUST be included inline.
6. COMPOUND CONDITIONS = SINGLE KEYWORD: When burn type + severity/complication form ONE clinical package (e.g., "high voltage electrical burns with limb loss"), output it as a SINGLE keyword, NOT split into separate keywords.
7. PRIMARY DISEASE FIRST: The first keyword MUST be the primary disease/procedure. Supportive procedures (blood transfusion, ICU, ventilator, extended LOS) must NEVER be the first keyword — they go after the primary condition if mentioned at all.
8. For orthopedic procedures, preserve the EXACT procedure name: "Total Knee Replacement" not "Arthroplasty", "Total Hip Replacement" not "Joint Replacement".
9. PATIENT TYPE EXTRACTION: If the query mentions "child", "infant", "toddler", "baby", "neonate", "paediatric", "pediatric", or an age under 18 (e.g., "4 year old", "5yr"), you MUST output patient_type as "Pediatric". Otherwise, output "Adult".
10. SPECIFIC BURN TYPES: Distinguish between "Flame burns", "Thermal burns", "Scald burns" (hot water/liquids), "Chemical burns", and "Electrical burns". Do not default all burns to thermal/flame.

AVAILABLE SPECIALTIES:
{specialties_str}

SAMPLE PACKAGE/PROCEDURE NAMES (use these as reference for keyword extraction):
{procedures_str}

EXAMPLES:
"heart attack" → {{"summary": "Acute Myocardial Infarction", "msso_instructions": "Admit under Cardiology. Book PTCA package.", "keywords": ["PTCA"], "patient_type": "Adult"}}
"kidney stone" → {{"summary": "Renal Calculi", "msso_instructions": "Admit under Urology. Book PCNL or Ureteroscopy.", "keywords": ["PCNL"], "patient_type": "Adult"}}
"Flame burns with 35% total body surface area in adult patient" → {{"summary": "35% TBSA Flame Burns", "msso_instructions": "Admit under Plastic Surgery. Book Flame Burns 35% TBSA package.", "keywords": ["Flame burns 35% TBSA"], "patient_type": "Adult"}}
"High voltage electrical burns with limb loss and amputation" → {{"summary": "HV Electrical Burns with Limb Loss", "msso_instructions": "Admit under Plastic Surgery. Book Electrical contact burns High voltage with limb loss.", "keywords": ["Electrical contact burns High voltage with limb loss"], "patient_type": "Adult"}}
"bilateral total knee replacement for osteoarthritis" → {{"summary": "Bilateral TKR for OA", "msso_instructions": "Admit under Orthopedics. Book Total Knee Replacement.", "keywords": ["Total Knee Replacement"], "patient_type": "Adult"}}
"Severe sepsis with blood transfusion and 3 days ICU" → {{"summary": "Severe Sepsis requiring ICU", "msso_instructions": "Admit under Medicine/ICU. Book Severe Sepsis package.", "keywords": ["Severe Sepsis"], "patient_type": "Adult"}}
"CABG with 3 vessel disease" → {{"summary": "CABG 3-vessel", "msso_instructions": "Admit under CTVS. Book CABG package.", "keywords": ["CABG"], "patient_type": "Adult"}}
"Appendicitis in child" → {{"summary": "Pediatric Appendicitis", "msso_instructions": "Admit under Pediatric Surgery. Book Appendicectomy.", "keywords": ["Appendicectomy"], "patient_type": "Pediatric"}}
"Scald burns 30% TBSA on a 4yr child" → {{"summary": "30% TBSA Scald Burns in Child", "msso_instructions": "Admit under Plastic Surgery/Pediatrics. Book Scald burns 30% TBSA package.", "keywords": ["Scald burns 30% TBSA"], "patient_type": "Pediatric"}}
"Planned repeat C-section" → {{"summary": "Repeat Caesarean Section", "msso_instructions": "Admit under Obstetrics. Book Caesarean Delivery.", "keywords": ["Caesarean Delivery"], "patient_type": "Adult"}}

Input: "{query}"

Return ONLY valid JSON in this format: {{"summary": "...", "msso_instructions": "...", "keywords": ["..."], "patient_type": "Adult"|"Pediatric"}}"""

    try:
        resp = await _async_groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            response_format={"type": "json_object"},
            timeout=3.0,
        )
        parsed = json.loads(resp.choices[0].message.content)

        keywords = parsed.get("keywords") or [query]
        # Hard cap at 6 keywords
        if len(keywords) > 6:
            keywords = keywords[:6]

        # Determine patient type from AI response + fallback heuristic
        ai_patient_type = parsed.get("patient_type", "").strip()
        if ai_patient_type not in ("Adult", "Pediatric"):
            ai_patient_type = _detect_patient_type_from_text(query)

        # Keep the exact medical keywords generated by the AI
        final_keywords = keywords
        
        # ── Post-LLM safety net: enforce primary-disease-first ordering ──
        _SUPPORTIVE_KEYWORDS = {"blood transfusion", "blood component", "transfusion", 
                                 "icu stay", "icu", "ventilator", "ventilator support",
                                 "extended los", "extended length of stay"}
        if len(final_keywords) > 1 and final_keywords[0].lower() in _SUPPORTIVE_KEYWORDS:
            non_supportive = [k for k in final_keywords if k.lower() not in _SUPPORTIVE_KEYWORDS]
            supportive = [k for k in final_keywords if k.lower() in _SUPPORTIVE_KEYWORDS]
            if non_supportive:
                final_keywords = non_supportive + supportive
                logger.info("Post-LLM reorder: moved supportive keyword '%s' after primary", supportive)
        
        # Use the clinical summary from the LLM if available, fallback to the original query
        clinical_summary = parsed.get("summary", query)
        msso_instructions = parsed.get("msso_instructions", "")

        return {
            "summary": clinical_summary,
            "msso_instructions": msso_instructions,
            "keywords": final_keywords,
            "patient_type": ai_patient_type,
        }
    except Exception as e:
        logger.error("Query analysis failed: %s. Using deterministic fallback.", e)
        return _deterministic_keyword_extractor(query)

@app.post("/interactive-search/analyze-query")
async def analyze_interactive_query(request: AnalyzeQueryRequest):
    """NLP analysis of free-text patient history → exact PMJAY package name keywords."""
    res = await _extract_keywords_from_history(request.query)
    return JSONResponse(content=res)


def _detect_patient_type_from_text(text: str) -> str:
    """Heuristic fallback to detect Adult vs Pediatric from free text."""
    t = (text or "").lower()

    # Check for explicit pediatric mentions
    pedia_words = ["pediatric", "paediatric", "child", "infant", "neonatal",
                   "neonate", "newborn", "toddler", "juvenile", "baby"]
    if any(w in t for w in pedia_words):
        return "Pediatric"

    # Check for age mentions
    age_match = re.search(r'(\d{1,3})\s*[-–]?\s*(?:year|yr|y/?o|yrs)', t)
    if age_match:
        age = int(age_match.group(1))
        return "Pediatric" if age < 18 else "Adult"

    # Check for explicit adult mentions
    adult_words = ["adult", "elderly", "geriatric"]
    if any(w in t for w in adult_words):
        return "Adult"

    return "Adult"  # Default


@app.post("/interactive-search/start", response_model=InteractiveSearchStartResponse)
async def start_interactive_search(request: InteractiveSearchStartRequest):
    """Start multi-step interactive package selection flow."""
    try:
        from tools.smart_search_flow import build_search_flow, _split_query_terms as flow_split, advance_past_empty_optional_steps
        import asyncio

        expanded_proc, expanded_dis = await asyncio.gather(
            _expand_abbreviations_llm(request.procedure),
            _expand_abbreviations_llm(request.disease)
        )

        corrections = {}
        pt_type = request.patient_type

        # Extract keywords if it's a long clinical history sentence/paragraph
        if len((request.query or "").split()) > 3:
            hist_res = await _extract_keywords_from_history(request.query)
            terms = hist_res.get("keywords", [])
            if not pt_type:
                pt_type = hist_res.get("patient_type", "")
        else:
            expanded_query = await _expand_abbreviations_llm(request.query)
            terms = flow_split(expanded_query)
            terms, corrections = _correct_query_terms_spelling(terms)

        if expanded_proc and expanded_proc not in terms:
            terms.insert(0, expanded_proc)
        if expanded_dis and expanded_dis not in terms:
            terms.append(expanded_dis)
            
        terms = _clean_query_terms(terms)
        if not terms:
            raise HTTPException(400, "Please provide a query, procedure, or disease")

        main_term = terms[0]
        addon_terms = terms[1:]
        if not pt_type:
            pt_type = _detect_patient_type_from_text(request.query)
        scheme = request.scheme  # 'maa', 'pmjay', or ''

        matching = _prioritize_exact_main_term_first(_search_packages_basic(main_term, 200, patient_type=pt_type), main_term)
        matching = _filter_by_scheme(matching, scheme)
        if request.disease:
            seen = {pkg_code(p) for p in matching}
            disease_pkgs = _filter_by_scheme(_search_packages_basic(request.disease, 120, patient_type=pt_type), scheme)
            for p in disease_pkgs:
                if pkg_code(p) not in seen:
                    matching.append(p)

        if not matching:
            _load_packages_cache()
            all_pkgs = _filter_by_scheme(_all_packages_cache, scheme)
            if not all_pkgs:
                scheme_label = {"maa": "MAA Yojana", "pmjay": "PMJAY"}.get(scheme, "any scheme")
                raise HTTPException(503, f"No packages loaded for {scheme_label}.")
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
            scheme_label = {"maa": "MAA Yojana", "pmjay": "PMJAY"}.get(scheme, "")
            raise HTTPException(404, f"No {scheme_label} packages found for: {main_term}".strip())

        _load_packages_cache()
        all_pkgs = _filter_by_scheme(_all_packages_cache, scheme)
        # ── Apply patient-type filter to the entire pool ──
        if pt_type:
            all_pkgs = [p for p in all_pkgs if _passes_patient_type(p, pt_type)]

        per_term: dict[str, list] = {main_term: matching}
        for t in addon_terms:
            tp = _prioritize_exact_main_term_first(_search_packages_basic(t, 200, patient_type=pt_type), t)
            tp = _filter_by_scheme(tp, scheme)
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
                         "patient_type": pt_type, "scheme": scheme},
        }
        _sync_session_db(sid, flow, [], pt_type=pt_type, scheme=scheme)

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
    flow, all_pkgs, _, _ = await _get_or_reconstruct_flow(session_id)
    from tools.smart_search_flow import advance_past_empty_optional_steps
    advance_past_empty_optional_steps(flow)
    _auto_advance_single_option_steps(flow, all_pkgs)

    if flow.flow_complete:
        return {"status": "complete", "final_recommendation": flow.final_recommendation, "selections": flow.selections}

    step = flow.steps[flow.current_step] if flow.current_step < len(flow.steps) else None
    if not step:
        raise HTTPException(500, "Invalid flow state")
    return _step_to_response(step).dict()


@app.post("/interactive-search/{session_id}/select")
async def submit_step_selection(session_id: str, selection: SelectionRequest):
    flow, all_pkgs, _, _ = await _get_or_reconstruct_flow(session_id)
    from tools.smart_search_flow import process_step_selection

    ok, err = process_step_selection(flow, {"id": selection.option_id, "notes": selection.notes,
                                             "manual_package": selection.manual_package}, all_pkgs)
    if not ok:
        return SelectionResponse(success=False, message=f"Error: {err}")

    sels = _interactive_flows[session_id].get("selections_list", [])
    sels.append({"id": selection.option_id, "notes": selection.notes, "manual_package": selection.manual_package})
    _interactive_flows[session_id]["selections_list"] = sels
    _pt = _interactive_flows.get(session_id, {}).get("request", {}).get("patient_type", "")
    _sync_session_db(session_id, flow, sels, pt_type=_pt)
    _auto_advance_single_option_steps(flow, all_pkgs)

    if flow.flow_complete:
        final = await _build_final_recommendation(flow, all_pkgs)
        flow.final_recommendation = final
        return SelectionResponse(success=True, message="Search complete!", flow_complete=True, final_recommendation=final)

    step = flow.steps[flow.current_step] if flow.current_step < len(flow.steps) else None
    if not step:
        return SelectionResponse(success=True, message="Flow completed.", flow_complete=True)
    return SelectionResponse(success=True, message="Selection received.", next_step=_step_to_response(step))


@app.post("/interactive-search/{session_id}/undo")
async def undo_step_selection(session_id: str):
    flow, _, _, _ = await _get_or_reconstruct_flow(session_id)
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

        pkg = code_to_pkg.get(sel_code) or _get_package_by_code(sel_code, packages)
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
        elif sel_id.startswith("variant_"):
            v_rate = float(sel.get("rate", 0))
            v_label = str(sel.get("label", ""))
            if g.get("main_package"):
                g["subtotal"] -= g["main_package"]["rate"]
                g["main_package"]["rate"] = v_rate
                g["main_package"]["package_category"] = v_label
                g["subtotal"] += v_rate
                if result["main_package"] and result["main_package"]["code"] == sel_code:
                    result["main_package"]["rate"] = v_rate
                    result["main_package"]["package_category"] = v_label
                for sp in result["selected_packages"]:
                    if sp["code"] == sel_code:
                        sp["rate"] = v_rate
                        sp["package_category"] = v_label
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
# PRO INTERACTIVE MULTI-STEP SEARCH (DYNAMIC AI-DRIVEN)
# ═══════════════════════════════════════════════════════════════════════

# ── Token-Efficient Helpers ──────────────────────────────────────────

# Keywords that indicate the query contains info relevant to each step type
_STRAT_RELEVANT_KEYWORDS = frozenset({
    "bilateral", "unilateral", "single", "double", "left", "right",
    "mild", "moderate", "severe", "grade", "stage", "type",
    "open", "laparoscopic", "lap", "minimally invasive",
    "with deformity", "without deformity", "high risk", "low risk",
    "primary", "revision", "redo", "recurrent",
    "tbsa", "%", "percent",
})

_IMPLANT_RELEVANT_KEYWORDS = frozenset({
    "implant", "stent", "des", "bms", "drug eluting", "bare metal",
    "valve", "pacemaker", "prosthesis", "prosthetic",
    "plate", "nail", "rod", "screw", "fixation",
    "with implant", "without implant", "no implant",
    "cemented", "uncemented", "hybrid",
})

_VARIANT_RELEVANT_KEYWORDS = frozenset({
    "nabh", "non-nabh", "non nabh", "public", "private",
    "government", "district", "tertiary",
})

def _query_contains_step_relevant_info(query: str, step_type: str) -> bool:
    """Quick keyword check: does the query contain info relevant to this step type?
    Used to skip LLM calls when the answer is obviously 'skip' or 'default'.
    """
    q = (query or "").lower()
    if step_type == "stratification":
        return any(kw in q for kw in _STRAT_RELEVANT_KEYWORDS)
    elif step_type == "implant":
        return any(kw in q for kw in _IMPLANT_RELEVANT_KEYWORDS)
    elif step_type == "variant":
        return any(kw in q for kw in _VARIANT_RELEVANT_KEYWORDS)
    return True  # For unknown step types, assume relevant

class ProAutoStepResponse(BaseModel):
    selected_option_ids: list[str]

@app.get("/pro-interactive-search/{session_id}/pro_auto_step", response_model=ProAutoStepResponse)
async def get_pro_auto_step(session_id: str):
    """Evaluates the current interactive step and returns the option IDs the AI would select."""
    if session_id not in _interactive_flows:
        raise HTTPException(404, "Session not found")
        
    session_data = _interactive_flows[session_id]
    flow = session_data["flow"]
    request_data = session_data["request"]
    
    if flow.flow_complete:
        return ProAutoStepResponse(selected_option_ids=[])
        
    # 1. Retrieve or run smart search
    smart_res = session_data.get("smart_res")
    if not smart_res:
        smart_req = SmartSearchRequest(
            query=request_data.get("query", ""),
            procedure=request_data.get("procedure", ""),
            disease=request_data.get("disease", ""),
            symptoms=request_data.get("symptoms", []),
            patient_age=request_data.get("patient_age", 0),
            patient_gender=request_data.get("patient_gender", ""),
            patient_type=request_data.get("patient_type", ""),
            scheme=request_data.get("scheme", ""),
            limit=50
        )
        smart_res = await smart_search(smart_req)
        session_data["smart_res"] = smart_res

    _load_packages_cache()
    query_text = request_data.get("query", "")
    current_step = flow.steps[flow.current_step]
    options = current_step.options

    is_primary = current_step.context.get("is_primary_selection", False)
    is_term = current_step.context.get("is_term_selection", False)
    is_variant = current_step.context.get("is_variant_selection", False)
    is_consolidated_addons = current_step.context.get("is_consolidated_addons", False)

    real_strat_opts = [opt for opt in options if opt["id"].startswith("strat_") and opt["id"] not in ("strat_skip", "manual_add_strat")]
    real_implant_opts = [opt for opt in options if opt["id"].startswith("implant_") and opt["id"] not in ("implant_skip", "manual_add_implant")]
    is_strat = len(real_strat_opts) > 0
    is_implant = len(real_implant_opts) > 0

    selected_options = []

    if is_primary or is_term:
        # ── Cross-validated package selection with add-on guard ──
        valid_opts = []
        for opt in options:
            if opt["id"] in ("package_skip", "manual_add_main"):
                continue
            opt_name = (opt.get("description", "") + " " + opt.get("label", "")).upper()
            if is_primary and any(tag in opt_name for tag in ("[ADD-ON", "[ADD ON", "ADD-ON PKG", "ADDON")):
                continue
            valid_opts.append(opt)
        
        recommended_code = None
        if is_primary:
            recommended_code = smart_res.main_package.package_code if (smart_res and smart_res.main_package) else None
        else:
            for add in (smart_res.suggested_addons if smart_res else []):
                if any(opt.get("code") == add.package_code for opt in options):
                    recommended_code = add.package_code
                    break
        if recommended_code:
            selected_option = next((opt for opt in valid_opts if opt.get("code") == recommended_code), None)
            # Cross-validation
            if selected_option and valid_opts and is_primary:
                top3_codes = {opt.get("code") for opt in valid_opts[:3]}
                if recommended_code not in top3_codes:
                    selected_option = valid_opts[0]
            if selected_option: selected_options.append(selected_option)
        if not selected_options and valid_opts:
            selected_options.append(valid_opts[0])

    elif is_strat:
        selected_option = None
        if _query_contains_step_relevant_info(query_text, "stratification"):
            selected_option = await _check_query_for_choice(
                query=query_text, options=options,
                step_type="stratification", client=_async_groq_client
            )
        if not selected_option:
            skip_opt = next((opt for opt in options if "skip" in opt["id"]), None)
            if skip_opt:
                selected_option = skip_opt
            elif real_strat_opts:
                selected_option = real_strat_opts[0]
        if selected_option: selected_options.append(selected_option)

    elif is_implant:
        selected_option = None
        has_implant_info = _query_contains_step_relevant_info(query_text, "implant")
        procedure_needs_implant = _procedure_implies_implant(query_text)
        
        if has_implant_info:
            selected_option = await _check_query_for_choice(
                query=query_text, options=options,
                step_type="implant", client=_async_groq_client
            )
        if not selected_option:
            if procedure_needs_implant and real_implant_opts:
                selected_option = real_implant_opts[0]
            else:
                skip_opt = next((opt for opt in options if "skip" in opt["id"]), None)
                if skip_opt:
                    selected_option = skip_opt
                elif real_implant_opts:
                    selected_option = real_implant_opts[0]
        if selected_option: selected_options.append(selected_option)

    elif is_variant:
        selected_option = None
        if _query_contains_step_relevant_info(query_text, "variant"):
            selected_option = await _check_query_for_choice(
                query=query_text, options=options,
                step_type="variant", client=_async_groq_client
            )
        if not selected_option:
            real_variants = [opt for opt in options if opt["id"].startswith("variant_")]
            if real_variants:
                selected_option = real_variants[0]
            else:
                selected_option = options[0] if options else None
        if selected_option: selected_options.append(selected_option)

    elif is_consolidated_addons:
        matched_options = await _check_query_for_multiple_choices(
            query=query_text, options=options,
            step_type="supportive care / stay duration",
            client=_async_groq_client
        )
        if matched_options:
            for opt in matched_options:
                selected_options.append(opt)
        
        selected_option = next((opt for opt in options if opt["id"] == "addon_skip"), None)
        if selected_option: selected_options.append(selected_option)
        
    if not selected_options:
        selected_option = next((opt for opt in options if "skip" in opt["id"]), None)
        if not selected_option and options:
            real_opts = [opt for opt in options if not opt["id"].startswith("manual_add")]
            selected_option = real_opts[0] if real_opts else options[0]
        if selected_option: selected_options.append(selected_option)
            
    return ProAutoStepResponse(selected_option_ids=[opt["id"] for opt in selected_options])


# Procedures that clinically require implants — auto-include the implant step
_PROCEDURES_IMPLYING_IMPLANTS = {
    "ptca", "angioplasty", "pci", "stent", "coronary stenting",
    "total knee replacement", "tkr", "total hip replacement", "thr",
    "total shoulder replacement", "arthroplasty",
    "pacemaker", "valve replacement", "aortic valve", "mitral valve",
    "cabg",  # Sometimes uses grafts
    "fixation", "plating", "nailing", "intramedullary",
}

def _procedure_implies_implant(query: str) -> bool:
    """Check if the procedure in the query clinically requires an implant."""
    q = (query or "").lower()
    return any(proc in q for proc in _PROCEDURES_IMPLYING_IMPLANTS)

async def _advance_pro_standard_flow(session_id: str) -> InteractiveSearchStartResponse:
    if session_id not in _interactive_flows:
        raise HTTPException(404, "Session not found")
        
    session_data = _interactive_flows[session_id]
    flow = session_data["flow"]
    all_packages = session_data["all_packages"]
    request_data = session_data["request"]
    sels = session_data.get("selections_list", [])
    
    # 1. Retrieve or run smart search
    smart_res = session_data.get("smart_res")
    if not smart_res:
        smart_req = SmartSearchRequest(
            query=request_data.get("query", ""),
            procedure=request_data.get("procedure", ""),
            disease=request_data.get("disease", ""),
            symptoms=request_data.get("symptoms", []),
            patient_age=request_data.get("patient_age", 0),
            patient_gender=request_data.get("patient_gender", ""),
            patient_type=request_data.get("patient_type", ""),
            scheme=request_data.get("scheme", ""),
            limit=50
        )
        smart_res = await smart_search(smart_req)
        session_data["smart_res"] = smart_res
        
    _load_packages_cache()
    
    asked = False
    from tools.smart_search_flow import process_step_selection
    
    max_iterations = 20
    iterations = 0
    
    query_text = request_data.get("query", "")
    
    while not flow.flow_complete and iterations < max_iterations:
        iterations += 1
        current_step = flow.steps[flow.current_step]
        options = current_step.options
        
        is_primary = current_step.context.get("is_primary_selection", False)
        is_term = current_step.context.get("is_term_selection", False)
        is_variant = current_step.context.get("is_variant_selection", False)
        is_consolidated_addons = current_step.context.get("is_consolidated_addons", False)
        
        real_strat_opts = [opt for opt in options if opt["id"].startswith("strat_") and opt["id"] not in ("strat_skip", "manual_add_strat")]
        real_implant_opts = [opt for opt in options if opt["id"].startswith("implant_") and opt["id"] not in ("implant_skip", "manual_add_implant")]
        is_strat = len(real_strat_opts) > 0
        is_implant = len(real_implant_opts) > 0
        
        selected_option = None
        
        if is_primary or is_term:
            # ── Cross-validated package selection with add-on guard ──
            from tools.smart_search_flow import _is_addon_package as _flow_is_addon
            
            # Build list of valid options (exclude skip/manual/add-on-as-primary)
            valid_opts = []
            for opt in options:
                if opt["id"] in ("package_skip", "manual_add_main"):
                    continue
                # Block add-on packages from being selected as PRIMARY
                opt_name = (opt.get("description", "") + " " + opt.get("label", "")).upper()
                if is_primary and any(tag in opt_name for tag in ("[ADD-ON", "[ADD ON", "ADD-ON PKG", "ADDON")):
                    logger.info("Pro: Blocking add-on package from primary: %s", opt.get("code"))
                    continue
                valid_opts.append(opt)
            
            recommended_code = None
            if is_primary:
                recommended_code = smart_res.main_package.package_code if (smart_res and smart_res.main_package) else None
            else:
                for add in (smart_res.suggested_addons if smart_res else []):
                    if any(opt.get("code") == add.package_code for opt in options):
                        recommended_code = add.package_code
                        break
                        
            if recommended_code:
                selected_option = next((opt for opt in valid_opts if opt.get("code") == recommended_code), None)
                
                # ── Cross-validation: verify AI pick is in flow's top 3 ranked options ──
                if selected_option and valid_opts and is_primary:
                    top3_codes = {opt.get("code") for opt in valid_opts[:3]}
                    if recommended_code not in top3_codes:
                        # AI recommended a package that's not in the flow's top 3
                        # Prefer the flow's #1 (which is deterministically ranked by relevance score)
                        logger.warning(
                            "Pro: AI recommended %s but flow top-3 is %s — preferring flow #1: %s",
                            recommended_code, top3_codes, valid_opts[0].get("code")
                        )
                        selected_option = valid_opts[0]
                
            if not selected_option and valid_opts:
                selected_option = valid_opts[0]
                    
        elif is_strat:
            # ── TOKEN-EFFICIENT: Check if query has stratification info first ──
            if _query_contains_step_relevant_info(query_text, "stratification"):
                selected_option = await _check_query_for_choice(
                    query=query_text, options=options,
                    step_type="stratification", client=_async_groq_client
                )
            if not selected_option:
                # Smart default: auto-skip stratification when query has no relevant info
                skip_opt = next((opt for opt in options if "skip" in opt["id"]), None)
                if skip_opt:
                    selected_option = skip_opt
                    logger.info("Pro: Auto-skipping strat (no relevant query info)")
                elif real_strat_opts:
                    # Fallback: select first stratification option
                    selected_option = real_strat_opts[0]
                    logger.info("Pro: Defaulting to first strat option: %s", selected_option.get("id"))
                
        elif is_implant:
            # ── TOKEN-EFFICIENT: Check if query mentions implant info OR procedure implies implant ──
            has_implant_info = _query_contains_step_relevant_info(query_text, "implant")
            procedure_needs_implant = _procedure_implies_implant(query_text)
            
            if has_implant_info:
                selected_option = await _check_query_for_choice(
                    query=query_text, options=options,
                    step_type="implant", client=_async_groq_client
                )
            
            if not selected_option:
                if procedure_needs_implant and real_implant_opts:
                    # Procedure clinically requires implant — auto-select first (most relevant)
                    selected_option = real_implant_opts[0]
                    logger.info("Pro: Auto-including implant (procedure implies it): %s", selected_option.get("id"))
                else:
                    # No implant info and procedure doesn't need it — skip
                    skip_opt = next((opt for opt in options if "skip" in opt["id"]), None)
                    if skip_opt:
                        selected_option = skip_opt
                        logger.info("Pro: Auto-skipping implant (no relevant query info)")
                    elif real_implant_opts:
                        selected_option = real_implant_opts[0]
                        logger.info("Pro: Defaulting to first implant option: %s", selected_option.get("id"))
                
        elif is_variant:
            # ── TOKEN-EFFICIENT: Check if query mentions variant info ──
            if _query_contains_step_relevant_info(query_text, "variant"):
                selected_option = await _check_query_for_choice(
                    query=query_text, options=options,
                    step_type="variant", client=_async_groq_client
                )
            if not selected_option:
                # Smart default: select first variant (most common/cheapest rate)
                real_variants = [opt for opt in options if opt["id"].startswith("variant_")]
                if real_variants:
                    selected_option = real_variants[0]
                    logger.info("Pro: Defaulting to first variant: %s", selected_option.get("id"))
                else:
                    selected_option = options[0] if options else None
                
        elif is_consolidated_addons:
            # ── Deterministic first, then LLM for add-ons ──
            matched_options = await _check_query_for_multiple_choices(
                query=query_text, options=options,
                step_type="supportive care / stay duration",
                client=_async_groq_client
            )
            if matched_options:
                for opt in matched_options:
                    if not any(s.get("id") == opt["id"] for s in sels):
                        sels.append({
                            "id": opt["id"],
                            "notes": "Pro auto-selected",
                            "manual_package": None
                        })
                        process_step_selection(flow, opt, all_packages)
            
            selected_option = next((opt for opt in options if opt["id"] == "addon_skip"), None)
            
        # ── Fallback: always pick skip or first option (never leave stuck) ──
        if not selected_option:
            selected_option = next((opt for opt in options if "skip" in opt["id"]), None)
            if not selected_option and options:
                # Filter out manual_add options and pick first real option
                real_opts = [opt for opt in options if not opt["id"].startswith("manual_add")]
                selected_option = real_opts[0] if real_opts else options[0]
                
        if selected_option:
            if not any(s.get("id") == selected_option["id"] for s in sels):
                sels.append({
                    "id": selected_option["id"],
                    "notes": "Pro auto-selected",
                    "manual_package": None
                })
            session_data["selections_list"] = sels
            
            success, err = process_step_selection(flow, selected_option, all_packages)
            if not success:
                logger.warning("Pro: Step selection failed: %s — retrying with skip", err)
                # Try skip option as last resort
                skip_opt = next((opt for opt in options if "skip" in opt["id"]), None)
                if skip_opt and skip_opt["id"] != selected_option["id"]:
                    sels[-1] = {"id": skip_opt["id"], "notes": "Pro auto-selected (fallback)", "manual_package": None}
                    success, err = process_step_selection(flow, skip_opt, all_packages)
                if not success:
                    asked = True
                    break
                
            _sync_session_db(session_id, flow, sels, pt_type=request_data.get("patient_type", ""), scheme=request_data.get("scheme", ""))
            _auto_advance_single_option_steps(flow, all_packages)
        else:
            asked = True
            break
            
    _sync_session_db(session_id, flow, sels, pt_type=request_data.get("patient_type", ""), scheme=request_data.get("scheme", ""))
    
    if asked:
        current_step = flow.steps[flow.current_step]
        return InteractiveSearchStartResponse(
            session_id=session_id,
            query=request_data.get("query", ""),
            parsed_terms=flow.parsed_terms,
            current_step=_step_to_response(current_step),
            message="Please provide the missing clinical details to complete selection.",
            status="interactive",
            final_recommendation=None
        )
    else:
        final_rec_dict = await _build_final_recommendation(flow, all_packages)
        flow.final_recommendation = final_rec_dict
        _sync_session_db(session_id, flow, sels, pt_type=request_data.get("patient_type", ""), scheme=request_data.get("scheme", ""))
        return InteractiveSearchStartResponse(
            session_id=session_id,
            query=request_data.get("query", ""),
            parsed_terms=flow.parsed_terms,
            current_step=None,
            message="AI successfully matched and selected all package details.",
            status="complete",
            final_recommendation=final_rec_dict
        )

def _deterministic_choice_matcher(query: str, options: list[dict], step_type: str) -> Optional[dict]:
    q = (query or "").lower()
    
    # Filter to only real options (exclude skip/manual)
    real_options = [opt for opt in options if "skip" not in str(opt.get("id", "")).lower() and not str(opt.get("id", "")).startswith("manual_add")]
    
    # 1. Skip / No Implant detection
    if "without implant" in q or "no implant" in q:
        for opt in options:
            opt_id = str(opt.get("id", "")).lower()
            opt_code = str(opt.get("code", "")).lower()
            if "skip" in opt_id or "no_implant" in opt_code:
                return opt

    # 2. Bilateral vs Unilateral
    if "bilateral" in q:
        for opt in options:
            label = str(opt.get("label", "")).lower()
            desc = str(opt.get("description", "")).lower()
            if "bilateral" in label or "bilateral" in desc:
                return opt
    elif "unilateral" in q or "single" in q:
        for opt in options:
            label = str(opt.get("label", "")).lower()
            desc = str(opt.get("description", "")).lower()
            if "unilateral" in label or "unilateral" in desc or "single" in label or "primary knee replacement" in label:
                return opt
    else:
        # Default: if bilateral/unilateral options exist but query doesn't specify,
        # prefer unilateral (most common)
        has_bilateral_opt = any("bilateral" in str(opt.get("label", "")).lower() for opt in options)
        if has_bilateral_opt:
            for opt in options:
                label = str(opt.get("label", "")).lower()
                desc = str(opt.get("description", "")).lower()
                if "unilateral" in label or "unilateral" in desc or "single" in label or "primary knee replacement" in label:
                    return opt

    # 3. Laparoscopic vs Open
    if "laparoscopic" in q or "lap " in q or "laparoscop" in q:
        for opt in options:
            label = str(opt.get("label", "")).lower()
            if "laparoscopic" in label or "lap." in label or "lap " in label or "laparoscop" in label:
                return opt
    elif "open" in q:
        for opt in options:
            label = str(opt.get("label", "")).lower()
            if "open" in label:
                return opt

    # 4. NABH vs Non-NABH / Public vs Private
    if "non-nabh" in q or "non nabh" in q:
        for opt in options:
            label = str(opt.get("label", "")).lower()
            if "non-nabh" in label or "non nabh" in label:
                return opt
    elif "nabh" in q:
        for opt in options:
            label = str(opt.get("label", "")).lower()
            if "nabh" in label and "non-nabh" not in label:
                return opt

    if "public" in q:
        for opt in options:
            label = str(opt.get("label", "")).lower()
            if "public" in label:
                return opt
    elif "private" in q:
        for opt in options:
            label = str(opt.get("label", "")).lower()
            if "private" in label:
                return opt

    # 5. Severity / Grade / Risk matching
    for severity in ["severe", "moderate", "mild", "high risk", "low risk"]:
        if severity in q:
            for opt in real_options:
                label = str(opt.get("label", "")).lower()
                desc = str(opt.get("description", "")).lower()
                if severity in label or severity in desc:
                    return opt

    # 6. Stent type matching (Cardiology)
    if "drug eluting" in q or "des" in q.split():
        for opt in real_options:
            label = str(opt.get("label", "")).lower()
            if "drug eluting" in label or "des" in label.split():
                return opt
    if "bare metal" in q or "bms" in q.split():
        for opt in real_options:
            label = str(opt.get("label", "")).lower()
            if "bare metal" in label or "bms" in label.split():
                return opt

    # 7. Cemented vs Uncemented (Orthopedics)
    if "cemented" in q and "uncemented" not in q:
        for opt in real_options:
            label = str(opt.get("label", "")).lower()
            if "cemented" in label and "uncemented" not in label:
                return opt
    elif "uncemented" in q:
        for opt in real_options:
            label = str(opt.get("label", "")).lower()
            if "uncemented" in label:
                return opt

    # 8. General keyword matching — check if any option code is in the query
    for opt in options:
        code = str(opt.get("code", "")).strip().lower()
        if code and len(code) > 4 and code in q:
            return opt

    return None

def _deterministic_multiple_choices_matcher(query: str, options: list[dict], step_type: str) -> list[dict]:
    q = (query or "").lower()
    matched = []
    
    # ── Extract numeric day counts from query (e.g., "5 days ICU", "3 day ward") ──
    day_matches = re.findall(r'(\d+)\s*(?:days?|nights?)\s*(icu|ward|general|private|stay)?', q)
    requested_icu_days = 0
    requested_ward_days = 0
    for count_str, ward_type in day_matches:
        days_val = int(count_str)
        if "icu" in (ward_type or ""):
            requested_icu_days = days_val
        else:
            requested_ward_days = days_val
    
    for opt in options:
        opt_id = str(opt.get("id", "")).lower()
        if "skip" in opt_id or "manual_add" in opt_id:
            continue
        label = str(opt.get("label", "")).lower()
        desc = str(opt.get("description", "")).lower()
        combined = label + " " + desc
        
        # Blood transfusion
        if "transfusion" in combined or "blood" in combined or "packed cell" in combined or "platelet" in combined:
            if "transfusion" in q or "blood" in q or "packed cell" in q or "platelet" in q or "sdp" in q or "ffp" in q:
                matched.append(opt)
                continue
                
        # ICU (with day-count matching)
        if "icu" in combined or "intensive care" in combined:
            if "icu" in q or "intensive care" in q:
                # If specific day count requested, try to match
                if requested_icu_days > 0:
                    day_in_label = re.search(r'(\d+)\s*(?:days?|nights?)', combined)
                    if day_in_label:
                        label_days = int(day_in_label.group(1))
                        if label_days == requested_icu_days:
                            matched.append(opt)
                            continue
                    else:
                        matched.append(opt)
                        continue
                else:
                    matched.append(opt)
                    continue
                
        # Ventilator
        if "ventilator" in combined or "ventilation" in combined or "respiratory support" in combined:
            if "ventilator" in q or "ventilation" in q or "intubated" in q or "intubation" in q:
                matched.append(opt)
                continue
                
        # Extended LOS
        if "extended los" in combined or "extended length of stay" in combined or "extended stay" in combined:
            if "extended los" in q or "extended length" in q or "extended stay" in q or " anticoagulation " in q or "dvt" in q:
                matched.append(opt)
                continue

        # General/Private Ward (with day-count matching)
        if ("general ward" in combined or "ward stay" in combined or "ward" in combined) and "icu" not in combined:
            if "general ward" in q or "ward stay" in q or ("ward" in q and "icu" not in q):
                if requested_ward_days > 0:
                    day_in_label = re.search(r'(\d+)\s*(?:days?|nights?)', combined)
                    if day_in_label:
                        label_days = int(day_in_label.group(1))
                        if label_days == requested_ward_days:
                            matched.append(opt)
                            continue
                    else:
                        matched.append(opt)
                        continue
                else:
                    matched.append(opt)
                    continue
                
    return matched

async def _check_query_for_choice(query: str, options: list[dict], step_type: str, client: Any) -> Optional[dict]:
    # Call deterministic matcher first
    det_match = _deterministic_choice_matcher(query, options, step_type)
    if det_match:
        logger.info(f"Deterministic choice match for {step_type}: {det_match.get('id')}")
        return det_match
        
    if not client:
        return None
    
    # Filter out manual/skip options so LLM focuses only on the real choices
    filtered_options = [
        {"id": opt.get("id"), "label": opt.get("label"), "description": opt.get("description"), "rate": opt.get("rate")}
        for opt in options if not str(opt.get("id", "")).startswith("manual_add") and "skip" not in str(opt.get("id", "")).lower()
    ]
    
    if not filtered_options:
        return None
        
    options_str = json.dumps(filtered_options, indent=2)
    
    # Find if there is a skip option
    skip_option = next((opt for opt in options if "skip" in str(opt.get("id", "")).lower()), None)
    
    # ── Compact prompt: send only id+label (not full description/rate) to save tokens ──
    compact_options = [{"id": o["id"], "label": o.get("label", "")[:60]} for o in filtered_options]
    compact_str = json.dumps(compact_options)
    
    prompt = f"""Clinical decision support. Query: "{query}"
Choosing: "{step_type}". Options:
{compact_str}

Rules:
1. If query specifies an option, return its id.
2. If NO implant/stratification/supportive care needed, return "skip".
3. If options are irrelevant to the patient's condition, return "skip".
4. If ambiguous/insufficient info, return null.

Return JSON: {{"matched_id": "id"|"skip"|null, "reason": "brief reason"}}
"""
    try:
        resp = await client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "system", "content": "You are a clinical decision support system. Always respond in JSON."},
                      {"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=300,
            response_format={"type": "json_object"},
            timeout=2.0,
        )
        data = json.loads(resp.choices[0].message.content)
        matched_id = data.get("matched_id")
        if matched_id == "skip" and skip_option:
            return skip_option
        if matched_id:
            for opt in options:
                if opt.get("id") == matched_id:
                    return opt
    except Exception as e:
        logger.error(f"Error in _check_query_for_choice: {e}")
    return None

async def _check_query_for_multiple_choices(query: str, options: list[dict], step_type: str, client: Any) -> list[dict]:
    # Call deterministic matcher first
    det_matches = _deterministic_multiple_choices_matcher(query, options, step_type)
    if det_matches:
        logger.info(f"Deterministic multiple choice matches for {step_type}: {[m.get('id') for m in det_matches]}")
        return det_matches
        
    if not client:
        return []
    
    filtered_options = [
        {"id": opt.get("id"), "label": opt.get("label", "")[:60]}
        for opt in options if not str(opt.get("id", "")).startswith("manual_add") and "skip" not in str(opt.get("id", "")).lower()
    ]
    if not filtered_options:
        return []
        
    compact_str = json.dumps(filtered_options)
    prompt = f"""Clinical decision support. Query: "{query}"
Selecting add-ons for: "{step_type}". Options:
{compact_str}

Return ids of options explicitly requested or clinically required. Empty list if none match.
Return JSON: {{"matched_ids": ["id1", ...]}}
"""
    try:
        resp = await client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "system", "content": "You are a clinical decision support system. Always respond in JSON."},
                      {"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=300,
            response_format={"type": "json_object"},
            timeout=2.0,
        )
        data = json.loads(resp.choices[0].message.content)
        matched_ids = data.get("matched_ids", [])
        matched_options = []
        if isinstance(matched_ids, list):
            for m_id in matched_ids:
                opt = next((o for o in options if o.get("id") == m_id), None)
                if opt:
                    matched_options.append(opt)
            return matched_options
    except Exception as e:
        logger.error(f"Error in _check_query_for_multiple_choices: {e}")
    return []

@app.post("/pro-interactive-search/start", response_model=InteractiveSearchStartResponse)
async def start_pro_interactive_search(request: InteractiveSearchStartRequest):
    # 1. Start standard search first to get parsed_terms and session setup
    res = await start_interactive_search(request)
    sid = res.session_id
    
    # 2. Advance the flow automatically using standard search steps
    return await _advance_pro_standard_flow(sid)

@app.post("/pro-interactive-search/recalculate")
async def recalculate_pro_recommendation(req: RecalculateRequest):
    flow, all_pkgs, _, _ = await _get_or_reconstruct_flow(req.session_id)
    if not flow:
        raise HTTPException(404, "Session not found.")
        
    _load_packages_cache()
    def get_full_package(code: str) -> dict:
        return _get_package_by_code(code, _all_packages_cache)

    def _entry(p: dict) -> dict:
        return {
            "code": pkg_code(p),
            "name": pkg_name(p)[:100],
            "rate": pkg_rate(p),
            "specialty": pkg_specialty(p),
            "package_category": pkg_category(p),
            "pre_auth_document": p.get("PRE AUTH DOCUMENT", p.get("Mandatory Documents", "")),
            "claim_document": p.get("CLAIM DOCUMENT", p.get("Mandatory Documents - Claim Processing", "")),
        }

    main_packages = []
    implant_packages = []
    addon_packages = []
    stratification_packages = []
    custom_rates = req.custom_rates or {}
    package_types = req.package_types or {}

    for code in req.package_codes:
        p = get_full_package(code)
        if not p:
            continue
        rate = custom_rates.get(code, pkg_rate(p))
        entry = _entry(p)
        entry["rate"] = rate
        
        c_up = pkg_category(p).upper()
        s_up = pkg_specialty(p).upper()
        n_up = pkg_name(p).upper()
        
        forced_type = package_types.get(code, "AUTO")
        
        if forced_type == "IMPLANT":
            entry["reason"] = "Manually selected implant package."
            implant_packages.append(entry)
        elif forced_type == "STRATIFICATION":
            entry["reason"] = "Manually selected stratification package."
            stratification_packages.append(entry)
        elif forced_type == "ADDON":
            entry["reason"] = "Manually selected addon package."
            addon_packages.append(entry)
        elif forced_type == "MAIN":
            entry["reason"] = "Manually selected main package."
            main_packages.append(entry)
        else:
            if "IMPLANT" in s_up or "IMPLANT" in c_up or "IMPLANT" in n_up:
                implant_packages.append(entry)
            elif "ADD-ON" in c_up or "ADDON" in c_up or "ADD-ON" in s_up or "ADDON" in s_up or "TRANSFUSION" in n_up:
                entry["reason"] = "Manually selected addon package."
                addon_packages.append(entry)
            elif "-STR" in code.upper() or "STRATIFICATION" in c_up or "STRATIFICATION" in n_up or "EXTENDED" in n_up:
                entry["reason"] = "Manually selected stratification package."
                stratification_packages.append(entry)
            else:
                main_packages.append(entry)

    main_pkg_dict = main_packages[0] if main_packages else None
    selected_packages = main_packages

    subtotal = (main_pkg_dict["rate"] if main_pkg_dict else 0.0) + sum(imp["rate"] for imp in implant_packages) + sum(strat["rate"] for strat in stratification_packages)
    
    term_groups = [{
        "term": flow.query or "Main Case",
        "main_package": main_pkg_dict,
        "implant_packages": implant_packages,
        "stratification_packages": stratification_packages,
        "subtotal": subtotal
    }]

    from tools.smart_search_flow import validate_package_combination
    blocked_rules = []
    
    if main_pkg_dict:
        full_main = get_full_package(main_pkg_dict["code"])
        if full_main:
            full_imp = get_full_package(implant_packages[0]["code"]) if implant_packages else None
            full_combo = []
            for sp in selected_packages[1:]:
                full_sp = get_full_package(sp["code"])
                if full_sp:
                    full_combo.append(full_sp)
            for a in addon_packages:
                full_a = get_full_package(a["code"])
                if full_a:
                    full_combo.append(full_a)
            
            valid, viols = validate_package_combination(full_main, full_imp, None, full_combo)
            blocked_rules = viols

    doctor_reasoning = "Calculated manually customized package bundle."
    approval_likelihood = "MEDIUM"
    if blocked_rules:
        approval_likelihood = "LOW"
    else:
        approval_likelihood = "HIGH"

    prompt = f"""You are a PMJAY/MAA Yojana package verification AI.
The doctor has manually edited the selected package bundle.

Case query/history: {flow.query}

Selected Bundle:
- Main package: {main_pkg_dict["name"] if main_pkg_dict else "None"} (Code: {main_pkg_dict["code"] if main_pkg_dict else "None"})
- Implants: {', '.join([imp["name"] for imp in implant_packages]) or "None"}
- Add-ons: {', '.join([add["name"] for add in addon_packages]) or "None"}

Please provide:
1. An AI clinical reason explaining why this combination of procedures/implants/addons is appropriate or if there are any PMJAY guidelines to note.
2. The overall AI claim approval likelihood ("HIGH", "MEDIUM", or "LOW").

Output ONLY raw valid JSON:
{{
  "doctor_reasoning": "A concise 2-3 sentence clinical reasoning explaining this specific package bundle combination.",
  "approval_likelihood": "HIGH" | "MEDIUM" | "LOW"
}}
"""
    try:
        content = ""
        if DEEPSEEK_API_KEY and _async_deepseek_client:
            try:
                resp = await _async_deepseek_client.chat.completions.create(
                    model=DEEPSEEK_MODEL,
                    messages=[{"role": "user", "content": prompt}],
                    stream=False,
                    reasoning_effort="high",
                    temperature=0.0,
                    response_format={"type": "json_object"},
                    extra_body={"thinking": {"type": "enabled"}}
                )
                content = resp.choices[0].message.content
            except Exception as d_err:
                logger.warning(f"DeepSeek recalculate request error: {d_err}")
                
        if not content and _async_groq_client:
            resp_groq = await _async_groq_client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                response_format={"type": "json_object"},
                timeout=3.0,
            )
            content = resp_groq.choices[0].message.content
            
        if content:
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            ai_res = json.loads(content)
            doctor_reasoning = ai_res.get("doctor_reasoning", doctor_reasoning)
            approval_likelihood = ai_res.get("approval_likelihood", approval_likelihood)
    except Exception as e:
        logger.error(f"Error calling AI for recalculation: {e}")

    if blocked_rules:
        approval_likelihood = "LOW"

    final_rec_dict = {
        "main_package": main_pkg_dict,
        "selected_packages": selected_packages,
        "implant_package": implant_packages[0] if implant_packages else None,
        "implant_packages": implant_packages,
        "stratification_package": stratification_packages[0] if stratification_packages else None,
        "stratification_packages": stratification_packages,
        "addon_packages": addon_packages,
        "term_groups": term_groups,
        "blocked_rules": blocked_rules,
        "approval_likelihood": approval_likelihood,
        "doctor_reasoning": doctor_reasoning
    }

    flow.final_recommendation = final_rec_dict
    _sync_session_db(req.session_id, flow, [], pt_type=getattr(flow, "patient_type", ""), scheme=getattr(flow, "scheme", ""))
    
    return final_rec_dict

@app.post("/pro-interactive-search/{session_id}/select")
async def submit_pro_step_selection(session_id: str, selection: SelectionRequest):
    flow, all_pkgs, _, _ = await _get_or_reconstruct_flow(session_id)
    if session_id not in _interactive_flows:
        raise HTTPException(404, "Session not found")
        
    session_data = _interactive_flows[session_id]
    
    # 1. Match the user's manual selection against the current step options
    current_step = flow.steps[flow.current_step]
    selected_option = None
    if selection.option_id.startswith("manual_add"):
        selected_option = {
            "id": selection.option_id,
            "manual_package": selection.manual_package
        }
    else:
        for opt in current_step.options:
            if opt.get("id") == selection.option_id:
                selected_option = opt
                break
                
    if not selected_option:
        raise HTTPException(400, "Invalid option ID")
        
    # 2. Process the manual selection
    from tools.smart_search_flow import process_step_selection
    sels = session_data.get("selections_list", [])
    
    if not any(s.get("id") == selected_option["id"] for s in sels):
        sels.append({
            "id": selected_option["id"],
            "notes": "User selected",
            "manual_package": selection.manual_package
        })
    session_data["selections_list"] = sels
    
    success, err = process_step_selection(flow, selected_option, all_pkgs)
    if not success:
        sels.pop()
        session_data["selections_list"] = sels
        raise HTTPException(400, err or "Validation failed")
        
    request_data = session_data["request"]
    _sync_session_db(session_id, flow, sels, pt_type=request_data.get("patient_type", ""), scheme=request_data.get("scheme", ""))
    
    _auto_advance_single_option_steps(flow, all_pkgs)
    
    # 3. Call auto-advance for the remaining steps and translate to SelectionResponse
    res = await _advance_pro_standard_flow(session_id)
    if res.status == "complete":
        return SelectionResponse(
            success=True,
            message=res.message,
            flow_complete=True,
            final_recommendation=res.final_recommendation
        )
    else:
        return SelectionResponse(
            success=True,
            message=res.message,
            next_step=res.current_step,
            flow_complete=False
        )

@app.get("/pro-interactive-search/{session_id}/step")
async def get_pro_current_step(session_id: str):
    return await get_current_step(session_id)

@app.get("/pro-interactive-search/{session_id}/status")
async def get_pro_flow_status(session_id: str):
    return await get_flow_status(session_id)

@app.post("/pro-interactive-search/{session_id}/undo")
async def undo_pro_step_selection(session_id: str):
    flow, _, _, _ = await _get_or_reconstruct_flow(session_id)
    from tools.smart_search_flow import undo_last_selection

    sels = _interactive_flows[session_id].get("selections_list", [])
    if not sels:
        return JSONResponse(status_code=400, content={"success": False, "message": "Already at the first step"})

    # Pop auto-selected steps first
    while sels:
        last_sel = sels[-1]
        is_auto = last_sel.get("notes") == "Pro auto-selected"
        
        ok, msg = undo_last_selection(flow)
        if not ok:
            break
        sels.pop()
        
        if not is_auto:
            # We popped the last user-selected step, stop here!
            break
            
    _interactive_flows[session_id]["selections_list"] = sels
    _pt = _interactive_flows.get(session_id, {}).get("request", {}).get("patient_type", "")
    _scheme = _interactive_flows.get(session_id, {}).get("request", {}).get("scheme", "")
    _sync_session_db(session_id, flow, sels, pt_type=_pt, scheme=_scheme)

    step = flow.steps[flow.current_step] if flow.current_step < len(flow.steps) else None
    if not step:
        return JSONResponse(status_code=500, content={"success": False, "message": "No current step after undo."})
    return {"success": True, "message": "Reverted to last choice step", "current_step": _step_to_response(step).dict()}

@app.post("/pro-interactive-search/{session_id}/feedback")
async def submit_pro_feedback(session_id: str, request: Request):
    try:
        data = await request.json()
        rating = data.get("rating")
        logger.info(f"PRO SEARCH FEEDBACK: session={session_id} rating={rating}")
        return {"success": True}
    except Exception as e:
        return {"success": False, "error": str(e)}


# ═══════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    uvicorn.run("main:app", host=SERVER_HOST, port=SERVER_PORT,
                workers=max(1, SERVER_WORKERS), reload=False)