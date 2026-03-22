"""
tools/smart_package_agent.py — Intelligent LLM-powered package selection system.

This agent thinks like a hospital package selector:
1. Understands the medical procedure/surgery from input
2. Applies MAA YOJANA booking rules DURING selection (not after)
3. Uses semantic understanding of medical terms
4. Returns properly grouped packages with reasoning

The 5 MAA YOJANA Booking Rules:
1. Surgical + medical management packages CANNOT be booked together
   (If surgical package exists, medical management packages with rate=0 are excluded)
2. Stand-alone packages CANNOT be booked with any other package
3. Add-on packages can ONLY be booked alongside a regular package
4. Implant packages automatically appear with their parent procedure
5. Extended LOS packages can ONLY be booked with a surgery package
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any
from functools import lru_cache

from config.settings import PACKAGES_JSON, TOP_K_PACKAGES, GROQ_API_KEY, GROQ_MODEL

logger = logging.getLogger("smart_package_agent")

# Robotic surgeries path
ROBOTIC_JSON = PACKAGES_JSON.parent / "maa_robotic_surgeries.json"

# ══════════════════════════════════════════════════════════════════════════════
# MEDICAL KNOWLEDGE BASE - Procedure to Specialty Mapping
# ══════════════════════════════════════════════════════════════════════════════

# Maps search terms to specialty keywords (partial match in package SPECIALITY field)
PROCEDURE_SPECIALTY_MAP = {
    # General Surgery
    "appendectomy": ["general surgery", "laparoscopic"],
    "appendicitis": ["general surgery", "laparoscopic"],
    "hernia": ["general surgery", "laparoscopic"],
    "inguinal hernia": ["general surgery"],
    "umbilical hernia": ["general surgery"],
    "cholecystectomy": ["general surgery", "laparoscopic"],
    "gallbladder": ["general surgery"],
    "gallstone": ["general surgery"],

    # Orthopedics
    "fracture": ["orthopaedics", "orthopedic", "ortho"],
    "joint replacement": ["orthopaedics", "orthopedic"],
    "knee replacement": ["orthopaedics", "orthopedic"],
    "hip replacement": ["orthopaedics", "orthopedic"],
    "arthroscopy": ["orthopaedics", "orthopedic"],
    "spine surgery": ["orthopaedics", "neurosurgery", "spine"],
    "disc": ["orthopaedics", "neurosurgery", "spine"],
    "tkr": ["orthopaedics", "orthopedic"],
    "thr": ["orthopaedics", "orthopedic"],

    # Cardiology & Cardio Thoracic
    "angioplasty": ["cardiology", "cardio"],
    "bypass": ["cardio-thoracic", "ctvs", "cardiac"],
    "cabg": ["cardio-thoracic", "ctvs", "cardiac"],
    "stent": ["cardiology", "cardio"],
    "pacemaker": ["cardiology", "cardio-thoracic", "ctvs"],
    "valve replacement": ["cardio-thoracic", "ctvs"],
    "coronary": ["cardiology", "cardio-thoracic", "ctvs"],
    "heart": ["cardiology", "cardio"],
    "cardiac": ["cardiology", "cardio-thoracic", "ctvs"],

    # Oncology
    "cancer": ["oncology", "surgical oncology", "medical oncology"],
    "tumor": ["oncology", "surgical oncology"],
    "chemotherapy": ["medical oncology", "oncology"],
    "radiation": ["radiation oncology", "oncology"],
    "mastectomy": ["surgical oncology", "oncology", "general surgery"],
    "carcinoma": ["oncology", "surgical oncology", "medical oncology"],

    # Neurosurgery
    "brain": ["neurosurgery", "neuro"],
    "craniotomy": ["neurosurgery", "neuro"],
    "spinal": ["neurosurgery", "spine", "orthopaedics"],

    # Urology
    "kidney": ["urology", "nephrology"],
    "prostate": ["urology"],
    "turp": ["urology"],
    "nephrectomy": ["urology"],
    "kidney stone": ["urology"],
    "lithotripsy": ["urology"],
    "renal": ["urology", "nephrology"],

    # Gynecology
    "hysterectomy": ["obstetrics", "gynaecology", "gynae"],
    "cesarean": ["obstetrics", "gynaecology", "gynae"],
    "c-section": ["obstetrics", "gynaecology"],
    "delivery": ["obstetrics", "gynaecology"],
    "fibroid": ["obstetrics", "gynaecology"],

    # Eye
    "cataract": ["ophthalmology", "eye"],
    "glaucoma": ["ophthalmology", "eye"],
    "retina": ["ophthalmology", "eye"],

    # ENT
    "tonsillectomy": ["ent"],
    "adenoidectomy": ["ent"],
    "septoplasty": ["ent"],
    "mastoidectomy": ["ent"],

    # Burns
    "burn": ["burns"],
    "thermal burn": ["burns"],
    "electrical burn": ["burns"],
    "skin graft": ["burns", "plastic surgery"],
}

# Medical management conditions (rate = 0)
MEDICAL_MANAGEMENT_KEYWORDS = {
    "diabetes", "hypertension", "fever", "infection", "pneumonia",
    "bronchitis", "gastritis", "dengue", "malaria", "typhoid",
    "anemia", "jaundice", "hepatitis", "diarrhea", "dysentery"
}

# Surgical keywords
SURGICAL_KEYWORDS = {
    "surgery", "operation", "excision", "resection", "removal",
    "repair", "reconstruction", "replacement", "transplant", "implant",
    "laparoscopic", "endoscopic", "arthroscopic", "bypass", "graft",
    "-ectomy", "-otomy", "-plasty", "-pexy", "-ostomy"
}


# ══════════════════════════════════════════════════════════════════════════════
# PACKAGE DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════

@lru_cache(maxsize=1)
def _load_all_packages() -> tuple[list[dict], list[dict]]:
    """Load both package files once and cache."""
    maa_packages = []
    robotic_packages = []

    if PACKAGES_JSON.exists():
        with open(PACKAGES_JSON, encoding="utf-8") as f:
            maa_packages = json.load(f)
        logger.info(f"Loaded {len(maa_packages)} MAA packages")

    if ROBOTIC_JSON.exists():
        with open(ROBOTIC_JSON, encoding="utf-8") as f:
            robotic_packages = json.load(f)
        logger.info(f"Loaded {len(robotic_packages)} robotic surgery packages")

    return maa_packages, robotic_packages


def _classify_package_type(pkg: dict, source: str) -> str:
    """Classify package type based on metadata."""
    if source == "robotic":
        pkg_type = (pkg.get("PACKAGE TYPE") or "").upper().strip()
        if pkg_type == "IMP":
            return "implant"
        proc_type = (pkg.get("Procedure Type") or "").lower()
        if "stand alone" in proc_type:
            return "standalone"
        if "add on" in proc_type:
            return "addon"
        if "day care" in proc_type:
            return "day_care"
        return "regular"

    # MAA packages
    name = (pkg.get("PACKAGE NAME") or "").upper()
    rate = pkg.get("RATE", 0)

    if "EXTENDED LOS" in name or "EXTENDED LENGTH OF STAY" in name:
        return "extended_los"
    if any(x in name for x in ["[STAND- ALONE]", "[STAND-ALONE]", "[STANDALONE]"]):
        return "standalone"
    if any(x in name for x in ["[ADD - ON PROCEDURE]", "[ADD-ON", "[GOVT RESERVE / ADD ON]"]):
        return "addon"

    strat = (pkg.get("STRATIFICATION PACKAGE") or "").strip()
    if rate == 0 and strat != "NO STRATIFICATION":
        return "medical_management"

    return "regular"


# ══════════════════════════════════════════════════════════════════════════════
# INTELLIGENT PACKAGE MATCHING WITH LLM
# ══════════════════════════════════════════════════════════════════════════════

def _get_groq_client():
    """Get Groq client for LLM calls."""
    if not GROQ_API_KEY:
        return None
    from groq import Groq
    return Groq(api_key=GROQ_API_KEY)


def _is_surgical_case(extracted_data: dict) -> bool:
    """Determine if this is a surgical case based on extracted data."""
    surgery_required = extracted_data.get("surgery_required", False)
    surgery_name = extracted_data.get("surgery_name", "")
    procedure_required = extracted_data.get("procedure_required", False)
    procedure_name = extracted_data.get("procedure_name", "")
    diagnosis = extracted_data.get("diagnosis", "").lower()

    # Direct surgical indicators
    if surgery_required or surgery_name:
        return True
    if procedure_required and procedure_name:
        # Check if procedure is surgical
        proc_lower = procedure_name.lower()
        for kw in SURGICAL_KEYWORDS:
            if kw in proc_lower:
                return True

    # Check diagnosis for surgical keywords
    for kw in SURGICAL_KEYWORDS:
        if kw in diagnosis:
            return True

    return False


def _build_search_context(extracted_data: dict) -> dict:
    """Build rich context for package search."""
    diagnosis = str(extracted_data.get("diagnosis", ""))
    secondary = extracted_data.get("secondary_diagnoses", [])
    procedures = extracted_data.get("procedures", [])
    surgery_name = str(extracted_data.get("surgery_name", ""))
    procedure_name = str(extracted_data.get("procedure_name", ""))
    department = str(extracted_data.get("department", ""))

    # Determine if surgical
    is_surgical = _is_surgical_case(extracted_data)

    # Find relevant specialties
    specialties = set()
    search_terms = [diagnosis.lower(), surgery_name.lower(),
                    procedure_name.lower()]
    search_terms.extend([str(p).lower() for p in procedures])

    for term in search_terms:
        for keyword, specs in PROCEDURE_SPECIALTY_MAP.items():
            if keyword in term:
                specialties.update(specs)

    if department:
        specialties.add(department)

    return {
        "diagnosis": diagnosis,
        "secondary_diagnoses": secondary,
        "procedures": procedures,
        "surgery_name": surgery_name,
        "procedure_name": procedure_name,
        "department": department,
        "is_surgical": is_surgical,
        "relevant_specialties": list(specialties),
        "search_terms": search_terms,
    }


def _score_package_intelligent(
    pkg: dict,
    source: str,
    context: dict,
) -> tuple[float, str]:
    """
    Score a package using intelligent matching.
    Returns (score, reasoning).
    """
    pkg_name = pkg.get("PACKAGE NAME", pkg.get("Package Name", "")).lower()
    pkg_specialty = pkg.get("SPECIALITY", pkg.get("Speciality", "")).lower()
    pkg_type = _classify_package_type(pkg, source)
    rate = pkg.get("RATE", pkg.get("Rate", 0))

    score = 0.0
    reasons = []

    # Rule 1: If surgical case, skip medical management (rate=0) packages
    if context["is_surgical"] and pkg_type == "medical_management":
        return 0.0, "Excluded: Medical management cannot combine with surgical"

    # Check specialty match (with partial matching)
    for spec in context["relevant_specialties"]:
        spec_lower = spec.lower()
        # Check if specialty keyword is in package specialty
        if spec_lower in pkg_specialty:
            score += 50  # High score for specialty match
            reasons.append(f"Specialty match: {spec}")
            break
        # Also check reverse
        if pkg_specialty and any(word in spec_lower for word in pkg_specialty.split()):
            score += 40
            reasons.append(f"Specialty related: {spec}")
            break

    # Check direct term matches in package name
    for term in context["search_terms"]:
        if term and len(term) > 2:
            if term in pkg_name:
                score += 35
                reasons.append(f"Term match: {term}")
            # Also check word boundaries
            elif f" {term}" in pkg_name or pkg_name.startswith(term):
                score += 25
                reasons.append(f"Partial term: {term}")

    # Check diagnosis match
    diagnosis_lower = context["diagnosis"].lower()
    if diagnosis_lower:
        diagnosis_words = [w for w in diagnosis_lower.split() if len(w) > 3]
        for word in diagnosis_words:
            if word in pkg_name:
                score += 30
                reasons.append(f"Diagnosis word: {word}")

    # Check surgery/procedure name match (highest priority)
    surgery = context["surgery_name"].lower()
    procedure = context["procedure_name"].lower()

    if surgery:
        surgery_words = [w for w in surgery.split() if len(w) > 3]
        for word in surgery_words:
            if word in pkg_name:
                score += 45
                reasons.append(f"Surgery term: {word}")

    if procedure:
        proc_words = [w for w in procedure.split() if len(w) > 3]
        for word in proc_words:
            if word in pkg_name:
                score += 40
                reasons.append(f"Procedure term: {word}")

    # Check department match
    dept = context.get("department", "").lower()
    if dept and dept in pkg_specialty:
        score += 35
        reasons.append(f"Department match: {dept}")

    # Boost for regular packages in surgical cases
    if context["is_surgical"] and pkg_type == "regular" and rate > 0:
        score += 5
        reasons.append("Surgical package bonus")

    # Process secondary diagnoses
    for sec in context["secondary_diagnoses"]:
        sec_lower = str(sec).lower()
        sec_words = [w for w in sec_lower.split() if len(w) > 3]
        for word in sec_words:
            if word in pkg_name:
                score += 15
                reasons.append(f"Secondary: {word}")

    reasoning = "; ".join(reasons) if reasons else "Low relevance"
    return score, reasoning


def _find_implant_packages(
    parent_pkg: dict,
    all_robotic: list[dict],
) -> list[dict]:
    """Find implant packages linked to a parent package."""
    implants = []
    parent_code = parent_pkg.get("PACKAGE CODE", "")

    if not parent_code:
        return implants

    # Check IMPLANT PACKAGE field
    implant_field = parent_pkg.get("IMPLANT PACKAGE", "")
    if implant_field and implant_field.upper() != "NO IMPLANT":
        # Parse implant codes from the field
        for code in re.findall(r"[\w-]+IMP\d*", implant_field):
            for rpkg in all_robotic:
                if rpkg.get("PACKAGE CODE", "").startswith(code):
                    implants.append(rpkg)

    # Also find robotic implants by code prefix
    for rpkg in all_robotic:
        r_code = rpkg.get("PACKAGE CODE", "")
        r_type = (rpkg.get("PACKAGE TYPE") or "").upper()
        if r_type == "IMP" and r_code.startswith(parent_code):
            if rpkg not in implants:
                implants.append(rpkg)

    return implants


# ══════════════════════════════════════════════════════════════════════════════
# LLM-POWERED INTELLIGENT PACKAGE SELECTION
# ══════════════════════════════════════════════════════════════════════════════

PACKAGE_SELECTION_PROMPT = """\
You are an expert MAA YOJANA (Ayushman Bharat) hospital package selector.

PATIENT CASE:
{case_summary}

CANDIDATE PACKAGES (top scored):
{package_list}

MAA YOJANA BOOKING RULES (MUST FOLLOW):
1. Surgical + medical management packages CANNOT be booked together
   - If a surgical package (rate > 0) is selected, medical management packages (rate = 0) are EXCLUDED
2. Stand-alone packages CANNOT be booked with any other package
   - If a standalone package is selected, ONLY that package + its linked implants allowed
3. Add-on packages can ONLY be booked alongside a regular package
   - If no regular package, add-ons are invalid
4. Implant packages automatically appear with their parent procedure
   - Already handled by system
5. Extended LOS packages can ONLY be booked with a surgery package
   - If no surgical package, extended LOS is invalid

YOUR TASK:
1. Analyze the patient case carefully
2. Select the MOST APPROPRIATE packages following the rules
3. Think like a hospital billing department selecting packages for pre-authorization

RESPOND IN JSON:
{{
  "selected_packages": [
    {{"code": "PACKAGE_CODE", "reason": "Why this package fits the case"}}
  ],
  "excluded_packages": [
    {{"code": "PACKAGE_CODE", "rule_violated": "Which rule was violated"}}
  ],
  "case_type": "surgical" or "medical_management",
  "primary_diagnosis_package": "The main package code for primary diagnosis",
  "reasoning": "Your overall reasoning for the selection"
}}
"""


def _call_llm_for_selection(
    context: dict,
    candidate_packages: list[dict],
) -> dict:
    """Use LLM to intelligently select packages."""
    client = _get_groq_client()

    if not client or not candidate_packages:
        return None

    # Build case summary
    case_summary = f"""
Diagnosis: {context['diagnosis']}
Secondary Diagnoses: {', '.join(str(d) for d in context['secondary_diagnoses']) or 'None'}
Surgery Required: {context['is_surgical']}
Surgery Name: {context['surgery_name'] or 'N/A'}
Procedure Name: {context['procedure_name'] or 'N/A'}
Department: {context['department'] or 'Unknown'}
Relevant Specialties: {', '.join(context['relevant_specialties']) or 'Unknown'}
"""

    # Build package list
    pkg_lines = []
    for pkg in candidate_packages[:15]:  # Limit to top 15 for LLM
        code = pkg.get("package_code", "")
        name = pkg.get("package_name", "")[:100]
        rate = pkg.get("rate", 0)
        ptype = pkg.get("package_type", "regular")
        score = pkg.get("alignment_score", 0)
        pkg_lines.append(
            f"- [{code}] {name} | Rate: {rate} | Type: {ptype} | Score: {score}")

    package_list = "\n".join(pkg_lines)

    prompt = PACKAGE_SELECTION_PROMPT.format(
        case_summary=case_summary,
        package_list=package_list
    )

    try:
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert MAA YOJANA hospital package selector. Always respond with valid JSON."
                },
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_completion_tokens=2000,
            response_format={"type": "json_object"},
        )

        text = response.choices[0].message.content
        if text:
            return json.loads(text)
    except Exception as e:
        logger.error(f"LLM package selection failed: {e}")

    return None


# ══════════════════════════════════════════════════════════════════════════════
# MAIN INTELLIGENT SEARCH FUNCTION
# ══════════════════════════════════════════════════════════════════════════════

def intelligent_package_search(
    extracted_data: dict,
    top_k: int = TOP_K_PACKAGES,
    use_llm: bool = True,
) -> dict[str, Any]:
    """
    Intelligent package search that thinks like a hospital package selector.

    Args:
        extracted_data: Extracted medical data from OCR
        top_k: Number of top packages to return
        use_llm: Whether to use LLM for intelligent selection

    Returns:
        Structured result with primary_packages, addon_packages, implant_packages,
        warnings, removed_packages, reasoning
    """
    maa_packages, robotic_packages = _load_all_packages()

    empty_result = {
        "primary_packages": [],
        "addon_packages": [],
        "implant_packages": [],
        "warnings": [],
        "removed_packages": [],
        "total_matched": 0,
        "reasoning": "",
        "case_type": "unknown",
    }

    if not maa_packages and not robotic_packages:
        return empty_result

    # Build search context
    context = _build_search_context(extracted_data)

    if not context["diagnosis"] and not context["surgery_name"]:
        empty_result["warnings"].append(
            "No diagnosis or surgery found in extracted data")
        return empty_result

    logger.info(f"Smart search - Diagnosis: {context['diagnosis']}, "
                f"Surgical: {context['is_surgical']}, "
                f"Specialties: {context['relevant_specialties']}")

    # ── Score all packages ──────────────────────────────────────────────
    scored_packages = []

    for pkg in maa_packages:
        score, reason = _score_package_intelligent(pkg, "maa", context)
        if score > 0:
            pkg_type = _classify_package_type(pkg, "maa")
            scored_packages.append({
                "source": "MAA YOJANA",
                "package_code": pkg.get("PACKAGE CODE", ""),
                "package_name": pkg.get("PACKAGE NAME", "").replace("\n", " ").strip(),
                "speciality": pkg.get("SPECIALITY", ""),
                "category": pkg.get("PACKAGE CATEGORY", ""),
                "rate": pkg.get("RATE", 0),
                "govt_reserve": pkg.get("GOVT RESERVE", "NO"),
                "pre_auth_documents": pkg.get("PRE AUTH DOCUMENT", ""),
                "claim_documents": pkg.get("CLAIM DOCUMENT", ""),
                "alignment_score": int(score),
                "package_type": pkg_type,
                "match_reasoning": reason,
                "_original": pkg,
            })

    for pkg in robotic_packages:
        score, reason = _score_package_intelligent(pkg, "robotic", context)
        if score > 0:
            pkg_type = _classify_package_type(pkg, "robotic")
            scored_packages.append({
                "source": "Robotic Surgery",
                "package_code": pkg.get("PACKAGE CODE", ""),
                "package_name": pkg.get("Package Name", "").replace("\n", " ").strip(),
                "speciality": pkg.get("Speciality", ""),
                "category": pkg.get("PACKAGE TYPE", ""),
                "rate": pkg.get("Rate", 0),
                "procedure_type": pkg.get("Procedure Type", ""),
                "mandatory_documents": pkg.get("Mandatory Documents", ""),
                "claim_documents": pkg.get("Mandatory Documents - Claim Processing", ""),
                "alignment_score": int(score),
                "package_type": pkg_type,
                "match_reasoning": reason,
                "_original": pkg,
            })

    # Sort by score
    scored_packages.sort(key=lambda x: x["alignment_score"], reverse=True)

    # Take candidates
    candidates = scored_packages[:top_k * 3]

    if not candidates:
        empty_result["warnings"].append(
            "No matching packages found for the given diagnosis")
        return empty_result

    # ── Apply booking rules ─────────────────────────────────────────────
    result = _apply_booking_rules_smart(
        candidates, context, maa_packages, robotic_packages)

    # ── Optional LLM refinement ─────────────────────────────────────────
    if use_llm and candidates:
        llm_selection = _call_llm_for_selection(context, candidates)
        if llm_selection:
            result["llm_reasoning"] = llm_selection.get("reasoning", "")
            result["case_type"] = llm_selection.get(
                "case_type", result.get("case_type", ""))

            # Apply LLM-suggested exclusions
            excluded_codes = {
                e.get("code") for e in llm_selection.get("excluded_packages", [])
            }
            if excluded_codes:
                for pkg_list in [result["primary_packages"], result["addon_packages"]]:
                    for pkg in pkg_list[:]:
                        if pkg["package_code"] in excluded_codes:
                            pkg["removal_reason"] = "LLM rule enforcement"
                            result["removed_packages"].append(pkg)
                            pkg_list.remove(pkg)

    # Limit results
    result["primary_packages"] = result["primary_packages"][:top_k]
    result["total_matched"] = (
        len(result["primary_packages"]) +
        len(result["addon_packages"]) +
        len(result["implant_packages"])
    )

    # Clean up internal fields
    for pkg_list in [result["primary_packages"], result["addon_packages"],
                     result["implant_packages"]]:
        for pkg in pkg_list:
            pkg.pop("_original", None)

    return result


def _apply_booking_rules_smart(
    packages: list[dict],
    context: dict,
    maa_packages: list[dict],
    robotic_packages: list[dict],
) -> dict[str, Any]:
    """
    Apply MAA YOJANA booking rules intelligently during selection.
    """
    primary = []
    addons = []
    implants = []
    extended_los = []
    standalones = []
    medical = []
    removed = []
    warnings = []

    # Classify packages
    for pkg in packages:
        ptype = pkg.get("package_type", "regular")
        if ptype == "standalone":
            standalones.append(pkg)
        elif ptype == "addon":
            addons.append(pkg)
        elif ptype == "implant":
            implants.append(pkg)
        elif ptype == "extended_los":
            extended_los.append(pkg)
        elif ptype == "medical_management":
            medical.append(pkg)
        else:
            primary.append(pkg)

    # ── RULE 2: Stand-alone isolation ───────────────────────────────────
    if standalones:
        best_standalone = max(
            standalones, key=lambda p: p.get("alignment_score", 0))

        # Find implants for this standalone
        sa_implants = []
        if "_original" in best_standalone:
            sa_implants = _find_implant_packages(
                best_standalone["_original"], robotic_packages)

        # Everything else gets removed
        all_others = primary + addons + extended_los + medical
        all_others += [s for s in standalones if s is not best_standalone]

        if all_others:
            warnings.append(
                f"🔒 RULE 2: '{best_standalone.get('package_name', 'Standalone')}' is STAND-ALONE "
                f"— {len(all_others)} other package(s) excluded."
            )
            for pkg in all_others:
                pkg["removal_reason"] = "Rule 2: Standalone isolation"
            removed.extend(all_others)

        # Format implants
        formatted_implants = []
        for imp in sa_implants:
            formatted_implants.append({
                "source": "Implant Package",
                "package_code": imp.get("PACKAGE CODE", ""),
                "package_name": imp.get("Package Name", "").replace("\n", " ").strip(),
                "speciality": imp.get("Speciality", ""),
                "rate": imp.get("Rate", 0),
                "package_type": "implant",
                "parent_code": best_standalone.get("package_code", ""),
            })

        return {
            "primary_packages": [best_standalone],
            "addon_packages": [],
            "implant_packages": formatted_implants,
            "warnings": warnings,
            "removed_packages": removed,
            "case_type": "standalone",
        }

    # ── RULE 1: Surgical + medical exclusion ────────────────────────────
    has_surgical = any(p.get("rate", 0) >
                       0 for p in primary) or context["is_surgical"]

    if has_surgical and medical:
        warnings.append(
            f"⚕️ RULE 1: Surgical case detected — {len(medical)} medical management package(s) excluded."
        )
        for pkg in medical:
            pkg["removal_reason"] = "Rule 1: Surgical + medical exclusion"
        removed.extend(medical)
        medical = []

    # ── RULE 3: Add-ons require regular package ─────────────────────────
    if addons and not primary:
        warnings.append(
            f"➕ RULE 3: {len(addons)} add-on package(s) need a regular package — none found."
        )
        for pkg in addons:
            pkg["removal_reason"] = "Rule 3: No regular package for add-on"
        removed.extend(addons)
        addons = []

    # ── RULE 5: Extended LOS requires surgery ───────────────────────────
    if extended_los and not has_surgical:
        warnings.append(
            f"🏥 RULE 5: Extended LOS packages require a surgery — none found."
        )
        for pkg in extended_los:
            pkg["removal_reason"] = "Rule 5: No surgery for extended LOS"
        removed.extend(extended_los)
        extended_los = []

    # ── RULE 4: Auto-include implants ───────────────────────────────────
    for pkg in primary:
        if "_original" in pkg:
            linked_implants = _find_implant_packages(
                pkg["_original"], robotic_packages)
            for imp in linked_implants:
                imp_code = imp.get("PACKAGE CODE", "")
                if not any(i.get("package_code") == imp_code for i in implants):
                    implants.append({
                        "source": "Implant Package",
                        "package_code": imp_code,
                        "package_name": imp.get("Package Name", "").replace("\n", " ").strip(),
                        "speciality": imp.get("Speciality", ""),
                        "rate": imp.get("Rate", 0),
                        "package_type": "implant",
                        "parent_code": pkg.get("package_code", ""),
                        "alignment_score": int(pkg.get("alignment_score", 0) * 0.8),
                    })

    # Add valid extended LOS to addons
    addons.extend(extended_los)

    # Add non-excluded medical to primary (if no surgical)
    if not has_surgical:
        primary.extend(medical)

    case_type = "surgical" if has_surgical else "medical_management"

    return {
        "primary_packages": primary,
        "addon_packages": addons,
        "implant_packages": implants,
        "warnings": warnings,
        "removed_packages": removed,
        "case_type": case_type,
    }


# ══════════════════════════════════════════════════════════════════════════════
# LEGACY COMPATIBILITY WRAPPER
# ══════════════════════════════════════════════════════════════════════════════

def search_packages_smart(
    diagnosis: str,
    speciality: str = "",
    extra_keywords: str = "",
    surgery_name: str = "",
    procedure_name: str = "",
    top_k: int = TOP_K_PACKAGES,
) -> dict[str, Any]:
    """
    Legacy-compatible wrapper for intelligent_package_search.
    """
    extracted_data = {
        "diagnosis": diagnosis,
        "department": speciality,
        "surgery_name": surgery_name,
        "procedure_name": procedure_name,
        "secondary_diagnoses": extra_keywords.split() if extra_keywords else [],
        "surgery_required": bool(surgery_name),
        "procedure_required": bool(procedure_name),
    }

    return intelligent_package_search(extracted_data, top_k=top_k)
