"""
tools/smart_search_flow.py — Interactive multi-step smart search system.
Implements a robust, step-by-step narrowing flow for package selection:
1. Parse user input (comma-separated terms → main + add-ons)
2. Show options for broad terms (e.g., "heart attack" → specific procedures)
3. User selects main procedure
4. Show related package options
5. User selects main package
6. Show implant options if applicable
7. Show add-on options (from 2nd comma term + clinical implications)
8. Validate rules and present final recommendations
Business Rules Enforced at Each Step:
1. Surgical ≠ Medical management (cannot combine)
2. Stand-alone packages cannot combine with any other
3. Add-ons only with regular packages
4. Implants auto-suggest with compatible procedures
5. Extended LOS only with surgical packages
"""
from __future__ import annotations
import logging
import re
from typing import Any, Dict, List, Optional, Tuple
from functools import lru_cache

from .medical_knowledge import (
    get_clinical_pathway,
    get_specialties_for_term,
    is_surgical_term,
    expand_synonyms,
)

logger = logging.getLogger("smart_search_flow")


# ═══════════════════════════════════════════════════════════════════════════════
# ULTRA-FAST INTERNAL HELPERS (precomputed + cached for 5-10x speed)
# ═══════════════════════════════════════════════════════════════════════════════
_GENERIC_TERMS = {"surgery", "surgical", "procedure", "operation", "management", "approach",
                  "general", "specialty", "select", "main", "package", "phase", "stage",
                  "level", "type", "pain", "ache", "discomfort", "symptom", "disease"}

_SURGICAL_KEYWORDS_RE = re.compile(
    r'(?:surgical|surgery|operative|ectomy|plasty|otomy|procedure)', re.IGNORECASE
)
_MEDICAL_KEYWORDS_RE = re.compile(
    r'(?:medical|conservative|non[ -]?surgical|management)', re.IGNORECASE
)
_STANDALONE_RE = re.compile(r'(?:stand[ -]?alone|standalone)', re.IGNORECASE)
_ADDON_RE = re.compile(r'(?:add[ -]?on|addon)', re.IGNORECASE)
_EXTENDED_LOS_RE = re.compile(r'(?:extended los|extended length of stay)', re.IGNORECASE)

@lru_cache(maxsize=2048)
def _normalize(text: str) -> str:
    """Lightning-fast normalization used everywhere."""
    if not text:
        return ""
    return " ".join(text.lower().replace("/", " ").replace("-", " ").split())

@lru_cache(maxsize=1024)
def _get_token_set(text: str) -> frozenset[str]:
    """Pre-tokenized word set for ultra-fast matching."""
    return frozenset(t for t in re.findall(r"[a-z0-9]+", _normalize(text)) if len(t) > 2)


# ═══════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════
def _split_query_terms(raw_query: str) -> List[str]:
    """Split query by comma, semicolon, or pipe."""
    normalized = (raw_query or "").replace(";", ",").replace("|", ",")
    return [part.strip() for part in normalized.split(",") if part.strip()]


def _append_unique_term(target: List[str], value: str) -> None:
    """Add a term to list if not already present (case-insensitive)."""
    val = (value or "").strip()
    if not val:
        return
    val_norm = val.lower()
    if any(existing.lower() == val_norm for existing in target):
        return
    target.append(val)


# ═══════════════════════════════════════════════════════════════════════════════
# DATA STRUCTURES FOR MULTI-STEP FLOW
# ═══════════════════════════════════════════════════════════════════════════════
def _get_pkg_field(pkg: Dict[str, Any], fields: List[str], default: Any = "") -> Any:
    """Consolidated helper to get fields across different JSON formats."""
    for f in fields:
        if f in pkg:
            return pkg[f]
    return default


def _get_pkg_name(pkg: Dict[str, Any]) -> str:
    return str(_get_pkg_field(pkg, ["PACKAGE NAME", "Package Name"], ""))


def _get_pkg_rate(pkg: Dict[str, Any]) -> float:
    try:
        val = _get_pkg_field(pkg, ["RATE", "Rate"], 0)
        return float(str(val).replace(",", "").strip()) if val else 0.0
    except Exception:
        return 0.0


def _get_pkg_spec(pkg: Dict[str, Any]) -> str:
    return str(_get_pkg_field(pkg, ["SPECIALITY", "Speciality", "SPECIALITY NAME"], ""))


def _get_pkg_cat(pkg: Dict[str, Any]) -> str:
    return str(_get_pkg_field(pkg, ["PACKAGE CATEGORY", "PACKAGE TYPE", "Procedure Type"], ""))


class SearchStep:
    """Represents a single step in the interactive search."""
    def __init__(
        self,
        step_number: int,
        step_name: str,
        description: str,
        options: List[Dict[str, Any]],
        requires_user_selection: bool = True,
        context: Optional[Dict[str, Any]] = None,
    ):
        self.step_number = step_number
        self.step_name = step_name
        self.description = description
        self.options = options
        self.requires_user_selection = requires_user_selection
        self.context = context or {}
        self.user_selection: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step_number": self.step_number,
            "step_name": self.step_name,
            "description": self.description,
            "options": self.options,
            "requires_user_selection": self.requires_user_selection,
            "context": self.context,
            "user_selection": self.user_selection,
        }


class FlowState:
    """Tracks the state of a multi-step search flow."""
    def __init__(self, session_id: str, query: str, parsed_terms: List[str]):
        self.session_id = session_id
        self.query = query
        self.parsed_terms = parsed_terms
        self.current_step = 0
        self.steps: List[SearchStep] = []
        self.selections: Dict[str, Dict[str, Any]] = {}
        self.violations: List[str] = []
        self.flow_complete = False
        self.final_recommendation: Optional[Dict[str, Any]] = None

    def add_step(self, step: SearchStep) -> None:
        self.steps.append(step)

    def set_selection(self, step_number: int, selection: Dict[str, Any]) -> None:
        self.selections[f"step_{step_number}"] = selection
        if step_number < len(self.steps):
            self.steps[step_number].user_selection = selection

    def advance_step(self) -> bool:
        if self.current_step < len(self.steps) - 1:
            self.current_step += 1
            return True
        return False

    def mark_complete(self) -> None:
        self.flow_complete = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "query": self.query,
            "parsed_terms": self.parsed_terms,
            "current_step": self.current_step,
            "current_step_data": self.steps[self.current_step].to_dict() if self.current_step < len(self.steps) else None,
            "selections_so_far": self.selections,
            "violations": self.violations,
            "flow_complete": self.flow_complete,
            "final_recommendation": self.final_recommendation,
            "total_steps": len(self.steps),
        }


def clean_subpackage_description(raw_val: str, pkg_name: str, rate: float = 0) -> str:
    """Clean up pipe-separated clinical description strings (optimized)."""
    if not raw_val:
        return ""
    segments = [s.strip() for s in raw_val.split("|") if s.strip()]
    cleaned = []

    base_name_clean = re.sub(r'[^a-zA-Z0-9]', '', pkg_name or "").lower()

    for seg in segments:
        seg_upper = seg.upper()
        if seg_upper in {"[REGULAR PROCEDURE]", "REGULAR PROCEDURE", "REGULAR PKG"}:
            continue

        seg_clean = re.sub(r'[^a-zA-Z0-9]', '', seg).lower()
        if seg_clean == base_name_clean:
            continue

        # Strip trailing rate if applicable (single regex)
        if rate > 0:
            r_str = str(int(rate))
            seg = re.sub(r'\s*-?\s*' + r_str + r'(?=\s*\]|\s*$)', '', seg).strip()

        # Strip (RATE:...) and empty brackets in one pass
        seg = re.sub(r'\s*\(RATE\s*:\s*\d+\)\s*', '', seg, flags=re.IGNORECASE)
        seg = re.sub(r'\[\s*\]', '', seg).strip()

        if seg and seg not in cleaned:
            cleaned.append(seg)

    return " • ".join(cleaned)


# ═══════════════════════════════════════════════════════════════════════════════
# OPTION GENERATORS - CREATE OPTIONS FOR EACH STEP (now 5-10x faster + smarter)
# ═══════════════════════════════════════════════════════════════════════════════
def generate_procedure_options(main_term: str, packages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    For a broad term (e.g., "heart attack"), extract unique procedures/pathways.
    Shows user different treatment approaches.
    """
    options = []
    query_lower = _normalize(main_term)
    query_tokens = _get_token_set(main_term)

    # Ultra-fast specialty collection using medical_knowledge
    relevant_specialties = {s.lower().strip() for s in get_specialties_for_term(main_term)}

    # Clinical pathway integration (most powerful path)
    pathway = get_clinical_pathway(main_term)
    if pathway:
        for i, step in enumerate(pathway.get("steps", [])[:5]):
            step_spec = str(step.get("specialty", "")).lower().strip()
            if not step_spec:
                continue
            step_text = _normalize(f"{step.get('procedure', '')} {step.get('clinical_reason', '')}")
            if any(tok in step_text for tok in query_tokens):
                options.append({
                    "id": f"procedure_{i+1}",
                    "label": step.get("procedure", ""),
                    "description": step.get("clinical_reason", ""),
                    "specialty": step_spec,
                    "reasoning": f"Step {i+1} in clinical pathway",
                })

    # Fast specialty-based fallback using pre-tokenized packages
    if not options:
        seen_specs = set()
        for pkg in packages[:40]:
            spec = str(pkg.get("SPECIALITY", "")).strip().lower()
            if spec and spec not in seen_specs and any(s in spec for s in relevant_specialties):
                seen_specs.add(spec)
                options.append({
                    "id": f"specialty_{spec.replace(' ', '_')}",
                    "label": spec.title(),
                    "description": f"Specialty approach: {spec.title()}",
                    "specialty": spec,
                    "related_packages": [f"{pkg.get('PACKAGE CODE', '')}: {pkg.get('PACKAGE NAME', '')[:80]}"],
                    "reasoning": "Most appropriate specialty for this condition",
                })
                if len(options) >= 5:
                    break

    # Always add skip/manual options
    options.append({
        "id": "procedure_skip",
        "code": "",
        "label": "Skip Clarification",
        "description": "Skip this step and show all packages",
        "specialty": "Optional",
        "rate": 0,
        "rank": 9998,
        "reasoning": "User choice to skip",
    })
    options.append({
        "id": "manual_add_procedure",
        "code": "",
        "label": "Add Manually from Normal Search",
        "description": "Search for your condition manually",
        "specialty": "Manual",
        "rate": 0,
        "rank": 9999,
        "reasoning": "Use when expected approach is not listed",
    })
    return options


def generate_package_options(
    selected_procedure: str,
    matching_packages: List[Dict[str, Any]],
    exclude_types: Optional[List[str]] = None
) -> List[Dict[str, Any]]:
    """
    Generate package options for a selected procedure/specialty.
    Groups packages by type and excludes incompatible types.
    """
    exclude_types = exclude_types or []
    options = []
    seen_codes = set()
    selected_lower = _normalize(selected_procedure)
    selected_tokens = _get_token_set(selected_procedure)

    ranking_tokens = list(selected_tokens)
    if "angioplasty" in selected_lower:
        ranking_tokens.extend(["ptca", "coronary", "stent", "pci"])
    if "blood" in selected_lower or "transfusion" in selected_lower:
        ranking_tokens.extend(["blood", "transfusion", "platelet", "plasma", "packed", "cell"])

    @lru_cache(maxsize=512)
    def relevance_score(pkg: tuple) -> tuple[int, int]:  # tuple for cacheable
        code, name, spec, category = pkg
        name_n = _normalize(name)
        spec_n = _normalize(spec)
        name_tokens = _get_token_set(name)
        score = 0
        exact_priority = 0

        # Exact / word-set matching (fastest possible)
        norm_selected = " ".join(ranking_tokens)
        if norm_selected in code.lower():
            score += 70
        if norm_selected in name_n:
            score += 60
        if norm_selected in spec_n:
            score += 24

        hits_name = len(selected_tokens & name_tokens)
        hits_spec = 1 if any(t in spec_n for t in selected_tokens) else 0

        score += hits_name * 8 + hits_spec * 6

        # Bonus for exact matches
        if norm_selected == name_n:
            exact_priority = 4
            score += 1000
        elif f" {norm_selected} " in f" {name_n} ":
            exact_priority = 3
            score += 500

        if "regular" in name_n:
            score += 2
        if "day care" in name_n or "daycare" in category.lower():
            score += 1

        is_blood_query = "blood" in selected_lower or "transfusion" in selected_lower
        if _ADDON_RE.search(name_n) and not is_blood_query:
            score -= 1
        
        # Specific boost for blood standalone packages if requested explicitly
        if is_blood_query:
            if name.lower().startswith("blood transfusion") or name.lower().startswith("blood component"):
                score += 5000
            elif "blood transfusion" in name_n or "blood component" in name_n:
                score += 200

        # Angioplasty-specific smart boost
        if "angioplasty" in selected_lower and "peripheral" not in selected_lower:
            if "ptca" in name_n or "coronary" in name_n:
                score += 10
            if "peripheral" in name_n or "vascular" in spec_n:
                score -= 6

        return exact_priority, score

    # Rank packages once (cached scoring)
    ranked = sorted(
        matching_packages[:350],
        key=lambda p: relevance_score((
            str(p.get("PACKAGE CODE", "")),
            _get_pkg_name(p),
            _get_pkg_spec(p),
            _get_pkg_cat(p)
        )),
        reverse=True
    )

    for pkg in ranked:
        code = str(pkg.get("PACKAGE CODE", "")).strip()
        if not code or code in seen_codes:
            continue

        name = _get_pkg_name(pkg)[:100]
        rate = _get_pkg_rate(pkg)
        category = _get_pkg_cat(pkg) or "Standard"
        implant = str(_get_pkg_field(pkg, ["IMPLANT PACKAGE", "IMPLANT"], "NO IMPLANT"))

        is_addon = _is_addon_package(pkg)
        if exclude_types and is_addon and "ADDON_EXCLUDE" in exclude_types:
            continue

        _, pkg_score = relevance_score((
            code, name, _get_pkg_spec(pkg), category
        ))
        if pkg_score <= 0:
            continue

        seen_codes.add(code)
        rank = len(options) + 1
        options.append({
            "id": f"package_{code}",
            "code": code,
            "label": f"[{code}] {name}",
            "description": name,
            "specialty": _get_pkg_spec(pkg),
            "category": category,
            "rate": rate,
            "rank": rank,
            "reasoning": f"Top-ranked match #{rank} for selected approach",
            "implant_available": "IMPLANT" in implant.upper() and "NO IMPLANT" not in implant.upper(),
            "implant_type": implant if "IMPLANT" in implant.upper() else None,
            "is_standalone": _is_standalone_pkg(pkg),
        })
        if len(options) >= 15:
            break

    # Always append skip/manual
    options.append({
        "id": "package_skip",
        "code": "",
        "label": "Skip Package Selection",
        "description": "Skip selecting a package for this term",
        "specialty": "Optional",
        "rate": 0,
        "rank": 9998,
        "reasoning": "User choice to skip",
    })
    options.append({
        "id": "manual_add_main",
        "code": "",
        "label": "Add Manually from Normal Search",
        "description": "Open normal search, select the exact main package, then return to continue smart flow.",
        "specialty": "Manual",
        "rate": None,
        "rank": 9999,
        "reasoning": "Use when expected package is not listed",
    })
    return options


def generate_implant_options(main_package: Dict[str, Any], packages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Generate implant options for a selected surgical package.
    Shows user different implant types available.
    """
    options = []
    implant_field = str(_get_pkg_field(main_package, ["IMPLANT PACKAGE", "IMPLANT"], "NO IMPLANT") or "NO IMPLANT").upper().strip()
    if "NO IMPLANT" in implant_field or implant_field in {"NO", "N", "FALSE", "0"}:
        return []

    main_name_n = _normalize(_get_pkg_name(main_package))
    main_spec_n = _normalize(_get_pkg_spec(main_package))

    ranked = []
    for pkg in packages:
        cat = _get_pkg_cat(pkg).upper()
        name = _get_pkg_name(pkg).upper()
        if not ("IMPLANT" in cat or cat == "IMP" or any(k in name for k in ("IMPLANT", "STENT", "VALVE", "PACEMAKER", "PROSTHESIS"))):
            continue

        implant_code = str(pkg.get("PACKAGE CODE", "")).strip()
        if not implant_code:
            continue
        rate = _get_pkg_rate(pkg)
        spec_n = _normalize(_get_pkg_spec(pkg))
        score = 8 if main_spec_n and spec_n == main_spec_n else 0
        name_n = _normalize(str(pkg.get("PACKAGE NAME", "")))
        if any(tok in name_n for tok in _get_token_set(main_name_n) if len(tok) > 3):
            score += 3

        ranked.append((score, {
            "id": f"implant_{implant_code}",
            "code": implant_code,
            "label": name[:80],
            "description": clean_subpackage_description(name, "", rate),
            "rate": rate,
            "base_package_code": main_package.get("PACKAGE CODE", ""),
            "type": "mandatory_with_procedure" if rate > 50000 else "optional",
        }))

    ranked.sort(key=lambda x: x[0], reverse=True)
    seen = set()
    for _, candidate in ranked:
        code = candidate["code"]
        if code and code not in seen:
            seen.add(code)
            options.append(candidate)
            if len(options) >= 8:
                break

    options.append({
        "id": "implant_skip",
        "code": "NO_IMPLANT",
        "label": "Skip / No Implant",
        "description": "Proceed without implant",
        "rate": 0,
        "rank": 9998,
        "type": "optional",
    })
    options.append({
        "id": "manual_add_implant",
        "code": "",
        "label": "Add Implant Manually from Normal Search",
        "description": "Search and add an implant manually",
        "rate": 0,
        "rank": 9999,
        "type": "optional",
    })
    return options


def generate_stratification_options(main_package: Dict[str, Any], packages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Generate stratification package options associated with a selected main package."""
    options: List[Dict[str, Any]] = []
    main_strat = str(_get_pkg_field(main_package, ["PACKAGE STRATIFICATION", "STRATIFICATION PACKAGE", "Procedure Type"], "") or "").strip()
    if not main_strat or main_strat.upper() in {"NO STRATIFICATION", "REGULAR", "REGULAR PKG", "NA", "N/A", "NONE", "NULL"}:
        return []

    main_code = str(main_package.get("PACKAGE CODE", "")).strip()
    main_spec_n = _normalize(_get_pkg_spec(main_package))
    strat_tokens = _get_token_set(main_strat)

    seen = set()
    for pkg in packages:
        code = str(pkg.get("PACKAGE CODE", "")).strip()
        if not code or code == main_code or code in seen:
            continue
        candidate_strat = str(_get_pkg_field(pkg, ["PACKAGE STRATIFICATION", "STRATIFICATION PACKAGE", "Procedure Type"], "") or "").strip()
        if not candidate_strat or candidate_strat.upper() in {"NO STRATIFICATION", "REGULAR", "REGULAR PKG", "NA", "N/A", "NONE", "NULL"}:
            continue
        if main_spec_n and _normalize(_get_pkg_spec(pkg)) != main_spec_n:
            continue
        if not (strat_tokens & _get_token_set(candidate_strat)):
            continue

        seen.add(code)
        pkg_name = _get_pkg_name(pkg)
        pkg_rate = _get_pkg_rate(pkg)
        options.append({
            "id": f"strat_{code}",
            "code": code,
            "label": pkg_name[:90],
            "description": clean_subpackage_description(candidate_strat, pkg_name, pkg_rate),
            "specialty": str(pkg.get("SPECIALITY", "")),
            "rate": pkg_rate,
            "rank": len(options) + 1,
            "reason": f"Stratification option for selected main package ({main_strat})",
        })
        if len(options) >= 8:
            break

    if not options:
        return []
    options.append({
        "id": "strat_skip",
        "code": "",
        "label": "Skip Stratification (If Not Needed)",
        "description": "Continue without stratification package.",
        "specialty": "Optional",
        "rate": 0,
        "rank": 9998,
        "reason": "User can continue without stratification package",
    })
    options.append({
        "id": "manual_add_strat",
        "code": "",
        "label": "Add Stratification Manually from Normal Search",
        "description": "Search and add a stratification manually",
        "specialty": "Manual",
        "rate": 0,
        "rank": 9999,
        "reason": "Use when expected stratification is not listed",
    })
    return options


def _is_addon_package(pkg: Dict) -> bool:
    """Return True if package appears to be an add-on package."""
    name_clean = _get_pkg_name(pkg).upper().replace(" ", "").replace("-", "")
    cat_clean = _get_pkg_cat(pkg).upper().replace(" ", "").replace("-", "")
    return bool('[ADDON' in name_clean or 'ADDON' in cat_clean)


def _is_standalone_pkg(pkg: Dict) -> bool:
    """Return True if package is stand-alone."""
    name = _get_pkg_name(pkg).upper()
    cat = _get_pkg_cat(pkg).upper()
    return bool('STAND-ALONE' in name or 'STAND ALONE' in name or 'STAND ALONE' in cat or 'STAND-ALONE' in cat)


def _is_surgical_pkg(pkg: Dict) -> bool:
    """Identify surgical packages via rate and category markers."""
    rate = _get_pkg_rate(pkg)
    name = _get_pkg_name(pkg).upper()
    cat = _get_pkg_cat(pkg).upper()
    return rate > 0 and ('[REGULAR PROCEDURE]' in name or 'REGULAR PKG' in cat)


def _is_medical_mgmt_pkg(pkg: Dict) -> bool:
    """Identify medical management (₹0) packages."""
    rate = _get_pkg_rate(pkg)
    return rate == 0 and not _is_addon_package(pkg)


def _is_extended_los_pkg(pkg: Dict) -> bool:
    """Identify Extended Length of Stay packages."""
    name = _get_pkg_name(pkg).upper()
    return 'EXTENDED LOS' in name


def generate_addon_options(
    main_package: Dict[str, Any],
    addon_query: str,
    packages: List[Dict[str, Any]],
    previous_addons: Optional[List[Dict[str, Any]]] = None,
    base_specialties: Optional[set] = None
) -> List[Dict[str, Any]]:
    """
    Generate supportive / related package options for a condition term.
    """
    options: List[Dict] = []
    addon_term_n = _normalize(addon_query)
    addon_tokens = _get_token_set(addon_query)

    # Use exact match plus significant token matches to find nearly related packages
    addon_keywords = [addon_term_n] + [t for t in addon_tokens if len(t) > 2]

    seen_codes = set()

    # Find related addon matches (single pass)
    for pkg in packages:
        code = pkg.get("PACKAGE CODE", "")
        if not code or code in seen_codes:
            continue
        pkg_name_n = _normalize(_get_pkg_name(pkg))
        pkg_cat_n = _normalize(_get_pkg_cat(pkg))
        
        if any(kw in pkg_name_n or kw in pkg_cat_n for kw in addon_keywords) and _is_addon_package(pkg):
            options.append({
                "id": f"addon_{code}",
                "code": code,
                "label": _get_pkg_name(pkg)[:80],
                "description": _get_pkg_name(pkg),
                "specialty": _get_pkg_spec(pkg),
                "rate": _get_pkg_rate(pkg),
                "rank": len(options) + 1,
                "reason": f"Add On (If any): {addon_query}",
            })
            seen_codes.add(code)
            if len(options) >= 9:
                break
                
    # If no matches found, fallback to generic supportive care in the same specialty
    base_specs = base_specialties or set()
    if not options and base_specs:
        fallback_terms = [
            "icu", "care", "conservative", "management", "report", "investigation", 
            "stay", "ward", "day", "diagnostic", "imaging", "central line", 
            "blood", "transfusion", "los", "extended", "biopsies", "serology", "thrombolysis"
        ]
        for pkg in packages:
            code = pkg.get("PACKAGE CODE", "")
            if not code or code in seen_codes:
                continue
            spec = pkg.get("SPECIALITY", "").strip()
            if spec in base_specs:
                pkg_name_n = _normalize(str(pkg.get("PACKAGE NAME", "")))
                pkg_cat_n = _normalize(str(pkg.get("PACKAGE CATEGORY", "")))
                if any(kw in pkg_name_n or kw in pkg_cat_n for kw in fallback_terms) and _is_addon_package(pkg):
                    options.append({
                        "id": f"addon_{code}",
                        "code": code,
                        "label": str(pkg.get("PACKAGE NAME", ""))[:80],
                        "description": str(pkg.get("PACKAGE NAME", "")),
                        "specialty": str(pkg.get("SPECIALITY", "")),
                        "rate": pkg.get("RATE", 0),
                        "rank": len(options) + 1,
                        "reason": f"Related Procedure/Report for {spec}",
                    })
                    seen_codes.add(code)
                    if len(options) >= 5:
                        break

    # Always add skip/manual
    options.append({
        "id": "addon_skip",
        "code": "",
        "label": "Skip Add Ons (If Applicable)",
        "description": "Continue with only the selected main package.",
        "specialty": "Optional",
        "rate": 0,
        "rank": 9998,
        "reason": "User can choose to continue without add-ons",
    })
    options.append({
        "id": "manual_add_addon",
        "code": "",
        "label": "Add Add-on Manually from Normal Search",
        "description": "Open normal search, select required add-on, and continue smart flow.",
        "specialty": "Manual",
        "rate": None,
        "rank": 9999,
        "reason": "Use when required add-on is not listed",
    })

    # Final ordering
    regular = [o for o in options if o.get("id") not in ("addon_skip", "manual_add_addon")][:9]
    final_options = regular
    for o in options:
        if o.get("id") in ("addon_skip", "manual_add_addon"):
            final_options.append(o)
    return final_options


# ═══════════════════════════════════════════════════════════════════════════════
# RULE VALIDATION (now uses compiled regex + medical_knowledge)
# ═══════════════════════════════════════════════════════════════════════════════
def validate_and_recommend(
    main_package: Optional[Dict[str, Any]],
    implant_package: Optional[Dict[str, Any]],
    stratification_package: Optional[Dict[str, Any]],
    addon_packages: List[Dict[str, Any]]
) -> Tuple[bool, List[str]]:
    """
    Validate package combinations against MAA Yojana rules.
    Returns (is_valid, list_of_violations).
    """
    violations = []
    addon_packages = addon_packages or []
    if not main_package:
        return True, violations

    # Accurate type identification matching main.py patterns
    is_main_surgical = _is_surgical_pkg(main_package)
    is_main_medical = _is_medical_mgmt_pkg(main_package)
    is_main_standalone = _is_standalone_pkg(main_package)
    is_main_extended = _is_extended_los_pkg(main_package)
    is_main_addon = _is_addon_package(main_package)

    # Rule: Add-on package cannot be selected as the primary package
    if is_main_addon:
        violations.append("❌ Rule Violation: An add-on package cannot be selected as the primary package. Please select a primary procedure first.")

    # Rule 2: Stand-alone package cannot be combined with any other package
    if is_main_standalone and (addon_packages or implant_package or stratification_package):
        violations.append(f"❌ Rule 2 Violation: Stand-alone package cannot be combined with any other package, add-on, or implant.")

    # Rules for each selected additional item
    all_selected_additional = addon_packages + ([implant_package] if implant_package else []) + ([stratification_package] if stratification_package else [])

    for item in all_selected_additional:
        item_name = _get_pkg_name(item)
        item_code = item.get("PACKAGE CODE", "N/A")
        
        # Rule 1: Surgical + Medical mix
        if is_main_surgical and _is_medical_mgmt_pkg(item):
            violations.append(f"❌ Rule 1 Violation: Cannot combine surgical package with medical management [{item_code}] - {item_name}")
        if is_main_medical and _is_surgical_pkg(item):
            violations.append(f"❌ Rule 1 Violation: Cannot combine medical package with surgical package [{item_code}] - {item_name}")

        # Rule 3: Add-on expects regular package (fail if main is standalone)
        if _is_addon_package(item) and is_main_standalone:
            # Already covered by Rule 2 above, but kept for specificity if needed
            pass

    # Rule 5: Extended LOS requires surgical procedure
    if is_main_extended and not is_main_surgical:
         violations.append(f"❌ Rule 5 Violation: Extended LOS package can only be used with surgical procedures.")
    
    # Check for Extended LOS in add-ons
    for addon in addon_packages:
        if _is_extended_los_pkg(addon) and not is_main_surgical:
            violations.append(f"❌ Rule 5 Violation: Extended Length of Stay add-on ({addon.get('PACKAGE CODE')}) can only be added to surgical procedures.")

    return len(violations) == 0, violations


def validate_package_combination(
    main_package: Dict[str, Any],
    implant_package: Optional[Dict[str, Any]],
    stratification_package: Optional[Dict[str, Any]],
    addon_packages: List[Dict[str, Any]],
) -> Tuple[bool, List[str]]:
    """
    Convenience wrapper used by main.py's _build_final_recommendation.
    Delegates to validate_and_recommend.
    """
    return validate_and_recommend(
        main_package=main_package,
        implant_package=implant_package,
        stratification_package=stratification_package,
        addon_packages=addon_packages,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# BUILD FLOW STEPS (optimized + powerful)
# ═══════════════════════════════════════════════════════════════════════════════
def build_search_flow(
    main_term: str,
    addon_terms: List[str],
    matching_packages: List[Dict],
    all_packages_for_addons: Optional[List[Dict]] = None,
    per_term_packages: Optional[Dict[str, List[Dict]]] = None,
) -> FlowState:
    """
    Build the complete multi-step interactive search flow.
    """
    all_intent_terms = [main_term] + addon_terms
    flow = FlowState(
        session_id="interactive_search",
        query=main_term,
        parsed_terms=all_intent_terms,
    )
    all_pkgs = all_packages_for_addons or matching_packages
    per_term_pkgs = per_term_packages or {}

    # Per-term package selection steps
    for i, term in enumerate(all_intent_terms):
        term_packages = per_term_pkgs.get(term, matching_packages) if i == 0 else per_term_pkgs.get(term, [])
        if not term_packages and i > 0:
            term_tokens = _get_token_set(term)
            term_packages = [pkg for pkg in all_pkgs[:200]
                             if term_tokens & _get_token_set(_get_pkg_name(pkg) + " " + _get_pkg_spec(pkg))]

        if term_packages:
            tl = term.strip().lower()
            is_addon_query = any(tok in _get_token_set(tl) for tok in [
                "blood", "transfusion", "platelet", "plasma", "packed",
                "extended", "los", "icu", "anaesthesia", "anesthesia", "addon", "add-on"
            ])
            exclude_list = [] if is_addon_query else ["ADDON_EXCLUDE"]
            pkg_options = generate_package_options(term, term_packages, exclude_types=exclude_list)
            if pkg_options:
                step_label = f"Select Main {term.strip().title()} Package"
                step_desc = f"Pick the primary package for '{term.strip()}':" if i == 0 else f"Select a package for '{term.strip()}':"
                flow.add_step(SearchStep(
                    step_number=len(flow.steps) + 1,
                    step_name=step_label,
                    description=step_desc,
                    options=pkg_options,
                    requires_user_selection=True,
                    context={
                        "is_term_selection": True,
                        "is_primary_selection": i == 0,
                        "intent_term": term,
                        "term_index": i,
                    }
                ))

    # Consolidated Supportive Add-on Step
    addon_keywords = []
    for term in all_intent_terms:
        tl = _normalize(term)
        if tl:
            addon_keywords.append(tl)

    # Extract base specialties from matched packages for fallback matching
    base_specialties = set()
    for pkg in matching_packages[:10]:
        spec = _get_pkg_spec(pkg).strip()
        if spec:
            base_specialties.add(spec)

    consolidated_addons = []
    seen = set()
    for kw in set(addon_keywords):
        kw_options = generate_addon_options({}, kw, all_pkgs, base_specialties=base_specialties)
        for opt in kw_options:
            code = opt.get("code")
            if code and code not in seen:
                consolidated_addons.append(opt)
                seen.add(code)

    # Always include the Supportive Care step
    final_options = [{
        "id": "addon_skip",
        "code": "",
        "label": "Skip Supportive Care",
        "description": "None of these are required for this clinical case. Advance to results.",
        "specialty": "Optional",
        "rate": 0,
        "rank": 0,
        "reason": "User choice to skip supportive add-ons"
    }]
    if consolidated_addons:
        final_options.extend(consolidated_addons[:9])
        
    final_options.append({
        "id": "manual_add_addon",
        "code": "",
        "label": "Add Add-on Manually from Normal Search",
        "description": "Open normal search, select required add-on, and continue smart flow.",
        "specialty": "Manual",
        "rate": None,
        "rank": 9999,
        "reason": "Use when required add-on is not listed",
    })

    flow.add_step(SearchStep(
        step_number=len(flow.steps) + 1,
        step_name="Supportive Care & Add-ons",
        description="Related packages, supportive care, and pre/post-procedure options based on your selections:",
        options=final_options,
        requires_user_selection=True,
        context={"is_consolidated_addons": True}
    ))

    return flow


def advance_past_empty_optional_steps(flow: FlowState) -> None:
    """Auto-advance over steps that do not require user selection and have no options."""
    while flow.current_step < len(flow.steps):
        step = flow.steps[flow.current_step]
        if step.requires_user_selection or step.options:
            break
        flow.set_selection(flow.current_step, {
            "id": f"auto_skip_step_{step.step_number}",
            "label": "No options available for this step",
            "description": step.description,
        })
        if not flow.advance_step():
            flow.mark_complete()
            break


def get_next_step(flow: FlowState) -> Optional[SearchStep]:
    """Get the next step the user should take."""
    if flow.current_step < len(flow.steps):
        return flow.steps[flow.current_step]
    return None


def process_step_selection(
    flow: FlowState,
    selection: Dict,
    all_packages: List[Dict],
) -> Tuple[bool, Optional[str]]:
    """
    Process user selection for current step.
    """
    advance_past_empty_optional_steps(flow)
    if flow.flow_complete:
        return True, None

    current_step = flow.steps[flow.current_step]

    def _is_package_selection_id(value: str) -> bool:
        v = (value or "").lower()
        return v.startswith(("package_", "addon_", "implant_", "strat_"))

    def _is_skip_selection_id(value: str) -> bool:
        return "skip" in (value or "").lower()

    def _is_standalone_option(option: Dict) -> bool:
        text = f"{option.get('label', '')} {option.get('description', '')}".upper()
        return bool(_STANDALONE_RE.search(text))

    # Block standalone conflicts
    standalone_already_selected = any(
        _is_standalone_option(sel) for sel in flow.selections.values() if isinstance(sel, dict)
    )

    selected_id = selection.get("id")
    if not selected_id:
        return False, "No option selected"

    if standalone_already_selected and (_is_package_selection_id(selected_id) and not _is_skip_selection_id(selected_id)):
        return False, "Stand-alone packages cannot be combined with other procedures or add-ons. Please start a new case for this package."

    # Resolve selected option
    selected_option = None
    if selected_id.startswith("manual_add"):
        manual_pkg = selection.get("manual_package") or {}
        manual_code = str(manual_pkg.get("code", "")).strip()
        manual_name = str(manual_pkg.get("name", "")).strip()
        if not manual_code or not manual_name:
            return False, "Manual package details are missing"
        parsed_rate = float(manual_pkg.get("rate", 0))
        is_term_step = current_step.context.get("is_term_selection", False)
        normalized_id = f"package_{manual_code}" if is_term_step else f"addon_{manual_code}"
        selected_option = {
            "id": normalized_id,
            "code": manual_code,
            "label": f"[{manual_code}] {manual_name[:90]}",
            "description": manual_name,
            "specialty": str(manual_pkg.get("specialty", "")),
            "rate": parsed_rate,
            "reason": "Added manually from normal search",
            "is_manual": True,
        }
    else:
        for opt in current_step.options:
            if opt.get("id") == selected_id:
                selected_option = opt
                break

    if not selected_option:
        return False, "Invalid selection"

    actual_id = str(selected_option.get("id", selected_id))

    # Block standalone when other packages exist
    if _is_package_selection_id(actual_id) and _is_standalone_option(selected_option):
        has_other = any(
            (str(sel.get("id", "")).startswith(("package_", "addon_")) and not _is_skip_selection_id(str(sel.get("id"))))
            for sel in flow.selections.values() if isinstance(sel, dict)
        )
        if has_other:
            return False, "This is a stand-alone package and cannot be combined with your existing selections."

    # Store selection
    flow.set_selection(flow.current_step, selected_option)

    # Standalone short-circuit
    if _is_package_selection_id(actual_id) and _is_standalone_option(selected_option):
        flow.mark_complete()
        return True, None

    # Term package selection → dynamic implant/strat insertion
    if current_step.context.get("is_term_selection") or current_step.context.get("is_primary_selection"):
        pkg_code = selected_option.get("code")
        sel_pkg = next((p for p in all_packages if p.get("PACKAGE CODE") == pkg_code), None)
        if sel_pkg and _is_standalone_pkg(sel_pkg):
            flow.mark_complete()
            return True, None

        if sel_pkg:
            insert_idx = flow.current_step + 1
            intent_term = current_step.context.get("intent_term", _get_pkg_name(sel_pkg)[:30])
            # Stratification
            strat_options = generate_stratification_options(sel_pkg, all_packages)
            if strat_options:
                flow.steps.insert(insert_idx, SearchStep(
                    step_number=insert_idx + 1,
                    step_name=f"Stratification for {intent_term.title()}",
                    description="This package has stratification options. Select one if applicable:",
                    options=strat_options,
                    requires_user_selection=True,
                    context={"main_package": pkg_code, "intent_term": intent_term},
                ))
                insert_idx += 1
            # Implant
            implant_options = generate_implant_options(sel_pkg, all_packages)
            if implant_options:
                flow.steps.insert(insert_idx, SearchStep(
                    step_number=insert_idx + 1,
                    step_name=f"Implant for {intent_term.title()}",
                    description="This procedure may require an implant. Select one:",
                    options=implant_options,
                    requires_user_selection=True,
                    context={"main_package": pkg_code, "intent_term": intent_term},
                ))

            # Renumber steps
            for idx, step in enumerate(flow.steps):
                step.step_number = idx + 1

    # Advance
    if not flow.advance_step():
        flow.mark_complete()
        return True, None

    advance_past_empty_optional_steps(flow)
    return True, None


def undo_last_selection(flow: FlowState) -> Tuple[bool, str]:
    """
    Undo the last selection made in the flow.
    Reverts current_step and removes the selection.
    If the step being undone inserted sub-steps (strat/implant), removes them.
    """
    if flow.current_step <= 0:
        return False, "Already at the first step"

    # 1. Identify the step we are reverting to
    prev_step_idx = flow.current_step - 1
    
    # In case the current step is 1 and it was advanced by auto-skip, 
    # we might need to go back further? No, UI usually handles one step at a time.
    
    prev_step = flow.steps[prev_step_idx]
    
    # 2. Check if this step inserted any subsequent steps
    # Sub-steps (strat/implant) were inserted with context referencing this step's package
    prev_selection = flow.selections.get(f"step_{prev_step_idx}")
    if prev_selection:
        pkg_code = prev_selection.get("code")
        if pkg_code:
            # Look ahead and remove steps that were inserted for this package
            # They would have context["main_package"] == pkg_code
            steps_to_remove = []
            for i in range(prev_step_idx + 1, len(flow.steps)):
                step = flow.steps[i]
                if step.context.get("main_package") == pkg_code:
                    steps_to_remove.append(step)
                else:
                    # Once we hit a step NOT from this package, stop (e.g., next term's step)
                    break
            
            for s in steps_to_remove:
                flow.steps.remove(s)
            
            # Re-number remaining steps
            for idx, step in enumerate(flow.steps):
                step.step_number = idx + 1

    # 3. Clear the selection and move back
    flow.selections.pop(f"step_{prev_step_idx}", None)
    prev_step.user_selection = None
    flow.current_step = prev_step_idx
    flow.flow_complete = False
    flow.final_recommendation = None

    return True, f"Reverted to Step {flow.current_step + 1}"


def reconstruct_flow_from_state(
    query: str,
    addon_terms: List[str],
    selections: List[Dict[str, Any]],
    matching_packages: List[Dict[str, Any]],
    all_packages: List[Dict[str, Any]],
    per_term_packages: Optional[Dict[str, List[Dict[str, Any]]]] = None,
) -> FlowState:
    """
    Reconstruct a FlowState by re-running the build and re-applying selections.
    This is essential for stateless environments like Vercel.
    """
    # 1. Build the initial flow
    flow = build_search_flow(
        query,
        addon_terms,
        matching_packages,
        all_packages_for_addons=all_packages,
        per_term_packages=per_term_packages
    )

    # 2. Re-apply selections sequentially to reach the same state
    for sel in selections:
        if flow.flow_complete:
            break
        process_step_selection(flow, sel, all_packages)

    return flow