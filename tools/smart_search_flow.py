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

logger = logging.getLogger("smart_search_flow")


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
        }



def clean_subpackage_description(raw_val: str, pkg_name: str, rate: float = 0) -> str:
    """Clean up pipe-separated clinical description strings."""
    if not raw_val: return ""
    segments = [s.strip() for s in raw_val.split("|") if s.strip()]
    cleaned = []
    
    base_name_clean = re.sub(r'[^a-zA-Z0-9]', '', pkg_name or "").lower()
    
    for seg in segments:
        seg_upper = seg.upper()
        if seg_upper in ["[REGULAR PROCEDURE]", "REGULAR PROCEDURE", "REGULAR PKG"]:
            continue
            
        seg_clean = re.sub(r'[^a-zA-Z0-9]', '', seg).lower()
        if seg_clean == base_name_clean:
            continue
            
        # Strip trailing rate if applicable
        if rate > 0:
            r_str = str(int(rate))
            seg = re.sub(r'\s*-?\s*' + r_str + r'(?=\s*\]|\s*$)', '', seg).strip()
            
        # Strip (RATE:...)
        seg = re.sub(r'\s*\(RATE\s*:\s*\d+\)\s*', '', seg, flags=re.IGNORECASE).strip()
        
        # Strip empty brackets
        seg = re.sub(r'\[\s*\]', '', seg).strip()
        
        if seg and seg not in cleaned:
            cleaned.append(seg)
            
    return " • ".join(cleaned)

# ═══════════════════════════════════════════════════════════════════════════════
# OPTION GENERATORS - CREATE OPTIONS FOR EACH STEP
# ═══════════════════════════════════════════════════════════════════════════════

def generate_procedure_options(main_term: str, packages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    For a broad term (e.g., "heart attack"), extract unique procedures/pathways.
    Shows user different treatment approaches.
    """
    from .medical_knowledge import get_clinical_pathway, get_specialties_for_term

    options = []

    generic_terms = {"surgery", "surgical",
                     "procedure", "operation", "management"}
    query_lower = (main_term or "").lower().strip()
    query_tokens = [t for t in query_lower.replace(
        "/", " ").split() if len(t) > 2]

    relevant_specialties = {s.lower().strip()
                            for s in get_specialties_for_term(main_term)}
    for token in query_tokens:
        if token in generic_terms:
            continue
        for spec in get_specialties_for_term(token):
            relevant_specialties.add(spec.lower().strip())

    # Strong clinical anchors for high-signal terms.
    if any(k in query_lower for k in ["heart", "cardiac", "coronary", "cabg", "angioplasty"]):
        relevant_specialties.update(["cardiology", "cardio thoracic surgery"])
    if any(k in query_lower for k in ["brain", "neuro", "craniotomy", "stroke"]):
        relevant_specialties.update(["neurosurgery", "neurology"])

    def _normalize_words(value: str) -> str:
        cleaned = " ".join((value or "").lower().replace(
            "/", " ").replace("-", " ").split())
        return f" {cleaned} "

    def _is_relevant_specialty(spec: str) -> bool:
        spec_lower = (spec or "").lower().strip()
        if not spec_lower:
            return False
        spec_norm = _normalize_words(spec_lower)
        if not relevant_specialties:
            # Fallback: only keep specialties that share at least one non-generic token.
            filtered_tokens = [
                t for t in query_tokens if t not in generic_terms]
            if not filtered_tokens:
                return True
            return any(_normalize_words(tok) in spec_norm for tok in filtered_tokens)
        return any(
            _normalize_words(
                rel) in spec_norm or spec_norm in _normalize_words(rel)
            for rel in relevant_specialties
        )

    # Get clinical pathway
    pathway = get_clinical_pathway(main_term)
    if pathway and "steps" in pathway:
        for i, step in enumerate(pathway["steps"][:5]):  # Max 5 options
            step_spec = step.get("specialty", "")
            if not _is_relevant_specialty(step_spec):
                continue

            # Keep only pathway options that are textually related to the user query.
            step_text = f"{step.get('procedure', '')} {step.get('clinical_reason', '')}".lower(
            )
            filtered_tokens = [
                t for t in query_tokens if t not in generic_terms]
            if filtered_tokens and not any(tok in step_text for tok in filtered_tokens):
                continue

            options.append({
                "id": f"procedure_{i+1}",
                "label": step.get("procedure", ""),
                "description": step.get("clinical_reason", ""),
                "specialty": step_spec,
                "reasoning": f"Step {i+1} in clinical pathway",
            })

    # Extract unique specialties from packages
    specialties = set()
    specialty_packages: Dict[str, List[str]] = {}
    for pkg in packages[:40]:
        spec = str(pkg.get("SPECIALITY", "") or "").strip()
        if not spec:
            continue
        if not _is_relevant_specialty(spec):
            continue
        specialties.add(spec)
        if spec not in specialty_packages:
            specialty_packages[spec] = []
        specialty_packages[spec].append(
            f"{pkg.get('PACKAGE CODE', '')}: {pkg.get('PACKAGE NAME', '')[:80]}"
        )

    # Add specialty-based options if pathway options are missing
    if not options and specialties:
        for spec in list(specialties)[:5]:
            related_packages = specialty_packages.get(spec, [])
            options.append({
                "id": f"specialty_{spec.replace(' ', '_')}",
                "label": spec,
                "description": f"Specialty approach: {spec}",
                "specialty": spec,
                "related_packages": related_packages[:2],
                "reasoning": "Most appropriate specialty for this condition",
            })

    # If no specialty appears in prefiltered packages but knowledge mapping exists,
    # show mapped specialties as safe clinician-first options.
    if not options and relevant_specialties:
        for spec in list(sorted(relevant_specialties))[:5]:
            pretty_spec = spec.title()
            options.append({
                "id": f"specialty_{pretty_spec.replace(' ', '_')}",
                "label": pretty_spec,
                "description": f"Specialty approach: {pretty_spec}",
                "specialty": pretty_spec,
                "related_packages": [],
                "reasoning": "Mapped from clinical term",
            })

    # Normalize generic pathway labels (e.g., "Phase 3") into query-focused labels.
    normalized_main = (main_term or "").strip().title()
    non_generic_query_tokens = [
        t for t in query_tokens if t not in generic_terms]
    for option in options:
        label = str(option.get("label", "") or "")
        label_lower = label.lower()
        if "phase" in label_lower and not any(tok in label_lower for tok in non_generic_query_tokens):
            specialty = str(option.get("specialty", "") or "").strip()
            if specialty:
                option["label"] = f"{specialty} ({normalized_main})"
            else:
                option["label"] = f"{normalized_main} focused approach"

    if not options:
        return []

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

    selected_lower = (selected_procedure or "").lower()

    def relevance_score(pkg: Dict) -> Tuple[int, int]:
        name = str(pkg.get("PACKAGE NAME", "")).lower()
        spec = str(pkg.get("SPECIALITY", "")).lower()
        code = str(pkg.get("PACKAGE CODE", "")).lower()
        category = str(pkg.get("PACKAGE CATEGORY", "")).lower()
        score = 0
        exact_priority = 0

        name_tokens = re.findall(r"[a-z0-9]+", name)
        spec_tokens = re.findall(r"[a-z0-9]+", spec)
        code_tokens = re.findall(r"[a-z0-9]+", code)

        def has_token_match(term: str, tokens: List[str], text: str) -> bool:
            if not term:
                return False
            if " " in term:
                return term in text
            return any(tok == term or tok.startswith(term) for tok in tokens)

        ranking_tokens = [t for t in selected_lower.replace(
            "/", " ").split() if len(t) > 2]
        if "angioplasty" in selected_lower:
            ranking_tokens.extend(["ptca", "coronary", "stent", "pci"])

        normalized_selected = " ".join(ranking_tokens).strip()
        if normalized_selected:
            if normalized_selected in code:
                score += 70
            if normalized_selected in name:
                score += 60
            if normalized_selected in spec:
                score += 24

            if normalized_selected == code:
                exact_priority = 5
            elif normalized_selected == name:
                exact_priority = 4
            elif f" {normalized_selected} " in f" {name} ":
                exact_priority = 3
            elif f" {normalized_selected} " in f" {spec} ":
                exact_priority = 2

        hits_name = 0
        hits_code = 0
        hits_spec = 0

        for token in ranking_tokens:
            if has_token_match(token, name_tokens, name):
                score += 5
                hits_name += 1
            if has_token_match(token, spec_tokens, spec):
                score += 4
                hits_spec += 1
            if has_token_match(token, code_tokens, code):
                score += 3
                hits_code += 1

        if ranking_tokens:
            total = len(ranking_tokens)
            if hits_name == total:
                score += 30
            if hits_code == total:
                score += 35
            if len(ranking_tokens) == 1 and has_token_match(ranking_tokens[0], name_tokens, name):
                score += 18
                if any(tok == ranking_tokens[0] for tok in name_tokens):
                    exact_priority = max(exact_priority, 4)
            if len(ranking_tokens) == 1 and any(tok == ranking_tokens[0] for tok in code_tokens):
                exact_priority = max(exact_priority, 4)
            if (hits_name + hits_code + hits_spec) == 0:
                score -= 8

        if "regular" in name:
            score += 2
        if "day care" in name or "daycare" in category:
            score += 1
        if "add - on" in name or "add-on" in name:
            score -= 1

        if "angioplasty" in selected_lower and "peripheral" not in selected_lower:
            if has_token_match("ptca", name_tokens, name) or has_token_match("coronary", name_tokens, name):
                score += 10
            if has_token_match("peripheral", name_tokens, name) or has_token_match("vascular", spec_tokens, spec):
                score -= 6
        return exact_priority, score

    ranked_packages = sorted(
        matching_packages[:350], key=relevance_score, reverse=True)

    for pkg in ranked_packages:
        code = str(pkg.get("PACKAGE CODE", "")).strip()
        if not code or code in seen_codes:
            continue

        name = str(pkg.get("PACKAGE NAME", ""))[:100]
        rate = pkg.get("RATE", 0)
        implant = str(pkg.get("IMPLANT PACKAGE", "NO IMPLANT"))
        category = str(pkg.get("PACKAGE CATEGORY", "Standard"))
        is_addon = "ADD-ON" in category.upper() or "ADD ON" in category.upper()

        if exclude_types and is_addon and "ADDON_EXCLUDE" in exclude_types:
            continue

        # Skip packages with no relevance to the search term
        pkg_priority, pkg_score = relevance_score(pkg)
        if pkg_score <= 0 and pkg_priority == 0:
            continue

        seen_codes.add(code)
        rank = len(options) + 1
        options.append({
            "id": f"package_{code}",
            "code": code,
            "label": f"[{code}] {name}",
            "description": name,
            "specialty": pkg.get("SPECIALITY", ""),
            "category": category,
            "rate": rate,
            "rank": rank,
            "reasoning": f"Top-ranked match #{rank} for selected approach",
            "implant_available": "IMPLANT" in implant.upper() and "NO IMPLANT" not in implant.upper(),
            "implant_type": implant if "IMPLANT" in implant.upper() else None,
            "is_standalone": "STAND-ALONE" in str(pkg.get("PACKAGE NAME", "")).upper(),
        })

        if len(options) >= 15:
            break

    options = options[:15]
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

    def _implant_value(pkg: Dict) -> str:
        return str(pkg.get("IMPLANT PACKAGE", pkg.get("IMPLANT", "NO IMPLANT")) or "NO IMPLANT")

    implant_field = _implant_value(main_package)
    implant_upper = implant_field.upper().strip()
    requires_implant = (
        ("IMPLANT" in implant_upper and "NO IMPLANT" not in implant_upper)
        or implant_upper in {"YES", "Y", "TRUE", "1"}
    )
    explicitly_no_implant = implant_upper in {
        "NO", "N", "FALSE", "0", "NO IMPLANT"}
    if explicitly_no_implant or not requires_implant:
        return []  # No implants for this package

    main_name = str(main_package.get("PACKAGE NAME", "")).lower()
    main_spec = str(main_package.get("SPECIALITY", "")).lower().strip()

    def _is_implant_candidate(pkg: Dict) -> bool:
        cat = str(pkg.get("PACKAGE CATEGORY", pkg.get(
            "PACKAGE TYPE", ""))).upper().strip()
        name = str(pkg.get("PACKAGE NAME", "")).upper()
        if "IMPLANT" in cat or cat == "IMP":
            return True
        implant_keywords = ["IMPLANT", "STENT", "VALVE",
                            "PACEMAKER", "PROSTHESIS", "RING", "DEVICE"]
        return any(k in name for k in implant_keywords)

    # Find implant packages related to this procedure
    ranked_candidates = []
    for pkg in packages:
        if not _is_implant_candidate(pkg):
            continue

        implant_name = str(pkg.get("PACKAGE NAME", ""))
        implant_code = str(pkg.get("PACKAGE CODE", "")).strip()
        if not implant_code:
            continue
        rate = pkg.get("RATE", 0)
        spec = str(pkg.get("SPECIALITY", "")).lower().strip()

        score = 0
        if main_spec and spec and main_spec == spec:
            score += 8
        name_lower = implant_name.lower()
        if any(tok in name_lower for tok in re.findall(r"[a-z0-9]+", main_name) if len(tok) > 3):
            score += 3
        if "stent" in main_name and "stent" in name_lower:
            score += 4
        if "valve" in main_name and "valve" in name_lower:
            score += 4
        if "pacemaker" in main_name and "pacemaker" in name_lower:
            score += 4

        ranked_candidates.append((score, {
            "id": f"implant_{implant_code}",
            "code": implant_code,
            "label": implant_name[:80],
            "description": clean_subpackage_description(implant_name, "", rate),
            "rate": rate,
            "base_package_code": main_package.get("PACKAGE CODE", ""),
            "type": "mandatory_with_procedure" if (isinstance(rate, (int, float)) and rate > 50000) else "optional",
        }))

    ranked_candidates.sort(key=lambda x: x[0], reverse=True)
    seen_codes = set()
    for _, candidate in ranked_candidates:
        code = candidate.get("code", "")
        if not code or code in seen_codes:
            continue
        seen_codes.add(code)
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

    def _strat_value(pkg: Dict[str, Any]) -> str:
        return str(pkg.get("PACKAGE STRATIFICATION", pkg.get("STRATIFICATION PACKAGE", "")) or "").strip()

    main_strat = _strat_value(main_package)
    if not main_strat:
        return []

    blocked_values = {
        "NO STRATIFICATION",
        "REGULAR",
        "REGULAR PKG",
        "NA",
        "N/A",
        "NONE",
        "NULL",
    }
    if main_strat.upper() in blocked_values:
        return []

    main_code = str(main_package.get("PACKAGE CODE", "")).strip()
    main_spec = str(main_package.get("SPECIALITY", "")).strip().lower()
    strat_tokens = [
        t for t in re.findall(r"[a-z0-9]+", main_strat.lower())
        if t not in {"stratification", "package", "type", "regular"}
    ]

    seen_codes = set()
    for pkg in packages:
        code = str(pkg.get("PACKAGE CODE", "")).strip()
        if not code or code == main_code or code in seen_codes:
            continue

        candidate_strat = _strat_value(pkg)
        if not candidate_strat:
            continue
        if candidate_strat.upper() in blocked_values:
            continue

        candidate_spec = str(pkg.get("SPECIALITY", "")).strip().lower()
        if main_spec and candidate_spec and main_spec != candidate_spec:
            continue

        candidate_text = f"{candidate_strat} {pkg.get('PACKAGE NAME', '')}".lower(
        )
        if strat_tokens and not any(tok in candidate_text for tok in strat_tokens):
            continue

        pkg_name = str(pkg.get("PACKAGE NAME", ""))
        pkg_rate = pkg.get("RATE", 0)
        
        seen_codes.add(code)
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
    name_raw = str(pkg.get("PACKAGE NAME", "")).upper()
    cat_raw = str(pkg.get("PACKAGE CATEGORY", "")).upper()
    normalized = (name_raw + " " + cat_raw).replace(" ", "").replace("-", "")
    return "ADDON" in normalized


def generate_addon_options(
    main_package: Dict[str, Any],
    addon_query: str,
    packages: List[Dict[str, Any]],
    previous_addons: Optional[List[Dict[str, Any]]] = None
) -> List[Dict[str, Any]]:
    """
    Generate supportive / related package options for a condition term.

    - include_direct_matches=True: include term-matching packages (good for comma-separated terms)
    - include_direct_matches=False: show only clinically suggested supportive options
    - include_skip_option=True: append explicit user choice to skip supportive package
    """
    options: List[Dict] = []

    clinical_addons = {
        "anemia": ["blood transfusion", "iron", "iron sucrose"],
        "hemorrhage": ["blood transfusion", "plasma", "ffp"],
        "heart attack": ["thrombolysis", "stent", "angioplasty", "critical care"],
        "sepsis": ["critical care", "icu", "antibiotic"],
        "infection": ["antibiotic", "infection"],
        "icu": ["icu", "critical care"],
        "extended los": ["extended", "los"],
    }

    addon_term_lower = (addon_query or "").lower().strip()
    addon_keywords = [addon_term_lower] + \
        clinical_addons.get(addon_term_lower, [addon_term_lower])

    seen_codes = set()

    # Pass 1: direct term matches
    # The original code had `include_direct_matches` as a parameter, but the instruction removes it.
    # Assuming `include_direct_matches` should be True by default based on the original logic.
    # If this assumption is wrong, the user will need to provide further clarification.
    # For now, I'll keep the logic that was inside the `if include_direct_matches:` block.
    for pkg in packages:
        code = pkg.get("PACKAGE CODE", "")
        if not code or code in seen_codes:
            continue

        pkg_name = str(pkg.get("PACKAGE NAME", "")).lower()
        pkg_cat = str(pkg.get("PACKAGE CATEGORY", "")).lower()
        if addon_term_lower and (addon_term_lower in pkg_name or addon_term_lower in pkg_cat):
            options.append({
                "id": f"addon_{code}",
                "code": code,
                "label": str(pkg.get("PACKAGE NAME", ""))[:80],
                "description": str(pkg.get("PACKAGE NAME", "")),
                "specialty": str(pkg.get("SPECIALITY", "")),
                "rate": pkg.get("RATE", 0),
                "rank": len(options) + 1,
                "reason": f"Direct match for: {addon_query}",
            })
            seen_codes.add(code)
            if len(options) >= 8:
                break

    # Pass 2: clinically related supportive add-ons
    for keyword in addon_keywords:
        if not keyword:
            continue
        for pkg in packages:
            code = pkg.get("PACKAGE CODE", "")
            if not code or code in seen_codes:
                continue

            pkg_name = str(pkg.get("PACKAGE NAME", "")).lower()
            pkg_cat = str(pkg.get("PACKAGE CATEGORY", "")).lower()
            is_addon = _is_addon_package(pkg)
            keyword_match = keyword in pkg_name or keyword in pkg_cat

            if is_addon and keyword_match:
                options.append({
                    "id": f"addon_{code}",
                    "code": code,
                    "label": str(pkg.get("PACKAGE NAME", ""))[:80],
                    "description": str(pkg.get("PACKAGE NAME", "")),
                    "specialty": str(pkg.get("SPECIALITY", "")),
                    "rate": pkg.get("RATE", 0),
                    "rank": len(options) + 1,
                    "reason": f"Add On (If any): {addon_query}",
                })
                seen_codes.add(code)
                if len(options) >= 8:
                    break
        if len(options) >= 8:
            break

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

    # Keep skip and manual-add options available even when list is capped.
    skip_option = next((o for o in options if o.get("id") == "addon_skip"), None)
    manual_option = next((o for o in options if o.get("id") == "manual_add_addon"), None)
    
    # Take regular options (not skip, not manual) up to 9
    regular = [o for o in options if o.get("id") not in ("addon_skip", "manual_add_addon")][:9]
    
    final_options = regular
    if skip_option:
        final_options.append(skip_option)
    if manual_option:
        final_options.append(manual_option)
        
    return final_options


# ═══════════════════════════════════════════════════════════════════════════════
# RULE VALIDATION
# ═══════════════════════════════════════════════════════════════════════════════

def validate_and_recommend(
    main_package: Optional[Dict[str, Any]],
    implant_package: Optional[Dict[str, Any]],
    stratification_package: Optional[Dict[str, Any]],
    addon_packages: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Validate package combinations against MAA Yojana rules.
    Returns (is_valid, list_of_violations).
    """
    violations = []
    addon_packages = addon_packages or []

    main_code = main_package.get("PACKAGE CODE", "")
    main_name = main_package.get("PACKAGE NAME", "").upper()
    main_cat = main_package.get("PACKAGE CATEGORY", "").upper()

    # Rule 1: Surgical ≠ Medical management
    surgical_keywords = ["SURGICAL", "SURGERY", "OPERATIVE", "ECTOMY", "PLASTY", "OTOMY", "PROCEDURE"]
    medical_keywords = ["MEDICAL", "CONSERVATIVE", "NON SURGICAL", "NON-SURGICAL", "MANAGEMENT"]

    is_main_surgical = any(k in main_name for k in surgical_keywords) or any(k in main_cat for k in surgical_keywords)
    is_main_medical = any(k in main_name for k in medical_keywords) or any(k in main_cat for k in medical_keywords)

    for addon in addon_packages:
        addon_name = addon.get("PACKAGE NAME", "").upper()
        addon_code = addon.get("PACKAGE CODE", "")
        addon_cat = addon.get("PACKAGE CATEGORY", "").upper()

        is_addon_surgical = any(k in addon_name for k in surgical_keywords) or any(k in addon_cat for k in surgical_keywords)
        is_addon_medical = any(k in addon_name for k in medical_keywords) or any(k in addon_cat for k in medical_keywords)

        if is_main_surgical and is_addon_medical:
            violations.append(
                f"❌ Rule 1 Violation: Cannot combine surgical package [{main_code}] "
                f"with medical management [{addon_code}]"
            )
        if is_main_medical and is_addon_surgical:
            violations.append(
                f"❌ Rule 1 Violation: Cannot combine medical package [{main_code}] "
                f"with surgical package [{addon_code}]"
            )

    # Rule 2: Stand-alone packages cannot combine
    main_name = str(main_package.get("PACKAGE NAME", main_package.get("name", ""))).upper()
    main_category = str(main_package.get("PACKAGE CATEGORY", main_package.get("category", ""))).upper()
    main_strat = str(main_package.get("PACKAGE STRATIFICATION", main_package.get("stratification", ""))).upper()
    
    is_main_standalone = (
        "STAND-ALONE" in main_name or "STANDALONE" in main_name or "STAND ALONE" in main_name or
        "STAND-ALONE" in main_category or "STANDALONE" in main_category or "STAND ALONE" in main_category or
        "STAND-ALONE" in main_strat or "STANDALONE" in main_strat or "STAND ALONE" in main_strat
    )

    if is_main_standalone and addon_packages:
        violations.append(
            f"❌ Rule 2 Violation: Stand-alone package [{main_code}] "
            f"cannot be combined with {len(addon_packages)} add-on(s)"
        )

    # Rule 3: Add-ons must be with regular packages
    for addon in addon_packages:
        addon_name = addon.get("PACKAGE NAME", "").upper()
        addon_code = addon.get("PACKAGE CODE", "")
        is_addon_addon = "ADD-ON" in addon.get("PACKAGE CATEGORY", "").upper()

        # If addon is truly an add-on and main is not compatible
        if is_addon_addon and is_main_standalone:
            violations.append(
                f"❌ Rule 3 Violation: Add-on [{addon_code}] requires a regular package, but [{main_code}] is stand-alone"
            )

    # Rule 5: Extended LOS only with surgery
    is_main_extended = "EXTENDED" in main_name or "EXTENDED LOS" in main_cat

    if is_main_extended and not is_main_surgical:
        violations.append(
            f"❌ Rule 5 Violation: Extended LOS package [{main_code}] can only be used with surgical procedures"
        )

    return len(violations) == 0, violations


def validate_package_combination(
    main_package: Dict[str, Any],
    implant_package: Optional[Dict[str, Any]],
    addon_packages: List[Dict[str, Any]],
) -> Tuple[bool, List[str]]:
    """
    Convenience wrapper used by main.py's _build_final_recommendation.

    Delegates to validate_and_recommend with stratification_package=None.
    """
    return validate_and_recommend(
        main_package=main_package,
        implant_package=implant_package,
        stratification_package=None,
        addon_packages=addon_packages,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# BUILD FLOW STEPS
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

    New Flow (per-term):
      For each input term (x, y, z):
        1. (Optional) Clarify specialty/approach if term is broad
        2. Select package for this term
        3. (Dynamic) Implant / Stratification inserted after selection
      Then:
        4. Consolidated supportive add-on step
        5. Result → Package List

    Standalone short-circuit: if user selects a standalone package,
    skip all remaining terms + add-ons → go to result.
    """
    all_intent_terms = [main_term] + addon_terms
    flow = FlowState(
        session_id="interactive_search",
        query=main_term,
        parsed_terms=all_intent_terms,
    )

    all_pkgs = all_packages_for_addons or matching_packages
    per_term_pkgs = per_term_packages or {}

    # ── Per-term package selection steps ──────────────────────────────────
    for i, term in enumerate(all_intent_terms):
        term_label = term.strip().title()
        term_packages = per_term_pkgs.get(term, matching_packages) if i == 0 else per_term_pkgs.get(term, [])

        # If no per-term packages found, search from full cache
        if not term_packages and i > 0:
            term_lower = term.lower().strip()
            term_tokens = [t for t in re.findall(r"[a-z0-9]+", term_lower) if len(t) > 2]
            if term_tokens:
                for pkg in all_pkgs:
                    name = str(pkg.get("PACKAGE NAME", "")).lower()
                    spec = str(pkg.get("SPECIALITY", "")).lower()
                    code_str = str(pkg.get("PACKAGE CODE", "")).lower()
                    text = f"{name} {spec} {code_str}"
                    if any(tok in text for tok in term_tokens):
                        term_packages.append(pkg)
                term_packages = term_packages[:200]

        is_first_term = (i == 0)

        # Step A: Clarification (only for first term if broad)
        if is_first_term and term_packages:
            proc_options = generate_procedure_options(term, term_packages)
            if len(proc_options) > 1:
                flow.add_step(SearchStep(
                    step_number=len(flow.steps) + 1,
                    step_name=f"Clarify '{term_label}' Specialty/Approach",
                    description=f"Select the most appropriate clinical specialty for '{term_label}':",
                    options=proc_options,
                    requires_user_selection=True,
                    context={
                        "is_clarification": True,
                        "intent_term": term,
                        "term_index": i,
                    }
                ))

        # Step B: Package selection for this term
        if term_packages:
            pkg_options = generate_package_options(term, term_packages)
            if pkg_options:
                step_label = f"Select Main {term_label} Package"
                step_desc = (
                    f"Pick the primary package for '{term_label}':"
                    if is_first_term
                    else f"Select a package for '{term_label}':"
                )
                flow.add_step(SearchStep(
                    step_number=len(flow.steps) + 1,
                    step_name=step_label,
                    description=step_desc,
                    options=pkg_options,
                    requires_user_selection=True,
                    context={
                        "is_term_selection": True,
                        "is_primary_selection": is_first_term,
                        "intent_term": term,
                        "term_index": i,
                    }
                ))

    # ── Consolidated Supportive Add-on Step ─────────────────────────────
    clinical_addons = {
        "anemia": ["blood transfusion", "iron", "iron sucrose"],
        "hemorrhage": ["blood transfusion", "plasma", "ffp"],
        "heart attack": ["thrombolysis", "stent", "angioplasty", "critical care"],
        "sepsis": ["critical care", "icu", "antibiotic"],
        "infection": ["antibiotic", "infection"],
        "icu": ["icu", "critical care"],
        "extended los": ["extended", "los"],
    }

    addon_keywords: List[str] = []
    for term in all_intent_terms:
        tl = term.lower().strip()
        addon_keywords.extend(clinical_addons.get(tl, []))

    addon_source = all_pkgs
    consolidated_addons: List[Dict] = []
    seen_addon_codes: set = set()
    
    unique_kws = list(dict.fromkeys([k for k in addon_keywords if k]))
    if not unique_kws:
        # If no specific clinical addons were triggered, just use the original terms to find loose additive matches
        unique_kws = [t.lower().strip() for t in all_intent_terms]

    for kw in unique_kws:
        kw_options = generate_addon_options(
            {}, kw, addon_source
        )
        for opt in kw_options:
            code = opt.get("code")
            if code and code not in seen_addon_codes:
                consolidated_addons.append(opt)
                seen_addon_codes.add(code)
                
    # Always ensure we have the Add-on step even if we found no specific packages,
    # so the user has the option to "Add Manually" or "Skip", unless they chose a standalone package.
    if not consolidated_addons:
        dummy_options = generate_addon_options({}, "", [])
        for opt in dummy_options:
            consolidated_addons.append(opt)

    if consolidated_addons:
        filtered_addons: List[Dict] = []
        has_manual = False
        for opt in consolidated_addons:
            if opt.get("id") == "manual_add_addon":
                if not has_manual:
                    filtered_addons.append(opt)
                    has_manual = True
            else:
                filtered_addons.append(opt)

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

        for idx, opt in enumerate(filtered_addons):
            opt["rank"] = idx + 1
            final_options.append(opt)

        flow.add_step(SearchStep(
            step_number=len(flow.steps) + 1,
            step_name="Supportive Care & Add-ons",
            description="Based on your selected clinical cases, these extra care packages may be needed:",
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
    Returns (success, error_message).

    Handles:
    - Per-term package selection (is_term_selection context flag)
    - Standalone short-circuit → marks flow complete
    - Dynamic implant/strat step insertion after any term's package selection
    - Clarification step → refines next step's package options
    """
    advance_past_empty_optional_steps(flow)
    if flow.flow_complete:
        return True, None

    current_step = flow.steps[flow.current_step]

    def _is_package_selection_id(value: str) -> bool:
        v = (value or "").lower()
        return v.startswith("package_") or v.startswith("addon_") or v.startswith("implant_") or v.startswith("strat_")

    def _is_skip_selection_id(value: str) -> bool:
        return "skip" in (value or "").lower()

    def _is_standalone_option(option: Dict) -> bool:
        text = f"{option.get('label', '')} {option.get('description', '')}".upper()
        cat = str(option.get("category", "")).upper()
        return cat == "STAND ALONE" or "[STAND ALONE]" in text or "[STAND-ALONE]" in text or "[STAND- ALONE]" in text

    def _is_standalone_pkg(pkg: Dict) -> bool:
        name = str(pkg.get("PACKAGE NAME", "")).upper()
        cat = str(pkg.get("PACKAGE CATEGORY", "")).upper()
        return cat == "STAND ALONE" or "[STAND ALONE]" in name or "[STAND-ALONE]" in name or "[STAND- ALONE]" in name

    standalone_already_selected = any(
        _is_standalone_option(sel)
        for sel in flow.selections.values()
        if isinstance(sel, dict)
    )

    selected_id = selection.get("id")
    if not selected_id:
        return False, "No option selected"

    # Block adding packages if a standalone is already selected
    is_new_primary_or_addon = selected_id.startswith("package_") or selected_id.startswith("addon_")
    if standalone_already_selected and is_new_primary_or_addon and not _is_skip_selection_id(selected_id):
        return False, "Stand-alone packages cannot be combined with other procedures or add-ons. Please start a new case for this package."

    # Find selected option from current step's options
    selected_option = None
    
    # Manual-add path
    if selected_id.startswith("manual_add"):
        manual_pkg = selection.get("manual_package") or {}
        manual_code = str(manual_pkg.get("code", "")).strip()
        manual_name = str(manual_pkg.get("name", "")).strip()
        if not manual_code or not manual_name:
            return False, "Manual package details are missing"

        rate_value = manual_pkg.get("rate", 0)
        try:
            parsed_rate = float(rate_value)
        except Exception:
            parsed_rate = 0.0

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

    # Block selecting standalone when other packages already selected
    if _is_package_selection_id(actual_id) and _is_standalone_option(selected_option):
        has_other_packages = any(
            (str(sel.get("id", "")).startswith("package_") or str(sel.get("id", "")).startswith("addon_"))
            and not _is_skip_selection_id(str(sel.get("id")))
            for sel in flow.selections.values()
            if isinstance(sel, dict)
        )
        if has_other_packages:
            return False, "This is a stand-alone package and cannot be combined with your existing selections. Please start a new search for this package."

    # Store selection
    flow.set_selection(flow.current_step, selected_option)

    # ── Standalone short-circuit ─────────────────────────────────────────
    if _is_package_selection_id(str(selected_option.get("id", selected_id))) and _is_standalone_option(selected_option):
        prior_package_exists = any(
            isinstance(sel, dict)
            and _is_package_selection_id(str(sel.get("id", "")))
            and not _is_skip_selection_id(str(sel.get("id", "")))
            for key, sel in flow.selections.items()
            if key != f"step_{flow.current_step}"
        )
        if prior_package_exists:
            return False, "Stand-alone package cannot be combined with other selected packages."

        flow.mark_complete()
        return True, None

    # ── Clarification step → refine next step's package options ──────────
    is_clarification = current_step.context.get("is_clarification", False)
    if is_clarification and flow.current_step + 1 < len(flow.steps):
        from tools.medical_knowledge import get_specialties_for_term, get_clinical_pathway

        generic_tokens = {
            "surgery", "surgical", "procedure", "management", "operation",
            "approach", "general", "specialty", "select", "main", "package",
            "phase", "stage", "level", "type", "pain", "ache", "discomfort", "symptom", "disease",
        }

        def _tokenize(text: str) -> List[str]:
            return [
                t for t in re.findall(r"[a-z0-9]+", (text or "").lower())
                if len(t) > 2 and t not in generic_tokens
            ]

        def _token_hit(token: str, text_tokens: List[str], raw_text: str) -> bool:
            if not token:
                return False
            return any(tok == token or tok.startswith(token) for tok in text_tokens) or token in raw_text

        intent_term = current_step.context.get("intent_term", flow.query)
        selected_hint = selected_option.get("label") or selected_option.get("specialty") or ""
        query_tokens = _tokenize(intent_term)
        hint_tokens = _tokenize(selected_hint)

        mapped_specs = {s.lower().strip() for s in get_specialties_for_term(intent_term)}
        pathway = get_clinical_pathway(intent_term)
        if pathway and pathway.get("steps"):
            for step in pathway.get("steps", []):
                step_spec = str(step.get("specialty", "")).strip().lower()
                if step_spec:
                    mapped_specs.add(step_spec)

        symptom_indicators = {"pain", "fever", "breath", "cough", "bleeding", "swelling", "weakness", "dizziness", "attack"}
        strict_intent = bool(mapped_specs) and any(ind in intent_term.lower() for ind in symptom_indicators)

        combined_tokens: List[str] = []
        for token in query_tokens + hint_tokens:
            if token not in combined_tokens:
                combined_tokens.append(token)
        procedure_hint = " ".join(combined_tokens) if combined_tokens else intent_term

        scored: List[Tuple[int, Dict]] = []
        for pkg in all_packages:
            name = str(pkg.get("PACKAGE NAME", "")).lower()
            spec = str(pkg.get("SPECIALITY", "")).lower()
            raw_text = f"{name} {spec}"
            text_tokens = re.findall(r"[a-z0-9]+", raw_text)
            q_hits = sum(1 for tok in query_tokens if _token_hit(tok, text_tokens, raw_text))
            h_hits = sum(1 for tok in hint_tokens if _token_hit(tok, text_tokens, raw_text))
            s_hit = 1 if any(ms in spec for ms in mapped_specs) else 0

            if query_tokens and q_hits == 0:
                continue
            score = (q_hits * 8) + (h_hits * 4) + (s_hit * 6)
            if strict_intent and s_hit == 0:
                continue
            if score > 0:
                scored.append((score, pkg))

        scored.sort(key=lambda x: x[0], reverse=True)
        refined = [pkg for _, pkg in scored[:500]]

        next_step = flow.steps[flow.current_step + 1]
        if refined:
            next_step.options = generate_package_options(procedure_hint, refined)
            next_step.description = f"Choose the package matched to your selected clinical approach:"
            next_step.context["procedure"] = procedure_hint

    # ── Term package selection → standalone check + implant/strat insertion ─
    is_term_selection = current_step.context.get("is_term_selection", False)
    # Also support legacy is_primary_selection flag
    is_primary = current_step.context.get("is_primary_selection", False)
    is_addon_selection = current_step.context.get("is_addon_selection", False)

    if is_term_selection or is_primary or is_addon_selection:
        pkg_code = selected_option.get("code")
        sel_pkg = None
        for pkg in all_packages:
            if pkg.get("PACKAGE CODE") == pkg_code:
                sel_pkg = pkg
                break

        if sel_pkg:
            # Standalone → complete flow immediately
            if _is_standalone_pkg(sel_pkg):
                flow.mark_complete()
                return True, None

            # Dynamic implant/strat insertion
            insert_idx = flow.current_step + 1
            intent_term = current_step.context.get("intent_term", "")
            if not intent_term:
                # Use the package name if intent_term is empty (e.g., for add-ons)
                intent_term = str(sel_pkg.get("PACKAGE NAME", sel_pkg.get("Package Name", ""))).split('|')[0][:30].strip()

            strat_options = generate_stratification_options(sel_pkg, all_packages)
            # Use unique step names per term to allow multiple terms with strat/implant
            strat_step_name = f"Stratification for {intent_term.title()}" if intent_term else "Select Stratification"
            implant_step_name = f"Implant for {intent_term.title()}" if intent_term else "Select Implant"

            already_has_strat = any(s.step_name == strat_step_name for s in flow.steps)
            if strat_options and not already_has_strat:
                flow.steps.insert(insert_idx, SearchStep(
                    step_number=insert_idx + 1,
                    step_name=strat_step_name,
                    description=f"This package has stratification options. Select one if applicable:",
                    options=strat_options,
                    requires_user_selection=True,
                    context={"main_package": pkg_code, "intent_term": intent_term},
                ))
                insert_idx += 1

            implant_options = generate_implant_options(sel_pkg, all_packages)
            already_has_implant = any(s.step_name == implant_step_name for s in flow.steps)
            if implant_options and not already_has_implant:
                flow.steps.insert(insert_idx, SearchStep(
                    step_number=insert_idx + 1,
                    step_name=implant_step_name,
                    description=f"This procedure may require an implant. Select one:",
                    options=implant_options,
                    requires_user_selection=True,
                    context={"main_package": pkg_code, "intent_term": intent_term},
                ))

            # Renumber steps
            for idx, step in enumerate(flow.steps):
                step.step_number = idx + 1

    # Advance to next step
    if not flow.advance_step():
        flow.mark_complete()
        return True, None

    advance_past_empty_optional_steps(flow)

    return True, None
