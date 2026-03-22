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

import json
import logging
import re
from typing import Dict, List, Optional, Tuple

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
        options: List[Dict],
        requires_user_selection: bool = True,
        context: Optional[Dict] = None,
    ):
        self.step_number = step_number
        self.step_name = step_name
        self.description = description
        self.options = options
        self.requires_user_selection = requires_user_selection
        self.context = context or {}
        self.user_selection = None

    def to_dict(self) -> Dict:
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
        self.selections: Dict[str, any] = {}
        self.violations: List[str] = []
        self.flow_complete = False
        self.final_recommendation: Optional[Dict] = None

    def add_step(self, step: SearchStep) -> None:
        self.steps.append(step)

    def set_selection(self, step_number: int, selection: Dict) -> None:
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

    def to_dict(self) -> Dict:
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


# ═══════════════════════════════════════════════════════════════════════════════
# OPTION GENERATORS - CREATE OPTIONS FOR EACH STEP
# ═══════════════════════════════════════════════════════════════════════════════

def generate_procedure_options(main_term: str, packages: List[Dict]) -> List[Dict]:
    """
    For a broad term (e.g., "heart attack"), extract unique procedures/pathways.
    Shows user different treatment approaches.
    """
    from tools.medical_knowledge import get_clinical_pathway, get_specialties_for_term

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
        options = [
            {
                "id": "direct_search",
                "label": f"Direct search for: {main_term}",
                "description": "Search for matching packages directly",
                "reasoning": "No specific procedures found, showing all matches",
            }
        ]

    return options


def generate_package_options(
    selected_procedure: str,
    matching_packages: List[Dict],
    exclude_types: Optional[List[str]] = None
) -> List[Dict]:
    """
    Generate package options for a selected procedure/specialty.
    Groups packages by type and excludes incompatible types.
    """
    exclude_types = exclude_types or []
    options = []
    seen_codes = set()

    selected_lower = (selected_procedure or "").lower()

    def relevance_score(pkg: Dict) -> int:
        name = str(pkg.get("PACKAGE NAME", "")).lower()
        spec = str(pkg.get("SPECIALITY", "")).lower()
        code = str(pkg.get("PACKAGE CODE", "")).lower()
        category = str(pkg.get("PACKAGE CATEGORY", "")).lower()
        score = 0

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
        return score

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
            return options[:15]

    return options[:15]


def generate_implant_options(
    main_package: Dict,
    all_packages: List[Dict]
) -> List[Dict]:
    """
    Generate implant options for a selected surgical package.
    Shows user different implant types available.
    """
    options = []

    implant_field = main_package.get("IMPLANT PACKAGE", "NO IMPLANT")
    if "NO IMPLANT" in implant_field.upper() or "IMPLANT" not in implant_field.upper():
        return []  # No implants for this package

    # Find implant packages related to this procedure
    for pkg in all_packages:
        if "IMPLANT" in pkg.get("PACKAGE CATEGORY", "").upper():
            implant_name = pkg.get("PACKAGE NAME", "")
            implant_code = pkg.get("PACKAGE CODE", "")
            rate = pkg.get("RATE", 0)

            # Check if implant matches main package specialty
            if main_package.get("SPECIALITY", "") in pkg.get("SPECIALITY", ""):
                options.append({
                    "id": f"implant_{implant_code}",
                    "code": implant_code,
                    "label": implant_name[:80],
                    "description": implant_name,
                    "rate": rate,
                    "base_package_code": main_package.get("PACKAGE CODE", ""),
                    "type": "mandatory_with_procedure" if rate > 50000 else "optional",
                })

    # If no specific implants found, show generic option
    if not options:
        options.append({
            "id": "implant_no_implant",
            "code": "NO_IMPLANT",
            "label": "No Implant (Conservative approach)",
            "description": "Proceed without implant",
            "rate": 0,
            "type": "optional",
        })

    return options


def _is_addon_package(pkg: Dict) -> bool:
    """Return True if package appears to be an add-on package."""
    name_raw = str(pkg.get("PACKAGE NAME", "")).upper()
    cat_raw = str(pkg.get("PACKAGE CATEGORY", "")).upper()
    normalized = (name_raw + " " + cat_raw).replace(" ", "").replace("-", "")
    return "ADDON" in normalized


def generate_addon_options(
    main_package: Dict,
    addon_term: str,
    all_packages: List[Dict],
    include_direct_matches: bool = True,
    include_skip_option: bool = False,
) -> List[Dict]:
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

    addon_term_lower = (addon_term or "").lower().strip()
    addon_keywords = [addon_term_lower] + \
        clinical_addons.get(addon_term_lower, [addon_term_lower])

    seen_codes = set()

    # Pass 1: direct term matches
    if include_direct_matches:
        for pkg in all_packages:
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
                    "reason": f"Direct match for: {addon_term}",
                })
                seen_codes.add(code)
                if len(options) >= 8:
                    break

    # Pass 2: clinically related supportive add-ons
    for keyword in addon_keywords:
        if not keyword:
            continue
        for pkg in all_packages:
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
                    "reason": f"Supportive suggestion for: {addon_term}",
                })
                seen_codes.add(code)
                if len(options) >= 8:
                    break
        if len(options) >= 8:
            break

    if include_skip_option:
        options.append({
            "id": "addon_skip",
            "code": "",
            "label": "Skip supportive package",
            "description": "Continue with only the selected main package.",
            "specialty": "Optional",
            "rate": 0,
            "reason": "User can choose to skip this suggestion",
        })

    return options[:9]


# ═══════════════════════════════════════════════════════════════════════════════
# RULE VALIDATION
# ═══════════════════════════════════════════════════════════════════════════════

def validate_package_combination(
    main_package: Dict,
    implant_package: Optional[Dict] = None,
    addon_packages: Optional[List[Dict]] = None,
) -> Tuple[bool, List[str]]:
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
    is_main_surgical = "SURGICAL" in main_name or "SURGERY" in main_name or "OPERATIVE" in main_name
    is_main_medical = "MEDICAL" in main_name or "CONSERVATIVE" in main_name

    for addon in addon_packages:
        addon_name = addon.get("PACKAGE NAME", "").upper()
        addon_code = addon.get("PACKAGE CODE", "")

        is_addon_surgical = "SURGICAL" in addon_name or "SURGERY" in addon_name
        is_addon_medical = "MEDICAL" in addon_name or "CONSERVATIVE" in addon_name

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
    is_main_standalone = "STAND-ALONE" in main_name or "STANDALONE" in main_name

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


# ═══════════════════════════════════════════════════════════════════════════════
# BUILD FLOW STEPS
# ═══════════════════════════════════════════════════════════════════════════════

def build_search_flow(
    main_term: str,
    addon_terms: List[str],
    matching_packages: List[Dict],
    all_packages_for_addons: Optional[List[Dict]] = None,
) -> FlowState:
    """
    Build the complete multi-step interactive search flow.

    Flow Steps:
    1. Parse and clarify main term (input or options)
    2. Show package options for main procedure
    3. Ask for implant selection (if applicable)
    4. Show add-on options for each addon term
    5. Validate rules and show final recommendation
    """
    flow = FlowState(
        session_id="interactive_search",
        query=main_term,
        parsed_terms=[main_term] + addon_terms,
    )

    # Step 1: Clarify main term (if broad)
    main_options = generate_procedure_options(main_term, matching_packages)

    step1 = SearchStep(
        step_number=1,
        step_name="Select Main Procedure/Approach",
        description=f"'{main_term}' is a broad term. Please select the most specific procedure approach:",
        options=main_options,
        requires_user_selection=len(main_options) > 1,
        context={"term": main_term},
    )
    flow.add_step(step1)

    # Step 2: Show package options for main procedure
    main_packages = generate_package_options(main_term, matching_packages)

    step2 = SearchStep(
        step_number=2,
        step_name="Select Main Package",
        description="Choose the primary package for your procedure:",
        options=main_packages,
        requires_user_selection=True,
        context={"procedure": main_term},
    )
    flow.add_step(step2)

    # Step 3+: supportive / related packages
    addon_source = all_packages_for_addons or matching_packages

    # Doctor-style optional supportive suggestions for the main condition.
    # Example only: anemia -> may suggest transfusion/iron, but user can skip.
    main_supportive_options = generate_addon_options(
        {},
        main_term,
        addon_source,
        include_direct_matches=False,
        include_skip_option=True,
    )
    if len(main_supportive_options) > 1:
        flow.add_step(
            SearchStep(
                step_number=len(flow.steps) + 1,
                step_name=f"Supportive Suggestions for: {main_term}",
                description=(
                    f"Based on '{main_term}', these supportive packages may also be needed. "
                    "Do you want to include any?"
                ),
                options=main_supportive_options,
                requires_user_selection=True,
                context={"addon_term": main_term, "suggestive": True},
            )
        )

    if addon_terms:
        # For additional comma-separated terms, include both direct and related options.
        for addon_term in addon_terms:
            addon_options = generate_addon_options(
                {},
                addon_term,
                addon_source,
                include_direct_matches=True,
                include_skip_option=True,
            )

            flow.add_step(
                SearchStep(
                    step_number=len(flow.steps) + 1,
                    step_name=f"Select Related Packages/Add-ons for: {addon_term}",
                    description=f"Direct and clinically related package options for '{addon_term}':",
                    options=addon_options,
                    requires_user_selection=len(addon_options) > 0,
                    context={"addon_term": addon_term},
                )
            )

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
    """
    # Ensure flow is not parked at an empty optional step.
    advance_past_empty_optional_steps(flow)
    if flow.flow_complete:
        return True, None

    current_step = flow.steps[flow.current_step]

    # Validate selection
    selected_id = selection.get("id")
    if not selected_id:
        return False, "No option selected"

    # Find selected option
    selected_option = None
    for opt in current_step.options:
        if opt.get("id") == selected_id:
            selected_option = opt
            break

    if not selected_option:
        return False, "Invalid selection"

    # Store selection
    flow.set_selection(flow.current_step, selected_option)

    # If procedure/specialty is selected in step 1, refine step 2 package options.
    if current_step.step_number == 1 and flow.current_step + 1 < len(flow.steps):
        from tools.medical_knowledge import get_specialties_for_term

        generic_tokens = {
            "surgery", "surgical", "procedure", "management", "operation",
            "approach", "general", "specialty", "select", "main", "package",
            "phase", "stage", "level", "type",
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

        selected_hint = selected_option.get(
            "label") or selected_option.get("specialty") or ""
        query_tokens = _tokenize(flow.query)
        hint_tokens = _tokenize(selected_hint)
        mapped_specs = {s.lower().strip()
                        for s in get_specialties_for_term(flow.query)}

        # Keep diagnosis anchors first, then selected non-generic hints.
        combined_tokens: List[str] = []
        for token in query_tokens + hint_tokens:
            if token not in combined_tokens:
                combined_tokens.append(token)

        procedure_hint = " ".join(
            combined_tokens) if combined_tokens else flow.query

        scored_candidates: List[Tuple[int, Dict]] = []
        for pkg in all_packages:
            name = str(pkg.get("PACKAGE NAME", "")).lower()
            spec = str(pkg.get("SPECIALITY", "")).lower()
            raw_text = f"{name} {spec}"
            text_tokens = re.findall(r"[a-z0-9]+", raw_text)
            query_hits = sum(1 for tok in query_tokens if _token_hit(
                tok, text_tokens, raw_text))
            hint_hits = sum(1 for tok in hint_tokens if _token_hit(
                tok, text_tokens, raw_text))
            spec_hit = 1 if any(ms in spec for ms in mapped_specs) else 0

            # Anchor on diagnosis for all searches, not only specific terms.
            if query_tokens and query_hits == 0:
                continue

            score = (query_hits * 8) + (hint_hits * 4) + (spec_hit * 6)
            if score > 0:
                scored_candidates.append((score, pkg))

        scored_candidates.sort(key=lambda item: item[0], reverse=True)
        refined_candidates = [pkg for _, pkg in scored_candidates[:500]]

        next_step = flow.steps[flow.current_step + 1]
        if refined_candidates:
            next_step.options = generate_package_options(
                procedure_hint, refined_candidates)
            next_step.description = "Choose the primary package matched to your selected clinical approach:"
            next_step.context["procedure"] = procedure_hint

    # If selecting main package, add implant step
    if current_step.step_number == 2:  # Main package step
        pkg_code = selected_option.get("code")
        main_pkg = None
        for pkg in all_packages:
            if pkg.get("PACKAGE CODE") == pkg_code:
                main_pkg = pkg
                break

        if main_pkg:
            implant_options = generate_implant_options(main_pkg, all_packages)
            if implant_options:
                implant_step = SearchStep(
                    step_number=3,
                    step_name="Select Implant (if needed)",
                    description="This procedure may require an implant. Select one:",
                    options=implant_options,
                    requires_user_selection=True,
                    context={"main_package": pkg_code},
                )
                # Insert implant step immediately after current step (index-based).
                insert_index = flow.current_step + 1
                already_has_implant_step = any(
                    s.step_name == "Select Implant (if needed)" for s in flow.steps
                )
                if not already_has_implant_step:
                    flow.steps.insert(insert_index, implant_step)

                    # Renumber all steps to keep numbering continuous and deterministic.
                    for idx, step in enumerate(flow.steps):
                        step.step_number = idx + 1

    # Advance to next step
    if not flow.advance_step():
        flow.mark_complete()
        return True, None

    # Skip any subsequent empty optional steps.
    advance_past_empty_optional_steps(flow)

    return True, None
