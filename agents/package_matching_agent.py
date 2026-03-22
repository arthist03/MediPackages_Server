"""
agents/package_matching_agent.py  —  Node 4: Intelligent MAA YOJANA package matching.

Uses the Smart Package Agent with:
1. LLM-powered semantic understanding of medical procedures
2. Medical knowledge base for procedure-specialty mapping
3. Rule enforcement DURING selection (not just post-hoc)
4. Hospital package selector thinking pattern

The 5 MAA YOJANA Booking Rules:
1. Surgical + medical management packages CANNOT be booked together
2. Stand-alone packages CANNOT be booked with any other package
3. Add-on packages can ONLY be booked alongside a regular package
4. Implant packages automatically appear with their parent procedure
5. Extended LOS packages can ONLY be booked with a surgery package
"""
from __future__ import annotations

import logging

from graph.state import OcrPipelineState
from tools.smart_package_agent import intelligent_package_search
from tools.medical_knowledge import (
    get_specialties_for_term,
    classify_case_type,
    expand_synonyms,
)

logger = logging.getLogger("package_matching_agent")


def package_matching_agent(state: OcrPipelineState) -> dict:
    """
    LangGraph node: Intelligent MAA YOJANA package matching.

    Thinks like a hospital package selector:
    1. Analyzes the medical case comprehensively
    2. Identifies relevant specialties and procedures
    3. Applies booking rules during selection
    4. Uses LLM for semantic understanding when beneficial
    """
    logger.info("🧠 Smart Package Agent: Intelligent MAA YOJANA matching")

    extracted = state.get("extracted", {})
    if not extracted:
        return {
            "best_packages": _empty_result(),
            "supervisor_notes": ["Package matching: no extracted data, skipped"],
        }

    # Extract all clinical signals
    diagnosis = str(extracted.get("diagnosis", ""))
    secondary_diagnoses = extracted.get("secondary_diagnoses", [])
    procedures = extracted.get("procedures", [])
    chief_complaints = extracted.get("chief_complaints", [])
    department = str(extracted.get("department", ""))
    surgery_name = str(extracted.get("surgery_name", ""))
    procedure_name = str(extracted.get("procedure_name", ""))
    surgery_required = extracted.get("surgery_required", False)
    procedure_required = extracted.get("procedure_required", False)
    lab_tests = extracted.get("lab_tests", [])

    # Fallback logic for missing diagnosis
    if not diagnosis:
        if surgery_name:
            diagnosis = surgery_name
            logger.info(f"Using surgery name as diagnosis: {surgery_name}")
        elif procedure_name:
            diagnosis = procedure_name
            logger.info(f"Using procedure name as diagnosis: {procedure_name}")
        elif secondary_diagnoses:
            diagnosis = str(secondary_diagnoses[0])
            logger.info(f"Using secondary diagnosis: {diagnosis}")
        elif chief_complaints:
            diagnosis = str(chief_complaints[0])
            logger.info(f"Using chief complaint as diagnosis: {diagnosis}")
        else:
            logger.info("No diagnosis found — skipping package matching")
            return {
                "best_packages": _empty_result(
                    warnings=["No diagnosis, surgery, or procedure found in extracted data"]
                ),
                "supervisor_notes": ["Package matching: no diagnosis found"],
            }

    # Enrich with medical knowledge
    specialties = get_specialties_for_term(diagnosis)
    if department and department not in specialties:
        specialties.insert(0, department)

    # Classify the case type
    case_type, case_reasoning = classify_case_type(
        diagnosis=diagnosis,
        procedures=[str(p) for p in procedures],
        surgery_name=surgery_name,
    )

    # Expand diagnosis with synonyms for better matching
    expanded_terms = expand_synonyms(diagnosis)

    logger.info(
        f"📋 Case Analysis:\n"
        f"   Diagnosis: {diagnosis}\n"
        f"   Case Type: {case_type} ({case_reasoning})\n"
        f"   Specialties: {specialties}\n"
        f"   Surgery: {surgery_name or 'N/A'}\n"
        f"   Procedure: {procedure_name or 'N/A'}"
    )

    try:
        # Build comprehensive extracted data for smart search
        search_data = {
            "diagnosis": diagnosis,
            "secondary_diagnoses": secondary_diagnoses,
            "procedures": procedures,
            "chief_complaints": chief_complaints,
            "department": department,
            "surgery_name": surgery_name,
            "procedure_name": procedure_name,
            "surgery_required": surgery_required or bool(surgery_name),
            "procedure_required": procedure_required or bool(procedure_name),
            "lab_tests": lab_tests,
            # Enriched fields
            "_expanded_terms": expanded_terms,
            "_specialties": specialties,
            "_case_type": case_type,
        }

        # Run intelligent package search
        results = intelligent_package_search(
            extracted_data=search_data,
            use_llm=True,  # Enable LLM for smart selection
        )

        # Build summary note
        total = results.get("total_matched", 0)
        case_type_result = results.get("case_type", case_type)

        note = f"🎯 {total} packages for '{diagnosis}' ({case_type_result})"
        if surgery_name:
            note += f" [Surgery: {surgery_name}]"
        if results.get("warnings"):
            note += f" ⚠️ {len(results['warnings'])} rule warning(s)"
        if results.get("llm_reasoning"):
            note += f"\n   LLM: {results['llm_reasoning'][:100]}..."

        logger.info(f"✅ {note}")

        # Add case analysis to results
        results["case_analysis"] = {
            "diagnosis": diagnosis,
            "case_type": case_type_result,
            "specialties": specialties,
            "surgery": surgery_name,
            "procedure": procedure_name,
        }

        return {
            "best_packages": results,
            "supervisor_notes": [f"Package matching: {note}"],
        }

    except Exception as e:
        logger.exception("Package matching error")
        return {
            "best_packages": _empty_result(
                warnings=[f"Error during package matching: {e}"]
            ),
            "supervisor_notes": [f"Package matching error: {e}"],
        }


def _empty_result(warnings: list = None) -> dict:
    """Return an empty result structure."""
    return {
        "primary_packages": [],
        "addon_packages": [],
        "implant_packages": [],
        "warnings": warnings or [],
        "removed_packages": [],
        "total_matched": 0,
        "case_type": "unknown",
    }
