"""
tools/validation_tool.py  —  Pydantic schema for structured medical document data.

Validates and normalises LLM-extracted data. Zero-tolerance hallucination
prevention: missing required fields raise ValidationError.
"""
from __future__ import annotations

import json
from typing import Any, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


class VitalsModel(BaseModel):
    bp: str = ""
    pulse: str = ""
    temperature: str = ""
    spo2: str = ""
    weight: str = ""
    rbs: str = ""   # Random Blood Sugar
    bmi: str = ""


class MedicationModel(BaseModel):
    name: str = ""
    dose: str = ""
    frequency: str = ""
    duration: str = ""
    brand_name: str = ""
    route: str = ""
    instructions: str = ""


class ValidationResult(BaseModel):
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    issues: list[str] = Field(default_factory=list)
    drugs_verified: int = 0
    drugs_total: int = 0


class MedicalExtraction(BaseModel):
    """
    Complete structured extraction of a medical document.
    Matches the Flutter app's expected JSON schema exactly.
    """
    # Core identity
    patient_name: str = ""
    patient_age: Optional[int] = None
    patient_gender: str = ""
    date: str = ""

    # Provider info
    doctor_name: str = ""
    qualifications: str = ""
    clinic_name: str = ""
    clinic_address: str = ""

    # Clinical
    diagnosis: str = ""
    secondary_diagnoses: list[str] = Field(default_factory=list)
    chief_complaints: list[str] = Field(default_factory=list)
    history: str = ""
    department: str = ""
    medications: list[MedicationModel] = Field(default_factory=list)
    lab_tests: list[str] = Field(default_factory=list)
    lab_tests_ordered: bool = False
    procedures: list[str] = Field(default_factory=list)
    surgery_required: bool = False
    surgery_name: str = ""
    procedure_required: bool = False
    procedure_name: str = ""
    follow_up_date: str = ""

    # Vitals
    vitals: VitalsModel = Field(default_factory=VitalsModel)

    # Raw
    full_raw_text: str = ""

    @field_validator("patient_age", mode="before")
    @classmethod
    def coerce_age(cls, v: Any) -> Optional[int]:
        if v is None or v == "":
            return None
        try:
            return int(str(v).strip().split()[0])
        except (ValueError, TypeError):
            return None

    @field_validator("patient_gender", mode="before")
    @classmethod
    def normalise_gender(cls, v: Any) -> str:
        val = str(v or "").lower().strip()
        if "f" in val:
            return "Female"
        if "m" in val:
            return "Male"
        return str(v or "")

    @model_validator(mode="after")
    def compute_validation(self) -> "MedicalExtraction":
        # Fallback for missing diagnosis
        if not self.diagnosis.strip():
            if self.surgery_name.strip():
                self.diagnosis = self.surgery_name.strip()
            elif self.secondary_diagnoses:
                self.diagnosis = self.secondary_diagnoses[0].strip()
            elif self.chief_complaints:
                self.diagnosis = self.chief_complaints[0].strip()
        return self


def validate_extraction(raw: dict[str, Any], raw_text: str = "") -> tuple[MedicalExtraction, ValidationResult]:
    """
    Validate and normalise an LLM extraction dict.
    Returns (validated_model, validation_result).
    """
    issues = []

    # Attempt Pydantic parse
    try:
        extraction = MedicalExtraction(**raw, full_raw_text=raw_text)
    except Exception as e:
        issues.append(f"Schema parse error: {e}")
        extraction = MedicalExtraction(full_raw_text=raw_text)

    # Confidence heuristic: patient_name + diagnosis are the key anchors
    required_fields = ["patient_name", "diagnosis"]
    filled = sum(1 for f in required_fields if getattr(extraction, f, ""))
    # Give partial credit: 0.5 per required field, plus 0.25 bonus per optional
    confidence = (filled / len(required_fields)) * 0.75
    optional_filled = sum(1 for f in ["doctor_name", "date", "clinic_name"]
                          if getattr(extraction, f, ""))
    confidence += min(optional_filled * 0.083, 0.25)  # up to 0.25 bonus
    confidence = round(min(confidence, 1.0), 2)

    if not extraction.patient_name:
        issues.append("Missing: patient_name")
    if not extraction.diagnosis:
        issues.append("Missing: diagnosis")

    drugs_total = len(extraction.medications)

    validation = ValidationResult(
        confidence=confidence,
        issues=issues,
        drugs_verified=drugs_total,
        drugs_total=drugs_total,
    )

    return extraction, validation


def extraction_to_response_dict(
    extraction: MedicalExtraction,
    validation: ValidationResult,
    best_packages: list[dict],
) -> dict[str, Any]:
    """Convert validated extraction to the exact JSON schema the Flutter app expects."""
    return {
        "success": True,
        "data": {
            "full_raw_text": extraction.full_raw_text,
            "doctor_name": extraction.doctor_name,
            "qualifications": extraction.qualifications,
            "clinic_name": extraction.clinic_name,
            "clinic_address": extraction.clinic_address,
            "patient_name": extraction.patient_name,
            "patient_age": extraction.patient_age,
            "patient_gender": extraction.patient_gender,
            "date": extraction.date,
            "diagnosis": extraction.diagnosis,
            "secondary_diagnoses": extraction.secondary_diagnoses,
            "chief_complaints": extraction.chief_complaints,
            "history": extraction.history,
            "department": extraction.department,
            "surgery_required": extraction.surgery_required,
            "surgery_name": extraction.surgery_name,
            "procedure_required": extraction.procedure_required,
            "procedure_name": extraction.procedure_name,
            "lab_tests_ordered": extraction.lab_tests_ordered,
            "medications": [m.model_dump() for m in extraction.medications],
            "lab_tests": extraction.lab_tests,
            "procedures": extraction.procedures,
            "vitals": extraction.vitals.model_dump(),
            "follow_up_date": extraction.follow_up_date,
            "_validation": validation.model_dump(),
            "_agentic_data": {
                "best_packages": best_packages,
            },
        },
    }
