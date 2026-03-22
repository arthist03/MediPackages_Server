"""
tools/medical_knowledge.py — Comprehensive medical knowledge base for intelligent package matching.

Contains:
1. ICD-10 to Specialty mappings
2. Procedure to Package type mappings
3. Synonym dictionaries for medical terms
4. Implicit rules for package combinations
"""
from __future__ import annotations

from typing import Dict, List, Set, Tuple


# ══════════════════════════════════════════════════════════════════════════════
# SPECIALTY MAPPINGS - Maps diagnoses/procedures to relevant specialties
# ══════════════════════════════════════════════════════════════════════════════

SPECIALTY_MAPPING: Dict[str, List[str]] = {
    # ── Burns & Trauma ─────────────────────────────────────────────────
    "burn": ["Burns Management"],
    "thermal burn": ["Burns Management"],
    "electrical burn": ["Burns Management"],
    "flame burn": ["Burns Management"],
    "chemical burn": ["Burns Management"],
    "scald": ["Burns Management"],
    "skin graft": ["Burns Management", "Plastic Surgery"],
    "contracture": ["Burns Management", "Plastic Surgery"],

    # ── Cardiology & Cardio Thoracic ────────────────────────────────────
    "heart": ["Cardiology", "Cardio Thoracic Surgery"],
    "cardiac": ["Cardiology", "Cardio Thoracic Surgery"],
    "coronary": ["Cardiology", "Cardio Thoracic Surgery"],
    "angioplasty": ["Cardiology"],
    "angiography": ["Cardiology"],
    "stent": ["Cardiology"],
    "pacemaker": ["Cardiology", "Cardio Thoracic Surgery"],
    "bypass": ["Cardio Thoracic Surgery"],
    "cabg": ["Cardio Thoracic Surgery"],
    "valve": ["Cardio Thoracic Surgery"],
    "valve replacement": ["Cardio Thoracic Surgery"],
    "mitral": ["Cardio Thoracic Surgery"],
    "aortic": ["Cardio Thoracic Surgery"],
    "arrhythmia": ["Cardiology"],
    "heart failure": ["Cardiology"],
    "myocardial infarction": ["Cardiology"],
    "mi": ["Cardiology"],
    "angina": ["Cardiology"],

    # ── General Surgery ─────────────────────────────────────────────────
    "appendix": ["General Surgery"],
    "appendicitis": ["General Surgery"],
    "appendectomy": ["General Surgery", "Laparoscopic Surgery"],
    "hernia": ["General Surgery"],
    "inguinal hernia": ["General Surgery"],
    "umbilical hernia": ["General Surgery"],
    "incisional hernia": ["General Surgery"],
    "ventral hernia": ["General Surgery"],
    "hiatal hernia": ["General Surgery"],
    "cholecystectomy": ["General Surgery", "Laparoscopic Surgery"],
    "gallbladder": ["General Surgery"],
    "gallstone": ["General Surgery"],
    "cholelithiasis": ["General Surgery"],
    "cholecystitis": ["General Surgery"],
    "intestinal obstruction": ["General Surgery"],
    "bowel": ["General Surgery"],
    "colectomy": ["General Surgery"],
    "hemorrhoid": ["General Surgery"],
    "piles": ["General Surgery"],
    "fissure": ["General Surgery"],
    "fistula": ["General Surgery"],
    "abscess": ["General Surgery"],
    "thyroid": ["General Surgery", "ENT"],
    "thyroidectomy": ["General Surgery", "ENT"],
    "breast": ["General Surgery", "Surgical Oncology"],
    "mastectomy": ["General Surgery", "Surgical Oncology"],
    "lumpectomy": ["General Surgery", "Surgical Oncology"],

    # ── Orthopaedics ────────────────────────────────────────────────────
    "fracture": ["Orthopaedics"],
    "bone": ["Orthopaedics"],
    "joint": ["Orthopaedics"],
    "knee": ["Orthopaedics"],
    "hip": ["Orthopaedics"],
    "shoulder": ["Orthopaedics"],
    "ankle": ["Orthopaedics"],
    "wrist": ["Orthopaedics"],
    "elbow": ["Orthopaedics"],
    "spine": ["Orthopaedics", "Neurosurgery"],
    "vertebra": ["Orthopaedics", "Neurosurgery"],
    "disc": ["Orthopaedics", "Neurosurgery"],
    "scoliosis": ["Orthopaedics"],
    "arthritis": ["Orthopaedics"],
    "arthroplasty": ["Orthopaedics"],
    "joint replacement": ["Orthopaedics"],
    "knee replacement": ["Orthopaedics"],
    "hip replacement": ["Orthopaedics"],
    "tkr": ["Orthopaedics"],
    "thr": ["Orthopaedics"],
    "arthroscopy": ["Orthopaedics"],
    "ligament": ["Orthopaedics"],
    "acl": ["Orthopaedics"],
    "mcl": ["Orthopaedics"],
    "meniscus": ["Orthopaedics"],
    "rotator cuff": ["Orthopaedics"],
    "tendon": ["Orthopaedics"],

    # ── Neurology & Neurosurgery ────────────────────────────────────────
    "brain": ["Neurosurgery", "Neurology"],
    "craniotomy": ["Neurosurgery"],
    "tumor brain": ["Neurosurgery"],
    "stroke": ["Neurology", "Neurosurgery"],
    "epilepsy": ["Neurology"],
    "seizure": ["Neurology"],
    "parkinson": ["Neurology"],
    "spinal cord": ["Neurosurgery"],
    "laminectomy": ["Neurosurgery", "Orthopaedics"],
    "discectomy": ["Neurosurgery", "Orthopaedics"],
    "hydrocephalus": ["Neurosurgery"],
    "shunt": ["Neurosurgery"],
    "aneurysm": ["Neurosurgery"],

    # ── Urology ─────────────────────────────────────────────────────────
    "kidney": ["Urology", "Nephrology"],
    "renal": ["Urology", "Nephrology"],
    "prostate": ["Urology"],
    "turp": ["Urology"],
    "bladder": ["Urology"],
    "ureter": ["Urology"],
    "urethra": ["Urology"],
    "nephrectomy": ["Urology"],
    "kidney stone": ["Urology"],
    "renal calculus": ["Urology"],
    "urolithiasis": ["Urology"],
    "lithotripsy": ["Urology"],
    "pcnl": ["Urology"],
    "ursl": ["Urology"],
    "cystectomy": ["Urology"],

    # ── Gynecology & Obstetrics ─────────────────────────────────────────
    "uterus": ["Obstetrics and Gynaecology"],
    "ovary": ["Obstetrics and Gynaecology"],
    "ovarian": ["Obstetrics and Gynaecology"],
    "hysterectomy": ["Obstetrics and Gynaecology"],
    "cesarean": ["Obstetrics and Gynaecology"],
    "c-section": ["Obstetrics and Gynaecology"],
    "lscs": ["Obstetrics and Gynaecology"],
    "delivery": ["Obstetrics and Gynaecology"],
    "pregnancy": ["Obstetrics and Gynaecology"],
    "fibroid": ["Obstetrics and Gynaecology"],
    "myomectomy": ["Obstetrics and Gynaecology"],
    "endometriosis": ["Obstetrics and Gynaecology"],
    "ectopic": ["Obstetrics and Gynaecology"],
    "tubal ligation": ["Obstetrics and Gynaecology"],

    # ── Ophthalmology ───────────────────────────────────────────────────
    "eye": ["Ophthalmology"],
    "cataract": ["Ophthalmology"],
    "glaucoma": ["Ophthalmology"],
    "retina": ["Ophthalmology"],
    "vitrectomy": ["Ophthalmology"],
    "cornea": ["Ophthalmology"],
    "lasik": ["Ophthalmology"],
    "squint": ["Ophthalmology"],

    # ── ENT ─────────────────────────────────────────────────────────────
    "ear": ["ENT"],
    "nose": ["ENT"],
    "throat": ["ENT"],
    "tonsil": ["ENT"],
    "tonsillectomy": ["ENT"],
    "adenoid": ["ENT"],
    "adenoidectomy": ["ENT"],
    "septoplasty": ["ENT"],
    "septum": ["ENT"],
    "sinusitis": ["ENT"],
    "sinus": ["ENT"],
    "fess": ["ENT"],
    "mastoid": ["ENT"],
    "mastoidectomy": ["ENT"],
    "cochlear": ["ENT"],
    "hearing": ["ENT"],
    "larynx": ["ENT"],
    "vocal cord": ["ENT"],

    # ── Oncology ────────────────────────────────────────────────────────
    "cancer": ["Surgical Oncology", "Medical Oncology", "Radiation Oncology"],
    "carcinoma": ["Surgical Oncology", "Medical Oncology"],
    "malignancy": ["Surgical Oncology", "Medical Oncology"],
    "tumor": ["Surgical Oncology", "Medical Oncology"],
    "lymphoma": ["Medical Oncology", "Radiation Oncology"],
    "leukemia": ["Medical Oncology"],
    "chemotherapy": ["Medical Oncology"],
    "radiation therapy": ["Radiation Oncology"],
    "brachytherapy": ["Radiation Oncology"],

    # ── Gastroenterology ────────────────────────────────────────────────
    "stomach": ["Gastroenterology", "General Surgery"],
    "gastric": ["Gastroenterology", "General Surgery"],
    "ulcer": ["Gastroenterology"],
    "endoscopy": ["Gastroenterology"],
    "colonoscopy": ["Gastroenterology"],
    "liver": ["Gastroenterology", "Surgical Gastroenterology"],
    "hepatic": ["Gastroenterology", "Surgical Gastroenterology"],
    "cirrhosis": ["Gastroenterology"],
    "pancreas": ["Gastroenterology", "Surgical Gastroenterology"],
    "pancreatitis": ["Gastroenterology"],
    "esophagus": ["Gastroenterology", "Surgical Gastroenterology"],
    "achalasia": ["Gastroenterology", "Surgical Gastroenterology"],
    "gi bleed": ["Gastroenterology"],

    # ── Pulmonology ─────────────────────────────────────────────────────
    "lung": ["Pulmonology", "Cardio Thoracic Surgery"],
    "pulmonary": ["Pulmonology"],
    "respiratory": ["Pulmonology"],
    "asthma": ["Pulmonology"],
    "copd": ["Pulmonology"],
    "pneumonia": ["Pulmonology"],
    "tuberculosis": ["Pulmonology"],
    "tb": ["Pulmonology"],
    "bronchitis": ["Pulmonology"],
    "pleural": ["Pulmonology", "Cardio Thoracic Surgery"],
    "thoracoscopy": ["Cardio Thoracic Surgery"],
    "lobectomy": ["Cardio Thoracic Surgery"],

    # ── Pediatric ───────────────────────────────────────────────────────
    "pediatric": ["Pediatric Surgery", "Pediatrics"],
    "congenital": ["Pediatric Surgery"],
    "neonatal": ["Pediatric Surgery", "Neonatology"],

    # ── Plastic Surgery ─────────────────────────────────────────────────
    "reconstructive": ["Plastic Surgery"],
    "cosmetic": ["Plastic Surgery"],
    "cleft lip": ["Plastic Surgery"],
    "cleft palate": ["Plastic Surgery"],

    # ── Vascular ────────────────────────────────────────────────────────
    "varicose": ["Vascular Surgery"],
    "vein": ["Vascular Surgery"],
    "artery": ["Vascular Surgery"],
    "aneurysm abdominal": ["Vascular Surgery"],
    "dvt": ["Vascular Surgery"],
    "embolism": ["Vascular Surgery"],

    # ── Nephrology ──────────────────────────────────────────────────────
    "dialysis": ["Nephrology"],
    "hemodialysis": ["Nephrology"],
    "peritoneal dialysis": ["Nephrology"],
    "renal failure": ["Nephrology"],
    "ckd": ["Nephrology"],
    "transplant kidney": ["Nephrology", "Urology"],

    # ── Endocrinology ───────────────────────────────────────────────────
    "diabetes": ["Endocrinology", "General Medicine"],
    "thyroid disorder": ["Endocrinology"],
    "adrenal": ["Endocrinology"],
    "pituitary": ["Endocrinology", "Neurosurgery"],

    # ── General Medicine (Medical Management) ───────────────────────────
    "fever": ["General Medicine"],
    "infection": ["General Medicine"],
    "dengue": ["General Medicine"],
    "malaria": ["General Medicine"],
    "typhoid": ["General Medicine"],
    "jaundice": ["General Medicine", "Gastroenterology"],
    "anemia": ["General Medicine", "Hematology"],
    "hypertension": ["General Medicine", "Cardiology"],
}


# ══════════════════════════════════════════════════════════════════════════════
# MEDICAL SYNONYMS - Helps match different terms for same condition
# ══════════════════════════════════════════════════════════════════════════════

MEDICAL_SYNONYMS: Dict[str, List[str]] = {
    "appendicitis": ["appendix", "appendicular", "appendectomy"],
    "cholecystitis": ["gallbladder", "gallstone", "cholelithiasis", "cholecystectomy"],
    "hernia": ["herniation", "herniorrhaphy", "hernioplasty"],
    "fracture": ["broken bone", "fx"],
    "hypertension": ["high blood pressure", "htn", "bp"],
    "diabetes": ["dm", "sugar", "diabetic", "t2dm"],
    "myocardial infarction": ["heart attack", "mi", "stemi", "nstemi"],
    "coronary artery disease": ["cad", "ischemic heart disease", "ihd"],
    "cesarean": ["c-section", "lscs", "cs"],
    "joint replacement": ["arthroplasty", "tkr", "thr"],
    "kidney stone": ["renal calculus", "urolithiasis", "nephrolithiasis"],
    "stroke": ["cva", "cerebrovascular accident"],
    "cancer": ["carcinoma", "ca", "malignancy", "tumor"],
}


# ══════════════════════════════════════════════════════════════════════════════
# PROCEDURE TYPE INDICATORS
# ══════════════════════════════════════════════════════════════════════════════

# Suffixes that indicate surgical procedures
SURGICAL_SUFFIXES: Set[str] = {
    "ectomy",   # removal (appendectomy, cholecystectomy)
    "otomy",    # cutting into (laparotomy, craniotomy)
    "plasty",   # repair/reconstruction (arthroplasty, rhinoplasty)
    "pexy",     # fixation (orchiopexy, nephropexy)
    "ostomy",   # creating opening (colostomy, tracheostomy)
    "rrhaphy",  # suturing (herniorrhaphy)
    "centesis",  # puncture (thoracentesis, paracentesis)
    "scopy",    # visualization (laparoscopy, endoscopy) - may be diagnostic
}

# Words that strongly indicate surgery
SURGERY_INDICATORS: Set[str] = {
    "surgery", "operation", "operative", "surgical",
    "excision", "resection", "repair", "reconstruction",
    "replacement", "implant", "implantation", "transplant",
    "bypass", "graft", "fusion", "fixation", "removal",
    "laparoscopic", "arthroscopic", "endoscopic", "robotic",
    "open", "minimal invasive", "mis",
}

# Words that indicate medical management (non-surgical)
MEDICAL_MANAGEMENT_INDICATORS: Set[str] = {
    "conservative", "medical management", "medication",
    "fever", "infection", "viral", "bacterial",
    "typhoid", "dengue", "malaria", "jaundice",
    "pneumonia", "bronchitis", "gastritis",
    "monitoring", "observation", "supportive care",
}


# ══════════════════════════════════════════════════════════════════════════════
# PACKAGE COMBINATION RULES (Domain Knowledge)
# ══════════════════════════════════════════════════════════════════════════════

# Procedures that commonly need implants
IMPLANT_PROCEDURES: Dict[str, List[str]] = {
    "joint replacement": ["prosthesis", "implant"],
    "fracture fixation": ["plate", "screw", "nail", "k-wire"],
    "pacemaker": ["generator", "lead"],
    "stent": ["stent"],
    "valve replacement": ["valve prosthesis"],
    "hernia repair": ["mesh"],
    "spine surgery": ["cage", "screw", "rod"],
}

# Procedures that commonly need extended LOS
EXTENDED_LOS_TRIGGERS: Set[str] = {
    "major surgery", "organ transplant", "complex trauma",
    "multiple fractures", "polytrauma", "icu required",
    "ventilator", "sepsis", "burns >40%",
    "cardiac surgery", "neurosurgery major",
}


# ══════════════════════════════════════════════════════════════════════════════
# CLINICAL PATHWAYS - Doctor-like reasoning for symptoms → diagnosis → treatment
# ══════════════════════════════════════════════════════════════════════════════

CLINICAL_PATHWAYS: Dict[str, Dict] = {
    # ── Cardiac Symptoms ─────────────────────────────────────────────────
    "chest pain": {
        "specialty": "Cardiology",
        "initial_workup": ["ECG", "Cardiac markers", "2D Echo"],
        "diagnosis_packages": ["Coronary Angiography"],
        "treatment_pathway": {
            "if_mi_confirmed": ["Systemic Thrombolysis (for MI)", "PTCA"],
            "if_blockage_found": ["PTCA", "CABG"],
            "if_heart_failure": ["Congestive heart failure"],
            "if_stable_angina": ["Medical management", "PTCA if severe"],
        },
        "package_codes": {
            # Coronary Angiography
            "diagnosis": ["1831-MC023A", "2831-MC023A"],
            "thrombolysis": ["1831-MC020A"],  # Systemic Thrombolysis
            "ptca": ["2831-MC026A"],  # PTCA/Angioplasty
            "cabg": ["2832-SV018A"],  # Bypass surgery
            "heart_failure": ["1831-MG038A"],  # CHF
        },
        "doctor_reasoning": "For chest pain, first confirm diagnosis via Coronary Angiography. If MI → Thrombolysis/PTCA. If blockages → PTCA (1-2 vessels) or CABG (multiple vessels)."
    },

    "heart attack": {
        "specialty": "Cardiology",
        "synonyms": ["mi", "myocardial infarction", "stemi", "nstemi"],
        "emergency": True,
        "treatment_pathway": {
            "primary": ["Systemic Thrombolysis (for MI)", "Primary PTCA"],
            "post_thrombolysis": ["Coronary Angiography", "PTCA if needed"],
            "if_multiple_blockages": ["CABG"],
        },
        "package_codes": {
            "thrombolysis": ["1831-MC020A"],
            "ptca": ["2831-MC026A"],
            "angiography": ["1831-MC023A"],
            "cabg": ["2832-SV018A"],
        },
        "doctor_reasoning": "Heart attack requires immediate action. STEMI → Primary PTCA or Thrombolysis. NSTEMI → Stabilize then Angiography. Multiple vessel disease → CABG."
    },

    "breathlessness": {
        "specialty": ["Cardiology", "Pulmonology"],
        "differential": ["Heart failure", "Pulmonary disease", "Anemia"],
        "initial_workup": ["2D Echo", "Chest X-ray", "Pulmonary function test"],
        "treatment_pathway": {
            "if_cardiac": ["Congestive heart failure", "Valve surgery if needed"],
            "if_pulmonary": ["Medical management", "Thoracoscopy if needed"],
        },
        "package_codes": {
            "heart_failure": ["1831-MG038A"],
            "valve_surgery": ["2832-SV004A"],
        },
        "doctor_reasoning": "Breathlessness needs cardiac vs pulmonary differentiation. Cardiac → CHF management or valve surgery. Pulmonary → Medical or surgical based on cause."
    },

    # ── Abdominal Pain Pathways ─────────────────────────────────────────
    "abdominal pain": {
        "specialty": ["General Surgery", "Gastroenterology"],
        "differential": ["Appendicitis", "Cholecystitis", "Pancreatitis", "Intestinal obstruction"],
        "location_based": {
            "right_lower": ["Appendicitis", "Appendectomy"],
            "right_upper": ["Cholecystitis", "Cholecystectomy"],
            "epigastric": ["Pancreatitis", "Peptic ulcer"],
            "generalized": ["Intestinal obstruction", "Peritonitis"],
        },
        "doctor_reasoning": "Abdominal pain location guides diagnosis. RLQ → Appendicitis. RUQ → Gallbladder. Epigastric → Pancreas/Stomach."
    },

    "appendicitis": {
        "specialty": "General Surgery",
        "synonyms": ["appendix pain", "appendix"],
        "treatment": ["Appendectomy - Laparoscopic", "Appendectomy - Open"],
        "package_codes": {
            "lap": ["1851-SV001A"],
            "open": ["1851-SV001B"],
        },
        "doctor_reasoning": "Appendicitis requires appendectomy. Laparoscopic preferred if available and patient stable."
    },

    "gallstone": {
        "specialty": "General Surgery",
        "synonyms": ["cholecystitis", "cholelithiasis", "gallbladder stone", "biliary colic"],
        "treatment": ["Cholecystectomy - Laparoscopic", "Cholecystectomy - Open"],
        "package_codes": {
            "lap": ["1851-SV002A"],
            "open": ["1851-SV002B"],
        },
        "doctor_reasoning": "Gallstones with symptoms require cholecystectomy. Laparoscopic is gold standard."
    },

    # ── Orthopedic Pathways ─────────────────────────────────────────────
    "fracture": {
        "specialty": "Orthopaedics",
        "treatment_by_type": {
            "simple": ["Closed reduction", "POP cast"],
            "compound": ["Open reduction", "Internal fixation (ORIF)"],
            "joint_involved": ["Arthroplasty may be needed"],
        },
        "implant_needed": True,
        "doctor_reasoning": "Fracture treatment depends on type and location. Simple → Conservative. Displaced/Compound → ORIF with implant."
    },

    "knee pain": {
        "specialty": "Orthopaedics",
        "differential": ["Osteoarthritis", "Ligament injury", "Meniscus tear"],
        "treatment_pathway": {
            "if_arthritis": ["Medical management", "Total Knee Replacement if severe"],
            "if_ligament": ["ACL reconstruction", "MCL repair"],
            "if_meniscus": ["Arthroscopic meniscectomy"],
        },
        "doctor_reasoning": "Knee pain in elderly → likely OA → TKR if severe. Young patient with injury → likely ligament/meniscus → Arthroscopy."
    },

    "hip pain": {
        "specialty": "Orthopaedics",
        "differential": ["Osteoarthritis", "Fracture neck femur", "AVN"],
        "treatment_pathway": {
            "if_fracture": ["Hemiarthroplasty", "Total Hip Replacement"],
            "if_arthritis": ["Total Hip Replacement"],
        },
        "doctor_reasoning": "Hip pain in elderly with history of fall → likely fracture → THR/Hemiarthroplasty. Chronic pain → OA → THR."
    },

    # ── Eye Pathways ────────────────────────────────────────────────────
    "vision loss": {
        "specialty": "Ophthalmology",
        "differential": ["Cataract", "Glaucoma", "Retinal detachment", "Diabetic retinopathy"],
        "treatment_pathway": {
            "if_cataract": ["Phacoemulsification with IOL"],
            "if_glaucoma": ["Trabeculectomy", "Medical management"],
            "if_retinal": ["Vitrectomy", "Laser treatment"],
        },
        "doctor_reasoning": "Gradual vision loss in elderly → likely cataract → Phaco with IOL. Sudden loss → Emergency - retinal issue or glaucoma."
    },

    "cataract": {
        "specialty": "Ophthalmology",
        "treatment": ["Phacoemulsification with IOL", "SICS with IOL"],
        "implant_needed": True,
        "package_codes": {
            "phaco": ["1842-SV001A"],
            "sics": ["1842-SV001B"],
        },
        "doctor_reasoning": "Cataract surgery: Phaco is gold standard. SICS for hard cataracts. Always includes IOL implant."
    },

    # ── Urology Pathways ────────────────────────────────────────────────
    "kidney stone": {
        "specialty": "Urology",
        "synonyms": ["renal calculus", "ureteric stone", "urolithiasis"],
        "treatment_by_size": {
            "small_<5mm": ["Medical expulsive therapy"],
            "medium_5-10mm": ["URSL", "ESWL"],
            "large_>10mm": ["PCNL", "Open surgery if very large"],
        },
        "package_codes": {
            "ursl": ["1877-SV003A"],
            "pcnl": ["1877-SV004A"],
            "eswl": ["1877-SV002A"],
        },
        "doctor_reasoning": "Stone size determines treatment. <5mm → Medical. 5-10mm → URSL. >10mm or staghorn → PCNL."
    },

    # ── OBG Pathways ────────────────────────────────────────────────────
    "pregnancy delivery": {
        "specialty": "Obstetrics and Gynaecology",
        "options": ["Normal vaginal delivery", "Cesarean section"],
        "indications_for_cs": ["Previous CS", "Fetal distress", "CPD", "Malpresentation"],
        "package_codes": {
            "nvd": ["1872-OB001A"],
            "lscs": ["1872-SV001A"],
        },
        "doctor_reasoning": "Delivery mode based on obstetric factors. Previous CS, CPD, fetal distress → LSCS. Otherwise → NVD."
    },

    "fibroid": {
        "specialty": "Obstetrics and Gynaecology",
        "synonyms": ["uterine fibroid", "myoma", "leiomyoma"],
        "treatment_pathway": {
            "if_symptomatic": ["Myomectomy", "Hysterectomy"],
            "if_fertility_desired": ["Myomectomy"],
            "if_completed_family": ["Hysterectomy"],
        },
        "doctor_reasoning": "Fibroid treatment depends on symptoms and fertility plans. Young patient → Myomectomy. Completed family → Hysterectomy."
    },
}


# ══════════════════════════════════════════════════════════════════════════════
# SYMPTOM TO PACKAGE DIRECT MAPPING
# ══════════════════════════════════════════════════════════════════════════════

SYMPTOM_PACKAGE_MAPPING: Dict[str, List[Dict]] = {
    "chest pain": [
        {"code": "1831-MC023A", "name": "Coronary Angiography",
            "reason": "First step: Diagnose cause of chest pain", "priority": 1},
        {"code": "2831-MC026A",
            "name": "PTCA (Angioplasty)", "reason": "If blockage found on angiography", "priority": 2},
        {"code": "1831-MC020A", "name": "Systemic Thrombolysis",
            "reason": "If acute MI confirmed", "priority": 2},
        {"code": "2832-SV018A", "name": "CABG",
            "reason": "If multiple vessel disease or left main disease", "priority": 3},
        {"code": "1831-MG038A", "name": "Congestive Heart Failure",
            "reason": "If heart failure is the cause", "priority": 2},
    ],
    "heart attack": [
        {"code": "1831-MC020A",
            "name": "Systemic Thrombolysis (for MI)", "reason": "Emergency: Dissolve clot if within window", "priority": 1},
        {"code": "2831-MC026A",
            "name": "PTCA (Primary Angioplasty)", "reason": "Primary PCI for STEMI", "priority": 1},
        {"code": "1831-MC023A", "name": "Coronary Angiography",
            "reason": "Assess coronary anatomy post-MI", "priority": 2},
        {"code": "2832-SV018A", "name": "CABG",
            "reason": "If triple vessel disease", "priority": 3},
    ],
    "breathlessness": [
        {"code": "1831-MG038A", "name": "Congestive Heart Failure",
            "reason": "If cardiac cause confirmed", "priority": 1},
        {"code": "1831-MC023A", "name": "Coronary Angiography",
            "reason": "To rule out ischemic cause", "priority": 2},
    ],
    "stomach pain": [
        {"code": "1851-SV001A", "name": "Appendectomy - Lap",
            "reason": "If right lower quadrant pain (appendicitis)", "priority": 1},
        {"code": "1851-SV002A", "name": "Cholecystectomy - Lap",
            "reason": "If right upper quadrant pain (gallstones)", "priority": 1},
    ],
    "eye problem": [
        {"code": "1842-SV001A", "name": "Cataract Surgery - Phaco",
            "reason": "If gradual vision loss with lens opacity", "priority": 1},
    ],
    "difficulty urinating": [
        {"code": "1877-SV005A", "name": "TURP",
            "reason": "For benign prostatic hyperplasia", "priority": 1},
    ],
    "blood in urine": [
        {"code": "1877-SV003A", "name": "Cystoscopy",
            "reason": "To evaluate source of hematuria", "priority": 1},
    ],
}


def get_clinical_pathway(symptom: str) -> Dict:
    """Get clinical pathway for a symptom or condition."""
    symptom_lower = symptom.lower().strip()

    # Direct match
    if symptom_lower in CLINICAL_PATHWAYS:
        return CLINICAL_PATHWAYS[symptom_lower]

    # Check synonyms in pathways
    for condition, pathway in CLINICAL_PATHWAYS.items():
        if "synonyms" in pathway:
            if symptom_lower in [s.lower() for s in pathway["synonyms"]]:
                return pathway
        if symptom_lower in condition or condition in symptom_lower:
            return pathway

    return {}


def get_packages_for_symptom(symptom: str) -> List[Dict]:
    """Get recommended packages for a symptom with doctor reasoning."""
    symptom_lower = symptom.lower().strip()

    # Direct match
    if symptom_lower in SYMPTOM_PACKAGE_MAPPING:
        return SYMPTOM_PACKAGE_MAPPING[symptom_lower]

    # Partial match
    for key, packages in SYMPTOM_PACKAGE_MAPPING.items():
        if key in symptom_lower or symptom_lower in key:
            return packages

    return []

    """Get relevant specialties for a medical term."""
    term_lower = term.lower()
    specialties = set()

    # Direct lookup
    if term_lower in SPECIALTY_MAPPING:
        specialties.update(SPECIALTY_MAPPING[term_lower])

    # Check if term contains any mapped keyword
    for keyword, specs in SPECIALTY_MAPPING.items():
        if keyword in term_lower or term_lower in keyword:
            specialties.update(specs)

    # Check synonyms
    for canonical, synonyms in MEDICAL_SYNONYMS.items():
        if term_lower in synonyms or any(syn in term_lower for syn in synonyms):
            if canonical in SPECIALTY_MAPPING:
                specialties.update(SPECIALTY_MAPPING[canonical])

    return list(specialties)


def is_surgical_term(term: str) -> bool:
    """Check if a term indicates a surgical procedure."""
    term_lower = term.lower()

    # Check for surgical suffixes
    for suffix in SURGICAL_SUFFIXES:
        if term_lower.endswith(suffix):
            return True

    # Check for surgical indicators
    for indicator in SURGERY_INDICATORS:
        if indicator in term_lower:
            return True

    return False


def is_medical_management_term(term: str) -> bool:
    """Check if a term indicates medical management (non-surgical)."""
    term_lower = term.lower()

    for indicator in MEDICAL_MANAGEMENT_INDICATORS:
        if indicator in term_lower:
            return True

    return False


def expand_synonyms(term: str) -> List[str]:
    """Expand a term to include all its synonyms."""
    term_lower = term.lower()
    result = [term]

    # Check if term is a synonym
    for canonical, synonyms in MEDICAL_SYNONYMS.items():
        if term_lower == canonical or term_lower in synonyms:
            result.extend(synonyms)
            result.append(canonical)

    return list(set(result))


def get_implant_types_for_procedure(procedure: str) -> List[str]:
    """Get expected implant types for a procedure."""
    procedure_lower = procedure.lower()

    for proc_type, implant_types in IMPLANT_PROCEDURES.items():
        if proc_type in procedure_lower:
            return implant_types

    return []


def needs_extended_los(case_description: str) -> bool:
    """Check if a case likely needs extended length of stay."""
    case_lower = case_description.lower()

    for trigger in EXTENDED_LOS_TRIGGERS:
        if trigger in case_lower:
            return True

    return False


def classify_case_type(
    diagnosis: str,
    procedures: List[str],
    surgery_name: str = "",
) -> Tuple[str, str]:
    """
    Classify a case as surgical or medical management.

    Returns:
        Tuple of (case_type, reasoning)
    """
    all_terms = [diagnosis] + procedures + [surgery_name]
    all_text = " ".join(all_terms).lower()

    # Check for explicit surgery
    if surgery_name:
        return "surgical", f"Surgery specified: {surgery_name}"

    # Check for surgical terms
    for term in all_terms:
        if is_surgical_term(term):
            return "surgical", f"Surgical procedure indicated: {term}"

    # Check for medical management terms
    for term in all_terms:
        if is_medical_management_term(term):
            return "medical_management", f"Medical management indicated: {term}"

    # Default based on diagnosis
    if any(ind in all_text for ind in SURGERY_INDICATORS):
        return "surgical", "Surgery indicators found in case description"

    return "unknown", "Case type could not be determined"


def get_related_packages_hint(
    diagnosis: str,
    specialties: List[str],
) -> Dict[str, List[str]]:
    """
    Get hints about what package types to look for.

    Returns dict with keys: 'search_terms', 'package_types', 'warnings'
    """
    search_terms = expand_synonyms(diagnosis)

    # Add specialty-specific terms
    for spec in specialties:
        spec_lower = spec.lower()
        for keyword, specs in SPECIALTY_MAPPING.items():
            if any(spec_lower in s.lower() for s in specs):
                search_terms.append(keyword)

    package_types = ["regular"]

    # Check for implant needs
    if get_implant_types_for_procedure(diagnosis):
        package_types.append("implant")

    # Check for extended LOS
    if needs_extended_los(diagnosis):
        package_types.append("extended_los")

    warnings = []

    # Add warnings for ambiguous cases
    if is_surgical_term(diagnosis) and is_medical_management_term(diagnosis):
        warnings.append(
            "Case has both surgical and medical indicators - verify case type")

    return {
        "search_terms": list(set(search_terms)),
        "package_types": package_types,
        "warnings": warnings,
    }
