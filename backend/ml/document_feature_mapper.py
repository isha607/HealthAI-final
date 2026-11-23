"""
HealthAI — Phase 6: Document to Model Feature Mapper
=====================================================
Safely maps extracted laboratory parameters, document header metadata,
and patient profiles to model-specific feature requirements.

CRITICAL RULES:
- Never invent missing values.
- Never use silent defaults unless explicitly requested or verified.
- Clearly distinguish extracted values from missing required inputs.
- Keep model-specific mappings strictly isolated.
"""

from typing import Dict, Any, List, Optional
import logging

logger = logging.getLogger("healthai.mapper")


def map_lab_results_to_dict(lab_results: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """
    Normalizes a list of parsed laboratory results into a key-indexed lookup.
    """
    mapped = {}
    for r in lab_results:
        tname = r.get("test_name", "").lower()
        val = r.get("result_value")
        unit = r.get("unit", "")
        if val is None:
            continue

        if "fasting blood glucose" in tname or "fasting glucose" in tname:
            mapped["glucose_fasting"] = {"val": float(val), "name": r.get("test_name"), "unit": unit}
        elif "random" in tname or "postprandial" in tname or "rbs" in tname:
            mapped["glucose_random"] = {"val": float(val), "name": r.get("test_name"), "unit": unit}
        elif "glucose" in tname or "sugar" in tname:
            if "glucose" not in mapped:
                mapped["glucose"] = {"val": float(val), "name": r.get("test_name"), "unit": unit}
        elif "hba1c" in tname:
            mapped["hba1c"] = {"val": float(val), "name": r.get("test_name"), "unit": unit}
        elif "total cholesterol" in tname or "serum cholesterol" in tname:
            mapped["cholesterol_total"] = {"val": float(val), "name": r.get("test_name"), "unit": unit}
        elif "hdl" in tname:
            mapped["cholesterol_hdl"] = {"val": float(val), "name": r.get("test_name"), "unit": unit}
        elif "ldl" in tname:
            mapped["cholesterol_ldl"] = {"val": float(val), "name": r.get("test_name"), "unit": unit}
        elif "triglyceride" in tname or "tg" in tname:
            mapped["triglycerides"] = {"val": float(val), "name": r.get("test_name"), "unit": unit}
        elif "systolic" in tname:
            mapped["systolic_bp"] = {"val": float(val), "name": r.get("test_name"), "unit": unit}
        elif "diastolic" in tname:
            mapped["diastolic_bp"] = {"val": float(val), "name": r.get("test_name"), "unit": unit}

    return mapped


def map_document_to_diabetes(
    lab_map: Dict[str, Dict[str, Any]],
    meta: Dict[str, Any],
    profile: Optional[Any] = None
) -> Dict[str, Any]:
    """
    Diabetes Model Features:
    ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Age']
    """
    extracted = {}
    missing = []

    # 1. Glucose
    if "glucose_fasting" in lab_map:
        extracted["Glucose"] = {
            "value": lab_map["glucose_fasting"]["val"],
            "source": f"Extracted: {lab_map['glucose_fasting']['name']}",
            "unit": lab_map["glucose_fasting"]["unit"]
        }
    elif "glucose" in lab_map:
        extracted["Glucose"] = {
            "value": lab_map["glucose"]["val"],
            "source": f"Extracted: {lab_map['glucose']['name']}",
            "unit": lab_map["glucose"]["unit"]
        }
    elif "glucose_random" in lab_map:
        extracted["Glucose"] = {
            "value": lab_map["glucose_random"]["val"],
            "source": f"Extracted: {lab_map['glucose_random']['name']}",
            "unit": lab_map["glucose_random"]["unit"]
        }
    else:
        missing.append({
            "key": "Glucose",
            "label": "Fasting / Blood Glucose",
            "unit": "mg/dL",
            "type": "number",
            "min": 40,
            "max": 400,
            "description": "Fasting blood glucose level is required for diabetes risk evaluation."
        })

    # 2. BloodPressure (Diastolic BP in Pima dataset)
    if "diastolic_bp" in lab_map:
        extracted["BloodPressure"] = {
            "value": lab_map["diastolic_bp"]["val"],
            "source": f"Extracted: {lab_map['diastolic_bp']['name']}",
            "unit": "mmHg"
        }
    else:
        missing.append({
            "key": "BloodPressure",
            "label": "Diastolic Blood Pressure",
            "unit": "mmHg",
            "type": "number",
            "min": 30,
            "max": 150,
            "description": "Resting diastolic blood pressure (lower number)."
        })

    # 3. Age
    if meta.get("age"):
        extracted["Age"] = {
            "value": meta["age"],
            "source": "Extracted from document header",
            "unit": "years"
        }
    elif profile and getattr(profile, "age", None):
        extracted["Age"] = {
            "value": profile.age,
            "source": "From patient health profile",
            "unit": "years"
        }
    else:
        missing.append({
            "key": "Age",
            "label": "Patient Age",
            "unit": "years",
            "type": "number",
            "min": 1,
            "max": 120,
            "description": "Patient age in years."
        })

    # 4. Pregnancies
    gender = meta.get("gender") or (getattr(profile, "gender", None) if profile else None)
    if gender and str(gender).lower() in ["male", "m"]:
        extracted["Pregnancies"] = {
            "value": 0,
            "source": "Standard clinical rule: 0 for male patients",
            "unit": "count"
        }
    else:
        missing.append({
            "key": "Pregnancies",
            "label": "Number of Pregnancies",
            "unit": "count",
            "type": "number",
            "min": 0,
            "max": 20,
            "default": 0,
            "description": "Total number of times pregnant (enter 0 if none or not applicable)."
        })

    # 5. SkinThickness
    missing.append({
        "key": "SkinThickness",
        "label": "Triceps Skinfold Thickness",
        "unit": "mm",
        "type": "number",
        "min": 0,
        "max": 99,
        "default": 20.0,
        "description": "Triceps skinfold measurement in mm (typical reference median is 20 mm)."
    })

    return {
        "model_type": "diabetes",
        "model_name": "Diabetes Risk Assessment",
        "extracted_features": extracted,
        "missing_features": missing
    }


def map_document_to_cardiovascular(
    lab_map: Dict[str, Dict[str, Any]],
    meta: Dict[str, Any],
    profile: Optional[Any] = None
) -> Dict[str, Any]:
    """
    Cardiovascular (Framingham / AHA) Model Features:
    ['Age', 'Gender', 'SystolicBP', 'TotalCholesterol', 'HDLCholesterol', 'Smoking', 'OnBPTreatment']
    """
    extracted = {}
    missing = []

    # 1. Total Cholesterol
    if "cholesterol_total" in lab_map:
        extracted["TotalCholesterol"] = {
            "value": lab_map["cholesterol_total"]["val"],
            "source": f"Extracted: {lab_map['cholesterol_total']['name']}",
            "unit": "mg/dL"
        }
    else:
        missing.append({
            "key": "TotalCholesterol",
            "label": "Total Cholesterol",
            "unit": "mg/dL",
            "type": "number",
            "min": 100,
            "max": 500,
            "description": "Total blood serum cholesterol level."
        })

    # 2. HDL Cholesterol
    if "cholesterol_hdl" in lab_map:
        extracted["HDLCholesterol"] = {
            "value": lab_map["cholesterol_hdl"]["val"],
            "source": f"Extracted: {lab_map['cholesterol_hdl']['name']}",
            "unit": "mg/dL"
        }
    else:
        missing.append({
            "key": "HDLCholesterol",
            "label": "HDL Cholesterol ('Good' Cholesterol)",
            "unit": "mg/dL",
            "type": "number",
            "min": 15,
            "max": 120,
            "description": "High-density lipoprotein cholesterol level."
        })

    # 3. Systolic Blood Pressure
    if "systolic_bp" in lab_map:
        extracted["SystolicBP"] = {
            "value": int(lab_map["systolic_bp"]["val"]),
            "source": f"Extracted: {lab_map['systolic_bp']['name']}",
            "unit": "mmHg"
        }
    else:
        missing.append({
            "key": "SystolicBP",
            "label": "Systolic Blood Pressure",
            "unit": "mmHg",
            "type": "number",
            "min": 70,
            "max": 260,
            "description": "Resting systolic blood pressure (upper number)."
        })

    # 4. Age
    if meta.get("age"):
        extracted["Age"] = {
            "value": meta["age"],
            "source": "Extracted from document header",
            "unit": "years"
        }
    elif profile and getattr(profile, "age", None):
        extracted["Age"] = {
            "value": profile.age,
            "source": "From patient health profile",
            "unit": "years"
        }
    else:
        missing.append({
            "key": "Age",
            "label": "Patient Age",
            "unit": "years",
            "type": "number",
            "min": 18,
            "max": 120,
            "description": "Patient age in years."
        })

    # 5. Gender
    if meta.get("gender"):
        extracted["Gender"] = {
            "value": meta["gender"],
            "source": "Extracted from document header",
            "unit": "category"
        }
    elif profile and getattr(profile, "gender", None):
        extracted["Gender"] = {
            "value": profile.gender,
            "source": "From patient health profile",
            "unit": "category"
        }
    else:
        missing.append({
            "key": "Gender",
            "label": "Biological Sex",
            "unit": "",
            "type": "select",
            "options": ["Male", "Female"],
            "description": "Biological sex for Framingham cardiac risk calculation."
        })

    # 6. Smoking
    if profile and hasattr(profile, "smoking_status") and profile.smoking_status:
        smk = profile.smoking_status.lower() in ["yes", "current", "former"]
        extracted["Smoking"] = {
            "value": smk,
            "source": f"From patient health profile ({profile.smoking_status})",
            "unit": "boolean"
        }
    else:
        missing.append({
            "key": "Smoking",
            "label": "Current Smoker",
            "unit": "",
            "type": "select",
            "options": ["No", "Yes"],
            "description": "Regular tobacco or cigarette smoking status."
        })

    # 7. On Blood Pressure Treatment
    if profile and hasattr(profile, "hypertension") and profile.hypertension:
        extracted["OnBPTreatment"] = {
            "value": True,
            "source": "From patient health profile (Hypertension on record)",
            "unit": "boolean"
        }
    else:
        missing.append({
            "key": "OnBPTreatment",
            "label": "On Blood Pressure Medication",
            "unit": "",
            "type": "select",
            "options": ["No", "Yes"],
            "description": "Currently taking prescribed antihypertensive medications."
        })

    return {
        "model_type": "cardiovascular",
        "model_name": "Cardiovascular Risk Assessment (Framingham/AHA)",
        "extracted_features": extracted,
        "missing_features": missing
    }


def map_document_to_alzheimer(
    lab_map: Dict[str, Dict[str, Any]],
    meta: Dict[str, Any],
    profile: Optional[Any] = None
) -> Dict[str, Any]:
    """
    Alzheimer's Model Features:
    32 features total (clinical biomarkers from lab/vitals, profile demographics, and cognitive scores).
    Routine laboratory reports cannot measure cognitive scores (MMSE, ADL, etc.), so they are flagged as missing.
    """
    extracted = {}
    missing = []

    # Blood & Vital markers
    if "cholesterol_total" in lab_map:
        extracted["CholesterolTotal"] = {
            "value": lab_map["cholesterol_total"]["val"],
            "source": f"Extracted: {lab_map['cholesterol_total']['name']}",
            "unit": "mg/dL"
        }
    else:
        missing.append({"key": "CholesterolTotal", "label": "Total Cholesterol", "unit": "mg/dL", "type": "number", "min": 100, "max": 400})

    if "cholesterol_ldl" in lab_map:
        extracted["CholesterolLDL"] = {
            "value": lab_map["cholesterol_ldl"]["val"],
            "source": f"Extracted: {lab_map['cholesterol_ldl']['name']}",
            "unit": "mg/dL"
        }
    else:
        missing.append({"key": "CholesterolLDL", "label": "LDL Cholesterol", "unit": "mg/dL", "type": "number", "min": 30, "max": 250})

    if "cholesterol_hdl" in lab_map:
        extracted["CholesterolHDL"] = {
            "value": lab_map["cholesterol_hdl"]["val"],
            "source": f"Extracted: {lab_map['cholesterol_hdl']['name']}",
            "unit": "mg/dL"
        }
    else:
        missing.append({"key": "CholesterolHDL", "label": "HDL Cholesterol", "unit": "mg/dL", "type": "number", "min": 20, "max": 120})

    if "triglycerides" in lab_map:
        extracted["CholesterolTriglycerides"] = {
            "value": lab_map["triglycerides"]["val"],
            "source": f"Extracted: {lab_map['triglycerides']['name']}",
            "unit": "mg/dL"
        }
    else:
        missing.append({"key": "CholesterolTriglycerides", "label": "Triglycerides", "unit": "mg/dL", "type": "number", "min": 30, "max": 500})

    if "systolic_bp" in lab_map:
        extracted["SystolicBP"] = {"value": int(lab_map["systolic_bp"]["val"]), "source": "Extracted: Systolic BP", "unit": "mmHg"}
    else:
        missing.append({"key": "SystolicBP", "label": "Systolic Blood Pressure", "unit": "mmHg", "type": "number", "min": 80, "max": 220})

    if "diastolic_bp" in lab_map:
        extracted["DiastolicBP"] = {"value": int(lab_map["diastolic_bp"]["val"]), "source": "Extracted: Diastolic BP", "unit": "mmHg"}
    else:
        missing.append({"key": "DiastolicBP", "label": "Diastolic Blood Pressure", "unit": "mmHg", "type": "number", "min": 40, "max": 140})

    # Profile & Demographic markers
    age_val = meta.get("age") or (getattr(profile, "age", None) if profile else None)
    if age_val:
        extracted["Age"] = {"value": age_val, "source": "Extracted / Profile", "unit": "years"}
    else:
        missing.append({"key": "Age", "label": "Age", "unit": "years", "type": "number", "min": 50, "max": 110})

    gender_val = meta.get("gender") or (getattr(profile, "gender", None) if profile else None)
    if gender_val:
        extracted["Gender"] = {"value": 0 if str(gender_val).lower().startswith("m") else 1, "source": "Extracted / Profile", "unit": "category"}
    else:
        missing.append({"key": "Gender", "label": "Gender (0=Male, 1=Female)", "unit": "", "type": "select", "options": ["0", "1"]})

    bmi_val = getattr(profile, "bmi", None) if profile else None
    if bmi_val:
        extracted["BMI"] = {"value": float(bmi_val), "source": "From patient profile", "unit": "kg/m²"}
    else:
        missing.append({"key": "BMI", "label": "BMI", "unit": "kg/m²", "type": "number", "min": 15, "max": 50})

    # Cognitive markers are NEVER present in standard blood tests — must be filled by user
    missing.append({
        "key": "MMSE",
        "label": "MMSE (Mini-Mental State Exam score, 0–30)",
        "unit": "points",
        "type": "number",
        "min": 0,
        "max": 30,
        "default": 27.0,
        "description": "Standard cognitive screening test score (normal range is 24–30)."
    })
    missing.append({
        "key": "FunctionalAssessment",
        "label": "Functional Assessment Score (0–10)",
        "unit": "points",
        "type": "number",
        "min": 0,
        "max": 10,
        "default": 7.5,
        "description": "Functional independence evaluation score."
    })
    missing.append({
        "key": "ADL",
        "label": "Activities of Daily Living Score (0–10)",
        "unit": "points",
        "type": "number",
        "min": 0,
        "max": 10,
        "default": 8.0,
        "description": "Self-care and daily activity capabilities."
    })
    missing.append({
        "key": "MemoryComplaints",
        "label": "Memory Complaints Noticed",
        "unit": "",
        "type": "select",
        "options": ["No", "Yes"],
        "description": "Subjective memory complaints reported by patient or family."
    })
    missing.append({
        "key": "FamilyHistoryAlzheimers",
        "label": "Family History of Alzheimer's",
        "unit": "",
        "type": "select",
        "options": ["No", "Yes"],
        "description": "First-degree relative diagnosed with Alzheimer's disease."
    })

    return {
        "model_type": "alzheimer",
        "model_name": "Alzheimer's Cognitive & Clinical Risk Model",
        "extracted_features": extracted,
        "missing_features": missing
    }


def map_document_to_model(
    model_type: str,
    lab_results: List[Dict[str, Any]],
    raw_text: str = "",
    profile: Optional[Any] = None
) -> Dict[str, Any]:
    """
    Main entry point for mapping parsed document results to the requested model.
    """
    from backend.ocr.document_extractor import extract_report_metadata

    lab_map = map_lab_results_to_dict(lab_results)
    meta = extract_report_metadata(raw_text)

    mtype = model_type.lower().strip()
    if mtype in ["diabetes", "diab"]:
        return map_document_to_diabetes(lab_map, meta, profile)
    elif mtype in ["cardiovascular", "cardio", "framingham", "cv"]:
        return map_document_to_cardiovascular(lab_map, meta, profile)
    elif mtype in ["alzheimer", "alzheimers", "alz"]:
        return map_document_to_alzheimer(lab_map, meta, profile)
    else:
        raise ValueError(f"Unsupported model type '{model_type}'. Choose 'diabetes', 'cardiovascular', or 'alzheimer'.")
