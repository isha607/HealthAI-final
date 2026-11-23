import re
from typing import List, Dict, Any, Optional

# Standard clinical reference ranges
LAB_REFERENCE_RANGES = {
    "glucose": {"name": "Fasting Blood Glucose", "low": 70.0, "high": 99.0, "unit": "mg/dL", 
                "high_exp": "This value is above the normal reference range (70–99 mg/dL). Elevated fasting blood glucose can be associated with increased diabetes risk or insulin resistance and should be evaluated by a healthcare professional.",
                "low_exp": "This value is below the normal reference range (70–99 mg/dL). Low glucose (hypoglycemia) can cause dizziness or fatigue and should be reviewed with a clinician."},
    "glucose_random": {"name": "Random Blood Glucose", "low": 70.0, "high": 140.0, "unit": "mg/dL",
                       "high_exp": "This value is above the standard random glucose reference threshold (<140 mg/dL). Elevated levels can indicate impaired glucose tolerance.",
                       "low_exp": "This value is below standard reference ranges and indicates hypoglycemia."},
    "hba1c": {"name": "Hemoglobin A1c (HbA1c)", "low": 4.0, "high": 5.6, "unit": "%",
              "high_exp": "This value is above the normal reference range (<5.7%). HbA1c between 5.7–6.4% can be associated with pre-diabetes, and >=6.5% with diabetes. Medical evaluation is strongly recommended.",
              "low_exp": "This value is below typical ranges, which is uncommon and should be evaluated in context of overall red blood cell health."},
    "cholesterol_total": {"name": "Total Cholesterol", "low": 125.0, "high": 200.0, "unit": "mg/dL",
                          "high_exp": "This value is above the desirable clinical range (<200 mg/dL). Elevated total cholesterol can be associated with increased cardiovascular risk and plaque development.",
                          "low_exp": "Total cholesterol is below typical levels; discuss with your physician."},
    "cholesterol_hdl": {"name": "HDL Cholesterol ('Good')", "low": 40.0, "high": 60.0, "unit": "mg/dL",
                        "high_exp": "Higher HDL levels are generally protective for cardiovascular health.",
                        "low_exp": "HDL is below the recommended minimum (>40 mg/dL for men, >50 mg/dL for women). Lower HDL can be associated with higher cardiovascular risk."},
    "cholesterol_ldl": {"name": "LDL Cholesterol ('Bad')", "low": 50.0, "high": 100.0, "unit": "mg/dL",
                        "high_exp": "This value is above optimal levels (<100 mg/dL). Elevated LDL cholesterol can be associated with atherosclerotic arterial risk and warrants dietary and clinical review.",
                        "low_exp": "LDL is well within the low/optimal range."},
    "triglycerides": {"name": "Triglycerides", "low": 50.0, "high": 150.0, "unit": "mg/dL",
                      "high_exp": "Triglycerides are above normal levels (<150 mg/dL). Elevated triglycerides can be associated with metabolic syndrome and cardiovascular risk.",
                      "low_exp": "Triglycerides are within the low/normal range."},
    "hemoglobin": {"name": "Hemoglobin", "low": 12.0, "high": 17.5, "unit": "g/dL",
                   "high_exp": "Hemoglobin is above typical levels. High values can be associated with dehydration or chronic respiratory conditions.",
                   "low_exp": "Hemoglobin is below normal reference ranges. Lower values can be associated with anemia and fatigue, requiring medical evaluation."},
    "blood_pressure_systolic": {"name": "Systolic Blood Pressure", "low": 90.0, "high": 120.0, "unit": "mmHg",
                                "high_exp": "Systolic blood pressure is elevated (>120 mmHg). Values between 120-139 indicate pre-hypertension, and >=140 indicate hypertension.",
                                "low_exp": "Systolic blood pressure is low (<90 mmHg). Can be associated with hypotension, lightheadedness, or dehydration."},
    "blood_pressure_diastolic": {"name": "Diastolic Blood Pressure", "low": 60.0, "high": 80.0, "unit": "mmHg",
                                 "high_exp": "Diastolic pressure is elevated (>80 mmHg). Persistent elevation can indicate hypertension risk.",
                                 "low_exp": "Diastolic blood pressure is below 60 mmHg; monitor for symptoms of dizziness."}
}

def parse_lab_text(text: str) -> List[Dict[str, Any]]:
    """
    Extracts laboratory test values from raw text or OCR output, compares
    against clinical reference thresholds, and generates status and cautious explanations.
    """
    results = []
    lines = text.splitlines()

    for key, ref in LAB_REFERENCE_RANGES.items():
        # Match test name in line
        pattern = re.compile(rf"(?i)(?:{key}|{ref['name']}|{key.replace('_', ' ')})[\s:=-]+([0-9]+(?:\.[0-9]+)?)")
        match = pattern.search(text)
        if not match:
            # Try searching individual lines
            for line in lines:
                if any(w in line.lower() for w in key.split("_")):
                    num_match = re.search(r"([0-9]+(?:\.[0-9]+)?)", line)
                    if num_match:
                        match = num_match
                        break
        
        if match:
            val = float(match.group(1))
            status = "NORMAL"
            explanation = "This result is within the standard reference range."

            if ref["high"] is not None and val > ref["high"]:
                status = "HIGH"
                explanation = ref["high_exp"]
            elif ref["low"] is not None and val < ref["low"]:
                status = "LOW"
                explanation = ref["low_exp"]

            results.append({
                "test_name": ref["name"],
                "result_value": val,
                "unit": ref["unit"],
                "reference_low": ref["low"],
                "reference_high": ref["high"],
                "status": status,
                "clinical_explanation": explanation
            })

    return results
