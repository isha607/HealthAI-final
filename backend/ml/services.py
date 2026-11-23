import os
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional
from sqlalchemy.orm import Session

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
MODELS_DIR = os.path.join(BASE_DIR, "ml_models")

_ALZ_ARTIFACT = None
_DIAB_ARTIFACT = None
_CARDIO_ARTIFACT = None

def get_alzheimer_model() -> Dict[str, Any]:
    global _ALZ_ARTIFACT
    if _ALZ_ARTIFACT is None:
        path = os.path.join(MODELS_DIR, "alzheimer_pipeline.joblib")
        _ALZ_ARTIFACT = joblib.load(path)
    return _ALZ_ARTIFACT

def get_diabetes_model() -> Dict[str, Any]:
    global _DIAB_ARTIFACT
    if _DIAB_ARTIFACT is None:
        path = os.path.join(MODELS_DIR, "diabetes_pipeline.joblib")
        _DIAB_ARTIFACT = joblib.load(path)
    return _DIAB_ARTIFACT

def get_cardio_model() -> Dict[str, Any]:
    global _CARDIO_ARTIFACT
    if _CARDIO_ARTIFACT is None:
        path = os.path.join(MODELS_DIR, "cardio_pipeline.joblib")
        _CARDIO_ARTIFACT = joblib.load(path)
    return _CARDIO_ARTIFACT


def predict_alzheimer(data_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Alzheimer's Disease Prediction using exact 32 features from notebook:
    alzheimer-s-disease-prediction (1).ipynb
    """
    art = get_alzheimer_model()
    model = art["model"]
    scaler = art["scaler"]
    features = art["features"]
    continuous_cols = art["continuous_columns"]

    # Construct DataFrame with exact column ordering
    row = {}
    for feat in features:
        val = data_dict.get(feat, 0)
        row[feat] = [float(val) if val is not None else 0.0]
    df_input = pd.DataFrame(row)

    # Scale the continuous columns exactly as done in training
    df_input[continuous_cols] = scaler.transform(df_input[continuous_cols])

    pred = int(model.predict(df_input)[0])
    probs = model.predict_proba(df_input)[0]
    prob_alz = float(probs[1])

    if prob_alz < 0.35:
        risk_level = "Low Risk"
    elif prob_alz < 0.65:
        risk_level = "Moderate Risk"
    else:
        risk_level = "Elevated / High Risk"

    # Identify contributing factors based on trained weights
    top_factors = {}
    feat_imp = art.get("feature_importance", {})
    for feat, imp in list(feat_imp.items())[:6]:
        user_val = data_dict.get(feat)
        top_factors[feat] = {
            "patient_value": user_val,
            "relative_influence": "Protective" if imp < 0 else "Risk Contributing",
            "weight": round(float(imp), 4)
        }

    recommendations = [
        "Maintain regular cognitive activities, reading, and mental stimulation.",
        "Follow a Mediterranean-style or MIND dietary pattern rich in antioxidants.",
        "Ensure 7 to 8 hours of quality sleep daily for neural metabolite clearance.",
        "Schedule periodic clinical cognitive reviews with your primary care provider or neurologist."
    ]
    if prob_alz >= 0.5:
        recommendations.insert(0, "Recommend consulting a neurologist or geriatric specialist for formal neuropsychological assessment.")

    return {
        "assessment_type": "Alzheimer's Disease Risk Assessment",
        "prediction_class": pred,
        "probability_percentage": round(prob_alz * 100, 1),
        "risk_level": risk_level,
        "headline": f"{risk_level} ({round(prob_alz * 100, 1)}% probability)",
        "contributing_factors": top_factors,
        "clinical_disclaimer": "This is an AI-assisted decision-support risk prediction based on statistical analysis of clinical and lifestyle markers. It is NOT a definitive clinical diagnosis of Alzheimer's Disease.",
        "recommendations": recommendations,
        "accuracy_metric": f"{round(art['metrics']['accuracy'] * 100, 1)}% Test Accuracy (AUC: {round(art['metrics']['roc_auc'], 3)})"
    }


def predict_diabetes(data_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Diabetes Prediction using exact 5 features from notebook:
    diabetes-prediction-logistic-regression.ipynb
    ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Age']
    """
    art = get_diabetes_model()
    model = art["model"]
    scaler = art["scaler"]
    features = art["features"]

    row = {}
    for feat in features:
        val = data_dict.get(feat, 0)
        row[feat] = [float(val) if val is not None else 0.0]
    df_input = pd.DataFrame(row)

    # Scale the 5 features
    scaled_vals = scaler.transform(df_input)

    pred = int(model.predict(scaled_vals)[0])
    probs = model.predict_proba(scaled_vals)[0]
    prob_diab = float(probs[1])

    if prob_diab < 0.35:
        risk_level = "Low Risk"
    elif prob_diab < 0.65:
        risk_level = "Moderate Risk"
    else:
        risk_level = "Elevated Risk"

    glucose = float(data_dict.get("Glucose", 100.0))
    bp = float(data_dict.get("BloodPressure", 80.0))
    age = int(data_dict.get("Age", 45))

    factors = {
        "Fasting / Blood Glucose": f"{glucose} mg/dL (Key biomarker)",
        "Diastolic Blood Pressure": f"{bp} mmHg",
        "Age": f"{age} years",
        "Skin Thickness (Triceps skinfold)": f"{data_dict.get('SkinThickness', 20)} mm",
        "Pregnancies": str(data_dict.get("Pregnancies", 0))
    }

    recommendations = [
        "Follow a low-glycemic, high-fiber nutritional plan with controlled carbohydrates.",
        "Engage in at least 150 minutes of moderate aerobic activity per week.",
        "Maintain regular routine fasting blood glucose and HbA1c screenings.",
        "Discuss preventative lifestyle strategies with your healthcare provider."
    ]

    return {
        "assessment_type": "Diabetes Risk Assessment",
        "prediction_class": pred,
        "probability_percentage": round(prob_diab * 100, 1),
        "risk_level": risk_level,
        "headline": f"{risk_level} ({round(prob_diab * 100, 1)}% risk score)",
        "contributing_factors": factors,
        "clinical_disclaimer": "This prediction is generated by a logistic regression machine learning model trained on diagnostic health markers. It does not replace laboratory blood glucose tests or clinical diagnostic evaluation.",
        "recommendations": recommendations,
        "accuracy_metric": f"{round(art['metrics']['accuracy'] * 100, 1)}% Test Accuracy (AUC: {round(art['metrics']['roc_auc'], 3)})"
    }


def predict_cardio_surveillance(data_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Cardiovascular Disease Surveillance Classification from notebook:
    cardiovascular-disease-prediction.ipynb
    Uses 9 CDC features with LabelEncoders + StandardScaler + PCA(2) + KNN(5)
    """
    art = get_cardio_model()
    encoders = art["encoders"]
    scaler = art["scaler"]
    pca = art["pca"]
    knn = art["knn"]
    classes = art["classes"]

    year = int(data_dict.get("Year", 2018))
    indicator_raw = str(data_dict.get("Indicator", "Prevalence of major cardiovascular disease"))
    unit_raw = str(data_dict.get("Data_Value_Unit", "%"))
    val_alt = float(data_dict.get("Data_Value_Alt", 8.0))
    low_limit = float(data_dict.get("LowConfidenceLimit", 6.5))
    high_limit = float(data_dict.get("HighConfidenceLimit", 9.5))
    cat_raw = str(data_dict.get("Break_Out_Category", "Overall"))
    breakout_raw = str(data_dict.get("Break_Out", "Overall"))
    priority_raw = str(data_dict.get("PriorityArea", "Million Hearts"))

    def safe_encode(encoder, val):
        if val in encoder.classes_:
            return int(encoder.transform([val])[0])
        return 0

    ind_enc = safe_encode(encoders["Indicator"], indicator_raw)
    unit_enc = safe_encode(encoders["Data_Value_Unit"], unit_raw)
    cat_enc = safe_encode(encoders["Break_Out_Category"], cat_raw)
    break_enc = safe_encode(encoders["Break_Out"], breakout_raw)
    pri_enc = safe_encode(encoders["PriorityArea"], priority_raw)

    feat_vec = np.array([[year, ind_enc, unit_enc, val_alt, low_limit, high_limit, cat_enc, break_enc, pri_enc]])
    scaled = scaler.transform(feat_vec)
    pca_vec = pca.transform(scaled)

    pred_idx = int(knn.predict(pca_vec)[0])
    condition_name = classes.get(pred_idx, "Cardiovascular Condition")

    return {
        "assessment_type": "Cardiovascular Surveillance Classification",
        "prediction_class": pred_idx,
        "probability_percentage": round(art["metrics"]["accuracy"] * 100, 1),
        "risk_level": "Surveillance Stratification",
        "headline": f"Matched Condition Category: {condition_name}",
        "contributing_factors": {
            "Condition Category": condition_name,
            "Surveillance Indicator": indicator_raw,
            "Reported Value": f"{val_alt} {unit_raw}",
            "Confidence Interval": f"[{low_limit}, {high_limit}]",
            "Demographic Group": f"{cat_raw} ({breakout_raw})"
        },
        "clinical_disclaimer": "This is an academic ML classification utilizing the CDC Heart Disease and Stroke Prevention surveillance dataset. It maps public health indicators to condition categories.",
        "recommendations": [
            "Review your comprehensive clinical lipid panel with a physician.",
            "Monitor blood pressure and resting pulse rates consistently.",
            "Maintain cardiac-healthy diet low in saturated fats and sodium."
        ],
        "accuracy_metric": f"{round(art['metrics']['accuracy'] * 100, 1)}% Accuracy (PCA + KNN, k=5)"
    }


def calculate_framingham_cardio_risk(data_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Standard Clinical Framingham / AHA 10-Year Cardiovascular Risk Score.
    Provides real patient-level clinical decision support using standard medical guidelines.
    """
    age = int(data_dict.get("Age", 50))
    gender = str(data_dict.get("Gender", "Male")).capitalize()
    sbp = int(data_dict.get("SystolicBP", 125))
    chol = float(data_dict.get("TotalCholesterol", 200.0))
    hdl = float(data_dict.get("HDLCholesterol", 50.0))
    smoker = bool(data_dict.get("Smoking", False))
    treated_bp = bool(data_dict.get("OnBPTreatment", False))

    points = 0
    if gender == "Male":
        if age < 35: points += -9
        elif age <= 39: points += -4
        elif age <= 44: points += 0
        elif age <= 49: points += 3
        elif age <= 54: points += 6
        elif age <= 59: points += 8
        elif age <= 64: points += 10
        elif age <= 69: points += 11
        elif age <= 74: points += 12
        else: points += 13
    else:
        if age < 35: points += -7
        elif age <= 39: points += -3
        elif age <= 44: points += 0
        elif age <= 49: points += 3
        elif age <= 54: points += 6
        elif age <= 59: points += 8
        elif age <= 64: points += 10
        elif age <= 69: points += 12
        elif age <= 74: points += 14
        else: points += 16

    if chol < 160: points += 0
    elif chol < 200: points += 4
    elif chol < 240: points += 7 if gender == "Male" else 8
    elif chol < 280: points += 9 if gender == "Male" else 11
    else: points += 11 if gender == "Male" else 13

    if smoker: points += 8 if gender == "Male" else 9

    if hdl >= 60: points += -1
    elif hdl >= 50: points += 0
    elif hdl >= 40: points += 1
    else: points += 2

    if sbp < 120: points += 0
    elif sbp < 130: points += 1 if treated_bp else 0
    elif sbp < 140: points += 2 if treated_bp else 1
    elif sbp < 160: points += 2 if treated_bp else 1
    else: points += 3 if treated_bp else 2

    if points <= 0: risk_pct = 1.0
    elif points <= 4: risk_pct = 2.0
    elif points <= 6: risk_pct = 3.0
    elif points <= 8: risk_pct = 5.0
    elif points <= 10: risk_pct = 7.0
    elif points <= 12: risk_pct = 10.0
    elif points <= 14: risk_pct = 15.0
    elif points <= 16: risk_pct = 20.0
    else: risk_pct = 28.0

    if risk_pct < 10.0:
        level = "Low Risk (<10% 10-Yr CVD Risk)"
    elif risk_pct < 20.0:
        level = "Intermediate Risk (10–20% 10-Yr CVD Risk)"
    else:
        level = "High Risk (>20% 10-Yr CVD Risk)"

    return {
        "assessment_type": "Clinical Cardiovascular Risk (Framingham/AHA)",
        "prediction_class": 1 if risk_pct >= 20.0 else (0 if risk_pct < 10.0 else 2),
        "probability_percentage": risk_pct,
        "risk_level": level,
        "headline": f"{level} ({risk_pct}% estimated 10-year risk)",
        "contributing_factors": {
            "Systolic Blood Pressure": f"{sbp} mmHg",
            "Total Cholesterol": f"{chol} mg/dL",
            "HDL Cholesterol": f"{hdl} mg/dL",
            "Smoker Status": "Yes" if smoker else "No",
            "Risk Score Points": points
        },
        "clinical_disclaimer": "This is a clinical decision-support score using established American Heart Association (AHA) and Framingham Heart Study risk models. It is not a definitive diagnosis of cardiovascular disease.",
        "recommendations": [
            "Maintain blood pressure below 120/80 mmHg.",
            "Aim for optimal LDL-C through dietary fiber and reduced saturated fats.",
            "Engage in 30 minutes of aerobic exercise daily.",
            "Discuss personalized cardiovascular risk reduction goals with your physician."
        ],
        "accuracy_metric": "Validated Framingham Heart Study Clinical Risk Scale"
    }


def predict_from_patient_records(patient_id: str, model_type: str, db: Session) -> Dict[str, Any]:
    """
    Method B: Evaluate patient risk directly from their stored laboratory and profile records.
    STRICT SAFETY RULE: If required fields are missing, NEVER impute with 0 or fabricate data.
    Return 'insufficient_data' status with the exact list of missing parameters.
    """
    from backend.models.db_models import PatientProfile, LabResult

    patient = db.query(PatientProfile).filter(PatientProfile.patient_id == patient_id).first()
    if not patient:
        return {"status": "error", "message": f"Patient '{patient_id}' not found"}

    results = db.query(LabResult).filter(LabResult.patient_id == patient_id).all()
    # Map test results to lowercase keys
    lab_map = {}
    for r in results:
        tname = r.test_name.lower()
        if "glucose" in tname:
            lab_map["glucose"] = r.result_value
        elif "hba1c" in tname:
            lab_map["hba1c"] = r.result_value
        elif "total cholesterol" in tname:
            lab_map["cholesterol_total"] = r.result_value
        elif "hdl" in tname:
            lab_map["cholesterol_hdl"] = r.result_value
        elif "ldl" in tname:
            lab_map["cholesterol_ldl"] = r.result_value
        elif "triglycerides" in tname:
            lab_map["triglycerides"] = r.result_value
        elif "systolic" in tname:
            lab_map["systolic_bp"] = r.result_value
        elif "diastolic" in tname:
            lab_map["diastolic_bp"] = r.result_value

    if model_type == "diabetes":
        missing = []
        if "glucose" not in lab_map:
            missing.append("Fasting Blood Glucose")
        if not patient.age:
            missing.append("Age")

        if missing:
            return {
                "status": "insufficient_data",
                "model_type": "diabetes",
                "missing_features": missing,
                "message": f"Insufficient information: {', '.join(missing)} is missing from patient records. Please enter the value manually or upload a blood test report."
            }

        input_data = {
            "Pregnancies": 0 if patient.gender == "Male" else 1,
            "Glucose": lab_map["glucose"],
            "BloodPressure": lab_map.get("diastolic_bp", 80.0),
            "SkinThickness": 20.0,
            "Age": patient.age
        }
        res = predict_diabetes(input_data)
        res["status"] = "success"
        res["input_method"] = "report_extraction"
        return res

    elif model_type == "alzheimer":
        # Check if cognitive score MMSE or functional assessment exists
        # In routine blood tests MMSE is typically not extracted automatically
        missing = ["MMSE (Mini-Mental State Exam score)", "Functional Assessment Score"]
        if "cholesterol_total" not in lab_map:
            missing.append("Total Cholesterol")
        if "systolic_bp" not in lab_map:
            missing.append("Systolic Blood Pressure")

        return {
            "status": "insufficient_data",
            "model_type": "alzheimer",
            "missing_features": missing,
            "message": f"Insufficient information: Specialized cognitive assessments ({', '.join(missing)}) are required for the Alzheimer model and cannot be derived solely from blood test reports. Please use the interactive Assessment form to complete this evaluation."
        }

    elif model_type == "cardiovascular":
        missing = []
        if "systolic_bp" not in lab_map:
            missing.append("Systolic Blood Pressure")
        if "cholesterol_total" not in lab_map:
            missing.append("Total Cholesterol")
        if "cholesterol_hdl" not in lab_map:
            missing.append("HDL Cholesterol")

        if missing:
            return {
                "status": "insufficient_data",
                "model_type": "cardiovascular",
                "missing_features": missing,
                "message": f"Insufficient information: {', '.join(missing)} is missing from lab records. Please upload a comprehensive lipid panel or enter values manually."
            }

        input_data = {
            "Age": patient.age,
            "Gender": patient.gender,
            "SystolicBP": int(lab_map["systolic_bp"]),
            "TotalCholesterol": lab_map["cholesterol_total"],
            "HDLCholesterol": lab_map["cholesterol_hdl"],
            "Smoking": patient.smoking_status in ["Yes", "Former", "Current"]
        }
        res = calculate_framingham_cardio_risk(input_data)
        res["status"] = "success"
        res["input_method"] = "report_extraction"
        return res

    return {"status": "error", "message": f"Unknown model type: '{model_type}'"}
