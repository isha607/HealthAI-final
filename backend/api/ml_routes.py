import os
import shutil
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form, Body, Header
from sqlalchemy.orm import Session
from typing import Dict, Any, List, Optional
import json
from datetime import datetime

from backend.database import get_db
from backend.models.db_models import MLAssessment, PatientProfile, LabReport, LabResult
from backend.models.schemas import AlzheimerInput, DiabetesInput, CardioSurveillanceInput, ClinicalCardioInput, PredictionResponse
from backend.ml.services import (
    predict_alzheimer, 
    predict_diabetes, 
    predict_cardio_surveillance, 
    calculate_framingham_cardio_risk,
    predict_from_patient_records,
    get_alzheimer_model,
    get_diabetes_model,
    get_cardio_model
)
from backend.ocr.document_extractor import extract_document_text, parse_laboratory_parameters
from backend.ml.document_feature_mapper import map_document_to_model
from backend.api.auth import verify_patient_access
from backend.api.documents import validate_uploaded_file

router = APIRouter(prefix="/api/ml", tags=["ML Predictions"])

def save_or_update_assessment(db: Session, patient_id: str, assessment_type: str, result: dict, input_data: dict, source_type: str = "manual"):
    """
    Saves an assessment record with its source type (manual vs uploaded document/report).
    Prevents exact duplicate rows if an identical assessment is recorded within the same session.
    """
    existing = db.query(MLAssessment).filter(
        MLAssessment.patient_id == patient_id,
        MLAssessment.assessment_type == assessment_type,
        MLAssessment.probability == result["probability_percentage"],
        MLAssessment.risk_level == result["risk_level"]
    ).order_by(MLAssessment.created_at.desc()).first()

    if existing:
        existing.created_at = datetime.utcnow()
        existing.input_payload = json.dumps(input_data)
        existing.contributing_factors = json.dumps(result.get("contributing_factors", {}))
        existing.source_type = source_type
        db.commit()
        db.refresh(existing)
        return existing

    assessment = MLAssessment(
        patient_id=patient_id,
        assessment_type=assessment_type,
        model_version="1.0.0",
        input_payload=json.dumps(input_data),
        prediction_class=result["prediction_class"],
        probability=result["probability_percentage"],
        risk_level=result["risk_level"],
        contributing_factors=json.dumps(result.get("contributing_factors", {})),
        clinical_disclaimer=result.get("clinical_disclaimer", ""),
        source_type=source_type
    )
    db.add(assessment)
    db.commit()
    db.refresh(assessment)
    return assessment

@router.post("/predict/alzheimer/{patient_id}")
def run_alzheimer_prediction(patient_id: str, payload: AlzheimerInput, authorization: Optional[str] = Header(None), db: Session = Depends(get_db)):
    verify_patient_access(patient_id, authorization)
    data_dict = payload.dict()
    result = predict_alzheimer(data_dict)
    save_or_update_assessment(db, patient_id, "alzheimer", result, data_dict, source_type="manual")
    return result

@router.post("/predict/diabetes/{patient_id}")
def run_diabetes_prediction(patient_id: str, payload: DiabetesInput, authorization: Optional[str] = Header(None), db: Session = Depends(get_db)):
    verify_patient_access(patient_id, authorization)
    data_dict = payload.dict()
    result = predict_diabetes(data_dict)
    save_or_update_assessment(db, patient_id, "diabetes", result, data_dict, source_type="manual")
    return result

@router.post("/predict/cardiovascular/{patient_id}")
def run_cardio_surveillance(patient_id: str, payload: CardioSurveillanceInput, authorization: Optional[str] = Header(None), db: Session = Depends(get_db)):
    verify_patient_access(patient_id, authorization)
    data_dict = payload.dict()
    result = predict_cardio_surveillance(data_dict)
    save_or_update_assessment(db, patient_id, "cardiovascular_cdc", result, data_dict, source_type="manual")
    return result

@router.post("/predict/cardiovascular-clinical/{patient_id}")
def run_cardio_clinical(patient_id: str, payload: ClinicalCardioInput, authorization: Optional[str] = Header(None), db: Session = Depends(get_db)):
    verify_patient_access(patient_id, authorization)
    data_dict = payload.dict()
    result = calculate_framingham_cardio_risk(data_dict)
    save_or_update_assessment(db, patient_id, "cardiovascular_framingham", result, data_dict, source_type="manual")
    return result

@router.post("/predict-from-report/{patient_id}/{model_type}")
def run_prediction_from_report(patient_id: str, model_type: str, authorization: Optional[str] = Header(None), db: Session = Depends(get_db)):
    """
    Method B: Evaluate risk directly from extracted lab reports.
    If required features are missing, returns 'insufficient_data' status without fabricating or imputing 0.
    """
    verify_patient_access(patient_id, authorization)
    result = predict_from_patient_records(patient_id, model_type.lower(), db)
    if result.get("status") == "success":
        save_or_update_assessment(db, patient_id, f"{model_type.lower()}_from_report", result, {"method": "report_extraction"}, source_type="uploaded_document")
    return result

# ==============================================================================
# PHASE 6: OCR / DOCUMENT EXTRACTION & FEATURE MAPPING ENDPOINTS
# ==============================================================================

@router.post("/extract-from-file")
async def extract_from_file_for_model(
    patient_id: str = Form(...),
    model_type: str = Form(...),
    file: Optional[UploadFile] = File(None),
    raw_text: Optional[str] = Form(None),
    authorization: Optional[str] = Header(None),
    db: Session = Depends(get_db)
):
    """
    Extracts candidate laboratory biomarkers and metadata from an uploaded report file or pasted text,
    and maps them specifically to the target model's input requirements.
    Never invents missing values; clearly categorizes extracted vs. missing features.
    """
    verify_patient_access(patient_id, authorization)
    validate_uploaded_file(file)
    profile = db.query(PatientProfile).filter(PatientProfile.patient_id == patient_id).first()
    text = ""

    if file and file.filename:
        upload_dir = os.path.join(os.path.dirname(__file__), "..", "..", "uploads", "temp")
        os.makedirs(upload_dir, exist_ok=True)
        temp_path = os.path.join(upload_dir, f"ml_scan_{datetime.now().strftime('%Y%m%d%H%M%S')}_{file.filename}")
        try:
            with open(temp_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            text = extract_document_text(temp_path)
        finally:
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except Exception:
                    pass
    elif raw_text:
        text = raw_text

    if not text or not text.strip():
        return {
            "status": "warning",
            "message": "We could not extract readable text from the provided document. Please check the image quality or enter values manually.",
            "extracted_features": {},
            "missing_features": [],
            "raw_text": ""
        }

    # Extract laboratory parameters from text
    parsed_params = parse_laboratory_parameters(text)

    # Map to model-specific inputs
    try:
        mapping = map_document_to_model(
            model_type=model_type,
            lab_results=parsed_params,
            raw_text=text,
            profile=profile
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    return {
        "status": "success",
        "raw_text": text[:1000],  # preview
        "parsed_parameters_count": len(parsed_params),
        **mapping
    }


@router.post("/extract-from-stored-report")
def extract_from_stored_report_for_model(
    payload: Dict[str, Any] = Body(...),
    authorization: Optional[str] = Header(None),
    db: Session = Depends(get_db)
):
    """
    Extracts features for the target model using a previously uploaded and confirmed lab report.
    Reuses existing structured data without requiring the patient to re-upload.
    """
    patient_id = payload.get("patient_id")
    model_type = payload.get("model_type")
    report_id = payload.get("report_id")

    if not patient_id or not model_type or not report_id:
        raise HTTPException(status_code=400, detail="patient_id, model_type, and report_id are required.")

    verify_patient_access(patient_id, authorization)

    # Patient isolation check
    report = db.query(LabReport).filter(LabReport.id == report_id, LabReport.patient_id == patient_id).first()
    if not report:
        raise HTTPException(status_code=404, detail="Lab report not found or does not belong to this patient.")

    profile = db.query(PatientProfile).filter(PatientProfile.patient_id == patient_id).first()

    # Convert results to dict format
    lab_results = []
    for r in report.results:
        lab_results.append({
            "test_name": r.test_name,
            "result_value": r.result_value,
            "unit": r.unit,
            "reference_low": r.reference_low,
            "reference_high": r.reference_high,
            "status": r.status
        })

    mapping = map_document_to_model(
        model_type=model_type,
        lab_results=lab_results,
        raw_text=report.report_title or "",
        profile=profile
    )

    return {
        "status": "success",
        "source_report_title": report.report_title,
        "source_report_date": report.report_date,
        "parsed_parameters_count": len(lab_results),
        **mapping
    }


@router.post("/predict-confirmed/{patient_id}/{model_type}")
def run_confirmed_prediction(
    patient_id: str,
    model_type: str,
    payload: Dict[str, Any] = Body(...),
    authorization: Optional[str] = Header(None),
    db: Session = Depends(get_db)
):
    """
    Executes prediction using the EXACT EXISTING models after user verification of extracted and missing inputs.
    source_type indicates whether input came from 'manual', 'uploaded_document', or 'stored_report'.
    """
    verify_patient_access(patient_id, authorization)
    mtype = model_type.lower().strip()
    source_type = payload.get("_source_type", "uploaded_document")
    input_data = payload.get("features", payload)
    # Normalize features if passed as {param: {"value": X}} dicts
    clean_data = {}
    for k, v in input_data.items():
        if isinstance(v, dict) and "value" in v:
            clean_data[k] = v["value"]
        else:
            clean_data[k] = v
    input_data = clean_data

    if mtype in ["diabetes", "diab"]:
        # Validate required features
        for req in ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Age"]:
            if req not in input_data:
                raise HTTPException(status_code=422, detail=f"Missing required parameter '{req}' for diabetes assessment.")
        result = predict_diabetes(input_data)
        assessment_type = "diabetes"

    elif mtype in ["cardiovascular", "cardio", "framingham", "cv"]:
        for req in ["Age", "Gender", "SystolicBP", "TotalCholesterol", "HDLCholesterol"]:
            if req not in input_data:
                raise HTTPException(status_code=422, detail=f"Missing required parameter '{req}' for cardiovascular assessment.")
        result = calculate_framingham_cardio_risk(input_data)
        assessment_type = "cardiovascular_framingham"

    elif mtype in ["alzheimer", "alzheimers", "alz"]:
        result = predict_alzheimer(input_data)
        assessment_type = "alzheimer"

    else:
        raise HTTPException(status_code=400, detail=f"Unsupported model type '{model_type}'.")

    # Persist with source_type
    save_or_update_assessment(db, patient_id, assessment_type, result, input_data, source_type=source_type)
    result["source_type"] = source_type
    return result


@router.delete("/assessment/{assessment_id}")
def delete_assessment(assessment_id: int, authorization: Optional[str] = Header(None), db: Session = Depends(get_db)):
    record = db.query(MLAssessment).filter(MLAssessment.id == assessment_id).first()
    if not record:
        raise HTTPException(status_code=404, detail="Assessment not found")
    verify_patient_access(record.patient_id, authorization)
    db.delete(record)
    db.commit()
    return {"status": "success", "message": "Assessment record deleted"}

@router.get("/history/{patient_id}")
def get_assessment_history(patient_id: str, authorization: Optional[str] = Header(None), db: Session = Depends(get_db)):
    verify_patient_access(patient_id, authorization)
    records = db.query(MLAssessment).filter(MLAssessment.patient_id == patient_id).order_by(MLAssessment.created_at.desc()).all()
    seen = set()
    history = []
    for r in records:
        key = (r.assessment_type, r.probability, r.risk_level)
        if key in seen:
            continue
        seen.add(key)
        history.append({
            "id": r.id,
            "assessment_type": r.assessment_type,
            "prediction_class": r.prediction_class,
            "probability": r.probability,
            "risk_level": r.risk_level,
            "source_type": getattr(r, "source_type", "manual") or "manual",
            "created_at": r.created_at.isoformat() if r.created_at else None,
            "clinical_disclaimer": r.clinical_disclaimer,
            "contributing_factors": json.loads(r.contributing_factors) if r.contributing_factors else {}
        })
    return history


@router.get("/model-info")
def get_model_info():
    alz = get_alzheimer_model()
    diab = get_diabetes_model()
    cardio = get_cardio_model()
    return {
        "models": [
            {
                "name": "Alzheimer's Disease Risk Model",
                "type": "alzheimer",
                "algorithm": "StandardScaler + Logistic Regression (C=1.0)",
                "features_count": len(alz.get("features", [])),
                "features": alz.get("features", []),
                "metrics": alz.get("metrics", {}),
                "dataset": "alzheimers_disease_data.csv (2,149 patient records)"
            },
            {
                "name": "Diabetes Prediction Model",
                "type": "diabetes",
                "algorithm": "StandardScaler + Logistic Regression",
                "features_count": len(diab.get("features", [])),
                "features": diab.get("features", []),
                "metrics": diab.get("metrics", {}),
                "dataset": "diabetes.csv (Pima Indians Diagnostic Dataset)"
            },
            {
                "name": "Cardiovascular Surveillance Model",
                "type": "cardiovascular_cdc",
                "algorithm": "LabelEncoder + StandardScaler + PCA(2) + KNeighborsClassifier(k=5)",
                "features_count": len(cardio.get("features", [])),
                "features": cardio.get("features", []),
                "classes": cardio.get("classes", {}),
                "metrics": cardio.get("metrics", {}),
                "dataset": "heart_disease_data.csv (CDC Heart Disease Surveillance)"
            },
            {
                "name": "Framingham Clinical CVD Risk Model",
                "type": "cardiovascular_framingham",
                "algorithm": "AHA / Framingham Heart Study 10-Year Cardiovascular Scoring",
                "features_count": 6,
                "features": ["Age", "Gender", "SystolicBP", "TotalCholesterol", "HDLCholesterol", "Smoking"],
                "dataset": "Framingham Heart Study Validated Clinical Score"
            }
        ]
    }
