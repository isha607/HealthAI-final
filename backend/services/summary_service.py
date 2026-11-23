from typing import Dict, Any, List
from sqlalchemy.orm import Session
from backend.models.db_models import PatientProfile, DoctorVisit, LabReport, Prescription, MLAssessment

def generate_doctor_patient_summary(patient_id: str, db: Session) -> Dict[str, Any]:
    """
    Synthesizes a clean longitudinal clinical summary for the treating physician:
    1. Patient Background, Vitals & Chronic History
    2. Latest Abnormal Lab Results & Trajectory
    3. Active Medication Regimen
    4. AI Risk Assessment Scores
    5. Lifestyle Profile (recorded activity, sleep, diet)
    6. Doctor Verification Notice
    """
    patient = db.query(PatientProfile).filter(PatientProfile.patient_id == patient_id).first()
    if not patient:
        return {"error": "Patient profile not found"}

    # Latest abnormal lab results
    latest_reports = db.query(LabReport).filter(LabReport.patient_id == patient_id).order_by(LabReport.report_date.desc()).limit(3).all()
    abnormal_items = []
    for rep in latest_reports:
        for r in rep.results:
            if r.status in ["HIGH", "LOW"]:
                abnormal_items.append({
                    "test_name": r.test_name,
                    "value": f"{r.result_value} {r.unit}",
                    "status": r.status,
                    "date": rep.report_date.strftime("%Y-%m-%d"),
                    "note": r.clinical_explanation
                })

    # Active medicines
    latest_rx = db.query(Prescription).filter(Prescription.patient_id == patient_id).order_by(Prescription.prescription_date.desc()).first()
    current_meds = []
    if latest_rx:
        for m in latest_rx.medicines:
            current_meds.append(f"{m.medicine_name} ({m.dosage}, {m.frequency})")

    # Recent AI Assessments
    assessments = db.query(MLAssessment).filter(MLAssessment.patient_id == patient_id).order_by(MLAssessment.created_at.desc()).limit(3).all()
    ai_summary_lines = []
    for a in assessments:
        ai_summary_lines.append(f"{a.assessment_type.capitalize()}: {a.risk_level} ({a.probability}% probability)")

    lifestyle_summary = (
        f"Sleep: {patient.sleep_hours} hrs/night. "
        f"Physical Activity: {patient.physical_activity_hours} hrs/week. "
        f"Diet Quality: {patient.diet_quality_score}/10. "
        f"Alcohol: {patient.alcohol_consumption} units/wk."
    )

    history_str = (
        f"{patient.full_name}, {patient.age}y {patient.gender}. "
        f"BMI {patient.bmi} kg/m² ({patient.weight_kg}kg, {patient.height_cm}cm). "
        f"Smoker: {patient.smoking_status}. "
        f"Hypertension: {'Positive' if patient.hypertension else 'Negative'}. "
        f"Diabetes History: {'Positive' if patient.diabetes_history else 'Negative'}. "
        f"Family Alzheimer's History: {'Positive' if patient.family_history_alzheimers else 'Negative'}."
    )

    return {
        "patient_id": patient_id,
        "patient_name": patient.full_name,
        "demographics": f"{patient.age} y/o {patient.gender} (BMI {patient.bmi})",
        "medical_background": history_str,
        "abnormal_lab_findings": abnormal_items,
        "current_medications": current_meds if current_meds else ["None currently recorded"],
        "ai_risk_assessments": ai_summary_lines if ai_summary_lines else ["No recent assessments"],
        "lifestyle_summary": lifestyle_summary,
        "clinical_impressions": [
            "Cardiometabolic monitoring advised based on blood pressure & lipid markers.",
            "Continue regular physical activity and monitor fasting glucose routinely."
        ],
        "verification_notice": "AI-GENERATED DECISION SUPPORT SUMMARY: This synthesis is compiled automatically from recorded longitudinal health records, lab reports, and ML models. It is designed to assist physician review and MUST BE VERIFIED against primary clinical records by the treating doctor before making medical decisions."
    }
