from typing import List, Dict, Any
from sqlalchemy.orm import Session
from backend.models.db_models import DoctorVisit, LabReport, Prescription, MLAssessment

def get_patient_timeline(patient_id: str, db: Session) -> List[Dict[str, Any]]:
    """
    Builds a unified chronological longitudinal health timeline for the patient.
    Consolidates visits, lab reports, prescriptions, ML assessments, and wearable milestones.
    """
    events = []

    # 1. Doctor Visits
    visits = db.query(DoctorVisit).filter(DoctorVisit.patient_id == patient_id).all()
    for v in visits:
        events.append({
            "type": "DOCTOR_VISIT",
            "title": f"Doctor Visit: {v.doctor_name}",
            "date": v.visit_date.strftime("%Y-%m-%d"),
            "display_date": v.visit_date.strftime("%b %d, %Y"),
            "badge": "Visit",
            "badge_color": "blue",
            "summary": f"Diagnosis: {v.diagnosis} | Chief Complaint: {v.chief_complaint}",
            "details": {
                "diagnosis": v.diagnosis,
                "chief_complaint": v.chief_complaint,
                "clinical_notes": v.clinical_notes,
                "vitals_bp": v.vitals_bp,
                "vitals_glucose": v.vitals_glucose,
                "follow_up": v.follow_up_date
            }
        })

    # 2. Lab Reports
    reports = db.query(LabReport).filter(LabReport.patient_id == patient_id).all()
    for rep in reports:
        abnormal_count = sum(1 for r in rep.results if r.status in ["HIGH", "LOW"])
        res_summary = ", ".join([f"{r.test_name}: {r.result_value} {r.unit} ({r.status})" for r in rep.results[:3]])
        events.append({
            "type": "LAB_REPORT",
            "title": f"Lab Report: {rep.report_title}",
            "date": rep.report_date.strftime("%Y-%m-%d"),
            "display_date": rep.report_date.strftime("%b %d, %Y"),
            "badge": f"{abnormal_count} Abnormal" if abnormal_count > 0 else "All Normal",
            "badge_color": "red" if abnormal_count > 0 else "green",
            "summary": f"Lab: {rep.lab_name} | {res_summary}",
            "details": {
                "lab_name": rep.lab_name,
                "results": [
                    {
                        "test": r.test_name,
                        "value": r.result_value,
                        "unit": r.unit,
                        "status": r.status,
                        "explanation": r.clinical_explanation
                    } for r in rep.results
                ]
            }
        })

    # 3. Prescriptions
    rxs = db.query(Prescription).filter(Prescription.patient_id == patient_id).all()
    for rx in rxs:
        med_names = ", ".join([m.medicine_name for m in rx.medicines])
        events.append({
            "type": "PRESCRIPTION",
            "title": f"Prescription: {rx.doctor_name}",
            "date": rx.prescription_date.strftime("%Y-%m-%d"),
            "display_date": rx.prescription_date.strftime("%b %d, %Y"),
            "badge": f"{len(rx.medicines)} Meds",
            "badge_color": "purple",
            "summary": f"Prescribed: {med_names or 'Prescription medications'}",
            "details": {
                "diagnosis": rx.diagnosis,
                "notes": rx.notes,
                "medicines": [
                    {
                        "name": m.medicine_name,
                        "dosage": m.dosage,
                        "frequency": m.frequency,
                        "purpose": m.purpose_explanation
                    } for m in rx.medicines
                ]
            }
        })

    # 4. ML Risk Assessments
    assessments = db.query(MLAssessment).filter(MLAssessment.patient_id == patient_id).all()
    for a in assessments:
        events.append({
            "type": "AI_ASSESSMENT",
            "title": f"AI Assessment: {a.assessment_type.capitalize()}",
            "date": a.created_at.strftime("%Y-%m-%d"),
            "display_date": a.created_at.strftime("%b %d, %Y"),
            "badge": f"{a.risk_level}",
            "badge_color": "orange" if "High" in a.risk_level or "Elevated" in a.risk_level else "green",
            "summary": f"Score: {a.probability}% risk probability | {a.risk_level}",
            "details": {
                "risk_level": a.risk_level,
                "probability": a.probability,
                "disclaimer": a.clinical_disclaimer
            }
        })

    # Sort descending by date
    events.sort(key=lambda x: x["date"], reverse=True)
    return events
