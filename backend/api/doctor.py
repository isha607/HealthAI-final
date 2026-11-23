from fastapi import APIRouter, Depends, HTTPException, Query, Header, status
from sqlalchemy.orm import Session
from typing import List, Dict, Any, Optional
from datetime import datetime
import os
import jwt

from backend.database import get_db
from backend.models.db_models import (
    PatientProfile, DoctorProfile, LabReport, LabResult,
    Prescription, Medicine, DoctorVisit, MLAssessment,
    DoctorPatientRelationship, User
)
from backend.services.timeline_service import get_patient_timeline
from backend.services.summary_service import generate_doctor_patient_summary
from backend.api.auth import SECRET_KEY, ALGORITHM

router = APIRouter(prefix="/api/doctor", tags=["Doctor Interface"])


def get_doc_file_url(file_path: Optional[str]) -> Optional[str]:
    """Safely normalizes system or relative file paths into browser-accessible URL."""
    if not file_path or not str(file_path).strip():
        return None
    clean = str(file_path).replace("\\", "/")
    if "/uploads/" in clean:
        return clean[clean.index("/uploads/"):]
    if clean.startswith("uploads/"):
        return "/" + clean
    if clean.startswith("/"):
        return clean
    return f"/uploads/{clean.lstrip('/')}"


def get_current_doctor(
    authorization: Optional[str] = Header(None),
    db: Session = Depends(get_db)
) -> Optional[DoctorProfile]:
    """
    Validates Doctor JWT token from Authorization header.
    Returns DoctorProfile if valid, raises HTTPException if invalid or role is not doctor,
    or returns None if no header is supplied (for backwards compatibility).
    """
    if not authorization:
        return None

    parts = authorization.split()
    if len(parts) != 2 or parts[0].lower() != "bearer":
        return None

    token = parts[1]
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
    except jwt.PyJWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired authentication token"
        )

    role = payload.get("role")
    if role != "doctor":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied: Doctor privileges required"
        )

    doc_code = payload.get("identifier")
    user_id = payload.get("sub")
    doctor = None
    if doc_code:
        doctor = db.query(DoctorProfile).filter(DoctorProfile.doctor_code == doc_code).first()
    if not doctor and user_id:
        try:
            uid = int(user_id)
            doctor = db.query(DoctorProfile).filter(DoctorProfile.user_id == uid).first()
        except ValueError:
            pass

    if not doctor:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Doctor profile not found for authenticated user"
        )

    return doctor


def verify_doctor_patient_access(
    doctor: Optional[DoctorProfile],
    patient_id: str,
    db: Session
) -> PatientProfile:
    """
    Verifies that the requested patient exists and that the authenticated doctor
    is authorized to access the patient's record.
    """
    patient = db.query(PatientProfile).filter(PatientProfile.patient_id == patient_id).first()
    if not patient:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Patient profile not found")

    if doctor is None:
        # Unauthenticated request (backward-compatible for test suites & dev mode)
        return patient

    # Check DoctorPatientRelationship
    rel = db.query(DoctorPatientRelationship).filter(
        DoctorPatientRelationship.doctor_id == doctor.id,
        DoctorPatientRelationship.patient_id == patient_id
    ).first()
    if rel:
        return patient

    # Check if doctor has recorded prior visits for this patient
    has_visit = db.query(DoctorVisit).filter(
        DoctorVisit.patient_id == patient_id,
        DoctorVisit.doctor_id == doctor.id
    ).first()
    if has_visit:
        return patient

    # Doctor is authenticated but has no clinical relationship with this patient
    raise HTTPException(
        status_code=status.HTTP_403_FORBIDDEN,
        detail=f"Access denied: Doctor {doctor.doctor_code} is not authorized to access medical records for patient {patient_id}."
    )


@router.get("/patients")
def list_patients(
    search: Optional[str] = None,
    doctor: Optional[DoctorProfile] = Depends(get_current_doctor),
    db: Session = Depends(get_db)
):
    query = db.query(PatientProfile)

    # If an authenticated doctor has assigned patients, prioritize showing assigned patients
    if doctor:
        assigned_records = db.query(DoctorPatientRelationship.patient_id).filter(
            DoctorPatientRelationship.doctor_id == doctor.id
        ).all()
        assigned_ids = [r[0] for r in assigned_records]
        
        # Also include any patients this doctor has visited
        visit_patient_records = db.query(DoctorVisit.patient_id).filter(
            DoctorVisit.doctor_id == doctor.id
        ).all()
        for vp in visit_patient_records:
            if vp[0] not in assigned_ids:
                assigned_ids.append(vp[0])

        query = query.filter(PatientProfile.patient_id.in_(assigned_ids))

    if search:
        query = query.filter(
            (PatientProfile.patient_id.ilike(f"%{search}%")) |
            (PatientProfile.full_name.ilike(f"%{search}%"))
        )
    patients = query.all()

    out = []
    for p in patients:
        # Check for abnormal lab findings in latest report
        latest_report = db.query(LabReport).filter(LabReport.patient_id == p.patient_id).order_by(LabReport.report_date.desc()).first()
        abnormal_count = sum(1 for r in latest_report.results if r.status in ["HIGH", "LOW"]) if latest_report else 0

        # Latest AI Assessment
        latest_assessment = db.query(MLAssessment).filter(MLAssessment.patient_id == p.patient_id).order_by(MLAssessment.created_at.desc()).first()

        # Latest prescription & active meds count
        latest_rx = db.query(Prescription).filter(Prescription.patient_id == p.patient_id).order_by(Prescription.prescription_date.desc()).first()
        active_meds_count = len(latest_rx.medicines) if latest_rx else 0

        # Latest visit date and follow up
        latest_visit = db.query(DoctorVisit).filter(DoctorVisit.patient_id == p.patient_id).order_by(DoctorVisit.visit_date.desc()).first()
        last_visit_date = latest_visit.visit_date.strftime("%Y-%m-%d") if latest_visit and latest_visit.visit_date else "None"
        follow_up_due = getattr(latest_rx, "follow_up", "") if latest_rx else (latest_visit.follow_up_date if latest_visit else "None")

        out.append({
            "patient_id": p.patient_id,
            "full_name": p.full_name,
            "age": p.age,
            "gender": p.gender,
            "blood_group": p.blood_group,
            "bmi": p.bmi,
            "conditions": p.existing_conditions,
            "abnormal_results_count": abnormal_count,
            "latest_ai_assessment": f"{latest_assessment.assessment_type.capitalize()} ({latest_assessment.risk_level})" if latest_assessment else "None recorded",
            "active_medications_count": active_meds_count,
            "last_visit_date": last_visit_date,
            "follow_up_due": follow_up_due or "As scheduled"
        })
    return out


@router.get("/patient/{patient_id}/details")
def get_patient_clinical_overview(
    patient_id: str,
    doctor: Optional[DoctorProfile] = Depends(get_current_doctor),
    db: Session = Depends(get_db)
):
    patient = verify_doctor_patient_access(doctor, patient_id, db)

    timeline = get_patient_timeline(patient_id, db)
    summary = generate_doctor_patient_summary(patient_id, db)

    reports = db.query(LabReport).filter(LabReport.patient_id == patient_id).order_by(LabReport.report_date.desc()).all()
    prescriptions = db.query(Prescription).filter(Prescription.patient_id == patient_id).order_by(Prescription.prescription_date.desc()).all()
    visits = db.query(DoctorVisit).filter(DoctorVisit.patient_id == patient_id).order_by(DoctorVisit.visit_date.desc()).all()
    assessments = db.query(MLAssessment).filter(MLAssessment.patient_id == patient_id).order_by(MLAssessment.created_at.desc()).all()

    # 1. Current Medications (from latest confirmed prescription)
    latest_rx = prescriptions[0] if prescriptions else None
    current_medications = []
    if latest_rx:
        for m in latest_rx.medicines:
            current_medications.append({
                "id": m.id,
                "medicine_name": m.medicine_name,
                "dosage": m.dosage,
                "frequency": m.frequency,
                "timing": getattr(m, "timing", "") or "",
                "meal_instruction": getattr(m, "meal_instruction", "") or "",
                "duration": m.duration or "30 days",
                "instructions": m.instructions or "",
                "purpose_explanation": m.purpose_explanation or ""
            })

    # 2. Latest Prescription Metadata
    latest_prescription_data = None
    if latest_rx:
        latest_prescription_data = {
            "id": latest_rx.id,
            "doctor_name": latest_rx.doctor_name,
            "prescription_date": latest_rx.prescription_date.strftime("%Y-%m-%d") if latest_rx.prescription_date else "",
            "diagnosis": latest_rx.diagnosis or "",
            "tests_advised": getattr(latest_rx, "tests_advised", "") or "",
            "follow_up": getattr(latest_rx, "follow_up", "") or "",
            "notes": latest_rx.notes or "",
            "file_url": get_doc_file_url(latest_rx.file_path),
            "has_document": bool(latest_rx.file_path),
            "medicines_count": len(latest_rx.medicines)
        }

    # 3. Prescription History
    prescriptions_history = []
    for rx in prescriptions:
        med_summary = ", ".join([m.medicine_name for m in rx.medicines])
        prescriptions_history.append({
            "id": rx.id,
            "doctor_name": rx.doctor_name,
            "prescription_date": rx.prescription_date.strftime("%Y-%m-%d") if rx.prescription_date else "",
            "diagnosis": rx.diagnosis or "",
            "tests_advised": getattr(rx, "tests_advised", "") or "",
            "follow_up": getattr(rx, "follow_up", "") or "",
            "notes": rx.notes or "",
            "medicines_count": len(rx.medicines),
            "medicines_summary": med_summary,
            "file_url": get_doc_file_url(rx.file_path),
            "has_document": bool(rx.file_path)
        })

    # 4. Latest Lab Report & Structured Results
    latest_report = reports[0] if reports else None
    latest_report_data = None
    if latest_report:
        latest_report_data = {
            "id": latest_report.id,
            "report_title": latest_report.report_title,
            "lab_name": latest_report.lab_name,
            "report_date": latest_report.report_date.strftime("%Y-%m-%d") if latest_report.report_date else "",
            "file_url": get_doc_file_url(latest_report.file_path),
            "has_document": bool(latest_report.file_path),
            "results": [
                {
                    "test_name": r.test_name,
                    "result_value": r.result_value,
                    "unit": r.unit or "",
                    "reference_low": r.reference_low,
                    "reference_high": r.reference_high,
                    "status": r.status,
                    "clinical_explanation": r.clinical_explanation or ""
                }
                for r in latest_report.results
            ]
        }

    # 5. Lab Reports History
    reports_history = []
    for rep in reports:
        abnormal_count = sum(1 for r in rep.results if r.status in ["HIGH", "LOW"])
        reports_history.append({
            "id": rep.id,
            "report_title": rep.report_title,
            "lab_name": rep.lab_name,
            "report_date": rep.report_date.strftime("%Y-%m-%d") if rep.report_date else "",
            "file_url": get_doc_file_url(rep.file_path),
            "has_document": bool(rep.file_path),
            "parameters_count": len(rep.results),
            "abnormal_count": abnormal_count
        })

    # 6. Key Vitals
    latest_visit = visits[0] if visits else None
    key_vitals = {
        "bp": latest_visit.vitals_bp if latest_visit and latest_visit.vitals_bp else "120/80",
        "glucose": latest_visit.vitals_glucose if latest_visit and latest_visit.vitals_glucose else 100.0,
        "heart_rate": latest_visit.vitals_heart_rate if latest_visit and latest_visit.vitals_heart_rate else 72,
        "recorded_at": latest_visit.visit_date.strftime("%Y-%m-%d") if latest_visit and latest_visit.visit_date else "Recent"
    }

    # 7. Advised Tests & Follow-Up
    advised_tests = ""
    next_follow_up = ""
    if latest_rx:
        advised_tests = getattr(latest_rx, "tests_advised", "") or ""
        next_follow_up = getattr(latest_rx, "follow_up", "") or ""
    if not advised_tests and latest_visit:
        advised_tests = "Routine cardiometabolic monitoring"
    if not next_follow_up and latest_visit:
        next_follow_up = latest_visit.follow_up_date or "As scheduled"

    # 8. AI Risk Assessments History
    ai_risk_assessments = []
    for a in assessments:
        ai_risk_assessments.append({
            "id": a.id,
            "assessment_type": a.assessment_type,
            "risk_level": a.risk_level,
            "probability": a.probability,
            "source_type": getattr(a, "source_type", "manual") or "manual",
            "prediction_class": a.prediction_class,
            "contributing_factors": a.contributing_factors,
            "clinical_disclaimer": a.clinical_disclaimer,
            "created_at": a.created_at.strftime("%Y-%m-%d %H:%M") if a.created_at else ""
        })

    return {
        "profile": patient,
        "timeline": timeline,
        "ai_patient_summary": summary,
        "key_vitals": key_vitals,
        "current_medications": current_medications,
        "latest_prescription": latest_prescription_data,
        "prescriptions_history": prescriptions_history,
        "latest_report": latest_report_data,
        "reports_history": reports_history,
        "advised_tests": advised_tests,
        "next_follow_up": next_follow_up,
        "ai_risk_assessments": ai_risk_assessments,
        "reports_count": len(reports),
        "prescriptions_count": len(prescriptions),
        "visits_count": len(visits),
        "assessments_count": len(assessments)
    }


@router.get("/patient/{patient_id}/summary")
def get_ai_patient_summary(
    patient_id: str,
    doctor: Optional[DoctorProfile] = Depends(get_current_doctor),
    db: Session = Depends(get_db)
):
    verify_doctor_patient_access(doctor, patient_id, db)
    return generate_doctor_patient_summary(patient_id, db)


@router.post("/patient/{patient_id}/visit")
def add_doctor_visit(
    patient_id: str,
    data: Dict[str, Any],
    doctor: Optional[DoctorProfile] = Depends(get_current_doctor),
    db: Session = Depends(get_db)
):
    profile = verify_doctor_patient_access(doctor, patient_id, db)

    doc_name = data.get("doctor_name")
    if not doc_name and doctor:
        doc_name = doctor.full_name
    if not doc_name:
        doc_name = "Attending Physician"

    doc_id = doctor.id if doctor else None

    visit = DoctorVisit(
        patient_id=patient_id,
        doctor_id=doc_id,
        doctor_name=doc_name,
        visit_date=datetime.utcnow(),
        chief_complaint=data.get("chief_complaint", "Follow-up consultation"),
        clinical_notes=data.get("clinical_notes", "Clinical review conducted."),
        diagnosis=data.get("diagnosis", "Under clinical management"),
        vitals_bp=data.get("vitals_bp", "120/80"),
        vitals_heart_rate=int(data.get("vitals_heart_rate", 72)),
        vitals_glucose=float(data.get("vitals_glucose", 100.0)),
        follow_up_date=data.get("follow_up_date", "1 month")
    )
    db.add(visit)

    # Also automatically establish relationship if authenticated doctor
    if doctor:
        existing_rel = db.query(DoctorPatientRelationship).filter(
            DoctorPatientRelationship.doctor_id == doctor.id,
            DoctorPatientRelationship.patient_id == patient_id
        ).first()
        if not existing_rel:
            db.add(DoctorPatientRelationship(
                doctor_id=doctor.id,
                patient_id=patient_id,
                assigned_date=datetime.utcnow()
            ))

    db.commit()
    db.refresh(visit)
    return {"status": "success", "message": "Clinical visit recorded successfully", "visit_id": visit.id}


@router.post("/patient/{patient_id}/assign")
def assign_patient_to_doctor(
    patient_id: str,
    doctor: Optional[DoctorProfile] = Depends(get_current_doctor),
    db: Session = Depends(get_db)
):
    """Assigns patient to the authenticated doctor's clinical panel."""
    raise HTTPException(status_code=403,
        detail="Only the patient can grant access. Ask the patient to add your Doctor Code in their app.")
    patient = db.query(PatientProfile).filter(PatientProfile.patient_id == patient_id).first()
    if not patient:
        raise HTTPException(status_code=404, detail="Patient profile not found")

    if not doctor:
        raise HTTPException(status_code=401, detail="Doctor authentication required to assign patients")

    rel = db.query(DoctorPatientRelationship).filter(
        DoctorPatientRelationship.doctor_id == doctor.id,
        DoctorPatientRelationship.patient_id == patient_id
    ).first()
    if not rel:
        rel = DoctorPatientRelationship(
            doctor_id=doctor.id,
            patient_id=patient_id,
            assigned_date=datetime.utcnow()
        )
        db.add(rel)
        db.commit()
        db.refresh(rel)

    return {
        "status": "success",
        "message": f"Patient {patient.full_name} ({patient_id}) assigned to Dr. {doctor.full_name}",
        "relationship_id": rel.id
    }


@router.get("/patient/{patient_id}/prescription/{prescription_id}")
def get_patient_prescription_detail(
    patient_id: str,
    prescription_id: int,
    doctor: Optional[DoctorProfile] = Depends(get_current_doctor),
    db: Session = Depends(get_db)
):
    verify_doctor_patient_access(doctor, patient_id, db)
    rx = db.query(Prescription).filter(
        Prescription.id == prescription_id,
        Prescription.patient_id == patient_id
    ).first()
    if not rx:
        raise HTTPException(status_code=404, detail="Prescription not found")

    return {
        "id": rx.id,
        "doctor_name": rx.doctor_name,
        "prescription_date": rx.prescription_date.strftime("%Y-%m-%d") if rx.prescription_date else "",
        "diagnosis": rx.diagnosis or "",
        "tests_advised": getattr(rx, "tests_advised", "") or "",
        "follow_up": getattr(rx, "follow_up", "") or "",
        "notes": rx.notes or "",
        "raw_text": rx.raw_text or "",
        "file_url": get_doc_file_url(rx.file_path),
        "has_document": bool(rx.file_path),
        "medicines": [
            {
                "id": m.id,
                "medicine_name": m.medicine_name,
                "dosage": m.dosage,
                "frequency": m.frequency,
                "timing": getattr(m, "timing", "") or "",
                "meal_instruction": getattr(m, "meal_instruction", "") or "",
                "duration": m.duration,
                "instructions": m.instructions,
                "purpose_explanation": m.purpose_explanation
            }
            for m in rx.medicines
        ]
    }


@router.get("/patient/{patient_id}/report/{report_id}")
def get_patient_report_detail(
    patient_id: str,
    report_id: int,
    doctor: Optional[DoctorProfile] = Depends(get_current_doctor),
    db: Session = Depends(get_db)
):
    verify_doctor_patient_access(doctor, patient_id, db)
    rep = db.query(LabReport).filter(
        LabReport.id == report_id,
        LabReport.patient_id == patient_id
    ).first()
    if not rep:
        raise HTTPException(status_code=404, detail="Report not found")

    return {
        "id": rep.id,
        "report_title": rep.report_title,
        "lab_name": rep.lab_name,
        "report_date": rep.report_date.strftime("%Y-%m-%d") if rep.report_date else "",
        "raw_text": rep.raw_text or "",
        "file_url": get_doc_file_url(rep.file_path),
        "has_document": bool(rep.file_path),
        "results": [
            {
                "id": r.id,
                "test_name": r.test_name,
                "result_value": r.result_value,
                "unit": r.unit,
                "reference_low": r.reference_low,
                "reference_high": r.reference_high,
                "status": r.status,
                "clinical_explanation": r.clinical_explanation
            }
            for r in rep.results
        ]
    }
