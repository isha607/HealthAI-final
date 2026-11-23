from fastapi import APIRouter, Depends, HTTPException, Header
from sqlalchemy.orm import Session
from typing import Dict, Any, List, Optional
from backend.database import get_db
from backend.models.db_models import PatientProfile, LabReport, LabResult, DoctorVisit
from backend.models.schemas import PatientProfileSchema
from backend.api.auth import verify_patient_access

router = APIRouter(prefix="/api/patient", tags=["Patient Profile"])

@router.get("/profile/{patient_id}")
def get_profile(patient_id: str, authorization: Optional[str] = Header(None), db: Session = Depends(get_db)):
    verify_patient_access(patient_id, authorization)
    profile = db.query(PatientProfile).filter(PatientProfile.patient_id == patient_id).first()
    if not profile:
        raise HTTPException(status_code=404, detail="Patient not found")
    return profile

@router.put("/profile/{patient_id}")
def update_profile(patient_id: str, data: Dict[str, Any], authorization: Optional[str] = Header(None), db: Session = Depends(get_db)):
    verify_patient_access(patient_id, authorization)
    profile = db.query(PatientProfile).filter(PatientProfile.patient_id == patient_id).first()
    if not profile:
        raise HTTPException(status_code=404, detail="Patient not found")

    for k, v in data.items():
        if hasattr(profile, k) and k != "patient_id":
            setattr(profile, k, v)

    # Recalculate BMI if height/weight changed
    if profile.height_cm and profile.height_cm > 0 and profile.weight_kg:
        h_m = profile.height_cm / 100.0
        profile.bmi = round(profile.weight_kg / (h_m * h_m), 1)

    db.commit()
    db.refresh(profile)
    return profile

@router.get("/visits/{patient_id}")
def get_patient_visits(patient_id: str, authorization: Optional[str] = Header(None), db: Session = Depends(get_db)):
    """
    Returns clinical doctor visits and follow-up consultation records for the patient.
    """
    verify_patient_access(patient_id, authorization)
    visits = db.query(DoctorVisit).filter(DoctorVisit.patient_id == patient_id).order_by(DoctorVisit.visit_date.desc()).all()
    return [
        {
            "id": v.id,
            "doctor_name": v.doctor_name,
            "visit_date": v.visit_date.strftime("%Y-%m-%d") if v.visit_date else "",
            "chief_complaint": v.chief_complaint,
            "diagnosis": v.diagnosis,
            "clinical_notes": v.clinical_notes,
            "vitals_bp": v.vitals_bp,
            "vitals_heart_rate": v.vitals_heart_rate,
            "vitals_glucose": v.vitals_glucose,
            "follow_up_date": v.follow_up_date
        }
        for v in visits
    ]

@router.get("/trends/{patient_id}")
def get_health_trends(patient_id: str, authorization: Optional[str] = Header(None), db: Session = Depends(get_db)):
    """
    Returns longitudinal data series for Glucose, Blood Pressure, and Cholesterol over time.
    """
    verify_patient_access(patient_id, authorization)
    reports = db.query(LabReport).filter(LabReport.patient_id == patient_id).order_by(LabReport.report_date.asc()).all()
    
    dates = []
    glucose_vals = []
    bp_sys_vals = []
    bp_dia_vals = []
    chol_vals = []

    for rep in reports:
        dt_str = rep.report_date.strftime("%b %Y")
        dates.append(dt_str)
        
        # Look for matching tests
        g_val, s_val, d_val, c_val = None, None, None, None
        for res in rep.results:
            name_low = res.test_name.lower()
            if "glucose" in name_low:
                g_val = res.result_value
            elif "systolic" in name_low:
                s_val = res.result_value
            elif "diastolic" in name_low:
                d_val = res.result_value
            elif "total cholesterol" in name_low or name_low == "cholesterol":
                c_val = res.result_value

        glucose_vals.append(g_val if g_val is not None else 100.0)
        bp_sys_vals.append(s_val if s_val is not None else 120.0)
        bp_dia_vals.append(d_val if d_val is not None else 80.0)
        chol_vals.append(c_val if c_val is not None else 200.0)

    summary_comparison = None
    if len(dates) >= 1:
        summary_comparison = {
            "glucose": f"Previous: {glucose_vals[-2] if len(glucose_vals)>1 else glucose_vals[0]} -> Current: {glucose_vals[-1]} mg/dL",
            "blood_pressure": f"Previous: {int(bp_sys_vals[-2])}/{int(bp_dia_vals[-2])} -> Current: {int(bp_sys_vals[-1])}/{int(bp_dia_vals[-1])} mmHg",
            "cholesterol": f"Previous: {chol_vals[-2] if len(chol_vals)>1 else chol_vals[0]} -> Current: {chol_vals[-1]} mg/dL"
        }

    return {
        "dates": dates,
        "glucose": glucose_vals,
        "systolic_bp": bp_sys_vals,
        "diastolic_bp": bp_dia_vals,
        "total_cholesterol": chol_vals,
        "summary_comparison": summary_comparison
    }


@router.post("/vitals/{patient_id}")
def add_patient_vitals(patient_id: str, data: Dict[str, Any], authorization: Optional[str] = Header(None), db: Session = Depends(get_db)):
    """
    Save new user-entered vitals (BP, Glucose, Cholesterol, Weight) directly into the database.
    Updates the patient profile and logs a new lab report record for longitudinal tracking.
    """
    verify_patient_access(patient_id, authorization)
    from datetime import datetime
    profile = db.query(PatientProfile).filter(PatientProfile.patient_id == patient_id).first()
    if not profile:
        raise HTTPException(status_code=404, detail="Patient not found")

    # Update profile fields if provided
    if "weight_kg" in data and data["weight_kg"]:
        profile.weight_kg = float(data["weight_kg"])
        if profile.height_cm and profile.height_cm > 0:
            h_m = profile.height_cm / 100.0
            profile.bmi = round(profile.weight_kg / (h_m * h_m), 1)

    if "phone" in data and data["phone"]:
        profile.phone = str(data["phone"])
    if "allergies" in data and data["allergies"]:
        profile.allergies = str(data["allergies"])
    if "existing_conditions" in data and data["existing_conditions"]:
        profile.existing_conditions = str(data["existing_conditions"])
    if "full_name" in data and data["full_name"]:
        profile.full_name = str(data["full_name"])
    if "age" in data and data["age"]:
        profile.age = int(data["age"])

    # Create new LabReport
    title = data.get("report_title", "User Recorded Vitals")
    lab_name = data.get("lab_name", "Patient Self-Check / Home Monitoring")
    now = datetime.utcnow()
    raw_summary = []

    report = LabReport(
        patient_id=patient_id,
        report_title=title,
        report_type="Vitals & Blood Check",
        report_date=now,
        lab_name=lab_name,
        raw_text=""
    )
    db.add(report)
    db.commit()
    db.refresh(report)

    # Insert test results
    # 1. Glucose
    if "glucose" in data and data["glucose"] is not None:
        g = float(data["glucose"])
        raw_summary.append(f"Fasting Glucose: {g} mg/dL")
        g_status = "HIGH" if g > 99.0 else ("LOW" if g < 70.0 else "NORMAL")
        g_exp = "Above normal fasting threshold (70–99 mg/dL)." if g_status == "HIGH" else "Within normal fasting range."
        db.add(LabResult(
            report_id=report.id, patient_id=patient_id, test_name="Fasting Blood Glucose",
            result_value=g, unit="mg/dL", reference_low=70.0, reference_high=99.0,
            status=g_status, clinical_explanation=g_exp
        ))

    # 2. Systolic BP
    if "systolic_bp" in data and data["systolic_bp"] is not None:
        s = float(data["systolic_bp"])
        raw_summary.append(f"Systolic BP: {s} mmHg")
        s_status = "HIGH" if s > 120.0 else ("LOW" if s < 90.0 else "NORMAL")
        s_exp = "Elevated systolic blood pressure (>120 mmHg)." if s_status == "HIGH" else "Optimal systolic blood pressure."
        db.add(LabResult(
            report_id=report.id, patient_id=patient_id, test_name="Systolic Blood Pressure",
            result_value=s, unit="mmHg", reference_low=90.0, reference_high=120.0,
            status=s_status, clinical_explanation=s_exp
        ))

    # 3. Diastolic BP
    if "diastolic_bp" in data and data["diastolic_bp"] is not None:
        d = float(data["diastolic_bp"])
        raw_summary.append(f"Diastolic BP: {d} mmHg")
        d_status = "HIGH" if d > 80.0 else ("LOW" if d < 60.0 else "NORMAL")
        d_exp = "Elevated diastolic blood pressure (>80 mmHg)." if d_status == "HIGH" else "Optimal diastolic blood pressure."
        db.add(LabResult(
            report_id=report.id, patient_id=patient_id, test_name="Diastolic Blood Pressure",
            result_value=d, unit="mmHg", reference_low=60.0, reference_high=80.0,
            status=d_status, clinical_explanation=d_exp
        ))

    # 4. Total Cholesterol
    if "cholesterol" in data and data["cholesterol"] is not None:
        c = float(data["cholesterol"])
        raw_summary.append(f"Total Cholesterol: {c} mg/dL")
        c_status = "HIGH" if c > 200.0 else ("LOW" if c < 125.0 else "NORMAL")
        c_exp = "Above desirable clinical threshold (<200 mg/dL)." if c_status == "HIGH" else "Desirable total cholesterol range."
        db.add(LabResult(
            report_id=report.id, patient_id=patient_id, test_name="Total Cholesterol",
            result_value=c, unit="mg/dL", reference_low=125.0, reference_high=200.0,
            status=c_status, clinical_explanation=c_exp
        ))

    # 5. HbA1c
    if "hba1c" in data and data["hba1c"] is not None:
        a = float(data["hba1c"])
        raw_summary.append(f"HbA1c: {a} %")
        a_status = "HIGH" if a >= 5.7 else ("LOW" if a < 4.0 else "NORMAL")
        a_exp = "Elevated HbA1c (pre-diabetes or diabetes range)." if a_status == "HIGH" else "Normal glycemic control."
        db.add(LabResult(
            report_id=report.id, patient_id=patient_id, test_name="Hemoglobin A1c (HbA1c)",
            result_value=a, unit="%", reference_low=4.0, reference_high=5.6,
            status=a_status, clinical_explanation=a_exp
        ))

    report.raw_text = "\n".join(raw_summary)
    db.commit()

    return {
        "status": "success",
        "message": "Vitals and patient data saved to database successfully!",
        "patient_id": patient_id,
        "report_id": report.id,
        "bmi": profile.bmi,
        "vitals_recorded": raw_summary
    }

