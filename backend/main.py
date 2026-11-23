import os
import time
import logging
from pathlib import Path
from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime, timedelta
import json

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("healthai")

import backend.database as _db   # imported as module so patches take effect at call time
from backend.database import Base, SessionLocal

from backend.models.db_models import (
    User, PatientProfile, DoctorProfile, DoctorVisit, 
    LabReport, LabResult, Prescription, Medicine, 
    MedicineSchedule, MLAssessment, DoctorPatientRelationship
)
from backend.api import auth, patient, doctor, documents, ml_routes, reminders, chatbot, timeline
from backend.api.auth import get_password_hash

app = FastAPI(
    title="HealthAI – Integrated Patient Healthcare & AI Risk Assessment Platform",
    description="Integrated clinical decision-support and patient health management platform featuring Alzheimer's, Diabetes, and Cardiovascular predictive models, OCR report extraction, and longitudinal records.",
    version="1.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security Headers Middleware
@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    response.headers["X-Process-Time"] = str(round(process_time * 1000, 2)) + "ms"
    logger.info(f"{request.method} {request.url.path} - {response.status_code} ({process_time*1000:.1f}ms)")
    return response

# Global Exception Handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled error on {request.url.path}: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "An internal server error occurred. Please try again.", "path": str(request.url.path)}
    )

# Include Routers
app.include_router(auth.router)
app.include_router(patient.router)
app.include_router(doctor.router)
app.include_router(documents.router)
app.include_router(ml_routes.router)
app.include_router(reminders.router)
app.include_router(chatbot.router)
app.include_router(timeline.router)

# Mount Static and Upload Folders
STATIC_DIR = Path(__file__).parent / "static"
_RENDER_PERSIST = "/opt/render/project/src/data_persist"
if os.path.isdir(_RENDER_PERSIST):
    UPLOADS_DIR = os.path.join(_RENDER_PERSIST, "uploads")
else:
    UPLOADS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "uploads"))
os.makedirs(STATIC_DIR, exist_ok=True)
os.makedirs(UPLOADS_DIR, exist_ok=True)
os.makedirs(os.path.join(UPLOADS_DIR, "reports"), exist_ok=True)
os.makedirs(os.path.join(UPLOADS_DIR, "prescriptions"), exist_ok=True)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
app.mount("/uploads", StaticFiles(directory=UPLOADS_DIR), name="uploads")

@app.get("/sw.js", include_in_schema=False)
def service_worker():
    return FileResponse(
        STATIC_DIR / "sw.js",
        media_type="application/javascript",
        headers={"Service-Worker-Allowed": "/"}
    )

@app.get("/manifest.json", include_in_schema=False)
def manifest():
    return FileResponse(
        STATIC_DIR / "manifest.json",
        media_type="application/manifest+json"
    )

@app.get("/")
def serve_index():
    return FileResponse(STATIC_DIR / "index.html")

@app.on_event("startup")
def startup_event():
    # Database migration for Phase 4: adding columns to existing SQLite database if they don't exist
    try:
        import sqlite3
        db_path = os.path.join(os.path.dirname(__file__), "..", "health_ai.db")
        if os.path.exists(db_path):
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            # Migrating Prescriptions
            cursor.execute("PRAGMA table_info(prescriptions)")
            presc_cols = [info[1] for info in cursor.fetchall()]
            if presc_cols and "tests_advised" not in presc_cols:
                cursor.execute("ALTER TABLE prescriptions ADD COLUMN tests_advised TEXT DEFAULT ''")
            if presc_cols and "follow_up" not in presc_cols:
                cursor.execute("ALTER TABLE prescriptions ADD COLUMN follow_up VARCHAR(100) DEFAULT ''")
            
            # Migrating Medicines
            cursor.execute("PRAGMA table_info(medicines)")
            med_cols = [info[1] for info in cursor.fetchall()]
            if med_cols and "timing" not in med_cols:
                cursor.execute("ALTER TABLE medicines ADD COLUMN timing VARCHAR(50) DEFAULT ''")
            if med_cols and "meal_instruction" not in med_cols:
                cursor.execute("ALTER TABLE medicines ADD COLUMN meal_instruction VARCHAR(50) DEFAULT ''")
                
            # Phase 5 migration: MedicineSchedule new columns
            cursor.execute("PRAGMA table_info(medicine_schedules)")
            sched_cols = [info[1] for info in cursor.fetchall()]
            if sched_cols and "prescription_id" not in sched_cols:
                cursor.execute("ALTER TABLE medicine_schedules ADD COLUMN prescription_id INTEGER DEFAULT NULL")
            if sched_cols and "frequency" not in sched_cols:
                cursor.execute("ALTER TABLE medicine_schedules ADD COLUMN frequency VARCHAR(50) DEFAULT ''")
            if sched_cols and "reminder_type" not in sched_cols:
                cursor.execute("ALTER TABLE medicine_schedules ADD COLUMN reminder_type VARCHAR(30) DEFAULT 'medicine'")

            # Phase 6 migration: MLAssessment source_type column
            cursor.execute("PRAGMA table_info(ml_assessments)")
            ml_cols = [info[1] for info in cursor.fetchall()]
            if ml_cols and "source_type" not in ml_cols:
                cursor.execute("ALTER TABLE ml_assessments ADD COLUMN source_type VARCHAR(30) DEFAULT 'manual'")
                
            conn.commit()
            conn.close()
            logger.info("Database migration completed successfully.")
    except Exception as e:
        logger.error(f"Migration error: {e}")

    # Create tables using the currently-bound engine (supports test patching)
    Base.metadata.create_all(bind=_db.engine)

    # Seed initial test patient and doctor if empty
    db = _db.SessionLocal()
    try:
        patient_user = db.query(User).filter(User.email == "patient@health.org").first()
        if not patient_user:
            print("Seeding initial demonstration accounts and longitudinal clinical records...")
            # 1. Create Patient User
            p_user = User(
                email="patient@health.org",
                hashed_password=get_password_hash("patient123"),
                role="patient"
            )
            db.add(p_user)
            db.commit()
            db.refresh(p_user)

            # 2. Patient Profile
            patient_profile = PatientProfile(
                patient_id="PAT-1001",
                user_id=p_user.id,
                full_name="John Doe",
                age=68,
                gender="Male",
                height_cm=175.0,
                weight_kg=78.5,
                bmi=25.6,
                phone="+1 (555) 234-5678",
                blood_group="A+",
                smoking_status="Former",
                alcohol_consumption=3.0,
                physical_activity_hours=2.0,
                diet_quality_score=6.0,
                sleep_hours=6.8,
                family_history_alzheimers=True,
                cardiovascular_disease=True,
                diabetes_history=False,
                hypertension=True,
                allergies="Penicillin",
                existing_conditions="Pre-hypertension, Borderline High Fasting Glucose"
            )
            db.add(patient_profile)
            db.commit()

            # 3. Create Doctor User
            d_user = User(
                email="doctor@hospital.org",
                hashed_password=get_password_hash("doctor123"),
                role="doctor"
            )
            db.add(d_user)
            db.commit()
            db.refresh(d_user)

            doctor_profile = DoctorProfile(
                doctor_code="DOC-501",
                user_id=d_user.id,
                full_name="Dr. Sarah Jenkins, MD",
                specialization="Cardiometabolic & Geriatric Medicine",
                hospital_clinic="St. Jude Medical Research Hospital",
                phone="+1 (555) 890-1234"
            )
            db.add(doctor_profile)
            db.commit()

            # 4. Longitudinal Doctor Visits
            v1 = DoctorVisit(
                patient_id="PAT-1001",
                doctor_name="Dr. Sarah Jenkins, MD",
                visit_date=datetime(2024, 6, 15),
                chief_complaint="Occasional mild fatigue and slight memory forgetfulness with complex schedules.",
                clinical_notes="Patient alert and oriented x3. MMSE scored at 27/30. Blood pressure mildly elevated at 132/84 mmHg. Recommended comprehensive lipid panel and lifestyle modifications.",
                diagnosis="Borderline Dyslipidemia & Mild Cognitive Complaint",
                vitals_bp="132/84",
                vitals_glucose=104.0,
                follow_up_date="6 months"
            )
            v2 = DoctorVisit(
                patient_id="PAT-1001",
                doctor_name="Dr. Sarah Jenkins, MD",
                visit_date=datetime(2025, 1, 20),
                chief_complaint="Follow-up review for blood pressure and fasting glucose monitoring.",
                clinical_notes="Blood pressure persistent at 136/88 mmHg. HbA1c elevated to 6.2%. Initiated low-dose Metformin and Lisinopril. Patient advised to maintain smartwatch step tracking.",
                diagnosis="Stage 1 Essential Hypertension & Impaired Glucose Tolerance",
                vitals_bp="136/88",
                vitals_glucose=118.0,
                follow_up_date="3 months"
            )
            db.add(v1)
            db.add(v2)

            # 5. Longitudinal Lab Reports
            rep1 = LabReport(
                patient_id="PAT-1001",
                report_title="Comprehensive Health Panel (Mid 2024)",
                report_type="Blood Report",
                report_date=datetime(2024, 6, 12),
                lab_name="Apex Diagnostic Center",
                raw_text="Fasting Glucose: 104 mg/dL\nTotal Cholesterol: 212 mg/dL\nHDL: 45 mg/dL\nLDL: 135 mg/dL\nTriglycerides: 160 mg/dL\nSystolic BP: 132 mmHg\nDiastolic BP: 84 mmHg"
            )
            db.add(rep1)
            db.commit()
            db.refresh(rep1)

            res1_items = [
                LabResult(report_id=rep1.id, patient_id="PAT-1001", test_name="Fasting Blood Glucose", result_value=104.0, unit="mg/dL", reference_low=70.0, reference_high=99.0, status="HIGH", clinical_explanation="This value is above standard normal range (70–99 mg/dL). Elevated fasting blood sugar can be associated with increased diabetes risk."),
                LabResult(report_id=rep1.id, patient_id="PAT-1001", test_name="Total Cholesterol", result_value=212.0, unit="mg/dL", reference_low=125.0, reference_high=200.0, status="HIGH", clinical_explanation="Total cholesterol is above the desirable clinical threshold (<200 mg/dL)."),
                LabResult(report_id=rep1.id, patient_id="PAT-1001", test_name="HDL Cholesterol ('Good')", result_value=45.0, unit="mg/dL", reference_low=40.0, reference_high=60.0, status="NORMAL", clinical_explanation="Within acceptable protective range."),
                LabResult(report_id=rep1.id, patient_id="PAT-1001", test_name="LDL Cholesterol ('Bad')", result_value=135.0, unit="mg/dL", reference_low=50.0, reference_high=100.0, status="HIGH", clinical_explanation="Above optimal range (<100 mg/dL); discuss lifestyle modifications."),
                LabResult(report_id=rep1.id, patient_id="PAT-1001", test_name="Systolic Blood Pressure", result_value=132.0, unit="mmHg", reference_low=90.0, reference_high=120.0, status="HIGH", clinical_explanation="Systolic BP is slightly elevated (pre-hypertension).")
            ]
            for r in res1_items: db.add(r)

            rep2 = LabReport(
                patient_id="PAT-1001",
                report_title="Metabolic & Cardiometabolic Panel (Early 2025)",
                report_type="Blood Report",
                report_date=datetime(2025, 1, 18),
                lab_name="Apex Diagnostic Center",
                raw_text="Fasting Glucose: 118 mg/dL\nHbA1c: 6.2 %\nTotal Cholesterol: 224 mg/dL\nHDL: 44 mg/dL\nLDL: 142 mg/dL\nSystolic BP: 136 mmHg\nDiastolic BP: 88 mmHg"
            )
            db.add(rep2)
            db.commit()
            db.refresh(rep2)

            res2_items = [
                LabResult(report_id=rep2.id, patient_id="PAT-1001", test_name="Fasting Blood Glucose", result_value=118.0, unit="mg/dL", reference_low=70.0, reference_high=99.0, status="HIGH", clinical_explanation="Elevated above reference range. Demonstrates an upward trend from previous report (104 -> 118 mg/dL)."),
                LabResult(report_id=rep2.id, patient_id="PAT-1001", test_name="Hemoglobin A1c (HbA1c)", result_value=6.2, unit="%", reference_low=4.0, reference_high=5.6, status="HIGH", clinical_explanation="Value between 5.7% and 6.4% corresponds to pre-diabetes classification according to American Diabetes Association guidelines."),
                LabResult(report_id=rep2.id, patient_id="PAT-1001", test_name="Total Cholesterol", result_value=224.0, unit="mg/dL", reference_low=125.0, reference_high=200.0, status="HIGH", clinical_explanation="Increased from previous 212 mg/dL. Statin therapy considered."),
                LabResult(report_id=rep2.id, patient_id="PAT-1001", test_name="Systolic Blood Pressure", result_value=136.0, unit="mmHg", reference_low=90.0, reference_high=120.0, status="HIGH", clinical_explanation="Blood pressure remains elevated; medication prescribed.")
            ]
            for r in res2_items: db.add(r)

            # 6. Prescriptions
            rx1 = Prescription(
                patient_id="PAT-1001",
                doctor_name="Dr. Sarah Jenkins, MD",
                prescription_date=datetime(2025, 1, 20),
                diagnosis="Hypertension and Pre-diabetes management",
                notes="Patient advised to take with meals and maintain daily medication log.",
                raw_text="Metformin 500 mg twice daily. Lisinopril 10 mg once daily morning. Atorvastatin 20 mg once daily bedtime."
            )
            db.add(rx1)
            db.commit()
            db.refresh(rx1)

            meds = [
                Medicine(prescription_id=rx1.id, patient_id="PAT-1001", medicine_name="Metformin", dosage="500 mg", frequency="Twice daily after meals", duration="90 days", instructions="Take with meals to minimize gastrointestinal discomfort.", purpose_explanation="Metformin is an antidiabetic medication commonly prescribed to improve insulin sensitivity and lower blood glucose levels."),
                Medicine(prescription_id=rx1.id, patient_id="PAT-1001", medicine_name="Lisinopril", dosage="10 mg", frequency="Once daily in the morning", duration="90 days", instructions="Take in morning with a full glass of water.", purpose_explanation="Lisinopril is an ACE inhibitor commonly prescribed to manage high blood pressure and protect cardiac health."),
                Medicine(prescription_id=rx1.id, patient_id="PAT-1001", medicine_name="Atorvastatin", dosage="20 mg", frequency="Once daily at bedtime", duration="90 days", instructions="Take at bedtime with water.", purpose_explanation="Atorvastatin is a statin medication widely prescribed to lower LDL ('bad') cholesterol and reduce cardiovascular risk.")
            ]
            for m in meds: db.add(m)

            # 7. Medicine Schedules
            scheds = [
                MedicineSchedule(patient_id="PAT-1001", medicine_name="Lisinopril 10 mg", dosage="1 tablet", scheduled_time="08:00 AM", meal_relation="After breakfast", taken_status=True, notes="Blood pressure support"),
                MedicineSchedule(patient_id="PAT-1001", medicine_name="Metformin 500 mg", dosage="1 tablet", scheduled_time="01:00 PM", meal_relation="After lunch", taken_status=True, notes="Blood sugar control"),
                MedicineSchedule(patient_id="PAT-1001", medicine_name="Metformin 500 mg", dosage="1 tablet", scheduled_time="08:00 PM", meal_relation="After dinner", taken_status=False, notes="Second daily dose"),
                MedicineSchedule(patient_id="PAT-1001", medicine_name="Atorvastatin 20 mg", dosage="1 tablet", scheduled_time="10:00 PM", meal_relation="At bedtime", taken_status=False, notes="Lipid management")
            ]
            for s in scheds: db.add(s)

            # 8. Initial ML Assessments
            a1 = MLAssessment(
                patient_id="PAT-1001",
                assessment_type="alzheimer",
                model_version="1.0.0",
                input_payload="{}",
                prediction_class=0,
                probability=38.4,
                risk_level="Low to Moderate Risk",
                contributing_factors=json.dumps({"MMSE": 27.0, "ADL": 8.5, "Memory Complaints": "Present"}),
                clinical_disclaimer="AI-assisted risk score based on cognitive and clinical markers. Not a diagnosis."
            )
            a2 = MLAssessment(
                patient_id="PAT-1001",
                assessment_type="diabetes",
                model_version="1.0.0",
                input_payload="{}",
                prediction_class=1,
                probability=68.5,
                risk_level="Elevated Risk",
                contributing_factors=json.dumps({"Fasting Glucose": "118 mg/dL", "Diastolic BP": "88 mmHg", "Age": "68 years"}),
                clinical_disclaimer="Machine learning statistical estimate based on metabolic parameters. Not a medical diagnosis."
            )
            db.add(a1)
            db.add(a2)
            db.commit()
            print("Seed data successfully added for PAT-1001 and DOC-501!")
        
        # Phase 8: Ensure DoctorPatientRelationship exists between DOC-501 and PAT-1001
        doc_prof = db.query(DoctorProfile).filter(DoctorProfile.doctor_code == "DOC-501").first()
        if doc_prof:
            rel = db.query(DoctorPatientRelationship).filter(
                DoctorPatientRelationship.doctor_id == doc_prof.id,
                DoctorPatientRelationship.patient_id == "PAT-1001"
            ).first()
            if not rel:
                db.add(DoctorPatientRelationship(
                    doctor_id=doc_prof.id,
                    patient_id="PAT-1001",
                    assigned_date=datetime.utcnow()
                ))
                db.commit()
    finally:
        db.close()
