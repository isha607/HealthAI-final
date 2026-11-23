"""
API endpoint integration tests.

Uses a session-scoped SQLite test database with a proper dependency override
so tables exist before any test queries run.
"""
import json
import pytest
from datetime import datetime
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# --- 1. Patch the DB engine BEFORE importing the app ---
from backend.database import Base, get_db
import backend.database as _db_module

TEST_DB_URL = "sqlite:///./test_health_ai.db"
_test_engine = create_engine(TEST_DB_URL, connect_args={"check_same_thread": False})
_TestSession = sessionmaker(autocommit=False, autoflush=False, bind=_test_engine)
# Patch BOTH engine and SessionLocal so startup_event in main.py uses the test DB
_db_module.engine = _test_engine
_db_module.SessionLocal = _TestSession


def _override_get_db():
    db = _TestSession()
    try:
        yield db
    finally:
        db.close()


# --- 2. Import app after patching ---
from backend.main import app  # noqa: E402
from backend.api.auth import get_password_hash  # noqa: E402
from backend.models.db_models import (  # noqa: E402
    User, PatientProfile, DoctorProfile, DoctorVisit,
    LabReport, LabResult, MLAssessment,
)

app.dependency_overrides[get_db] = _override_get_db


# --- 3. Session-scoped fixture: create tables + seed once ---
@pytest.fixture(scope="session", autouse=True)
def _seed_db():
    Base.metadata.create_all(bind=_test_engine)
    db = _TestSession()
    try:
        p = User(email="patient@health.org",
                 hashed_password=get_password_hash("patient123"), role="patient")
        db.add(p); db.commit(); db.refresh(p)

        db.add(PatientProfile(
            patient_id="PAT-1001", user_id=p.id, full_name="John Doe", age=68,
            gender="Male", height_cm=175.0, weight_kg=78.5, bmi=25.6,
            phone="+1555", blood_group="A+", smoking_status="Former",
            alcohol_consumption=3.0, physical_activity_hours=2.0,
            diet_quality_score=6.0, sleep_hours=6.8,
            family_history_alzheimers=True, cardiovascular_disease=True,
            diabetes_history=False, hypertension=True,
            allergies="Penicillin", existing_conditions="Hypertension",
        ))
        db.commit()

        d = User(email="doctor@hospital.org",
                 hashed_password=get_password_hash("doctor123"), role="doctor")
        db.add(d); db.commit(); db.refresh(d)

        db.add(DoctorProfile(
            doctor_code="DOC-501", user_id=d.id,
            full_name="Dr. Sarah Jenkins, MD",
            specialization="Cardiology", hospital_clinic="St. Jude",
            phone="+1555890",
        ))
        db.commit()

        db.add(DoctorVisit(
            patient_id="PAT-1001", doctor_name="Dr. Sarah Jenkins, MD",
            visit_date=datetime(2024, 6, 15), chief_complaint="Fatigue",
            clinical_notes="MMSE 27.", diagnosis="Dyslipidemia",
            vitals_bp="132/84", vitals_glucose=104.0, follow_up_date="6 months",
        ))

        rep = LabReport(
            patient_id="PAT-1001", report_title="Panel 2024",
            report_type="Blood Report", report_date=datetime(2024, 6, 12),
            lab_name="Apex", raw_text="Fasting Glucose: 104 mg/dL",
        )
        db.add(rep); db.commit(); db.refresh(rep)

        db.add(LabResult(
            report_id=rep.id, patient_id="PAT-1001",
            test_name="Fasting Blood Glucose", result_value=104.0,
            unit="mg/dL", reference_low=70.0, reference_high=99.0,
            status="HIGH", clinical_explanation="Above range.",
        ))

        db.add(MLAssessment(
            patient_id="PAT-1001", assessment_type="alzheimer",
            model_version="1.0.0", input_payload="{}",
            prediction_class=0, probability=38.4,
            risk_level="Low Risk",
            contributing_factors=json.dumps({"MMSE": 27.0}),
            clinical_disclaimer="AI estimate.",
        ))
        db.commit()
    finally:
        db.close()

    yield

    Base.metadata.drop_all(bind=_test_engine)


# --- 4. Test client fixture ---
@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


# --- 5. Tests ---
def test_homepage_serves(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "HealthAI" in r.text


def test_get_patient_profile(client):
    r = client.get("/api/patient/profile/PAT-1001")
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["patient_id"] == "PAT-1001"
    assert d["full_name"] == "John Doe"


def test_patient_timeline(client):
    r = client.get("/api/timeline/PAT-1001")
    assert r.status_code == 200, r.text
    events = r.json()
    assert isinstance(events, list)
    assert len(events) > 0
    print(f"Retrieved {len(events)} timeline events for PAT-1001")


def test_doctor_patient_summary(client):
    r = client.get("/api/doctor/patient/PAT-1001/summary")
    assert r.status_code == 200, r.text
    s = r.json()
    assert "patient_name" in s
    assert "verification_notice" in s


def test_chatbot_query(client):
    r = client.post(
        "/api/chatbot/PAT-1001/query",
        json={"message": "What is high in my blood report?"},
    )
    assert r.status_code == 200, r.text
    d = r.json()
    assert "response" in d
    reply = d["response"].lower()
    assert any(kw in reply for kw in ["glucose", "abnormal", "high", "cholesterol", "john"]), \
        f"Unexpected reply: {d['response']}"
