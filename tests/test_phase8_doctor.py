"""
Test Suite for Phase 8: Doctor Portal + Patient Data Integration + Clinical History View.

Verifies:
1. Doctor login and token authentication.
2. Doctor-Patient authorization and privacy enforcement (blocking unauthorized doctors and patient tokens).
3. Retrieval of organized clinical snapshot for authorized patient (Profile, Medications, Latest Report, Tests Advised, Follow-up, AI Assessments, Vitals).
4. Safe document URL handling (no system path leakage).
5. Individual report & prescription inspection endpoints.
6. Clinical visit recording.
7. Patient clinical panel assignment.
"""
from fastapi.testclient import TestClient
from datetime import datetime
import json

from backend.main import app
from backend.database import SessionLocal
from backend.models.db_models import (
    User, PatientProfile, DoctorProfile, DoctorVisit,
    LabReport, LabResult, Prescription, Medicine,
    MLAssessment, DoctorPatientRelationship
)
from backend.api.auth import create_access_token, get_password_hash

client = TestClient(app)


def test_phase8_doctor_suite():
    db = SessionLocal()
    try:
        # --- 1. SETUP TEST ENTITIES ---
        # Ensure Doctor DOC-501 exists
        doc_user = db.query(User).filter(User.email == "doctor@hospital.org").first()
        if not doc_user:
            doc_user = User(email="doctor@hospital.org", hashed_password=get_password_hash("doctor123"), role="doctor")
            db.add(doc_user)
            db.commit()
            db.refresh(doc_user)

        doc_profile = db.query(DoctorProfile).filter(DoctorProfile.user_id == doc_user.id).first()
        if not doc_profile:
            doc_profile = DoctorProfile(
                doctor_code="DOC-501",
                user_id=doc_user.id,
                full_name="Dr. Sarah Jenkins, MD",
                specialization="Cardiology & Geriatrics"
            )
            db.add(doc_profile)
            db.commit()
            db.refresh(doc_profile)

        # Create a second Doctor (DOC-502) to test authorization isolation
        doc2_user = db.query(User).filter(User.email == "doctor2@hospital.org").first()
        if not doc2_user:
            doc2_user = User(email="doctor2@hospital.org", hashed_password=get_password_hash("doc2pass"), role="doctor")
            db.add(doc2_user)
            db.commit()
            db.refresh(doc2_user)

        doc2_profile = db.query(DoctorProfile).filter(DoctorProfile.user_id == doc2_user.id).first()
        if not doc2_profile:
            doc2_profile = DoctorProfile(
                doctor_code="DOC-502",
                user_id=doc2_user.id,
                full_name="Dr. Marcus Vance, MD",
                specialization="Neurology"
            )
            db.add(doc2_profile)
            db.commit()
            db.refresh(doc2_profile)

        # Ensure Patient PAT-1001 exists
        p1 = db.query(PatientProfile).filter(PatientProfile.patient_id == "PAT-1001").first()
        if not p1:
            u1 = User(email="patient@health.org", hashed_password=get_password_hash("patient123"), role="patient")
            db.add(u1)
            db.commit()
            db.refresh(u1)
            p1 = PatientProfile(
                patient_id="PAT-1001",
                user_id=u1.id,
                full_name="Arisha Khan",
                age=48,
                gender="Female",
                blood_group="B+",
                allergies="Penicillin",
                existing_conditions="Hypertension"
            )
            db.add(p1)
            db.commit()
            db.refresh(p1)

        # Ensure PAT-1001 is assigned to DOC-501
        rel1 = db.query(DoctorPatientRelationship).filter(
            DoctorPatientRelationship.doctor_id == doc_profile.id,
            DoctorPatientRelationship.patient_id == "PAT-1001"
        ).first()
        if not rel1:
            db.add(DoctorPatientRelationship(
                doctor_id=doc_profile.id,
                patient_id="PAT-1001",
                assigned_date=datetime.utcnow()
            ))
            db.commit()

        # Create isolated Patient PAT-2002 assigned ONLY to Doctor 2 (DOC-502)
        p2 = db.query(PatientProfile).filter(PatientProfile.patient_id == "PAT-2002").first()
        if not p2:
            u2 = User(email="patient2@health.org", hashed_password=get_password_hash("patient2pass"), role="patient")
            db.add(u2)
            db.commit()
            db.refresh(u2)
            p2 = PatientProfile(
                patient_id="PAT-2002",
                user_id=u2.id,
                full_name="Robert Smith",
                age=55,
                gender="Male",
                blood_group="O+",
                allergies="None",
                existing_conditions="Type 2 Diabetes"
            )
            db.add(p2)
            db.commit()
            db.refresh(p2)

        # Ensure PAT-2002 is NOT assigned to DOC-501 (for cross-patient access test)
        db.query(DoctorPatientRelationship).filter(
            DoctorPatientRelationship.doctor_id == doc_profile.id,
            DoctorPatientRelationship.patient_id == "PAT-2002"
        ).delete()
        db.commit()

        rel2 = db.query(DoctorPatientRelationship).filter(
            DoctorPatientRelationship.doctor_id == doc2_profile.id,
            DoctorPatientRelationship.patient_id == "PAT-2002"
        ).first()
        if not rel2:
            db.add(DoctorPatientRelationship(
                doctor_id=doc2_profile.id,
                patient_id="PAT-2002",
                assigned_date=datetime.utcnow()
            ))
            db.commit()

        # Generate JWT tokens
        doc1_token = create_access_token({"sub": str(doc_user.id), "role": "doctor", "identifier": "DOC-501"})
        doc2_token = create_access_token({"sub": str(doc2_user.id), "role": "doctor", "identifier": "DOC-502"})
        pat_token = create_access_token({"sub": str(p1.user_id), "role": "patient", "identifier": "PAT-1001"})

        doc1_headers = {"Authorization": f"Bearer {doc1_token}"}
        doc2_headers = {"Authorization": f"Bearer {doc2_token}"}
        pat_headers = {"Authorization": f"Bearer {pat_token}"}

        print("\n--- RUNNING PHASE 8 VERIFICATION TESTS ---")

        # TEST 1: Doctor Login & Directory Retrieval
        res_dir = client.get("/api/doctor/patients", headers=doc1_headers)
        assert res_dir.status_code == 200, f"Failed listing doctor patients: {res_dir.text}"
        patients = res_dir.json()
        assert len(patients) > 0, "Expected at least 1 patient in roster"
        pat_ids = [p["patient_id"] for p in patients]
        assert "PAT-1001" in pat_ids, "PAT-1001 should be present in DOC-501 roster"
        print("[PASS] Test 1: Doctor Login & Patient Directory Roster")

        # TEST 2: Patient Token Blocked from Doctor API (Role Authorization)
        res_pat_block = client.get("/api/doctor/patients", headers=pat_headers)
        assert res_pat_block.status_code == 403, f"Patient token should be forbidden from doctor endpoint: {res_pat_block.status_code}"
        print("[PASS] Test 2: Role Authorization Enforcement (Patient Token Blocked)")

        # TEST 3: Doctor Cross-Patient Privacy / Isolation
        # DOC-501 should be BLOCKED from accessing PAT-2002 (assigned exclusively to DOC-502)
        res_unauth = client.get("/api/doctor/patient/PAT-2002/details", headers=doc1_headers)
        assert res_unauth.status_code == 403, f"DOC-501 should be forbidden from accessing PAT-2002: {res_unauth.status_code}"
        assert "not authorized" in res_unauth.json()["detail"].lower()
        print("[PASS] Test 3: Doctor Privacy & Access Control (Cross-Patient Access Blocked)")

        # TEST 4: Nonexistent Patient Returns 404
        res_404 = client.get("/api/doctor/patient/PAT-9999/details", headers=doc1_headers)
        assert res_404.status_code == 404, f"Nonexistent patient should return 404: {res_404.status_code}"
        print("[PASS] Test 4: Nonexistent Patient ID Returns 404 Not Found")

        # TEST 5: Authorized Patient Clinical Snapshot
        res_details = client.get("/api/doctor/patient/PAT-1001/details", headers=doc1_headers)
        assert res_details.status_code == 200, f"Authorized detail request failed: {res_details.text}"
        data = res_details.json()

        # A. Profile & Snapshot
        prof = data["profile"]
        assert prof["patient_id"] == "PAT-1001"
        assert prof["full_name"] == "Arisha Khan"
        assert prof["blood_group"] == "B+"
        assert prof["allergies"] == "Penicillin"

        # B. Current Medications
        assert "current_medications" in data
        assert isinstance(data["current_medications"], list)
        if len(data["current_medications"]) > 0:
            med0 = data["current_medications"][0]
            assert "medicine_name" in med0
            assert "dosage" in med0
            assert "frequency" in med0
            assert "timing" in med0
            assert "meal_instruction" in med0

        # C. Latest Lab Report & Parameters
        assert "latest_report" in data
        if data["latest_report"]:
            rep = data["latest_report"]
            assert "report_title" in rep
            assert "results" in rep
            assert isinstance(rep["results"], list)

        # D. Advised Tests & Follow-Up
        assert "advised_tests" in data
        assert "next_follow_up" in data

        # E. AI Risk Assessment History
        assert "ai_risk_assessments" in data
        assert isinstance(data["ai_risk_assessments"], list)
        if len(data["ai_risk_assessments"]) > 0:
            asm0 = data["ai_risk_assessments"][0]
            assert "assessment_type" in asm0
            assert "risk_level" in asm0
            assert "probability" in asm0
            assert "source_type" in asm0

        # F. Key Vitals
        assert "key_vitals" in data
        assert "bp" in data["key_vitals"]
        assert "glucose" in data["key_vitals"]

        # G. Longitudinal Timeline
        assert "timeline" in data
        assert len(data["timeline"]) > 0

        print("[PASS] Test 5: Comprehensive Clinical Snapshot Retrieved for Doctor")

        # TEST 6: Document Path Normalization (No System Leaks)
        # Check that file_url (if present) begins with /uploads/ and does not contain C: or \\
        for rx in data.get("prescriptions_history", []):
            if rx.get("file_url"):
                assert rx["file_url"].startswith("/uploads/"), f"Insecure or invalid document URL: {rx['file_url']}"
                assert "\\" not in rx["file_url"], f"Backslashes leaked in URL: {rx['file_url']}"

        for rp in data.get("reports_history", []):
            if rp.get("file_url"):
                assert rp["file_url"].startswith("/uploads/"), f"Insecure or invalid document URL: {rp['file_url']}"
                assert "\\" not in rp["file_url"], f"Backslashes leaked in URL: {rp['file_url']}"
        print("[PASS] Test 6: Document URL Security & System Path Sanitization")

        # TEST 7: Single Report & Prescription Inspection Endpoints
        latest_rx = data.get("latest_prescription")
        if latest_rx:
            rx_id = latest_rx["id"]
            res_rx = client.get(f"/api/doctor/patient/PAT-1001/prescription/{rx_id}", headers=doc1_headers)
            assert res_rx.status_code == 200
            assert res_rx.json()["id"] == rx_id
            assert "medicines" in res_rx.json()

        latest_rep = data.get("latest_report")
        if latest_rep:
            rep_id = latest_rep["id"]
            res_rep = client.get(f"/api/doctor/patient/PAT-1001/report/{rep_id}", headers=doc1_headers)
            assert res_rep.status_code == 200
            assert res_rep.json()["id"] == rep_id
            assert "results" in res_rep.json()
        print("[PASS] Test 7: Individual Report & Prescription Inspection Endpoints")

        # TEST 8: Clinical Visit Recording
        visit_payload = {
            "doctor_name": "Dr. Sarah Jenkins, MD",
            "chief_complaint": "Follow-up for cardiometabolic review",
            "diagnosis": "Stage 1 Essential Hypertension (Well Controlled)",
            "clinical_notes": "Patient reports strict medication adherence. Blood pressure normalized at 124/80 mmHg.",
            "vitals_bp": "124/80",
            "vitals_heart_rate": 70,
            "vitals_glucose": 98.0,
            "follow_up_date": "6 months"
        }
        res_visit = client.post("/api/doctor/patient/PAT-1001/visit", json=visit_payload, headers=doc1_headers)
        assert res_visit.status_code == 200, f"Visit recording failed: {res_visit.text}"
        assert res_visit.json()["status"] == "success"
        print("[PASS] Test 8: Clinical Doctor Visit Recording")

        # TEST 9: Patient Panel Assignment Endpoint
        # Assign PAT-2002 to DOC-501 via formal assignment
        res_assign = client.post("/api/doctor/patient/PAT-2002/assign", headers=doc1_headers)
        assert res_assign.status_code == 200, f"Assignment failed: {res_assign.text}"
        assert "assigned" in res_assign.json()["message"].lower()

        # Now DOC-501 should be permitted to access PAT-2002
        res_assigned_access = client.get("/api/doctor/patient/PAT-2002/details", headers=doc1_headers)
        assert res_assigned_access.status_code == 200, f"Access after assignment failed: {res_assigned_access.status_code}"
        print("[PASS] Test 9: Doctor-Patient Panel Assignment & Access Grant")

        # TEST 10: Backwards Compatibility with Unauthenticated Summary
        res_summary = client.get("/api/doctor/patient/PAT-1001/summary")
        assert res_summary.status_code == 200
        assert "patient_name" in res_summary.json()
        print("[PASS] Test 10: Backwards Compatibility for Unauthenticated Summary")

        print("\n==========================================")
        print("ALL 10 PHASE 8 DOCTOR PORTAL TESTS PASSED!")
        print("==========================================")

    finally:
        db.close()


if __name__ == "__main__":
    test_phase8_doctor_suite()
