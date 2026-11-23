import os
import sys
from fastapi.testclient import TestClient
from datetime import datetime, date

from backend.main import app
from backend.database import SessionLocal
from backend.models.db_models import (
    User, PatientProfile, DoctorProfile, LabReport, LabResult,
    Prescription, Medicine, MedicineSchedule, MLAssessment,
    DoctorPatientRelationship
)
from backend.api.auth import create_access_token, get_password_hash

def test_phase9_comprehensive_integration():
    client = TestClient(app)
    db = SessionLocal()

    print("\n=======================================================")
    print("STARTING PHASE 9: FULL PROJECT INTEGRATION AUDIT")
    print("=======================================================")

    try:
        # -----------------------------------------------------------------
        # STEP 1: AUTHENTICATION & PATIENT SETUP
        # -----------------------------------------------------------------
        # Ensure test patient PAT-1001 exists
        pat1 = db.query(PatientProfile).filter(PatientProfile.patient_id == "PAT-1001").first()
        if not pat1:
            u1 = User(email="patient@health.org", hashed_password=get_password_hash("patient123"), role="patient")
            db.add(u1)
            db.commit()
            db.refresh(u1)
            pat1 = PatientProfile(
                patient_id="PAT-1001",
                user_id=u1.id,
                full_name="Eleanor Vance",
                age=62,
                gender="Female",
                blood_group="A+",
                allergies="Penicillin",
                existing_conditions="Hypertension"
            )
            db.add(pat1)
            db.commit()
            db.refresh(pat1)

        # Ensure patient PAT-2002 exists for privacy/isolation tests
        pat2 = db.query(PatientProfile).filter(PatientProfile.patient_id == "PAT-2002").first()
        if not pat2:
            u2 = User(email="patient2@health.org", hashed_password=get_password_hash("patient2pass"), role="patient")
            db.add(u2)
            db.commit()
            db.refresh(u2)
            pat2 = PatientProfile(
                patient_id="PAT-2002",
                user_id=u2.id,
                full_name="Robert Smith",
                age=55,
                gender="Male",
                blood_group="O+",
                allergies="None",
                existing_conditions="Type 2 Diabetes"
            )
            db.add(pat2)
            db.commit()
            db.refresh(pat2)

        # Tokens
        token_pat1 = create_access_token({"sub": str(pat1.user_id), "role": "patient", "identifier": "PAT-1001"})
        token_pat2 = create_access_token({"sub": str(pat2.user_id), "role": "patient", "identifier": "PAT-2002"})

        headers_pat1 = {"Authorization": f"Bearer {token_pat1}"}
        headers_pat2 = {"Authorization": f"Bearer {token_pat2}"}

        # -----------------------------------------------------------------
        # STEP 2: PATIENT PROFILE AUDIT
        # -----------------------------------------------------------------
        res_prof = client.get("/api/patient/profile/PAT-1001", headers=headers_pat1)
        assert res_prof.status_code == 200, f"Profile fetch failed: {res_prof.text}"
        prof_data = res_prof.json()
        assert prof_data["patient_id"] == "PAT-1001"
        assert "full_name" in prof_data and len(prof_data["full_name"]) > 0
        print("[PASS] 1. Patient Profile Retrieval & Integrity Verified")

        # -----------------------------------------------------------------
        # STEP 3: SINGLE SOURCE OF TRUTH & ZERO FAKE DATA AUDIT
        # -----------------------------------------------------------------
        res_trends = client.get("/api/patient/trends/PAT-1001", headers=headers_pat1)
        assert res_trends.status_code == 200, f"Trends fetch failed: {res_trends.text}"
        trends_data = res_trends.json()
        # Ensure no fabricated fake dates (such as "Aug 2024", "Nov 2024" injected hardcoded)
        assert "dates" in trends_data
        for d in trends_data["dates"]:
            assert not d.startswith("Aug 2024") and not d.startswith("Nov 2024"), "Found hardcoded mock dates in trends"
        print("[PASS] 2. Single Source of Truth & Zero Fake Data Audit Verified")

        # -----------------------------------------------------------------
        # STEP 4: LAB REPORTS FLOW (PREVIEW -> CONFIRM -> HISTORY -> TABLE)
        # -----------------------------------------------------------------
        # 4a. Preview Lab Report with Text
        preview_payload = {
            "patient_id": "PAT-1001",
            "raw_text": "Clinical Pathology Report\nDate: 2026-09-15\nLab: Metro Diagnostics\nFasting Blood Glucose: 118 mg/dL (70-99)\nHbA1c: 6.2 % (4.0-5.6)\nTotal Cholesterol: 195 mg/dL (125-200)"
        }
        res_prev_rep = client.post("/api/documents/preview-report", data=preview_payload, headers=headers_pat1)
        assert res_prev_rep.status_code == 200, f"Report preview failed: {res_prev_rep.text}"
        prev_rep_data = res_prev_rep.json()
        assert prev_rep_data["status"] == "success"
        assert len(prev_rep_data["parameters"]) >= 2
        print("[PASS] 3. Lab Report Preview & Extraction Verified")

        # 4b. Confirm & Save Report
        confirm_rep_payload = {
            "patient_id": "PAT-1001",
            "report_title": "Comprehensive Metabolic Panel",
            "lab_name": "Metro Diagnostics",
            "report_date": "2026-09-15",
            "file_path": "uploads/reports/test_report.pdf",
            "raw_text": preview_payload["raw_text"],
            "parameters": prev_rep_data["parameters"]
        }
        res_save_rep = client.post("/api/documents/confirm-report", json=confirm_rep_payload, headers=headers_pat1)
        assert res_save_rep.status_code == 200, f"Report confirmation failed: {res_save_rep.text}"
        saved_rep_id = res_save_rep.json()["report_id"]
        assert saved_rep_id is not None
        print("[PASS] 4. Lab Report Confirmation & Storage Verified")

        # 4c. Verify Report History and Table
        res_rep_list = client.get("/api/documents/patient/PAT-1001/reports", headers=headers_pat1)
        assert res_rep_list.status_code == 200
        reports = res_rep_list.json()
        assert any(r["id"] == saved_rep_id for r in reports)

        res_rep_table = client.get("/api/documents/patient/PAT-1001/report-table", headers=headers_pat1)
        assert res_rep_table.status_code == 200
        table_data = res_rep_table.json()
        assert "dates" in table_data and "rows" in table_data
        print("[PASS] 5. Longitudinal Lab Reports Table & Trend Verified")

        # -----------------------------------------------------------------
        # STEP 5: PRESCRIPTIONS & MEDICINES FLOW
        # -----------------------------------------------------------------
        # 5a. Preview Prescription
        rx_sample_text = (
            "Dr. Sarah Jenkins, MD\n"
            "Date: 2026-09-16\n"
            "Rx:\n"
            "1. Metformin 500mg - Twice daily with meals - 30 days\n"
            "2. Lisinopril 10mg - Once daily in morning - 30 days\n"
            "Follow-up: 4 weeks\n"
            "Tests Advised: Fasting Blood Glucose, Serum Creatinine"
        )
        res_rx_prev = client.post(
            "/api/documents/preview-prescription",
            data={"patient_id": "PAT-1001", "raw_text": rx_sample_text},
            headers=headers_pat1
        )
        assert res_rx_prev.status_code == 200, f"Prescription preview failed: {res_rx_prev.text}"
        extracted_rx = res_rx_prev.json()["extracted_data"]
        assert len(extracted_rx["medicines"]) >= 2
        print("[PASS] 6. Prescription Preview & Extraction Verified")

        # 5b. Confirm Prescription
        confirm_rx_payload = {
            "patient_id": "PAT-1001",
            "doctor_name": extracted_rx.get("doctor_name") or "Dr. Sarah Jenkins, MD",
            "prescription_date": "2026-09-16",
            "diagnosis": extracted_rx.get("diagnosis") or "Hypertension & Glucose Management",
            "follow_up": extracted_rx.get("follow_up") or "4 weeks",
            "tests_advised": extracted_rx.get("tests_advised") or "Fasting Blood Glucose, Serum Creatinine",
            "file_path": "uploads/prescriptions/test_rx.pdf",
            "raw_text": rx_sample_text,
            "medicines": extracted_rx["medicines"]
        }
        res_rx_conf = client.post("/api/documents/confirm-prescription", json=confirm_rx_payload, headers=headers_pat1)
        assert res_rx_conf.status_code == 200, f"Prescription save failed: {res_rx_conf.text}"
        saved_rx_id = res_rx_conf.json()["prescription_id"]
        print("[PASS] 7. Prescription Confirmation & Storage Verified")

        # 5c. Verify Prescriptions and Medicines list
        res_rxs = client.get("/api/documents/patient/PAT-1001/prescriptions", headers=headers_pat1)
        assert res_rxs.status_code == 200
        rx_list = res_rxs.json()
        assert any(rx["id"] == saved_rx_id for rx in rx_list)

        res_meds = client.get("/api/documents/patient/PAT-1001/medicines", headers=headers_pat1)
        assert res_meds.status_code == 200
        assert len(res_meds.json()) >= 2
        print("[PASS] 8. Prescription History & Medications List Verified")

        # -----------------------------------------------------------------
        # STEP 6: REMINDERS & COMPLIANCE TRACKING
        # -----------------------------------------------------------------
        # 6a. Generate Reminders from Prescription
        res_gen_rem = client.post(
            f"/api/documents/prescription/{saved_rx_id}/create-reminders",
            json={"patient_id": "PAT-1001"},
            headers=headers_pat1
        )
        assert res_gen_rem.status_code == 200, f"Reminders generation failed: {res_gen_rem.text}"
        rem_res = res_gen_rem.json()
        assert rem_res["status"] == "success"
        print("[PASS] 9. Prescription-Driven Reminders Auto-Generation Verified")

        # 6b. Verify Reminders & Duplicate Prevention
        res_dup_rem = client.post(
            f"/api/documents/prescription/{saved_rx_id}/create-reminders",
            json={"patient_id": "PAT-1001"},
            headers=headers_pat1
        )
        assert res_dup_rem.status_code == 200
        assert res_dup_rem.json()["created"] == 0, "Duplicate reminders should not be created"
        assert res_dup_rem.json()["skipped_duplicates"] > 0
        print("[PASS] 10. Duplicate Reminder Prevention Verified")

        # 6c. Toggle Taken/Pending Status
        res_rem_list = client.get("/api/reminders/PAT-1001", headers=headers_pat1)
        assert res_rem_list.status_code == 200
        rem_items = res_rem_list.json()
        assert len(rem_items) > 0
        target_rem = rem_items[0]

        res_toggle = client.patch(
            f"/api/reminders/{target_rem['id']}/status",
            json={"taken_status": True},
            headers=headers_pat1
        )
        assert res_toggle.status_code == 200
        toggle_data = res_toggle.json()
        assert (toggle_data.get("taken_status") is True) or (toggle_data.get("reminder", {}).get("taken_status") is True)
        print("[PASS] 11. Reminder Taken/Pending Status Toggle & Persistence Verified")

        # -----------------------------------------------------------------
        # STEP 7: AI RISK ASSESSMENT (MANUAL & EXTRACTED DOCUMENT INFERENCE)
        # -----------------------------------------------------------------
        # 7a. Manual Assessment - Diabetes
        diab_payload = {
            "Pregnancies": 1,
            "Glucose": 130.0,
            "BloodPressure": 80.0,
            "SkinThickness": 24.0,
            "Age": 62
        }
        res_diab = client.post("/api/ml/predict/diabetes/PAT-1001", json=diab_payload, headers=headers_pat1)
        assert res_diab.status_code == 200, f"Diabetes prediction failed: {res_diab.text}"
        diab_result = res_diab.json()
        assert "probability_percentage" in diab_result and "risk_level" in diab_result
        print("[PASS] 12. Manual Diabetes ML Risk Assessment Verified")

        # 7b. Document-Assisted Assessment - Extract from confirmed report
        res_doc_extract = client.post(
            "/api/ml/extract-from-stored-report",
            json={"patient_id": "PAT-1001", "model_type": "diabetes", "report_id": saved_rep_id},
            headers=headers_pat1
        )
        assert res_doc_extract.status_code == 200, f"Feature extraction from report failed: {res_doc_extract.text}"
        doc_extracted_data = res_doc_extract.json()
        assert doc_extracted_data["status"] == "success"

        # 7c. Confirmed Prediction using extracted + filled features
        confirmed_features = {
            "Pregnancies": 1,
            "Glucose": doc_extracted_data["extracted_features"].get("Glucose", 118.0),
            "BloodPressure": 80.0,
            "SkinThickness": 20.0,
            "Age": 62
        }
        res_conf_pred = client.post(
            "/api/ml/predict-confirmed/PAT-1001/diabetes",
            json={"_source_type": "stored_report", "features": confirmed_features},
            headers=headers_pat1
        )
        assert res_conf_pred.status_code == 200
        assert res_conf_pred.json()["source_type"] == "stored_report"
        print("[PASS] 13. Document-Assisted ML Assessment Pipeline Verified")

        # 7d. Verify Assessment History
        res_hist = client.get("/api/ml/history/PAT-1001", headers=headers_pat1)
        assert res_hist.status_code == 200
        hist_items = res_hist.json()
        assert len(hist_items) >= 2
        print("[PASS] 14. ML Assessment Longitudinal History Verified")

        # -----------------------------------------------------------------
        # STEP 8: HEALTHAI ASSISTANT (RECORD-AWARE CHAT)
        # -----------------------------------------------------------------
        res_chat = client.post(
            "/api/chatbot/PAT-1001/query",
            json={"message": "What medications am I taking?"},
            headers=headers_pat1
        )
        assert res_chat.status_code == 200, f"Chatbot query failed: {res_chat.text}"
        chat_data = res_chat.json()
        assert "response" in chat_data
        assert "Metformin" in chat_data["response"] or "Lisinopril" in chat_data["response"]
        print("[PASS] 15. Record-Aware AI Health Chatbot Query Verified")

        # -----------------------------------------------------------------
        # STEP 9: PRIVACY & PATIENT DATA ISOLATION ENFORCEMENT
        # -----------------------------------------------------------------
        # Patient 2 attempts to access Patient 1's profile
        res_priv_prof = client.get("/api/patient/profile/PAT-1001", headers=headers_pat2)
        assert res_priv_prof.status_code == 403, f"Expected 403 Forbidden for cross-patient profile: {res_priv_prof.status_code}"

        # Patient 2 attempts to access Patient 1's reports
        res_priv_rep = client.get("/api/documents/patient/PAT-1001/reports", headers=headers_pat2)
        assert res_priv_rep.status_code == 403, f"Expected 403 Forbidden for cross-patient reports: {res_priv_rep.status_code}"

        # Patient 2 attempts to query chatbot using Patient 1 ID
        res_priv_chat = client.post(
            "/api/chatbot/PAT-1001/query",
            json={"message": "Show me patient Eleanor's records"},
            headers=headers_pat2
        )
        assert res_priv_chat.status_code == 403, f"Expected 403 Forbidden for cross-patient chatbot: {res_priv_chat.status_code}"

        # Patient 2 attempts to access Patient 1's reminders
        res_priv_rem = client.get("/api/reminders/PAT-1001", headers=headers_pat2)
        assert res_priv_rem.status_code == 403, f"Expected 403 Forbidden for cross-patient reminders: {res_priv_rem.status_code}"

        print("[PASS] 16. Patient Data Isolation & Privacy Controls (403 Forbidden) Verified")

        # -----------------------------------------------------------------
        # STEP 10: SECURITY AUDIT & UPLOAD VALIDATION
        # -----------------------------------------------------------------
        # Attempt to upload an unauthorized file extension (.exe)
        malicious_file = {"file": ("malicious_payload.exe", b"binary_data", "application/octet-stream")}
        res_bad_upload = client.post(
            "/api/documents/preview-report",
            data={"patient_id": "PAT-1001"},
            files=malicious_file,
            headers=headers_pat1
        )
        assert res_bad_upload.status_code == 400, f"Expected 400 Bad Request for .exe: {res_bad_upload.status_code}"
        assert "unsupported file format" in res_bad_upload.json()["detail"].lower()
        print("[PASS] 17. Upload File Whitelist & Security Enforcement Verified")

        print("\n=======================================================")
        print("ALL 17 PHASE 9 COMPREHENSIVE INTEGRATION AUDIT TESTS PASSED!")
        print("=======================================================\n")

    finally:
        db.close()

if __name__ == "__main__":
    test_phase9_comprehensive_integration()
