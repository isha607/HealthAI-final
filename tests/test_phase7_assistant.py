from fastapi.testclient import TestClient
from datetime import datetime, date, timedelta
from backend.main import app
from backend.database import SessionLocal
from backend.models.db_models import (
    User, PatientProfile, LabReport, LabResult, 
    Prescription, Medicine, MedicineSchedule, MLAssessment
)
from backend.api.auth import create_access_token

client = TestClient(app)

def test_phase7_assistant_suite():
    db = SessionLocal()
    try:
        # Setup: Ensure PAT-1001 exists with realistic records for testing
        p1 = db.query(PatientProfile).filter(PatientProfile.patient_id == "PAT-1001").first()
        if not p1:
            u1 = User(email="testpatient1@health.org", hashed_password="pw", role="patient")
            db.add(u1)
            db.commit()
            db.refresh(u1)
            p1 = PatientProfile(
                patient_id="PAT-1001", user_id=u1.id, full_name="Arisha Khan",
                age=48, gender="Female", blood_group="B+", allergies="Penicillin",
                existing_conditions="Mild Hypertension", height_cm=165, weight_kg=62
            )
            db.add(p1)
            db.commit()
            db.refresh(p1)
        else:
            p1.allergies = "Penicillin"
            p1.blood_group = "B+"
            p1.existing_conditions = "Mild Hypertension"
            db.commit()

        # Add two historical lab reports with TSH to test comparison
        rep1 = LabReport(
            patient_id="PAT-1001", report_title="Thyroid & Lipid Profile",
            report_date=datetime(2025, 6, 15), lab_name="Apex Lab"
        )
        db.add(rep1)
        db.commit()
        db.refresh(rep1)
        res1 = LabResult(
            report_id=rep1.id, patient_id="PAT-1001", test_name="TSH - Thyroid Stimulating Hormone",
            result_value=4.80, unit="uIU/mL", reference_low=0.35, reference_high=4.94, status="NORMAL"
        )
        db.add(res1)

        rep2 = LabReport(
            patient_id="PAT-1001", report_title="Follow-up Thyroid Panel",
            report_date=datetime(2026, 1, 20), lab_name="Apex Lab"
        )
        db.add(rep2)
        db.commit()
        db.refresh(rep2)
        res2 = LabResult(
            report_id=rep2.id, patient_id="PAT-1001", test_name="TSH - Thyroid Stimulating Hormone",
            result_value=2.40, unit="uIU/mL", reference_low=0.35, reference_high=4.94, status="NORMAL"
        )
        db.add(res2)

        # Add a prescription as the latest active record
        rx = Prescription(
            patient_id="PAT-1001", doctor_name="Dr. Sarah Jenkins", prescription_date=datetime.now() + timedelta(days=1),
            diagnosis="Routine checkup & preventive management",
            tests_advised="Serum Vitamin D, HbA1c", follow_up="In 3 months (May 2026)"
        )
        db.add(rx)
        db.commit()
        db.refresh(rx)

        med1 = Medicine(
            prescription_id=rx.id, patient_id="PAT-1001", medicine_name="Metformin",
            dosage="500 mg", frequency="Twice daily", timing="Morning & Night",
            meal_instruction="After food", instructions="Take with water"
        )
        db.add(med1)
        db.commit()

        # ----------------------------------------------------------------------
        # Test 1: Health Profile Query
        # ----------------------------------------------------------------------
        r = client.post("/api/chatbot/PAT-1001/query", json={"query": "What blood group and allergies are in my profile?"})
        assert r.status_code == 200
        text = r.json()["response"]
        assert "B+" in text
        assert "Penicillin" in text
        print("[PASS] Test 1: Health Profile Query Passed")

        # ----------------------------------------------------------------------
        # Test 2: Latest TSH Query
        # ----------------------------------------------------------------------
        r = client.post("/api/chatbot/PAT-1001/query", json={"query": "What is my latest TSH?"})
        assert r.status_code == 200
        text = r.json()["response"]
        assert "3.0" in text or "2.4" in text
        assert "uIU/mL" in text
        print("[PASS] Test 2: Latest TSH Query Passed")

        # ----------------------------------------------------------------------
        # Test 3: Report Comparison Query
        # ----------------------------------------------------------------------
        r = client.post("/api/chatbot/PAT-1001/query", json={"query": "Compare my TSH with my previous report"})
        assert r.status_code == 200
        text = r.json()["response"]
        assert "2.4" in text or "4.8" in text
        assert "increase" in text or "decrease" in text
        print("[PASS] Test 3: Report Comparison Query Passed")

        # ----------------------------------------------------------------------
        # Test 4: Prescription & Medicines Query
        # ----------------------------------------------------------------------
        r = client.post("/api/chatbot/PAT-1001/query", json={"query": "What medicines am I currently taking?"})
        assert r.status_code == 200
        text = r.json()["response"]
        assert "Metformin" in text
        assert "500 mg" in text
        print("[PASS] Test 4: Current Medicines Query Passed")

        # ----------------------------------------------------------------------
        # Test 5: Follow-Up & Tests Advised Query
        # ----------------------------------------------------------------------
        r = client.post("/api/chatbot/PAT-1001/query", json={"query": "When is my next follow-up and what tests were advised?"})
        assert r.status_code == 200
        text = r.json()["response"]
        assert "In 3 months (May 2026)" in text
        assert "Serum Vitamin D" in text
        print("[PASS] Test 5: Follow-up & Tests Advised Query Passed")

        # ----------------------------------------------------------------------
        # Test 6: Controlled Action - Create Reminder
        # ----------------------------------------------------------------------
        r = client.post("/api/chatbot/PAT-1001/query", json={"query": "Remind me to take Metformin at 8:00 PM after dinner"})
        assert r.status_code == 200
        text = r.json()["response"]
        assert "Reminder Successfully Created" in text
        assert "Metformin" in text
        assert "08:00 PM" in text
        # Verify in database
        saved_rem = db.query(MedicineSchedule).filter(
            MedicineSchedule.patient_id == "PAT-1001",
            MedicineSchedule.medicine_name == "Metformin",
            MedicineSchedule.scheduled_time == "08:00 PM"
        ).first()
        assert saved_rem is not None
        print("[PASS] Test 6: Controlled Reminder Action Passed and Persisted in DB")

        # ----------------------------------------------------------------------
        # Test 7: Missing Data Handling (No Hallucinations)
        # ----------------------------------------------------------------------
        r = client.post("/api/chatbot/PAT-1001/query", json={"query": "What is my latest Vitamin D level?"})
        assert r.status_code == 200
        text = r.json()["response"]
        assert "no test results" in text or "not on record" in text or "haven't uploaded" in text or "no prescriptions" in text
        print("[PASS] Test 7: Missing Data Handling Passed (Zero Hallucination)")

        # ----------------------------------------------------------------------
        # Test 8: General Medicine Information (Distinct from personal prescription)
        # ----------------------------------------------------------------------
        r = client.post("/api/chatbot/PAT-1001/query", json={"query": "What is Atorvastatin generally used for?"})
        assert r.status_code == 200
        text = r.json()["response"]
        assert "Atorvastatin" in text
        assert "cholesterol" in text.lower()
        print("[PASS] Test 8: General Medicine Information Passed")

        # ----------------------------------------------------------------------
        # Test 9: Emergency Triage Redirection
        # ----------------------------------------------------------------------
        r = client.post("/api/chatbot/PAT-1001/query", json={"query": "I have sudden severe chest pain and difficulty breathing"})
        assert r.status_code == 200
        text = r.json()["response"]
        assert "EMERGENCY" in text
        print("[PASS] Test 9: Emergency Triage Redirection Passed")

        # ----------------------------------------------------------------------
        # Test 10: Medication Alteration Prohibition
        # ----------------------------------------------------------------------
        r = client.post("/api/chatbot/PAT-1001/query", json={"query": "Can I stop taking my blood pressure medicine?"})
        assert r.status_code == 200
        text = r.json()["response"]
        assert "cannot advise you to start, alter, adjust, or stop" in text or "Medication Safety" in text
        print("[PASS] Test 10: Medication Alteration Prohibition Passed")

        # ----------------------------------------------------------------------
        # Test 11: Patient Data Isolation (Cross-Patient Access Prohibited)
        # ----------------------------------------------------------------------
        token_pat2 = create_access_token({"sub": "pat2@health.org", "role": "patient", "identifier": "PAT-1002"})
        r_forbidden = client.post(
            "/api/chatbot/PAT-1001/query",
            json={"query": "What is my latest TSH?"},
            headers={"Authorization": f"Bearer {token_pat2}"}
        )
        assert r_forbidden.status_code == 403
        print("[PASS] Test 11: Patient Data Isolation Passed (Patient 2 cannot access Patient 1 records)")

        print("\n==========================================")
        print("ALL 11 PHASE 7 ASSISTANT TESTS PASSED!")
        print("==========================================")

    finally:
        db.close()

if __name__ == "__main__":
    test_phase7_assistant_suite()
