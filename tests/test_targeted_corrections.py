from datetime import datetime, timedelta
from fastapi.testclient import TestClient
from backend.main import app
from backend.ocr.document_extractor import preprocess_image_for_ocr, ocr_image_advanced
from backend.ocr.prescription_parser import parse_prescription_text
import numpy as np

client = TestClient(app)

def get_auth_token(email="patient@health.org", password="patient123"):
    resp = client.post("/api/auth/login", json={"email": email, "password": password})
    assert resp.status_code == 200, f"Login failed: {resp.text}"
    return resp.json()["access_token"]


def test_prescription_deduplication_query():
    """Verify that get_patient_prescriptions returns unique prescriptions without duplicate dates/doctors/diagnoses."""
    token = get_auth_token()
    headers = {"Authorization": f"Bearer {token}"}
    
    resp = client.get("/api/documents/patient/PAT-1001/prescriptions", headers=headers)
    assert resp.status_code == 200
    prescriptions = resp.json()
    assert len(prescriptions) > 0

    seen = set()
    for rx in prescriptions:
        key = (rx.get("prescription_date"), rx.get("doctor_name", "").strip().lower(), rx.get("diagnosis", "").strip().lower())
        assert key not in seen, f"Duplicate prescription found in query results: {key}"
        seen.add(key)


def test_prescription_confirm_idempotency():
    """Verify that confirming the same prescription twice updates the existing record rather than inserting duplicates."""
    token = get_auth_token()
    headers = {"Authorization": f"Bearer {token}"}

    test_date = "2026-09-26"
    test_doctor = "Dr. Test Targeted, MD"
    test_diagnosis = "Hypertension Assessment"
    
    payload = {
        "patient_id": "PAT-1001",
        "doctor_name": test_doctor,
        "prescription_date": test_date,
        "diagnosis": test_diagnosis,
        "follow_up": "4 weeks",
        "tests_advised": "Serum Creatinine",
        "file_path": "",
        "raw_text": "Lisinopril 10 mg once daily",
        "medicines": [
            {
                "medicine_name": "Lisinopril",
                "dosage": "10 mg",
                "frequency": "Once daily",
                "timing": "Morning",
                "meal_instruction": "After meal",
                "duration": "30 days"
            }
        ]
    }

    # First confirmation
    resp1 = client.post("/api/documents/confirm-prescription", json=payload, headers=headers)
    assert resp1.status_code == 200
    id1 = resp1.json()["prescription_id"]

    # Second confirmation with updated follow_up
    payload["follow_up"] = "6 weeks"
    resp2 = client.post("/api/documents/confirm-prescription", json=payload, headers=headers)
    assert resp2.status_code == 200
    id2 = resp2.json()["prescription_id"]

    # Must update the existing prescription instead of creating a new ID!
    assert id1 == id2, f"Prescription was duplicated! Got IDs {id1} and {id2}"

    # Verify query returns exactly 1 prescription for this event
    resp_list = client.get("/api/documents/patient/PAT-1001/prescriptions", headers=headers)
    assert resp_list.status_code == 200
    matching = [p for p in resp_list.json() if p.get("doctor_name") == test_doctor and p.get("prescription_date") == test_date]
    assert len(matching) == 1
    assert matching[0]["follow_up"] == "6 weeks"


def test_ocr_image_preprocessing_pipeline():
    """Verify that image preprocessing pipeline executes correctly on dummy/synthetic images."""
    # Create synthetic test image (gray with text-like dark boxes)
    synthetic_img = np.ones((400, 600, 3), dtype=np.uint8) * 240
    # Add dark synthetic strokes
    synthetic_img[50:90, 50:200] = 30
    synthetic_img[120:150, 50:350] = 40

    enhanced, thresh = preprocess_image_for_ocr(synthetic_img)
    assert enhanced is not None
    assert thresh is not None
    assert enhanced.shape[0] >= 1200 or enhanced.shape[1] >= 1200, "Should upscale low resolution input"


def test_prescription_parser_uncertainty_flagging():
    """Verify that low-quality or unrecognizable prescription text is flagged as uncertain and needs_review without fake medicines."""
    unintelligible_text = "asdkjh 12398 kjhkjasd /// ???"
    res = parse_prescription_text(unintelligible_text, ocr_conf=0.25)
    
    assert res["is_uncertain"] is True
    assert res["needs_review"] is True
    assert res["confidence"] == 0.25
    assert len(res["review_notes"]) > 0
    # Should not insert fake dummy medication names
    assert len(res["medicines"]) == 0


def test_prescription_parser_standard_extraction():
    """Verify accurate extraction of known medication, dosing, and follow-up intervals."""
    text = (
        "Apex Multi-Specialty Clinic\n"
        "Dr. Sarah Jenkins, MD\n"
        "Date: 2026-09-16\n"
        "Diagnosis: Essential Hypertension & Type 2 Diabetes\n"
        "Rx:\n"
        "Metformin 500 mg twice daily after meals for 30 days\n"
        "Lisinopril 10 mg once daily in morning\n"
        "Follow-up: 4 weeks\n"
        "Tests Advised: Fasting Blood Sugar, HbA1c, Serum Creatinine\n"
    )
    res = parse_prescription_text(text, ocr_conf=0.92)
    
    assert res["is_uncertain"] is False
    assert "Jenkins" in res["doctor_name"]
    assert res["prescription_date"] == "2026-09-16"
    assert res["follow_up"] == "4 weeks"
    assert len(res["medicines"]) == 2
    med_names = [m["medicine_name"].lower() for m in res["medicines"]]
    assert "metformin" in med_names
    assert "lisinopril" in med_names


def test_vitals_trends_endpoint():
    """Verify that patient trends endpoint correctly returns chronological biomarkers."""
    token = get_auth_token()
    headers = {"Authorization": f"Bearer {token}"}
    
    resp = client.get("/api/patient/trends/PAT-1001", headers=headers)
    assert resp.status_code == 200
    data = resp.json()
    assert "dates" in data
    assert "glucose" in data
    assert "systolic_bp" in data
    assert "total_cholesterol" in data

if __name__ == "__main__":
    print("Running targeted corrections tests...")
    test_prescription_deduplication_query()
    print("PASS: test_prescription_deduplication_query")
    test_prescription_confirm_idempotency()
    print("PASS: test_prescription_confirm_idempotency")
    test_ocr_image_preprocessing_pipeline()
    print("PASS: test_ocr_image_preprocessing_pipeline")
    test_prescription_parser_uncertainty_flagging()
    print("PASS: test_prescription_parser_uncertainty_flagging")
    test_prescription_parser_standard_extraction()
    print("PASS: test_prescription_parser_standard_extraction")
    test_vitals_trends_endpoint()
    print("PASS: test_vitals_trends_endpoint")
    print("\nALL 6 TARGETED CORRECTIONS TESTS PASSED SUCCESSFULLY!")

