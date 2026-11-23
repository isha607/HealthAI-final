import pytest
from backend.ocr.report_parser import parse_lab_text
from backend.ocr.prescription_parser import parse_prescription_text

def test_lab_report_parsing():
    raw_report = """
    Fasting Blood Glucose: 125 mg/dL
    Total Cholesterol: 240 mg/dL
    HDL: 38 mg/dL
    Systolic BP: 142 mmHg
    Hemoglobin: 14.5 g/dL
    """
    results = parse_lab_text(raw_report)
    assert len(results) >= 4
    
    # Check glucose is HIGH
    glucose_item = next(r for r in results if "Glucose" in r["test_name"])
    assert glucose_item["result_value"] == 125.0
    assert glucose_item["status"] == "HIGH"
    assert "above the normal reference range" in glucose_item["clinical_explanation"]

    # Check cholesterol is HIGH
    chol_item = next(r for r in results if "Cholesterol" in r["test_name"])
    assert chol_item["result_value"] == 240.0
    assert chol_item["status"] == "HIGH"
    print("Lab OCR parsing test passed!")

def test_prescription_parsing():
    raw_rx = """
    Metformin 500 mg twice daily after meals.
    Atorvastatin 20 mg once daily at bedtime.
    """
    meds = parse_prescription_text(raw_rx)
    assert len(meds) == 2
    assert meds[0]["medicine_name"] == "Metformin"
    assert "antidiabetic" in meds[0]["purpose_explanation"].lower()
    assert meds[1]["medicine_name"] == "Atorvastatin"
    assert "statin" in meds[1]["purpose_explanation"].lower()
    print("Prescription parsing test passed!")
