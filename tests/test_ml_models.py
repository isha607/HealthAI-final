import pytest
from backend.ml.services import (
    predict_alzheimer, 
    predict_diabetes, 
    predict_cardio_surveillance, 
    calculate_framingham_cardio_risk
)

def test_alzheimer_prediction():
    sample_input = {
        "Age": 72, "Gender": 0, "EducationLevel": 2, "BMI": 26.0, "Smoking": 0,
        "AlcoholConsumption": 1.0, "PhysicalActivity": 3.0, "DietQuality": 7.0, "SleepQuality": 8.0,
        "FamilyHistoryAlzheimers": 0, "CardiovascularDisease": 0, "Diabetes": 0, "Depression": 0,
        "HeadInjury": 0, "Hypertension": 0, "SystolicBP": 120, "DiastolicBP": 80,
        "CholesterolTotal": 190.0, "MMSE": 28.0, "FunctionalAssessment": 9.0,
        "MemoryComplaints": 0, "BehavioralProblems": 0, "ADL": 9.0,
        "Confusion": 0, "Disorientation": 0, "PersonalityChanges": 0, "DifficultyCompletingTasks": 0
    }
    res = predict_alzheimer(sample_input)
    assert "prediction_class" in res
    assert "probability_percentage" in res
    assert "risk_level" in res
    assert res["prediction_class"] in [0, 1]
    assert "clinical_disclaimer" in res
    print("Alzheimer test passed. Result:", res["risk_level"], res["probability_percentage"])

def test_diabetes_prediction():
    sample_input = {
        "Pregnancies": 2,
        "Glucose": 145.0,
        "BloodPressure": 85.0,
        "SkinThickness": 28.0,
        "Age": 52
    }
    res = predict_diabetes(sample_input)
    assert res["prediction_class"] in [0, 1]
    assert "probability_percentage" in res
    assert "risk_level" in res
    print("Diabetes test passed. Result:", res["risk_level"], res["probability_percentage"])

def test_framingham_cardio():
    sample_input = {
        "Age": 65, "Gender": "Male", "SystolicBP": 140, "TotalCholesterol": 230.0,
        "HDLCholesterol": 42.0, "Smoking": True
    }
    res = calculate_framingham_cardio_risk(sample_input)
    assert "probability_percentage" in res
    assert res["probability_percentage"] > 0
    assert "High Risk" in res["risk_level"] or "Intermediate" in res["risk_level"]
    print("Framingham Cardio test passed. Risk:", res["risk_level"])
