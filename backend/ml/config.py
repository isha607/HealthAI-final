"""
HealthAI — Machine Learning Configuration & Feature Definitions
================================================================
You can easily edit feature lists, hyperparameters, and thresholds here.
After modifying this file, simply run:
    python -m backend.ml.train_and_export
to retrain and save your models.
"""

import os

# Base Directories
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "saved_models")

# ==============================================================================
# 1. ALZHEIMER'S DISEASE MODEL SETTINGS (from Project (5) (1).ipynb)
# ==============================================================================
ALZHEIMER_CONFIG = {
    "csv_file": os.path.join(DATA_DIR, "alzheimers_disease_data.csv"),
    "target_column": "Diagnosis",
    # Columns dropped during cleaning as specified in your notebook
    "drop_columns": [
        'PatientID', 'DoctorInCharge', 'CholesterolLDL', 'Ethnicity', 
        'CholesterolHDL', 'CholesterolTriglycerides', 'Forgetfulness'
    ],
    # 27 clinical, cognitive, and lifestyle features
    "features": [
        'Age', 'Gender', 'EducationLevel', 'BMI', 'Smoking', 'AlcoholConsumption',
        'PhysicalActivity', 'DietQuality', 'SleepQuality', 'FamilyHistoryAlzheimers',
        'CardiovascularDisease', 'Diabetes', 'Depression', 'HeadInjury', 'Hypertension',
        'SystolicBP', 'DiastolicBP', 'CholesterolTotal', 'MMSE', 'FunctionalAssessment',
        'MemoryComplaints', 'BehavioralProblems', 'ADL', 'Confusion', 'Disorientation',
        'PersonalityChanges', 'DifficultyCompletingTasks'
    ],
    "test_size": 1/3,          # Matching Project (5) (1).ipynb
    "random_state": 35,        # Matching Project (5) (1).ipynb
    "model_type": "LogisticRegression",
    "model_params": {
        "random_state": 0,
        "max_iter": 1000
    }
}

# ==============================================================================
# 2. DIABETES PREDICTION MODEL SETTINGS (from diabetes-prediction notebook)
# ==============================================================================
DIABETES_CONFIG = {
    "csv_file": os.path.join(DATA_DIR, "diabetes.csv"),
    "target_column": "Outcome",
    # 5 selected features matching your notebook
    "features": [
        "Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Age"
    ],
    # Dropped columns
    "drop_columns": [
        "Outcome", "Insulin", "BMI", "DiabetesPedigreeFunction"
    ],
    "test_size": 0.12,         # Matching notebook test_size
    "random_state": 42,
    "model_type": "LogisticRegression",
    "model_params": {
        "random_state": 42,
        "max_iter": 1000
    }
}

# ==============================================================================
# 3. CARDIOVASCULAR SURVEILLANCE MODEL SETTINGS (from cardio notebook)
# ==============================================================================
CARDIO_CONFIG = {
    "csv_file": os.path.join(DATA_DIR, "heart_disease_data.csv"),
    "target_column": "Topic",
    "features": [
        'Year', 'Indicator', 'Data_Value_Unit', 'Data_Value_Alt',
        'LowConfidenceLimit', 'HighConfidenceLimit', 'Break_Out_Category',
        'Break_Out', 'PriorityArea'
    ],
    "sample_size": 10000,
    "test_size": 0.20,
    "random_state": 100,
    "pca_components": 2,
    "knn_neighbors": 5
}
