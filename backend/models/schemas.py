from pydantic import BaseModel, EmailStr, field_validator, model_validator
from typing import List, Optional, Dict, Any
from datetime import datetime

# --- Auth Schemas ---
class UserRegister(BaseModel):
    email: EmailStr
    password: str
    role: str # 'patient' or 'doctor'
    full_name: str
    age: Optional[int] = 35
    gender: Optional[str] = "Female"

    @field_validator("role")
    @classmethod
    def validate_role(cls, v: str) -> str:
        if v not in ("patient", "doctor"):
            raise ValueError("Role must be 'patient' or 'doctor'")
        return v

    @field_validator("password")
    @classmethod
    def validate_password(cls, v: str) -> str:
        if len(v) < 6:
            raise ValueError("Password must be at least 6 characters")
        return v

    @field_validator("full_name")
    @classmethod
    def validate_full_name(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Full name cannot be empty")
        return v.strip()

class UserLogin(BaseModel):
    email: EmailStr
    password: str

class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user_id: int
    role: str
    identifier: str # patient_id or doctor_code
    full_name: str

# --- Profile Schemas ---
class PatientProfileSchema(BaseModel):
    patient_id: str
    full_name: str
    age: int
    gender: str
    height_cm: float
    weight_kg: float
    bmi: float
    phone: str
    blood_group: str
    smoking_status: str
    alcohol_consumption: float
    physical_activity_hours: float
    diet_quality_score: float
    sleep_hours: float
    family_history_alzheimers: bool
    cardiovascular_disease: bool
    diabetes_history: bool
    hypertension: bool
    allergies: str
    existing_conditions: str

    class Config:
        from_attributes = True

# --- ML Schemas ---
class AlzheimerInput(BaseModel):
    Age: int = 70
    Gender: int = 0  # 0: Male, 1: Female
    Ethnicity: int = 0 # 0: Caucasian, 1: African American, 2: Asian, 3: Other
    EducationLevel: int = 2 # 0: None, 1: High School, 2: Bachelor, 3: Higher
    BMI: float = 23.5
    Smoking: int = 0
    AlcoholConsumption: float = 2.0
    PhysicalActivity: float = 3.0
    DietQuality: float = 7.0
    SleepQuality: float = 7.5
    FamilyHistoryAlzheimers: int = 0
    CardiovascularDisease: int = 0
    Diabetes: int = 0
    Depression: int = 0
    HeadInjury: int = 0
    Hypertension: int = 0
    SystolicBP: int = 125
    DiastolicBP: int = 80
    CholesterolTotal: float = 195.0
    CholesterolLDL: float = 110.0
    CholesterolHDL: float = 55.0
    CholesterolTriglycerides: float = 145.0
    MMSE: float = 27.0 # Mini-Mental State Examination (0-30)
    FunctionalAssessment: float = 8.5 # 0-10
    MemoryComplaints: int = 0
    BehavioralProblems: int = 0
    ADL: float = 8.5 # Activities of Daily Living (0-10)
    Confusion: int = 0
    Disorientation: int = 0
    PersonalityChanges: int = 0
    DifficultyCompletingTasks: int = 0
    Forgetfulness: int = 0

    @field_validator("MMSE")
    @classmethod
    def validate_mmse(cls, v: float) -> float:
        if not (0 <= v <= 30):
            raise ValueError("MMSE must be between 0 and 30")
        return v

    @field_validator("ADL", "FunctionalAssessment")
    @classmethod
    def validate_zero_to_ten(cls, v: float) -> float:
        if not (0 <= v <= 10):
            raise ValueError("Value must be between 0 and 10")
        return v

    @field_validator("Age")
    @classmethod
    def validate_age(cls, v: int) -> int:
        if not (18 <= v <= 120):
            raise ValueError("Age must be between 18 and 120")
        return v

    @field_validator("BMI")
    @classmethod
    def validate_bmi(cls, v: float) -> float:
        if not (10.0 <= v <= 70.0):
            raise ValueError("BMI must be between 10 and 70")
        return v


class DiabetesInput(BaseModel):
    Pregnancies: int = 1
    Glucose: float = 110.0
    BloodPressure: float = 75.0
    SkinThickness: float = 22.0
    Age: int = 45

    @field_validator("Glucose")
    @classmethod
    def validate_glucose(cls, v: float) -> float:
        if not (0 < v < 600):
            raise ValueError("Glucose must be between 0 and 600 mg/dL")
        return v

    @field_validator("Age")
    @classmethod
    def validate_age(cls, v: int) -> int:
        if not (1 <= v <= 120):
            raise ValueError("Age must be between 1 and 120")
        return v

class CardioSurveillanceInput(BaseModel):
    Year: int = 2018
    Indicator: str = "Prevalence of major cardiovascular disease"
    Data_Value_Unit: str = "%"
    Data_Value_Alt: float = 8.5
    LowConfidenceLimit: float = 7.2
    HighConfidenceLimit: float = 9.8
    Break_Out_Category: str = "Overall"
    Break_Out: str = "Overall"
    PriorityArea: str = "Million Hearts"

class ClinicalCardioInput(BaseModel):
    Age: int = 55
    Gender: str = "Male"
    SystolicBP: int = 135
    TotalCholesterol: float = 220.0
    HDLCholesterol: float = 48.0
    Smoking: bool = False
    Diabetes: bool = False
    OnBPTreatment: bool = False

    @field_validator("Age")
    @classmethod
    def validate_age(cls, v: int) -> int:
        if not (18 <= v <= 120):
            raise ValueError("Age must be between 18 and 120")
        return v

    @field_validator("SystolicBP")
    @classmethod
    def validate_bp(cls, v: int) -> int:
        if not (60 <= v <= 300):
            raise ValueError("Systolic BP must be between 60 and 300 mmHg")
        return v

    @field_validator("Gender")
    @classmethod
    def validate_gender(cls, v: str) -> str:
        if v not in ("Male", "Female"):
            raise ValueError("Gender must be 'Male' or 'Female'")
        return v

class PredictionResponse(BaseModel):
    assessment_type: str
    prediction_class: int
    probability_percentage: float
    risk_level: str
    headline: str
    contributing_factors: Dict[str, Any]
    clinical_disclaimer: str
    recommendations: List[str]

# --- Lab Result & Report Schemas ---
class LabResultItem(BaseModel):
    test_name: str
    result_value: float
    unit: str
    reference_low: Optional[float]
    reference_high: Optional[float]
    status: str # NORMAL, HIGH, LOW
    clinical_explanation: str

class LabReportCreate(BaseModel):
    report_title: str
    report_type: str = "Blood Report"
    report_date: Optional[str] = None
    lab_name: str = "Apex Diagnostic Lab"
    results: List[LabResultItem]

# --- Medicine & Prescription Schemas ---
class MedicineItem(BaseModel):
    medicine_name: str
    dosage: str
    frequency: str
    duration: str = "30 days"
    instructions: str = ""
    purpose_explanation: str = ""

class PrescriptionCreate(BaseModel):
    doctor_name: str
    prescription_date: Optional[str] = None
    diagnosis: str
    notes: str = ""
    medicines: List[MedicineItem]

# --- Medicine Reminder Schemas ---
class ReminderCreate(BaseModel):
    medicine_name: str
    dosage: str
    scheduled_time: str
    meal_relation: str = "After meal"
    schedule_date: Optional[str] = None
    notes: str = ""

    @field_validator("medicine_name")
    @classmethod
    def validate_medicine_name(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Medicine name cannot be empty")
        return v.strip()

    @field_validator("scheduled_time")
    @classmethod
    def validate_scheduled_time(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Scheduled time cannot be empty")
        return v.strip()


class ReminderStatusUpdate(BaseModel):
    taken_status: bool

# --- Chatbot Schema ---
class ChatQuery(BaseModel):
    message: Optional[str] = None
    query: Optional[str] = None

class ChatResponse(BaseModel):
    response: str
    referenced_context: List[str]
    disclaimer: str
