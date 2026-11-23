from sqlalchemy import Column, Integer, String, Float, Boolean, Text, DateTime, ForeignKey
from sqlalchemy.orm import relationship
from datetime import datetime
from backend.database import Base

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String(100), unique=True, index=True, nullable=False)
    hashed_password = Column(String(200), nullable=False)
    role = Column(String(20), nullable=False)  # 'patient' or 'doctor'
    created_at = Column(DateTime, default=datetime.utcnow)

    patient_profile = relationship("PatientProfile", back_populates="user", uselist=False)
    doctor_profile = relationship("DoctorProfile", back_populates="user", uselist=False)

class PatientProfile(Base):
    __tablename__ = "patient_profiles"

    patient_id = Column(String(50), primary_key=True, index=True) # e.g. PAT-1001
    user_id = Column(Integer, ForeignKey("users.id"), unique=True, nullable=False)
    full_name = Column(String(100), nullable=False)
    age = Column(Integer, nullable=False)
    gender = Column(String(20), nullable=False)
    height_cm = Column(Float, default=170.0)
    weight_kg = Column(Float, default=70.0)
    bmi = Column(Float, default=24.2)
    phone = Column(String(20), default="")
    blood_group = Column(String(10), default="O+")
    smoking_status = Column(String(20), default="No")
    alcohol_consumption = Column(Float, default=0.0)
    physical_activity_hours = Column(Float, default=2.5)
    diet_quality_score = Column(Float, default=6.5)
    sleep_hours = Column(Float, default=7.0)
    family_history_alzheimers = Column(Boolean, default=False)
    cardiovascular_disease = Column(Boolean, default=False)
    diabetes_history = Column(Boolean, default=False)
    hypertension = Column(Boolean, default=False)
    allergies = Column(Text, default="None reported")
    existing_conditions = Column(Text, default="None")
    created_at = Column(DateTime, default=datetime.utcnow)

    user = relationship("User", back_populates="patient_profile")
    visits = relationship("DoctorVisit", back_populates="patient")
    lab_reports = relationship("LabReport", back_populates="patient")
    prescriptions = relationship("Prescription", back_populates="patient")
    medicine_schedules = relationship("MedicineSchedule", back_populates="patient")
    assessments = relationship("MLAssessment", back_populates="patient")
    chat_messages = relationship("ChatMessage", back_populates="patient")

class DoctorProfile(Base):
    __tablename__ = "doctors"

    id = Column(Integer, primary_key=True, index=True)
    doctor_code = Column(String(50), unique=True, index=True) # e.g. DOC-501
    user_id = Column(Integer, ForeignKey("users.id"), unique=True, nullable=False)
    full_name = Column(String(100), nullable=False)
    specialization = Column(String(100), default="General Medicine & Cardiology")
    hospital_clinic = Column(String(150), default="Central Health Institute")
    phone = Column(String(20), default="")

    user = relationship("User", back_populates="doctor_profile")
    visits = relationship("DoctorVisit", back_populates="doctor")

class DoctorPatientRelationship(Base):
    __tablename__ = "doctor_patient_relationships"

    id = Column(Integer, primary_key=True, index=True)
    doctor_id = Column(Integer, ForeignKey("doctors.id"), nullable=False)
    patient_id = Column(String(50), ForeignKey("patient_profiles.patient_id"), nullable=False)
    assigned_date = Column(DateTime, default=datetime.utcnow)

class DoctorVisit(Base):
    __tablename__ = "doctor_visits"

    id = Column(Integer, primary_key=True, index=True)
    patient_id = Column(String(50), ForeignKey("patient_profiles.patient_id"), nullable=False)
    doctor_id = Column(Integer, ForeignKey("doctors.id"), nullable=True)
    doctor_name = Column(String(100), default="Dr. Sarah Jenkins, MD")
    visit_date = Column(DateTime, default=datetime.utcnow)
    chief_complaint = Column(String(255), nullable=False)
    clinical_notes = Column(Text, nullable=False)
    diagnosis = Column(String(255), nullable=False)
    vitals_bp = Column(String(30), default="120/80")
    vitals_heart_rate = Column(Integer, default=72)
    vitals_glucose = Column(Float, default=95.0)
    follow_up_date = Column(String(50), default="")

    patient = relationship("PatientProfile", back_populates="visits")
    doctor = relationship("DoctorProfile", back_populates="visits")

class LabReport(Base):
    __tablename__ = "lab_reports"

    id = Column(Integer, primary_key=True, index=True)
    patient_id = Column(String(50), ForeignKey("patient_profiles.patient_id"), nullable=False)
    report_title = Column(String(150), nullable=False)
    report_type = Column(String(50), default="Blood Report")
    report_date = Column(DateTime, default=datetime.utcnow)
    lab_name = Column(String(150), default="Apex Clinical Diagnostic Lab")
    file_path = Column(String(255), default="")
    raw_text = Column(Text, default="")
    created_at = Column(DateTime, default=datetime.utcnow)

    patient = relationship("PatientProfile", back_populates="lab_reports")
    results = relationship("LabResult", back_populates="report", cascade="all, delete-orphan")

class LabResult(Base):
    __tablename__ = "lab_results"

    id = Column(Integer, primary_key=True, index=True)
    report_id = Column(Integer, ForeignKey("lab_reports.id"), nullable=False)
    patient_id = Column(String(50), nullable=False)
    test_name = Column(String(100), nullable=False)
    result_value = Column(Float, nullable=False)
    unit = Column(String(30), default="")
    reference_low = Column(Float, nullable=True)
    reference_high = Column(Float, nullable=True)
    status = Column(String(20), nullable=False) # 'NORMAL', 'HIGH', 'LOW'
    clinical_explanation = Column(Text, default="")

    report = relationship("LabReport", back_populates="results")

class Prescription(Base):
    __tablename__ = "prescriptions"

    id = Column(Integer, primary_key=True, index=True)
    patient_id = Column(String(50), ForeignKey("patient_profiles.patient_id"), nullable=False)
    doctor_name = Column(String(100), default="Dr. Sarah Jenkins, MD")
    prescription_date = Column(DateTime, default=datetime.utcnow)
    diagnosis = Column(String(255), default="")
    notes = Column(Text, default="")
    tests_advised = Column(Text, default="")
    follow_up = Column(String(100), default="")
    file_path = Column(String(255), default="")
    raw_text = Column(Text, default="")

    patient = relationship("PatientProfile", back_populates="prescriptions")
    medicines = relationship("Medicine", back_populates="prescription", cascade="all, delete-orphan")

class Medicine(Base):
    __tablename__ = "medicines"

    id = Column(Integer, primary_key=True, index=True)
    prescription_id = Column(Integer, ForeignKey("prescriptions.id"), nullable=False)
    patient_id = Column(String(50), nullable=False)
    medicine_name = Column(String(100), nullable=False)
    dosage = Column(String(50), nullable=False) # e.g. "500 mg"
    frequency = Column(String(50), nullable=False) # e.g. "Twice daily after meals"
    timing = Column(String(50), default="") # e.g. "Morning, Night"
    meal_instruction = Column(String(50), default="") # e.g. "After meal"
    duration = Column(String(50), default="30 days")
    instructions = Column(Text, default="Take with full glass of water")
    purpose_explanation = Column(Text, default="")

    prescription = relationship("Prescription", back_populates="medicines")

class MedicineSchedule(Base):
    __tablename__ = "medicine_schedules"

    id = Column(Integer, primary_key=True, index=True)
    patient_id = Column(String(50), ForeignKey("patient_profiles.patient_id"), nullable=False)
    prescription_id = Column(Integer, ForeignKey("prescriptions.id"), nullable=True)
    medicine_name = Column(String(100), nullable=False)
    dosage = Column(String(50), nullable=False)
    frequency = Column(String(50), default="")
    scheduled_time = Column(String(20), nullable=False) # e.g. "08:00 AM"
    meal_relation = Column(String(30), default="After meal")
    schedule_date = Column(String(30), default="") # YYYY-MM-DD
    taken_status = Column(Boolean, default=False)
    notes = Column(String(255), default="")
    reminder_type = Column(String(30), default="medicine") # 'medicine', 'follow_up', 'test'

    patient = relationship("PatientProfile", back_populates="medicine_schedules")


class MLAssessment(Base):
    __tablename__ = "ml_assessments"

    id = Column(Integer, primary_key=True, index=True)
    patient_id = Column(String(50), ForeignKey("patient_profiles.patient_id"), nullable=False)
    assessment_type = Column(String(50), nullable=False) # 'alzheimer', 'diabetes', 'cardiovascular'
    model_version = Column(String(20), default="1.0.0")
    input_payload = Column(Text, nullable=False) # JSON string of inputs
    prediction_class = Column(Integer, nullable=False)
    probability = Column(Float, nullable=False)
    risk_level = Column(String(30), nullable=False) # 'Low Risk', 'Moderate Risk', 'High Risk'
    contributing_factors = Column(Text, default="{}") # JSON string
    clinical_disclaimer = Column(Text, nullable=False)
    source_type = Column(String(30), default="manual") # 'manual' or 'uploaded_document'
    created_at = Column(DateTime, default=datetime.utcnow)

    patient = relationship("PatientProfile", back_populates="assessments")

class ChatMessage(Base):
    __tablename__ = "chat_messages"

    id = Column(Integer, primary_key=True, index=True)
    patient_id = Column(String(50), ForeignKey("patient_profiles.patient_id"), nullable=False)
    sender = Column(String(20), nullable=False) # 'user' or 'assistant'
    message = Column(Text, nullable=False)
    referenced_records = Column(Text, default="") # JSON or metadata
    timestamp = Column(DateTime, default=datetime.utcnow)

    patient = relationship("PatientProfile", back_populates="chat_messages")
