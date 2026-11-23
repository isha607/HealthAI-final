from sqlalchemy import (
    Column, Integer, String, Float, Boolean, Text, DateTime, ForeignKey
)
from sqlalchemy.orm import relationship
from datetime import datetime
from backend.database.db import Base

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String(120), unique=True, index=True, nullable=False)
    password_hash = Column(String(255), nullable=False)
    role = Column(String(20), nullable=False, default="patient") # 'patient' or 'doctor'
    created_at = Column(DateTime, default=datetime.utcnow)

    patient_profile = relationship("PatientProfile", back_populates="user", uselist=False)
    doctor_profile = relationship("DoctorProfile", back_populates="user", uselist=False)


class PatientProfile(Base):
    __tablename__ = "patients"

    patient_id = Column(String(50), primary_key=True, index=True) # e.g. PAT-1001
    user_id = Column(Integer, ForeignKey("users.id"), unique=True, nullable=False)
    full_name = Column(String(120), nullable=False)
    age = Column(Integer, nullable=False)
    gender = Column(String(20), nullable=False)
    height_cm = Column(Float, default=170.0)
    weight_kg = Column(Float, default=70.0)
    bmi = Column(Float, default=24.2)
    phone = Column(String(30), default="")
    blood_group = Column(String(10), default="O+")
    allergies = Column(Text, default="None reported")
    existing_conditions = Column(Text, default="None")
    created_at = Column(DateTime, default=datetime.utcnow)

    user = relationship("User", back_populates="patient_profile")
    lab_reports = relationship("LabReport", back_populates="patient", cascade="all, delete-orphan")
    lab_results = relationship("LabResult", back_populates="patient", cascade="all, delete-orphan")
    prescriptions = relationship("Prescription", back_populates="patient", cascade="all, delete-orphan")
    medicines = relationship("Medicine", back_populates="patient", cascade="all, delete-orphan")
    reminders = relationship("Reminder", back_populates="patient", cascade="all, delete-orphan")
    ml_assessments = relationship("MLAssessment", back_populates="patient", cascade="all, delete-orphan")
    medical_events = relationship("MedicalEvent", back_populates="patient", cascade="all, delete-orphan")


class DoctorProfile(Base):
    __tablename__ = "doctors"

    doctor_id = Column(String(50), primary_key=True, index=True) # e.g. DOC-501
    user_id = Column(Integer, ForeignKey("users.id"), unique=True, nullable=False)
    full_name = Column(String(120), nullable=False)
    specialization = Column(String(120), default="Internal Medicine & Cardiology")
    hospital_name = Column(String(150), default="Metro Health Hospital")
    phone = Column(String(30), default="")
    created_at = Column(DateTime, default=datetime.utcnow)

    user = relationship("User", back_populates="doctor_profile")


class LabReport(Base):
    __tablename__ = "lab_reports"

    id = Column(Integer, primary_key=True, index=True)
    patient_id = Column(String(50), ForeignKey("patients.patient_id"), nullable=False, index=True)
    report_title = Column(String(200), nullable=False)
    lab_name = Column(String(150), default="")
    report_date = Column(DateTime, default=datetime.utcnow)
    file_path = Column(String(500), default="")
    raw_text = Column(Text, default="")
    created_at = Column(DateTime, default=datetime.utcnow)

    patient = relationship("PatientProfile", back_populates="lab_reports")
    results = relationship("LabResult", back_populates="report", cascade="all, delete-orphan")


class LabResult(Base):
    __tablename__ = "lab_results"

    id = Column(Integer, primary_key=True, index=True)
    report_id = Column(Integer, ForeignKey("lab_reports.id"), nullable=False, index=True)
    patient_id = Column(String(50), ForeignKey("patients.patient_id"), nullable=False, index=True)
    test_name = Column(String(150), nullable=False)
    result_value = Column(Float, nullable=False)
    unit = Column(String(50), default="")
    reference_low = Column(Float, nullable=True)
    reference_high = Column(Float, nullable=True)
    status = Column(String(30), default="NORMAL") # 'HIGH', 'LOW', 'NORMAL', 'NOT AVAILABLE'
    clinical_explanation = Column(Text, default="")
    created_at = Column(DateTime, default=datetime.utcnow)

    report = relationship("LabReport", back_populates="results")
    patient = relationship("PatientProfile", back_populates="lab_results")


class Prescription(Base):
    __tablename__ = "prescriptions"

    id = Column(Integer, primary_key=True, index=True)
    patient_id = Column(String(50), ForeignKey("patients.patient_id"), nullable=False, index=True)
    doctor_name = Column(String(150), nullable=False)
    prescription_date = Column(DateTime, default=datetime.utcnow)
    file_path = Column(String(500), default="")
    raw_text = Column(Text, default="")
    notes = Column(Text, default="")
    created_at = Column(DateTime, default=datetime.utcnow)

    patient = relationship("PatientProfile", back_populates="prescriptions")
    medicines = relationship("Medicine", back_populates="prescription", cascade="all, delete-orphan")


class Medicine(Base):
    __tablename__ = "medicines"

    id = Column(Integer, primary_key=True, index=True)
    prescription_id = Column(Integer, ForeignKey("prescriptions.id"), nullable=True)
    patient_id = Column(String(50), ForeignKey("patients.patient_id"), nullable=False, index=True)
    medicine_name = Column(String(150), nullable=False)
    dosage = Column(String(80), nullable=False)
    frequency = Column(String(80), default="Once daily")
    timing = Column(String(100), default="After breakfast")
    instructions = Column(Text, default="")
    start_date = Column(String(50), default="")
    end_date = Column(String(50), default="")
    prescribed_by = Column(String(150), default="Attending Physician")
    created_at = Column(DateTime, default=datetime.utcnow)

    prescription = relationship("Prescription", back_populates="medicines")
    patient = relationship("PatientProfile", back_populates="medicines")


class Reminder(Base):
    __tablename__ = "reminders"

    id = Column(Integer, primary_key=True, index=True)
    patient_id = Column(String(50), ForeignKey("patients.patient_id"), nullable=False, index=True)
    title = Column(String(200), nullable=False)
    reminder_type = Column(String(50), default="medicine") # 'medicine', 'appointment', 'followup', 'test', 'routine'
    date_time = Column(String(80), nullable=False)
    frequency = Column(String(80), default="Daily")
    notes = Column(Text, default="")
    is_completed = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    patient = relationship("PatientProfile", back_populates="reminders")


class MLAssessment(Base):
    __tablename__ = "ml_assessments"

    id = Column(Integer, primary_key=True, index=True)
    patient_id = Column(String(50), ForeignKey("patients.patient_id"), nullable=False, index=True)
    model_type = Column(String(50), nullable=False) # 'alzheimer', 'diabetes', 'cardiovascular'
    input_method = Column(String(30), default="manual") # 'manual' or 'report'
    input_data_json = Column(Text, nullable=False)
    prediction_class = Column(Integer, nullable=False)
    probability = Column(Float, nullable=True)
    risk_level = Column(String(80), nullable=False)
    contributing_factors = Column(Text, default="{}")
    clinical_disclaimer = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    patient = relationship("PatientProfile", back_populates="ml_assessments")


class MedicalEvent(Base):
    __tablename__ = "medical_events"

    id = Column(Integer, primary_key=True, index=True)
    patient_id = Column(String(50), ForeignKey("patients.patient_id"), nullable=False, index=True)
    event_date = Column(DateTime, default=datetime.utcnow)
    event_type = Column(String(50), nullable=False) # 'registration', 'visit', 'report', 'prescription', 'assessment'
    title = Column(String(200), nullable=False)
    description = Column(Text, default="")
    source = Column(String(100), default="System")
    created_at = Column(DateTime, default=datetime.utcnow)

    patient = relationship("PatientProfile", back_populates="medical_events")
