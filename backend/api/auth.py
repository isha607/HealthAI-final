from fastapi import APIRouter, Depends, HTTPException, status, Header
from sqlalchemy.orm import Session
from pydantic import BaseModel, EmailStr
from typing import Optional
from datetime import datetime, timedelta
import jwt
import bcrypt

from backend.database import get_db
from backend.models.db_models import User, PatientProfile, DoctorProfile, DoctorPatientRelationship
from backend.models.schemas import UserRegister, UserLogin, TokenResponse

router = APIRouter(prefix="/api/auth", tags=["Authentication"])

SECRET_KEY = "health-ai-app-super-secret-key-for-jwt-tokens"
ALGORITHM = "HS256"

def verify_password(plain_password: str, hashed_password: str) -> bool:
    try:
        if isinstance(hashed_password, str):
            hashed_password = hashed_password.encode("utf-8")
        return bcrypt.checkpw(plain_password[:72].encode("utf-8"), hashed_password)
    except Exception:
        return False

def get_password_hash(password: str) -> str:
    salt = bcrypt.gensalt()
    return bcrypt.hashpw(password[:72].encode("utf-8"), salt).decode("utf-8")

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(days=7))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

def get_current_user_payload(authorization: Optional[str] = Header(None)) -> Optional[dict]:
    """Decodes JWT payload from Authorization header if present."""
    if not authorization or not authorization.startswith("Bearer "):
        return None
    token = authorization.split(" ", 1)[1]
    try:
        return jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
    except Exception:
        return None

def verify_patient_access(patient_id: str, authorization: Optional[str] = Header(None)):
    """
    Enforces patient privacy and cross-patient isolation.
    If a Patient JWT token is supplied, ensures the patient can only access their own records.
    If an unauthorized patient tries to access another patient's data, raises 403 Forbidden.
    """
    payload = get_current_user_payload(authorization)
    if not payload:
        return  # demo / local test without header
    role = payload.get("role")
    identifier = payload.get("identifier")
    if role == "patient" and identifier and identifier != patient_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Access denied: You are not authorized to access medical records for patient {patient_id}."
        )

@router.post("/register", response_model=TokenResponse)
def register(req: UserRegister, db: Session = Depends(get_db)):
    existing = db.query(User).filter(User.email == req.email).first()
    if existing:
        raise HTTPException(status_code=400, detail="Email already registered")

    user = User(
        email=req.email,
        hashed_password=get_password_hash(req.password),
        role=req.role.lower()
    )
    db.add(user)
    db.commit()
    db.refresh(user)

    identifier = ""
    if user.role == "patient":
        # Create Patient Profile with unique Patient ID
        patient_count = db.query(PatientProfile).count()
        identifier = f"PAT-{1000 + patient_count + 1}"
        profile = PatientProfile(
            patient_id=identifier,
            user_id=user.id,
            full_name=req.full_name,
            age=req.age or 45,
            gender=req.gender or "Female"
        )
        db.add(profile)
        db.commit()
    else:
        # Create Doctor Profile
        doc_count = db.query(DoctorProfile).count()
        identifier = f"DOC-{500 + doc_count + 1}"
        doc = DoctorProfile(
            doctor_code=identifier,
            user_id=user.id,
            full_name=req.full_name,
            specialization="General Medicine & Chronic Care"
        )
        db.add(doc)
        db.commit()

    token = create_access_token({"sub": str(user.id), "role": user.role, "identifier": identifier})
    return TokenResponse(
        access_token=token,
        user_id=user.id,
        role=user.role,
        identifier=identifier,
        full_name=req.full_name
    )

@router.post("/login", response_model=TokenResponse)
def login(req: UserLogin, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == req.email).first()
    if not user or not verify_password(req.password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Invalid email or password")

    identifier = ""
    full_name = "User"
    if user.role == "patient" and user.patient_profile:
        identifier = user.patient_profile.patient_id
        full_name = user.patient_profile.full_name
    elif user.role == "doctor" and user.doctor_profile:
        identifier = user.doctor_profile.doctor_code
        full_name = user.doctor_profile.full_name

    token = create_access_token({"sub": str(user.id), "role": user.role, "identifier": identifier})
    return TokenResponse(
        access_token=token,
        user_id=user.id,
        role=user.role,
        identifier=identifier,
        full_name=full_name
    )


class DoctorAccessRequest(BaseModel):
    doctor_code: str


def _require_patient(authorization: Optional[str]) -> str:
    payload = get_current_user_payload(authorization)
    if not payload or payload.get("role") != "patient" or not payload.get("identifier"):
        raise HTTPException(status_code=401, detail="Patient login required")
    return payload["identifier"]


@router.post("/grant-doctor-access")
def grant_doctor_access(req: DoctorAccessRequest, authorization: Optional[str] = Header(None),
                        db: Session = Depends(get_db)):
    patient_id = _require_patient(authorization)
    doc = db.query(DoctorProfile).filter(
        DoctorProfile.doctor_code == req.doctor_code.strip().upper()).first()
    if not doc:
        raise HTTPException(status_code=404, detail="No doctor found with that code")
    exists = db.query(DoctorPatientRelationship).filter(
        DoctorPatientRelationship.doctor_id == doc.id,
        DoctorPatientRelationship.patient_id == patient_id).first()
    if not exists:
        db.add(DoctorPatientRelationship(doctor_id=doc.id, patient_id=patient_id))
        db.commit()
    return {"message": f"Access granted to Dr. {doc.full_name}"}


@router.get("/my-doctors")
def my_doctors(authorization: Optional[str] = Header(None), db: Session = Depends(get_db)):
    patient_id = _require_patient(authorization)
    rels = db.query(DoctorPatientRelationship).filter(
        DoctorPatientRelationship.patient_id == patient_id).all()
    out = []
    for r in rels:
        d = db.query(DoctorProfile).filter(DoctorProfile.id == r.doctor_id).first()
        if d:
            out.append({"doctor_code": d.doctor_code, "full_name": d.full_name})
    return out
