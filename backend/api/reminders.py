from fastapi import APIRouter, Depends, HTTPException, Header
from sqlalchemy.orm import Session
from typing import List, Dict, Any, Optional
from backend.database import get_db
from backend.models.db_models import MedicineSchedule
from backend.models.schemas import ReminderCreate, ReminderStatusUpdate
from backend.api.auth import verify_patient_access
from datetime import datetime, date

router = APIRouter(prefix="/api/reminders", tags=["Medicine Reminders"])

@router.get("/{patient_id}")
def get_reminders(patient_id: str, date_filter: Optional[str] = None, authorization: Optional[str] = Header(None), db: Session = Depends(get_db)):
    verify_patient_access(patient_id, authorization)
    query = db.query(MedicineSchedule).filter(MedicineSchedule.patient_id == patient_id)
    if date_filter:
        query = query.filter(MedicineSchedule.schedule_date == date_filter)
    items = query.order_by(MedicineSchedule.schedule_date.desc(), MedicineSchedule.scheduled_time).all()
    
    today_str = date.today().isoformat()
    
    result = []
    for r in items:
        # Compute overdue: pending + today + time is in the past
        is_overdue = False
        if (not r.taken_status and r.schedule_date == today_str):
            try:
                rem_time = datetime.strptime(r.scheduled_time, "%I:%M %p")
                if rem_time.time() < datetime.now().time():
                    is_overdue = True
            except:
                pass
        
        result.append({
            "id": r.id,
            "patient_id": r.patient_id,
            "prescription_id": r.prescription_id if hasattr(r, 'prescription_id') else None,
            "medicine_name": r.medicine_name,
            "dosage": r.dosage,
            "frequency": r.frequency if hasattr(r, 'frequency') else "",
            "scheduled_time": r.scheduled_time,
            "meal_relation": r.meal_relation,
            "schedule_date": r.schedule_date,
            "taken_status": r.taken_status,
            "notes": r.notes,
            "reminder_type": r.reminder_type if hasattr(r, 'reminder_type') else "medicine",
            "is_overdue": is_overdue
        })
    return result

@router.post("/{patient_id}")
def create_reminder(patient_id: str, req: ReminderCreate, authorization: Optional[str] = Header(None), db: Session = Depends(get_db)):
    verify_patient_access(patient_id, authorization)
    
    # Check for existing duplicate reminder
    sched_date = req.schedule_date or date.today().isoformat()
    existing = db.query(MedicineSchedule).filter(
        MedicineSchedule.patient_id == patient_id,
        MedicineSchedule.medicine_name == req.medicine_name,
        MedicineSchedule.scheduled_time == req.scheduled_time,
        MedicineSchedule.schedule_date == sched_date
    ).first()
    if existing:
        return existing

    rem = MedicineSchedule(
        patient_id=patient_id,
        medicine_name=req.medicine_name,
        dosage=req.dosage,
        scheduled_time=req.scheduled_time,
        meal_relation=req.meal_relation,
        schedule_date=sched_date,
        taken_status=False,
        notes=req.notes
    )
    db.add(rem)
    db.commit()
    db.refresh(rem)
    return rem

@router.patch("/{reminder_id}/status")
def update_reminder_status(reminder_id: int, req: ReminderStatusUpdate, authorization: Optional[str] = Header(None), db: Session = Depends(get_db)):
    rem = db.query(MedicineSchedule).filter(MedicineSchedule.id == reminder_id).first()
    if not rem:
        raise HTTPException(status_code=404, detail="Reminder not found")
    verify_patient_access(rem.patient_id, authorization)
    rem.taken_status = req.taken_status
    db.commit()
    db.refresh(rem)
    return rem

@router.delete("/{reminder_id}")
def delete_reminder(reminder_id: int, authorization: Optional[str] = Header(None), db: Session = Depends(get_db)):
    rem = db.query(MedicineSchedule).filter(MedicineSchedule.id == reminder_id).first()
    if not rem:
        raise HTTPException(status_code=404, detail="Reminder not found")
    verify_patient_access(rem.patient_id, authorization)
    db.delete(rem)
    db.commit()
    return {"message": "Reminder deleted successfully"}
