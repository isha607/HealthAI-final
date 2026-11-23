"""
Timeline API — returns a merged chronological timeline of all patient events.
"""
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from backend.database import get_db
from backend.services.timeline_service import get_patient_timeline

router = APIRouter(prefix="/api/timeline", tags=["Timeline"])


@router.get("/{patient_id}")
def patient_timeline(patient_id: str, db: Session = Depends(get_db)):
    """
    Return a unified longitudinal timeline for the given patient_id.
    Events include doctor visits, lab reports, AI assessments, prescriptions,
    and smartwatch snapshots sorted in reverse-chronological order.
    """
    try:
        timeline = get_patient_timeline(patient_id, db)
        return timeline
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
