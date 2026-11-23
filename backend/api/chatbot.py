from fastapi import APIRouter, Depends, Request, HTTPException
from sqlalchemy.orm import Session
import jwt
from backend.database import get_db
from backend.models.schemas import ChatQuery, ChatResponse
from backend.services.chatbot_service import handle_chatbot_query
from backend.api.auth import SECRET_KEY, ALGORITHM

router = APIRouter(prefix="/api/chatbot", tags=["AI Health Chatbot"])

def get_authenticated_patient_id(request: Request) -> str:
    auth_header = request.headers.get("Authorization")
    if auth_header and auth_header.startswith("Bearer "):
        token = auth_header.split(" ")[1]
        try:
            payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
            return payload.get("identifier")
        except Exception:
            pass
    return None

@router.post("/{patient_id}/query", response_model=ChatResponse)
def query_chatbot(patient_id: str, req: ChatQuery, request: Request, db: Session = Depends(get_db)):
    auth_patient_id = get_authenticated_patient_id(request)
    
    # Enforce patient data isolation: If token is provided, ensure patient cannot query another patient's records
    if auth_patient_id and auth_patient_id.startswith("PAT-") and auth_patient_id != patient_id:
        raise HTTPException(
            status_code=403, 
            detail="Access forbidden: You cannot access or query another patient's medical records."
        )

    # Scoped strictly to authenticated patient
    effective_patient_id = auth_patient_id if (auth_patient_id and auth_patient_id.startswith("PAT-")) else patient_id
    msg = req.message or req.query or ""
    
    try:
        return handle_chatbot_query(effective_patient_id, msg, db)
    except Exception as e:
        import logging
        logging.getLogger("healthai.chatbot").error(f"Chatbot query error for {effective_patient_id}: {e}", exc_info=True)
        return {
            "response": "I couldn't process that health record inquiry right now. Please verify your query or try again.",
            "referenced_context": ["Internal Error Handler"],
            "disclaimer": "Automated healthcare assistant error recovery."
        }
