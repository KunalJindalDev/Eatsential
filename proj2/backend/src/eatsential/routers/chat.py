from typing import List

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from ..db.database import get_db
from ..models.models import UserDB
from ..schemas.chat import ChatRequest, ChatResponse, ChatSessionResponse
from ..services.chat import ChatService
from ..services.auth_service import get_current_user

router = APIRouter(
    prefix="/chat",
    tags=["chat"],
    responses={404: {"description": "Not found"}},
)

@router.post("/", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    current_user: UserDB = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Send a message to the AI Health Concierge.
    """
    service = ChatService(db)
    return await service.process_message(current_user.id, request)

@router.get("/sessions", response_model=List[ChatSessionResponse])
def get_sessions(
    current_user: UserDB = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Get all chat sessions for the current user.
    """
    service = ChatService(db)
    return service.get_user_sessions(current_user.id)

@router.get("/sessions/{session_id}", response_model=ChatSessionResponse)
def get_session(
    session_id: str,
    current_user: UserDB = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Get a specific chat session with history.
    """
    service = ChatService(db)
    session = service.get_session_history(current_user.id, session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return session
