import os
import uuid
from datetime import datetime, timezone
from typing import List, Optional

from fastapi import HTTPException
from google import genai
from sqlalchemy.orm import Session, selectinload

from ..models.chat import ChatMessage, ChatSession
from ..models.models import GoalDB, HealthProfileDB, UserDB, UserAllergyDB, DietaryPreferenceDB
from ..schemas.chat import ChatRequest, ChatResponse
from .health_service import HealthProfileService

class ChatService:
    def __init__(self, db: Session):
        self.db = db
        self.api_key = os.environ.get("GOOGLE_API_KEY")
        if not self.api_key:
            print("WARNING: GOOGLE_API_KEY not set")
        self.client = genai.Client(api_key=self.api_key)

    def _get_system_prompt(self, user_id: str) -> str:
        """Constructs a system prompt based on the user's profile."""
        
        # Fetch user profile data
        user = self.db.query(UserDB).filter(UserDB.id == user_id).first()
        if not user:
            return "You are a helpful nutrition assistant."

        health_profile = (
            self.db.query(HealthProfileDB)
            .options(
                selectinload(HealthProfileDB.allergies).selectinload(UserAllergyDB.allergen),
                selectinload(HealthProfileDB.dietary_preferences)
            )
            .filter(HealthProfileDB.user_id == user_id)
            .first()
        )
        
        goals = self.db.query(GoalDB).filter(GoalDB.user_id == user_id, GoalDB.status == "active").all()

        prompt_parts = [
            "You are a personalized AI Health Concierge for Eatsential.",
            f"User: {user.username}",
        ]

        if health_profile:
            prompt_parts.append("Health Profile:")
            if health_profile.height_cm:
                prompt_parts.append(f"- Height: {health_profile.height_cm} cm")
            if health_profile.weight_kg:
                prompt_parts.append(f"- Weight: {health_profile.weight_kg} kg")
            if health_profile.activity_level:
                prompt_parts.append(f"- Activity Level: {health_profile.activity_level}")
            
            if health_profile.allergies:
                allergies = [a.allergen.name for a in health_profile.allergies]
                prompt_parts.append(f"- Allergies: {', '.join(allergies)}")
            
            if health_profile.dietary_preferences:
                prefs = [f"{p.preference_type}: {p.preference_name}" for p in health_profile.dietary_preferences]
                prompt_parts.append(f"- Preferences: {', '.join(prefs)}")

        if goals:
            prompt_parts.append("Current Goals:")
            for goal in goals:
                prompt_parts.append(f"- {goal.goal_type}: {goal.target_type} {goal.target_value} (End: {goal.end_date})")

        prompt_parts.append("\nYour role is to answer nutrition questions and give real-time, personalized advice based on the user's profile and goals. Be encouraging, scientific but accessible, and practical.")
        
        return "\n".join(prompt_parts)

    async def process_message(self, user_id: str, request: ChatRequest) -> ChatResponse:
        """Processes a user message and returns the AI response."""
        
        # 1. Get or Create Session
        if request.session_id:
            session = self.db.query(ChatSession).filter(ChatSession.id == request.session_id, ChatSession.user_id == user_id).first()
            if not session:
                raise HTTPException(status_code=404, detail="Session not found")
        else:
            session = ChatSession(id=str(uuid.uuid4()), user_id=user_id)
            self.db.add(session)
            self.db.commit()
            self.db.refresh(session)

        # 2. Save User Message
        user_message = ChatMessage(
            id=str(uuid.uuid4()),
            session_id=session.id,
            role="user",
            content=request.message
        )
        self.db.add(user_message)
        
        # 3. Construct Context and Call AI
        system_prompt = self._get_system_prompt(user_id)
        
        # Fetch recent history for context (last 10 messages)
        history = (
            self.db.query(ChatMessage)
            .filter(ChatMessage.session_id == session.id)
            .order_by(ChatMessage.created_at.asc())
            .limit(10)
            .all()
        )
        
        try:
            # chat = self.client.chats.create(model="gemini-2.0-flash-exp") # Unused and causing 429

            
            history_messages = []
            
            full_prompt = system_prompt + "\n\nConversation History:\n"
            for msg in history:
                full_prompt += f"{msg.role.capitalize()}: {msg.content}\n"
            
            self.db.commit()
            
            history = (
                self.db.query(ChatMessage)
                .filter(ChatMessage.session_id == session.id)
                .order_by(ChatMessage.created_at.asc())
                .limit(10) # Last 10
                .all()
            )
             
            full_prompt = system_prompt + "\n\nConversation History:\n"
            for msg in history: # This includes the current message
                 full_prompt += f"{msg.role.capitalize()}: {msg.content}\n"
            
            full_prompt += "\nAssistant:"

            response = self.client.models.generate_content(
                model="gemini-flash-latest",
                contents=full_prompt
            )
            
            ai_response_text = response.text
            
        except Exception as e:
            print(f"Error calling GenAI: {e}")
            ai_response_text = "I'm sorry, I'm having trouble connecting to my brain right now. Please try again later."

        # 4. Save AI Response
        ai_message = ChatMessage(
            id=str(uuid.uuid4()),
            session_id=session.id,
            role="model",
            content=ai_response_text
        )
        self.db.add(ai_message)
        
        # Update session timestamp
        session.updated_at = datetime.now(timezone.utc).replace(tzinfo=None)
        
        self.db.commit()
        
        return ChatResponse(response=ai_response_text, session_id=session.id)

    def get_user_sessions(self, user_id: str) -> List[ChatSession]:
        return (
            self.db.query(ChatSession)
            .filter(ChatSession.user_id == user_id)
            .order_by(ChatSession.updated_at.desc())
            .all()
        )

    def get_session_history(self, user_id: str, session_id: str) -> Optional[ChatSession]:
        return (
            self.db.query(ChatSession)
            .options(selectinload(ChatSession.messages))
            .filter(ChatSession.id == session_id, ChatSession.user_id == user_id)
            .first()
        )
