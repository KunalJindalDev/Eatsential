import pytest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from src.eatsential.models import UserDB, AccountStatus
from src.eatsential.services.chat import ChatService
from src.eatsential.models.chat import ChatSession, ChatMessage

# Mock the GenAI Client
@pytest.fixture
def mock_genai_client():
    with patch("src.eatsential.services.chat.genai.Client") as MockClient:
        mock_instance = MockClient.return_value
        # Mock the generate_content method
        mock_response = MagicMock()
        mock_response.text = "This is a mock AI response."
        mock_instance.models.generate_content.return_value = mock_response
        yield mock_instance

@pytest.fixture
def authenticated_user(db: Session, client: TestClient):
    """Create a user and return authentication headers"""
    # Create user
    user = UserDB(
        id="test_user_id",
        email="chat_test@example.com",
        username="chattest",
        password_hash="hashed_password",
        account_status=AccountStatus.VERIFIED,
        email_verified=True
    )
    db.add(user)
    db.commit()
    return user

def test_chat_flow(client: TestClient, db: Session, authenticated_user, mock_genai_client):
    """Test the full chat flow: send message, get response, check history"""
    
    # Override get_current_user to return our test user
    from src.eatsential.services.auth_service import get_current_user
    app = client.app
    app.dependency_overrides[get_current_user] = lambda: authenticated_user

    # 1. Send a new message
    response = client.post(
        "/api/chat/",
        json={"message": "Hello AI"}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["response"] == "This is a mock AI response."
    assert "session_id" in data
    session_id = data["session_id"]

    # 2. Verify session and messages in DB
    session = db.query(ChatSession).filter(ChatSession.id == session_id).first()
    assert session is not None
    assert session.user_id == authenticated_user.id
    
    messages = db.query(ChatMessage).filter(ChatMessage.session_id == session_id).all()
    assert len(messages) == 2 # User message + AI response
    assert messages[0].role == "user"
    assert messages[0].content == "Hello AI"
    assert messages[1].role == "model"
    assert messages[1].content == "This is a mock AI response."

    # 3. Send another message in the same session
    response = client.post(
        "/api/chat/",
        json={"message": "Follow up question", "session_id": session_id}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["session_id"] == session_id # Should be same session

    # 4. Get Sessions List
    response = client.get("/api/chat/sessions")
    assert response.status_code == 200
    sessions = response.json()
    assert len(sessions) == 1
    assert sessions[0]["id"] == session_id

    # 5. Get Specific Session History
    response = client.get(f"/api/chat/sessions/{session_id}")
    assert response.status_code == 200
    history = response.json()
    assert history["id"] == session_id
    assert len(history["messages"]) == 4 # 2 exchanges * 2 messages each

    # Clean up dependency override
    app.dependency_overrides = {}

def test_chat_ai_failure(client: TestClient, db: Session, authenticated_user):
    """Test that the system handles AI failures gracefully"""
    
    # 1. Override the AI client to raise an Error instead of returning text
    with patch("src.eatsential.services.chat.genai.Client") as MockClient:
        mock_instance = MockClient.return_value
        # Make the generate_content method crash
        mock_instance.models.generate_content.side_effect = Exception("Google is down")
        
        # 2. Setup Auth Override
        from src.eatsential.services.auth_service import get_current_user
        app = client.app
        app.dependency_overrides[get_current_user] = lambda: authenticated_user

        # 3. Send message
        response = client.post("/api/chat/", json={"message": "Hello?"})
        
        # 4. Verify we get a 200 OK (not 500 Crash) and the fallback message
        assert response.status_code == 200
        data = response.json()
        assert "I'm sorry" in data["response"]
        assert "trouble connecting to my brain" in data["response"]
        
        # Clean up
        app.dependency_overrides = {}

def test_access_other_user_session(client: TestClient, db: Session, authenticated_user):
    """Test that a user cannot access another user's chat session"""
    
    # 1. Create a session for a DIFFERENT user
    other_user_id = "hacker_target_id"
    other_session = ChatSession(id="secret_session_123", user_id=other_user_id)
    db.add(other_session)
    db.commit()

    # 2. Setup Auth Override (We are logged in as 'authenticated_user', NOT 'hacker_target_id')
    from src.eatsential.services.auth_service import get_current_user
    app = client.app
    app.dependency_overrides[get_current_user] = lambda: authenticated_user

    # 3. Try to get that session history
    response = client.get(f"/api/chat/sessions/{other_session.id}")
    
    # 4. Should be 404 Not Found (or 403 Forbidden depending on your logic)
    assert response.status_code == 404
    assert response.json()["detail"] == "Session not found"

    # Clean up
    app.dependency_overrides = {}

def test_chat_invalid_session_id(client: TestClient, authenticated_user):
    """Test sending a message with a non-existent session ID"""
    
    # Setup Auth
    from src.eatsential.services.auth_service import get_current_user
    app = client.app
    app.dependency_overrides[get_current_user] = lambda: authenticated_user

    # Try to reply to a session that doesn't exist
    response = client.post(
        "/api/chat/",
        json={
            "message": "I'm replying to nothing", 
            "session_id": "fake-uuid-12345"
        }
    )

    # Should return 404 Not Found
    assert response.status_code == 404
    
    # Clean up
    app.dependency_overrides = {}