"""User service containing user-related business logic."""

import json
import uuid
from datetime import datetime, timedelta, timezone
from typing import Optional

from fastapi import HTTPException
from sqlalchemy.orm import Session

from ..models import AccountStatus, UserAuditLogDB, UserDB
from ..schemas import UserCreate, UserLogin
from ..utils.auth_util import create_access_token, get_password_hash, verify_password
from .emailer import send_verification_email


async def create_user(db: Session, user_data: UserCreate) -> UserDB:
    """Create a new user in the database with enhanced validation

    Args:
        db: Database session
        user_data: User registration data

    Returns:
        Created user object

    Raises:
        HTTPException: If validation fails or user already exists

    """
    # Email validation is already done by Pydantic EmailStr
    # Username reserved validation is already done by Pydantic field_validator

    # Check if email exists (case-insensitive)
    email_str = str(user_data.email)
    if db.query(UserDB).filter(UserDB.email.ilike(email_str)).first():
        raise HTTPException(
            status_code=422,
            detail=[
                {
                    "loc": ["body", "email"],
                    "msg": "This email address is already registered",
                    "type": "value_error",
                }
            ],
        )

    # Check if username exists (case-insensitive)
    if db.query(UserDB).filter(UserDB.username.ilike(user_data.username)).first():
        raise HTTPException(
            status_code=422,
            detail=[
                {
                    "loc": ["body", "username"],
                    "msg": "This username is already taken",
                    "type": "value_error",
                }
            ],
        )

    # Generate secure verification token
    verification_token = str(uuid.uuid4())
    token_expiry = datetime.now(timezone.utc).replace(tzinfo=None) + timedelta(hours=24)

    # Create user object
    db_user = UserDB(
        id=str(uuid.uuid4()),
        email=email_str.lower(),  # Store email in lowercase
        username=user_data.username,
        password_hash=get_password_hash(user_data.password),
        verification_token=verification_token,
        verification_token_expires=token_expiry,
        account_status=AccountStatus.PENDING,
    )

    try:
        # Save to database
        db.add(db_user)
        db.commit()
        db.refresh(db_user)

        # Send verification email
        email_sent = await send_verification_email(db_user.email, verification_token)
        if not email_sent:
            # Rollback if email sending fails
            db.delete(db_user)
            db.commit()
            raise HTTPException(
                status_code=500,
                detail="Failed to send verification email. Please try again later.",
            )

        return db_user

    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=500,
            detail="An error occurred during registration. Please try again later.",
        ) from e


async def login_user_service(db: Session, user_data: UserLogin) -> tuple[UserDB, str]:
    """Login a user and generate JWT token

    Args:
        db: Database session
        user_data: User login data

    Returns:
        Tuple of (logged in user object, JWT access token)

    Raises:
        HTTPException: If login fails

    """
    # Find user by email (case-insensitive)
    email_str = str(user_data.email)
    user = db.query(UserDB).filter(UserDB.email.ilike(email_str)).first()

    if not user:
        raise HTTPException(status_code=401, detail="Invalid email")

    # Verify password
    if not verify_password(user_data.password, user.password_hash):
        raise HTTPException(status_code=401, detail="Invalid password")

    if not user.email_verified:
        raise HTTPException(status_code=403, detail="Email not verified")

    # Generate JWT token
    access_token = create_access_token(data={"sub": user.id})

    return user, access_token


async def verify_user_email(db: Session, token: str) -> dict:
    """Verify user's email address

    Args:
        db: Database session
        token: Email verification token

    Returns:
        Success message dictionary

    Raises:
        HTTPException: If token is invalid or expired

    """
    # Find user by verification token
    current_time = datetime.now(timezone.utc).replace(tzinfo=None)
    user = (
        db.query(UserDB)
        .filter(
            UserDB.verification_token == token,
            UserDB.verification_token_expires > current_time,
        )
        .first()
    )

    if not user:
        raise HTTPException(
            status_code=400, detail="Invalid or expired verification token"
        )

    # Update user status
    user.email_verified = True
    user.account_status = AccountStatus.VERIFIED
    user.verification_token = None
    user.verification_token_expires = None

    db.commit()

    return {"message": "Email verified successfully"}


async def resend_verification_email(db: Session, email: str) -> dict:
    """Resend verification email to user

    Args:
        db: Database session
        email: User's email address

    Returns:
        Success message dictionary

    Raises:
        HTTPException: If user not found or already verified

    """
    user = db.query(UserDB).filter(UserDB.email == email).first()

    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    if user.email_verified:
        raise HTTPException(status_code=400, detail="Email already verified")

    # Generate new verification token
    verification_token = str(uuid.uuid4())
    user.verification_token = verification_token
    user.verification_token_expires = datetime.now(timezone.utc).replace(
        tzinfo=None
    ) + timedelta(hours=24)
    db.commit()

    # Send new verification email
    await send_verification_email(user.email, verification_token)

    return {"message": "Verification email sent"}


# --- Admin User Management with Audit Logging ---


def create_user_audit_log(
    db: Session,
    target_user_id: str,
    target_username: str,
    action: str,
    admin_user_id: str,
    admin_username: str,
    changes: Optional[dict] = None,
) -> None:
    """Create an audit log entry for user management operations.

    Args:
        db: Database session
        target_user_id: ID of the user being modified
        target_username: Username of the user being modified
        action: Action type (role_change, status_change, profile_update, etc.)
        admin_user_id: ID of the admin user performing the action
        admin_username: Username of the admin user
        changes: Optional dictionary of changes made (old/new values)

    """
    audit_log = UserAuditLogDB(
        id=str(uuid.uuid4()),
        target_user_id=target_user_id,
        target_username=target_username,
        action=action,
        admin_user_id=admin_user_id,
        admin_username=admin_username,
        changes=json.dumps(changes) if changes else None,
    )

    db.add(audit_log)
    db.commit()


def get_user_audit_logs(
    db: Session,
    target_user_id: Optional[str] = None,
    limit: int = 100,
) -> list:
    """Get audit logs for user management operations.

    Args:
        db: Database session
        target_user_id: Optional user ID to filter logs
        limit: Maximum number of logs to return

    Returns:
        List of UserAuditLogDB objects

    """
    query = db.query(UserAuditLogDB).order_by(UserAuditLogDB.created_at.desc())

    if target_user_id:
        query = query.filter(UserAuditLogDB.target_user_id == target_user_id)

    return query.limit(limit).all()


async def update_user_profile_with_audit(
    db: Session,
    user_id: str,
    user_update: dict,
    admin_user_id: str,
    admin_username: str,
) -> UserDB:
    """Update a user's profile and create audit log entries.

    Args:
        db: Database session
        user_id: ID of the user to update
        user_update: Dictionary of fields to update
        admin_user_id: ID of the admin performing the update
        admin_username: Username of the admin performing the update

    Returns:
        Updated UserDB object

    Raises:
        HTTPException: If user not found or validation fails

    """
    user = db.query(UserDB).filter(UserDB.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail=f"User with ID {user_id} not found")

    changes = {}

    # Track changes for audit log
    if "username" in user_update and user_update["username"] is not None:
        if user_update["username"] != user.username:
            # Check if username already exists
            existing = (
                db.query(UserDB)
                .filter(UserDB.username == user_update["username"])
                .first()
            )
            if existing:
                raise HTTPException(status_code=400, detail="Username already exists")
            changes["username"] = {
                "old": user.username,
                "new": user_update["username"],
            }
            user.username = user_update["username"]

    if "email" in user_update and user_update["email"] is not None:
        if user_update["email"] != user.email:
            # Check if email already exists
            existing = (
                db.query(UserDB).filter(UserDB.email == user_update["email"]).first()
            )
            if existing:
                raise HTTPException(status_code=400, detail="Email already exists")
            changes["email"] = {"old": user.email, "new": user_update["email"]}
            user.email = user_update["email"]

    if "role" in user_update and user_update["role"] is not None:
        if user_update["role"] != user.role:
            changes["role"] = {"old": user.role, "new": user_update["role"]}
            user.role = user_update["role"]
            # Create specific audit log for role change
            create_user_audit_log(
                db=db,
                target_user_id=user.id,
                target_username=user.username,
                action="role_change",
                admin_user_id=admin_user_id,
                admin_username=admin_username,
                changes={"old": changes["role"]["old"], "new": changes["role"]["new"]},
            )

    if "account_status" in user_update and user_update["account_status"] is not None:
        if user_update["account_status"] != user.account_status:
            changes["account_status"] = {
                "old": user.account_status,
                "new": user_update["account_status"],
            }
            user.account_status = user_update["account_status"]
            # Create specific audit log for status change
            create_user_audit_log(
                db=db,
                target_user_id=user.id,
                target_username=user.username,
                action="status_change",
                admin_user_id=admin_user_id,
                admin_username=admin_username,
                changes={
                    "old": changes["account_status"]["old"],
                    "new": changes["account_status"]["new"],
                },
            )

    if "email_verified" in user_update and user_update["email_verified"] is not None:
        if user_update["email_verified"] != user.email_verified:
            changes["email_verified"] = {
                "old": user.email_verified,
                "new": user_update["email_verified"],
            }
            user.email_verified = user_update["email_verified"]
            # Create specific audit log for email verification change
            create_user_audit_log(
                db=db,
                target_user_id=user.id,
                target_username=user.username,
                action="email_verify",
                admin_user_id=admin_user_id,
                admin_username=admin_username,
                changes={
                    "old": changes["email_verified"]["old"],
                    "new": changes["email_verified"]["new"],
                },
            )

    # Create general profile update log if there were other changes
    if any(
        key in changes
        for key in ["username", "email"]
        if key not in ["role", "account_status", "email_verified"]
    ):
        profile_changes = {
            k: v for k, v in changes.items() if k in ["username", "email"]
        }
        if profile_changes:
            create_user_audit_log(
                db=db,
                target_user_id=user.id,
                target_username=user.username,
                action="profile_update",
                admin_user_id=admin_user_id,
                admin_username=admin_username,
                changes=profile_changes,
            )

    db.commit()
    db.refresh(user)

    return user
