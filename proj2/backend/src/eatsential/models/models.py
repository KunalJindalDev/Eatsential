"""SQLAlchemy ORM models for database tables."""

from datetime import date, datetime, timezone
from enum import Enum
from typing import Optional

from sqlalchemy import (
    Boolean,
    Column,
    Date,
    DateTime,
    ForeignKey,
    Numeric,
    String,
    Table,
    Text,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from ..db.database import Base


def utcnow():
    """Return current UTC time as naive datetime (UTC)."""
    return datetime.now(timezone.utc).replace(tzinfo=None)


class AccountStatus(str, Enum):
    """User account status"""

    PENDING = "pending"
    VERIFIED = "verified"
    SUSPENDED = "suspended"


class UserRole(str, Enum):
    """User role for access control"""

    USER = "user"
    ADMIN = "admin"


class UserDB(Base):
    """SQLAlchemy model for user database table"""

    __tablename__ = "users"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    email: Mapped[str] = mapped_column(String, unique=True, nullable=False, index=True)
    username: Mapped[str] = mapped_column(
        String(20), unique=True, nullable=False, index=True
    )
    password_hash: Mapped[str] = mapped_column(String, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=utcnow, nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=utcnow, onupdate=utcnow, nullable=False
    )
    account_status: Mapped[str] = mapped_column(
        String, nullable=False, default=AccountStatus.PENDING
    )
    email_verified: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    verification_token: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    verification_token_expires: Mapped[Optional[datetime]] = mapped_column(
        DateTime, nullable=True
    )
    role: Mapped[str] = mapped_column(
        String, nullable=False, default=UserRole.USER, index=True
    )
    timezone: Mapped[str] = mapped_column(
        String(50), nullable=False, default="America/New_York"
    )

    # Relationships
    health_profile: Mapped[Optional["HealthProfileDB"]] = relationship(
        "HealthProfileDB",
        back_populates="user",
        uselist=False,
        cascade="all, delete-orphan",
    )
    meals: Mapped[list["MealDB"]] = relationship(
        "MealDB", back_populates="user", cascade="all, delete-orphan"
    )
    goals: Mapped[list["GoalDB"]] = relationship(
        "GoalDB", back_populates="user", cascade="all, delete-orphan"
    )
    mood_logs: Mapped[list["MoodLogDB"]] = relationship(
        "MoodLogDB", back_populates="user", cascade="all, delete-orphan"
    )
    stress_logs: Mapped[list["StressLogDB"]] = relationship(
        "StressLogDB", back_populates="user", cascade="all, delete-orphan"
    )
    sleep_logs: Mapped[list["SleepLogDB"]] = relationship(
        "SleepLogDB", back_populates="user", cascade="all, delete-orphan"
    )
    recommendation_feedback: Mapped[list["RecommendationFeedbackDB"]] = relationship(
        "RecommendationFeedbackDB",
        cascade="all, delete-orphan",
    )


class ActivityLevel(str, Enum):
    """Activity level for health profile"""

    SEDENTARY = "sedentary"
    LIGHT = "light"
    MODERATE = "moderate"
    ACTIVE = "active"
    VERY_ACTIVE = "very_active"


class AllergySeverity(str, Enum):
    """Allergy severity levels"""

    MILD = "mild"
    MODERATE = "moderate"
    SEVERE = "severe"
    LIFE_THREATENING = "life_threatening"


class PreferenceType(str, Enum):
    """Dietary preference types"""

    DIET = "diet"
    CUISINE = "cuisine"
    INGREDIENT = "ingredient"
    PREPARATION = "preparation"


class HealthProfileDB(Base):
    """SQLAlchemy model for health profile table"""

    __tablename__ = "health_profiles"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    user_id: Mapped[str] = mapped_column(
        String, ForeignKey("users.id", ondelete="CASCADE"), unique=True, nullable=False
    )

    # Biometric Data
    height_cm: Mapped[Optional[float]] = mapped_column(Numeric(5, 2), nullable=True)
    weight_kg: Mapped[Optional[float]] = mapped_column(Numeric(5, 2), nullable=True)
    activity_level: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    metabolic_rate: Mapped[Optional[int]] = mapped_column(String, nullable=True)

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=utcnow, nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=utcnow, onupdate=utcnow, nullable=False
    )

    # Relationships
    user: Mapped["UserDB"] = relationship("UserDB", back_populates="health_profile")
    allergies: Mapped[list["UserAllergyDB"]] = relationship(
        "UserAllergyDB", back_populates="health_profile", cascade="all, delete-orphan"
    )
    dietary_preferences: Mapped[list["DietaryPreferenceDB"]] = relationship(
        "DietaryPreferenceDB",
        back_populates="health_profile",
        cascade="all, delete-orphan",
    )


# ============================================================================
# Menu Item - Allergen Association Table (defined early for use in models)
# ============================================================================

menu_item_allergens = Table(
    "menu_item_allergens",
    Base.metadata,
    Column("menu_item_id", String, ForeignKey("menu_items.id"), primary_key=True),
    Column("allergen_id", String, ForeignKey("allergen_database.id"), primary_key=True),
)


# ============================================================================
# Allergen Models
# ============================================================================


class AllergenDB(Base):
    """SQLAlchemy model for allergen database table"""

    __tablename__ = "allergen_database"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    name: Mapped[str] = mapped_column(String(100), unique=True, nullable=False)
    category: Mapped[str] = mapped_column(String(50), nullable=False)
    is_major_allergen: Mapped[bool] = mapped_column(
        Boolean, default=False, nullable=False
    )
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=utcnow, nullable=False
    )

    # Relationships
    user_allergies: Mapped[list["UserAllergyDB"]] = relationship(
        "UserAllergyDB", back_populates="allergen"
    )
    menu_items: Mapped[list["MenuItem"]] = relationship(
        "MenuItem", secondary=menu_item_allergens, back_populates="allergens"
    )


class UserAllergyDB(Base):
    """SQLAlchemy model for user allergies table"""

    __tablename__ = "user_allergies"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    health_profile_id: Mapped[str] = mapped_column(
        String, ForeignKey("health_profiles.id", ondelete="CASCADE"), nullable=False
    )
    allergen_id: Mapped[str] = mapped_column(
        String, ForeignKey("allergen_database.id"), nullable=False
    )

    # Allergy Information
    severity: Mapped[str] = mapped_column(String(20), nullable=False)
    diagnosed_date: Mapped[Optional[date]] = mapped_column(Date, nullable=True)
    reaction_type: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    notes: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Verification
    is_verified: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=utcnow, nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=utcnow, onupdate=utcnow, nullable=False
    )

    # Relationships
    health_profile: Mapped["HealthProfileDB"] = relationship(
        "HealthProfileDB", back_populates="allergies"
    )
    allergen: Mapped["AllergenDB"] = relationship(
        "AllergenDB", back_populates="user_allergies"
    )


class DietaryPreferenceDB(Base):
    """SQLAlchemy model for dietary preferences table"""

    __tablename__ = "dietary_preferences"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    health_profile_id: Mapped[str] = mapped_column(
        String, ForeignKey("health_profiles.id", ondelete="CASCADE"), nullable=False
    )

    # Preference Details
    preference_type: Mapped[str] = mapped_column(String(50), nullable=False)
    preference_name: Mapped[str] = mapped_column(String(100), nullable=False)
    is_strict: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)

    # Reason and Notes
    reason: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    notes: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=utcnow, nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=utcnow, onupdate=utcnow, nullable=False
    )

    # Relationships
    health_profile: Mapped["HealthProfileDB"] = relationship(
        "HealthProfileDB", back_populates="dietary_preferences"
    )


class MealType(str, Enum):
    """Meal type enum"""

    BREAKFAST = "breakfast"
    LUNCH = "lunch"
    DINNER = "dinner"
    SNACK = "snack"


class MealDB(Base):
    """SQLAlchemy model for meal logs table"""

    __tablename__ = "meals"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    user_id: Mapped[str] = mapped_column(
        String, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )

    # Meal Information
    meal_type: Mapped[str] = mapped_column(String(20), nullable=False, index=True)
    meal_time: Mapped[datetime] = mapped_column(DateTime, nullable=False, index=True)
    notes: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    photo_url: Mapped[Optional[str]] = mapped_column(String, nullable=True)

    # Nutritional Summary (calculated from food items)
    total_calories: Mapped[Optional[float]] = mapped_column(
        Numeric(10, 2), nullable=True
    )
    total_protein_g: Mapped[Optional[float]] = mapped_column(
        Numeric(10, 2), nullable=True
    )
    total_carbs_g: Mapped[Optional[float]] = mapped_column(
        Numeric(10, 2), nullable=True
    )
    total_fat_g: Mapped[Optional[float]] = mapped_column(Numeric(10, 2), nullable=True)

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=utcnow, nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=utcnow, onupdate=utcnow, nullable=False
    )

    # Relationships
    user: Mapped["UserDB"] = relationship("UserDB")
    food_items: Mapped[list["MealFoodItemDB"]] = relationship(
        "MealFoodItemDB", back_populates="meal", cascade="all, delete-orphan"
    )


class MealFoodItemDB(Base):
    """SQLAlchemy model for food items in a meal"""

    __tablename__ = "meal_food_items"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    meal_id: Mapped[str] = mapped_column(
        String, ForeignKey("meals.id", ondelete="CASCADE"), nullable=False, index=True
    )

    # Food Information
    food_name: Mapped[str] = mapped_column(String(200), nullable=False)
    portion_size: Mapped[float] = mapped_column(Numeric(10, 2), nullable=False)
    portion_unit: Mapped[str] = mapped_column(String(20), nullable=False)

    # Nutritional Information
    calories: Mapped[Optional[float]] = mapped_column(Numeric(10, 2), nullable=True)
    protein_g: Mapped[Optional[float]] = mapped_column(Numeric(10, 2), nullable=True)
    carbs_g: Mapped[Optional[float]] = mapped_column(Numeric(10, 2), nullable=True)
    fat_g: Mapped[Optional[float]] = mapped_column(Numeric(10, 2), nullable=True)

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=utcnow, nullable=False
    )

    # Relationships
    meal: Mapped["MealDB"] = relationship("MealDB", back_populates="food_items")


class GoalType(str, Enum):
    """Enumeration of goal types"""

    NUTRITION = "nutrition"
    WELLNESS = "wellness"


class GoalStatus(str, Enum):
    """Enumeration of goal statuses"""

    ACTIVE = "active"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


class GoalDB(Base):
    """SQLAlchemy model for health goals"""

    __tablename__ = "goals"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    user_id: Mapped[str] = mapped_column(
        String, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )

    # Goal Definition
    goal_type: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    target_type: Mapped[str] = mapped_column(
        String(100), nullable=False
    )  # e.g., "daily_calories", "weekly_protein"
    target_value: Mapped[float] = mapped_column(Numeric(10, 2), nullable=False)
    current_value: Mapped[float] = mapped_column(
        Numeric(10, 2), nullable=False, default=0
    )

    # Date Range
    start_date: Mapped[date] = mapped_column(Date, nullable=False, index=True)
    end_date: Mapped[date] = mapped_column(Date, nullable=False)

    # Status
    status: Mapped[str] = mapped_column(
        String(20), nullable=False, default=GoalStatus.ACTIVE.value, index=True
    )

    # Optional
    notes: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=utcnow, nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=utcnow, onupdate=utcnow, nullable=False
    )

    # Relationships
    user: Mapped["UserDB"] = relationship("UserDB", back_populates="goals")


# ============================================================================
# Mental Wellness Models
# ============================================================================


class LogType(str, Enum):
    """Log type enum"""

    MOOD = "mood"
    STRESS = "stress"
    SLEEP = "sleep"


class MoodLogDB(Base):
    """SQLAlchemy model for mood logging"""

    __tablename__ = "mood_logs"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    user_id: Mapped[str] = mapped_column(
        String, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )

    # Mood Data
    occurred_at_utc: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, index=True
    )
    mood_score: Mapped[int] = mapped_column(Numeric(2, 0), nullable=False)  # 1-10 scale

    # Encrypted sensitive data (optional notes)
    encrypted_notes: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=utcnow, nullable=False, index=True
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=utcnow, onupdate=utcnow, nullable=False
    )

    # Relationships
    user: Mapped["UserDB"] = relationship("UserDB", back_populates="mood_logs")


class StressLogDB(Base):
    """SQLAlchemy model for stress logging"""

    __tablename__ = "stress_logs"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    user_id: Mapped[str] = mapped_column(
        String, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )

    # Stress Data
    occurred_at_utc: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, index=True
    )
    stress_level: Mapped[int] = mapped_column(
        Numeric(2, 0), nullable=False
    )  # 1-10 scale

    # Encrypted sensitive data (triggers and notes)
    encrypted_triggers: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    encrypted_notes: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=utcnow, nullable=False, index=True
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=utcnow, onupdate=utcnow, nullable=False
    )

    # Relationships
    user: Mapped["UserDB"] = relationship("UserDB", back_populates="stress_logs")


class SleepLogDB(Base):
    """SQLAlchemy model for sleep logging"""

    __tablename__ = "sleep_logs"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    user_id: Mapped[str] = mapped_column(
        String, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )

    # Sleep Data
    occurred_at_utc: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, index=True
    )
    duration_hours: Mapped[float] = mapped_column(Numeric(4, 2), nullable=False)
    quality_score: Mapped[int] = mapped_column(
        Numeric(2, 0), nullable=False
    )  # 1-10 scale

    # Encrypted sensitive data (optional notes)
    encrypted_notes: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=utcnow, nullable=False, index=True
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=utcnow, onupdate=utcnow, nullable=False
    )

    # Relationships
    user: Mapped["UserDB"] = relationship("UserDB", back_populates="sleep_logs")


# ============================================================================
# Restaurant Models
# ============================================================================


class Restaurant(Base):
    """SQLAlchemy model representing a restaurant."""

    __tablename__ = "restaurants"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    name: Mapped[str] = mapped_column(String(200), nullable=False, index=True)
    address: Mapped[Optional[str]] = mapped_column(String(300), nullable=True)
    cuisine: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=utcnow, nullable=False
    )

    # Relationships
    menu_items: Mapped[list["MenuItem"]] = relationship(
        "MenuItem", back_populates="restaurant", cascade="all, delete-orphan"
    )


# ============================================================================
# MenuItem Model
# ============================================================================


class MenuItem(Base):
    """SQLAlchemy model representing a single menu item for a restaurant."""

    __tablename__ = "menu_items"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    restaurant_id: Mapped[str] = mapped_column(
        String, ForeignKey("restaurants.id", ondelete="CASCADE"), nullable=False
    )
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    calories: Mapped[Optional[float]] = mapped_column(Numeric(7, 2), nullable=True)
    price: Mapped[Optional[float]] = mapped_column(Numeric(8, 2), nullable=True)

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=utcnow, nullable=False
    )

    # Relationships
    restaurant: Mapped["Restaurant"] = relationship(
        "Restaurant", back_populates="menu_items"
    )
    allergens: Mapped[list["AllergenDB"]] = relationship(
        "AllergenDB", secondary=menu_item_allergens, back_populates="menu_items"
    )

# ============================================================================
# Orders Model
# ============================================================================

class Orders(Base):
    """SQLAlchemy model representing a potentially scheduled order for a user."""

    __tablename__ = "orders"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    menu_item_id : Mapped[str] = mapped_column(
        String, ForeignKey("menu_items.id", ondelete="CASCADE"), nullable=False
    )
    meal_id : Mapped[str] = mapped_column(
        String, ForeignKey("meals.id", ondelete="CASCADE"), nullable=False
    )


# ============================================================================
# Audit Log Models
# ============================================================================


class AuditAction(str, Enum):
    """Audit log action types"""

    CREATE = "create"
    UPDATE = "update"
    DELETE = "delete"
    BULK_IMPORT = "bulk_import"
    # User-specific actions
    ROLE_CHANGE = "role_change"
    STATUS_CHANGE = "status_change"
    PROFILE_UPDATE = "profile_update"
    EMAIL_VERIFY = "email_verify"


class AllergenAuditLogDB(Base):
    """SQLAlchemy model for allergen audit log table"""

    __tablename__ = "allergen_audit_logs"

    id: Mapped[str] = mapped_column(String, primary_key=True)

    # What was changed
    allergen_id: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    allergen_name: Mapped[str] = mapped_column(String(100), nullable=False)

    # Action details
    action: Mapped[str] = mapped_column(String(20), nullable=False)

    # Who made the change
    admin_user_id: Mapped[str] = mapped_column(
        String, ForeignKey("users.id"), nullable=False
    )
    admin_username: Mapped[str] = mapped_column(String(20), nullable=False)

    # Change details (JSON string)
    changes: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Timestamp
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=utcnow, nullable=False
    )

    # Relationships
    admin_user: Mapped["UserDB"] = relationship("UserDB")


class UserAuditLogDB(Base):
    """SQLAlchemy model for user audit log table

    Records all administrative actions performed on user accounts,
    including role changes, status updates, and profile modifications.
    """

    __tablename__ = "user_audit_logs"

    id: Mapped[str] = mapped_column(String, primary_key=True)

    # What was changed
    target_user_id: Mapped[str] = mapped_column(String, nullable=False)
    target_username: Mapped[str] = mapped_column(String(20), nullable=False)

    # Action details
    action: Mapped[str] = mapped_column(String(20), nullable=False)

    # Who made the change
    admin_user_id: Mapped[str] = mapped_column(
        String, ForeignKey("users.id"), nullable=False
    )
    admin_username: Mapped[str] = mapped_column(String(20), nullable=False)

    # Change details (JSON string containing old/new values)
    changes: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Timestamp
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=utcnow, nullable=False
    )

    # Relationships
    admin_user: Mapped["UserDB"] = relationship("UserDB", foreign_keys=[admin_user_id])


# ============================================================================
# Recommendation Feedback Models
# ============================================================================


class FeedbackType(str, Enum):
    """Feedback type enum"""

    LIKE = "like"
    DISLIKE = "dislike"


class RecommendationFeedbackDB(Base):
    """SQLAlchemy model for recommendation feedback table

    Stores user feedback (like/dislike) on recommended items to improve
    future recommendations by learning user preferences.
    """

    __tablename__ = "recommendation_feedback"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    user_id: Mapped[str] = mapped_column(
        String, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )
    item_id: Mapped[str] = mapped_column(
        String, nullable=False, index=True
    )  # Can be menu_item_id or restaurant_id
    item_type: Mapped[str] = mapped_column(
        String(20), nullable=False, index=True
    )  # "meal" or "restaurant"
    feedback_type: Mapped[str] = mapped_column(
        String(20), nullable=False, index=True
    )  # "like" or "dislike"

    # Optional context
    notes: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=utcnow, nullable=False, index=True
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=utcnow, onupdate=utcnow, nullable=False
    )

    # Relationships
    user: Mapped["UserDB"] = relationship("UserDB")
