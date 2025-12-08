"""Service for handling recommendation feedback."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy.orm import Session

from ..models.models import RecommendationFeedbackDB, FeedbackType
from ..schemas.recommendation_schemas import FeedbackRequest, FeedbackResponse


class FeedbackService:
    """Service for managing recommendation feedback."""

    def __init__(self, db: Session) -> None:
        self.db = db

    def submit_feedback(
        self, user_id: str, request: FeedbackRequest
    ) -> FeedbackResponse:
        """Submit feedback for a recommended item.

        Args:
            user_id: ID of the user submitting feedback
            request: Feedback request containing item_id, item_type, feedback_type, and optional notes

        Returns:
            FeedbackResponse with the created feedback record

        Raises:
            ValueError: If feedback_type is invalid
        """
        # Validate feedback_type
        if request.feedback_type not in ["like", "dislike"]:
            raise ValueError(f"Invalid feedback_type: {request.feedback_type}")

        # Check if user already has feedback for this item (update existing)
        existing_feedback = (
            self.db.query(RecommendationFeedbackDB)
            .filter(
                RecommendationFeedbackDB.user_id == user_id,
                RecommendationFeedbackDB.item_id == request.item_id,
                RecommendationFeedbackDB.item_type == request.item_type,
            )
            .first()
        )

        if existing_feedback:
            # Update existing feedback
            existing_feedback.feedback_type = request.feedback_type
            existing_feedback.notes = request.notes
            existing_feedback.updated_at = datetime.now(timezone.utc).replace(tzinfo=None)
            self.db.commit()
            self.db.refresh(existing_feedback)

            return FeedbackResponse(
                id=existing_feedback.id,
                item_id=existing_feedback.item_id,
                item_type=existing_feedback.item_type,
                feedback_type=existing_feedback.feedback_type,
                created_at=existing_feedback.created_at.isoformat(),
            )

        # Create new feedback
        feedback_id = str(uuid.uuid4())
        feedback = RecommendationFeedbackDB(
            id=feedback_id,
            user_id=user_id,
            item_id=request.item_id,
            item_type=request.item_type,
            feedback_type=request.feedback_type,
            notes=request.notes,
        )

        self.db.add(feedback)
        self.db.commit()
        self.db.refresh(feedback)

        return FeedbackResponse(
            id=feedback.id,
            item_id=feedback.item_id,
            item_type=feedback.item_type,
            feedback_type=feedback.feedback_type,
            created_at=feedback.created_at.isoformat(),
        )

    def get_user_disliked_items(
        self, user_id: str, item_type: str | None = None
    ) -> set[str]:
        """Get set of item IDs that the user has disliked.

        Args:
            user_id: ID of the user
            item_type: Optional filter for item type ('meal' or 'restaurant')

        Returns:
            Set of item IDs that the user has disliked
        """
        query = self.db.query(RecommendationFeedbackDB).filter(
            RecommendationFeedbackDB.user_id == user_id,
            RecommendationFeedbackDB.feedback_type == FeedbackType.DISLIKE.value,
        )

        if item_type:
            query = query.filter(RecommendationFeedbackDB.item_type == item_type)

        disliked_items = query.all()
        return {item.item_id for item in disliked_items}

    def get_user_liked_items(
        self, user_id: str, item_type: str | None = None
    ) -> set[str]:
        """Get set of item IDs that the user has liked.

        Args:
            user_id: ID of the user
            item_type: Optional filter for item type ('meal' or 'restaurant')

        Returns:
            Set of item IDs that the user has liked
        """
        query = self.db.query(RecommendationFeedbackDB).filter(
            RecommendationFeedbackDB.user_id == user_id,
            RecommendationFeedbackDB.feedback_type == FeedbackType.LIKE.value,
        )

        if item_type:
            query = query.filter(RecommendationFeedbackDB.item_type == item_type)

        liked_items = query.all()
        return {item.item_id for item in liked_items}

    def get_user_feedback_for_items(
        self, user_id: str, item_ids: list[str], item_type: str
    ) -> dict[str, str]:
        """Get feedback type for specific items.

        Args:
            user_id: ID of the user
            item_ids: List of item IDs to check
            item_type: Item type ('meal' or 'restaurant')

        Returns:
            Dictionary mapping item_id to feedback_type ('like' or 'dislike')
        """
        if not item_ids:
            return {}

        feedback_records = (
            self.db.query(RecommendationFeedbackDB)
            .filter(
                RecommendationFeedbackDB.user_id == user_id,
                RecommendationFeedbackDB.item_id.in_(item_ids),
                RecommendationFeedbackDB.item_type == item_type,
            )
            .all()
        )

        return {record.item_id: record.feedback_type for record in feedback_records}

