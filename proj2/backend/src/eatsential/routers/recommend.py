"""Recommendation API endpoints for meals and restaurants."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, Query, status
from sqlalchemy.orm import Session

from ..db.database import get_db
from ..models.models import UserDB
from ..schemas.recommendation_schemas import (
    FeedbackRequest,
    FeedbackResponse,
    RecommendationRequest,
    RecommendationResponse,
)
from ..services.auth_service import get_current_user
from ..services.engine import RecommendationService
from ..services.feedback_service import FeedbackService

router = APIRouter(prefix="/recommend", tags=["recommendations"])

SessionDep = Annotated[Session, Depends(get_db)]
CurrentUserDep = Annotated[UserDB, Depends(get_current_user)]


def _build_service(db: Session) -> RecommendationService:
    """Instantiate the recommendation service for the request lifecycle."""
    return RecommendationService(db)


@router.post(
    "/meal",
    response_model=RecommendationResponse,
    status_code=status.HTTP_200_OK,
)
async def recommend_meal(
    request: RecommendationRequest,
    current_user: CurrentUserDep,
    db: SessionDep,
) -> RecommendationResponse:
    """Return personalized meal recommendations using the LLM-enabled engine."""
    service = _build_service(db)
    return await service.get_meal_recommendations(user=current_user, request=request)


@router.post(
    "/restaurant",
    response_model=RecommendationResponse,
    status_code=status.HTTP_200_OK,
)
async def recommend_restaurant(
    request: RecommendationRequest,
    current_user: CurrentUserDep,
    db: SessionDep,
) -> RecommendationResponse:
    """Return restaurant recommendations using the LLM-enabled engine."""
    service = _build_service(db)
    return await service.get_restaurant_recommendations(user=current_user, request=request)


@router.post(
    "/feedback",
    response_model=FeedbackResponse,
    status_code=status.HTTP_201_CREATED,
)
def submit_feedback(
    request: FeedbackRequest,
    current_user: CurrentUserDep,
    db: SessionDep,
) -> FeedbackResponse:
    """Submit feedback (like/dislike) for a recommended item.

    This endpoint allows users to provide feedback on recommendations,
    which will be used to improve future recommendations by filtering
    out disliked items and prioritizing liked items.
    """
    service = FeedbackService(db)
    return service.submit_feedback(user_id=current_user.id, request=request)


@router.get(
    "/feedback",
    status_code=status.HTTP_200_OK,
)
async def get_feedback(
    current_user: CurrentUserDep,
    db: SessionDep,
    item_ids: str = Query(..., description="Comma-separated list of item IDs"),
    item_type: str = Query(..., description="Item type: 'meal' or 'restaurant'"),
) -> dict[str, str]:
    """Get feedback for specific items.

    Returns a dictionary mapping item_id to feedback_type ('like' or 'dislike').
    """
    item_id_list = [id.strip() for id in item_ids.split(",") if id.strip()]
    service = FeedbackService(db)
    feedback_map = service.get_user_feedback_for_items(
        user_id=current_user.id, item_ids=item_id_list, item_type=item_type
    )
    return await feedback_map
