"""Recommendation API endpoints for meals and restaurants."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, status
from sqlalchemy.orm import Session

from ..db.database import get_db
from ..models.models import UserDB
from ..schemas.recommendation_schemas import (
    RecommendationRequest,
    RecommendationResponse,
)
from ..services.auth_service import get_current_user
from ..services.engine import RecommendationService

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
