# ruff: noqa: UP006,UP035,UP045
"""Schemas for recommendation endpoints.

These models capture request filters, control which recommendation engine
mode runs (baseline vs LLM), and define the response payload delivered
to the frontend consumers of the recommendation API.
"""

from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


class RecommendationFilters(BaseModel):
    """Optional filters provided by the client to refine recommendations."""

    diet: Optional[List[str]] = Field(
        default=None, description="Dietary labels to include such as 'vegan'."
    )
    cuisine: Optional[List[str]] = Field(
        default=None, description="Preferred cuisines such as 'italian'."
    )
    price_range: Optional[str] = Field(
        default=None,
        description="Desired price range (e.g. '$', '$$', '$$$').",
    )


class RecommendationRequest(BaseModel):
    """Request body accepted by the recommendation endpoints."""

    filters: Optional[RecommendationFilters] = None
    mode: Optional[Literal["llm", "baseline"]] = Field(
        default="llm",
        description="Determines which ranking engine to use.",
    )


class RecommendedItem(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    item_id: str
    name: str
    score: float = Field(ge=0.0, le=1.0)
    explanation: str
    price: Optional[float] = None
    calories: Optional[float] = None

    # NEW FIELDS
    restaurant_name: Optional[str] = None
    restaurant_address: Optional[str] = None
    restaurant_place_id: Optional[str] = None  # Google Places place_id for direct map lookup


class RecommendationResponse(BaseModel):
    """Response payload returned by the recommendation endpoints."""

    items: List[RecommendedItem]


# ============================================================================
# Feedback Schemas
# ============================================================================


class FeedbackRequest(BaseModel):
    """Request body for submitting recommendation feedback."""

    item_id: str = Field(..., description="ID of the recommended item (meal or restaurant)")
    item_type: Literal["meal", "restaurant"] = Field(
        ..., description="Type of item: 'meal' or 'restaurant'"
    )
    feedback_type: Literal["like", "dislike"] = Field(
        ..., description="Type of feedback: 'like' or 'dislike'"
    )
    notes: Optional[str] = Field(
        default=None, description="Optional notes about the feedback"
    )


class FeedbackResponse(BaseModel):
    """Response payload for feedback submission."""

    id: str
    item_id: str
    item_type: str
    feedback_type: str
    created_at: str
