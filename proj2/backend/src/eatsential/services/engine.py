"""Recommendation engine for safety filtering, baseline scoring, and LLM ranking."""

from __future__ import annotations

import json
import logging
import os
import random
from collections.abc import Sequence
from dataclasses import dataclass
from decimal import Decimal
from typing import TYPE_CHECKING, cast

from google import genai
from google.genai import types as genai_types
from pydantic import BaseModel
from sqlalchemy.orm import Session, selectinload

if TYPE_CHECKING:
    from google.genai.client import Client as GenAiClient

MAX_LLM_CANDIDATES = 100

from ..models.models import (
    GoalDB,
    GoalStatus,
    GoalType,
    HealthProfileDB,
    MenuItem,
    PreferenceType,
    Restaurant,
    UserAllergyDB,
    UserDB,
)
from ..schemas.recommendation_schemas import (
    RecommendationFilters,
    RecommendationRequest,
    RecommendationResponse,
    RecommendedItem,
)
from .feedback_service import FeedbackService

logger = logging.getLogger(__name__)


PRICE_RANGE_MAP: dict[str, tuple[float | None, float | None]] = {
    "$": (None, 10.0),
    "$$": (10.0, 25.0),
    "$$$": (25.0, 45.0),
    "$$$$": (45.0, None),
}
STRICT_DIET_EXCLUSIONS: dict[str, tuple[str, ...]] = {
    "vegan": (
        "beef",
        "pork",
        "chicken",
        "fish",
        "shrimp",
        "egg",
        "cheese",
        "milk",
        "honey",
        "butter",
        "yogurt",
    ),
    "vegetarian": (
        "beef",
        "pork",
        "chicken",
        "turkey",
        "fish",
        "shrimp",
        "bacon",
    ),
    "gluten-free": ("wheat", "barley", "rye", "gluten", "bread", "pasta"),
    "keto": ("sugar", "bread", "pasta", "rice", "noodle", "potato"),
}


@dataclass
class _UserContext:
    """Lightweight container for user health profile data used by the recommender."""

    user: UserDB
    allergies: list[str]
    strict_dietary_preferences: list[str]
    preferred_cuisines: list[str]
    health_goals: list[GoalDB]


class RecommendationService:
    """Orchestrates filtering, ranking, and LLM calls for recommendations."""

    def __init__(
        self,
        db: Session,
        *,
        llm_model: str | None = None,
        llm_api_key: str | None = None,
        llm_temperature: float | None = None,
        max_results: int = 5,
    ) -> None:
        self.db = db
        self.llm_api_key = llm_api_key or os.getenv("GEMINI_API_KEY")
        self.llm_model = llm_model or os.getenv("GEMINI_MODEL") or "gemini-2.5-flash"
        temperature_env = os.getenv("GEMINI_TEMPERATURE")
        if llm_temperature is not None:
            self.llm_temperature = llm_temperature
        else:
            self.llm_temperature = 0.2
            if temperature_env:
                try:
                    self.llm_temperature = float(temperature_env)
                except ValueError:
                    logger.warning(
                        "Invalid GEMINI_TEMPERATURE value '%s'; defaulting to 0.2",
                        temperature_env,
                    )
        self._llm_client: GenAiClient | None = None
        self.max_results = max_results

    # ------------------------------------------------------------------ #
    # Public APIs
    # ------------------------------------------------------------------ #

    def get_meal_recommendations(
        self,
        *,
        user: UserDB,
        request: RecommendationRequest,
    ) -> RecommendationResponse:
        """Return meal recommendations for the given user."""
        context = self._load_user_context(user)
        filters = request.filters or RecommendationFilters()

        candidates = self._get_menu_item_candidates()
        safe_candidates = self._apply_safety_filters(context, candidates)

        if not safe_candidates:
            return RecommendationResponse(items=[])

        # Filter out disliked items based on user feedback
        feedback_service = FeedbackService(self.db)
        disliked_items = feedback_service.get_user_disliked_items(
            user_id=user.id, item_type="meal"
        )
        liked_items = feedback_service.get_user_liked_items(
            user_id=user.id, item_type="meal"
        )

        # Remove disliked items from candidates
        filtered_candidates = [
            item for item in safe_candidates if str(item.id) not in disliked_items
        ]

        if not filtered_candidates:
            # If all items were disliked, return empty or fall back to safe_candidates
            # (depending on business logic - here we return empty to respect user preferences)
            logger.warning(
                "All meal candidates were filtered out due to user dislikes for user %s",
                user.id,
            )
            return RecommendationResponse(items=[])
        if len(safe_candidates) > MAX_LLM_CANDIDATES:
            safe_candidates = random.sample(
                safe_candidates, MAX_LLM_CANDIDATES
            )

        if (request.mode or "llm") == "baseline":
            baseline = self._get_baseline_meals(context, filtered_candidates, filters)
            # Boost scores for liked items
            baseline = self._apply_feedback_boosts(baseline, liked_items)
            return RecommendationResponse(items=baseline[: self.max_results])

        try:
            llm = self._get_llm_recommendations(
                context=context,
                items=filtered_candidates,
                filters=filters,
                entity_type="meal",
            )
            if llm:
                # Boost scores for liked items
                llm = self._apply_feedback_boosts(llm, liked_items)
                return RecommendationResponse(items=llm[: self.max_results])
        except Exception as exc:
            logger.exception(
                "LLM recommendation failed, falling back to baseline: %s", exc
            )

        baseline = self._get_baseline_meals(context, filtered_candidates, filters)
        # Boost scores for liked items
        baseline = self._apply_feedback_boosts(baseline, liked_items)
        return RecommendationResponse(items=baseline[: self.max_results])

    def get_restaurant_recommendations(
        self,
        *,
        user: UserDB,
        request: RecommendationRequest,
    ) -> RecommendationResponse:
        """Return restaurant recommendations for the given user."""
        context = self._load_user_context(user)
        filters = request.filters or RecommendationFilters()

        candidates = self._get_restaurant_candidates()
        safe_restaurants, menu_map = self._apply_restaurant_safety_filters(
            context, candidates
        )

        if not safe_restaurants:
            return RecommendationResponse(items=[])

        # Filter out disliked restaurants based on user feedback
        feedback_service = FeedbackService(self.db)
        disliked_items = feedback_service.get_user_disliked_items(
            user_id=user.id, item_type="restaurant"
        )
        liked_items = feedback_service.get_user_liked_items(
            user_id=user.id, item_type="restaurant"
        )

        # Remove disliked restaurants from candidates
        filtered_restaurants = [
            restaurant
            for restaurant in safe_restaurants
            if str(restaurant.id) not in disliked_items
        ]

        if not filtered_restaurants:
            logger.warning(
                "All restaurant candidates were filtered out due to user dislikes for user %s",
                user.id,
            )
            return RecommendationResponse(items=[])

        if (request.mode or "llm") == "baseline":
            baseline = self._get_baseline_restaurants(
                context, filtered_restaurants, menu_map, filters
            )
            # Boost scores for liked restaurants
            baseline = self._apply_feedback_boosts(baseline, liked_items)
            return RecommendationResponse(items=baseline[: self.max_results])

        try:
            llm = self._get_llm_recommendations(
                context=context,
                items=filtered_restaurants,
                filters=filters,
                entity_type="restaurant",
                restaurant_menu_map=menu_map,
            )
            if llm:
                # Boost scores for liked restaurants
                llm = self._apply_feedback_boosts(llm, liked_items)
                return RecommendationResponse(items=llm[: self.max_results])
        except Exception as exc:
            logger.exception(
                "LLM restaurant recommendation failed, falling back to baseline: %s",
                exc,
            )

        baseline = self._get_baseline_restaurants(
            context, filtered_restaurants, menu_map, filters
        )
        # Boost scores for liked restaurants
        baseline = self._apply_feedback_boosts(baseline, liked_items)
        return RecommendationResponse(items=baseline[: self.max_results])

    # ------------------------------------------------------------------ #
    # Data access helpers
    # ------------------------------------------------------------------ #

    def _load_user_context(self, user: UserDB) -> _UserContext:
        """Eagerly load related data needed for recommendations."""
        refreshed = (
            self.db.query(UserDB)
            .options(
                selectinload(UserDB.health_profile)
                .selectinload(HealthProfileDB.allergies)
                .selectinload(UserAllergyDB.allergen),
                selectinload(UserDB.health_profile).selectinload(
                    HealthProfileDB.dietary_preferences
                ),
                selectinload(UserDB.goals),
            )
            .filter(UserDB.id == user.id)
            .first()
        )
        if not refreshed:
            raise ValueError("User not found")

        allergies = []
        strict_diets = []
        preferred_cuisines = []

        if refreshed.health_profile:
            for allergy in refreshed.health_profile.allergies:
                if allergy.allergen:
                    allergies.append(allergy.allergen.name.lower())
            for pref in refreshed.health_profile.dietary_preferences:
                if pref.preference_type == PreferenceType.DIET.value and pref.is_strict:
                    strict_diets.append(pref.preference_name.lower())
                if pref.preference_type == PreferenceType.CUISINE.value:
                    preferred_cuisines.append(pref.preference_name.lower())

        active_goals = [
            goal for goal in refreshed.goals if goal.status == GoalStatus.ACTIVE.value
        ]

        return _UserContext(
            user=refreshed,
            allergies=allergies,
            strict_dietary_preferences=strict_diets,
            preferred_cuisines=preferred_cuisines,
            health_goals=active_goals,
        )

    def _get_menu_item_candidates(self) -> list[MenuItem]:
        """Fetch menu item candidates from active restaurants."""
        return (
            self.db.query(MenuItem)
            .join(Restaurant)
            .options(
                selectinload(MenuItem.restaurant),
                selectinload(MenuItem.allergens),
            )
            .filter(Restaurant.is_active.is_(True))
            .all()
        )

    def _get_restaurant_candidates(self) -> list[Restaurant]:
        """Fetch restaurant candidates with their menu items."""
        return (
            self.db.query(Restaurant)
            .options(
                selectinload(Restaurant.menu_items).selectinload(MenuItem.allergens)
            )
            .filter(Restaurant.is_active.is_(True))
            .all()
        )

    # ------------------------------------------------------------------ #
    # Safety filtering
    # ------------------------------------------------------------------ #

    def _apply_safety_filters(
        self,
        context: _UserContext,
        items: Sequence[MenuItem],
    ) -> list[MenuItem]:
        """Filter menu items that violate allergy or strict dietary rules.

        Uses two-tier allergen checking:
        1. Database relationships (MenuItem.allergens) - most reliable
        2. Text-based fallback for items without allergen data
        """
        if not context.allergies and not context.strict_dietary_preferences:
            return list(items)

        safe_items: list[MenuItem] = []
        for item in items:
            # Tier 1: Check database allergen relationships (most reliable)
            if context.allergies and item.allergens:
                item_allergen_names = {
                    allergen.name.lower() for allergen in item.allergens
                }
                user_allergen_set = set(context.allergies)
                if item_allergen_names & user_allergen_set:
                    # Item contains user allergen via database relationship
                    continue

            # Tier 2: Fallback to text-based checking for items without allergen data
            # or for additional safety
            text = f"{item.name} {item.description or ''}".lower()
            if self._contains_allergen(text, context.allergies):
                continue
            if self._violates_strict_diet(text, context.strict_dietary_preferences):
                continue
            safe_items.append(item)
        return safe_items

    def _apply_restaurant_safety_filters(
        self,
        context: _UserContext,
        restaurants: Sequence[Restaurant],
    ) -> tuple[list[Restaurant], dict[str, list[MenuItem]]]:
        """Filter restaurants to those that have at least one compliant menu item."""
        safe_restaurants: list[Restaurant] = []
        safe_menu_items: dict[str, list[MenuItem]] = {}

        for restaurant in restaurants:
            compliant_items = self._apply_safety_filters(
                context, restaurant.menu_items or []
            )
            if compliant_items:
                safe_restaurants.append(restaurant)
                safe_menu_items[str(restaurant.id)] = compliant_items

        return safe_restaurants, safe_menu_items

    def _contains_allergen(self, text: str, allergens: Sequence[str]) -> bool:
        """Return True if text contains any allergen term."""
        return any(allergen in text for allergen in allergens)

    def _violates_strict_diet(
        self,
        text: str,
        strict_diets: Sequence[str],
    ) -> bool:
        """Return True if text breaks a strict dietary rule."""
        for diet in strict_diets:
            exclusions = STRICT_DIET_EXCLUSIONS.get(diet)
            if exclusions and any(term in text for term in exclusions):
                return True
        return False

    # ------------------------------------------------------------------ #
    # Baseline logic
    # ------------------------------------------------------------------ #

    def _get_baseline_meals(
        self,
        context: _UserContext,
        items: Sequence[MenuItem],
        filters: RecommendationFilters,
    ) -> list[RecommendedItem]:
        """Compute heuristic ranking for menu items."""
        results: list[RecommendedItem] = []
        allowed_cuisines = {c.lower() for c in filters.cuisine or []}

        for item in items:
            cuisine = (item.restaurant.cuisine or "").lower() if item.restaurant else ""
            price = item.price
            calories = item.calories
            text = f"{item.name} {item.description or ''}".lower()

            if filters.price_range and not self._price_in_range(
                price, filters.price_range
            ):
                continue
            if allowed_cuisines and cuisine and cuisine not in allowed_cuisines:
                continue

            score = 0.35
            explanation_bits: list[str] = []

            if cuisine:
                explanation_bits.append(f"Cuisine: {item.restaurant.cuisine}")
                if cuisine in context.preferred_cuisines:
                    score += 0.2
                if cuisine in allowed_cuisines:
                    score += 0.15

            if price is not None:
                explanation_bits.append(f"Price: ${price:.2f}")
                if filters.price_range:
                    score += 0.15
                else:
                    score += 0.05

            if filters.diet:
                matches = [
                    diet for diet in (filters.diet or []) if diet.lower() in text
                ]
                if matches:
                    score += 0.1
                    explanation_bits.append(f"Matches diet: {', '.join(matches)}")

            if calories is not None:
                explanation_bits.append(f"{calories:.0f} kcal")
                if self._supports_calorie_goal(context.health_goals, calories):
                    score += 0.1

            if context.health_goals and self._mentions_goal_keywords(
                text, context.health_goals
            ):
                score += 0.05

            score = max(0.0, min(score, 1.0))
            explanation = "; ".join(explanation_bits) or "Matches user preferences"
            results.append(
                RecommendedItem(
                    item_id=str(item.id),
                    name=item.name,
                    score=score,
                    explanation=explanation,
                    price=price,
                    calories=calories,
                )
            )

        results.sort(key=lambda rec: (-rec.score, rec.item_id))
        return results

    def _get_baseline_restaurants(
        self,
        context: _UserContext,
        restaurants: Sequence[Restaurant],
        menu_map: dict[str, list[MenuItem]],
        filters: RecommendationFilters,
    ) -> list[RecommendedItem]:
        """Compute baseline ranking for restaurants."""
        results: list[RecommendedItem] = []
        allowed_cuisines = {c.lower() for c in filters.cuisine or []}

        for restaurant in restaurants:
            cuisine = (restaurant.cuisine or "").lower()
            if allowed_cuisines and cuisine and cuisine not in allowed_cuisines:
                continue

            menu_items = menu_map.get(str(restaurant.id), [])
            avg_price = self._average_price(menu_items)
            price_ok = self._price_in_range(avg_price, filters.price_range)
            if filters.price_range and not price_ok:
                continue

            text_blob = " ".join(
                f"{item.name} {item.description or ''}".lower() for item in menu_items
            )

            score = 0.4
            explanation_bits: list[str] = []

            if restaurant.cuisine:
                explanation_bits.append(f"Cuisine: {restaurant.cuisine}")
                if cuisine in context.preferred_cuisines:
                    score += 0.2
                if cuisine in allowed_cuisines:
                    score += 0.15

            if avg_price is not None:
                explanation_bits.append(f"Avg. price ≈ ${avg_price:.2f}")
                if filters.price_range:
                    score += 0.15
                else:
                    score += 0.05

            if filters.diet:
                matches = [
                    diet for diet in (filters.diet or []) if diet.lower() in text_blob
                ]
                if matches:
                    explanation_bits.append(f"Menu mentions {', '.join(matches)}")
                    score += 0.1

            if context.health_goals and self._mentions_goal_keywords(
                text_blob, context.health_goals
            ):
                score += 0.05

            score = max(0.0, min(score, 1.0))
            explanation = "; ".join(explanation_bits) or "Menu aligns with preferences"

            results.append(
                RecommendedItem(
                    item_id=str(restaurant.id),
                    name=restaurant.name,
                    score=score,
                    explanation=explanation,
                )
            )

        results.sort(key=lambda rec: (-rec.score, rec.item_id))
        return results

    # ------------------------------------------------------------------ #
    # LLM logic
    # ------------------------------------------------------------------ #

    def _get_llm_client(self) -> GenAiClient:
        """Create or reuse a Gemini client."""
        if not self.llm_api_key:
            raise RuntimeError("LLM API key is not configured")
        if self._llm_client is None:
            self._llm_client = genai.Client(api_key=self.llm_api_key)
        return self._llm_client

    def _get_llm_recommendations(
        self,
        *,
        context: _UserContext,
        items: Sequence[MenuItem] | Sequence[Restaurant],
        filters: RecommendationFilters,
        entity_type: str,
        restaurant_menu_map: dict[str, list[MenuItem]] | None = None,
    ) -> list[RecommendedItem]:
        """Call the Gemini API via google-genai for ranking and explanations."""
        client = self._get_llm_client()

        prompt = self._build_prompt(
            context=context,
            items=items,
            filters=filters,
            entity_type=entity_type,
            restaurant_menu_map=restaurant_menu_map,
        )

        config = genai_types.GenerateContentConfig(
            temperature=self.llm_temperature,
            response_mime_type="application/json",
        )

        response = client.models.generate_content(
            model=self.llm_model,
            contents=[prompt],
            config=config,
        )

        structured = self._extract_llm_suggestions(response)

        candidate_lookup: dict[str, MenuItem | Restaurant]
        if entity_type == "meal":
            menu_items = cast(Sequence[MenuItem], items)
            candidate_lookup = {str(item.id): item for item in menu_items}
        else:
            restaurants = cast(Sequence[Restaurant], items)
            candidate_lookup = {
                str(restaurant.id): restaurant for restaurant in restaurants
            }

        recommendations: list[RecommendedItem] = []
        for entry in structured:
            item_id = entry.get("item_id")
            if item_id is None:
                continue

            item = candidate_lookup.get(str(item_id))
            if not item:
                continue

            name_raw = entry.get("name")
            if isinstance(name_raw, str) and name_raw:
                name = name_raw
            else:
                name = getattr(item, "name", "")

            raw_score = entry.get("score", 0.0)
            try:
                if isinstance(raw_score, (int, float, str)):
                    score = float(raw_score)
                else:
                    score = 0.0
            except (TypeError, ValueError):
                score = 0.0
            score = max(0.0, min(score, 1.0))

            explanation_raw = entry.get("explanation")
            if isinstance(explanation_raw, str):
                explanation_candidate = explanation_raw
            elif explanation_raw is None:
                explanation_candidate = ""
            else:
                explanation_candidate = str(explanation_raw)
            explanation = explanation_candidate.strip() or "Selected by LLM ranking"

            price = getattr(item, "price", None)
            calories = getattr(item, "calories", None)

            recommendations.append(
                RecommendedItem(
                    item_id=str(item.id),
                    name=name,
                    score=score,
                    explanation=explanation,
                    price=price,
                    calories=calories,
                )
            )

        recommendations.sort(key=lambda rec: (-rec.score, rec.item_id))
        
        # Deduplicate by item_id (keep the one with highest score)
        seen_ids: dict[str, RecommendedItem] = {}
        for rec in recommendations:
            if rec.item_id not in seen_ids:
                seen_ids[rec.item_id] = rec
            elif rec.score > seen_ids[rec.item_id].score:
                seen_ids[rec.item_id] = rec
        
        return list(seen_ids.values())

    def _build_prompt(
        self,
        *,
        context: _UserContext,
        items: Sequence[MenuItem] | Sequence[Restaurant],
        filters: RecommendationFilters,
        entity_type: str,
        restaurant_menu_map: dict[str, list[MenuItem]] | None = None,
    ) -> str:
        """Construct a structured prompt for Gemini."""
        user_profile = self._serialize_user_profile(context)
        filters_payload = {
            "diet": filters.diet or [],
            "cuisine": filters.cuisine or [],
            "price_range": filters.price_range,
        }

        if entity_type == "meal":
            menu_items = cast(Sequence[MenuItem], items)
            candidates_payload = [
                self._serialize_menu_item(item) for item in menu_items
            ]
        else:
            restaurants = cast(Sequence[Restaurant], items)
            candidates_payload = [
                self._serialize_restaurant(
                    restaurant,
                    (restaurant_menu_map or {}).get(str(restaurant.id), []),
                )
                for restaurant in restaurants
            ]

        prompt = (
            "You are a helpful nutrition and dining assistant.\n\n"
            f"User Profile:\n{json.dumps(user_profile, indent=2)}\n\n"
            f"Request Filters:\n{json.dumps(filters_payload, indent=2)}\n\n"
            f"Candidate {entity_type.title()}s:\n"
            f"{json.dumps(candidates_payload, indent=2)}\n\n"
            f"Task: From the candidate list provided, select and rank the top {self.max_results} items "
            "that best match the user's profile, health context, and request filters. "
            "For each item, provide a score between 0.0 and 1.0 "
            "and a short explanation for why it's a good match.\n\n"
            "Output Format: Return your response only as a valid JSON list in the "
            'following format: [{"item_id": "...", "name": "...", "score": 0.9, '
            '"explanation": "..."}]'
        )
        return prompt

    def _extract_llm_suggestions(self, data: object) -> list[dict[str, object]]:
        """Normalize LLM response into a list of suggestion dictionaries."""
        if isinstance(data, genai_types.GenerateContentResponse):
            if data.text:
                return self._parse_json_payload(data.text)
            if data.parsed:
                return self._parse_json_payload(data.parsed)
            return self._parse_json_payload(data.model_dump())

        if isinstance(data, list):
            return [self._ensure_dict(entry) for entry in data]

        if isinstance(data, dict):
            if "output" in data:
                return self._parse_json_payload(data["output"])
            if "result" in data:
                return self._parse_json_payload(data["result"])
            if "candidates" in data:
                for candidate in data["candidates"]:
                    parts = candidate.get("content", {}).get("parts")  # type: ignore[assignment]
                    if not parts:
                        continue
                    text = "".join(
                        part.get("text", "") for part in parts if isinstance(part, dict)
                    )
                    if text:
                        return self._parse_json_payload(text)
            return self._parse_json_payload(data)

        raise ValueError("LLM response not in a recognized format")

    def _parse_json_payload(self, payload: object) -> list[dict[str, object]]:
        """Parse a payload that may contain JSON in string or dict form."""
        if isinstance(payload, BaseModel):
            return self._parse_json_payload(payload.model_dump())

        if isinstance(payload, list):
            normalized = [
                entry.model_dump() if isinstance(entry, BaseModel) else entry
                for entry in payload
            ]
            return [self._ensure_dict(entry) for entry in normalized]

        if isinstance(payload, dict):
            candidate = payload.get("data") or payload.get("items")
            if candidate:
                return self._parse_json_payload(candidate)
            return [self._ensure_dict(payload)]

        if isinstance(payload, str):
            stripped = payload.strip()
            try:
                data = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError("LLM response is not valid JSON") from exc
            return self._parse_json_payload(data)

        raise ValueError("LLM response is not valid JSON")

    def _ensure_dict(self, entry: object) -> dict[str, object]:
        """Ensure entry is a dictionary."""
        if isinstance(entry, BaseModel):
            return entry.model_dump()
        if isinstance(entry, dict):
            return entry
        raise ValueError("LLM suggestion entry is not a dictionary")

    # ------------------------------------------------------------------ #
    # Serialization helpers
    # ------------------------------------------------------------------ #

    def _serialize_user_profile(self, context: _UserContext) -> dict[str, object]:
        """Serialize user context for the prompt."""
        health_profile = context.user.health_profile
        profile: dict[str, object] = {
            "user_id": context.user.id,
            "allergies": context.allergies,
            "strict_dietary_preferences": context.strict_dietary_preferences,
            "preferred_cuisines": context.preferred_cuisines,
            "health_goals": [
                {
                    "id": goal.id,
                    "type": goal.goal_type,
                    "target_type": goal.target_type,
                    "target_value": float(goal.target_value),
                }
                for goal in context.health_goals
            ],
        }

        if health_profile:
            profile["biometrics"] = {
                "height_cm": self._decimal_to_float(health_profile.height_cm),
                "weight_kg": self._decimal_to_float(health_profile.weight_kg),
                "activity_level": health_profile.activity_level,
            }

        return profile

    def _serialize_menu_item(self, item: MenuItem) -> dict[str, object]:
        """Serialize menu item information for the LLM."""
        return {
            "item_id": str(item.id),
            "name": item.name,
            "restaurant": item.restaurant.name if item.restaurant else None,
            "description": item.description,
            "calories": self._decimal_to_float(item.calories),
            "price": self._decimal_to_float(item.price),
            "cuisine": item.restaurant.cuisine if item.restaurant else None,
        }

    def _serialize_restaurant(
        self,
        restaurant: Restaurant,
        menu_items: Sequence[MenuItem],
    ) -> dict[str, object]:
        """Serialize restaurant information for the LLM."""
        sample_menu = [
            {
                "item_id": str(item.id),
                "name": item.name,
                "description": item.description,
                "calories": self._decimal_to_float(item.calories),
                "price": self._decimal_to_float(item.price),
            }
            for item in menu_items[:10]
        ]

        return {
            "item_id": str(restaurant.id),
            "name": restaurant.name,
            "cuisine": restaurant.cuisine,
            "address": restaurant.address,
            "sample_menu_items": sample_menu,
        }

    # ------------------------------------------------------------------ #
    # Utility helpers
    # ------------------------------------------------------------------ #

    def _decimal_to_float(self, value: Decimal | float | None) -> float | None:
        """Convert Decimal values to float for serialization."""
        if value is None:
            return None
        if isinstance(value, Decimal):
            return float(value)
        return float(value)

    def _price_in_range(
        self,
        price: float | None,
        price_range: str | None,
    ) -> bool:
        """Check if price falls within the selected price range."""
        if price_range is None or price is None:
            return True
        bounds = PRICE_RANGE_MAP.get(price_range)
        if not bounds:
            return True
        lower, upper = bounds
        if lower is not None and price < lower:
            return False
        if upper is not None and price > upper:
            return False
        return True

    def _average_price(self, items: Sequence[MenuItem]) -> float | None:
        """Compute average price for a set of menu items."""
        prices: list[float] = []
        for item in items:
            price_val = self._decimal_to_float(item.price)
            if price_val is not None:
                prices.append(price_val)
        if not prices:
            return None
        return sum(prices) / len(prices)

    def _supports_calorie_goal(
        self,
        goals: Sequence[GoalDB],
        calories: float | None,
    ) -> bool:
        """Return True if the item aligns with a calorie-focused goal."""
        if calories is None:
            return False

        for goal in goals:
            if goal.goal_type != GoalType.NUTRITION.value:
                continue
            target_type = goal.target_type.lower()
            target_value = float(goal.target_value)
            if "calorie" in target_type and calories <= target_value:
                return True
        return False

    def _mentions_goal_keywords(
        self,
        text: str,
        goals: Sequence[GoalDB],
    ) -> bool:
        """Return True if text references keywords from the user's active goals."""
        keywords: list[str] = []
        for goal in goals:
            lower = goal.target_type.lower()
            if "protein" in lower:
                keywords.append("protein")
            if "fiber" in lower:
                keywords.append("fiber")
            if "sodium" in lower:
                keywords.append("low sodium")
        return any(keyword in text for keyword in keywords)

    def _apply_feedback_boosts(
        self,
        recommendations: list[RecommendedItem],
        liked_items: set[str],
    ) -> list[RecommendedItem]:
        """Apply score boosts to items that the user has liked.

        Args:
            recommendations: List of recommended items
            liked_items: Set of item IDs that the user has liked

        Returns:
            List of recommendations with boosted scores for liked items
        """
        if not liked_items:
            return recommendations

        boosted = []
        for rec in recommendations:
            if rec.item_id in liked_items:
                # Boost score by 10% (capped at 1.0)
                new_score = min(1.0, rec.score * 1.1)
                boosted.append(
                    RecommendedItem(
                        item_id=rec.item_id,
                        name=rec.name,
                        score=new_score,
                        explanation=f"{rec.explanation} (You liked this before)",
                    )
                )
            else:
                boosted.append(rec)

        # Re-sort by score after boosting
        boosted.sort(key=lambda r: (-r.score, r.item_id))
        
        # Deduplicate by item_id (keep the one with highest score)
        seen_ids: dict[str, RecommendedItem] = {}
        for rec in boosted:
            if rec.item_id not in seen_ids:
                seen_ids[rec.item_id] = rec
            elif rec.score > seen_ids[rec.item_id].score:
                seen_ids[rec.item_id] = rec
        
        return list(seen_ids.values())
