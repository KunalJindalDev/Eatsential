"""Integration tests for restaurant recommendation diversity."""

import pytest
from sqlalchemy.orm import Session

from src.eatsential.models.models import UserDB, Restaurant, MenuItem
from src.eatsential.utils.auth_util import create_access_token


@pytest.fixture
def test_user(db: Session) -> UserDB:
    """Create a test user."""
    user = UserDB(
        id="diversity_test_user",
        email="diversity@test.com",
        username="diversity_user",
        password_hash="hashed_password",  # Static hash since we create JWT directly
        email_verified=True,
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


@pytest.fixture
def auth_headers(test_user: UserDB) -> dict[str, str]:
    """Get authentication headers."""
    access_token = create_access_token(data={"sub": test_user.id})
    return {"Authorization": f"Bearer {access_token}"}


@pytest.fixture
def test_restaurants(db: Session):
    """Create test restaurants with menu items."""
    restaurants_data = [
        {"name": "Luigi's Pizza", "cuisine": "Italian", "place_id": "place_italian_1"},
        {"name": "Roma Trattoria", "cuisine": "Italian", "place_id": "place_italian_2"},
        {"name": "Venezia Cafe", "cuisine": "Italian", "place_id": "place_italian_3"},
        {"name": "El Taco Loco", "cuisine": "Mexican", "place_id": "place_mexican_1"},
        {"name": "Casa Mexico", "cuisine": "Mexican", "place_id": "place_mexican_2"},
    ]

    for r_data in restaurants_data:
        restaurant = Restaurant(
            id=r_data["place_id"],
            name=r_data["name"],
            address=f"{r_data['name']} Address",
            cuisine=r_data["cuisine"],
        )
        db.add(restaurant)
        
        # Add menu items
        for i in range(4):
            item = MenuItem(
                id=f"{r_data['place_id']}_item_{i}",
                name=f"{r_data['cuisine']} Dish {i}",
                restaurant_id=r_data["place_id"],
                calories=500.0 + i * 50.0,
                price=12.99 + i * 2.0,
            )
            db.add(item)
    
    db.commit()
    return restaurants_data


class TestRecommendationDiversityByCuisine:
    """Test diversity filtering across different cuisines."""

    def test_italian_recommendations_diversity_baseline(
        self, client, auth_headers, test_restaurants
    ):
        """Test Italian cuisine recommendations have diverse restaurants (baseline mode)."""
        response = client.post(
            "/api/recommend/meal",
            json={
                "filters": {"cuisine": ["Italian"]},
                "mode": "baseline"
            },
            headers=auth_headers,
        )

        assert response.status_code == 200
        data = response.json()

        # Should have recommendations
        recommendations = data.get("items", [])
        assert len(recommendations) > 0

        # Count unique restaurants
        place_ids = [r["restaurant_place_id"] for r in recommendations]
        unique_restaurants = len(set(place_ids))

        # Should have diversity (multiple restaurants)
        assert unique_restaurants >= 2, f"Expected diverse restaurants but got {unique_restaurants}"

    def test_mexican_recommendations_diversity_llm(self, client, auth_headers, test_restaurants):
        """Test Mexican cuisine recommendations with LLM mode."""
        response = client.post(
            "/api/recommend/meal",
            json={
                "filters": {"cuisine": ["Mexican"]},
                "mode": "llm"
            },
            headers=auth_headers,
        )

        assert response.status_code == 200
        data = response.json()

        recommendations = data.get("items", [])
        assert len(recommendations) > 0

        # Verify diversity even in LLM mode
        place_ids = [r["restaurant_place_id"] for r in recommendations]
        unique_restaurants = len(set(place_ids))
        assert unique_restaurants >= 2

    def test_recommendations_with_multiple_cuisines(self, client, auth_headers, test_restaurants):
        """Test that different cuisines return diverse results."""
        italian_response = client.post(
            "/api/recommend/meal",
            json={"filters": {"cuisine": ["Italian"]}, "mode": "baseline"},
            headers=auth_headers,
        )
        mexican_response = client.post(
            "/api/recommend/meal",
            json={"filters": {"cuisine": ["Mexican"]}, "mode": "baseline"},
            headers=auth_headers,
        )

        assert italian_response.status_code == 200
        assert mexican_response.status_code == 200

        italian_data = italian_response.json().get("items", [])
        mexican_data = mexican_response.json().get("items", [])

        # Both should have recommendations
        assert len(italian_data) > 0
        assert len(mexican_data) > 0

        # Italian should have different place_ids than Mexican
        italian_places = {r["restaurant_place_id"] for r in italian_data}
        mexican_places = {r["restaurant_place_id"] for r in mexican_data}
        
        # Should be no overlap (different cuisines)
        assert len(italian_places & mexican_places) == 0

    def test_baseline_vs_llm_both_have_diversity(self, client, auth_headers, test_restaurants):
        """Test that both baseline and LLM modes produce diverse results."""
        baseline_response = client.post(
            "/api/recommend/meal",
            json={"filters": {"cuisine": ["Italian"]}, "mode": "baseline"},
            headers=auth_headers,
        )
        llm_response = client.post(
            "/api/recommend/meal",
            json={"filters": {"cuisine": ["Italian"]}, "mode": "llm"},
            headers=auth_headers,
        )

        assert baseline_response.status_code == 200
        assert llm_response.status_code == 200

        baseline_data = baseline_response.json().get("items", [])
        llm_data = llm_response.json().get("items", [])

        # Both should have diverse restaurants
        baseline_unique = len(set(r["restaurant_place_id"] for r in baseline_data))
        llm_unique = len(set(r["restaurant_place_id"] for r in llm_data))

        assert baseline_unique >= 2, "Baseline mode should have diverse restaurants"
        assert llm_unique >= 2, "LLM mode should have diverse restaurants"


class TestRecommendationResponseQuality:
    """Test that diversity filtering maintains response quality."""

    def test_all_recommendations_have_restaurant_data(
        self, client, auth_headers, test_restaurants
    ):
        """Test that all recommendations include complete restaurant data."""
        response = client.post(
            "/api/recommend/meal",
            json={"filters": {"cuisine": ["Italian"]}, "mode": "baseline"},
            headers=auth_headers,
        )

        assert response.status_code == 200
        data = response.json()

        recommendations = data.get("items", [])
        assert len(recommendations) > 0

        # Every recommendation should have restaurant data
        for rec in recommendations:
            assert "restaurant_name" in rec
            assert "restaurant_address" in rec
            assert "restaurant_place_id" in rec
            assert rec["restaurant_name"] is not None
            assert rec["restaurant_place_id"] is not None

    def test_recommendations_sorted_by_score(self, client, auth_headers, test_restaurants):
        """Test that recommendations maintain score ordering despite diversity."""
        response = client.post(
            "/api/recommend/meal",
            json={"filters": {"cuisine": ["Italian"]}, "mode": "baseline"},
            headers=auth_headers,
        )

        assert response.status_code == 200
        data = response.json()

        recommendations = data.get("items", [])
        if len(recommendations) > 1:
            # Scores should be in descending order
            scores = [r.get("score", 0) for r in recommendations]
            assert scores == sorted(scores, reverse=True), "Scores should be sorted descending"

    def test_diversity_does_not_exclude_high_quality_items(
        self, client, auth_headers, test_restaurants
    ):
        """Test that diversity filtering doesn't exclude high-scoring items."""
        response = client.post(
            "/api/recommend/meal",
            json={"filters": {"cuisine": ["Italian"]}, "mode": "baseline"},
            headers=auth_headers,
        )

        assert response.status_code == 200
        data = response.json()

        recommendations = data.get("items", [])
        assert len(recommendations) > 0

        # Should have items with decent scores (> 0)
        scores = [r.get("score", 0) for r in recommendations]
        assert all(s > 0 for s in scores), "All recommendations should have positive scores"
