"""Integration tests for Meal Logging API endpoints."""

from datetime import datetime, timedelta

from fastapi import status
from fastapi.testclient import TestClient

from src.eatsential.models.models import MealType
from src.eatsential.schemas.schemas import MealCreate, MealFoodItemCreate
from src.eatsential.services.meal_service import MealService


class TestCreateMealEndpoint:
    """Tests for POST /api/meals endpoint."""

    def test_create_meal_success(self, client: TestClient, auth_headers: dict, db):
        """Test successful meal creation."""
        meal_data = {
            "meal_type": MealType.BREAKFAST.value,
            "meal_time": (datetime.now() - timedelta(hours=2)).isoformat(),
            "notes": "Healthy breakfast",
            "food_items": [
                {
                    "food_name": "Oatmeal",
                    "portion_size": 1.0,
                    "portion_unit": "cup",
                    "calories": 150,
                    "protein_g": 5.0,
                    "carbs_g": 27.0,
                    "fat_g": 3.0,
                }
            ],
        }

        response = client.post("/api/meals", json=meal_data, headers=auth_headers)

        assert response.status_code == status.HTTP_201_CREATED
        data = response.json()
        assert data["meal_type"] == MealType.BREAKFAST.value
        assert data["notes"] == "Healthy breakfast"
        assert data["total_calories"] == 150
        assert len(data["food_items"]) == 1

    def test_create_meal_requires_authentication(self, client: TestClient):
        """Test that creating meal requires authentication."""
        meal_data = {
            "meal_type": MealType.LUNCH.value,
            "meal_time": datetime.now().isoformat(),
            "food_items": [
                {
                    "food_name": "Sandwich",
                    "portion_size": 1.0,
                    "portion_unit": "whole",
                }
            ],
        }

        response = client.post("/api/meals", json=meal_data)

        assert response.status_code == status.HTTP_403_FORBIDDEN

    def test_create_meal_allows_future_meal_times(
        self, client: TestClient, auth_headers: dict
    ):
        """Test that meal_time can be in the future (within 30 days)."""
        meal_data = {
            "meal_type": MealType.DINNER.value,
            "meal_time": (datetime.now() + timedelta(hours=1)).isoformat(),
            "food_items": [
                {
                    "food_name": "Pasta",
                    "portion_size": 2.0,
                    "portion_unit": "cups",
                }
            ],
        }

        response = client.post("/api/meals", json=meal_data, headers=auth_headers)

        assert response.status_code == status.HTTP_201_CREATED
        data = response.json()
        assert data["meal_type"] == MealType.DINNER.value

    def test_create_meal_validates_meal_time_within_30_days(
        self, client: TestClient, auth_headers: dict
    ):
        """Test that meal_time must be within 30 days (past or future)."""
        meal_data = {
            "meal_type": MealType.SNACK.value,
            "meal_time": (datetime.now() - timedelta(days=31)).isoformat(),
            "food_items": [
                {
                    "food_name": "Nuts",
                    "portion_size": 1.0,
                    "portion_unit": "oz",
                }
            ],
        }

        response = client.post("/api/meals", json=meal_data, headers=auth_headers)

        assert response.status_code == status.HTTP_422_UNPROCESSABLE_CONTENT
        assert "30 days" in response.json()["detail"][0]["msg"].lower()

    def test_create_meal_validates_meal_time_not_beyond_30_days_future(
        self, client: TestClient, auth_headers: dict
    ):
        """Test that meal_time cannot be more than 30 days in the future."""
        meal_data = {
            "meal_type": MealType.SNACK.value,
            "meal_time": (datetime.now() + timedelta(days=31)).isoformat(),
            "food_items": [
                {
                    "food_name": "Nuts",
                    "portion_size": 1.0,
                    "portion_unit": "oz",
                }
            ],
        }

        response = client.post("/api/meals", json=meal_data, headers=auth_headers)

        assert response.status_code == status.HTTP_422_UNPROCESSABLE_CONTENT
        assert "30 days" in response.json()["detail"][0]["msg"].lower()

    def test_create_meal_requires_at_least_one_food_item(
        self, client: TestClient, auth_headers: dict
    ):
        """Test that at least one food item is required."""
        meal_data = {
            "meal_type": MealType.BREAKFAST.value,
            "meal_time": datetime.now().isoformat(),
            "food_items": [],  # Empty list
        }

        response = client.post("/api/meals", json=meal_data, headers=auth_headers)

        assert response.status_code == status.HTTP_422_UNPROCESSABLE_CONTENT


class TestGetMealsEndpoint:
    """Tests for GET /api/meals endpoint."""

    def test_get_meals_success(
        self, client: TestClient, auth_headers: dict, test_user, db
    ):
        """Test getting list of meals."""
        # Create test meals
        for i in range(3):
            meal_data = MealCreate(
                meal_type=MealType.LUNCH,
                meal_time=datetime.now() - timedelta(hours=i),
                food_items=[
                    MealFoodItemCreate(
                        food_name=f"Food {i}",
                        portion_size=1.0,
                        portion_unit="serving",
                    )
                ],
            )
            MealService.create_meal(db, test_user.id, meal_data)

        response = client.get("/api/meals", headers=auth_headers)

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "meals" in data
        assert "total" in data
        assert data["total"] == 3
        assert len(data["meals"]) == 3

    def test_get_meals_pagination(
        self, client: TestClient, auth_headers: dict, test_user, db
    ):
        """Test meal list pagination."""
        # Create 5 meals
        for i in range(5):
            meal_data = MealCreate(
                meal_type=MealType.SNACK,
                meal_time=datetime.now() - timedelta(hours=i),
                food_items=[
                    MealFoodItemCreate(
                        food_name=f"Snack {i}",
                        portion_size=1.0,
                        portion_unit="serving",
                    )
                ],
            )
            MealService.create_meal(db, test_user.id, meal_data)

        # Get page 1 with page_size=2
        response = client.get("/api/meals?page=1&page_size=2", headers=auth_headers)

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["total"] == 5
        assert len(data["meals"]) == 2
        assert data["page"] == 1
        assert data["page_size"] == 2

        # Get page 2
        response = client.get("/api/meals?page=2&page_size=2", headers=auth_headers)

        data = response.json()
        assert len(data["meals"]) == 2

    def test_get_meals_filter_by_meal_type(
        self, client: TestClient, auth_headers: dict, test_user, db
    ):
        """Test filtering meals by meal type."""
        # Create different meal types
        meal_types = [
            MealType.BREAKFAST,
            MealType.LUNCH,
            MealType.DINNER,
            MealType.BREAKFAST,
        ]

        for meal_type in meal_types:
            meal_data = MealCreate(
                meal_type=meal_type,
                meal_time=datetime.now() - timedelta(hours=1),
                food_items=[
                    MealFoodItemCreate(
                        food_name="Food",
                        portion_size=1.0,
                        portion_unit="serving",
                    )
                ],
            )
            MealService.create_meal(db, test_user.id, meal_data)

        # Filter for BREAKFAST
        response = client.get(
            f"/api/meals?meal_type={MealType.BREAKFAST.value}",
            headers=auth_headers,
        )

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["total"] == 2
        assert all(
            meal["meal_type"] == MealType.BREAKFAST.value for meal in data["meals"]
        )

    def test_get_meals_filter_by_date_range(
        self, client: TestClient, auth_headers: dict, test_user, db
    ):
        """Test filtering meals by date range."""
        now = datetime.now()

        # Create meals at different times
        meal_times = [
            now - timedelta(days=5),
            now - timedelta(days=3),
            now - timedelta(days=1),
        ]

        for meal_time in meal_times:
            meal_data = MealCreate(
                meal_type=MealType.LUNCH,
                meal_time=meal_time,
                food_items=[
                    MealFoodItemCreate(
                        food_name="Food",
                        portion_size=1.0,
                        portion_unit="serving",
                    )
                ],
            )
            MealService.create_meal(db, test_user.id, meal_data)

        # Filter for last 2 days
        start_date = (now - timedelta(days=2)).isoformat()
        response = client.get(
            f"/api/meals?start_date={start_date}",
            headers=auth_headers,
        )

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["total"] == 1

    def test_get_meals_requires_authentication(self, client: TestClient):
        """Test that getting meals requires authentication."""
        response = client.get("/api/meals")

        assert response.status_code == status.HTTP_403_FORBIDDEN

    def test_get_meals_user_isolation(
        self, client: TestClient, auth_headers: dict, test_user, test_user_2, db
    ):
        """Test that users only see their own meals."""
        # Create meals for test_user
        meal_data = MealCreate(
            meal_type=MealType.DINNER,
            meal_time=datetime.now() - timedelta(hours=2),
            food_items=[
                MealFoodItemCreate(
                    food_name="Food",
                    portion_size=1.0,
                    portion_unit="serving",
                )
            ],
        )
        MealService.create_meal(db, test_user.id, meal_data)
        MealService.create_meal(db, test_user_2.id, meal_data)

        # Get meals for authenticated user (test_user)
        response = client.get("/api/meals", headers=auth_headers)

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        # Should only see test_user's meals
        assert data["total"] == 1


class TestGetMealEndpoint:
    """Tests for GET /api/meals/{meal_id} endpoint."""

    def test_get_meal_success(
        self, client: TestClient, auth_headers: dict, test_user, db
    ):
        """Test getting a specific meal."""
        meal_data = MealCreate(
            meal_type=MealType.BREAKFAST,
            meal_time=datetime.now() - timedelta(hours=2),
            notes="Test meal",
            food_items=[
                MealFoodItemCreate(
                    food_name="Eggs",
                    portion_size=2.0,
                    portion_unit="large",
                    calories=143,
                    protein_g=12.6,
                    carbs_g=0.7,
                    fat_g=9.5,
                )
            ],
        )

        created_meal = MealService.create_meal(db, test_user.id, meal_data)

        response = client.get(f"/api/meals/{created_meal.id}", headers=auth_headers)

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["id"] == created_meal.id
        assert data["notes"] == "Test meal"
        assert len(data["food_items"]) == 1

    def test_get_meal_not_found(self, client: TestClient, auth_headers: dict):
        """Test getting non-existent meal returns 404."""
        import uuid

        fake_id = str(uuid.uuid4())

        response = client.get(f"/api/meals/{fake_id}", headers=auth_headers)

        assert response.status_code == status.HTTP_404_NOT_FOUND

    def test_get_meal_requires_authentication(self, client: TestClient, test_user, db):
        """Test that getting specific meal requires authentication."""
        meal_data = MealCreate(
            meal_type=MealType.LUNCH,
            meal_time=datetime.now() - timedelta(hours=1),
            food_items=[
                MealFoodItemCreate(
                    food_name="Sandwich",
                    portion_size=1.0,
                    portion_unit="whole",
                )
            ],
        )

        created_meal = MealService.create_meal(db, test_user.id, meal_data)

        response = client.get(f"/api/meals/{created_meal.id}")

        assert response.status_code == status.HTTP_403_FORBIDDEN

    def test_get_meal_user_isolation(
        self, client: TestClient, auth_headers: dict, test_user_2, db
    ):
        """Test users can't access other users' meals."""
        # Create meal for test_user_2
        meal_data = MealCreate(
            meal_type=MealType.DINNER,
            meal_time=datetime.now() - timedelta(hours=3),
            food_items=[
                MealFoodItemCreate(
                    food_name="Steak",
                    portion_size=8.0,
                    portion_unit="oz",
                )
            ],
        )

        other_user_meal = MealService.create_meal(db, test_user_2.id, meal_data)

        # Try to access with test_user's auth
        response = client.get(f"/api/meals/{other_user_meal.id}", headers=auth_headers)

        assert response.status_code == status.HTTP_404_NOT_FOUND


class TestUpdateMealEndpoint:
    """Tests for PUT /api/meals/{meal_id} endpoint."""

    def test_update_meal_success(
        self, client: TestClient, auth_headers: dict, test_user, db
    ):
        """Test successful meal update."""
        meal_data = MealCreate(
            meal_type=MealType.LUNCH,
            meal_time=datetime.now() - timedelta(hours=2),
            notes="Original notes",
            food_items=[
                MealFoodItemCreate(
                    food_name="Salad",
                    portion_size=2.0,
                    portion_unit="cups",
                    calories=100,
                )
            ],
        )

        created_meal = MealService.create_meal(db, test_user.id, meal_data)

        # Update meal
        update_data = {
            "notes": "Updated notes",
            "meal_type": MealType.DINNER.value,
        }

        response = client.put(
            f"/api/meals/{created_meal.id}", json=update_data, headers=auth_headers
        )

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["notes"] == "Updated notes"
        assert data["meal_type"] == MealType.DINNER.value

    def test_update_meal_replace_food_items(
        self, client: TestClient, auth_headers: dict, test_user, db
    ):
        """Test replacing food items and recalculating nutritional totals."""
        meal_data = MealCreate(
            meal_type=MealType.BREAKFAST,
            meal_time=datetime.now() - timedelta(hours=2),
            food_items=[
                MealFoodItemCreate(
                    food_name="Pancakes",
                    portion_size=3.0,
                    portion_unit="pieces",
                    calories=350,
                )
            ],
        )

        created_meal = MealService.create_meal(db, test_user.id, meal_data)

        # Replace with different food
        update_data = {
            "food_items": [
                {
                    "food_name": "Oatmeal",
                    "portion_size": 1.0,
                    "portion_unit": "cup",
                    "calories": 150,
                    "protein_g": 5.0,
                    "carbs_g": 27.0,
                    "fat_g": 3.0,
                }
            ]
        }

        response = client.put(
            f"/api/meals/{created_meal.id}", json=update_data, headers=auth_headers
        )

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert len(data["food_items"]) == 1
        assert data["food_items"][0]["food_name"] == "Oatmeal"
        assert data["total_calories"] == 150

    def test_update_meal_not_found(self, client: TestClient, auth_headers: dict):
        """Test updating non-existent meal returns 404."""
        import uuid

        fake_id = str(uuid.uuid4())
        update_data = {"notes": "New notes"}

        response = client.put(
            f"/api/meals/{fake_id}", json=update_data, headers=auth_headers
        )

        assert response.status_code == status.HTTP_404_NOT_FOUND

    def test_update_meal_requires_authentication(
        self, client: TestClient, test_user, db
    ):
        """Test that updating meal requires authentication."""
        meal_data = MealCreate(
            meal_type=MealType.SNACK,
            meal_time=datetime.now() - timedelta(hours=1),
            food_items=[
                MealFoodItemCreate(
                    food_name="Apple",
                    portion_size=1.0,
                    portion_unit="medium",
                )
            ],
        )

        created_meal = MealService.create_meal(db, test_user.id, meal_data)

        update_data = {"notes": "Hacked"}

        response = client.put(f"/api/meals/{created_meal.id}", json=update_data)

        assert response.status_code == status.HTTP_403_FORBIDDEN


class TestDeleteMealEndpoint:
    """Tests for DELETE /api/meals/{meal_id} endpoint."""

    def test_delete_meal_success(
        self, client: TestClient, auth_headers: dict, test_user, db
    ):
        """Test successful meal deletion."""
        meal_data = MealCreate(
            meal_type=MealType.DINNER,
            meal_time=datetime.now() - timedelta(hours=4),
            food_items=[
                MealFoodItemCreate(
                    food_name="Burger",
                    portion_size=1.0,
                    portion_unit="whole",
                    calories=500,
                )
            ],
        )

        created_meal = MealService.create_meal(db, test_user.id, meal_data)

        response = client.delete(f"/api/meals/{created_meal.id}", headers=auth_headers)

        assert response.status_code == status.HTTP_204_NO_CONTENT

        # Verify meal is deleted
        get_response = client.get(f"/api/meals/{created_meal.id}", headers=auth_headers)
        assert get_response.status_code == status.HTTP_404_NOT_FOUND

    def test_delete_meal_not_found(self, client: TestClient, auth_headers: dict):
        """Test deleting non-existent meal returns 404."""
        import uuid

        fake_id = str(uuid.uuid4())

        response = client.delete(f"/api/meals/{fake_id}", headers=auth_headers)

        assert response.status_code == status.HTTP_404_NOT_FOUND

    def test_delete_meal_requires_authentication(
        self, client: TestClient, test_user, db
    ):
        """Test that deleting meal requires authentication."""
        meal_data = MealCreate(
            meal_type=MealType.SNACK,
            meal_time=datetime.now() - timedelta(hours=1),
            food_items=[
                MealFoodItemCreate(
                    food_name="Chips",
                    portion_size=1.0,
                    portion_unit="bag",
                )
            ],
        )

        created_meal = MealService.create_meal(db, test_user.id, meal_data)

        response = client.delete(f"/api/meals/{created_meal.id}")

        assert response.status_code == status.HTTP_403_FORBIDDEN

    def test_delete_meal_user_isolation(
        self, client: TestClient, auth_headers: dict, test_user_2, db
    ):
        """Test users can't delete other users' meals."""
        # Create meal for test_user_2
        meal_data = MealCreate(
            meal_type=MealType.LUNCH,
            meal_time=datetime.now() - timedelta(hours=2),
            food_items=[
                MealFoodItemCreate(
                    food_name="Pizza",
                    portion_size=2.0,
                    portion_unit="slices",
                )
            ],
        )

        other_user_meal = MealService.create_meal(db, test_user_2.id, meal_data)

        # Try to delete with test_user's auth
        response = client.delete(
            f"/api/meals/{other_user_meal.id}", headers=auth_headers
        )

        assert response.status_code == status.HTTP_404_NOT_FOUND

        # Verify meal still exists
        meal = MealService.get_meal_by_id(db, test_user_2.id, other_user_meal.id)
        assert meal is not None
