"""Unit tests for MealService."""

import uuid
from datetime import datetime, timedelta

import pytest
from sqlalchemy.orm import Session

from src.eatsential.models.models import MealFoodItemDB, MealType, UserDB
from src.eatsential.schemas.schemas import (
    MealCreate,
    MealFoodItemCreate,
    MealUpdate,
)
from src.eatsential.services.meal_service import MealService


@pytest.fixture
def test_user(db: Session) -> UserDB:
    """Create a test user."""
    user = UserDB(
        id=str(uuid.uuid4()),
        email="testuser@example.com",
        username="testuser",
        password_hash="hashedpassword123",
        email_verified=True,
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


@pytest.fixture
def test_user_2(db: Session) -> UserDB:
    """Create a second test user for isolation tests."""
    user = UserDB(
        id=str(uuid.uuid4()),
        email="testuser2@example.com",
        username="testuser2",
        password_hash="hashedpassword456",
        email_verified=True,
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


class TestCreateMeal:
    """Tests for MealService.create_meal."""

    def test_create_meal_with_single_food_item(self, db: Session, test_user: UserDB):
        """Test creating a meal with one food item."""
        meal_data = MealCreate(
            meal_type=MealType.BREAKFAST,
            meal_time=datetime.now() - timedelta(hours=2),
            notes="Morning breakfast",
            food_items=[
                MealFoodItemCreate(
                    food_name="Oatmeal",
                    portion_size=1.0,
                    portion_unit="cup",
                    calories=150,
                    protein_g=5.0,
                    carbs_g=27.0,
                    fat_g=3.0,
                )
            ],
        )

        meal = MealService.create_meal(db, test_user.id, meal_data)

        assert meal.id is not None
        assert meal.user_id == test_user.id
        assert meal.meal_type == MealType.BREAKFAST.value
        assert meal.notes == "Morning breakfast"
        assert meal.total_calories == 150
        assert meal.total_protein_g == 5.0
        assert meal.total_carbs_g == 27.0
        assert meal.total_fat_g == 3.0
        assert len(meal.food_items) == 1
        assert meal.food_items[0].food_name == "Oatmeal"

    def test_create_meal_with_multiple_food_items(self, db: Session, test_user: UserDB):
        """Test creating a meal with multiple food items and nutritional calculation."""
        meal_data = MealCreate(
            meal_type=MealType.LUNCH,
            meal_time=datetime.now() - timedelta(hours=1),
            notes="Healthy lunch",
            food_items=[
                MealFoodItemCreate(
                    food_name="Grilled Chicken",
                    portion_size=6.0,
                    portion_unit="oz",
                    calories=280,
                    protein_g=53.0,
                    carbs_g=0.0,
                    fat_g=6.0,
                ),
                MealFoodItemCreate(
                    food_name="Brown Rice",
                    portion_size=1.0,
                    portion_unit="cup",
                    calories=216,
                    protein_g=5.0,
                    carbs_g=45.0,
                    fat_g=1.8,
                ),
                MealFoodItemCreate(
                    food_name="Steamed Broccoli",
                    portion_size=1.0,
                    portion_unit="cup",
                    calories=55,
                    protein_g=3.7,
                    carbs_g=11.0,
                    fat_g=0.6,
                ),
            ],
        )

        meal = MealService.create_meal(db, test_user.id, meal_data)

        assert meal.id is not None
        assert meal.user_id == test_user.id
        assert meal.meal_type == MealType.LUNCH.value
        # Test nutritional totals calculation (with portion size multipliers)
        # Chicken: 280 * 6 = 1680, Rice: 216 * 1 = 216, Broccoli: 55 * 1 = 55, Total = 1951
        assert meal.total_calories == (280 * 6) + (216 * 1) + (55 * 1)
        assert float(meal.total_protein_g) == (53.0 * 6) + (5.0 * 1) + (3.7 * 1)
        assert float(meal.total_carbs_g) == (0.0 * 6) + (45.0 * 1) + (11.0 * 1)
        assert float(meal.total_fat_g) == (6.0 * 6) + (1.8 * 1) + (0.6 * 1)
        assert len(meal.food_items) == 3

    def test_create_meal_with_partial_nutritional_info(
        self, db: Session, test_user: UserDB
    ):
        """Test creating a meal with food items missing nutritional data."""
        meal_data = MealCreate(
            meal_type=MealType.SNACK,
            meal_time=datetime.now() - timedelta(minutes=30),
            food_items=[
                MealFoodItemCreate(
                    food_name="Apple",
                    portion_size=1.0,
                    portion_unit="medium",
                    # No nutritional data provided
                )
            ],
        )

        meal = MealService.create_meal(db, test_user.id, meal_data)

        assert meal.id is not None
        assert meal.total_calories == 0
        assert meal.total_protein_g == 0.0
        assert meal.total_carbs_g == 0.0
        assert meal.total_fat_g == 0.0
        assert len(meal.food_items) == 1


class TestGetMealById:
    """Tests for MealService.get_meal_by_id."""

    def test_get_existing_meal(self, db: Session, test_user: UserDB):
        """Test retrieving an existing meal."""
        meal_data = MealCreate(
            meal_type=MealType.DINNER,
            meal_time=datetime.now() - timedelta(hours=3),
            notes="Test dinner",
            food_items=[
                MealFoodItemCreate(
                    food_name="Salmon",
                    portion_size=4.0,
                    portion_unit="oz",
                    calories=200,
                    protein_g=25.0,
                    carbs_g=0.0,
                    fat_g=11.0,
                )
            ],
        )

        created_meal = MealService.create_meal(db, test_user.id, meal_data)

        # Retrieve the meal
        retrieved_meal = MealService.get_meal_by_id(db, test_user.id, created_meal.id)

        assert retrieved_meal is not None
        assert retrieved_meal.id == created_meal.id
        assert retrieved_meal.user_id == test_user.id
        assert len(retrieved_meal.food_items) == 1

    def test_get_nonexistent_meal(self, db: Session, test_user: UserDB):
        """Test retrieving a non-existent meal returns None."""
        fake_meal_id = str(uuid.uuid4())

        meal = MealService.get_meal_by_id(db, test_user.id, fake_meal_id)

        assert meal is None

    def test_get_meal_with_wrong_user_returns_none(
        self, db: Session, test_user: UserDB, test_user_2: UserDB
    ):
        """Test that users can't access other users' meals."""
        meal_data = MealCreate(
            meal_type=MealType.BREAKFAST,
            meal_time=datetime.now() - timedelta(hours=1),
            food_items=[
                MealFoodItemCreate(
                    food_name="Toast",
                    portion_size=2.0,
                    portion_unit="slices",
                    calories=160,
                    protein_g=6.0,
                    carbs_g=30.0,
                    fat_g=2.0,
                )
            ],
        )

        created_meal = MealService.create_meal(db, test_user.id, meal_data)

        # Try to access with different user
        meal = MealService.get_meal_by_id(db, test_user_2.id, created_meal.id)

        assert meal is None


class TestGetUserMeals:
    """Tests for MealService.get_user_meals."""

    def test_get_user_meals_pagination(self, db: Session, test_user: UserDB):
        """Test pagination works correctly."""
        # Create 5 meals
        for i in range(5):
            meal_data = MealCreate(
                meal_type=MealType.SNACK,
                meal_time=datetime.now() - timedelta(hours=i),
                notes=f"Meal {i}",
                food_items=[
                    MealFoodItemCreate(
                        food_name=f"Food {i}",
                        portion_size=1.0,
                        portion_unit="serving",
                        calories=100,
                    )
                ],
            )
            MealService.create_meal(db, test_user.id, meal_data)

        # Get first page
        meals, total = MealService.get_user_meals(db, test_user.id, skip=0, limit=3)

        assert total == 5
        assert len(meals) == 3

        # Get second page
        meals, total = MealService.get_user_meals(db, test_user.id, skip=3, limit=3)

        assert total == 5
        assert len(meals) == 2

    def test_get_user_meals_filter_by_meal_type(self, db: Session, test_user: UserDB):
        """Test filtering meals by meal type."""
        # Create meals of different types
        meal_types = [
            MealType.BREAKFAST,
            MealType.LUNCH,
            MealType.DINNER,
            MealType.SNACK,
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

        # Filter for BREAKFAST only
        meals, total = MealService.get_user_meals(
            db, test_user.id, meal_type=MealType.BREAKFAST.value
        )

        assert total == 2
        assert all(meal.meal_type == MealType.BREAKFAST.value for meal in meals)

    def test_get_user_meals_filter_by_date_range(self, db: Session, test_user: UserDB):
        """Test filtering meals by date range."""
        now = datetime.now()

        # Create meals at different times
        meal_times = [
            now - timedelta(days=5),
            now - timedelta(days=3),
            now - timedelta(days=1),
            now - timedelta(hours=2),
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
        start_date = now - timedelta(days=2)
        meals, total = MealService.get_user_meals(
            db, test_user.id, start_date=start_date
        )

        assert total == 2
        assert all(meal.meal_time >= start_date for meal in meals)

        # Filter for 3-5 days ago
        start_date = now - timedelta(days=6)
        end_date = now - timedelta(days=2)
        meals, total = MealService.get_user_meals(
            db, test_user.id, start_date=start_date, end_date=end_date
        )

        assert total == 2

    def test_get_user_meals_user_isolation(
        self, db: Session, test_user: UserDB, test_user_2: UserDB
    ):
        """Test that users only see their own meals."""
        # Create meals for both users
        for _ in range(3):
            meal_data = MealCreate(
                meal_type=MealType.LUNCH,
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
            MealService.create_meal(db, test_user_2.id, meal_data)

        # Get meals for user 1
        meals_user_1, total_user_1 = MealService.get_user_meals(db, test_user.id)

        # Get meals for user 2
        meals_user_2, total_user_2 = MealService.get_user_meals(db, test_user_2.id)

        assert total_user_1 == 3
        assert total_user_2 == 3
        assert all(meal.user_id == test_user.id for meal in meals_user_1)
        assert all(meal.user_id == test_user_2.id for meal in meals_user_2)


class TestUpdateMeal:
    """Tests for MealService.update_meal."""

    def test_update_meal_partial_fields(self, db: Session, test_user: UserDB):
        """Test updating only some fields."""
        meal_data = MealCreate(
            meal_type=MealType.BREAKFAST,
            meal_time=datetime.now() - timedelta(hours=2),
            notes="Original notes",
            food_items=[
                MealFoodItemCreate(
                    food_name="Cereal",
                    portion_size=1.0,
                    portion_unit="cup",
                    calories=110,
                )
            ],
        )

        created_meal = MealService.create_meal(db, test_user.id, meal_data)

        # Update only notes
        update_data = MealUpdate(notes="Updated notes")

        updated_meal = MealService.update_meal(
            db, test_user.id, created_meal.id, update_data
        )

        assert updated_meal is not None
        assert updated_meal.notes == "Updated notes"
        assert updated_meal.meal_type == MealType.BREAKFAST.value
        assert len(updated_meal.food_items) == 1

    def test_update_meal_replace_food_items(self, db: Session, test_user: UserDB):
        """Test replacing food items and recalculating nutritional totals."""
        meal_data = MealCreate(
            meal_type=MealType.LUNCH,
            meal_time=datetime.now() - timedelta(hours=3),
            food_items=[
                MealFoodItemCreate(
                    food_name="Pizza",
                    portion_size=2.0,
                    portion_unit="slices",
                    calories=570,
                    protein_g=24.0,
                    carbs_g=68.0,
                    fat_g=22.0,
                )
            ],
        )

        created_meal = MealService.create_meal(db, test_user.id, meal_data)

        # Replace with healthier food
        update_data = MealUpdate(
            food_items=[
                MealFoodItemCreate(
                    food_name="Salad",
                    portion_size=2.0,
                    portion_unit="cups",
                    calories=150,
                    protein_g=8.0,
                    carbs_g=15.0,
                    fat_g=9.0,
                ),
                MealFoodItemCreate(
                    food_name="Grilled Chicken",
                    portion_size=4.0,
                    portion_unit="oz",
                    calories=187,
                    protein_g=35.0,
                    carbs_g=0.0,
                    fat_g=4.0,
                ),
            ]
        )

        updated_meal = MealService.update_meal(
            db, test_user.id, created_meal.id, update_data
        )

        assert updated_meal is not None
        assert len(updated_meal.food_items) == 2
        # Check recalculated totals (with portion size multipliers)
        # Salad: 150 * 2 = 300, Chicken: 187 * 4 = 748, Total = 1048
        assert updated_meal.total_calories == (150 * 2) + (187 * 4)
        assert updated_meal.total_protein_g == (8.0 * 2) + (35.0 * 4)
        assert updated_meal.total_carbs_g == (15.0 * 2) + (0.0 * 4)
        assert updated_meal.total_fat_g == (9.0 * 2) + (4.0 * 4)

    def test_update_nonexistent_meal(self, db: Session, test_user: UserDB):
        """Test updating a non-existent meal returns None."""
        fake_meal_id = str(uuid.uuid4())
        update_data = MealUpdate(notes="New notes")

        updated_meal = MealService.update_meal(
            db, test_user.id, fake_meal_id, update_data
        )

        assert updated_meal is None

    def test_update_meal_with_wrong_user_returns_none(
        self, db: Session, test_user: UserDB, test_user_2: UserDB
    ):
        """Test that users can't update other users' meals."""
        meal_data = MealCreate(
            meal_type=MealType.DINNER,
            meal_time=datetime.now() - timedelta(hours=4),
            food_items=[
                MealFoodItemCreate(
                    food_name="Steak",
                    portion_size=8.0,
                    portion_unit="oz",
                    calories=614,
                )
            ],
        )

        created_meal = MealService.create_meal(db, test_user.id, meal_data)

        # Try to update with different user
        update_data = MealUpdate(notes="Hacked notes")

        updated_meal = MealService.update_meal(
            db, test_user_2.id, created_meal.id, update_data
        )

        assert updated_meal is None


class TestDeleteMeal:
    """Tests for MealService.delete_meal."""

    def test_delete_existing_meal(self, db: Session, test_user: UserDB):
        """Test deleting an existing meal."""
        meal_data = MealCreate(
            meal_type=MealType.SNACK,
            meal_time=datetime.now() - timedelta(hours=1),
            food_items=[
                MealFoodItemCreate(
                    food_name="Yogurt",
                    portion_size=6.0,
                    portion_unit="oz",
                    calories=150,
                )
            ],
        )

        created_meal = MealService.create_meal(db, test_user.id, meal_data)

        # Delete the meal
        success = MealService.delete_meal(db, test_user.id, created_meal.id)

        assert success is True

        # Verify it's gone
        deleted_meal = MealService.get_meal_by_id(db, test_user.id, created_meal.id)
        assert deleted_meal is None

    def test_delete_meal_cascades_to_food_items(self, db: Session, test_user: UserDB):
        """Test that deleting a meal also deletes its food items (CASCADE)."""
        meal_data = MealCreate(
            meal_type=MealType.DINNER,
            meal_time=datetime.now() - timedelta(hours=5),
            food_items=[
                MealFoodItemCreate(
                    food_name="Food 1",
                    portion_size=1.0,
                    portion_unit="serving",
                ),
                MealFoodItemCreate(
                    food_name="Food 2",
                    portion_size=1.0,
                    portion_unit="serving",
                ),
            ],
        )

        created_meal = MealService.create_meal(db, test_user.id, meal_data)

        # Count food items before delete
        food_items_before = (
            db.query(MealFoodItemDB)
            .filter(MealFoodItemDB.meal_id == created_meal.id)
            .count()
        )
        assert food_items_before == 2

        # Delete the meal
        MealService.delete_meal(db, test_user.id, created_meal.id)

        # Verify food items are also deleted
        food_items_after = (
            db.query(MealFoodItemDB)
            .filter(MealFoodItemDB.meal_id == created_meal.id)
            .count()
        )
        assert food_items_after == 0

    def test_delete_nonexistent_meal(self, db: Session, test_user: UserDB):
        """Test deleting a non-existent meal returns False."""
        fake_meal_id = str(uuid.uuid4())

        success = MealService.delete_meal(db, test_user.id, fake_meal_id)

        assert success is False

    def test_delete_meal_with_wrong_user_returns_false(
        self, db: Session, test_user: UserDB, test_user_2: UserDB
    ):
        """Test that users can't delete other users' meals."""
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

        # Try to delete with different user
        success = MealService.delete_meal(db, test_user_2.id, created_meal.id)

        assert success is False

        # Verify meal still exists
        meal = MealService.get_meal_by_id(db, test_user.id, created_meal.id)
        assert meal is not None
