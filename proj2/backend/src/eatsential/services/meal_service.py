"""Meal logging service for CRUD operations."""

import uuid
from datetime import datetime
from typing import Optional

from sqlalchemy import and_, desc
from sqlalchemy.orm import Session, selectinload

from ..models.models import MealDB, MealFoodItemDB
from ..schemas.schemas import MealCreate, MealUpdate


class MealService:
    """Service class for meal logging operations"""

    @staticmethod
    def create_meal(db: Session, user_id: str, meal_data: MealCreate) -> MealDB:
        """Create a new meal log with food items.

        Args:
            db: Database session
            user_id: User ID
            meal_data: Meal creation data

        Returns:
            Created meal database object

        """
        # Calculate nutritional totals
        total_calories = sum(
            (item.calories or 0) * item.portion_size for item in meal_data.food_items
        )
        total_protein = sum(
            (item.protein_g or 0) * item.portion_size for item in meal_data.food_items
        )
        total_carbs = sum(
            (item.carbs_g or 0) * item.portion_size for item in meal_data.food_items
        )
        total_fat = sum(
            (item.fat_g or 0) * item.portion_size for item in meal_data.food_items
        )

        # Create meal record
        db_meal = MealDB(
            id=str(uuid.uuid4()),
            user_id=user_id,
            meal_type=meal_data.meal_type.value,
            meal_time=meal_data.meal_time,
            notes=meal_data.notes,
            photo_url=meal_data.photo_url,
            total_calories=total_calories or 0,
            total_protein_g=total_protein or 0,
            total_carbs_g=total_carbs or 0,
            total_fat_g=total_fat or 0,
        )

        # Create food items
        for food_item_data in meal_data.food_items:
            db_food_item = MealFoodItemDB(
                id=str(uuid.uuid4()),
                meal_id=db_meal.id,
                food_name=food_item_data.food_name,
                portion_size=food_item_data.portion_size,
                portion_unit=food_item_data.portion_unit,
                calories=food_item_data.calories,
                protein_g=food_item_data.protein_g,
                carbs_g=food_item_data.carbs_g,
                fat_g=food_item_data.fat_g,
            )
            db_meal.food_items.append(db_food_item)

        db.add(db_meal)
        db.commit()
        db.refresh(db_meal)

        return db_meal

    @staticmethod
    def get_meal_by_id(db: Session, user_id: str, meal_id: str) -> Optional[MealDB]:
        """Get a meal by ID for a specific user.

        Args:
            db: Database session
            user_id: User ID
            meal_id: Meal ID

        Returns:
            Meal database object or None if not found

        """
        return (
            db.query(MealDB)
            .options(selectinload(MealDB.food_items))
            .filter(and_(MealDB.id == meal_id, MealDB.user_id == user_id))
            .first()
        )

    @staticmethod
    def get_user_meals(
        db: Session,
        user_id: str,
        skip: int = 0,
        limit: int = 20,
        meal_type: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> tuple[list[MealDB], int]:
        """Get meals for a user with optional filters.

        Args:
            db: Database session
            user_id: User ID
            skip: Number of records to skip (pagination)
            limit: Maximum number of records to return
            meal_type: Optional filter by meal type
            start_date: Optional filter for meals after this date
            end_date: Optional filter for meals before this date

        Returns:
            Tuple of (list of meals, total count)

        """
        # Build query
        query = db.query(MealDB).filter(MealDB.user_id == user_id)

        # Apply filters
        if meal_type:
            query = query.filter(MealDB.meal_type == meal_type)
        if start_date:
            query = query.filter(MealDB.meal_time >= start_date)
        if end_date:
            query = query.filter(MealDB.meal_time <= end_date)

        # Get total count
        total = query.count()

        # Get paginated results
        meals = (
            query.options(selectinload(MealDB.food_items))
            .order_by(desc(MealDB.meal_time))
            .offset(skip)
            .limit(limit)
            .all()
        )

        return meals, total

    @staticmethod
    def update_meal(
        db: Session, user_id: str, meal_id: str, meal_data: MealUpdate
    ) -> Optional[MealDB]:
        """Update an existing meal log.

        Args:
            db: Database session
            user_id: User ID
            meal_id: Meal ID
            meal_data: Meal update data

        Returns:
            Updated meal database object or None if not found

        """
        db_meal = MealService.get_meal_by_id(db, user_id, meal_id)

        if not db_meal:
            return None

        # Update meal fields
        update_data = meal_data.model_dump(exclude_unset=True, exclude={"food_items"})
        for field, value in update_data.items():
            if field == "meal_type" and value:
                setattr(db_meal, field, value.value)
            else:
                setattr(db_meal, field, value)

        # Update food items if provided
        if meal_data.food_items is not None:
            # Delete existing food items
            db.query(MealFoodItemDB).filter(MealFoodItemDB.meal_id == meal_id).delete()

            # Create new food items
            total_calories = 0.0
            total_protein = 0.0
            total_carbs = 0.0
            total_fat = 0.0

            for food_item_data in meal_data.food_items:
                db_food_item = MealFoodItemDB(
                    id=str(uuid.uuid4()),
                    meal_id=db_meal.id,
                    food_name=food_item_data.food_name,
                    portion_size=food_item_data.portion_size,
                    portion_unit=food_item_data.portion_unit,
                    calories=food_item_data.calories,
                    protein_g=food_item_data.protein_g,
                    carbs_g=food_item_data.carbs_g,
                    fat_g=food_item_data.fat_g,
                )
                db_meal.food_items.append(db_food_item)

                # Calculate totals with portion size multiplier
                if food_item_data.calories:
                    total_calories += food_item_data.calories * food_item_data.portion_size
                if food_item_data.protein_g:
                    total_protein += food_item_data.protein_g * food_item_data.portion_size
                if food_item_data.carbs_g:
                    total_carbs += food_item_data.carbs_g * food_item_data.portion_size
                if food_item_data.fat_g:
                    total_fat += food_item_data.fat_g * food_item_data.portion_size

            # Update nutritional totals
            db_meal.total_calories = total_calories or 0
            db_meal.total_protein_g = total_protein or 0
            db_meal.total_carbs_g = total_carbs or 0
            db_meal.total_fat_g = total_fat or 0

        db.commit()
        db.refresh(db_meal)

        return db_meal

    @staticmethod
    def delete_meal(db: Session, user_id: str, meal_id: str) -> bool:
        """Delete a meal log.

        Args:
            db: Database session
            user_id: User ID
            meal_id: Meal ID

        Returns:
            True if meal was deleted, False if not found

        """
        db_meal = MealService.get_meal_by_id(db, user_id, meal_id)

        if not db_meal:
            return False

        db.delete(db_meal)
        db.commit()

        return True
