"""API routes for orders (menu items associated with meal logs)."""

from typing import Annotated
from datetime import datetime, timedelta

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from ..db.database import get_db
from ..models.models import MenuItem, MealDB, Orders, UserDB, Restaurant, MealFoodItemDB
from ..schemas.schemas import OrderCreate, OrderResponse, ScheduledOrderResponse, UserResponse, MealUpdate, MealFoodItemCreate
from ..services.auth_service import get_current_user
from ..services.meal_service import MealService
from uuid import uuid4

router = APIRouter(prefix="/orders", tags=["orders"])

SessionDep = Annotated[Session, Depends(get_db)]
CurrentUserDep = Annotated[UserResponse, Depends(get_current_user)]


@router.post("", response_model=OrderResponse, status_code=status.HTTP_201_CREATED)
def create_order(
    order_data: OrderCreate,
    current_user: CurrentUserDep,
    db: SessionDep,
):
    """Create an order linking a menu item to a user's meal log.

    This endpoint associates a restaurant menu item with a logged meal,
    creating an order record. Useful for tracking which specific menu items
    were included in which meals.

    Args:
        order_data: Order creation data (menu_item_id and meal_id)
        current_user: Authenticated user
        db: Database session

    Returns:
        Created order

    Raises:
        HTTPException 400: If menu item or meal not found, or meal doesn't belong to user
        HTTPException 500: If creation fails

    """
    try:
        # Verify that the menu item exists
        menu_item = db.scalars(
            select(MenuItem).filter(MenuItem.id == order_data.menu_item_id)
        ).first()
        if not menu_item:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Menu item '{order_data.menu_item_id}' not found",
            )

        # Verify that the meal exists and belongs to the current user
        meal = db.scalars(
            select(MealDB).filter(
                MealDB.id == order_data.meal_id,
                MealDB.user_id == current_user.id,
            )
        ).first()
        if not meal:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Meal not found or does not belong to the current user",
            )

        # Create the order
        order = Orders(
            id=str(uuid4()),
            menu_item_id=order_data.menu_item_id,
            meal_id=order_data.meal_id,
        )
        db.add(order)
        db.commit()
        db.refresh(order)

        return OrderResponse.model_validate(order)

    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create order: {e!s}",
        )


@router.get("/scheduled", response_model=list[ScheduledOrderResponse])
def get_scheduled_orders(
    current_user: CurrentUserDep,
    db: SessionDep,
    days: int = 7,
):
    """Get scheduled orders for the next N days.

    Retrieves orders for the current user that are scheduled for the next N days,
    including meal and menu item information. Orders are sorted by meal_time.

    Args:
        current_user: Authenticated user
        db: Database session
        days: Number of days to look ahead (default: 7)

    Returns:
        List of scheduled orders with related meal and menu item data

    Raises:
        HTTPException 500: If retrieval fails
    """
    try:
        now = datetime.utcnow()
        future_cutoff = now + timedelta(days=days)

        # Query for orders in the next N days
        orders = db.scalars(
            select(Orders)
            .join(MealDB, Orders.meal_id == MealDB.id)
            .join(MenuItem, Orders.menu_item_id == MenuItem.id)
            .join(Restaurant, MenuItem.restaurant_id == Restaurant.id)
            .filter(
                MealDB.user_id == current_user.id,
                MealDB.meal_time >= now,
                MealDB.meal_time <= future_cutoff,
            )
            .order_by(MealDB.meal_time)
        ).all()

        # Build response objects
        results = []
        for order in orders:
            meal = db.scalars(
                select(MealDB).filter(MealDB.id == order.meal_id)
            ).first()
            menu_item = db.scalars(
                select(MenuItem).filter(MenuItem.id == order.menu_item_id)
            ).first()
            restaurant = db.scalars(
                select(Restaurant).filter(Restaurant.id == menu_item.restaurant_id)
            ).first()

            if meal and menu_item and restaurant:
                # Get the portion info from the meal's food items
                # Use the first food item's portion as the portion for this order
                # This is a placeholder until we add functionality to add multiple items to an order
                portion_size = 1.0
                portion_unit = "serving"
                
                if meal.food_items:
                    portion_size = meal.food_items[0].portion_size
                    portion_unit = meal.food_items[0].portion_unit
                
                results.append(
                    ScheduledOrderResponse(
                        id=order.id,
                        menu_item_id=order.menu_item_id,
                        meal_id=order.meal_id,
                        meal_type=meal.meal_type,
                        meal_time=meal.meal_time,
                        menu_item_name=menu_item.name,
                        restaurant_name=restaurant.name,
                        calories=menu_item.calories,
                        price=menu_item.price,
                        portion_size=portion_size,
                        portion_unit=portion_unit,
                    )
                )

        return results

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve scheduled orders: {e!s}",
        )


@router.delete("/{order_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_order(
    order_id: str,
    current_user: CurrentUserDep,
    db: SessionDep,
):
    """Delete an order (only if it's more than 30 minutes in the future).

    Args:
        order_id: ID of the order to delete
        current_user: Authenticated user
        db: Database session

    Returns:
        204 No Content

    Raises:
        HTTPException 404: If order not found
        HTTPException 403: If order belongs to another user or is within 30 minutes
        HTTPException 500: If deletion fails
    """
    try:
        # Get the order
        order = db.scalars(select(Orders).filter(Orders.id == order_id)).first()
        if not order:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Order not found",
            )

        # Get the meal to verify ownership and check time
        meal = db.scalars(
            select(MealDB).filter(
                MealDB.id == order.meal_id,
                MealDB.user_id == current_user.id,
            )
        ).first()
        if not meal:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Order does not belong to the current user",
            )

        # Check if meal is more than 30 minutes in the future
        now = datetime.utcnow()
        min_deletion_time = now + timedelta(minutes=30)
        if meal.meal_time < min_deletion_time:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Order cannot be deleted less than 30 minutes before meal time",
            )

        # Delete the order (which will cascade delete the meal due to foreign key constraint)
        db.delete(order)
        # Also explicitly delete the meal to ensure it's removed
        db.delete(meal)
        db.commit()

    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete order: {e!s}",
        )


@router.put("/{order_id}/meal", response_model=ScheduledOrderResponse)
def update_order_meal(
    order_id: str,
    meal_update: MealUpdate,
    current_user: CurrentUserDep,
    db: SessionDep,
):
    """Update the meal associated with an order.

    This allows updating meal details like food items, notes, etc.

    Args:
        order_id: ID of the order
        meal_update: Updated meal data
        current_user: Authenticated user
        db: Database session

    Returns:
        Updated ScheduledOrderResponse

    Raises:
        HTTPException 404: If order not found
        HTTPException 403: If order belongs to another user
        HTTPException 500: If update fails
    """
    try:
        # Get the order
        order = db.scalars(select(Orders).filter(Orders.id == order_id)).first()
        if not order:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Order not found",
            )

        # Get the meal to verify ownership
        meal = db.scalars(
            select(MealDB).filter(
                MealDB.id == order.meal_id,
                MealDB.user_id == current_user.id,
            )
        ).first()
        if not meal:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Order does not belong to the current user",
            )

        # Update the meal using MealService
        updated_meal = MealService.update_meal(
            db, current_user.id, order.meal_id, meal_update
        )
        if not updated_meal:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Failed to update meal",
            )

        # Get menu item and restaurant for response
        menu_item = db.scalars(
            select(MenuItem).filter(MenuItem.id == order.menu_item_id)
        ).first()
        restaurant = db.scalars(
            select(Restaurant).filter(Restaurant.id == menu_item.restaurant_id)
        ).first()

        # Get portion from meal food items
        portion_size = 1.0
        portion_unit = "serving"
        if updated_meal.food_items:
            portion_size = updated_meal.food_items[0].portion_size
            portion_unit = updated_meal.food_items[0].portion_unit

        return ScheduledOrderResponse(
            id=order.id,
            menu_item_id=order.menu_item_id,
            meal_id=order.meal_id,
            meal_type=updated_meal.meal_type,
            meal_time=updated_meal.meal_time,
            menu_item_name=menu_item.name,
            restaurant_name=restaurant.name,
            calories=menu_item.calories,
            price=menu_item.price,
            portion_size=portion_size,
            portion_unit=portion_unit,
        )

    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update order: {e!s}",
        )
