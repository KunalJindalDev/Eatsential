"""Seed restaurant and menu item data.

This script populates the database with sample restaurants and their menu items.
"""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from eatsential.db.database import DATABASE_URL
from eatsential.models import AllergenDB, MenuItem, Restaurant


def load_restaurant_data():
    """Load restaurant data from JSON file."""
    data_file = Path(__file__).parent / "data" / "restaurants.json"
    with open(data_file) as f:
        return json.load(f)


def seed_restaurants(session: Session):
    """Seed restaurants and menu items from JSON data.

    Args:
        session: Database session

    """
    print("\nSeeding restaurant data...")

    restaurants_data = load_restaurant_data()

    for rest_data in restaurants_data:
        # Create restaurant
        restaurant = Restaurant(
            id=str(uuid4()),
            name=rest_data["name"],
            address=rest_data.get("address"),
            cuisine=rest_data.get("cuisine"),
            is_active=True,
            created_at=datetime.now(timezone.utc).replace(tzinfo=None),
        )
        session.add(restaurant)

        # Create menu items for this restaurant
        for item_data in rest_data["menu_items"]:
            menu_item = MenuItem(
                id=str(uuid4()),
                restaurant_id=restaurant.id,
                name=item_data["name"],
                description=item_data.get("description"),
                calories=item_data.get("calories"),
                price=item_data.get("price"),
                created_at=datetime.now(timezone.utc).replace(tzinfo=None),
            )
            
            # Associate allergens if provided
            allergen_names = item_data.get("allergens", [])
            if allergen_names:
                for allergen_name in allergen_names:
                    allergen = session.query(AllergenDB).filter(
                        AllergenDB.name.ilike(allergen_name)
                    ).first()
                    if allergen:
                        menu_item.allergens.append(allergen)
                    else:
                        print(f"Warning: Allergen '{allergen_name}' not found for menu item '{item_data['name']}'")
            
            session.add(menu_item)

    session.commit()

    # Count results
    restaurant_count = session.query(Restaurant).count()
    menu_item_count = session.query(MenuItem).count()

    print(f"✅ Seeded {restaurant_count} restaurants with {menu_item_count} menu items")


def main():
    """Seed restaurant data into the database."""
    print("=" * 70)
    print("RESTAURANT DATA SEEDING")
    print("=" * 70)

    # Create engine and session
    engine = create_engine(DATABASE_URL)
    session_local = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    session = session_local()

    try:
        seed_restaurants(session)
        print("\n✅ Restaurant seeding completed successfully!")
    except Exception as e:
        print(f"\n❌ Error seeding restaurants: {e}")
        session.rollback()
        import traceback

        traceback.print_exc()
        sys.exit(1)
    finally:
        session.close()


if __name__ == "__main__":
    main()
