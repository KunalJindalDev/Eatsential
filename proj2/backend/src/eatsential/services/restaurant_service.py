"""Service for managing restaurants from Google Places API."""

import logging
from typing import Optional
from sqlalchemy.orm import Session
from sqlalchemy import or_

from ..models.models import Restaurant

logger = logging.getLogger(__name__)


def extract_cuisine_from_types(types: list[str]) -> Optional[str]:
    """Extract cuisine type from Google Places types list."""
    if not types:
        return None
    
    # Common cuisine keywords to look for in types
    cuisine_keywords = {
        "japanese": ["japanese", "sushi"],
        "chinese": ["chinese"],
        "italian": ["italian", "pizza"],
        "mexican": ["mexican", "taco"],
        "indian": ["indian"],
        "thai": ["thai"],
        "american": ["american", "burger", "steakhouse"],
        "mediterranean": ["mediterranean", "greek", "middle_eastern"],
        "french": ["french"],
        "korean": ["korean"],
        "vietnamese": ["vietnamese"],
        "indian": ["indian"],
    }
    
    types_lower = [t.lower() for t in types]
    
    for cuisine, keywords in cuisine_keywords.items():
        if any(keyword in " ".join(types_lower) for keyword in keywords):
            return cuisine
    
    return None


def save_restaurant_from_google_places(
    db: Session,
    place_data: dict,
    cuisine: Optional[str] = None,
) -> Optional[Restaurant]:
    """Save or update a restaurant from Google Places API data.
    
    Args:
        db: Database session
        place_data: Dictionary containing Google Places API response data
            Expected keys: place_id, name, formatted_address, types, geometry
        cuisine: Optional cuisine type to use (takes precedence over extracted cuisine)
    
    Returns:
        Restaurant object if saved successfully, None otherwise
    """
    place_id = place_data.get("place_id")
    name = place_data.get("name")
    
    if not place_id or not name:
        logger.warning(f"Cannot save restaurant: missing place_id or name. Data: {place_data}")
        return None
    
    # Determine cuisine: use provided cuisine, or extract from types, or keep existing
    types = place_data.get("types", [])
    extracted_cuisine = extract_cuisine_from_types(types) if types else None
    final_cuisine = cuisine or extracted_cuisine
    
    # Check if restaurant already exists by place_id (using id field) or name
    existing = db.query(Restaurant).filter(
        or_(
            Restaurant.id == place_id,
            Restaurant.name == name
        )
    ).first()
    
    if existing:
        # Update existing restaurant with latest data
        if place_data.get("formatted_address") and not existing.address:
            existing.address = place_data.get("formatted_address")
        
        # Update website URL if available
        if place_data.get("website") and not existing.website_url:
            existing.website_url = place_data.get("website")
        
        # Update cuisine if we have new information
        # Prefer provided cuisine, then extracted, and only update if we have better info
        if final_cuisine:
            # Update if: no cuisine set, or we have a provided cuisine (more reliable)
            if not existing.cuisine or cuisine:
                existing.cuisine = final_cuisine
        
        # Ensure place_id is stored in id field if it wasn't before
        if existing.id != place_id:
            # If id is different, we might have a duplicate - keep the one with place_id
            if existing.id == place_id or not existing.id.startswith("ChIJ"):  # place_ids start with ChIJ
                existing.id = place_id
        
        try:
            db.commit()
            db.refresh(existing)
            logger.info(f"Updated existing restaurant: {name} (place_id: {place_id}, cuisine: {existing.cuisine})")
            return existing
        except Exception as e:
            db.rollback()
            logger.error(f"Error updating restaurant {name}: {e}")
            return existing
    
    # Create new restaurant
    address = place_data.get("formatted_address")
    website_url = place_data.get("website")
    
    restaurant = Restaurant(
        id=place_id,  # Use place_id as the primary key for uniqueness
        name=name,
        address=address,
        website_url=website_url,
        cuisine=final_cuisine,
        is_active=True,
    )
    
    try:
        db.add(restaurant)
        db.commit()
        db.refresh(restaurant)
        logger.info(f"Saved new restaurant: {name} (place_id: {place_id}, cuisine: {final_cuisine})")
        return restaurant
    except Exception as e:
        db.rollback()
        logger.error(f"Error saving restaurant {name}: {e}")
        return None

