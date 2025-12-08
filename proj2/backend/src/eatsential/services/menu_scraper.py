"""
Menu scraper service for pulling menu items from restaurants.

This service attempts to scrape menu items from restaurant websites and APIs.
It tries multiple strategies:
1. Google Places API (if available)
2. Website scraping (BeautifulSoup-based)
3. Third-party APIs (Yelp, etc.)
"""

import asyncio
import logging
from typing import Optional
from urllib.parse import urljoin

import httpx
from bs4 import BeautifulSoup
from sqlalchemy.orm import Session

from eatsential.models.models import MenuItem, Restaurant
from eatsential.services.google_places import GOOGLE_MAPS_API_KEY

logger = logging.getLogger(__name__)

# Keywords to identify menu sections
MENU_KEYWORDS = {
    "appetizers", "starters", "entrees", "mains", "main course", "sides",
    "desserts", "drinks", "beverages", "soup", "salad", "pasta", "pizza",
    "sandwich", "burger", "seafood", "poultry", "vegetarian", "vegan",
    "menu", "food", "items", "dishes", "meals", "specials", "offerings",
}

PRICE_KEYWORDS = ["$", "price", "cost", "fee"]


async def scrape_website_menu(website_url: str) -> list[dict]:
    """
    Attempt to scrape menu items from a restaurant's website.

    Returns:
        List of dictionaries with 'name', 'description', 'price' keys
    """
    if not website_url:
        return []

    try:
        async with httpx.AsyncClient(timeout=10.0, follow_redirects=True) as client:
            response = await client.get(website_url)
            response.raise_for_status()

        soup = BeautifulSoup(response.content, "html.parser")
        menu_items = []

        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()

        # Look for common menu section patterns
        menu_sections = _find_menu_sections(soup)

        for section in menu_sections:
            items = _extract_items_from_section(section)
            menu_items.extend(items)

        logger.info(f"Scraped {len(menu_items)} menu items from {website_url}")
        return menu_items

    except httpx.HTTPError as e:
        logger.warning(f"HTTP error scraping menu from {website_url}: {e}")
        return []
    except Exception as e:
        logger.warning(f"Failed to scrape menu from {website_url}: {e}")
        return []


def _find_menu_sections(soup: BeautifulSoup) -> list:
    """Find likely menu sections in the parsed HTML."""
    sections = []

    # Look for divs, sections with menu-related class names
    for element in soup.find_all(["div", "section", "article"]):
        class_str = " ".join(element.get("class", [])).lower()
        id_str = element.get("id", "").lower()

        for keyword in MENU_KEYWORDS:
            if keyword in class_str or keyword in id_str:
                sections.append(element)
                break

    # If no sections found, try to parse the whole body
    if not sections:
        sections = [soup.body] if soup.body else [soup]

    return sections


def _extract_items_from_section(section) -> list[dict]:
    """Extract individual menu items from a section."""
    items = []
    seen_names = set()

    # Get all text lines from this section
    text_content = section.get_text("\n", strip=True)
    lines = [line.strip() for line in text_content.split("\n") if line.strip()]
    
    # Price pattern: $12.99 or $12 or 12.99
    price_pattern = r'\$?\s*(\d+(?:\.\d{2})?)'
    
    i = 0
    while i < len(lines):
        line = lines[i]
        
        # Skip empty or very short lines
        if len(line) < 3:
            i += 1
            continue
        
        # Skip common non-menu keywords
        if any(skip in line.lower() for skip in [
            "please", "order online", "catering", "about", "contact", "hours",
            "location", "phone", "email", "delivery", "copyright", "terms"
        ]):
            i += 1
            continue
        
        # Look for price in this line
        price_match = re.search(price_pattern, line)
        
        if price_match:
            price_str = price_match.group(1)
            try:
                price = float(price_str)
                
                # Extract name (everything before the price)
                name = line[:price_match.start()].strip()
                
                # Remove common delimiters
                name = re.sub(r'[-–—•.]+$', '', name).strip()
                
                # Description: next line if it looks like description
                description = ""
                if i + 1 < len(lines):
                    next_line = lines[i + 1]
                    # If next line is short and doesn't have a price, it's likely description
                    if len(next_line) < 200 and not re.search(price_pattern, next_line):
                        if next_line and not any(skip in next_line.lower() for skip in ["please", "order", "contact"]):
                            description = next_line
                            i += 1
                
                # Validate name (not too generic or too long)
                if len(name) >= 2 and len(name) <= 200:
                    if name.lower() not in ["item", "menu", "special", "dish"]:
                        if name not in seen_names:
                            items.append({
                                "name": name,
                                "price": price,
                                "description": description[:500] if description else "",
                            })
                            seen_names.add(name)
            except ValueError:
                pass
        
        i += 1
    
    return items


async def scrape_menu_for_restaurant(
    db: Session, restaurant: Restaurant
) -> list[MenuItem]:
    """
    Main function to scrape menu for a restaurant.
    Tries multiple approaches and drops restaurants with no valid menu data.

    Returns:
        List of MenuItem objects created for the restaurant, or empty list if no menu found
    """
    existing_items = db.query(MenuItem).filter_by(restaurant_id=restaurant.id).count()
    if existing_items > 0:
        logger.info(f"Restaurant {restaurant.name} already has {existing_items} menu items")
        return []

    logger.info(f"Scraping menu for {restaurant.name}...")

    menu_items = []

    # Try 1: Website scraping
    if restaurant.website_url:
        logger.debug(f"  Trying website: {restaurant.website_url}")
        menu_items = await scrape_website_menu(restaurant.website_url)

    # Try 2: Common menu URLs (if no items found yet)
    if not menu_items and restaurant.website_url:
        common_paths = ["/menu", "/menu.pdf", "/menus", "/pdf/menu", "/documents/menu"]
        base_url = restaurant.website_url.rstrip("/")
        
        for path in common_paths:
            if menu_items:
                break
            try:
                menu_url = base_url + path
                logger.debug(f"  Trying common path: {menu_url}")
                menu_items = await scrape_website_menu(menu_url)
            except Exception:
                pass

    # If no menu items found, drop the restaurant
    if not menu_items:
        logger.warning(f"No menu items found for {restaurant.name}, dropping from recommendations")
        # Mark as inactive so it won't be recommended
        restaurant.is_active = False
        db.commit()
        return []

    # Deduplicate items by name and price
    seen = set()
    unique_items = []
    for item_data in menu_items:
        # Skip empty dicts
        if not item_data or not isinstance(item_data, dict):
            continue
        
        # Skip items without name or price
        if not item_data.get("name") or item_data.get("price") is None:
            continue
            
        key = (item_data.get("name", ""), item_data.get("price"))
        if key not in seen:
            seen.add(key)
            unique_items.append(item_data)

    # Convert scraped items to MenuItem objects
    created_items = []
    for idx, item_data in enumerate(unique_items[:20]):  # Limit to 20 items per restaurant
        try:
            # Only create items that have both name and price
            if not item_data or not item_data.get("name") or item_data.get("price") is None:
                continue
                
            menu_item = MenuItem(
                id=f"{restaurant.id}_item_{idx}",
                restaurant_id=restaurant.id,
                name=item_data.get("name", "Unknown Item")[:200],
                description=item_data.get("description", "")[:1000] or None,
                price=item_data.get("price"),
            )
            db.add(menu_item)
            created_items.append(menu_item)
        except Exception as e:
            logger.error(f"Failed to create menu item for {restaurant.name}: {e}")

    try:
        db.commit()
        if created_items:
            logger.info(f"Created {len(created_items)} unique menu items for {restaurant.name}")
        else:
            logger.warning(f"No valid menu items created for {restaurant.name}, dropping restaurant")
            restaurant.is_active = False
            db.commit()
    except Exception as e:
        db.rollback()
        logger.error(f"Failed to commit menu items for {restaurant.name}: {e}")

    return created_items
