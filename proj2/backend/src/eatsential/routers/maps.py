import httpx
import os
import urllib.parse
from typing import Annotated
from fastapi import APIRouter, HTTPException, Query, Depends
from sqlalchemy.orm import Session

from ..db.database import get_db
from ..services.restaurant_service import save_restaurant_from_google_places

router = APIRouter(prefix="/maps", tags=["maps"])

GOOGLE_MAPS_API_KEY = os.getenv("GOOGLE_MAPS_API_KEY")
SessionDep = Annotated[Session, Depends(get_db)]


@router.get("/search")
async def search_place(
    db: SessionDep,
    query: str = Query(...),
):
    if not GOOGLE_MAPS_API_KEY:
        raise HTTPException(500, "GOOGLE_MAPS_API_KEY is not configured")

    # -----------------------------
    # NC STATE CAMPUS COORDINATES
    # -----------------------------
    lat = 35.7847
    lng = -78.6821
    radius = 16000  # 16 km ≈ 10 miles

    # -----------------------------
    # IMPORTANT: Clean + validate + encode
    # -----------------------------
    cleaned = query.strip()
    
    # Validate query - reject obviously invalid queries
    if not cleaned or len(cleaned) < 2:
        raise HTTPException(400, "Query must be at least 2 characters long")
    
    # Reject queries that look like fragments of explanations
    invalid_patterns = [
        "falls within",
        "price range",
        "and falls",
        "within the",
    ]
    cleaned_lower = cleaned.lower()
    if any(pattern in cleaned_lower for pattern in invalid_patterns):
        raise HTTPException(400, f"Invalid query: '{cleaned}' does not appear to be a valid place name")
    
    # Reject queries that are mostly numbers or special characters
    if len([c for c in cleaned if c.isalnum()]) < 2:
        raise HTTPException(400, f"Invalid query: '{cleaned}' does not appear to be a valid place name")
    
    encoded_query = urllib.parse.quote_plus(cleaned)

    url = (
        "https://maps.googleapis.com/maps/api/place/textsearch/json"
        f"?query={encoded_query}"
        f"&location={lat},{lng}"
        f"&radius={radius}"
        f"&key={GOOGLE_MAPS_API_KEY}"
    )

    async with httpx.AsyncClient() as client:
        response = await client.get(url)
        data = response.json()

    # Bubble up Google errors
    if "error_message" in data:
        raise HTTPException(500, f"Google API error: {data['error_message']}")

    if not data.get("results"):
        raise HTTPException(404, f"No nearby places found for '{cleaned}'")

    # Return up to 5 results
    results = data["results"][:5]

    # Save restaurants to database automatically
    for r in results:
        place_id = r.get("place_id")
        if place_id:
            # Save restaurant to database
            save_restaurant_from_google_places(
                db=db,
                place_data={
                    "place_id": place_id,
                    "name": r.get("name"),
                    "formatted_address": r.get("formatted_address"),
                    "types": r.get("types", []),
                    "geometry": r.get("geometry", {}),
                }
            )

    return [
        {
            "name": r.get("name"),
            "address": r.get("formatted_address"),
            "location": r.get("geometry", {}).get("location"),
            "place_id": r.get("place_id"),
        }
        for r in results
    ]
