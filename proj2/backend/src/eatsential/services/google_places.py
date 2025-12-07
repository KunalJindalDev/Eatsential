import os
import httpx

GOOGLE_MAPS_API_KEY = os.getenv("GOOGLE_MAPS_API_KEY")

NC_STATE_LAT = 35.7847
NC_STATE_LNG = -78.6821
SEARCH_RADIUS_METERS = 16000  # 10 miles

async def search_places_for_cuisine(cuisine: str) -> list[dict]:
    if not GOOGLE_MAPS_API_KEY:
        raise RuntimeError("GOOGLE_MAPS_API_KEY is not configured")

    url = (
        "https://maps.googleapis.com/maps/api/place/textsearch/json"
        f"?query={cuisine}+restaurant"
        f"&location={NC_STATE_LAT},{NC_STATE_LNG}"
        f"&radius={SEARCH_RADIUS_METERS}"
        f"&key={GOOGLE_MAPS_API_KEY}"
    )

    async with httpx.AsyncClient() as client:
        r = await client.get(url)
        data = r.json()

    if "error_message" in data:
        raise RuntimeError(data["error_message"])

    results = data.get("results", [])
    places = []

    for r in results:
        places.append({
            "place_id": r.get("place_id"),
            "name": r.get("name"),
            "address": r.get("formatted_address"),
            "rating": r.get("rating"),
            "price_level": r.get("price_level"),
            "location": r.get("geometry", {}).get("location"),
            "types": r.get("types", []),  # Include types for cuisine extraction
        })

    return places[:10]   # limit to reduce LLM cost
