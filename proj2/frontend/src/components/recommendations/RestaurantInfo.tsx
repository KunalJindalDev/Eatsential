import { MapPin, Utensils, DollarSign } from "lucide-react";
import { Card, CardContent } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { useGooglePlaceSearch } from "@/hooks/useGooglePlaces";
import type { RestaurantInfo as RestaurantInfoType, MenuItemInfo } from "@/lib/api";

interface RestaurantInfoProps {
  restaurant: RestaurantInfoType;
  menuItem?: MenuItemInfo;
}

export function RestaurantInfo({ restaurant, menuItem }: RestaurantInfoProps) {
  // IMPORTANT: Use restaurant.restaurant_name, not restaurant.name
  const lookupName =
    restaurant.restaurant_name || restaurant.name || menuItem?.restaurant_name;

  const { data: place } = useGooglePlaceSearch(lookupName);

  return (
    <Card className="mt-4">
      <CardContent className="space-y-3 pt-4">
        {/* Restaurant Title */}
        <div className="flex items-start justify-between">
          <div className="flex items-center gap-2">
            <Utensils className="h-5 w-5 text-primary" />
            <h3 className="text-lg font-semibold">
              {restaurant.restaurant_name || restaurant.name}
            </h3>
          </div>
          {restaurant.is_active ? (
            <Badge variant="default" className="bg-green-500">
              Open
            </Badge>
          ) : (
            <Badge variant="secondary">Closed</Badge>
          )}
        </div>

        {/* Cuisine */}
        {restaurant.cuisine && (
          <div className="flex items-center gap-2 text-sm text-muted-foreground">
            <Utensils className="h-4 w-4" />
            <span>{restaurant.cuisine}</span>
          </div>
        )}

        {/* Address */}
        {(restaurant.restaurant_address || restaurant.address) && (
          <div className="flex items-start gap-2 text-sm text-muted-foreground">
            <MapPin className="mt-0.5 h-4 w-4 flex-shrink-0" />
            <span>
              {restaurant.restaurant_address || restaurant.address}
            </span>
          </div>
        )}

        {/* Google Maps Preview */}
        {place && (
          <div className="space-y-2">
            <iframe
              width="100%"
              height="200"
              loading="lazy"
              allowFullScreen
              className="rounded-md"
              src={`https://www.google.com/maps/embed/v1/place?key=${
                import.meta.env.VITE_GOOGLE_MAPS_API_KEY
              }&q=place_id:${place.place_id}`}
            />
            <a
              href={`https://www.google.com/maps/search/?api=1&query=place_id:${place.place_id}`}
              target="_blank"
              className="text-sm text-blue-600 underline"
            >
              View on Google Maps
            </a>
          </div>
        )}

        {/* Menu Item */}
        {menuItem && (
          <div className="space-y-2 border-t pt-3">
            <h4 className="font-medium">{menuItem.name}</h4>

            {menuItem.description && (
              <p className="text-sm text-muted-foreground">
                {menuItem.description}
              </p>
            )}

            <div className="mt-2 flex flex-wrap gap-2">
              {menuItem.price != null && (
                <Badge variant="outline" className="flex items-center gap-1">
                  <DollarSign className="h-3 w-3" />
                  <span>${menuItem.price.toFixed(2)}</span>
                </Badge>
              )}

              {menuItem.calories != null && (
                <Badge variant="outline">
                  {Math.round(menuItem.calories)} cal
                </Badge>
              )}
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  );
}
