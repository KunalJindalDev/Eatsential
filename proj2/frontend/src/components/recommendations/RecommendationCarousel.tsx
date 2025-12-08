import { useEffect, useMemo, useState } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Label } from '@/components/ui/label';
import { ToggleGroup, ToggleGroupItem } from '@/components/ui/toggle-group';
import { Plus } from 'lucide-react';
import { Progress } from '@/components/ui/progress';
import { ChefHat, RefreshCw, Sparkles, UtensilsCrossed, Eraser, Bot, Baseline, ThumbsUp, ThumbsDown, MapPin} from 'lucide-react';
import { useMealRecommendations } from '@/hooks/useRecommendations';
import { useGooglePlaceSearch } from "@/hooks/useGooglePlaces";
import { recommendationApi } from '@/lib/api';
import { QuickMealPlanner } from './OrderDrawer';
import type {
  MealRecommendationResponse,
  RecommendationFiltersPayload,
  RecommendationMode,
} from '@/lib/api';

interface RecommendationCarouselProps {
  userId: string;
  initialFilters?: RecommendationFiltersPayload;
  initialMode?: RecommendationMode;
}

interface NormalizedRecommendation {
  id: string;
  name: string;
  score: number;
  explanation: string;
  highlights: string[];
  restaurant?: string;
  restaurant_place_id?: string;
  description?: string;
  price?: number;
  calories?: number;
}

const priceRanges = [
  { value: '$', label: 'Budget ($)' },
  { value: '$$', label: 'Casual ($$)' },
  { value: '$$$', label: 'Premium ($$$)' },
  { value: '$$$$', label: 'Fine Dining ($$$$)' },
] as const;

// Common dietary preferences
const COMMON_DIETS = ['vegetarian', 'vegan', 'gluten-free', 'dairy-free', 'keto', 'paleo'] as const;

// Common cuisines
const COMMON_CUISINES = [
  'italian',
  'chinese',
  'japanese',
  'mexican',
  'indian',
  'thai',
  'american',
  'mediterranean',
] as const;

const clampScore = (value: unknown): number => {
  const parsed =
    typeof value === 'number' ? value : typeof value === 'string' ? Number.parseFloat(value) : 0;
  if (!Number.isFinite(parsed)) {
    return 0;
  }
  return Math.min(1, Math.max(0, parsed));
};

const isValidRestaurantName = (name: string): boolean => {
  if (!name || name.length < 2 || name.length > 100) {
    return false;
  }
  
  // Reject if it contains common non-restaurant phrases
  const invalidPatterns = [
    /falls within/i,
    /price range/i,
    /\$\$/,
    /and\s+falls/i,
    /within\s+the/i,
    /range\s*\(/i,
    /^\d+\.\d+$/,
  ];
  
  for (const pattern of invalidPatterns) {
    if (pattern.test(name)) {
      return false;
    }
  }
  
  // Should start with a letter and contain mostly letters/spaces/common restaurant chars
  if (!/^[A-Za-z]/.test(name)) {
    return false;
  }
  
  // Should not be just numbers or special characters
  if (/^[\d\s$().]+$/.test(name)) {
    return false;
  }
  
  return true;
};

const extractRestaurant = (explanation: string, fallback?: string): string | undefined => {
  // First try to use the fallback if it exists and looks valid
  if (fallback && isValidRestaurantName(fallback)) {
    return fallback;
  }
  
  // Try to extract restaurant name from explanation
  // Look for patterns like "restaurant: Name" or "at Restaurant Name" or "from Restaurant Name"
  const patterns = [
    /restaurant[:\s-]+([A-Z][A-Za-z0-9\s&'-]{2,40})(?:[;,\n]|$)/i,
    /(?:at|from|via)\s+([A-Z][A-Za-z0-9\s&'-]{2,40})\s+(?:restaurant|eatery|dining)/i,
  ];
  
  for (const pattern of patterns) {
    const match = explanation.match(pattern);
    if (match && match[1]) {
      const extracted = match[1].trim();
      if (isValidRestaurantName(extracted)) {
        return extracted;
      }
    }
  }
  
  return fallback && isValidRestaurantName(fallback) ? fallback : undefined;
};

const splitHighlights = (explanation: string): string[] => {
  if (!explanation) {
    return [];
  }

  const segments = explanation
    .split(/[\n;]+/)
    .map((segment) => segment.trim())
    .filter(Boolean);

  if (segments.length > 1) {
    return segments;
  }

  return explanation
    .split(',')
    .map((segment) => segment.trim())
    .filter(Boolean);
};

const normalizeRecommendationResponse = (
  data?: MealRecommendationResponse
): NormalizedRecommendation[] => {
  if (!data) {
    return [];
  }

  // Deduplicate items by id
  const seenIds = new Set<string>();
  const deduplicatedItems: NormalizedRecommendation[] = [];

  if ('items' in data) {
    for (const item of data.items) {
      const itemId = String(item.item_id);
      if (seenIds.has(itemId)) {
        continue; // Skip duplicates
      }
      seenIds.add(itemId);
      const record = item as Record<string, unknown>;
      const score = clampScore(item.score);
      const explanation = item.explanation?.trim() ?? '';

      // Use restaurant_name directly from API response (most reliable - from database)
      // DO NOT extract from explanation as LLM may mention wrong restaurants
      const restaurantNameFromAPI = 
        typeof item.restaurant_name === 'string' && item.restaurant_name.trim() ? item.restaurant_name.trim() :
        typeof record.restaurant_name === 'string' && record.restaurant_name.trim() ? record.restaurant_name.trim() :
        undefined;

      // Try to get restaurant from other fields as fallback (but not from explanation)
      const restaurantField = (() => {
        if (restaurantNameFromAPI) {
          return restaurantNameFromAPI;
        }
        const direct = record.restaurant;
        if (typeof direct === 'string' && direct.trim()) {
          return direct.trim();
        }
        if (direct && typeof direct === 'object' && 'name' in direct) {
          const maybeName = (direct as Record<string, unknown>).name;
          if (typeof maybeName === 'string' && maybeName.trim()) {
            return maybeName.trim();
          }
        }
        return undefined;
      })();

      // ONLY use API restaurant_name or restaurant field - never extract from explanation
      // The explanation may contain incorrect restaurant names from LLM
      const restaurant = restaurantNameFromAPI || restaurantField || undefined;
      const highlights = splitHighlights(explanation).filter(
        (entry) => !entry.toLowerCase().startsWith('restaurant')
      );

      deduplicatedItems.push({
        id: String(item.item_id),
        name: item.name?.trim() || 'Recommended item',
        score,
        explanation,
        highlights,
        restaurant,
        restaurant_place_id: typeof item.restaurant_place_id === 'string' ? item.restaurant_place_id : undefined,
        description: typeof record.description === 'string' ? record.description : undefined,
        price:
          typeof record.price === 'number'
            ? record.price
            : typeof record.price === 'string'
              ? Number.parseFloat(record.price)
              : undefined,
        calories:
          typeof record.calories === 'number'
            ? record.calories
            : typeof record.calories === 'string'
              ? Number.parseFloat(record.calories)
              : undefined,
      });
    }
  }

  if ('recommendations' in data) {
    for (const item of data.recommendations) {
      const itemId = String(item.menu_item_id);
      if (seenIds.has(itemId)) {
        continue; // Skip duplicates
      }
      seenIds.add(itemId);
      const explanation = item.explanation?.trim() ?? '';

      // Use restaurant_name from API (database) - never extract from explanation
      // The LLM explanation may contain incorrect restaurant names
      const restaurantName = 
        (item.restaurant_name && typeof item.restaurant_name === 'string' && item.restaurant_name.trim()) 
          ? item.restaurant_name.trim() :
        (item.restaurant?.name && typeof item.restaurant.name === 'string' && item.restaurant.name.trim())
          ? item.restaurant.name.trim() :
        undefined;
      
      const highlights = splitHighlights(explanation).filter(
        (entry) => !entry.toLowerCase().startsWith('restaurant')
      );

      deduplicatedItems.push({
        id: String(item.item_id),
        name: item.name ?? 'Recommended item',
        score: clampScore(item.score),
        explanation,
        highlights,
        restaurant: restaurantName,
        restaurant_place_id: typeof item.restaurant_place_id === 'string' ? item.restaurant_place_id : undefined,
        description: item.menu_item?.description,
        price: item.menu_item?.price ?? undefined,
        calories: item.menu_item?.calories ?? undefined,
      });
    }
  }

  return deduplicatedItems;
};

interface Props {
  restaurant?: string;
  placeId?: string;
}

export default function RecommendationMapPreview({ restaurant, placeId }: Props) {
  // Debug logging
  console.log('[RecommendationMapPreview] Input:', { restaurant, placeId });
  
  // If we have place_id, use it directly (most reliable)
  if (placeId && placeId.trim().length > 0) {
    console.log('[RecommendationMapPreview] Using placeId:', placeId);
    const mapUrl = `https://www.google.com/maps/embed/v1/place?key=${
      import.meta.env.VITE_GOOGLE_MAPS_API_KEY
    }&q=place_id:${placeId}`;
    console.log('[RecommendationMapPreview] Map URL:', mapUrl);
    
    return (
      <div className="mt-3 space-y-2">
        <div className="flex items-center gap-1 text-xs text-muted-foreground">
          <MapPin className="h-3 w-3" />
          <span>Location preview</span>
        </div>

        <iframe
          width="100%"
          height="180"
          loading="lazy"
          allowFullScreen
          className="rounded-md border"
          src={mapUrl}
        />

        <a
          href={`https://www.google.com/maps/search/?api=1&query=place_id:${placeId}`}
          target="_blank"
          rel="noopener noreferrer"
          className="text-xs text-blue-600 underline"
        >
          View on Google Maps
        </a>
      </div>
    );
  }

  // Fallback: search by restaurant name if place_id not available
  if (!restaurant || !isValidRestaurantName(restaurant)) {
    console.log('[RecommendationMapPreview] Invalid or missing restaurant name:', restaurant);
    return null;
  }
  
  console.log('[RecommendationMapPreview] Searching for restaurant:', restaurant);
  const { data } = useGooglePlaceSearch(restaurant);
  console.log('[RecommendationMapPreview] Search result:', data);

  // Normalize: backend may return an array OR an object
  const place = Array.isArray(data) ? data[0] : data;
  console.log('[RecommendationMapPreview] Place data:', place);

  if (!place || !place.place_id || place.place_id.trim().length === 0) {
    console.log('[RecommendationMapPreview] No valid place_id found');
    return null;
  }

  const mapUrl = `https://www.google.com/maps/embed/v1/place?key=${
    import.meta.env.VITE_GOOGLE_MAPS_API_KEY
  }&q=place_id:${place.place_id}`;
  console.log('[RecommendationMapPreview] Fallback Map URL:', mapUrl);

  return (
    <div className="mt-3 space-y-2">
      <div className="flex items-center gap-1 text-xs text-muted-foreground">
        <MapPin className="h-3 w-3" />
        <span>Location preview</span>
      </div>

      <iframe
        width="100%"
        height="180"
        loading="lazy"
        allowFullScreen
        className="rounded-md border"
        src={mapUrl}
      />

      <a
        href={`https://www.google.com/maps/search/?api=1&query=place_id:${place.place_id}`}
        target="_blank"
        rel="noopener noreferrer"
        className="text-xs text-blue-600 underline"
      >
        View on Google Maps
      </a>
    </div>
  );
}

export function RecommendationCarousel({
  userId,
  initialFilters,
  initialMode = 'llm',
}: RecommendationCarouselProps) {
  const [mode, setMode] = useState<RecommendationMode>(initialMode);
  const [appliedFilters, setAppliedFilters] = useState<RecommendationFiltersPayload | undefined>(
    initialFilters
  );
  const [formState, setFormState] = useState<{
    diet: string[];
    cuisine: string[];
    priceRange: string;
  }>(() => ({
    diet: initialFilters?.diet ?? [],
    cuisine: initialFilters?.cuisine ?? [],
    priceRange: initialFilters?.price_range ?? '',
  }));
  const [openPlanner, setOpenPlanner] = useState(false);
  const [selectedMealName, setSelectedMealName] = useState<string | undefined>(undefined);
  const [selectedMealCalories, setSelectedMealCalories] = useState<string | undefined>(undefined);
  const [selectedMenuItemId, setSelectedMenuItemId] = useState<string | undefined>(undefined);

  // Track whether user has manually requested recommendations
  const [hasRequestedRecommendations, setHasRequestedRecommendations] = useState(false);

  const appliedOptions = useMemo(() => {
    const filters =
      appliedFilters && Object.keys(appliedFilters).length > 0 ? appliedFilters : undefined;
    return { mode, filters };
  }, [mode, appliedFilters]);

  const { data, isLoading, isError, error, refetch, isFetching } = useMealRecommendations(
    userId,
    appliedOptions,
    hasRequestedRecommendations // Only fetch when user requests
  );

  const recommendations = useMemo(() => normalizeRecommendationResponse(data), [data]);
  const [feedbackLoading, setFeedbackLoading] = useState<Record<string, boolean>>({});
  const [itemFeedback, setItemFeedback] = useState<Record<string, 'like' | 'dislike'>>({});

  // Fetch existing feedback when recommendations change
  useEffect(() => {
    const fetchFeedback = async () => {
      if (recommendations.length > 0 && userId) {
        const itemIds = recommendations.map((r) => r.id);
        try {
          const feedback = await recommendationApi.getFeedback(itemIds, 'meal');
          setItemFeedback(feedback);
        } catch (error) {
          console.error('Failed to fetch feedback:', error);
        }
      }
    };
    void fetchFeedback();
  }, [recommendations, userId]);

  const handleApplyFilters = () => {
    const filters: RecommendationFiltersPayload = {};

    if (formState.diet.length > 0) {
      filters.diet = formState.diet;
    }
    if (formState.cuisine.length > 0) {
      filters.cuisine = formState.cuisine;
    }
    if (formState.priceRange) {
      filters.price_range = formState.priceRange as RecommendationFiltersPayload['price_range'];
    }

    setAppliedFilters(Object.keys(filters).length > 0 ? filters : undefined);
  };

  const handleGetRecommendations = () => {
    handleApplyFilters();
    setHasRequestedRecommendations(true);
    if (hasRequestedRecommendations) {
      void refetch();
    }
  };

  const handleUpdateRecommendations = () => {
    handleApplyFilters();
    void refetch();
  };

  const handleRefresh = () => {
    void refetch();
  };

  const handleModeSelect = (nextMode: RecommendationMode) => {
    setMode(nextMode);
  };

  const handleFeedback = async (itemId: string, feedbackType: 'like' | 'dislike') => {
    setFeedbackLoading((prev) => ({ ...prev, [itemId]: true }));
    try {
      await recommendationApi.submitFeedback({
        item_id: itemId,
        item_type: 'meal',
        feedback_type: feedbackType,
      });
      // Update local feedback state to highlight the button
      setItemFeedback((prev) => ({
        ...prev,
        [itemId]: feedbackType,
      }));
      // Optionally refetch recommendations to see updated results
      // void refetch();
    } catch (error) {
      console.error('Failed to submit feedback:', error);
      // You could add toast notification here
    } finally {
      setFeedbackLoading((prev) => ({ ...prev, [itemId]: false }));
    }
  };

  const handleClearFilters = () => {
    setFormState({
      diet: [],
      cuisine: [],
      priceRange: '',
    });
  };

  const loadMealPlanner = (menuItemId: string, name: string, calories: string): void => {
    setSelectedMenuItemId(menuItemId);
    setSelectedMealName(name);
    setSelectedMealCalories(calories);
    setOpenPlanner(true);
  };

  // Initial state - user hasn't requested recommendations yet
  if (!hasRequestedRecommendations) {
    return (
      <Card className="border-gray-100 bg-linear-to-br from-white to-emerald-50/30 transition-shadow hover:shadow-lg">
        <CardHeader>
          <div className="flex items-center gap-2">
            <Sparkles className="h-5 w-5 text-emerald-600" />
            <CardTitle>Meal Recommendations</CardTitle>
          </div>
          <CardDescription>
            Get personalized meal suggestions based on your preferences
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="space-y-6">
            {/* Mode Selection */}
            <div className="space-y-2">
              <Label className="text-sm font-medium">Recommendation Engine</Label>
              <div className="flex gap-2">
                {(['llm', 'baseline'] as RecommendationMode[]).map((option) => (
                  <Button
                    key={option}
                    type="button"
                    size="sm"
                    variant={mode === option ? 'default' : 'outline'}
                    onClick={() => setMode(option)}
                    className={
                      mode === option
                        ? 'gap-2 bg-emerald-500 text-white hover:bg-emerald-600'
                        : 'gap-2 border-gray-300 text-gray-700 hover:bg-gray-50'
                    }
                  >
                    {option === 'llm' ? (
                      <>
                        <Bot className="h-4 w-4" />
                        AI Powered
                      </>
                    ) : (
                      <>
                        <Baseline className="h-4 w-4" />
                        Basic
                      </>
                    )}
                  </Button>
                ))}
              </div>
              <p className="text-xs text-muted-foreground">
                {mode === 'llm'
                  ? 'Use advanced AI to analyze your preferences and health profile'
                  : 'Quick recommendations based on basic criteria'}
              </p>
            </div>

            {/* Filters Form */}
            <div className="space-y-4">
              <div className="space-y-2">
                <Label>Dietary Restrictions</Label>
                <ToggleGroup
                  type="multiple"
                  value={formState.diet}
                  onValueChange={(value) => setFormState((prev) => ({ ...prev, diet: value }))}
                  className="flex-wrap justify-start"
                  variant="outline"
                  spacing={2}
                  size="sm"
                >
                  {COMMON_DIETS.map((diet) => (
                    <ToggleGroupItem
                      key={diet}
                      value={diet}
                      className="border-gray-300 text-gray-700 capitalize hover:bg-gray-50 data-[state=on]:border-emerald-500 data-[state=on]:bg-emerald-500 data-[state=on]:text-white"
                    >
                      {diet}
                    </ToggleGroupItem>
                  ))}
                </ToggleGroup>
              </div>

              <div className="space-y-2">
                <Label>Preferred Cuisines</Label>
                <ToggleGroup
                  type="multiple"
                  value={formState.cuisine}
                  onValueChange={(value) => setFormState((prev) => ({ ...prev, cuisine: value }))}
                  className="flex-wrap justify-start"
                  variant="outline"
                  spacing={2}
                  size="sm"
                >
                  {COMMON_CUISINES.map((cuisine) => (
                    <ToggleGroupItem
                      key={cuisine}
                      value={cuisine}
                      className="border-gray-300 text-gray-700 capitalize hover:bg-gray-50 data-[state=on]:border-emerald-500 data-[state=on]:bg-emerald-500 data-[state=on]:text-white"
                    >
                      {cuisine}
                    </ToggleGroupItem>
                  ))}
                </ToggleGroup>
              </div>

              <div className="space-y-2">
                <Label>Price Range</Label>
                <ToggleGroup
                  type="single"
                  value={formState.priceRange}
                  onValueChange={(value) =>
                    setFormState((prev) => ({ ...prev, priceRange: value || '' }))
                  }
                  className="flex-wrap justify-start"
                  variant="outline"
                  spacing={2}
                  size="sm"
                >
                  {priceRanges.map((range) => (
                    <ToggleGroupItem
                      key={range.value}
                      value={range.value}
                      className="border-gray-300 text-gray-700 hover:bg-gray-50 data-[state=on]:border-emerald-500 data-[state=on]:bg-emerald-500 data-[state=on]:text-white"
                    >
                      {range.label}
                    </ToggleGroupItem>
                  ))}
                </ToggleGroup>
              </div>

              <div className="flex gap-2">
                <Button
                  type="button"
                  variant="outline"
                  size="sm"
                  onClick={handleClearFilters}
                  disabled={
                    formState.diet.length === 0 &&
                    formState.cuisine.length === 0 &&
                    !formState.priceRange
                  }
                  className="gap-2 border-gray-300 text-gray-700 hover:bg-gray-50"
                >
                  <Eraser className="h-4 w-4" />
                  Clear Filters
                </Button>
              </div>
            </div>

            {/* Get Recommendations Button */}
            <div className="border-t border-gray-200 pt-4">
              <Button
                onClick={handleGetRecommendations}
                className="w-full bg-emerald-500 text-white shadow-lg hover:bg-emerald-600"
                size="lg"
              >
                <ChefHat className="mr-2 h-5 w-5" />
                Get Recommendations
              </Button>
              <p className="mt-2 text-center text-xs text-muted-foreground">
                Click to get personalized meal suggestions
              </p>
            </div>
          </div>
        </CardContent>
      </Card>
    );
  }

  // Loading state
  if (isLoading) {
    return (
      <Card className="bg-linear-to-br from-white to-emerald-50/30 transition-shadow hover:shadow-lg">
        <CardHeader>
          <div className="flex items-center gap-2">
            <Sparkles className="h-5 w-5 text-emerald-600" />
            <CardTitle>Meal Recommendations</CardTitle>
          </div>
          <CardDescription>Loading personalized recommendations...</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="flex items-center justify-center py-12">
            <RefreshCw className="h-8 w-8 animate-spin text-gray-500" />
          </div>
        </CardContent>
      </Card>
    );
  }

  // Error state
  if (isError) {
    return (
      <Card className="bg-linear-to-br from-white to-emerald-50/30 transition-shadow hover:shadow-lg">
        <CardHeader>
          <div className="flex items-center gap-2">
            <Sparkles className="h-5 w-5 text-emerald-600" />
            <CardTitle>Meal Recommendations</CardTitle>
          </div>
          <CardDescription>Unable to load recommendations</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="flex flex-col items-center gap-4 py-8">
            <p className="text-sm text-red-600">
              {error instanceof Error ? error.message : 'Failed to load recommendations'}
            </p>
            <Button
              onClick={handleRefresh}
              variant="outline"
              size="sm"
              className="border-gray-300 text-gray-700 hover:bg-gray-50"
            >
              <RefreshCw className="mr-2 h-4 w-4" />
              Try Again
            </Button>
          </div>
        </CardContent>
      </Card>
    );
  }

  // Empty state
  if (recommendations.length === 0) {
    return (
      <Card className="bg-linear-to-br from-white to-emerald-50/30">
        <CardHeader>
          <div className="flex items-center gap-2">
            <Sparkles className="h-5 w-5 text-emerald-600" />
            <CardTitle>Meal Recommendations</CardTitle>
          </div>
          <CardDescription>No recommendations available yet</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="flex flex-col items-center gap-4 py-8">
            <UtensilsCrossed className="h-12 w-12 text-emerald-400" />
            <p className="text-center text-sm text-muted-foreground">
              No meals match your current filters. Try adjusting your preferences or clearing
              filters.
            </p>
            <div className="flex gap-2">
              <Button
                onClick={() => {
                  setHasRequestedRecommendations(false);
                  handleClearFilters();
                }}
                variant="default"
                size="sm"
                className="gap-2 bg-emerald-500 text-white hover:bg-emerald-600"
              >
                <Eraser className="h-4 w-4" />
                Adjust Filters
              </Button>
              <Button
                onClick={handleRefresh}
                variant="outline"
                size="sm"
                className="border-gray-300 text-gray-700 hover:bg-gray-50"
              >
                <RefreshCw className="mr-2 h-4 w-4" />
                Try Again
              </Button>
            </div>
          </div>
        </CardContent>
      </Card>
    );
  }

  return (
    <>
      <Card className="border-gray-100 bg-linear-to-br from-white to-emerald-50/30 transition-shadow hover:shadow-lg">
        <CardHeader>
          <div className="flex flex-col gap-4 lg:flex-row lg:items-center lg:justify-between">
            <div className="flex items-start gap-3">
              <Sparkles className="mt-1 h-5 w-5 text-emerald-600" />
              <div>
                <CardTitle>Meal Recommendations</CardTitle>
                <CardDescription>
                  Personalized suggestions powered by{' '}
                  {mode === 'llm' ? 'LLM insights' : 'baseline heuristics'}
                </CardDescription>
              </div>
            </div>
            <div className="flex flex-wrap items-center gap-2">
              <span className="text-xs font-medium text-gray-600 uppercase">Engine</span>
              {(['llm', 'baseline'] as RecommendationMode[]).map((option) => (
                <Button
                  key={option}
                  type="button"
                  size="sm"
                  variant={mode === option ? 'default' : 'outline'}
                  onClick={() => handleModeSelect(option)}
                  className={
                    mode === option
                      ? 'gap-2 bg-emerald-500 text-white hover:bg-emerald-600'
                      : 'gap-2 border-gray-300 text-gray-700 hover:bg-gray-50'
                  }
                >
                  {option === 'llm' ? (
                    <>
                      <Bot className="h-4 w-4" />
                      AI
                    </>
                  ) : (
                    <>
                      <Baseline className="h-4 w-4" />
                      Baseline
                    </>
                  )}
                </Button>
              ))}
              <Button
                type="button"
                variant="ghost"
                size="sm"
                onClick={handleRefresh}
                aria-label="Refresh recommendations"
                className="text-emerald-700 hover:bg-emerald-50"
              >
                <RefreshCw className={`h-4 w-4 ${isFetching ? 'animate-spin' : ''}`} />
              </Button>
            </div>
          </div>
        </CardHeader>
        <CardContent className="space-y-6">
          <div className="grid gap-4 rounded-lg bg-white/80" aria-label="Recommendation filters">
            <div className="space-y-2">
              <Label>Dietary Restrictions</Label>
              <ToggleGroup
                type="multiple"
                value={formState.diet}
                onValueChange={(value) => setFormState((prev) => ({ ...prev, diet: value }))}
                className="flex-wrap justify-start"
                variant="outline"
                spacing={2}
                size="sm"
              >
                {COMMON_DIETS.map((diet) => (
                  <ToggleGroupItem
                    key={diet}
                    value={diet}
                    className="border-gray-300 text-gray-700 capitalize hover:bg-gray-50 data-[state=on]:border-emerald-500 data-[state=on]:bg-emerald-500 data-[state=on]:text-white"
                  >
                    {diet}
                  </ToggleGroupItem>
                ))}
              </ToggleGroup>
            </div>

            <div className="space-y-2">
              <Label>Preferred Cuisines</Label>
              <ToggleGroup
                type="multiple"
                value={formState.cuisine}
                onValueChange={(value) => setFormState((prev) => ({ ...prev, cuisine: value }))}
                className="flex-wrap justify-start"
                variant="outline"
                spacing={2}
                size="sm"
              >
                {COMMON_CUISINES.map((cuisine) => (
                  <ToggleGroupItem
                    key={cuisine}
                    value={cuisine}
                    className="border-gray-300 text-gray-700 capitalize hover:bg-gray-50 data-[state=on]:border-emerald-500 data-[state=on]:bg-emerald-500 data-[state=on]:text-white"
                  >
                    {cuisine}
                  </ToggleGroupItem>
                ))}
              </ToggleGroup>
            </div>

            <div className="grid gap-4 md:grid-cols-2">
              <div className="space-y-2">
                <Label>Price Range</Label>
                <ToggleGroup
                  type="single"
                  value={formState.priceRange}
                  onValueChange={(value) =>
                    setFormState((prev) => ({ ...prev, priceRange: value || '' }))
                  }
                  className="flex-wrap justify-start"
                  variant="outline"
                  spacing={2}
                  size="sm"
                >
                  {priceRanges.map((range) => (
                    <ToggleGroupItem
                      key={range.value}
                      value={range.value}
                      className="border-gray-300 text-gray-700 hover:bg-gray-50 data-[state=on]:border-emerald-500 data-[state=on]:bg-emerald-500 data-[state=on]:text-white"
                    >
                      {range.label}
                    </ToggleGroupItem>
                  ))}
                </ToggleGroup>
              </div>

              <div className="flex items-end gap-2">
                <Button
                  type="button"
                  variant="outline"
                  size="sm"
                  onClick={handleClearFilters}
                  disabled={
                    formState.diet.length === 0 &&
                    formState.cuisine.length === 0 &&
                    !formState.priceRange
                  }
                  className="gap-2 border-gray-300 text-gray-700 hover:bg-gray-50"
                  aria-label="Clear filters"
                >
                  <Eraser className="h-4 w-4" />
                  Clear Filters
                </Button>
                <Button
                  type="button"
                  variant="default"
                  size="sm"
                  onClick={handleUpdateRecommendations}
                  className="gap-2 bg-emerald-500 text-white hover:bg-emerald-600"
                  aria-label="Update recommendations"
                >
                  <Sparkles className="h-4 w-4" />
                  Update
                </Button>
              </div>
            </div>
          </div>

          <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-3">
            {recommendations.map((item) => {
              const scorePercent = Math.round(item.score * 100);
              const metaBadges = [
                item.price !== undefined ? `$${item.price.toFixed(2)}` : undefined,
                item.calories !== undefined ? `${Math.round(item.calories)} kcal` : undefined,
              ].filter(Boolean) as string[];

              return (
                <div
                  key={item.id}
                  className="flex h-full flex-col rounded-lg border bg-card p-5 shadow-sm transition-shadow hover:shadow-md"
                >
                  <div className="mb-4 flex items-start justify-between gap-2">
                    <div>
                      <div className="mb-1 flex items-center gap-2">
                        <Badge
                          variant={scorePercent >= 80 ? 'default' : 'secondary'}
                          className={
                            scorePercent >= 80
                              ? 'bg-emerald-500 text-xs font-semibold text-white'
                              : 'bg-emerald-100 text-xs font-semibold text-emerald-800'
                          }
                        >
                          {scorePercent}% match
                        </Badge>
                      </div>
                      <p className="text-base font-semibold text-gray-900">{item.name}</p>
                      {item.restaurant && (
                        <p className="flex items-center gap-1 text-sm text-muted-foreground">
                          <ChefHat className="h-4 w-4 text-gray-500" />
                          {item.restaurant}
                        </p>
                      )}
                    </div>
                  </div>

                  <div className="mb-2 *:space-y-2">
                    <Progress
                      value={scorePercent}
                      className="h-2 bg-gray-100 [&>div]:bg-emerald-500"
                    />
                    <p className="text-xs text-muted-foreground">Explanation:</p>
                  </div>

                  {item.description && (
                    <p className="mb-3 line-clamp-3 text-sm text-muted-foreground">
                      {item.description}
                    </p>
                  )}

                  {metaBadges.length > 0 && (
                    <div className="mb-3 flex flex-wrap gap-2">
                      {metaBadges.map((label) => (
                        <Badge
                          key={`${item.id}-${label}`}
                          variant="secondary"
                          className="bg-emerald-50 text-gray-700"
                        >
                          {label}
                        </Badge>
                      ))}
                    </div>
                  )}

                  {item.highlights.length > 0 && (
                    <ul className="mb-4 space-y-1 text-sm text-muted-foreground">
                      {item.highlights.map((highlight, index) => (
                        <li key={`${item.id}-highlight-${index}`}>• {highlight}</li>
                      ))}
                    </ul>
                  )}
                
                  <div className="flex gap-2">
                    <Button
                      type="button"
                      variant={itemFeedback[item.id] === 'like' ? 'default' : 'outline'}
                      size="sm"
                      onClick={() => handleFeedback(item.id, 'like')}
                      disabled={feedbackLoading[item.id]}
                      className={`flex-1 gap-2 ${
                        itemFeedback[item.id] === 'like'
                          ? 'bg-green-500 text-white hover:bg-green-600 border-green-500'
                          : 'border-green-300 text-green-700 hover:bg-green-50'
                      }`}
                      aria-label="Like this recommendation"
                    >
                      <ThumbsUp className="h-4 w-4" />
                      Like
                    </Button>
                    <Button
                      type="button"
                      variant={itemFeedback[item.id] === 'dislike' ? 'default' : 'outline'}
                      size="sm"
                      onClick={() => handleFeedback(item.id, 'dislike')}
                      disabled={feedbackLoading[item.id]}
                      className={`flex-1 gap-2 ${
                        itemFeedback[item.id] === 'dislike'
                          ? 'bg-red-500 text-white hover:bg-red-600 border-red-500'
                          : 'border-red-300 text-red-700 hover:bg-red-50'
                      }`}
                      aria-label="Dislike this recommendation"
                    >
                      <ThumbsDown className="h-4 w-4" />
                      Dislike
                    </Button>
                  </div>
                  <Button
                    size="lg"
                    onClick={() => loadMealPlanner(item.id, item.name, item.calories !== undefined ? `${Math.round(item.calories)}` : '0')}
                    className="px-4 w-full gap-2 bg-emerald-600 hover:bg-emerald-700 margin-top:15px"
                    style={{marginTop:10}}
                  >
                    <Plus className="size-5" />
                    Schedule Meal
                  </Button>
                  {/* Google Maps Mini Preview */}
                  <RecommendationMapPreview restaurant={item.restaurant} placeId={item.restaurant_place_id} />

                  <div className="mt-auto flex items-center justify-between text-xs text-muted-foreground">
                    <span>Reasoned by {mode === 'llm' ? 'LLM' : 'baseline'} engine</span>
                    <span>{item.explanation ? 'Explainer available' : 'No explanation'}</span>
                  </div>
                </div>
              );
            })}
          </div>
        </CardContent>
      </Card>
      <QuickMealPlanner open={openPlanner} onOpenChange={setOpenPlanner} mealName={selectedMealName} mealCalories={selectedMealCalories} menuItemId={selectedMenuItemId} />
    </>
  );
}