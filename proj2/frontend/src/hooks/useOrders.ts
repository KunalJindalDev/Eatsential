import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';

import {
  mealApi,
  ordersApi,
  type MealCreateRequest,
  type MealListFilters,
  type MealListResponse,
  type MealLogResponse,
  type OrderCreate,
  type OrderResponse,
} from '@/lib/api';

// Query keys for meal logging features
export const mealKeys = {
  all: ['meals'] as const,
  list: (filtersKey: string) => [...mealKeys.all, 'list', filtersKey] as const,
};

const sanitizeFilters = (filters?: MealListFilters) => {
  if (!filters) {
    return undefined;
  }

  const sanitized = Object.entries(filters).reduce<Record<string, string | number>>(
    (acc, [key, value]) => {
      if (value === undefined || value === null || value === '') {
        return acc;
      }
      acc[key] = value as string | number;
      return acc;
    },
    {}
  );

  return Object.keys(sanitized).length > 0 ? (sanitized as MealListFilters) : undefined;
};

/**
 * Fetch meal logs for the authenticated user with optional filters.
 * Utilises TanStack Query for caching and background refreshes.
 */
export function useOrders(filters?: MealListFilters, options?: { enabled?: boolean }) {
  const sanitized = sanitizeFilters(filters);
  const filtersKey = sanitized ? JSON.stringify(sanitized) : 'default';

  return useQuery<MealListResponse, Error>({
    queryKey: mealKeys.list(filtersKey),
    queryFn: async (): Promise<MealListResponse> => {
      const response = await mealApi.getMeals(sanitized);
      return response;
    },
    staleTime: 1000 * 30, // 30 seconds freshness
    enabled: options?.enabled ?? true,
  });
}

/**
 * Payload for creating both a meal and an order in a single operation
 */
export interface MealWithOrderPayload extends MealCreateRequest {
  menu_item_id?: string; // Optional menu item ID to link to order
}

interface UseLogOrderOptions {
  onSuccess?: (data: MealLogResponse) => void;
  onError?: (error: unknown) => void;
}

/**
 * Mutation hook for logging a new meal and optionally creating an order.
 * If menu_item_id is provided, creates both meal and order entries.
 * Automatically invalidates cached meal lists on success.
 */
export function useLogOrder(options?: UseLogOrderOptions) {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: async (payload: MealWithOrderPayload) => {
      // Create the meal first
      const mealData = await mealApi.logMeal(payload);

      // If menu_item_id is provided, create the order linking meal to menu item
      if (payload.menu_item_id && mealData.id) {
        try {
          await ordersApi.createOrder({
            menu_item_id: payload.menu_item_id,
            meal_id: mealData.id,
          });
        } catch (error) {
          // Log warning but don't fail - meal was created successfully
          console.warn('Failed to create order:', error);
        }
      }

      return mealData;
    },
    onSuccess: (data) => {
      queryClient.invalidateQueries({ queryKey: mealKeys.all });
      options?.onSuccess?.(data);
    },
    onError: (error) => {
      options?.onError?.(error);
    },
  });
}

interface UseCreateOrderOptions {
  onSuccess?: (data: OrderResponse) => void;
  onError?: (error: unknown) => void;
}

/**
 * Mutation hook for creating an order linking a menu item to a meal.
 * Automatically invalidates relevant queries on success.
 */
export function useCreateOrder(options?: UseCreateOrderOptions) {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: async (payload: OrderCreate) => ordersApi.createOrder(payload),
    onSuccess: (data) => {
      queryClient.invalidateQueries({ queryKey: mealKeys.all });
      options?.onSuccess?.(data);
    },
    onError: (error: unknown) => {
      options?.onError?.(error);
    },
  });
}
