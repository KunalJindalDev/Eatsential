import { useMemo, useState } from 'react';
import { format } from 'date-fns';
import { Calendar as CalendarIcon, Filter, Loader2, RefreshCw } from 'lucide-react';

import { Button } from '@/components/ui/button';
import {
  Card,
  CardContent,
  CardDescription,
  CardFooter,
  CardHeader,
  CardTitle,
} from '@/components/ui/card';
import { Label } from '@/components/ui/label';
import { Popover, PopoverContent, PopoverTrigger } from '@/components/ui/popover';
import { Calendar } from '@/components/ui/calendar';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import { Separator } from '@/components/ui/separator';
import { toast } from 'sonner';

import { useMeals } from '@/hooks/useMeals';
import type { MealListFilters, MealLogResponse, MealTypeOption } from '@/lib/api';
import { cn } from '@/lib/utils';

const MEAL_TYPE_LABELS: Record<MealTypeOption, string> = {
  breakfast: 'Breakfast',
  lunch: 'Lunch',
  dinner: 'Dinner',
  snack: 'Snack',
};

const PAGE_SIZE = 10;

const mealDateFormatter = new Intl.DateTimeFormat(undefined, {
  dateStyle: 'medium',
  timeStyle: 'short',
});

const hasTimezone = (value: string) => /([zZ]|[+-]\d{2}:\d{2})$/.test(value);

const coerceToDate = (value: string): Date | undefined => {
  if (!value) return undefined;
  const normalized = hasTimezone(value) ? value : `${value}Z`;
  const parsed = new Date(normalized);
  return Number.isNaN(parsed.getTime()) ? undefined : parsed;
};

const toISOStringOrUndefined = (date?: Date, endOfDay = false) => {
  if (!date) return undefined;
  const boundary = new Date(date);
  if (endOfDay) {
    boundary.setHours(23, 59, 59, 999);
  } else {
    boundary.setHours(0, 0, 0, 0);
  }
  return boundary.toISOString();
};

const formatMealTime = (isoTimestamp: string) => {
  try {
    const parsed = coerceToDate(isoTimestamp);
    if (!parsed) {
      return isoTimestamp;
    }
    return mealDateFormatter.format(parsed);
  } catch {
    return isoTimestamp;
  }
};

export function MealHistory() {
  const [page, setPage] = useState(1);
  const [filters, setFilters] = useState<{
    meal_type: '' | MealTypeOption;
    start_date?: Date;
    end_date?: Date;
  }>({
    meal_type: '',
    start_date: undefined,
    end_date: undefined,
  });
  const [isStartDateOpen, setIsStartDateOpen] = useState(false);
  const [isEndDateOpen, setIsEndDateOpen] = useState(false);

  const queryFilters: MealListFilters = useMemo(() => {
    return {
      page,
      page_size: PAGE_SIZE,
      meal_type: filters.meal_type || undefined,
      start_date: toISOStringOrUndefined(filters.start_date),
      end_date: toISOStringOrUndefined(filters.end_date, true),
    };
  }, [filters, page]);

  const { data, isLoading, isFetching, isError, error, refetch } = useMeals(queryFilters, {
    enabled: true,
  });

  const handleMealTypeChange = (value: string) => {
    const normalized = value === 'all' ? '' : (value as MealTypeOption);
    setPage(1);
    setFilters((prev) => ({
      ...prev,
      meal_type: normalized,
    }));
  };

  const handleStartDateChange = (date: Date | undefined) => {
    setPage(1);
    setFilters((prev) => {
      const sanitizedEnd = prev.end_date && date && prev.end_date < date ? date : prev.end_date;
      return {
        ...prev,
        start_date: date,
        end_date: sanitizedEnd,
      };
    });
  };

  const handleEndDateChange = (date: Date | undefined) => {
    setPage(1);
    setFilters((prev) => {
      if (date && prev.start_date && date < prev.start_date) {
        return {
          ...prev,
          end_date: prev.start_date,
        };
      }
      return {
        ...prev,
        end_date: date,
      };
    });
  };

  const handleRefresh = () => {
    refetch();
    toast.success('Meal history refreshed');
  };

  const totalPages = data ? Math.max(1, Math.ceil(data.total / data.page_size)) : 1;
  const meals: MealLogResponse[] = data?.meals ?? [];
  const showEmptyState = !isLoading && meals.length === 0;

  return (
    <Card className="h-full">
      <CardHeader>
        <CardTitle>Meal History</CardTitle>
        <CardDescription>
          Review past meals, filter by type or date range, and monitor nutrition trends.
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-6">
        <form className="grid items-end gap-4 md:grid-cols-4" aria-label="Meal history filters">
          <div className="space-y-2">
            <Label htmlFor="meal_type" className="flex items-center gap-2">
              <Filter className="h-4 w-4 text-purple-500" />
              Meal type
            </Label>
            <Select value={filters.meal_type || 'all'} onValueChange={handleMealTypeChange}>
              <SelectTrigger id="meal_type" className="mb-0 w-full">
                <SelectValue placeholder="All meals" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="all">All meals</SelectItem>
                {Object.entries(MEAL_TYPE_LABELS).map(([value, label]) => (
                  <SelectItem key={value} value={value}>
                    {label}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>

          <div className="space-y-2">
            <Label htmlFor="start_date" className="flex items-center gap-2">
              <CalendarIcon className="h-4 w-4 text-purple-500" />
              Start date
            </Label>
            <Popover open={isStartDateOpen} onOpenChange={setIsStartDateOpen}>
              <PopoverTrigger asChild>
                <Button
                  id="start_date"
                  variant="outline"
                  className={cn(
                    'w-full justify-start text-left font-normal',
                    !filters.start_date && 'text-muted-foreground'
                  )}
                >
                  <CalendarIcon className="mr-2 h-4 w-4" />
                  {filters.start_date ? format(filters.start_date, 'PPP') : 'Pick a start date'}
                </Button>
              </PopoverTrigger>
              <PopoverContent className="w-auto p-0" align="start">
                <Calendar
                  mode="single"
                  selected={filters.start_date}
                  onSelect={(date) => {
                    handleStartDateChange(date);
                    setIsStartDateOpen(false);
                  }}
                  disabled={(date) => (filters.end_date ? date > filters.end_date : false)}
                  initialFocus
                />
              </PopoverContent>
            </Popover>
          </div>

          <div className="space-y-2">
            <Label htmlFor="end_date" className="flex items-center gap-2">
              <CalendarIcon className="h-4 w-4 text-purple-500" />
              End date
            </Label>
            <Popover open={isEndDateOpen} onOpenChange={setIsEndDateOpen}>
              <PopoverTrigger asChild>
                <Button
                  id="end_date"
                  variant="outline"
                  className={cn(
                    'w-full justify-start text-left font-normal',
                    !filters.end_date && 'text-muted-foreground'
                  )}
                >
                  <CalendarIcon className="mr-2 h-4 w-4" />
                  {filters.end_date ? format(filters.end_date, 'PPP') : 'Pick an end date'}
                </Button>
              </PopoverTrigger>
              <PopoverContent className="w-auto p-0" align="start">
                <Calendar
                  mode="single"
                  selected={filters.end_date}
                  onSelect={(date) => {
                    handleEndDateChange(date);
                    setIsEndDateOpen(false);
                  }}
                  disabled={(date) => (filters.start_date ? date < filters.start_date : false)}
                  initialFocus
                />
              </PopoverContent>
            </Popover>
          </div>

          <div className="flex items-end gap-3">
            <Button
              type="button"
              variant="outline"
              className="w-full sm:w-auto"
              onClick={handleRefresh}
              disabled={isFetching}
            >
              <RefreshCw className="mr-2 h-4 w-4" />
              Refresh
            </Button>
            <Button
              type="button"
              variant="ghost"
              className="w-full sm:w-auto"
              onClick={() => {
                setFilters({
                  meal_type: '',
                  start_date: undefined,
                  end_date: undefined,
                });
                setPage(1);
              }}
              disabled={!filters.meal_type && !filters.start_date && !filters.end_date}
            >
              Clear
            </Button>
          </div>
        </form>

        {isLoading ? (
          <div className="flex items-center justify-center py-10 text-purple-600">
            <Loader2 className="h-6 w-6 animate-spin" />
            <span className="ml-2 text-sm">Loading meal history...</span>
          </div>
        ) : isError ? (
          <div className="rounded-md border border-red-200 bg-red-50 p-4 text-red-600">
            <p className="font-medium">Unable to load meal history</p>
            <p className="mt-1 text-sm">{(error as Error)?.message ?? 'Please try again.'}</p>
            <Button className="mt-4" variant="outline" onClick={() => refetch()}>
              Try again
            </Button>
          </div>
        ) : showEmptyState ? (
          <div className="rounded-md border border-dashed border-gray-300 p-10 text-center text-gray-600">
            <p className="text-lg font-semibold">No meals logged for this range.</p>
            <p className="mt-2 text-sm">
              Adjust the filters or log a new meal using the Quick Meal Logger.
            </p>
          </div>
        ) : (
          <div className="space-y-5">
            {meals.map((meal) => (
              <div
                key={meal.id}
                className="rounded-lg border border-gray-200 bg-white p-5 shadow-sm transition hover:border-purple-200"
              >
                <div className="flex flex-col justify-between gap-4 sm:flex-row sm:items-start">
                  <div>
                    <p className="text-sm font-semibold tracking-wide text-purple-600 uppercase">
                      {MEAL_TYPE_LABELS[meal.meal_type]}
                    </p>
                    <p className="text-lg font-semibold text-gray-900">
                      {formatMealTime(meal.meal_time)}
                    </p>
                    {meal.notes && (
                      <p className="mt-2 text-sm text-gray-700">
                        <span className="font-medium text-gray-900">Notes:</span> {meal.notes}
                      </p>
                    )}
                  </div>
                  <div className="flex flex-col items-start text-sm text-gray-600 sm:items-end">
                    <p>
                      Total calories:{' '}
                      <span className="font-medium text-gray-900">
                        {typeof meal.total_calories === 'number'
                          ? `${Math.round(meal.total_calories)} kcal`
                          : '—'}
                      </span>
                    </p>
                    <p>
                      Protein:{' '}
                      <span className="font-medium text-gray-900">
                        {typeof meal.total_protein_g === 'number'
                          ? `${meal.total_protein_g.toFixed(1)} g`
                          : '—'}
                      </span>
                    </p>
                    <p>
                      Carbs:{' '}
                      <span className="font-medium text-gray-900">
                        {typeof meal.total_carbs_g === 'number'
                          ? `${meal.total_carbs_g.toFixed(1)} g`
                          : '—'}
                      </span>
                    </p>
                    <p>
                      Fat:{' '}
                      <span className="font-medium text-gray-900">
                        {typeof meal.total_fat_g === 'number'
                          ? `${meal.total_fat_g.toFixed(1)} g`
                          : '—'}
                      </span>
                    </p>
                  </div>
                </div>

                {meal.photo_url && (
                  <img
                    src={meal.photo_url}
                    alt={`Meal ${MEAL_TYPE_LABELS[meal.meal_type]} photo`}
                    className="mt-4 max-h-56 w-full rounded-md object-cover"
                  />
                )}

                <Separator className="my-4" />

                <div className="grid gap-3 sm:grid-cols-2">
                  {meal.food_items.map((item) => (
                    <div
                      key={item.id}
                      className="rounded-md border border-gray-200 bg-gray-50 p-3 text-sm"
                    >
                      <p className="font-medium text-gray-900">{item.food_name}</p>
                      <p className="mt-1 text-gray-600">
                        {item.portion_size} {item.portion_unit}{item.portion_size == 1 ? '': 's'} 
                      </p>
                      <div className="mt-2 flex flex-wrap gap-3 text-xs text-gray-500">
                        {typeof item.calories === 'number' && (
                          <span>{Math.round(item.calories)} kcal</span>
                        )}
                        {typeof item.protein_g === 'number' && (
                          <span>{item.protein_g.toFixed(1)} g protein</span>
                        )}
                        {typeof item.carbs_g === 'number' && (
                          <span>{item.carbs_g.toFixed(1)} g carbs</span>
                        )}
                        {typeof item.fat_g === 'number' && (
                          <span>{item.fat_g.toFixed(1)} g fat</span>
                        )}
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            ))}
          </div>
        )}
      </CardContent>
      <CardFooter className="flex flex-col gap-4 border-t border-gray-100 pt-4 sm:flex-row sm:items-center sm:justify-between">
        <div className="text-sm text-gray-600">
          Page {page} of {totalPages}
          {isFetching && !isLoading && (
            <span className="ml-2 inline-flex items-center gap-1 text-xs text-purple-600">
              <Loader2 className="h-3 w-3 animate-spin" /> Updating…
            </span>
          )}
        </div>
        <div className="flex gap-2">
          <Button
            type="button"
            variant="outline"
            onClick={() => setPage((prev) => Math.max(1, prev - 1))}
            disabled={page === 1 || isFetching}
          >
            Previous
          </Button>
          <Button
            type="button"
            variant="outline"
            onClick={() => setPage((prev) => Math.min(totalPages, prev + 1))}
            disabled={page >= totalPages || isFetching}
          >
            Next
          </Button>
        </div>
      </CardFooter>
    </Card>
  );
}
