import { useEffect, useState } from 'react';
import { format, addDays, startOfDay } from 'date-fns';
import { Card } from '@/components/ui/card';
import { Skeleton } from '@/components/ui/skeleton';
import { useQuery } from '@tanstack/react-query';
import { ordersApi, type ScheduledOrderResponse } from '@/lib/api';
import { OrderEditDrawer } from '@/components/orders/OrderEditDrawer';

const MEAL_TYPE_LABELS: Record<string, string> = {
  breakfast: 'Breakfast',
  lunch: 'Lunch',
  dinner: 'Dinner',
  snack: 'Snack',
};

interface DayColumn {
  date: Date;
  orders: ScheduledOrderResponse[];
}

const hasTimezone = (value: string) => /([zZ]|[+-]\d{2}:\d{2})$/.test(value);

const coerceToDate = (value: string): Date | undefined => {
  if (!value) return undefined;
  const normalized = hasTimezone(value) ? value : `${value}Z`;
  const parsed = new Date(normalized);
  return Number.isNaN(parsed.getTime()) ? undefined : parsed;
};

const formatMealTime = (isoTimestamp: string) => {
  try {
    const parsed = coerceToDate(isoTimestamp);
    if (!parsed) {
      return isoTimestamp;
    }
    return parsed;
  } catch {
    return isoTimestamp;
  }
};

function getMealTypeColor(mealType: string): string {
  const colors: Record<string, string> = {
    breakfast: 'bg-orange-50 border-orange-200',
    lunch: 'bg-blue-50 border-blue-200',
    dinner: 'bg-purple-50 border-purple-200',
    snack: 'bg-green-50 border-green-200',
  };
  return colors[mealType] || 'bg-gray-50 border-gray-200';
}

function getMealTypeBadgeColor(mealType: string): string {
  const colors: Record<string, string> = {
    breakfast: 'bg-orange-200 text-orange-900',
    lunch: 'bg-blue-200 text-blue-900',
    dinner: 'bg-purple-200 text-purple-900',
    snack: 'bg-green-200 text-green-900',
  };
  return colors[mealType] || 'bg-gray-200 text-gray-900';
}

export function ScheduledMealsWidget() {
  const [days, setDays] = useState<DayColumn[]>([]);
  const [selectedOrder, setSelectedOrder] = useState<ScheduledOrderResponse | null>(null);
  const [editDrawerOpen, setEditDrawerOpen] = useState(false);

  const { data: orders, isLoading, error } = useQuery({
    queryKey: ['scheduledOrders'],
    queryFn: () => ordersApi.getScheduledOrders(7),
    refetchInterval: 60000, // Refetch every minute
  });

  useEffect(() => {
    if (!orders) return;

    // Group orders by day
    const daysMap = new Map<string, ScheduledOrderResponse[]>();
    const today = startOfDay(new Date());

    // Initialize 7 days starting from today
    for (let i = 0; i < 7; i++) {
      const date = addDays(today, i);
      const dateKey = format(date, 'yyyy-MM-dd');
      daysMap.set(dateKey, []);
    }

    // Add orders to their respective days
    orders.forEach((order) => {
      const orderDate = startOfDay(new Date(formatMealTime(order.meal_time)));
      const dateKey = format(orderDate, 'yyyy-MM-dd');
      if (daysMap.has(dateKey)) {
        daysMap.get(dateKey)!.push(order);
      }
    });

    // Sort orders within each day by meal_time
    daysMap.forEach((dayOrders) => {
      dayOrders.sort((a, b) => {
        return new Date(a.meal_time).getTime() - new Date(b.meal_time).getTime();
      });
    });

    // Convert map to array of day columns
    const dayColumns: DayColumn[] = [];
    for (let i = 0; i < 7; i++) {
      const date = addDays(today, i);
      const dateKey = format(date, 'yyyy-MM-dd');
      dayColumns.push({
        date,
        orders: daysMap.get(dateKey) || [],
      });
    }

    setDays(dayColumns);
  }, [orders]);

  if (isLoading) {
    return (
      <Card className="p-4 sm:p-6">
        <h2 className="mb-4 text-lg font-semibold">Scheduled Orders</h2>
        <div className="flex gap-4">
          {[...Array(7)].map((_, i) => (
            <div key={i} className="flex-1">
              <Skeleton className="mb-2 h-8 w-full" />
              <Skeleton className="h-32 w-full" />
            </div>
          ))}
        </div>
      </Card>
    );
  }

  if (error) {
    return (
      <Card className="p-4 sm:p-6">
        <h2 className="mb-4 text-lg font-semibold">Scheduled Orders</h2>
        <p className="text-center text-sm text-red-600">
          Failed to load scheduled orders.
        </p>
      </Card>
    );
  }

  return (
    <>
    <Card className="p-4 sm:p-6">
      <h2 className="mb-4 text-lg font-semibold">Scheduled Orders - Next 7 Days</h2>
      <div className="overflow-x-auto">
        <div className="flex gap-2 pb-4" style={{ minWidth: 'min-content' }}>
          {days.map((day) => (
            <div
              key={format(day.date, 'yyyy-MM-dd')}
              className="flex flex-col gap-2"
              style={{ minWidth: '160px' }}
            >
              {/* Day Header */}
              <div className="rounded-lg bg-gray-100 p-2 text-center">
                <div className="text-xs font-semibold text-gray-600">
                  {format(day.date, 'EEE')}
                </div>
                <div className="text-sm font-bold text-gray-900">
                  {format(day.date, 'MMM d')}
                </div>
              </div>

              {/* Orders for the day */}
              <div className="flex flex-col gap-2">
                {day.orders.length === 0 ? (
                  <div className="rounded-lg border-2 border-dashed border-gray-200 p-3 text-center">
                    <p className="text-xs text-gray-500">No meals scheduled</p>
                  </div>
                ) : (
                  day.orders.map((order) => (
                    <button
                      key={order.id}
                      onClick={() => {
                        setSelectedOrder(order);
                        setEditDrawerOpen(true);
                      }}
                      className={`w-full rounded-lg border-2 p-2 text-xs text-left transition-all hover:shadow-md ${getMealTypeColor(
                        order.meal_type
                      )}`}
                    >
                      {/* Meal Type Badge */}
                      <div className="mb-1 flex items-center gap-1">
                        <span
                          className={`inline-block rounded px-2 py-0.5 text-xs font-semibold ${getMealTypeBadgeColor(
                            order.meal_type
                          )}`}
                        >
                          {MEAL_TYPE_LABELS[order.meal_type]}
                        </span>
                      </div>

                      {/* Time */}
                      <div className="mb-1 font-semibold text-gray-700">
                        {format(formatMealTime(order.meal_time), 'hh:mm aa')}
                      </div>

                      {/* Menu Item Name */}
                      <div className="mb-1 line-clamp-2 font-medium text-gray-900">
                        {order.menu_item_name}
                      </div>

                      {/* Restaurant */}
                      <div className="mb-1 text-xs text-gray-600">
                        {order.restaurant_name}
                      </div>

                      {/* Portion */}
                      <div className="mb-1 text-xs font-medium text-gray-700">
                        {order.portion_size} {order.portion_unit}{order.portion_size == 1 ? '': 's'} 
                      </div>

                      {/* Nutrition Info */}
                      <div className="mt-auto flex items-center justify-between text-xs text-muted-foreground">
                        <span className="font-semibold">{order.calories !== undefined && order.calories !== null ? Math.round(order.calories) : 0 } cal</span>
                        <span className="font-semibold">${order.price !== undefined && order.price !== null ? order.price.toFixed(2) : 0.00}</span>
                      </div>
                    </button>
                  ))
                )}
              </div>
            </div>
          ))}
        </div>
      </div>
    </Card>

    {selectedOrder && (
      <OrderEditDrawer open={editDrawerOpen} onOpenChange={setEditDrawerOpen} order={selectedOrder} />
    )}
  </>
);
}

export default ScheduledMealsWidget;
