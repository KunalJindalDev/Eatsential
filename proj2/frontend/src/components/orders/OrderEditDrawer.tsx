import { useState, useEffect } from 'react';
import { format, differenceInMinutes } from 'date-fns';
import { useForm } from 'react-hook-form';
import { zodResolver } from '@hookform/resolvers/zod';
import { Calendar as CalendarIcon, Loader2, Trash2 } from 'lucide-react';
import * as z from 'zod';

import {
  Drawer,
  DrawerClose,
  DrawerContent,
  DrawerDescription,
  DrawerFooter,
  DrawerHeader,
  DrawerTitle,
} from '@/components/ui/drawer';
import { Button } from '@/components/ui/button';
import { Field, FieldError, FieldGroup, FieldLabel } from '@/components/ui/field';
import { Input } from '@/components/ui/input';
import { Calendar } from '@/components/ui/calendar';
import { Popover, PopoverContent, PopoverTrigger } from '@/components/ui/popover';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import { toast } from 'sonner';
import { ordersApi, type ScheduledOrderResponse, type MealTypeOption } from '@/lib/api';
import { useMutation, useQueryClient } from '@tanstack/react-query';
import { cn } from '@/lib/utils';

const MEAL_TYPES: { label: string; value: MealTypeOption }[] = [
  { label: 'Breakfast', value: 'breakfast' },
  { label: 'Lunch', value: 'lunch' },
  { label: 'Dinner', value: 'dinner' },
  { label: 'Snack', value: 'snack' },
];

const numberString = z
  .string()
  .optional()
  .refine(
    (val) =>
      val === undefined ||
      val === '' ||
      (!Number.isNaN(Number(val)) && Number(val) >= 0 && Number(val) !== Infinity),
    'Enter a non-negative number'
  );

const orderEditSchema = z.object({
  meal_type: z.enum(['breakfast', 'lunch', 'dinner', 'snack']),
  meal_time: z
    .string()
    .min(1, 'Meal time is required')
    .refine((val) => !Number.isNaN(Date.parse(val)), 'Enter a valid date and time'),
  food_name: z.string().min(1, 'Food name is required'),
  portion_size: z
    .string()
    .min(1, 'Portion size is required')
    .refine(
      (val) => !Number.isNaN(Number(val)) && Number(val) > 0,
      'Portion size must be greater than 0'
    ),
  portion_unit: z.string().min(1, 'Portion unit is required'),
  calories: numberString,
  protein_g: numberString,
  carbs_g: numberString,
  fat_g: numberString,
});

type OrderEditFormValues = z.infer<typeof orderEditSchema>;

interface OrderEditDrawerProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  order: ScheduledOrderResponse;
}

export function OrderEditDrawer({ open, onOpenChange, order }: OrderEditDrawerProps) {
  const [isMealDateOpen, setIsMealDateOpen] = useState(false);
  const [mealTimeValidationError, setMealTimeValidationError] = useState<string | null>(null);
  const queryClient = useQueryClient();

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
  
  const {
    register,
    handleSubmit,
    formState: { errors },
    setValue,
    watch,
    reset,
  } = useForm<OrderEditFormValues>({
    resolver: zodResolver(orderEditSchema),
    defaultValues: {
      meal_type: order.meal_type as MealTypeOption,
      meal_time: format(new Date(order.meal_time), "yyyy-MM-dd'T'HH:mm"),
      food_name: order.menu_item_name,
      portion_size: order.portion_size.toString(),
      portion_unit: order.portion_unit,
      calories: order.calories?.toString() || '0',
      protein_g: '',
      carbs_g: '',
      fat_g: '',
    },
  });

  // Update form when order changes
  useEffect(() => {
    reset({
      meal_type: order.meal_type as MealTypeOption,
      meal_time: format(new Date(formatMealTime(order.meal_time)), "yyyy-MM-dd'T'HH:mm"),
      food_name: order.menu_item_name,
      portion_size: order.portion_size.toString(),
      portion_unit: order.portion_unit,
      calories: order.calories?.toString() || '0',
      protein_g: '',
      carbs_g: '',
      fat_g: '',
    });
  }, [order, reset]);

  const mealTime = watch('meal_time');
  const now = new Date();
  const originalMealTime = new Date(order.meal_time);

  // Check if order can be deleted (more than 30 minutes in future)
  const canDelete = differenceInMinutes(originalMealTime, now) > 30;

  // Mutation for updating order
  const { mutate: updateOrder, isPending: isUpdating } = useMutation({
    mutationFn: async (data: OrderEditFormValues) => {
      return ordersApi.updateOrderMeal(order.id, {
        meal_type: data.meal_type,
        meal_time: new Date(data.meal_time).toISOString(),
        food_items: [
          {
            food_name: data.food_name,
            portion_size: Number(data.portion_size),
            portion_unit: data.portion_unit,
            calories: data.calories ? Number(data.calories) : undefined,
            protein_g: data.protein_g ? Number(data.protein_g) : undefined,
            carbs_g: data.carbs_g ? Number(data.carbs_g) : undefined,
            fat_g: data.fat_g ? Number(data.fat_g) : undefined,
          },
        ],
      });
    },
    onSuccess: () => {
      toast.success('Order updated successfully!');
      queryClient.invalidateQueries({ queryKey: ['scheduledOrders'] });
      onOpenChange(false);
    },
    onError: (error: unknown) => {
      const message =
        error instanceof Error ? error.message : 'Failed to update order. Please try again.';
      toast.error(message);
    },
  });

  // Mutation for deleting order
  const { mutate: deleteOrder, isPending: isDeleting } = useMutation({
    mutationFn: async () => {
      return ordersApi.deleteOrder(order.id);
    },
    onSuccess: () => {
      toast.success('Order deleted successfully!');
      queryClient.invalidateQueries({ queryKey: ['scheduledOrders'] });
      onOpenChange(false);
    },
    onError: (error: unknown) => {
      const message =
        error instanceof Error ? error.message : 'Failed to delete order. Please try again.';
      toast.error(message);
    },
  });

  const parseMealTimeValue = (value?: string): Date | undefined => {
    if (!value) return undefined;
    const parsed = new Date(value);
    return Number.isNaN(parsed.getTime()) ? undefined : parsed;
  };

  const onSubmit = (data: OrderEditFormValues) => {
    updateOrder(data);
  };

  return (
    <Drawer open={open} onOpenChange={onOpenChange}>
      <DrawerContent className="max-h-[95vh]">
        <DrawerHeader>
          <DrawerTitle>Edit Order</DrawerTitle>
          <DrawerDescription>Update your scheduled meal order</DrawerDescription>
        </DrawerHeader>

        <form
          onSubmit={handleSubmit(onSubmit)}
          className="mx-auto w-full space-y-3 overflow-y-auto px-4 sm:max-w-2xl"
        >
          {/* Meal Type and Time */}
          <div className="grid gap-3 md:grid-cols-2">
            <FieldGroup className="gap-1">
              <FieldLabel>Meal Type</FieldLabel>
              <Field>
                <Select
                  value={watch('meal_type')}
                  onValueChange={(value) => setValue('meal_type', value as MealTypeOption)}
                >
                  <SelectTrigger>
                    <SelectValue placeholder="Select meal type" />
                  </SelectTrigger>
                  <SelectContent>
                    {MEAL_TYPES.map((type) => (
                      <SelectItem key={type.value} value={type.value}>
                        {type.label}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </Field>
              {errors.meal_type && <FieldError>{errors.meal_type.message}</FieldError>}
            </FieldGroup>

            <FieldGroup className="gap-1">
              <FieldLabel>Meal Time</FieldLabel>
              <div className="flex flex-col gap-3 sm:flex-row sm:items-center">
                <Popover open={isMealDateOpen} onOpenChange={setIsMealDateOpen}>
                  <PopoverTrigger asChild>
                    <Button
                      type="button"
                      variant="outline"
                      className={cn(
                        'w-full justify-start text-left font-normal sm:max-w-[220px]',
                        !mealTime && 'text-muted-foreground'
                      )}
                    >
                      <CalendarIcon className="mr-2 size-4" />
                      {mealTime ? format(new Date(mealTime), 'PPP') : 'Select date'}
                    </Button>
                  </PopoverTrigger>
                  <PopoverContent className="w-auto p-0" align="start">
                    <Calendar
                      mode="single"
                      selected={parseMealTimeValue(mealTime)}
                      onSelect={(date) => {
                        if (date) {
                          const selectedDate = parseMealTimeValue(mealTime);
                          const next = new Date(date);
                          const baseline = selectedDate ?? new Date();
                          next.setHours(baseline.getHours(), baseline.getMinutes(), 0, 0);
                          setMealTimeValidationError(null);
                          // Check if the meal_time is being reduced to below 30 minutes of the current time
                          if (differenceInMinutes(originalMealTime, next) > 0 && differenceInMinutes(next, now) < 30) {
                            setMealTimeValidationError('Cannot reduce meal time to below 30 minutes of the current time');
                          }
                          setValue('meal_time', format(next, "yyyy-MM-dd'T'HH:mm"));
                          setIsMealDateOpen(false);
                        }
                      }}
                      disabled={(date) => date < now}
                    />
                  </PopoverContent>
                </Popover>
                <div className="flex w-full flex-col gap-3 sm:w-auto sm:max-w-[180px]">
                  <Input
                    type="time"
                    step={60}
                    value={
                      mealTime ? format(parseMealTimeValue(mealTime) ?? new Date(), 'HH:mm') : ''
                    }
                    onChange={(event) => {
                      const time = event.target.value;
                      if (!time) {
                        const selectedDate = parseMealTimeValue(mealTime);
                        if (selectedDate) {
                          const reset = new Date(selectedDate);
                          reset.setHours(0, 0, 0, 0);
                          setValue('meal_time', format(reset, "yyyy-MM-dd'T'HH:mm"));
                        }
                        return;
                      }

                      const [hoursStr, minutesStr] = time.split(':');
                      const hours = Number(hoursStr);
                      const minutes = Number(minutesStr);

                      if (Number.isNaN(hours) || Number.isNaN(minutes)) {
                        return;
                      }

                      const base = parseMealTimeValue(mealTime) ?? new Date();
                      const next = new Date(base);
                      next.setHours(hours, minutes, 0, 0);
                      
                      setMealTimeValidationError(null);
                      // Check if the meal_time is being reduced to below 30 minutes of the current time
                      if (differenceInMinutes(originalMealTime, next) > 0 && differenceInMinutes(next, now) < 30) {
                        setMealTimeValidationError('Cannot reduce meal time to below 30 minutes of the current time');
                      }
                      setValue('meal_time', format(next, "yyyy-MM-dd'T'HH:mm"));
                    }}
                    className={cn(
                      'appearance-none bg-background [&::-webkit-calendar-picker-indicator]:hidden [&::-webkit-calendar-picker-indicator]:appearance-none',
                      'w-full sm:w-auto sm:max-w-[140px]'
                    )}
                  />
                </div>
              </div>
              {errors.meal_time && <FieldError>{errors.meal_time.message}</FieldError>}
              {mealTimeValidationError && <FieldError>{mealTimeValidationError}</FieldError>}
            </FieldGroup>
          </div>

          {/* Food Item */}
          <FieldGroup className="gap-1">
            <FieldLabel>Food Name</FieldLabel>
            <Field>
              <Input type="text" {...register('food_name')} />
            </Field>
            {errors.food_name && <FieldError>{errors.food_name.message}</FieldError>}
          </FieldGroup>

          {/* Nutrition Label */}
          <div className="gap-1 text-center">
            Nutrition Information per Portion (Optional)
          </div>

          {/* Portion */}
          <FieldGroup className="gap-1">
            <FieldLabel>Portions</FieldLabel>
            <Field>
              <Input {...register('portion_size')} type="number" step="0.5" min="0" />
            </Field>
            {errors.portion_size && <FieldError>{errors.portion_size.message}</FieldError>}
          </FieldGroup>

          {/* Nutrition */}
          <div className="grid gap-3 md:grid-cols-2">
            <FieldGroup className="gap-1">
              <FieldLabel>Calories</FieldLabel>
              <Field>
                <Input type="text" {...register('calories')} min="0" />
              </Field>
              {errors.calories && <FieldError>{errors.calories.message}</FieldError>}
            </FieldGroup>

            <FieldGroup className="gap-1">
              <FieldLabel>Protein (g)</FieldLabel>
              <Field>
                <Input {...register('protein_g')} type="number" step="1" placeholder="0" min="0" />
              </Field>
              {errors.protein_g && <FieldError>{errors.protein_g.message}</FieldError>}
            </FieldGroup>

            <FieldGroup className="gap-1">
              <FieldLabel>Carbs (g) </FieldLabel>
              <Field>
                <Input {...register('carbs_g')} type="number" step="1" placeholder="0" min="0" />
              </Field>
              {errors.carbs_g && <FieldError>{errors.carbs_g.message}</FieldError>}
            </FieldGroup>

            <FieldGroup className="gap-1">
              <FieldLabel>Fat (g) </FieldLabel>
              <Field>
                <Input {...register('fat_g')} type="number" step="1" placeholder="0" min="0" />
              </Field>
              {errors.fat_g && <FieldError>{errors.fat_g.message}</FieldError>}
            </FieldGroup>
          </div>

          <DrawerFooter className="flex flex-col gap-2 px-0 sm:flex-row">
            <Button
              type="submit"
              disabled={isUpdating || isDeleting || !!mealTimeValidationError}
              className="flex-1 bg-blue-600 hover:bg-blue-700"
            >
              {isUpdating ? (
                <>
                  <Loader2 className="mr-2 size-4 animate-spin" />
                  Updating...
                </>
              ) : (
                'Update Order'
              )}
            </Button>

            {canDelete && (
              <Button
                type="button"
                disabled={isUpdating || isDeleting}
                onClick={() => {
                  if (
                    confirm(
                      'Are you sure you want to delete this order? This action cannot be undone.'
                    )
                  ) {
                    deleteOrder();
                  }
                }}
                variant="destructive"
                className="flex-1"
              >
                {isDeleting ? (
                  <>
                    <Loader2 className="mr-2 size-4 animate-spin" />
                    Deleting...
                  </>
                ) : (
                  <>
                    <Trash2 className="mr-2 size-4" />
                    Delete Order
                  </>
                )}
              </Button>
            )}

            <DrawerClose asChild>
              <Button variant="outline" type="button" className="flex-1">
                Cancel
              </Button>
            </DrawerClose>
          </DrawerFooter>
        </form>
      </DrawerContent>
    </Drawer>
  );
}
