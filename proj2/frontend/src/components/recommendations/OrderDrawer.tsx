import { useState, useEffect } from 'react';
import { format } from 'date-fns';
import { useForm } from 'react-hook-form';
import { zodResolver } from '@hookform/resolvers/zod';
import { Calendar as CalendarIcon, Loader2 } from 'lucide-react';
import * as z from 'zod';
import { useQueryClient } from '@tanstack/react-query';

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
import { useLogOrder } from '@/hooks/useOrders';
import type { MealTypeOption } from '@/lib/api';
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

const quickMealSchema = z.object({
  meal_type: z.enum(['breakfast', 'lunch', 'dinner', 'snack']),
  meal_time: z
    .string()
    .min(1, 'Meal time is required')
    .refine((val) => !Number.isNaN(Date.parse(val)), 'Enter a valid date and time')
    .refine(
      (val) => {
        const selectedTime = new Date(val);
        const minimumTime = new Date();
        minimumTime.setMinutes(minimumTime.getMinutes() + 30);
        return selectedTime >= minimumTime;
      },
      'Meal time must be at least 30 minutes in the future to allow for prep and delivery'
    ),
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

type QuickMealFormValues = z.infer<typeof quickMealSchema>;

const getDefaultDateTime = () => format(new Date(), "yyyy-MM-dd'T'HH:mm");

interface QuickMealPlannerProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  mealName?: string;
  mealCalories?: string;
  menuItemId?: string;
  onOrderSuccess?: () => void;
}

export function QuickMealPlanner({ open, onOpenChange, mealName, mealCalories, menuItemId, onOrderSuccess }: QuickMealPlannerProps) {
  const queryClient = useQueryClient();
  const [isMealDateOpen, setIsMealDateOpen] = useState(false);

  const {
    register,
    handleSubmit,
    formState: { errors },
    setValue,
    watch,
    reset,
  } = useForm<QuickMealFormValues>({
    resolver: zodResolver(quickMealSchema),
    defaultValues: {
      meal_type: 'breakfast',
      meal_time: getDefaultDateTime(),
      food_name: mealName || '',
      portion_size: '',
      portion_unit: 'serving',
      calories: mealCalories || '0',
      protein_g: '',
      carbs_g: '',
      fat_g: '',
    },
  });

  // Sync prop changes to form when drawer opens
  useEffect(() => {
    if (open) {
      if (mealName) setValue('food_name', mealName);
      if (mealCalories) setValue('calories', mealCalories);
      setValue('portion_size', '1')
    }
  }, [open, mealName, mealCalories, setValue]);

  const mealTime = watch('meal_time');

  const { mutate: logOrder, isPending } = useLogOrder({
    onSuccess: () => {
      toast.success('Meal logged successfully!');
      reset({
        meal_type: 'breakfast',
        meal_time: getDefaultDateTime(),
        food_name: mealName || '',
        portion_size: '',
        portion_unit: 'serving',
        calories: mealCalories || '0',
        protein_g: '',
        carbs_g: '',
        fat_g: '',
      });
      onOrderSuccess?.();
      queryClient.invalidateQueries({ queryKey: ['scheduledOrders'] });
      onOpenChange(false);
    },
    onError: (error: unknown) => {
      const message =
        error instanceof Error ? error.message : 'Failed to log meal. Please try again.';
      toast.error(message);
    },
  });

  const parseMealTimeValue = (value?: string): Date | undefined => {
    if (!value) return undefined;
    const parsed = new Date(value);
    return Number.isNaN(parsed.getTime()) ? undefined : parsed;
  };

  const getMinimumDateTime = (): Date => {
    const now = new Date();
    now.setMinutes(now.getMinutes() + 30);
    return now;
  };

  const onSubmit = (data: QuickMealFormValues) => {
    const payload = {
      meal_type: data.meal_type,
      meal_time: new Date(data.meal_time).toISOString(),
      menu_item_id: menuItemId,
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
    };

    logOrder(payload);
  };

  return (
    <Drawer open={open} onOpenChange={onOpenChange}>
      <DrawerContent className="max-h-[95vh]">
        <DrawerHeader>
          <DrawerTitle>Schedule Meal</DrawerTitle>
          <DrawerDescription>Schedule a meal in advance </DrawerDescription>
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
                          const baseline = selectedDate ?? getMinimumDateTime();
                          next.setHours(baseline.getHours(), baseline.getMinutes(), 0, 0);
                          setValue('meal_time', format(next, "yyyy-MM-dd'T'HH:mm"));
                          setIsMealDateOpen(false);
                        }
                      }}
                      disabled={(date) => date < getMinimumDateTime()}
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
            </FieldGroup>
          </div>

          {/* Food Item */}
          <FieldGroup className="gap-1">
            <FieldLabel>Food Name</FieldLabel>
            <Field>
              <Input type="text" {...register('food_name')} disabled />
            </Field>
            {errors.food_name && <FieldError>{errors.food_name.message}</FieldError>}
          </FieldGroup>

          {/* Portion */}
          <FieldGroup className="gap-1">
            <FieldLabel>Portions</FieldLabel>
            <Field>
              <Input {...register('portion_size')} type="number" step="1" min="0"/>
            </Field>
            {errors.portion_size && <FieldError>{errors.portion_size.message}</FieldError>}
          </FieldGroup>

          {/* Nutrition Label */}
          <div className="gap-1 text-center">
            Nutrition Information per Portion (Optional)
          </div>

          {/* Nutrition */}
          <div className="grid gap-3 md:grid-cols-2">
            <FieldGroup className="gap-1">
              <FieldLabel>Calories </FieldLabel>
              <Field>
                <Input type="text" {...register('calories')} min="0"/>
              </Field>
              {errors.calories && <FieldError>{errors.calories.message}</FieldError>}
            </FieldGroup>

            <FieldGroup className="gap-1">
              <FieldLabel>Protein (g)</FieldLabel>
              <Field>
                <Input {...register('protein_g')} type="number" step="1" placeholder="0" min="0"/>
              </Field>
              {errors.protein_g && <FieldError>{errors.protein_g.message}</FieldError>}
            </FieldGroup>

            <FieldGroup className="gap-1">
              <FieldLabel>Carbs (g)</FieldLabel>
              <Field>
                <Input {...register('carbs_g')} type="number" step="1" placeholder="0" min="0"/>
              </Field>
              {errors.carbs_g && <FieldError>{errors.carbs_g.message}</FieldError>}
            </FieldGroup>

            <FieldGroup className="gap-1">
              <FieldLabel>Fat (g)</FieldLabel>
              <Field>
                <Input {...register('fat_g')} type="number" step="1" placeholder="0" min="0"/>
              </Field>
              {errors.fat_g && <FieldError>{errors.fat_g.message}</FieldError>}
            </FieldGroup>
          </div>

          <DrawerFooter className="px-0">
            <Button
              type="submit"
              disabled={isPending}
              className="w-full bg-emerald-600 hover:bg-emerald-700"
            >
              {isPending ? (
                <>
                  <Loader2 className="mr-2 size-4 animate-spin" />
                  Submitting...
                </>
              ) : (
                'Submit'
              )}
            </Button>
            <DrawerClose asChild>
              <Button variant="outline" type="button">
                Cancel
              </Button>
            </DrawerClose>
          </DrawerFooter>
        </form>
      </DrawerContent>
    </Drawer>
  );
}
