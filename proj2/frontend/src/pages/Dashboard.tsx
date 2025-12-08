import { useNavigate } from 'react-router';
import { useEffect, useState } from 'react';
import { DashboardNavbar } from '@/components/DashboardNavbar';
import { WellnessOverviewCard } from '@/components/wellness';
import { RecommendationCarousel } from '@/components/recommendations/RecommendationCarousel';
import { ScheduledMealsWidget } from '@/components/dashboard/ScheduledMealsWidget';
import {
  DailyCalorieGoal,
  MacronutrientBalance,
  LogMealWidget,
  MealsLoggedWidget,
  HealthProfileOverview,
} from '@/components/dashboard';
import apiClient, { getAuthToken } from '@/lib/api';

function Dashboard() {
  const navigate = useNavigate();
  const [isVerifying, setIsVerifying] = useState(true);
  const [userId, setUserId] = useState<string | null>(null);

  // Verify if token is valid
  useEffect(() => {
    const verifyToken = async () => {
      const token = getAuthToken();

      // If no token, redirect to home page
      if (!token) {
        navigate('/404-not-found');
        return;
      }

      try {
        // Call user endpoint to check if token is valid
        const response = await apiClient.get('/users/me');
        setUserId(response.data.id);
        setIsVerifying(false);
      } catch {
        // Token is invalid, clear and redirect
        navigate('/404-not-found');
      }
    };

    verifyToken();
  }, [navigate]);

  // Handler for navigating to detailed daily view
  const handleViewDailyDetails = () => {
    navigate('/daily-log');
  };

  // Show loading state while verifying
  if (isVerifying) {
    return (
      <div className="flex min-h-screen items-center justify-center bg-gray-50">
        <div className="text-center">
          <div className="mb-4 text-lg text-gray-600">Verifying...</div>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gray-50">
      <DashboardNavbar />

      {/* Main Content */}
      <main className="mx-auto max-w-7xl px-4 py-8 sm:px-6 lg:px-8">
        <div className="mb-6">
          <h2 className="mb-2 text-2xl font-bold text-gray-800">Welcome to Your Dashboard</h2>
          <p className="text-gray-600">
            Track your health, wellness, and nutrition journey all in one place
          </p>
        </div>

        {/* Primary Widgets Grid - Daily Goals & Nutrition */}
        <div className="mb-8 grid gap-4 lg:grid-cols-3">
          {/* Small Cards: Calorie Goal & Macros */}
          <DailyCalorieGoal />

          {/* Large Card: Meals Logged */}
          <MealsLoggedWidget onViewDetails={handleViewDailyDetails} />

          <MacronutrientBalance />
          <div className="lg:col-span-2">
            <div className="grid gap-4 md:grid-cols-3">
              <div className="md:col-span-2">
                <WellnessOverviewCard />
              </div>
              <div className="grid grid-rows-2 gap-2">
                <LogMealWidget />
                <HealthProfileOverview />
              </div>
            </div>
          </div>
        </div>

        {/* Meal Recommendations */}
        {userId && <RecommendationCarousel userId={userId} />}

        {/* Scheduled Meals Widget */}
        <div className="mb-8">
          <ScheduledMealsWidget />
        </div>

        {/* Other Dashboard Cards */}
      </main>
    </div>
  );
}

export default Dashboard;
