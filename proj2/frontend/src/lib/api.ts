import axios from 'axios';

// Token management
export const TOKEN_KEY = 'eatsential_auth_token';
export const USER_ROLE_KEY = 'eatsential_user_role';

export const setAuthToken = (token: string) => {
  localStorage.setItem(TOKEN_KEY, token);
};

export const getAuthToken = (): string | null => {
  return localStorage.getItem(TOKEN_KEY);
};

export const clearAuthToken = () => {
  localStorage.removeItem(TOKEN_KEY);
  localStorage.removeItem(USER_ROLE_KEY);
};

export const setUserRole = (role: string) => {
  localStorage.setItem(USER_ROLE_KEY, role);
};

export const getUserRole = (): string | null => {
  return localStorage.getItem(USER_ROLE_KEY);
};

export const isAdmin = (): boolean => {
  return getUserRole() === 'admin';
};

// Create axios instance
const apiClient = axios.create({
  baseURL: '/api', // Use relative path, Vite will proxy automatically
  headers: {
    'Content-Type': 'application/json',
  },
});

// Request interceptor: automatically add JWT token
apiClient.interceptors.request.use(
  (config) => {
    const token = localStorage.getItem('eatsential_auth_token');
    if (token) {
      config.headers.Authorization = `Bearer ${token}`;
    }
    return config;
  },
  (error) => {
    return Promise.reject(error);
  }
);

// Response interceptor: handle 401 errors
apiClient.interceptors.response.use(
  (response) => {
    return response;
  },
  (error) => {
    if (error.response?.status === 401) {
      // Token is invalid or expired, clear and redirect to login page
      clearAuthToken();
      window.location.href = '/';
    }
    return Promise.reject(error);
  }
);

// User and Auth Types
export type UserRole = 'user' | 'admin';

export interface User {
  id: string;
  username: string;
  email: string;
  role: UserRole;
}

export interface LoginResponse extends User {
  access_token: string;
  token_type: string;
  message: string;
  has_completed_wizard: boolean;
}

export interface UserResponse extends User {
  message: string;
}

// Health Profile API Types
export type AllergySeverity = 'mild' | 'moderate' | 'severe' | 'life_threatening';
export type ActivityLevel = 'sedentary' | 'light' | 'moderate' | 'active' | 'very_active';
export type PreferenceType = 'diet' | 'cuisine' | 'ingredient' | 'preparation';

export interface Allergen {
  id: string;
  name: string;
  category: string;
  is_major_allergen: boolean;
  description?: string;
}

export interface UserAllergy {
  id: string;
  health_profile_id: string;
  allergen_id: string;
  severity: string;
  diagnosed_date?: string;
  reaction_type?: string;
  notes?: string;
  is_verified: boolean;
  created_at: string;
  updated_at: string;
}

export interface DietaryPreference {
  id: string;
  health_profile_id: string;
  preference_type: string;
  preference_name: string;
  is_strict: boolean;
  reason?: string;
  notes?: string;
  created_at: string;
  updated_at: string;
}

export interface HealthProfile {
  id: string;
  user_id: string;
  height_cm?: number;
  weight_kg?: number;
  activity_level?: string;
  metabolic_rate?: number;
  created_at: string;
  updated_at: string;
  allergies: UserAllergy[];
  dietary_preferences: DietaryPreference[];
}

// Health Profile API functions
export const healthProfileApi = {
  // Get user's health profile
  getProfile: () => apiClient.get<HealthProfile>('/health/profile'),

  // Create health profile
  createProfile: (data: {
    height_cm?: number;
    weight_kg?: number;
    activity_level?: ActivityLevel;
    metabolic_rate?: number;
  }) => apiClient.post<HealthProfile>('/health/profile', data),

  // Update health profile
  updateProfile: (data: {
    height_cm?: number;
    weight_kg?: number;
    activity_level?: ActivityLevel;
    metabolic_rate?: number;
  }) => apiClient.put<HealthProfile>('/health/profile', data),

  // Delete health profile
  deleteProfile: () => apiClient.delete('/health/profile'),

  // List all allergens
  listAllergens: () => apiClient.get<Allergen[]>('/health/allergens'),

  // Add allergy
  addAllergy: (data: {
    allergen_id: string;
    severity: AllergySeverity;
    diagnosed_date?: string;
    reaction_type?: string;
    notes?: string;
    is_verified?: boolean;
  }) => apiClient.post<UserAllergy>('/health/allergies', data),

  // Update allergy
  updateAllergy: (
    allergyId: string,
    data: {
      severity?: AllergySeverity;
      diagnosed_date?: string;
      reaction_type?: string;
      notes?: string;
      is_verified?: boolean;
    }
  ) => apiClient.put<UserAllergy>(`/health/allergies/${allergyId}`, data),

  // Delete allergy
  deleteAllergy: (allergyId: string) => apiClient.delete(`/health/allergies/${allergyId}`),

  // Add dietary preference
  addDietaryPreference: (data: {
    preference_type: PreferenceType;
    preference_name: string;
    is_strict?: boolean;
    reason?: string;
    notes?: string;
  }) => apiClient.post<DietaryPreference>('/health/dietary-preferences', data),

  // Update dietary preference
  updateDietaryPreference: (
    preferenceId: string,
    data: {
      preference_name?: string;
      is_strict?: boolean;
      reason?: string;
      notes?: string;
    }
  ) => apiClient.put<DietaryPreference>(`/health/dietary-preferences/${preferenceId}`, data),

  // Delete dietary preference
  deleteDietaryPreference: (preferenceId: string) =>
    apiClient.delete(`/health/dietary-preferences/${preferenceId}`),
};

export default apiClient;

// --- Health Profile API ---

export interface HealthProfileCreate {
  height_cm?: number;
  weight_kg?: number;
  activity_level?: 'sedentary' | 'light' | 'moderate' | 'active' | 'very_active';
  metabolic_rate?: number;
}

export interface HealthProfileResponse {
  id: string;
  user_id: string;
  height_cm?: number;
  weight_kg?: number;
  activity_level?: string;
  metabolic_rate?: number;
  created_at: string;
  updated_at: string;
  allergies: UserAllergyResponse[];
  dietary_preferences: DietaryPreferenceResponse[];
}

export interface UserAllergyCreate {
  allergen_id: string;
  severity: 'mild' | 'moderate' | 'severe' | 'life_threatening';
  diagnosed_date?: string;
  reaction_type?: string;
  notes?: string;
  is_verified?: boolean;
}

export interface UserAllergyResponse {
  id: string;
  health_profile_id: string;
  allergen_id: string;
  severity: string;
  diagnosed_date?: string;
  reaction_type?: string;
  notes?: string;
  is_verified: boolean;
  created_at: string;
  updated_at: string;
}

export interface DietaryPreferenceCreate {
  preference_type: 'diet' | 'cuisine' | 'ingredient' | 'preparation';
  preference_name: string;
  is_strict?: boolean;
  reason?: string;
  notes?: string;
}

export interface DietaryPreferenceResponse {
  id: string;
  health_profile_id: string;
  preference_type: string;
  preference_name: string;
  is_strict: boolean;
  reason?: string;
  notes?: string;
  created_at: string;
  updated_at: string;
}

export interface AllergenResponse {
  id: string;
  name: string;
  category: string;
  is_major_allergen: boolean;
  description?: string;
}

export interface AllergenCreate {
  name: string;
  category: string;
  is_major_allergen: boolean;
  description?: string;
}

export interface AllergenUpdate {
  name?: string;
  category?: string;
  is_major_allergen?: boolean;
  description?: string;
}

// Bulk import types
export interface AllergenBulkImportItem {
  name: string;
  category: string;
  is_major_allergen: boolean;
  description?: string;
}

export interface AllergenBulkImportResponse {
  imported: number;
  skipped: number;
  errors: string[];
}

// Audit log types
export interface AllergenAuditLog {
  id: string;
  allergen_id: string | null;
  allergen_name: string;
  action: string;
  admin_user_id: string;
  admin_username: string;
  changes: string | null;
  created_at: string;
}

export interface UserAuditLog {
  id: string;
  target_user_id: string;
  target_username: string;
  action: string;
  admin_user_id: string;
  admin_username: string;
  changes: string | null;
  created_at: string;
}

// Create health profile
export const createHealthProfile = async (
  data: HealthProfileCreate
): Promise<HealthProfileResponse> => {
  const response = await apiClient.post('/health/profile', data);
  return response.data;
};

// Get health profile
export const getHealthProfile = async (): Promise<HealthProfileResponse> => {
  const response = await apiClient.get('/health/profile');
  return response.data;
};

// Add allergy
export const addAllergy = async (data: UserAllergyCreate): Promise<UserAllergyResponse> => {
  const response = await apiClient.post('/health/allergies', data);
  return response.data;
};

// Add dietary preference
export const addDietaryPreference = async (
  data: DietaryPreferenceCreate
): Promise<DietaryPreferenceResponse> => {
  const response = await apiClient.post('/health/dietary-preferences', data);
  return response.data;
};

// Get all allergens from database
export const getAllergens = async (): Promise<AllergenResponse[]> => {
  const response = await apiClient.get('/health/allergens');
  return response.data;
};

// Admin User Management Types
export interface UserListItem {
  id: string;
  username: string;
  email: string;
  role: string;
  account_status: string;
  email_verified: boolean;
  created_at: string;
  updated_at: string;
}

export interface UserDetailData {
  id: string;
  username: string;
  email: string;
  role: string;
  account_status: string;
  email_verified: boolean;
  verification_token_expires?: string | null;
  created_at: string;
  updated_at: string;
}

export interface UserProfileUpdate {
  username?: string;
  role?: string;
  account_status?: string;
  email_verified?: boolean;
}

// Auth API functions
export const getCurrentUser = async (): Promise<UserResponse> => {
  const response = await apiClient.get('/users/me');
  return response.data;
};

// Admin API functions
export const adminApi = {
  // Get all users (admin only)
  getAllUsers: async (): Promise<UserListItem[]> => {
    const response = await apiClient.get('/users/admin/users');
    return response.data;
  },

  // Get user details (admin only)
  getUserDetails: async (userId: string): Promise<UserDetailData> => {
    const response = await apiClient.get(`/users/admin/users/${userId}`);
    return response.data;
  },

  // Update user profile (admin only)
  updateUser: async (userId: string, data: UserProfileUpdate): Promise<UserDetailData> => {
    const response = await apiClient.put(`/users/admin/users/${userId}`, data);
    return response.data;
  },

  // Allergen management (admin only)
  getAllergens: async (): Promise<AllergenResponse[]> => {
    const response = await apiClient.get('/health/allergens');
    return response.data;
  },

  createAllergen: async (data: AllergenCreate): Promise<AllergenResponse> => {
    const response = await apiClient.post('/health/admin/allergens', data);
    return response.data;
  },

  updateAllergen: async (allergenId: string, data: AllergenUpdate): Promise<AllergenResponse> => {
    const response = await apiClient.put(`/health/admin/allergens/${allergenId}`, data);
    return response.data;
  },

  deleteAllergen: async (allergenId: string): Promise<void> => {
    await apiClient.delete(`/health/admin/allergens/${allergenId}`);
  },

  // Bulk operations
  bulkImportAllergens: async (
    allergens: AllergenBulkImportItem[]
  ): Promise<AllergenBulkImportResponse> => {
    const response = await apiClient.post('/health/admin/allergens/bulk', {
      allergens,
    });
    return response.data;
  },

  // Search allergens
  searchAllergens: async (params: {
    name?: string;
    category?: string;
    is_major_allergen?: boolean;
    skip?: number;
    limit?: number;
  }): Promise<AllergenResponse[]> => {
    const response = await apiClient.get('/health/admin/allergens/search', {
      params,
    });
    return response.data;
  },

  // Export allergens
  exportAllergens: async (format: 'json' | 'csv'): Promise<Blob> => {
    const response = await apiClient.get('/health/admin/allergens/export', {
      params: { format },
      responseType: 'blob',
    });
    return response.data;
  },

  // Get allergen audit logs
  getAllergenAuditLogs: async (allergenId?: string, limit = 100): Promise<AllergenAuditLog[]> => {
    const response = await apiClient.get('/health/admin/allergens/audit-logs', {
      params: { allergen_id: allergenId, limit },
    });
    return response.data;
  },

  // Get user audit logs
  getUserAuditLogs: async (userId: string, limit = 100): Promise<UserAuditLog[]> => {
    const response = await apiClient.get(`/users/admin/users/${userId}/audit-logs`, {
      params: { limit },
    });
    return response.data;
  },

  // Get all user audit logs
  getAllUserAuditLogs: async (limit = 100): Promise<UserAuditLog[]> => {
    const response = await apiClient.get('/users/admin/audit-logs', {
      params: { limit },
    });
    return response.data;
  },
};

// Mental Wellness API Types
export interface MoodLogCreate {
  occurred_at: string; // ISO8601 with timezone (e.g., new Date().toISOString())
  mood_score: number; // 1-10
  notes?: string;
}

export interface StressLogCreate {
  occurred_at: string; // ISO8601 with timezone (e.g., new Date().toISOString())
  stress_level: number; // 1-10
  triggers?: string;
  notes?: string;
}

export interface SleepLogCreate {
  occurred_at: string; // ISO8601 with timezone (e.g., new Date().toISOString())
  duration_hours: number; // Sleep duration in hours
  quality_score: number; // Sleep quality from 1 to 10
  notes?: string;
}

export interface GoalCreate {
  goal_type: 'nutrition' | 'wellness';
  target_type: string; // e.g., "calories", "mood_score", "steps"
  target_value: number;
  start_date: string; // YYYY-MM-DD format
  end_date: string; // YYYY-MM-DD format (was target_date)
  notes?: string;
}

export interface WellnessLogResponse {
  id: string;
  user_id: string;
  mood_score?: number;
  stress_level?: number;
  duration_hours?: number; // Sleep duration in hours
  quality_score?: number; // Sleep quality score 1-10
  notes?: string;
  triggers?: string;
  occurred_at_utc: string; // UTC datetime from backend
  created_at: string;
  updated_at: string;
}

export interface WellnessLogsResponse {
  mood_logs: WellnessLogResponse[];
  stress_logs: WellnessLogResponse[];
  sleep_logs: WellnessLogResponse[];
  total_count: number;
}

export interface GoalResponse {
  id: string;
  user_id: string;
  goal_type: string;
  target_type: string;
  target_value: number;
  current_value: number;
  start_date: string;
  end_date: string;
  status: string;
  notes?: string;
  completion_percentage: number;
  is_active: boolean;
  created_at: string;
  updated_at: string;
}

export interface GoalListResponse {
  goals: GoalResponse[];
  total: number;
  page: number;
  page_size: number;
}

// Mental Wellness API
export const wellnessApi = {
  // Mood logs
  createMoodLog: async (data: MoodLogCreate): Promise<WellnessLogResponse> => {
    const response = await apiClient.post('/wellness/mood-logs', data);
    return response.data;
  },

  // Stress logs
  createStressLog: async (data: StressLogCreate): Promise<WellnessLogResponse> => {
    const response = await apiClient.post('/wellness/stress-logs', data);
    return response.data;
  },

  // Sleep logs
  createSleepLog: async (data: SleepLogCreate): Promise<WellnessLogResponse> => {
    const response = await apiClient.post('/wellness/sleep-logs', data);
    return response.data;
  },

  // Get all wellness logs
  getWellnessLogs: async (params?: {
    start_date?: string;
    end_date?: string;
    log_type?: string;
  }): Promise<WellnessLogResponse[]> => {
    const response = await apiClient.get<WellnessLogsResponse>('/wellness/logs', { params });

    // Flatten the response into a single array
    const { mood_logs = [], stress_logs = [], sleep_logs = [] } = response.data;

    // Combine all logs into a single array
    const allLogs: WellnessLogResponse[] = [];

    // Add mood logs
    mood_logs.forEach((log) => {
      allLogs.push({ ...log, mood_score: log.mood_score });
    });

    // Add stress logs
    stress_logs.forEach((log) => {
      allLogs.push({ ...log, stress_level: log.stress_level });
    });

    // Add sleep logs
    sleep_logs.forEach((log) => {
      allLogs.push({
        ...log,
        quality_score: log.quality_score,
        duration_hours: log.duration_hours,
      });
    });

    // Sort by date (newest first)
    return allLogs.sort(
      (a, b) => new Date(b.occurred_at_utc).getTime() - new Date(a.occurred_at_utc).getTime()
    );
  },

  // Goals
  createGoal: async (data: GoalCreate): Promise<GoalResponse> => {
    const response = await apiClient.post('/goals', data);
    return response.data;
  },

  getGoals: async (params?: {
    goal_type?: string;
    status?: string;
    page?: number;
    page_size?: number;
  }): Promise<GoalResponse[]> => {
    const response = await apiClient.get<GoalListResponse>('/goals', { params });
    // Extract the goals array from the paginated response
    return response.data.goals || [];
  },

  deleteGoal: async (goalId: string): Promise<void> => {
    await apiClient.delete(`/goals/${goalId}`);
  },
};

// Meal Logging API Types
export type MealTypeOption = 'breakfast' | 'lunch' | 'dinner' | 'snack';

export interface MealFoodItemInput {
  food_name: string;
  portion_size: number;
  portion_unit: string;
  calories?: number;
  protein_g?: number;
  carbs_g?: number;
  fat_g?: number;
}

export interface MealCreateRequest {
  meal_type: MealTypeOption;
  meal_time: string;
  notes?: string;
  photo_url?: string;
  food_items: MealFoodItemInput[];
}

export interface MealUpdate {
  meal_type?: MealTypeOption;
  meal_time?: string;
  notes?: string;
  photo_url?: string;
  food_items?: MealFoodItemInput[];
}

export interface MealFoodItemResponse {
  id: string;
  food_name: string;
  portion_size: number;
  portion_unit: string;
  calories?: number | null;
  protein_g?: number | null;
  carbs_g?: number | null;
  fat_g?: number | null;
  created_at: string;
}

export interface MealLogResponse {
  id: string;
  user_id: string;
  meal_type: MealTypeOption;
  meal_time: string;
  notes?: string | null;
  photo_url?: string | null;
  total_calories?: number | null;
  total_protein_g?: number | null;
  total_carbs_g?: number | null;
  total_fat_g?: number | null;
  food_items: MealFoodItemResponse[];
  created_at: string;
  updated_at: string;
}

export interface MealListResponse {
  meals: MealLogResponse[];
  total: number;
  page: number;
  page_size: number;
}

export interface MealListFilters {
  page?: number;
  page_size?: number;
  meal_type?: MealTypeOption | '';
  start_date?: string;
  end_date?: string;
}

export const mealApi = {
  logMeal: async (data: MealCreateRequest): Promise<MealLogResponse> => {
    const response = await apiClient.post<MealLogResponse>('/meals', data);
    return response.data;
  },
  getMeals: async (params?: MealListFilters): Promise<MealListResponse> => {
    const response = await apiClient.get<MealListResponse>('/meals', {
      params,
    });
    return response.data;
  },
};

// Orders API Types
export interface OrderCreate {
  menu_item_id: string;
  meal_id: string;
}

export interface OrderResponse {
  id: string;
  menu_item_id: string;
  meal_id: string;
}

export interface ScheduledOrderResponse {
  id: string;
  menu_item_id: string;
  meal_id: string;
  meal_type: string;
  meal_time: string; // ISO datetime string
  menu_item_name: string;
  restaurant_name: string;
  calories: number | null;
  price: number | null;
  portion_size: number;
  portion_unit: string;
}

// Orders API
export const ordersApi = {
  createOrder: async (data: OrderCreate): Promise<OrderResponse> => {
    const response = await apiClient.post<OrderResponse>('/orders', data);
    return response.data;
  },
  
  getScheduledOrders: async (days: number = 7): Promise<ScheduledOrderResponse[]> => {
    const response = await apiClient.get<ScheduledOrderResponse[]>('/orders/scheduled', {
      params: { days },
    });
    return response.data;
  },

  deleteOrder: async (orderId: string): Promise<void> => {
    await apiClient.delete(`/orders/${orderId}`);
  },

  updateOrderMeal: async (orderId: string, mealUpdate: MealUpdate): Promise<ScheduledOrderResponse> => {
    const response = await apiClient.put<ScheduledOrderResponse>(`/orders/${orderId}/meal`, mealUpdate);
    return response.data;
  },
};

// Meal Recommendation API Types
export type RecommendationMode = 'llm' | 'baseline';

export interface RecommendationFiltersPayload {
  diet?: string[];
  cuisine?: string[];
  price_range?: '$' | '$$' | '$$$' | '$$$$';
}

export interface RecommendationQueryOptions {
  mode?: RecommendationMode;
  filters?: RecommendationFiltersPayload;
  legacyConstraints?: Record<string, unknown>;
}

export interface RestaurantInfo {
  id: string;
  name: string;
  cuisine?: string;
  address?: string;
  is_active: boolean;
}

export interface MenuItemInfo {
  id: string;
  name: string;
  description?: string;
  price?: number;
  calories?: number;
}

export interface RecommendationItemLegacy {
  menu_item_id: string;
  score: number;
  explanation: string;
  menu_item?: MenuItemInfo;
  restaurant?: RestaurantInfo;
}

export interface MealRecommendationResponseLegacy {
  user_id?: string;
  recommendations: RecommendationItemLegacy[];
}

export interface RecommendationItemV2 {
  item_id: string;
  name: string;
  score: number;
  explanation: string;
  [key: string]: unknown;
}

export interface MealRecommendationResponseV2 {
  items: RecommendationItemV2[];
}

export type MealRecommendationResponse =
  | MealRecommendationResponseV2
  | MealRecommendationResponseLegacy;

// Feedback types
export interface FeedbackRequest {
  item_id: string;
  item_type: 'meal' | 'restaurant';
  feedback_type: 'like' | 'dislike';
  notes?: string;
}

export interface FeedbackResponse {
  id: string;
  item_id: string;
  item_type: string;
  feedback_type: string;
  created_at: string;
}

// Meal Recommendation API
export const recommendationApi = {
  getMealRecommendations: async (
    _userId: string,
    options?: RecommendationQueryOptions
  ): Promise<MealRecommendationResponse> => {
    const payload: Record<string, unknown> = {};

    // Default to LLM mode unless explicitly overridden
    payload.mode = options?.mode ?? 'llm';

    if (options?.filters) {
      payload.filters = options.filters;
    }

    if (options?.legacyConstraints) {
      payload.constraints = options.legacyConstraints;
    }

    const response = await apiClient.post<MealRecommendationResponse>(
      '/recommend/meal',
      Object.keys(payload).length > 0 ? payload : undefined
    );
    return response.data;
  },

  submitFeedback: async (request: FeedbackRequest): Promise<FeedbackResponse> => {
    const response = await apiClient.post<FeedbackResponse>('/recommend/feedback', request);
    return response.data;
  },

  getFeedback: async (itemIds: string[], itemType: 'meal' | 'restaurant'): Promise<Record<string, 'like' | 'dislike'>> => {
    const itemIdsParam = itemIds.join(',');
    const response = await apiClient.get<Record<string, 'like' | 'dislike'>>(
      `/recommend/feedback?item_ids=${encodeURIComponent(itemIdsParam)}&item_type=${itemType}`
    );
    return response.data;
  },
};

// GitHub API types
export interface GitHubIssue {
  number: number;
  title: string;
  state: string;
  created_at: string;
  updated_at: string;
  user: {
    login: string;
    avatar_url: string;
  };
  html_url: string;
  labels: Array<{
    name: string;
    color: string;
  }>;
  pull_request?: {
    url: string;
  };
}

// GitHub API client (no auth needed for public repos)
const GITHUB_REPO = 'Asoingbob225/CSC510'; // Your repo

export const githubApi = {
  getRecentIssues: async (limit = 5): Promise<GitHubIssue[]> => {
    const response = await axios.get(`https://api.github.com/repos/${GITHUB_REPO}/issues`, {
      params: {
        state: 'all',
        sort: 'updated',
        direction: 'desc',
        per_page: limit,
      },
    });
    return response.data;
  },

  getRecentPRs: async (limit = 5): Promise<GitHubIssue[]> => {
    const response = await axios.get(`https://api.github.com/repos/${GITHUB_REPO}/pulls`, {
      params: {
        state: 'all',
        sort: 'updated',
        direction: 'desc',
        per_page: limit,
      },
    });
    return response.data;
  },
};

// Chat API Types
export interface ChatRequest {
  message: string;
  session_id?: string;
}

export interface ChatResponse {
  response: string;
  session_id: string;
}

export interface ChatMessageResponse {
  role: string;
  content: string;
  id: string;
  created_at: string;
}

export interface ChatSessionResponse {
  id: string;
  title?: string;
  created_at: string;
  updated_at: string;
  messages: ChatMessageResponse[];
}

export const chatApi = {
  // Send message
  sendMessage: async (data: ChatRequest): Promise<ChatResponse> => {
    const token = localStorage.getItem("eatsential_auth_token");

    const response = await apiClient.post<ChatResponse>('/chat/', data, {
      headers: {
        Authorization: `Bearer ${token}`,
      },
    });
    return response.data;
  },

  // Get all sessions
  getSessions: async (): Promise<ChatSessionResponse[]> => {
    const token = localStorage.getItem("eatsential_auth_token");

    const response = await apiClient.get<ChatSessionResponse[]>('/chat/sessions', {
      headers: {
        Authorization: `Bearer ${token}`,
      },
    });
    return response.data;
  },

  // Get specific session
  getSession: async (sessionId: string): Promise<ChatSessionResponse> => {
    const token = localStorage.getItem("eatsential_auth_token");

    const response = await apiClient.get<ChatSessionResponse>(`/chat/sessions/${sessionId}`, {
      headers: {
        Authorization: `Bearer ${token}`,
      },
    });
    return response.data;
  },
};
