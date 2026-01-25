/**
 * Type definitions for API responses and domain models
 */

// User & Profile Types
export interface UserProfile {
  id: string;
  name: string;
  age: number;
  height: number;
  weight: number;
  gender: string;
  ref_photos: string[];
  is_full_body: boolean;
  body_detection_meta?: Record<string, unknown>;
  style_preference?: string;
  skin_tone?: string;
  created_at?: string;
}

export interface OnboardResponse {
  id: string;
  name: string;
  age: number;
  height_cm: number;
  weight_kg: number;
  gender: string;
  style_preference: string;
  is_full_body: boolean;
}

// Try-On Types
export interface TryOnResponse {
  result_url: string;
  processing_time: number;
  metadata: {
    model: string;
    resolution: string;
  };
}

// Wardrobe/History Types  
export interface WardrobeItem {
  id: number | string;
  user_id: string;
  image: string;
  type: string;
  created_at: string;
  date?: string; // Computed field
}

// Recommendation Types
export interface RecommendationItem {
  name: string;
  description: string;
  color_theory: string;
  items: string[];
  ebay_tags: string;
}

// Shop Types
export interface ShopItem {
  id: string;
  name: string;
  description: string;
  price: string;
  link: string;
  category: string;
}

// Body Generation Types
export interface BodyGenerationOption {
  id: string;
  url: string;
}

export interface BodyGenerationResponse {
  options: BodyGenerationOption[];
}

// Health Check Types
export interface HealthCheckResponse {
  status: 'healthy' | 'degraded' | 'unhealthy';
  checks: {
    compute?: {
      gpu_available: boolean;
      device: string;
    };
    database?: {
      status: string;
      error?: string;
    };
    ai_service?: {
      configured: boolean;
      masked_key?: string | null;
    };
    environment?: {
      log_level: string;
      cors_origins: number;
    };
  };
}
