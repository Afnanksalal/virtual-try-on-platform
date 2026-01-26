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
  personalImage?: {
    url: string;
    type: 'head-only' | 'full-body';
    uploadedAt: Date;
  } | null;
  preferences?: {
    ethnicity?: string;
    bodyType?: string;
    height?: number;
    weight?: number;
  };
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
export interface TryOnResult {
  id: string;
  userId: string;
  personalImageUrl: string;
  garmentUrl: string;
  resultUrl: string;
  createdAt: Date;
  status: 'processing' | 'completed' | 'failed';
}

export interface TryOnResponse {
  result_url: string;
  processing_time: number;
  metadata: {
    model: string;
    resolution: string;
  };
}

// Garment Types
export interface Garment {
  id: string;
  userId: string;
  url: string;
  thumbnailUrl: string;
  name: string;
  uploadedAt: Date;
  metadata: {
    width: number;
    height: number;
    size: number;
    format: string;
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

// Recommendation Types - matches backend API response
export interface Recommendation {
  id: string;
  name: string;
  image_url: string;
  price: number;
  currency: string;
  category: string;
  ebay_url: string;
  condition?: string;
  shipping?: number;
  relevance_score?: number;
}

// Alias for backward compatibility
export type RecommendationItem = Recommendation;

// Shop Types
export interface ShopItem {
  id: string;
  name: string;
  description: string;
  price: string;
  link: string;
  category: string;
}

// Image Analysis Types
export interface ImageAnalysis {
  type: 'head-only' | 'full-body';
  confidence: number;
  detectedFeatures: {
    hasFace: boolean;
    hasFullBody: boolean;
    bodyParts: string[];
  };
}

// Body Parameters Types
export interface BodyParameters {
  ethnicity: string;
  bodyType: 'slim' | 'athletic' | 'average' | 'curvy' | 'plus-size';
  height: number; // in cm
  weight: number; // in kg
  gender: 'male' | 'female' | 'other';
}

// Body Generation Types
export interface BodyGenerationOption {
  id: string;
  url: string;
}

export interface BodyGenerationResponse {
  options: BodyGenerationOption[];
}

// Supabase Storage Types
export interface StorageObject {
  name: string;
  id: string;
  updated_at: string;
  created_at: string;
  last_accessed_at: string;
  metadata: Record<string, any>;
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
