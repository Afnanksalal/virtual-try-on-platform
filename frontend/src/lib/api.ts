import { supabase } from './supabase';
import { withRetry, handleError } from './errorHandling';
import type { 
  Garment, 
  TryOnResult, 
  Recommendation, 
  ImageAnalysis, 
  BodyParameters,
  StorageObject,
  GarmentType 
} from './types';

// Type definitions
type HealthCheckResponse = {
  status: string;
  service?: string;
  version?: string;
};

type RecommendationItem = {
  id: string;
  name: string;
  image_url: string;
  price: number;
  category: string;
  ebay_url: string;
};

type TryOnResponse = {
  message: string;
  status: string;
  result_url: string;
  processing_time: number;
};

type BodyGenerationResponse = {
  message: string;
  request_id: string;
  count: number;
  images: Array<{ id: string; url: string }>;
};

type ImageAnalysisResponse = {
  type: "head_only" | "full_body";
  confidence: number;
};

type ImageCompositionResponse = {
  message: string;
  request_id: string;
  image_url: string;
};

type APIConfig = {
  timeout?: number;
  headers?: Record<string, string>;
};

// API Configuration
const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
// No timeouts - wait indefinitely for ML operations
const DEFAULT_TIMEOUT = 0; // Infinite
const LONG_TIMEOUT = 0; // Infinite

class APIError extends Error {
  constructor(public status: number, message: string) {
    super(message);
    this.name = "APIError";
  }
}

// Core API client
const api = {
  async getAuthHeaders(): Promise<Record<string, string>> {
    try {
      const { data: { session } } = await supabase.auth.getSession();
      if (session?.access_token) {
        return {
          'Authorization': `Bearer ${session.access_token}`,
        };
      }
      return {};
    } catch (error) {
      console.error('Failed to get auth token:', error);
      return {};
    }
  },

  async request<T>(endpoint: string, options: RequestInit = {}, config: APIConfig = {}): Promise<T> {
    const timeout = config.timeout || DEFAULT_TIMEOUT;
    const controller = new AbortController();
    let timeoutId: NodeJS.Timeout | undefined;
    
    // Only set timeout if it's not 0 (infinite)
    if (timeout > 0) {
      timeoutId = setTimeout(() => controller.abort(), timeout);
    }

    try {
      // Get auth headers
      const authHeaders = await this.getAuthHeaders();
      
      const response = await fetch(`${API_BASE_URL}${endpoint}`, {
        ...options,
        headers: {
          ...authHeaders,
          ...config.headers,
          ...options.headers,
        },
        signal: controller.signal,
      });

      if (timeoutId) clearTimeout(timeoutId);

      if (!response.ok) {
        const errorText = await response.text();
        throw new APIError(response.status, errorText || "Request failed");
      }

      return await response.json();
    } catch (error) {
      if (timeoutId) clearTimeout(timeoutId);
      if (error instanceof APIError) throw error;
      if (error instanceof Error) {
        if (error.name === 'AbortError') {
          throw new APIError(408, 'Request timeout');
        }
        throw new APIError(500, error.message);
      }
      throw new APIError(500, "Unknown error");
    }
  },

  get<T>(endpoint: string, config?: APIConfig): Promise<T> {
    return this.request<T>(endpoint, { method: "GET" }, config);
  },

  post<T>(endpoint: string, body?: unknown, config?: APIConfig): Promise<T> {
    const options: RequestInit = {
      method: "POST",
    };

    if (body instanceof FormData) {
      options.body = body;
    } else if (body) {
      options.headers = { "Content-Type": "application/json" };
      options.body = JSON.stringify(body);
    }

    return this.request<T>(endpoint, options, config);
  },

  delete<T>(endpoint: string, config?: APIConfig): Promise<T> {
    return this.request<T>(endpoint, { method: "DELETE" }, config);
  },
};

// Exported ML-only endpoints
export const endpoints = {
  // Health check
  health: (): Promise<HealthCheckResponse> => 
    api.get("/api/v1/health"),

  // ML Operations Only
  getRecommendations: (
    userPhoto: File,
    wardrobeImages?: File[],
    generatedImages?: File[]
  ): Promise<Recommendation[]> => {
    const formData = new FormData();
    formData.append("user_photo", userPhoto);
    
    if (wardrobeImages) {
      wardrobeImages.forEach(img => formData.append("wardrobe_images", img));
    }
    
    if (generatedImages) {
      generatedImages.forEach(img => formData.append("generated_images", img));
    }
    
    return api.post("/api/v1/recommend", formData);
  },

  processTryOn: (
    userImage: File, 
    garmentImage: File, 
    options?: {
      garment_type?: GarmentType;
      num_inference_steps?: number;
      guidance_scale?: number;
      seed?: number;
    }
  ): Promise<TryOnResponse> => {
    const formData = new FormData();
    formData.append("user_image", userImage);
    formData.append("garment_image", garmentImage);
    formData.append("garment_type", options?.garment_type || "upper_body");
    formData.append("num_inference_steps", String(options?.num_inference_steps || 30));
    formData.append("guidance_scale", String(options?.guidance_scale || 2.5));
    formData.append("seed", String(options?.seed || 42));
    return api.post("/api/v1/process-tryon", formData);
  },

  // Alias for TryOnWidget compatibility
  tryOn: (formData: FormData): Promise<TryOnResponse> => {
    return api.post("/api/v1/process-tryon", formData);
  },

  generateBody: (data: {
    ethnicity: string;
    height_cm: number;
    weight_kg: number;
    body_type: string;
    count?: number;
  }): Promise<BodyGenerationResponse> => {
    const formData = new FormData();
    formData.append("ethnicity", data.ethnicity);
    formData.append("height_cm", data.height_cm.toString());
    formData.append("weight_kg", data.weight_kg.toString());
    formData.append("body_type", data.body_type);
    formData.append("count", (data.count || 4).toString());
    return api.post("/api/v1/generate-body", formData);
  },

  // Smart onboarding endpoints
  analyzeImage: (image: File): Promise<ImageAnalysisResponse> => {
    const formData = new FormData();
    formData.append("image", image);
    return api.post("/api/v1/analyze-image", formData);
  },

  generateBodies: (data: {
    ethnicity: string;
    skin_tone: string;
    body_type: string;
    height_cm: number;
    weight_kg: number;
  }): Promise<BodyGenerationResponse> => {
    const formData = new FormData();
    formData.append("ethnicity", data.ethnicity);
    formData.append("skin_tone", data.skin_tone);
    formData.append("body_type", data.body_type);
    formData.append("height_cm", data.height_cm.toString());
    formData.append("weight_kg", data.weight_kg.toString());
    return api.post("/api/v1/generate-bodies", formData);
  },

  combineHeadBody: (headImage: File, bodyImage: File): Promise<ImageCompositionResponse> => {
    const formData = new FormData();
    formData.append("head_image", headImage);
    formData.append("body_image", bodyImage);
    return api.post("/api/v1/combine-head-body", formData);
  },

  // Garment Management
  uploadGarment: async (file: File, userId: string): Promise<Garment> => {
    return withRetry(
      async () => {
        const formData = new FormData();
        formData.append('file', file);
        
        const response = await api.post<{
          message: string;
          garment: {
            id: string;
            url: string;
            name: string;
            path: string;
            uploaded_at: string;
            size_mb: number;
            content_type: string;
          };
        }>(`/api/v1/garments/upload?user_id=${encodeURIComponent(userId)}`, formData, {
          timeout: LONG_TIMEOUT,
        });

        return {
          id: response.garment.id,
          userId,
          url: response.garment.url,
          thumbnailUrl: response.garment.url,
          name: response.garment.name,
          uploadedAt: new Date(response.garment.uploaded_at),
          metadata: {
            width: 0,
            height: 0,
            size: Math.round(response.garment.size_mb * 1024 * 1024),
            format: response.garment.content_type,
          },
        };
      },
      'upload garment',
      { maxAttempts: 2 } // Only retry once for uploads
    );
  },

  listGarments: async (userId: string): Promise<Garment[]> => {
    return withRetry(
      async () => {
        const response = await api.get<{
          garments: Array<{
            id: string;
            url: string;
            name: string;
            path: string;
            uploaded_at: string;
            size_bytes: number;
          }>;
          count: number;
          user_id: string;
        }>(`/api/v1/garments/list?user_id=${encodeURIComponent(userId)}`);

        return response.garments.map((garment) => ({
          id: garment.id,
          userId,
          url: garment.url,
          thumbnailUrl: garment.url,
          name: garment.name,
          uploadedAt: new Date(garment.uploaded_at),
          metadata: {
            width: 0,
            height: 0,
            size: garment.size_bytes,
            format: '',
          },
        }));
      },
      'list garments'
    );
  },

  deleteGarment: async (garmentId: string, userId: string): Promise<void> => {
    return withRetry(
      async () => {
        await api.delete<{
          message: string;
          garment_id: string;
          deleted: boolean;
        }>(`/api/v1/garments/${encodeURIComponent(garmentId)}?user_id=${encodeURIComponent(userId)}`);
      },
      'delete garment',
      { maxAttempts: 2 }
    );
  },

  // Try-On Operations
  generateTryOn: async (
    personalImageUrl: string, 
    garmentUrl: string,
    options?: {
      garment_type?: GarmentType;
      num_inference_steps?: number;
      guidance_scale?: number;
      seed?: number;
    }
  ): Promise<TryOnResult> => {
    return withRetry(
      async () => {
        // Fetch images from URLs
        const [personalResponse, garmentResponse] = await Promise.all([
          fetch(personalImageUrl),
          fetch(garmentUrl),
        ]);

        const [personalBlob, garmentBlob] = await Promise.all([
          personalResponse.blob(),
          garmentResponse.blob(),
        ]);

        const personalFile = new File([personalBlob], 'personal.jpg', { type: personalBlob.type });
        const garmentFile = new File([garmentBlob], 'garment.jpg', { type: garmentBlob.type });

        const response = await endpoints.processTryOn(personalFile, garmentFile, options);

        return {
          id: `tryon-${Date.now()}`,
          userId: '',
          personalImageUrl,
          garmentUrl,
          resultUrl: response.result_url,
          createdAt: new Date(),
          status: 'completed',
        };
      },
      'generate try-on',
      { maxAttempts: 2 } // ML operations are expensive, limit retries
    );
  },

  // Personal Image Management
  getPersonalImage: async (userId: string): Promise<{
    url: string;
    type: 'head-only' | 'full-body';
    uploadedAt: Date;
  } | null> => {
    try {
      const { data, error } = await supabase
        .from('profiles')
        .select('photo_url, is_full_body, created_at, updated_at')
        .eq('id', userId)
        .single();

      if (error || !data || !data.photo_url) {
        return null;
      }

      return {
        url: data.photo_url,
        type: data.is_full_body ? 'full-body' : 'head-only',
        uploadedAt: new Date(data.updated_at || data.created_at),
      };
    } catch (error) {
      console.error('Failed to fetch personal image:', error);
      return null;
    }
  },

  updatePersonalImage: async (userId: string, file: File): Promise<{
    url: string;
    type: 'head-only' | 'full-body';
    uploadedAt: Date;
  }> => {
    try {
      // First analyze the image to determine type
      const analysis = await endpoints.analyzeImage(file);
      
      // Upload to Supabase storage
      const fileName = `${userId}/${Date.now()}_${file.name}`;
      const { error: uploadError } = await supabase.storage
        .from('user-uploads')
        .upload(fileName, file, { cacheControl: '3600', upsert: false });

      if (uploadError) {
        throw new APIError(500, `Upload failed: ${uploadError.message}`);
      }

      // Get public URL
      const { data: { publicUrl } } = supabase.storage
        .from('user-uploads')
        .getPublicUrl(fileName);

      // Update profile with image type from analysis
      const isFullBody = analysis.type === 'full_body';
      const { error: updateError } = await supabase
        .from('profiles')
        .update({
          photo_url: publicUrl,
          is_full_body: isFullBody,
          updated_at: new Date().toISOString(),
        })
        .eq('id', userId);

      if (updateError) {
        throw new APIError(500, `Profile update failed: ${updateError.message}`);
      }

      return {
        url: publicUrl,
        type: isFullBody ? 'full-body' : 'head-only',
        uploadedAt: new Date(),
      };
    } catch (error) {
      if (error instanceof APIError) throw error;
      throw new APIError(500, error instanceof Error ? error.message : 'Failed to update personal image');
    }
  },

  // ========== WARDROBE ENDPOINTS ==========

  getWardrobe: async (userId: string): Promise<Garment[]> => {
    return withRetry(
      async () => {
        const response = await api.get<{
          items: Array<{
            id: string;
            url: string;
            name: string;
            path?: string;
            uploaded_at: string;
            size_bytes?: number;
            user_id?: string;
          }>;
          count: number;
        }>(`/api/v1/wardrobe/${encodeURIComponent(userId)}`);

        return response.items.map((item) => ({
          id: item.id,
          userId: item.user_id || userId,
          url: item.url,
          thumbnailUrl: item.url,
          name: item.name,
          uploadedAt: new Date(item.uploaded_at),
          metadata: {
            width: 0,
            height: 0,
            size: item.size_bytes || 0,
            format: '',
          },
        }));
      },
      'get wardrobe'
    );
  },

  // ========== TRY-ON HISTORY ENDPOINTS ==========

  getTryOnHistory: async (userId: string, limit: number = 50): Promise<TryOnResult[]> => {
    return withRetry(
      async () => {
        const response = await api.get<{
          results: Array<{
            id: string;
            user_id: string;
            personal_image_url: string;
            garment_url: string;
            result_url: string;
            status: string;
            created_at: string;
            metadata?: any;
          }>;
          count: number;
        }>(`/api/v1/tryon/history/${encodeURIComponent(userId)}?limit=${limit}`);

        return response.results.map((result) => ({
          id: result.id,
          userId: result.user_id,
          personalImageUrl: result.personal_image_url,
          garmentUrl: result.garment_url,
          resultUrl: result.result_url,
          createdAt: new Date(result.created_at),
          status: (result.status as 'processing' | 'completed' | 'failed') || 'completed',
        }));
      },
      'get try-on history'
    );
  },
};

// Supabase Storage Helpers
export const supabaseStorage = {
  uploadToSupabase: async (file: File, bucket: string, path: string): Promise<string> => {
    const { data, error } = await supabase.storage
      .from(bucket)
      .upload(path, file);

    if (error) {
      throw new APIError(500, `Upload failed: ${error.message}`);
    }

    const { data: { publicUrl } } = supabase.storage
      .from(bucket)
      .getPublicUrl(path);

    return publicUrl;
  },

  deleteFromSupabase: async (bucket: string, path: string): Promise<void> => {
    const { error } = await supabase.storage
      .from(bucket)
      .remove([path]);

    if (error) {
      throw new APIError(500, `Delete failed: ${error.message}`);
    }
  },

  listFromSupabase: async (bucket: string, prefix: string): Promise<StorageObject[]> => {
    const { data, error } = await supabase.storage
      .from(bucket)
      .list(prefix);

    if (error) {
      throw new APIError(500, `List failed: ${error.message}`);
    }

    return data || [];
  },
};

export { APIError };
