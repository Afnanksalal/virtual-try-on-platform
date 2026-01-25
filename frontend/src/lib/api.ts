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
const DEFAULT_TIMEOUT = 30000;

class APIError extends Error {
  constructor(public status: number, message: string) {
    super(message);
    this.name = "APIError";
  }
}

// Core API client
const api = {
  async request<T>(endpoint: string, options: RequestInit = {}, config: APIConfig = {}): Promise<T> {
    const controller = new AbortController();
    const timeout = config.timeout || DEFAULT_TIMEOUT;
    const timeoutId = setTimeout(() => controller.abort(), timeout);

    try {
      const response = await fetch(`${API_BASE_URL}${endpoint}`, {
        ...options,
        signal: controller.signal,
        headers: {
          ...config.headers,
          ...options.headers,
        },
      });

      clearTimeout(timeoutId);

      if (!response.ok) {
        const errorText = await response.text();
        throw new APIError(response.status, errorText || "Request failed");
      }

      return await response.json();
    } catch (error) {
      clearTimeout(timeoutId);
      if (error instanceof APIError) throw error;
      if (error instanceof Error) {
        if (error.name === "AbortError") {
          throw new APIError(408, "Request timeout");
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
  ): Promise<RecommendationItem[]> => {
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

  processTryOn: (userImage: File, garmentImage: File): Promise<TryOnResponse> => {
    const formData = new FormData();
    formData.append("user_image", userImage);
    formData.append("garment_image", garmentImage);
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
};

export { APIError };
