# API Documentation

Base URL: `http://localhost:8000/api/v1`

All endpoints require authentication unless specified otherwise. Include JWT token in Authorization header:
```
Authorization: Bearer <jwt-token>
```

## Health & Status

### GET /health
Health check endpoint - no authentication required.

**Response**:
```json
{
  "status": "healthy",
  "service": "ml-api",
  "checks": {
    "compute": {
      "gpu_available": true,
      "device": "NVIDIA GeForce RTX 3050"
    },
    "leffa": {
      "available": true,
      "path": "/path/to/Leffa",
      "loaded": true
    },
    "ai_service": {
      "configured": true,
      "masked_key": "AIzaSyB..."
    },
    "environment": {
      "log_level": "INFO",
      "cors_origins": 1
    },
    "memory": {
      "gpu_allocated_mb": 1024.5,
      "gpu_usage_percent": 25.6
    }
  }
}
```

## Virtual Try-On

### POST /process-tryon
Process 2D virtual try-on using Leffa.

**Request** (multipart/form-data):
- `user_image`: File (required) - User/person image
- `garment_image`: File (required) - Garment image
- `garment_type`: String (default: "upper_body") - "upper_body", "lower_body", or "dresses"
- `num_inference_steps`: Integer (default: 30) - Range: 10-50
- `guidance_scale`: Float (default: 2.5) - Range: 1.0-5.0
- `seed`: Integer (default: 42) - Random seed
- `model_type`: String (default: "viton_hd") - "viton_hd" or "dress_code"
- `ref_acceleration`: Boolean (default: false) - Speed up reference UNet
- `repaint`: Boolean (default: false) - Better edge handling

**Response**:
```json
{
  "message": "Virtual try-on processed successfully",
  "status": "success",
  "request_id": "550e8400-e29b-41d4-a716-446655440000",
  "result_url": "https://supabase.co/storage/v1/object/public/results/...",
  "processing_time": 12.34,
  "metadata": {
    "garment_type": "upper_body",
    "num_inference_steps": 30,
    "guidance_scale": 2.5,
    "seed": 42,
    "model_type": "viton_hd"
  }
}
```

### POST /process-tryon-batch
Process batch virtual try-on with multiple garments.

**Request** (multipart/form-data):
- `user_image`: File (required) - User/person image
- `garment_images`: File[] (required) - Multiple garment images
- Same options as `/process-tryon`

**Response**:
```json
{
  "message": "Batch try-on processing complete",
  "batch_id": "batch-uuid",
  "total_count": 5,
  "successful_count": 5,
  "failed_count": 0,
  "results": [
    {
      "status": "success",
      "garment_index": 1,
      "garment_url": "https://...",
      "request_id": "batch-uuid_1",
      "result_url": "https://...",
      "processing_time": 12.34
    }
  ],
  "total_processing_time": 61.7
}
```

## AI Recommendations

### POST /recommend
Get AI-powered outfit recommendations using Gemini + eBay.

**Request** (multipart/form-data):
- `user_photo`: File (required) - User's photo
- `wardrobe_images`: File[] (optional) - Wardrobe items
- `generated_images`: File[] (optional) - Generated body images
- `height_cm`: Float (optional) - Height in cm (140-220)
- `weight_kg`: Float (optional) - Weight in kg (40-200)
- `body_type`: String (optional) - "slim", "athletic", "average", "curvy", "plus_size"
- `ethnicity`: String (optional) - Ethnicity for cultural preferences
- `gender`: String (optional) - Gender for style preferences
- `skin_tone`: String (optional) - Skin tone category
- `style_preference`: String (optional) - Style preferences

**Response**:
```json
{
  "recommendations": [
    {
      "id": "item-1",
      "name": "Blue Denim Jacket",
      "image_url": "https://...",
      "price": "$49.99",
      "ebay_url": "https://ebay.com/...",
      "description": "Classic denim jacket"
    }
  ],
  "count": 20,
  "sources": {
    "wardrobe_count": 5,
    "tryon_history_count": 3,
    "generated_count": 2
  }
}
```

## Body Generation

### POST /generate-bodies
Generate multiple body model variations using SDXL.

**Request** (application/json):
```json
{
  "ethnicity": "caucasian",
  "height_cm": 170,
  "weight_kg": 65,
  "body_type": "athletic",
  "count": 4
}
```

**Response**:
```json
{
  "message": "Body generation complete",
  "bodies": [
    {
      "id": "body-1",
      "url": "https://supabase.co/storage/v1/object/public/bodies/...",
      "metadata": {
        "ethnicity": "caucasian",
        "height_cm": 170,
        "weight_kg": 65,
        "body_type": "athletic"
      }
    }
  ],
  "count": 4,
  "processing_time": 45.6
}
```

### POST /generate-identity-body
Generate full-body images with identity preservation using InstantID.

This endpoint uses InstantID to generate bodies with the user's actual face natively embedded (not stitched), resulting in natural-looking images. Falls back to SDXL if InstantID is unavailable.

**Request** (multipart/form-data):
- `face_image`: File (required) - User's face photo
- `body_type`: String (default: "average") - "athletic", "slim", "muscular", "average", "curvy"
- `height_cm`: Float (default: 170.0) - Height reference (50-300)
- `weight_kg`: Float (default: 65.0) - Weight reference (20-500)
- `gender`: String (default: "female") - "male", "female", "other"
- `ethnicity`: String (optional) - Gemini auto-detects if not provided
- `skin_tone`: String (optional) - Gemini auto-detects if not provided
- `num_images`: Integer (default: 4) - Number of variations (1-4)
- `use_gemini_analysis`: Boolean (default: true) - Enable Gemini facial analysis

**Response**:
```json
{
  "success": true,
  "request_id": "uuid",
  "count": 4,
  "images": [
    {
      "id": "identity_body_0",
      "url": "https://supabase.co/storage/v1/object/public/generated/..."
    },
    {
      "id": "identity_body_1",
      "url": "https://supabase.co/storage/v1/object/public/generated/..."
    }
  ],
  "analysis": {
    "skin_tone": "warm ivory with peachy undertones",
    "skin_tone_category": "fair",
    "ethnicity": "East Asian",
    "age_range": "25-30"
  },
  "params_used": {
    "body_type": "athletic",
    "ethnicity": "East Asian",
    "skin_tone": "warm ivory with peachy undertones"
  },
  "method": "instantid",
  "note": null
}
```

**Method Field**:
- `"instantid"` - Used InstantID for identity-preserving generation (best quality)
- `"sdxl_fallback"` - Used SDXL fallback (InstantID unavailable, may require head stitching)

### POST /analyze-face-features
Analyze facial features using Gemini Vision without generation.

**Request** (multipart/form-data):
- `image`: File (required) - Face photo to analyze

**Response**:
```json
{
  "success": true,
  "analysis": {
    "skin_tone": "rich dark brown with golden undertones",
    "skin_tone_category": "deep",
    "ethnicity": "African",
    "age_range": "30-35",
    "gender_presentation": "masculine",
    "facial_features": {
      "face_shape": "oval",
      "distinctive_features": ["strong jawline", "high cheekbones"]
    },
    "hair": {
      "color": "black",
      "style": "short",
      "length": "short"
    }
  }
}
```

### POST /preview-body-types
Get reference images for different body types.

**Response**:
```json
{
  "body_types": [
    {
      "id": "athletic",
      "label": "Athletic",
      "description": "Toned muscles, fit physique",
      "preview_url": null
    },
    {
      "id": "slim",
      "label": "Slim",
      "description": "Lean, slender build",
      "preview_url": null
    }
  ],
  "note": "Select your preferred body type for personalized generation"
}
```

## Image Analysis

### POST /analyze-body
Analyze if image is head-only or full-body.

**Request** (multipart/form-data):
- `image`: File (required) - Image to analyze

**Response**:
```json
{
  "image_type": "head_only",
  "confidence": 0.95,
  "details": {
    "has_full_body": false,
    "has_face": true,
    "body_coverage": 0.15
  }
}
```

### POST /combine-head-body
Combine head shot with generated body.

**Request** (multipart/form-data):
- `head_image`: File (required) - Head-only photo
- `body_image`: File (required) - Generated body image

**Response**:
```json
{
  "message": "Head-body composition complete",
  "result_url": "https://...",
  "processing_time": 5.2
}
```

## 3D Reconstruction

### POST /reconstruct-3d
Generate 3D mesh from 2D image.

**Request** (multipart/form-data):
- `image`: File (required) - Input image
- `resolution`: Integer (default: 256) - Mesh resolution (128, 256, 512)
- `use_depth`: Boolean (default: true) - Use depth preprocessing

**Response**:
```json
{
  "message": "3D reconstruction complete",
  "mesh_url": "https://supabase.co/storage/v1/object/public/meshes/model.glb",
  "preview_url": "https://supabase.co/storage/v1/object/public/meshes/preview.png",
  "processing_time": 35.8,
  "metadata": {
    "resolution": 256,
    "vertices": 50000,
    "faces": 100000
  }
}
```

## Wardrobe Management

### GET /wardrobe/{user_id}
Get all wardrobe items for a user.

**Response**:
```json
{
  "items": [
    {
      "id": "garment-1",
      "name": "Blue Shirt",
      "image_url": "https://...",
      "category": "upper_body",
      "created_at": "2026-02-15T10:30:00Z"
    }
  ],
  "count": 10
}
```

### POST /garments/upload
Upload garment to wardrobe.

**Request** (multipart/form-data):
- `image`: File (required) - Garment image
- `name`: String (optional) - Garment name
- `category`: String (optional) - "upper_body", "lower_body", "dresses"

**Response**:
```json
{
  "message": "Garment uploaded successfully",
  "garment": {
    "id": "garment-1",
    "name": "Blue Shirt",
    "image_url": "https://...",
    "category": "upper_body"
  }
}
```

### DELETE /garments/{garment_id}
Delete garment from wardrobe.

**Response**:
```json
{
  "message": "Garment deleted successfully",
  "garment_id": "garment-1"
}
```

## Try-On History

### GET /tryon/history/{user_id}
Get try-on history for a user.

**Query Parameters**:
- `limit`: Integer (default: 50) - Max results to return

**Response**:
```json
{
  "results": [
    {
      "id": "result-1",
      "personal_image_url": "https://...",
      "garment_url": "https://...",
      "result_url": "https://...",
      "created_at": "2026-02-15T10:30:00Z",
      "metadata": {
        "garment_type": "upper_body",
        "processing_time": 12.34
      }
    }
  ],
  "count": 25
}
```

## Error Responses

All endpoints return errors in this format:

```json
{
  "error": {
    "type": "user_error",
    "code": "INVALID_INPUT",
    "message": "Invalid input provided",
    "details": {
      "field": "image",
      "reason": "File too large"
    },
    "request_id": "550e8400-e29b-41d4-a716-446655440000"
  }
}
```

### Error Types
- `user_error`: Client-side error (400-499)
- `system_error`: Server-side error (500-599)

### Common Error Codes
- `INVALID_INPUT`: Invalid request parameters
- `FILE_TOO_LARGE`: File exceeds 10MB limit
- `INVALID_MIME_TYPE`: Unsupported file type
- `TRYON_FAILED`: Virtual try-on processing failed
- `RECOMMENDATION_FAILED`: Recommendation generation failed
- `FORBIDDEN`: Access denied (user data isolation)

## Rate Limits

- No rate limits in development
- Production: TBD

## Authentication

All endpoints (except `/health`) require JWT authentication via Supabase.

**Get token**:
```typescript
import { supabase } from '@/lib/supabase';

const { data, error } = await supabase.auth.signInWithPassword({
  email: 'user@example.com',
  password: 'password'
});

const token = data.session?.access_token;
```

**Use token**:
```typescript
fetch('http://localhost:8000/api/v1/process-tryon', {
  method: 'POST',
  headers: {
    'Authorization': `Bearer ${token}`
  },
  body: formData
});
```

---

**Last Updated**: February 15, 2026
