# Identity-Preserving Body Generation

## Overview

This document describes the new approach for generating full-body images from head-only photos, replacing the problematic "cut-and-paste" method with **identity-preserving AI generation**.

## The Problem with the Old Approach

The previous system used:
1. **SDXL** to generate generic full-body images from text prompts
2. **Simple face detection + paste** to crop the user's head and paste onto the generated body

This resulted in:
- Unnatural transitions between head and body
- Mismatched skin tones and lighting
- Obvious "stitching" artifacts
- Placeholder-looking results

## The New Approach: InstantID + Gemini

### How It Works

```
User Photo → Gemini Analysis → InstantID Generation → Natural Full-Body Image
     ↓              ↓                    ↓
 Face Only    Detect features     Generate body WITH
             (skin tone, ethnicity)   user's face natively
```

### Key Technologies

1. **InstantID** (https://github.com/InstantID/InstantID)
   - Zero-shot identity-preserving generation
   - Uses face embedding + ControlNet + IP-Adapter
   - Generates the face AS PART OF the image (not stitched)
   - Result: Natural-looking full-body images

2. **InsightFace (antelopev2)**
   - Extracts facial embeddings and keypoints
   - Provides identity information for InstantID

3. **Gemini Vision API**
   - Analyzes user's facial features
   - Detects skin tone, ethnicity, age range accurately
   - Provides detailed prompts for better generation

### Flow

1. **User uploads head-only photo**
2. **Gemini analyzes facial features:**
   ```json
   {
     "skin_tone": "warm ivory with peachy undertones",
     "skin_tone_category": "fair",
     "ethnicity": "East Asian",
     "age_range": "25-30",
     "hair": { "color": "dark brown", "length": "medium" }
   }
   ```
3. **User selects body type preference** (athletic, slim, muscular, average)
4. **InstantID generates full-body images:**
   - Face embedding extracted from user photo
   - Facial keypoints guide generation
   - Body type and Gemini analysis inform the prompt
5. **User selects their favorite** - directly used as profile photo

### Fallback to SDXL

If InstantID is unavailable (no GPU, model not downloaded), the system falls back to:
1. Improved SDXL generation using Gemini-analyzed prompts
2. Head stitching (like before, but with better prompts)

The API response indicates which method was used:
```json
{
  "method": "instantid",  // or "sdxl_fallback"
  "note": null  // or warning message
}
```

## Installation

### Backend Dependencies

Add to requirements.txt:
```
insightface==0.7.3
albumentations==1.4.23
```

### Download InstantID Models

The models download automatically on first use, or manually:
```python
from huggingface_hub import hf_hub_download

hf_hub_download(repo_id="InstantX/InstantID", filename="ControlNetModel/config.json", local_dir="./checkpoints")
hf_hub_download(repo_id="InstantX/InstantID", filename="ControlNetModel/diffusion_pytorch_model.safetensors", local_dir="./checkpoints")
hf_hub_download(repo_id="InstantX/InstantID", filename="ip-adapter.bin", local_dir="./checkpoints")
```

### Download Face Model (Required)

Download antelopev2 face model manually:
https://github.com/deepinsight/insightface/issues/1896#issuecomment-1023867304

Place in: `backend/ml_engine/weights/instantid/models/antelopev2/`

## API Endpoints

### POST /api/v1/generate-identity-body

Generate full-body images preserving facial identity.

**Request (multipart/form-data):**
| Field | Type | Description |
|-------|------|-------------|
| face_image | File | User's face photo |
| body_type | string | "athletic", "slim", "muscular", "average", "curvy" |
| height_cm | float | Height reference |
| weight_kg | float | Weight reference |
| gender | string | "male", "female", "other" |
| ethnicity | string? | Optional override (Gemini auto-detects) |
| skin_tone | string? | Optional override (Gemini auto-detects) |
| num_images | int | 1-4 variations |
| use_gemini_analysis | bool | Enable Gemini facial analysis |

**Response:**
```json
{
  "success": true,
  "request_id": "uuid",
  "count": 4,
  "images": [
    { "id": "identity_body_0", "url": "https://..." },
    { "id": "identity_body_1", "url": "https://..." }
  ],
  "analysis": {
    "skin_tone": "medium skin tone",
    "ethnicity": "South Asian"
  },
  "params_used": {
    "body_type": "athletic",
    "ethnicity": "South Asian",
    "skin_tone": "medium skin tone"
  },
  "method": "instantid",
  "note": null
}
```

### POST /api/v1/analyze-face-features

Analyze facial features without generation.

**Request:** `image` (File)

**Response:**
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

## Frontend Integration

The onboarding page automatically uses the new endpoint:

```typescript
const result = await endpoints.generateIdentityBody({
  faceImage: uploadedImage,
  body_type: bodyParameters.bodyType,
  height_cm: bodyParameters.height,
  weight_kg: bodyParameters.weight,
  gender: bodyParameters.gender,
  num_images: 4,
  use_gemini_analysis: true,
});

// Check method used
if (result.method === "sdxl_fallback") {
  // Need to combine head+body manually
  await endpoints.combineHeadBody(userPhoto, selectedBody);
} else {
  // InstantID: face is already in the image, use directly
  await saveProfile(selectedBodyUrl);
}
```

## Benefits

1. **Natural Results** - No visible stitching or transitions
2. **Accurate Skin Tones** - Gemini detects and InstantID preserves
3. **Identity Preservation** - User looks like themselves
4. **Better Try-On** - More realistic base for virtual try-on
5. **Graceful Fallback** - Works even without GPU (degraded quality)

## File Structure

```
backend/
├── ml_engine/pipelines/
│   ├── identity_preserving.py  # InstantID pipeline
│   └── instantid_pipeline.py   # (Optional) Custom pipeline
├── app/api/
│   └── identity_body.py        # API endpoints
├── app/services/
│   └── face_analysis.py        # Gemini face analysis
└── requirements.txt            # Updated dependencies

frontend/src/
├── lib/api.ts                  # New API methods
└── app/onboard/page.tsx        # Updated onboarding flow
```

## Troubleshooting

### "InstantID not available"
- Ensure GPU is available
- Check model downloads completed
- Verify InsightFace/antelopev2 model is present

### "No face detected"
- Ensure face is clearly visible in photo
- Photo shouldn't be too dark or blurry
- Face should be at least 128x128 pixels

### "SDXL fallback used"
- InstantID couldn't load, falling back to previous method
- Results will require head stitching
- Check logs for specific error
