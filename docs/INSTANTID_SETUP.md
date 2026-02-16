# InstantID Setup Guide

Complete guide for setting up InstantID identity-preserving body generation.

## Overview

InstantID enables generating full-body images while preserving the user's facial identity. Unlike simple cut-and-paste methods, InstantID natively embeds the face into the generation process, resulting in natural-looking images.

## System Requirements

- **GPU**: NVIDIA GPU with 6GB+ VRAM (8GB+ recommended)
- **CUDA**: 12.4 (same as main project)
- **Python**: 3.10.x
- **PyTorch**: 2.6.0+cu124 (already installed)
- **Disk Space**: ~2GB for models

## Installation Steps

### Step 1: Install Dependencies

The required packages are already in `requirements.txt`:

```bash
cd backend
pip install insightface==0.7.3 albumentations==1.4.23
```

**Compatibility verified**:
- ✅ insightface 0.7.3 works with PyTorch 2.6.0+cu124
- ✅ albumentations 1.4.23 works with numpy 1.26.4
- ✅ No conflicts with existing packages

### Step 2: Download Models

**Option A: Automatic Download (Recommended)**

Run the PowerShell script:

```powershell
cd backend
.\download_instantid_models.ps1
```

This will download:
- InstantID ControlNet (~800MB)
- InstantID IP-Adapter (~700MB)
- InsightFace antelopev2 face model (~200MB)

**Option B: Manual Download**

If the script fails, download manually:

1. **InstantID Models** (from HuggingFace):
```python
from huggingface_hub import hf_hub_download

model_dir = "./ml_engine/weights/instantid"

# ControlNet
hf_hub_download(
    repo_id="InstantX/InstantID",
    filename="ControlNetModel/config.json",
    local_dir=model_dir
)
hf_hub_download(
    repo_id="InstantX/InstantID",
    filename="ControlNetModel/diffusion_pytorch_model.safetensors",
    local_dir=model_dir
)

# IP-Adapter
hf_hub_download(
    repo_id="InstantX/InstantID",
    filename="ip-adapter.bin",
    local_dir=model_dir
)
```

2. **InsightFace antelopev2**:
   - Download: https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip
   - Extract to: `backend/ml_engine/weights/instantid/models/antelopev2/`

### Step 3: Verify Installation

Check that all files are present:

```
backend/ml_engine/weights/instantid/
├── ControlNetModel/
│   ├── config.json
│   └── diffusion_pytorch_model.safetensors
├── ip-adapter.bin
└── models/
    └── antelopev2/
        ├── 1k3d68.onnx
        ├── 2d106det.onnx
        ├── genderage.onnx
        └── glintr100.onnx
```

Run verification:

```python
python -c "from ml_engine.pipelines.identity_preserving import get_identity_pipeline; pipeline = get_identity_pipeline(); pipeline.load_models(); print('✓ InstantID loaded successfully!')"
```

## Usage

### API Endpoint

**POST /api/v1/generate-identity-body**

Generate full-body images with identity preservation:

```bash
curl -X POST http://localhost:8000/api/v1/generate-identity-body \
  -F "face_image=@user_photo.jpg" \
  -F "body_type=athletic" \
  -F "height_cm=170" \
  -F "weight_kg=65" \
  -F "gender=female" \
  -F "num_images=4" \
  -F "use_gemini_analysis=true"
```

### Python Example

```python
from PIL import Image
from ml_engine.pipelines.identity_preserving import get_identity_pipeline

# Load pipeline
pipeline = get_identity_pipeline()
pipeline.load_models()

# Load user's face photo
face_image = Image.open("user_photo.jpg")

# Generate body images
body_params = {
    "body_type": "athletic",
    "height_cm": 170,
    "weight_kg": 65,
    "gender": "female",
    "ethnicity": "East Asian",
    "skin_tone": "fair"
}

result = pipeline(
    face_image=face_image,
    body_params=body_params,
    num_images=4
)

# Save results
for i, img in enumerate(result["images"]):
    img.save(f"body_{i}.png")
```

## How It Works

### Pipeline Flow

```
User Photo → Face Analysis → InstantID Generation → Full-Body Image
     ↓              ↓                    ↓
 Face Only    Extract features    Generate body WITH
             (InsightFace)        user's face natively
```

### Key Components

1. **InsightFace (antelopev2)**
   - Extracts 512-dimensional face embedding
   - Detects facial keypoints (eyes, nose, mouth)
   - Provides identity information

2. **InstantID ControlNet**
   - Guides generation using facial keypoints
   - Ensures correct face positioning and orientation

3. **IP-Adapter**
   - Injects face embedding into diffusion process
   - Preserves facial identity throughout generation

4. **Gemini Vision (Optional)**
   - Analyzes skin tone, ethnicity, age
   - Provides accurate prompts for better results

### Generation Parameters

```python
pipeline.generate(
    face_image=img,
    body_type="athletic",        # slim, athletic, muscular, average, curvy
    height_cm=170,                # 50-300
    weight_kg=65,                 # 20-500
    gender="female",              # male, female, other
    skin_tone="fair",             # Gemini auto-detects if not provided
    ethnicity="East Asian",       # Gemini auto-detects if not provided
    pose="standing",              # standing, walking, sitting
    clothing="casual minimal",    # Clothing description
    num_images=4,                 # 1-4 variations
    num_inference_steps=20,       # 20 recommended with LCM
    guidance_scale=0.0,           # 0.0 with LCM, higher without
    controlnet_conditioning_scale=0.8,  # ControlNet strength
    ip_adapter_scale=0.8,         # Identity preservation strength
    seed=42                       # For reproducibility
)
```

## Performance

### Memory Usage

- **Idle**: ~0.5GB VRAM
- **Loading models**: ~2GB VRAM
- **Generation (1024x768)**: ~4-6GB VRAM
- **With CPU offload**: ~3-4GB VRAM

### Processing Time

On RTX 3050 4GB:
- Model loading: ~30 seconds (first time only)
- Face analysis: ~2 seconds
- Generation per image: ~15-20 seconds with LCM
- Total for 4 images: ~60-80 seconds

### Optimization Tips

1. **Enable CPU offload** (for 4-6GB VRAM):
```python
pipeline.pipe.enable_model_cpu_offload()
```

2. **Enable VAE tiling** (reduces memory):
```python
pipeline.pipe.enable_vae_tiling()
```

3. **Use LCM-LoRA** (faster inference):
   - Already enabled by default
   - Reduces steps from 50 to 20
   - 2-3x faster generation

4. **Batch processing**:
   - Generate multiple images in one call
   - Reuses loaded models
   - More efficient than separate calls

## Troubleshooting

### "No face detected in the image"

**Cause**: Face not clearly visible or too small

**Solution**:
- Ensure face is at least 128x128 pixels
- Face should be well-lit and in focus
- Avoid extreme angles or occlusions

### "InstantID not available" or "No module named 'ip_adapter'"

**Cause**: Missing IP-Adapter module or models not downloaded

**Solution**:
1. Verify IP-Adapter module exists: `backend/ml_engine/pipelines/ip_adapter/`
2. Check models are in `ml_engine/weights/instantid/`
3. Verify dependencies: `pip list | grep -E "insightface|albumentations|einops"`
4. Check GPU availability: `python -c "import torch; print(torch.cuda.is_available())"`

**Note**: The IP-Adapter module is now included in the codebase at `backend/ml_engine/pipelines/ip_adapter/`

### "CUDA out of memory"

**Cause**: Insufficient VRAM

**Solution**:
1. Enable CPU offload: `pipeline.pipe.enable_model_cpu_offload()`
2. Reduce resolution: `height=768, width=512`
3. Generate fewer images at once
4. Close other GPU applications

### "Fallback to SDXL used"

**Cause**: InstantID failed to load, using SDXL instead

**Solution**:
- Check logs for specific error
- Verify all models are downloaded
- Results will require head stitching (lower quality)

### Face doesn't look like the user

**Cause**: Low IP-Adapter scale or poor face embedding

**Solution**:
1. Increase `ip_adapter_scale` to 0.9-1.0
2. Use higher quality input photo
3. Enable Gemini analysis for better prompts
4. Adjust `controlnet_conditioning_scale`

## Comparison: InstantID vs SDXL Fallback

| Feature | InstantID | SDXL Fallback |
|---------|-----------|---------------|
| Face Quality | ✅ Natural, preserved | ⚠️ Generic, may need stitching |
| Skin Tone | ✅ Accurate | ⚠️ Approximate |
| Processing Time | ~15-20s per image | ~5-10s per image |
| VRAM Required | 4-6GB | 2-3GB |
| Setup Complexity | Medium (models required) | Low (already installed) |
| Result Quality | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |

## API Response

### Success (InstantID)

```json
{
  "success": true,
  "request_id": "uuid",
  "count": 4,
  "images": [
    {"id": "identity_body_0", "url": "https://..."},
    {"id": "identity_body_1", "url": "https://..."}
  ],
  "analysis": {
    "skin_tone": "warm ivory",
    "ethnicity": "East Asian"
  },
  "method": "instantid",
  "note": null
}
```

### Fallback (SDXL)

```json
{
  "success": true,
  "method": "sdxl_fallback",
  "note": "Used SDXL fallback (InstantID not available). Results may require head stitching."
}
```

## References

- **InstantID Paper**: https://arxiv.org/abs/2401.07519
- **InstantID GitHub**: https://github.com/instantX-research/InstantID
- **InsightFace**: https://github.com/deepinsight/insightface
- **Diffusers**: https://huggingface.co/docs/diffusers

## Support

If you encounter issues:

1. Check this guide's troubleshooting section
2. Verify all models are downloaded correctly
3. Check GPU memory with `nvidia-smi`
4. Review logs in `backend/logs/app.log`
5. Test with the verification script above

---

**Last Updated**: February 15, 2026
**Tested On**: Windows 11, RTX 3050 4GB, CUDA 12.4, Python 3.10
