# CatVTON Setup Instructions

## What I've Done

1. **Rewrote `backend/ml_engine/pipelines/idm_vton.py`** to use CatVTON's actual implementation
   - Uses `CatVTONPipeline` from the official repo
   - Uses `AutoMasker` for automatic mask generation (DensePose + SCHP)
   - Downloads checkpoints automatically from HuggingFace
   - Follows the exact pattern from CatVTON's app.py

2. **Updated `backend/requirements.txt`** with CatVTON dependencies
   - Torch 2.4.0 (CatVTON tested version)
   - Diffusers from git (latest version required)
   - All CatVTON dependencies: peft, fvcore, pycocotools, etc.
   - Removed incompatible packages (rembg, controlnet-aux, etc.)

3. **Updated `backend/app/services/tryon_service.py`** to match CatVTON API
   - Supports cloth types: "upper", "lower", "overall"
   - Default CFG: 2.5 (CatVTON recommended)
   - Default resolution: 768x1024
   - Proper error handling and validation

## What YOU Need to Do

### Step 1: Clone CatVTON Repository

```bash
cd backend
git clone https://github.com/Zheng-Chong/CatVTON
```

This will create: `backend/CatVTON/`

The pipeline will import from this directory:
- `model.pipeline.CatVTONPipeline`
- `model.cloth_masker.AutoMasker`
- `utils.resize_and_crop`, `utils.resize_and_padding`

### Step 2: Install Dependencies

```bash
cd backend
pip install -r requirements.txt
```

This will:
- Install PyTorch 2.4.0 + torchvision 0.19.0
- Install diffusers from git (latest)
- Install all CatVTON dependencies
- May take 10-15 minutes

### Step 3: Verify CatVTON Path (Already Done)

The pipeline automatically adds CatVTON to Python path at runtime.
Check `backend/ml_engine/pipelines/idm_vton.py` lines 11-15:

```python
# Add CatVTON to Python path
CATVTON_PATH = os.path.join(os.path.dirname(__file__), '..', '..', 'CatVTON')
if os.path.exists(CATVTON_PATH) and CATVTON_PATH not in sys.path:
    sys.path.insert(0, CATVTON_PATH)
```

This means you just need to clone the repo to `backend/CatVTON/` and it will work automatically.

### Step 4: Verify Installation

Check if CatVTON modules can be imported:
```bash
cd backend
python -c "import sys; sys.path.insert(0, 'CatVTON'); from model.pipeline import CatVTONPipeline; print('✓ CatVTON modules found')"
```

### Step 5: Test the Pipeline

```bash
cd backend
python main.py
```

The first run will:
1. Download CatVTON checkpoint from HuggingFace (~900MB)
2. Download DensePose model
3. Download SCHP model
4. Load everything into GPU memory (~8GB VRAM)

## How CatVTON Works

1. **Input**: Person image + Garment image + Cloth type ("upper"/"lower"/"overall")
2. **Auto-masking**: DensePose + SCHP generate mask automatically
3. **Inference**: CatVTON pipeline runs diffusion (50 steps, CFG 2.5)
4. **Output**: Person wearing the garment

## API Usage

```python
# From frontend
POST /api/v1/process-tryon
{
  "person_image": <file>,
  "garment_image": <file>,
  "options": {
    "garment_description": "upper",  // "upper", "lower", or "overall"
    "num_inference_steps": 50,
    "guidance_scale": 2.5,
    "seed": 42,
    "width": 768,
    "height": 1024
  }
}
```

## Troubleshooting

### Import Error: "No module named 'model'"
- CatVTON repo not cloned or not in Python path
- Solution: Follow Step 1 and Step 3

### CUDA Out of Memory
- CatVTON needs ~8GB VRAM for 1024x768
- Solution: Reduce resolution to 512x768 or use CPU (slow)

### Checkpoint Download Fails
- HuggingFace connection issue
- Solution: Check internet, try again, or manually download from https://huggingface.co/zhengchong/CatVTON

### Mask Generation Fails
- DensePose/SCHP models not downloaded
- Solution: Wait for first run to complete, models download automatically

## Key Differences from Previous Attempts

| Feature | IDM-VTON (failed) | StableVITON (failed) | CatVTON (current) |
|---------|-------------------|----------------------|-------------------|
| Repo clone | Required | Required | Required |
| Preprocessing | Complex | Complex | Automatic |
| Mask generation | Manual | Manual | Automatic (DensePose+SCHP) |
| Text prompts | Required | Not needed | Not needed |
| VRAM usage | >12GB | >10GB | <8GB |
| Resolution | 512x768 | 512x768 | 768x1024 |
| HuggingFace | No | Partial | Full support |

## References

- GitHub: https://github.com/Zheng-Chong/CatVTON
- Paper: https://arxiv.org/abs/2407.15886
- HuggingFace: https://huggingface.co/zhengchong/CatVTON
- Demo: https://huggingface.co/spaces/zhengchong/CatVTON
