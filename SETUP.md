# Virtual Try-On Platform - Complete Setup Guide

**Last Updated**: February 16, 2026  
**Tested On**: Windows 11, RTX 3050 4GB, CUDA 12.4, Python 3.10.x, Node.js 18+

---

## Table of Contents

1. [System Requirements](#system-requirements)
2. [Prerequisites Installation](#prerequisites-installation)
3. [Python Environment Setup](#python-environment-setup)
4. [Frontend Setup](#frontend-setup)
5. [Backend Setup](#backend-setup)
6. [Model Downloads](#model-downloads)
7. [Database Setup](#database-setup)
8. [Environment Configuration](#environment-configuration)
9. [Running the Application](#running-the-application)
10. [Troubleshooting](#troubleshooting)
11. [Advanced Configuration](#advanced-configuration)

---

## System Requirements

### Minimum Requirements
- **OS**: Windows 10/11 (64-bit), Linux, or macOS
- **CPU**: 4+ cores recommended
- **RAM**: 16GB minimum, 32GB recommended
- **Storage**: 50GB+ free space (for models and dependencies)
- **GPU**: NVIDIA GPU with 4GB+ VRAM (tested on RTX 3050 4GB)
- **Internet**: Stable connection for model downloads (20-30GB total)

### Software Requirements
- **Python**: 3.10.x (3.10.11 recommended)
- **Node.js**: 18+ (18.17.0 or higher)
- **CUDA**: 12.4 (for GPU support)
- **Git**: Latest version
- **Visual Studio 2022 Build Tools** (Windows only, for compiling C++ extensions)

### GPU Requirements
- **Minimum**: 4GB VRAM (RTX 3050, GTX 1650)
- **Recommended**: 8GB+ VRAM (RTX 3060, RTX 4060)
- **Optimal**: 12GB+ VRAM (RTX 3080, RTX 4070)

**Note**: CPU-only mode is supported but 10-20x slower.

---

## Prerequisites Installation

### 1. Visual Studio 2022 Build Tools (Windows Only)

**Required for**: Compiling torchmcubes and other C++ extensions

1. Download Visual Studio 2022 Build Tools:
   ```
   https://visualstudio.microsoft.com/downloads/
   ```

2. Run installer and select:
   - ✅ Desktop development with C++
   - ✅ MSVC v143 - VS 2022 C++ x64/x86 build tools
   - ✅ Windows 10/11 SDK (latest version)
   - ✅ C++ CMake tools for Windows

3. Note installation paths (needed later):
   ```
   Compiler: C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Tools\MSVC\<version>\bin\Hostx64\x64\cl.exe
   SDK: C:\Program Files (x86)\Windows Kits\10\
   ```

### 2. CUDA 12.4 Toolkit

**Required for**: GPU acceleration with PyTorch 2.6.0

1. Download CUDA 12.4:
   ```
   https://developer.nvidia.com/cuda-12-4-0-download-archive
   ```

2. Install to default location:
   ```
   C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4
   ```

3. Verify installation:
   ```bash
   nvcc --version
   ```
   Expected output: `Cuda compilation tools, release 12.4`

4. Add to PATH (if not automatic):
   ```
   C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4\bin
   ```

### 3. Node.js 18+

1. Download from: https://nodejs.org/
2. Install LTS version (18.17.0 or higher)
3. Verify:
   ```bash
   node --version  # Should show v18.x.x or higher
   npm --version   # Should show 9.x.x or higher
   ```

### 4. Git

1. Download from: https://git-scm.com/
2. Install with default settings
3. Verify:
   ```bash
   git --version
   ```

---

## Python Environment Setup

### Step 1: Clone Repository

```bash
cd C:\Users\<YourUsername>\Downloads  # Or your preferred location
git clone <your-repo-url> virtual-try-on-platform
cd virtual-try-on-platform
```

### Step 2: Create Virtual Environment

```bash
cd backend
python -m venv venv
```

### Step 3: Activate Virtual Environment

**Windows (PowerShell)**:
```powershell
.\venv\Scripts\Activate.ps1
```

**Windows (CMD)**:
```cmd
venv\Scripts\activate.bat
```

**Linux/macOS**:
```bash
source venv/bin/activate
```

### Step 4: Upgrade pip

```bash
python -m pip install --upgrade pip
```

### Step 5: Install PyTorch with CUDA 12.4

**CRITICAL**: Install PyTorch FIRST before other packages.

```bash
pip install torch==2.6.0+cu124 torchvision==0.21.0+cu124 torchaudio==2.6.0+cu124 --index-url https://download.pytorch.org/whl/cu124
```

**Verify installation**:
```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.version.cuda}'); print(f'CUDA Available: {torch.cuda.is_available()}')"
```

Expected output:
```
PyTorch: 2.6.0+cu124
CUDA: 12.4
CUDA Available: True
```

### Step 6: Install Core Dependencies

```bash
pip install -r requirements.txt
```

This installs:
- FastAPI, Uvicorn (web framework)
- Diffusers, Transformers, Accelerate (ML models)
- OpenCV, Pillow, NumPy (image processing)
- Supabase, Google GenAI (cloud services)
- And 50+ other dependencies

**Installation time**: 10-15 minutes

### Step 7: Install Git-Based Packages

```bash
# SAM 2.1 (Segment Anything Model 2)
pip install git+https://github.com/facebookresearch/segment-anything-2.git@2b90b9f5ceec907a1c18123530e92e794ad901a4

# torchmcubes (for 3D mesh extraction)
pip install git+https://github.com/tatsy/torchmcubes.git@3381600ddc3d2e4d74222f8495866be5fafbace4
```

**Note**: torchmcubes compilation may take 5-10 minutes. See [Troubleshooting](#troubleshooting) if it fails.

### Step 8: Install InstantID Dependencies (Optional)

For identity-preserving body generation:

```bash
pip install insightface==0.7.3 albumentations==1.4.23
```

---

## Frontend Setup

### Step 1: Navigate to Frontend Directory

```bash
cd ../frontend  # From backend directory
```

### Step 2: Install Dependencies

```bash
npm install
```

This installs:
- Next.js 16.1.3, React 19.2.3
- Three.js, React Three Fiber (3D visualization)
- Tailwind CSS 4, Framer Motion (styling & animations)
- Supabase client (auth & storage)

**Installation time**: 3-5 minutes

### Step 3: Configure Environment

```bash
cp .env.local.example .env.local
```

Edit `.env.local` with your values (see [Environment Configuration](#environment-configuration))

---

## Backend Setup

### Step 1: Clone Leffa Repository

**IMPORTANT**: Leffa must be cloned at the PROJECT ROOT (same level as backend/ and frontend/)

```bash
cd ..  # Go to project root
git clone https://github.com/franciszzj/Leffa
```

Directory structure should be:
```
virtual-try-on-platform/
├── backend/
├── frontend/
├── Leffa/          ← Must be here
└── README.md
```

### Step 2: Configure Backend Environment

```bash
cd backend
cp .env.example .env
```

Edit `.env` with your API keys (see [Environment Configuration](#environment-configuration))

### Step 3: Create Required Directories

```bash
mkdir -p logs
mkdir -p data/uploads
mkdir -p data/results
mkdir -p 3d/models
mkdir -p 3d/outputs
mkdir -p ml_engine/weights/instantid
```

---

## Model Downloads

Models will auto-download on first run, but you can pre-download them:

### 1. Leffa (Virtual Try-On) - ~8GB

**Auto-downloads on first backend startup**

Repository: `franciszzj/Leffa` on HuggingFace

**Sub-checkpoints included**:

1. **Stable Diffusion Inpainting** (~4GB)
   - Base model for virtual try-on
   - Path: `ckpts/stable-diffusion-inpainting/`

2. **Virtual Try-On Models** (~1.5GB)
   - VITON-HD model: `ckpts/virtual_tryon.pth`
   - DressCode model: `ckpts/virtual_tryon_dc.pth` (optional)

3. **DensePose** (~200MB)
   - Config: `ckpts/densepose/densepose_rcnn_R_50_FPN_s1x.yaml`
   - Weights: `ckpts/densepose/model_final_162be9.pkl`
   - Used for body pose estimation

4. **Human Parsing** (~200MB)
   - ATR model: `ckpts/humanparsing/parsing_atr.onnx`
   - LIP model: `ckpts/humanparsing/parsing_lip.onnx`
   - Used for semantic segmentation

5. **OpenPose** (~200MB)
   - Body model: `ckpts/openpose/body_pose_model.pth`
   - Used for keypoint detection

6. **SCHP (Self Correction Human Parsing)** (~200MB)
   - Path: `ckpts/schp/`
   - Used for garment-agnostic mask generation

7. **Pose Transfer Model** (~1.5GB, optional)
   - Model: `ckpts/pose_transfer.pth`
   - SDXL Inpainting: `ckpts/stable-diffusion-xl-1.0-inpainting-0.1/`
   - Only needed for pose transfer feature

**Total size**: ~8GB (without pose transfer) or ~10GB (with pose transfer)

**Manual download** (optional):
```python
from huggingface_hub import snapshot_download
snapshot_download(repo_id="franciszzj/Leffa", local_dir="./Leffa/ckpts")
```

**Note**: First download takes 10-30 minutes depending on connection speed.

### 2. SAM 2.1 (Segmentation) - ~900MB

**Auto-downloads on first 3D reconstruction**

Model: `facebook/sam2.1-hiera-large`

Used for automatic person segmentation in 3D reconstruction pipeline.

**Manual download** (optional):
```python
from huggingface_hub import hf_hub_download
hf_hub_download(
    repo_id='facebook/sam2-hiera-large',
    filename='sam2_hiera_large.pt',
    local_dir='./backend/3d/models'
)
```

### 3. TripoSR (3D Reconstruction) - ~500MB

**Auto-downloads on first 3D reconstruction**

Model: `stabilityai/TripoSR`

### 4. SDXL (Body Generation) - ~7GB

**Auto-downloads on first body generation**

Model: `stabilityai/sdxl-turbo` (faster variant)

Alternative: `stabilityai/stable-diffusion-xl-base-1.0` (higher quality)

### 5. InstantID (Identity-Preserving) - ~1.5GB

**Optional**: For identity-preserving body generation

Run download script:
```powershell
cd backend
.\download_instantid_models.ps1
```

Or manually:
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

InsightFace model (antelopev2):
- Download: https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip
- Extract to: `backend/ml_engine/weights/instantid/models/antelopev2/`

### 6. Segformer (Clothes Parsing) - ~100MB

**Auto-downloads on first use**

Model: `mattmdjaga/segformer_b2_clothes`

### 7. OpenPose (Pose Detection) - ~200MB

**Auto-downloads on first use**

Model: `lllyasviel/ControlNet`

---

## Database Setup

### Supabase Setup

1. Create account at: https://supabase.com/
2. Create new project
3. Go to Project Settings → API
4. Copy:
   - Project URL
   - `anon` public key (for frontend)
   - `service_role` secret key (for backend)

### Database Schema

Run the SQL migration:

```sql
-- From backend/database_schema_complete.sql
-- Copy and paste into Supabase SQL Editor
```

Creates tables:
- `users` - User profiles
- `garments` - Wardrobe items
- `tryon_history` - Try-on results
- `user_preferences` - User settings

### Storage Buckets

Create storage buckets in Supabase:
1. `garments` - For uploaded garment images
2. `results` - For try-on results
3. `avatars` - For user profile photos

Set policies for authenticated access only.

---

## Environment Configuration

### Frontend (.env.local)

```env
# API Configuration
NEXT_PUBLIC_API_URL=http://localhost:8000

# Supabase Configuration
NEXT_PUBLIC_SUPABASE_URL=https://your-project.supabase.co
NEXT_PUBLIC_SUPABASE_ANON_KEY=your-anon-key-here
```

### Backend (.env)

```env
# Gemini API (for AI recommendations)
GEMINI_API_KEY=your-gemini-api-key-here

# Hugging Face (for model downloads)
HUGGINGFACE_TOKEN=your-huggingface-token-here

# RapidAPI (for eBay product search)
RAPIDAPI_KEY=your-rapidapi-key-here
RAPIDAPI_HOST=ebay32.p.rapidapi.com

# GPU Configuration
USE_GPU=true

# Supabase Configuration
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_KEY=your-service-role-key-here

# Server Configuration
LOG_LEVEL=INFO
ALLOWED_ORIGINS=*

# Model Preloading (set to empty for faster startup during development)
PRELOAD_MODELS=
WARMUP_MODELS=false
```

### Getting API Keys

**Gemini API Key**:
1. Go to: https://makersuite.google.com/app/apikey
2. Create API key
3. Copy to `.env`

**HuggingFace Token**:
1. Go to: https://huggingface.co/settings/tokens
2. Create new token (read access)
3. Copy to `.env`

**RapidAPI Key** (optional):
1. Go to: https://rapidapi.com/
2. Subscribe to eBay API
3. Copy API key to `.env`

---

## Running the Application

### Option 1: Start Both Services (Recommended)

**Windows PowerShell**:
```powershell
.\dev_start.ps1
```

This starts:
- Backend at http://localhost:8000
- Frontend at http://localhost:3000

### Option 2: Start Services Separately

**Terminal 1 - Backend**:
```bash
cd backend
python main.py
```

**Terminal 2 - Frontend**:
```bash
cd frontend
npm run dev
```

### First Run

On first startup, the backend will:
1. Download Leffa checkpoints (~8GB, 10-30 minutes)
2. Initialize ML models
3. Create log files

**Expected output**:
```
============================================================
Starting Virtual Try-On ML API...
============================================================
Compute device: NVIDIA GeForce RTX 3050 4GB Laptop GPU
Preloading ML models...
Downloading Leffa checkpoints from HuggingFace...
This may take 10-30 minutes on first run (several GB)
...
Leffa checkpoints downloaded successfully
ML models preloaded successfully
============================================================
Virtual Try-On ML API is ready!
============================================================
INFO:     Uvicorn running on http://0.0.0.0:8000
```

### Access the Application

- **Frontend**: http://localhost:3000
- **Backend API Docs**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health

---

## Troubleshooting

### Issue 1: PyTorch CUDA Not Available

**Symptom**: `torch.cuda.is_available()` returns `False`

**Solutions**:
1. Verify CUDA 12.4 is installed: `nvcc --version`
2. Reinstall PyTorch:
   ```bash
   pip uninstall torch torchvision torchaudio -y
   pip install torch==2.6.0+cu124 torchvision==0.21.0+cu124 torchaudio==2.6.0+cu124 --index-url https://download.pytorch.org/whl/cu124
   ```
3. Check GPU drivers are up to date
4. Restart computer after CUDA installation

### Issue 2: torchmcubes Compilation Fails (Windows)

**Symptom**: `error: Microsoft Visual C++ 14.0 or greater is required`

**Solution**:
1. Install Visual Studio 2022 Build Tools (see Prerequisites)
2. Set environment variables:
   ```powershell
   $env:PATH = "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Tools\MSVC\<version>\bin\Hostx64\x64;" + $env:PATH
   $env:CUDA_PATH = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4"
   ```
3. Retry installation:
   ```bash
   pip install --no-cache-dir git+https://github.com/tatsy/torchmcubes.git
   ```

See `docs/TRIPOSR_SETUP.md` for detailed compilation guide.

### Issue 3: Leffa Not Found

**Symptom**: `Leffa repository not found` or `ImportError: No module named 'leffa'`

**Solution**:
1. Verify Leffa is cloned at project root:
   ```bash
   ls Leffa/leffa  # Should show leffa module files
   ```
2. If missing, clone it:
   ```bash
   cd <project-root>
   git clone https://github.com/franciszzj/Leffa
   ```
3. Restart backend

### Issue 4: Out of Memory (OOM) Errors

**Symptom**: `CUDA out of memory` during inference

**Solutions**:
1. Close other GPU applications
2. Reduce batch size in requests
3. Enable CPU offload in `.env`:
   ```env
   USE_GPU=false  # Use CPU for preprocessing
   ```
4. For 4GB VRAM, the app is already optimized:
   - SAM2 runs on CPU
   - Depth Anything V2 runs on CPU
   - TripoSR uses FP16 precision
   - SDXL uses CPU offloading

### Issue 5: Frontend Build Errors

**Symptom**: `Module not found` or TypeScript errors

**Solutions**:
1. Delete node_modules and reinstall:
   ```bash
   cd frontend
   rm -rf node_modules package-lock.json
   npm install
   ```
2. Clear Next.js cache:
   ```bash
   rm -rf .next
   npm run dev
   ```
3. Verify Node.js version:
   ```bash
   node --version  # Should be 18+
   ```

### Issue 6: Supabase Connection Errors

**Symptom**: `Invalid API key` or `Failed to fetch`

**Solutions**:
1. Verify API keys in `.env` and `.env.local`
2. Check Supabase project is active
3. Verify CORS settings in Supabase dashboard
4. Test connection:
   ```bash
   curl https://your-project.supabase.co/rest/v1/
   ```

### Issue 7: Model Download Fails

**Symptom**: `Connection timeout` or `Failed to download`

**Solutions**:
1. Check internet connection
2. Set HuggingFace token in `.env`:
   ```env
   HUGGINGFACE_TOKEN=your-token-here
   ```
3. Use VPN if HuggingFace is blocked
4. Download models manually (see Model Downloads section)
5. Increase timeout in code if needed

### Issue 8: Port Already in Use

**Symptom**: `Address already in use: 8000` or `3000`

**Solutions**:
1. Kill process using port:
   ```bash
   # Windows
   netstat -ano | findstr :8000
   taskkill /PID <PID> /F
   
   # Linux/macOS
   lsof -ti:8000 | xargs kill -9
   ```
2. Or change port in configuration

---

## Advanced Configuration

### GPU Memory Optimization

For 4GB VRAM, edit `backend/.env`:
```env
# Disable model preloading for faster startup
PRELOAD_MODELS=
WARMUP_MODELS=false

# Force CPU for preprocessing
USE_GPU=true  # Keep true for TripoSR, but SAM2/Depth use CPU automatically
```

### Model Quality vs Speed

**Faster (lower quality)**:
- Use `stabilityai/sdxl-turbo` (2-4 steps)
- Reduce inference steps: `num_inference_steps=20`
- Enable `ref_acceleration` in Leffa

**Higher Quality (slower)**:
- Use `stabilityai/stable-diffusion-xl-base-1.0`
- Increase inference steps: `num_inference_steps=50`
- Disable `ref_acceleration`

### Production Deployment

1. **Set CORS origins**:
   ```env
   ALLOWED_ORIGINS=https://yourdomain.com
   ```

2. **Enable HTTPS**:
   - Use reverse proxy (nginx, Caddy)
   - Configure SSL certificates

3. **Use production server**:
   ```bash
   uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4
   ```

4. **Build frontend**:
   ```bash
   cd frontend
   npm run build
   npm run start
   ```

5. **Set up monitoring**:
   - Enable performance metrics
   - Configure error tracking
   - Set up log aggregation

### Docker Deployment

```bash
docker-compose up --build
```

See `docker-compose.yml` for configuration.

---

## Performance Benchmarks

### RTX 3050 4GB (Tested Configuration)

| Operation | Time | VRAM Usage |
|-----------|------|------------|
| Leffa Try-On | 15-20s | 3.5GB |
| Body Generation (SDXL) | 10-15s | 4GB (with offload) |
| 3D Reconstruction | 35-40s | 1.7GB |
| InstantID Generation | 60-80s | 4-6GB |
| AI Recommendations | 3-5s | N/A (API call) |

### RTX 3060 12GB

| Operation | Time | VRAM Usage |
|-----------|------|------------|
| Leffa Try-On | 8-12s | 5GB |
| Body Generation (SDXL) | 5-8s | 7GB |
| 3D Reconstruction | 20-25s | 3GB |
| InstantID Generation | 30-40s | 8GB |

### CPU Only (Not Recommended)

| Operation | Time |
|-----------|------|
| Leffa Try-On | 5-10 minutes |
| Body Generation | 2-5 minutes |
| 3D Reconstruction | 10-15 minutes |

---

## File Structure

```
virtual-try-on-platform/
├── backend/
│   ├── app/
│   │   ├── api/              # API endpoints
│   │   ├── core/             # Core utilities
│   │   ├── models/           # Data models
│   │   └── services/         # Business logic
│   ├── ml_engine/
│   │   ├── loader.py         # Model loader (singleton)
│   │   ├── pipelines/        # ML pipelines
│   │   └── weights/          # Model weights
│   ├── 3d/
│   │   ├── TripoSR/          # TripoSR repository
│   │   ├── models/           # SAM 2.1 weights
│   │   └── outputs/          # Generated 3D meshes
│   ├── logs/                 # Application logs
│   ├── main.py               # FastAPI entry point
│   ├── requirements.txt      # Python dependencies
│   └── .env                  # Environment variables
├── frontend/
│   ├── src/
│   │   ├── app/              # Next.js pages
│   │   ├── components/       # React components
│   │   └── lib/              # Utilities
│   ├── public/               # Static assets
│   ├── package.json          # Node dependencies
│   └── .env.local            # Environment variables
├── Leffa/                    # Leffa repository (clone here)
│   └── ckpts/                # Leffa checkpoints (auto-downloaded)
├── data/                     # Runtime data
│   ├── uploads/              # User uploads
│   └── results/              # Generated results
├── docs/                     # Documentation
├── .kiro/                    # Kiro configuration
├── SETUP.md                  # This file
└── README.md                 # Project overview
```

---

## Next Steps

After successful setup:

1. **Test the health endpoint**:
   ```bash
   curl http://localhost:8000/health
   ```

2. **Try the frontend**:
   - Open http://localhost:3000
   - Sign up / Log in
   - Upload a photo
   - Try virtual try-on

3. **Explore API docs**:
   - Visit http://localhost:8000/docs
   - Test endpoints interactively

4. **Read documentation**:
   - Product overview: `.kiro/steering/product.md`
   - Tech stack: `.kiro/steering/tech.md`
   - API guide: `docs/API.md`

---

## Support & Resources

- **Documentation**: See `docs/` folder
- **3D Setup**: `docs/TRIPOSR_SETUP.md`
- **InstantID Setup**: `docs/INSTANTID_SETUP.md`
- **Error Handling**: `backend/app/core/ERROR_HANDLING_GUIDE.md`

---

## Credits

- **Leffa**: https://github.com/franciszzj/Leffa
- **TripoSR**: https://github.com/VAST-AI-Research/TripoSR
- **SAM 2**: https://github.com/facebookresearch/sam2
- **InstantID**: https://github.com/instantX-research/InstantID
- **Depth Anything V2**: https://github.com/DepthAnything/Depth-Anything-V2

---

**Setup complete! 🎉**

If you encounter any issues not covered here, check the troubleshooting section or refer to the detailed setup guides in the `docs/` folder.
