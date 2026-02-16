# Technology Stack

## Frontend

**Framework**: Next.js 16.1.3 (React 19.2.3)
- App Router architecture
- TypeScript 5
- Server and Client Components

**Styling**: 
- Tailwind CSS 4
- Framer Motion 12.26.2 for animations
- Lucide React 0.562.0 for icons
- Plus Jakarta Sans font

**3D Graphics**:
- Three.js 0.182.0
- React Three Fiber 9.5.0
- React Three Drei 10.7.7

**State & UI**:
- Sonner 2.0.7 for toast notifications
- clsx + tailwind-merge for className utilities
- file-saver 2.0.5 for downloads

**Authentication**: Supabase client 2.90.1 (@supabase/supabase-js)

**Development**:
- ESLint 9 with Next.js config
- React Compiler (Babel plugin 1.0.0)

## Backend

**Framework**: FastAPI 0.128.5
- Uvicorn 0.40.0 ASGI server
- Pydantic 2.12.5 for data validation
- Python-multipart 0.0.22 for file uploads
- Python 3.10.x

**ML/AI Stack**:
- PyTorch 2.6.0+cu124 + TorchVision 0.21.0+cu124
- Diffusers 0.36.0
- Transformers 5.1.0
- Accelerate 1.12.0
- safetensors 0.7.0
- huggingface-hub 1.4.1
- PEFT 0.18.1 (Parameter-Efficient Fine-Tuning)

**2D Virtual Try-On (Leffa)**:
- Leffa repository (cloned at project root)
- Auto-downloads checkpoints from HuggingFace (franciszzj/Leffa)
- Supports upper_body, lower_body, and dresses
- Model variants: viton_hd (recommended), dress_code
- Advanced options: ref_acceleration, repaint mode

**3D Reconstruction**:
- TripoSR (stabilityai/TripoSR)
- SAM 2.1 (facebook/sam2.1-hiera-large) for segmentation
- Depth Anything V2 (depth-anything/Depth-Anything-V2-Large-hf)
- torchmcubes (compiled from source)
- open3d 0.19.0, trimesh 4.0.5, pymeshlab 2025.7.post1

**Body Generation**:
- SDXL via Diffusers for generic body generation
- **InstantID** (InstantX/InstantID) for identity-preserving generation
  - InsightFace 0.7.3 (antelopev2) for face embedding extraction
  - ControlNet for facial keypoint guidance
  - IP-Adapter for identity injection into diffusion process
  - Albumentations 1.4.23 for image augmentation
- Supports ethnicity, body type, height, weight parameters
- Gemini Vision for facial feature analysis (skin tone, ethnicity)

**Image Processing**:
- Pillow 10.1.0
- NumPy 1.26.4
- OpenCV 4.11.0.86
- rembg 2.0.69 for background removal
- segment-anything 1.0

**AI Services**:
- Google Gen AI SDK 1.10.0+ (Gemini 2.5 Flash)
  - Uses `google-genai` package (GA as of May 2025)
  - Replaces deprecated `google-generativeai`
  - Multimodal support (text, images, audio, video)
- Supabase 2.14.0 (Storage, Auth, Database)

**Infrastructure**:
- CUDA 12.4 (required for GPU support)
- CORS middleware for cross-origin requests
- Comprehensive error handling system
- Structured logging with coloredlogs
- Background task for temp file cleanup (15 min intervals)

## Environment Variables

### Frontend (.env.local)
```
NEXT_PUBLIC_API_URL=http://localhost:8000
NEXT_PUBLIC_SUPABASE_URL=<your-supabase-url>
NEXT_PUBLIC_SUPABASE_ANON_KEY=<your-supabase-key>
```

### Backend (.env)
```
SUPABASE_URL=<your-supabase-url>
SUPABASE_KEY=<your-supabase-key>
GEMINI_API_KEY=<your-gemini-key>
HUGGINGFACE_TOKEN=<your-huggingface-token>
USE_GPU=true
LOG_LEVEL=INFO
ALLOWED_ORIGINS=http://localhost:3000
WARMUP_MODELS=true
PRELOAD_MODELS=tryon,segmentation
```

## Common Commands

### Frontend Development
```bash
cd frontend
npm install          # Install dependencies
npm run dev          # Start dev server (http://localhost:3000)
npm run build        # Production build
npm run start        # Start production server
npm run lint         # Run ESLint
```

### Backend Development
```bash
cd backend
pip install -r requirements.txt    # Install dependencies
python main.py                     # Start dev server (http://localhost:8000)
uvicorn main:app --reload          # Alternative dev server
```

### Docker Deployment
```bash
docker-compose up --build          # Build and start services
docker-compose down                # Stop services
```

### Development Script
```powershell
.\dev_start.ps1                    # Start both frontend and backend
```

## API Endpoints

Base URL: `http://localhost:8000/api/v1`

**Core Endpoints**:
- `GET /health` - Health check (GPU, Leffa, Gemini, memory)
- `POST /recommend` - AI outfit recommendations (Gemini + eBay)
- `POST /process-tryon` - 2D virtual try-on (Leffa)
- `POST /process-tryon-batch` - Batch try-on (multiple garments)

**Body Generation**:
- `POST /generate-bodies` - Generate body variations (SDXL)
- `POST /generate-full-body` - Generate full body with identity (legacy)
- `POST /generate-identity-body` - Identity-preserving generation (InstantID)
- `POST /analyze-face-features` - Analyze facial features with Gemini

**Image Analysis**:
- `POST /analyze-body` - Detect if image is head-only or full-body
- `POST /combine-head-body` - Combine head with generated body

**3D Reconstruction**:
- `POST /reconstruct-3d` - Generate 3D mesh from image

**Wardrobe & History**:
- `GET /wardrobe/{user_id}` - Get user's garment collection
- `GET /tryon/history/{user_id}` - Get try-on history
- `POST /garments/upload` - Upload garment to wardrobe
- `DELETE /garments/{garment_id}` - Delete garment

## GPU Requirements

**Minimum**: NVIDIA GPU with 4GB VRAM (tested on RTX 3050 4GB)
**Recommended**: 8GB+ VRAM for optimal performance
**CUDA**: 12.4 (required for PyTorch 2.6.0+cu124)

CPU fallback available but significantly slower (10-20x).
