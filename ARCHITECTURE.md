# Virtual Try-On Platform - System Architecture & Methodology

**Version**: 1.0  
**Last Updated**: February 16, 2026  
**Status**: Production

---

## Table of Contents

1. [System Overview](#system-overview)
2. [Architecture Layers](#architecture-layers)
3. [Core Pipelines](#core-pipelines)
4. [ML Model Methodology](#ml-model-methodology)
5. [Data Flow](#data-flow)
6. [Memory Management](#memory-management)
7. [API Design](#api-design)
8. [Security Architecture](#security-architecture)
9. [Performance Optimization](#performance-optimization)
10. [Deployment Architecture](#deployment-architecture)

---

## System Overview

### Vision

An AI-powered fashion technology platform that enables users to virtually try on clothing, generate personalized body models, receive AI-driven outfit recommendations, and visualize garments in 3D.

### Architecture Philosophy

1. **Modular Design**: Each feature is a self-contained pipeline
2. **Lazy Loading**: Models load on-demand to minimize memory footprint
3. **Singleton Pattern**: Single model instance shared across requests
4. **Aggressive Memory Management**: GPU memory cleared between operations
5. **Graceful Degradation**: Fallbacks when services unavailable
6. **API-First**: Backend exposes RESTful APIs consumed by frontend

### Technology Stack Summary

**Frontend**: Next.js 16 (React 19) + TypeScript + Tailwind CSS + Three.js  
**Backend**: FastAPI + PyTorch 2.6.0 + CUDA 12.4  
**ML Models**: Leffa, TripoSR, SAM 2.1, SDXL, InstantID, Gemini 2.5 Flash  
**Storage**: Supabase (PostgreSQL + Object Storage)  
**Auth**: Supabase JWT  
**Deployment**: Docker + Uvicorn

---

## Architecture Layers

### 1. Presentation Layer (Frontend)

```
┌─────────────────────────────────────────────────────────┐
│                    Next.js Frontend                      │
├─────────────────────────────────────────────────────────┤
│  Pages:                                                  │
│  - Landing (/)                                           │
│  - Auth (/auth)                                          │
│  - Onboarding (/onboard)                                 │
│  - Studio (/studio) - Virtual Try-On                     │
│  - Wardrobe (/wardrobe) - Garment Management            │
│  - Shop (/shop) - AI Recommendations                     │
├─────────────────────────────────────────────────────────┤
│  Components:                                             │
│  - TryOnWidget - Upload & process try-on                 │
│  - ModelViewer - 3D visualization (Three.js)             │
│  - Recommendations - AI outfit suggestions               │
│  - ProtectedRoute - Auth guard                           │
├─────────────────────────────────────────────────────────┤
│  Services:                                               │
│  - api.ts - Backend API client                           │
│  - supabase.ts - Auth & storage client                   │
└─────────────────────────────────────────────────────────┘
```

### 2. API Layer (Backend)

```
┌─────────────────────────────────────────────────────────┐
│                   FastAPI Backend                        │
├─────────────────────────────────────────────────────────┤
│  Endpoints (app/api/):                                   │
│  - endpoints.py - Core (health, recommend, try-on)       │
│  - body_generation.py - Body model generation            │
│  - identity_body.py - InstantID generation               │
│  - image_analysis.py - Body type detection               │
│  - image_composition.py - Head-body composition          │
│  - garment_management.py - Wardrobe CRUD                 │
│  - reconstruction_3d.py - 3D mesh generation             │
├─────────────────────────────────────────────────────────┤
│  Middleware:                                             │
│  - CORS - Cross-origin protection                        │
│  - RequestLogging - Request/response logging             │
│  - ErrorHandler - Centralized error handling             │
│  - Auth - JWT validation (Supabase)                      │
└─────────────────────────────────────────────────────────┘
```

### 3. Service Layer (Business Logic)

```
┌─────────────────────────────────────────────────────────┐
│                  Service Layer                           │
├─────────────────────────────────────────────────────────┤
│  - tryon_service.py - Virtual try-on orchestration       │
│  - recommendation.py - AI recommendations engine         │
│  - body_generation.py - Body model generation            │
│  - face_analysis.py - Gemini Vision face analysis        │
│  - body_detection.py - Full-body vs head-only detection  │
│  - image_collage.py - Multi-image collage creation       │
│  - supabase_storage.py - Storage & database operations   │
│  - temp_file_manager.py - Temporary file cleanup         │
└─────────────────────────────────────────────────────────┘
```

### 4. ML Engine Layer (Model Management)

```
┌─────────────────────────────────────────────────────────┐
│                    ML Engine                             │
├─────────────────────────────────────────────────────────┤
│  loader.py - Singleton model loader                      │
│  - load_tryon() - Leffa pipeline                         │
│  - load_triposr() - 3D reconstruction                    │
│  - load_sam2() - Segmentation                            │
│  - load_sdxl() - Body generation                         │
│  - load_segmentation() - Clothes parsing                 │
│  - load_pose() - Pose estimation                         │
├─────────────────────────────────────────────────────────┤
│  pipelines/ - ML pipeline implementations                │
│  - tryon.py - Leffa virtual try-on                       │
│  - reconstruction_3d.py - TripoSR + SAM2                 │
│  - body_gen.py - SDXL body generation                    │
│  - instantid_pipeline.py - Identity-preserving gen       │
│  - segmentation.py - SAM 2.1 segmentation                │
└─────────────────────────────────────────────────────────┘
```

### 5. Data Layer

```
┌─────────────────────────────────────────────────────────┐
│                  Supabase (PostgreSQL)                   │
├─────────────────────────────────────────────────────────┤
│  Tables:                                                 │
│  - users - User profiles & preferences                   │
│  - garments - Wardrobe items                             │
│  - tryon_history - Try-on results                        │
│  - user_preferences - Settings                           │
├─────────────────────────────────────────────────────────┤
│  Storage Buckets:                                        │
│  - garments - Uploaded garment images                    │
│  - results - Generated try-on results                    │
│  - avatars - User profile photos                         │
└─────────────────────────────────────────────────────────┘
```

---

## Core Pipelines

### Pipeline 1: Virtual Try-On (Leffa)

**Purpose**: Generate realistic images of users wearing different garments

**Methodology**:

1. **Input Processing**
   - User uploads person image + garment image
   - Images validated (format, size, dimensions)
   - Garment type detected (upper_body, lower_body, dresses)

2. **Preprocessing (Leffa Built-in)**
   - **DensePose**: Extract body pose and surface representation
   - **Human Parsing**: Segment body parts (ATR + LIP models)
   - **OpenPose**: Detect keypoints (joints, limbs)
   - **SCHP**: Generate garment-agnostic mask
   - **AutoMasker**: Create mask for garment region

3. **Virtual Try-On Generation**
   - **Base Model**: Stable Diffusion Inpainting (4GB)
   - **Leffa Model**: Flow-based diffusion (VITON-HD or DressCode)
   - **Reference UNet**: Preserves garment details
   - **Inference**: 30 steps (default), guidance scale 2.5
   - **Output**: 768x1024 realistic try-on image

4. **Post-Processing**
   - Result saved to Supabase storage
   - Metadata stored in database
   - Public URL returned to frontend

**Key Features**:
- Preserves garment texture and patterns
- Maintains body pose and proportions
- Handles complex garments (prints, logos, wrinkles)
- Supports batch processing (multiple garments)

**Performance**:
- RTX 3050 4GB: 15-20 seconds per image
- RTX 3060 12GB: 8-12 seconds per image
- CPU: 5-10 minutes (not recommended)

---

### Pipeline 2: Body Generation (SDXL + InstantID)

**Purpose**: Generate full-body images from head-only photos or create synthetic bodies

**Methodology**:

#### Option A: SDXL (Generic Body Generation)

1. **Face Analysis (Gemini Vision)**
   - Extract skin tone, ethnicity, age, gender
   - Analyze facial features, hair color/style
   - Generate descriptive prompt additions

2. **Prompt Construction**
   - Combine face analysis + user parameters
   - Body type: slim, athletic, muscular, average, curvy, plus
   - Height/weight: 140-220cm, 40-200kg
   - Pose: standing, walking, sitting
   - Clothing: casual minimal, formal, etc.

3. **SDXL Generation**
   - Model: stabilityai/sdxl-turbo (fast) or sdxl-base-1.0 (quality)
   - Resolution: 1024x768 (portrait)
   - Steps: 4 (turbo) or 50 (base)
   - Guidance: 0.0 (turbo) or 7.5 (base)
   - CPU offloading enabled for 4GB VRAM

4. **Output**
   - Generated body image
   - May require head stitching for identity preservation

#### Option B: InstantID (Identity-Preserving Generation)

1. **Face Embedding Extraction (InsightFace)**
   - Load antelopev2 face model
   - Extract 512-dimensional face embedding
   - Detect facial keypoints (eyes, nose, mouth)

2. **Keypoint Image Generation**
   - Draw facial keypoints on canvas
   - Used for ControlNet guidance

3. **InstantID Generation**
   - **Base**: SDXL 1.0
   - **ControlNet**: Guides face positioning
   - **IP-Adapter**: Injects face embedding into diffusion
   - **LCM-LoRA**: Fast inference (20 steps)
   - **Face embedding**: Preserves identity throughout generation

4. **Output**
   - Full-body image with user's actual face
   - No stitching required - face is natively embedded
   - Higher quality identity preservation

**Comparison**:

| Feature | SDXL | InstantID |
|---------|------|-----------|
| Identity Preservation | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| Speed | Fast (5-10s) | Medium (15-20s) |
| VRAM | 2-3GB | 4-6GB |
| Setup | Simple | Complex (models required) |
| Quality | Good | Excellent |

---

### Pipeline 3: AI Recommendations (Gemini + eBay)

**Purpose**: Provide personalized outfit recommendations based on scientific color theory

**Methodology**:

1. **Skin Tone Analysis (Gemini Vision)**
   - Analyze user photo
   - Determine skin tone category (12 categories)
   - Identify undertone (cool, warm, neutral)
   - Fitzpatrick scale (1-6)

2. **Color Theory Mapping**
   - Map skin tone to color palette
   - Best colors: Colors that complement skin tone
   - Avoid colors: Colors that clash
   - Metals: Gold vs silver recommendations
   - Neutrals: Safe base colors

3. **Image Collage Creation**
   - Combine user photo + wardrobe + generated bodies
   - Add labels for context
   - Create visual summary for Gemini

4. **Keyword Extraction (Gemini Vision)**
   - Analyze collage with color theory context
   - Extract 8-10 fashion keywords
   - Consider user profile (gender, style, body type)
   - Prioritize colors from "best colors" palette

5. **eBay Product Search (RapidAPI)**
   - Search eBay for each keyword
   - Filter by category (clothing only)
   - Get top 4 products per keyword
   - Deduplicate by product ID

6. **Result Ranking**
   - Return top 20 products
   - Include: name, price, image, eBay link
   - Sorted by relevance

**Color Theory Categories**:
- fair_cool, fair_warm
- light_cool, light_warm
- medium_cool, medium_warm
- olive
- tan_warm, tan_cool
- deep_warm, deep_cool, deep_neutral

**Circuit Breaker Pattern**:
- Protects against API failures
- Fails fast when service is down
- Auto-recovery after timeout
- Fallback recommendations on failure

---

### Pipeline 4: 3D Reconstruction (TripoSR + SAM2)

**Purpose**: Generate 3D meshes from 2D images for advanced visualization

**Methodology**:

1. **Segmentation (SAM 2.1)**
   - **Model**: facebook/sam2.1-hiera-large
   - **Method**: Automatic mask generation
   - **Strategy**: Select largest mask (person)
   - **Output**: RGBA image with transparent background
   - **Device**: CPU (forced for 4GB VRAM optimization)

2. **Image Preprocessing**
   - Resize to 512x512
   - Handle RGBA: composite on gray background
   - Foreground centering (85% fill)

3. **3D Mesh Generation (TripoSR)**
   - **Model**: stabilityai/TripoSR
   - **Method**: Single-image 3D reconstruction
   - **Resolution**: Adaptive (1024 → 512 → 256 → 128)
   - **Chunk Size**: 8192 (for 4GB VRAM)
   - **Precision**: FP16 on GPU
   - **Features**: Adaptive threshold, post-processing

4. **Mesh Post-Processing**
   - Coordinate transformation (Y-up to Z-up)
   - Rotation (60° around Y-axis)
   - Vertex color preservation
   - Mesh smoothing (optional)

5. **Export**
   - Formats: GLB, OBJ, PLY
   - Includes vertex colors (texture)
   - Optimized for web viewing

**Memory Strategy (4GB VRAM)**:
- SAM2 on CPU (offloaded immediately)
- TripoSR on GPU with FP16
- Nuclear memory cleanup between stages
- Adaptive resolution fallback on OOM

**Performance**:
- RTX 3050 4GB: 35-40 seconds
- RTX 3060 12GB: 20-25 seconds
- Resolution: 256 (4GB) or 512 (8GB+)

---

### Pipeline 5: Smart Onboarding

**Purpose**: Analyze user photos and generate complete avatars

**Methodology**:

1. **Body Type Detection**
   - Analyze uploaded photo
   - Detect if full-body or head-only
   - Use pose estimation (keypoint coverage)
   - Calculate coverage metric (0.0-1.0)

2. **Decision Tree**
   ```
   Is Full Body?
   ├─ YES → Use photo directly for try-on
   └─ NO → Generate full body
       ├─ Extract face features (Gemini)
       ├─ Generate body (SDXL/InstantID)
       └─ Optional: Stitch head to body
   ```

3. **Head-Body Composition** (if needed)
   - Detect face in head-only photo
   - Detect face in generated body
   - Align and blend faces
   - Smooth transition at neck
   - Color correction for consistency

4. **Avatar Creation**
   - Save complete avatar to storage
   - Link to user profile
   - Ready for virtual try-on

---

## ML Model Methodology

### Model Loading Strategy (Singleton Pattern)

```python
class ModelLoader:
    _instance = None  # Singleton
    
    def __init__(self):
        self.models = {}  # Loaded models
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    def load_tryon(self):
        if "tryon" not in self.models:
            # Load Leffa pipeline
            self.models["tryon"] = LeffaPipeline(device=self.device)
            self.models["tryon"].load_models()
        return self.models["tryon"]
```

**Benefits**:
- Single model instance shared across requests
- Lazy loading (load on first use)
- Memory efficient (no duplicate models)
- Thread-safe with locks

### Model Specifications

#### 1. Leffa (Virtual Try-On)

**Architecture**: Flow-based diffusion model  
**Base**: Stable Diffusion Inpainting  
**Size**: ~8GB total

**Sub-Models**:
1. Stable Diffusion Inpainting (~4GB)
2. Virtual Try-On weights (~1.5GB)
3. DensePose (~200MB)
4. Human Parsing (~200MB)
5. OpenPose (~200MB)
6. SCHP (~200MB)

**Input**: Person image (768x1024) + Garment image (768x1024)  
**Output**: Try-on result (768x1024)  
**Inference**: 30 steps, ~15-20s on RTX 3050

#### 2. TripoSR (3D Reconstruction)

**Architecture**: Transformer-based 3D reconstruction  
**Size**: ~500MB  
**Input**: Single RGB image (512x512)  
**Output**: 3D mesh (vertices + faces + colors)  
**Resolution**: 128-1024 (adaptive)  
**Inference**: ~11s on RTX 3050 (GPU), ~7s preprocessing (CPU)

#### 3. SAM 2.1 (Segmentation)

**Architecture**: Vision Transformer + Mask Decoder  
**Size**: ~900MB  
**Input**: RGB image (any size)  
**Output**: Segmentation masks  
**Method**: Automatic mask generation  
**Inference**: ~13s on CPU (forced for memory optimization)

#### 4. SDXL (Body Generation)

**Architecture**: Latent Diffusion Model  
**Variants**:
- sdxl-turbo: 4 steps, fast (~5-10s)
- sdxl-base-1.0: 50 steps, quality (~30-40s)

**Size**: ~7GB  
**Input**: Text prompt  
**Output**: 1024x768 image  
**Memory**: 4GB with CPU offloading

#### 5. InstantID (Identity-Preserving)

**Architecture**: SDXL + ControlNet + IP-Adapter  
**Size**: ~1.5GB (additional to SDXL)

**Components**:
- ControlNet: Face keypoint guidance (~800MB)
- IP-Adapter: Identity injection (~700MB)
- InsightFace: Face embedding (~200MB)

**Input**: Face image + text prompt  
**Output**: Full-body with preserved identity  
**Inference**: ~15-20s with LCM-LoRA

#### 6. Gemini 2.5 Flash (AI Analysis)

**Type**: Cloud API (Google)  
**Capabilities**: Vision + Text  
**Use Cases**:
- Face analysis (skin tone, ethnicity)
- Fashion keyword extraction
- Color theory recommendations

**Latency**: 2-5 seconds per request  
**Cost**: Pay-per-use (API calls)

---

## Data Flow

### Request Flow: Virtual Try-On

```
User (Frontend)
    ↓ POST /api/v1/process-tryon
    ↓ {person_image, garment_image, options}
    ↓
FastAPI Endpoint (endpoints.py)
    ↓ Validate inputs
    ↓ Extract options
    ↓
TryOnService (tryon_service.py)
    ↓ Load Leffa pipeline (lazy)
    ↓
Leffa Pipeline (tryon.py)
    ↓ Preprocess (DensePose, Parsing, OpenPose, SCHP)
    ↓ Generate try-on (Diffusion)
    ↓ Return result image
    ↓
TryOnService
    ↓ Save to Supabase storage
    ↓ Store metadata in database
    ↓
FastAPI Endpoint
    ↓ Return {result_url, metadata}
    ↓
User (Frontend)
    ↓ Display result
```

### Request Flow: AI Recommendations

```
User (Frontend)
    ↓ POST /api/v1/recommend
    ↓ {user_photo, wardrobe_images, user_profile}
    ↓
FastAPI Endpoint (endpoints.py)
    ↓ Validate inputs
    ↓
RecommendationEngine (recommendation.py)
    ↓
Step 1: Skin Tone Analysis
    ↓ Gemini Vision API
    ↓ Extract skin tone category
    ↓ Map to color palette
    ↓
Step 2: Create Collage
    ↓ Combine all images
    ↓ Add labels
    ↓
Step 3: Extract Keywords
    ↓ Gemini Vision API
    ↓ Analyze with color theory
    ↓ Return 8-10 keywords
    ↓
Step 4: Search eBay
    ↓ RapidAPI (eBay)
    ↓ 4 products per keyword
    ↓ Deduplicate
    ↓
Step 5: Return Results
    ↓ Top 20 products
    ↓ {name, price, image, link}
    ↓
User (Frontend)
    ↓ Display recommendations
```

### Request Flow: 3D Reconstruction

```
User (Frontend)
    ↓ POST /api/v1/reconstruct-3d
    ↓ {image, options}
    ↓
FastAPI Endpoint (reconstruction_3d.py)
    ↓ Validate input
    ↓
ThreeDReconstructionPipeline (reconstruction_3d.py)
    ↓
Stage 1: Segmentation (SAM2 on CPU)
    ↓ Automatic mask generation
    ↓ Select largest mask
    ↓ Create RGBA image
    ↓ Offload SAM2 immediately
    ↓
Stage 2: 3D Generation (TripoSR on GPU)
    ↓ Nuclear memory cleanup
    ↓ Load TripoSR
    ↓ Preprocess image
    ↓ Encode to scene codes
    ↓ Extract mesh (adaptive resolution)
    ↓ Transform coordinates
    ↓
Stage 3: Export
    ↓ Convert to GLB/OBJ/PLY
    ↓ Return file bytes
    ↓
FastAPI Endpoint
    ↓ Return {file_bytes, format}
    ↓
User (Frontend)
    ↓ Display in ModelViewer (Three.js)
```

---

## Memory Management

### Challenge: 4GB VRAM Constraint

The platform is optimized to run on NVIDIA RTX 3050 (4GB VRAM), which requires aggressive memory management.

### Strategies

#### 1. Model Offloading

```python
# Force models to CPU when not in use
def offload_to_cpu(model_name):
    if model_name in self.models:
        self.models[model_name].to('cpu')
        torch.cuda.empty_cache()
        gc.collect()
```

**Applied to**:
- SAM2: Always on CPU (large model, used once)
- Depth models: CPU only
- SDXL: CPU offloading enabled

#### 2. Nuclear Memory Reset

```python
def reset_cuda_memory():
    # Move all models to CPU
    for model in self.models.values():
        model.to('cpu')
    
    # Multiple rounds of cleanup
    for _ in range(3):
        gc.collect()
        torch.cuda.empty_cache()
    
    # Synchronize and collect IPC
    torch.cuda.synchronize()
    torch.cuda.ipc_collect()
    
    # Reset stats
    torch.cuda.reset_peak_memory_stats()
```

**Applied**:
- Between 3D reconstruction stages
- After large model inference
- Before loading new models

#### 3. FP16 Precision

```python
# Use half precision on GPU
model = model.half()  # FP32 → FP16 (50% memory reduction)
```

**Applied to**:
- TripoSR: FP16 on GPU
- SDXL: FP16 with CPU offload
- Leffa: FP16 on CUDA

#### 4. Adaptive Resolution

```python
# Try high resolution, fallback on OOM
resolutions = [1024, 512, 256, 128]
for res in resolutions:
    try:
        mesh = generate_mesh(resolution=res)
        break
    except RuntimeError as e:
        if "out of memory" in str(e):
            continue  # Try lower resolution
```

**Applied to**:
- TripoSR mesh extraction
- SDXL generation (reduce size on OOM)

#### 5. Lazy Loading

```python
# Load models only when needed
def load_tryon(self):
    if "tryon" not in self.models:
        self.models["tryon"] = LeffaPipeline()
    return self.models["tryon"]
```

**Benefits**:
- Faster startup
- Lower idle memory
- Load only what's used

### Memory Usage Breakdown

| Operation | VRAM Usage | Strategy |
|-----------|------------|----------|
| Idle | ~0.14GB | Minimal |
| Leffa Try-On | ~3.5GB | FP16, efficient attention |
| SDXL Generation | ~4GB | CPU offloading |
| TripoSR | ~1.7GB | FP16, chunked extraction |
| SAM2 | ~0GB | Forced to CPU |
| InstantID | ~4-6GB | CPU offload + FP16 |

---

## API Design

### RESTful Principles

1. **Resource-Based URLs**: `/api/v1/garments/{id}`
2. **HTTP Methods**: GET, POST, PUT, DELETE
3. **Status Codes**: 200 (OK), 201 (Created), 400 (Bad Request), 401 (Unauthorized), 404 (Not Found), 500 (Server Error)
4. **JSON Responses**: Consistent structure

### API Structure

```
/api/v1/
├── health                    GET    Health check
├── recommend                 POST   AI recommendations
├── process-tryon             POST   Virtual try-on
├── process-tryon-batch       POST   Batch try-on
├── generate-bodies           POST   Body generation (SDXL)
├── generate-identity-body    POST   Identity-preserving (InstantID)
├── analyze-body              POST   Body type detection
├── analyze-face-features     POST   Face analysis (Gemini)
├── combine-head-body         POST   Head-body composition
├── reconstruct-3d            POST   3D mesh generation
├── garments/
│   ├── upload                POST   Upload garment
│   ├── {garment_id}          GET    Get garment
│   └── {garment_id}          DELETE Delete garment
├── wardrobe/{user_id}        GET    Get user wardrobe
└── tryon/history/{user_id}   GET    Get try-on history
```

### Request/Response Format

**Standard Success Response**:
```json
{
  "success": true,
  "data": { ... },
  "metadata": {
    "processing_time": 15.23,
    "request_id": "uuid"
  }
}
```

**Standard Error Response**:
```json
{
  "success": false,
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Invalid image format",
    "details": { ... }
  }
}
```

### Authentication

**Method**: JWT (JSON Web Tokens) via Supabase

**Flow**:
1. User logs in via Supabase Auth
2. Frontend receives JWT token
3. Token included in `Authorization: Bearer <token>` header
4. Backend validates token with Supabase
5. User ID extracted from token
6. Data isolation enforced (users see only their data)

---

## Security Architecture

### 1. Authentication & Authorization

- **JWT Tokens**: Supabase-issued, validated on each request
- **User Isolation**: Database queries filtered by user_id
- **Role-Based Access**: Future: admin, user, guest roles

### 2. Input Validation

```python
# File upload validation
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
ALLOWED_TYPES = ["image/jpeg", "image/png", "image/webp"]

def validate_image(file):
    if file.size > MAX_FILE_SIZE:
        raise ValueError("File too large")
    if file.content_type not in ALLOWED_TYPES:
        raise ValueError("Invalid file type")
```

### 3. CORS Protection

```python
# Configurable origins
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*").split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)
```

### 4. Rate Limiting

**Future Implementation**:
- Per-user rate limits
- Per-IP rate limits
- Circuit breakers for external APIs

### 5. Data Privacy

- **User Data Isolation**: Users can only access their own data
- **Secure Storage**: Supabase with RLS (Row Level Security)
- **Temporary Files**: Auto-cleanup every 15 minutes
- **No PII Logging**: Sensitive data not logged

---

## Performance Optimization

### 1. Model Caching (Singleton)

- Single model instance per process
- Shared across all requests
- Reduces load time from 30s to 0s (after first load)

### 2. Async Processing

```python
# Non-blocking I/O
@app.post("/recommend")
async def recommend(request: RecommendRequest):
    result = await recommendation_engine.get_recommendations(...)
    return result
```

### 3. Connection Pooling

```python
# HTTP client with connection pooling
http_client = httpx.AsyncClient(
    limits=httpx.Limits(
        max_connections=100,
        max_keepalive_connections=20
    ),
    http2=True
)
```

### 4. Background Tasks

```python
# Cleanup temp files every 15 minutes
async def periodic_cleanup():
    while True:
        await asyncio.sleep(900)
        temp_file_manager.cleanup_expired_files()
```

### 5. Circuit Breakers

```python
# Fail fast when external service is down
class CircuitBreaker:
    states = ["CLOSED", "OPEN", "HALF_OPEN"]
    
    def call(self, func):
        if self.state == "OPEN":
            raise Exception("Circuit breaker is OPEN")
        try:
            result = func()
            self.on_success()
            return result
        except:
            self.on_failure()
            raise
```

**Applied to**:
- Gemini API calls
- eBay API calls
- External service integrations

### 6. Caching Strategy

**Future Implementation**:
- Redis for session data
- Model output caching (same inputs → cached result)
- CDN for static assets

---

## Deployment Architecture

### Development

```
┌─────────────────────────────────────────────────────────┐
│                  Local Development                       │
├─────────────────────────────────────────────────────────┤
│  Frontend: npm run dev (localhost:3000)                  │
│  Backend: python main.py (localhost:8000)                │
│  Database: Supabase Cloud                                │
│  Storage: Supabase Cloud                                 │
└─────────────────────────────────────────────────────────┘
```

### Production (Docker)

```
┌─────────────────────────────────────────────────────────┐
│                   Docker Compose                         │
├─────────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌─────────────────┐              │
│  │  Frontend       │  │  Backend        │              │
│  │  (Next.js)      │  │  (FastAPI)      │              │
│  │  Port: 3000     │  │  Port: 8000     │              │
│  └─────────────────┘  └─────────────────┘              │
│           │                    │                         │
│           └────────┬───────────┘                         │
│                    │                                     │
│           ┌────────▼────────┐                           │
│           │   Supabase      │                           │
│           │   (Cloud)       │                           │
│           └─────────────────┘                           │
└─────────────────────────────────────────────────────────┘
```

### Scaling Strategy

**Horizontal Scaling**:
- Multiple backend instances behind load balancer
- Shared model cache (Redis)
- Stateless API design

**Vertical Scaling**:
- Upgrade GPU (8GB → 12GB → 24GB)
- Increase CPU cores for parallel processing
- More RAM for model caching

**GPU Optimization**:
- Batch inference (process multiple requests together)
- Model quantization (INT8 for faster inference)
- TensorRT optimization (future)

---

## Conclusion

This architecture provides:

✅ **Modularity**: Each feature is independent  
✅ **Scalability**: Horizontal and vertical scaling  
✅ **Performance**: Optimized for 4GB VRAM  
✅ **Reliability**: Circuit breakers, error handling  
✅ **Security**: JWT auth, input validation, CORS  
✅ **Maintainability**: Clean code, logging, monitoring  

**Next Steps**:
1. Implement caching layer (Redis)
2. Add rate limiting
3. Set up monitoring (Prometheus + Grafana)
4. Implement A/B testing framework
5. Add analytics and user behavior tracking

---

**Document Version**: 1.0  
**Last Updated**: February 16, 2026  
**Maintained By**: Development Team
