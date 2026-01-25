# Technology Stack

## Frontend

**Framework**: Next.js 16.1.3 (React 19.2.3)
- App Router architecture
- TypeScript 5
- Server and Client Components

**Styling**: 
- Tailwind CSS 4
- Framer Motion for animations
- Lucide React for icons

**3D Graphics**:
- Three.js
- React Three Fiber
- React Three Drei

**State & UI**:
- Sonner for toast notifications
- clsx + tailwind-merge for className utilities

**Authentication**: Supabase client (@supabase/supabase-js)

**Development**:
- ESLint with Next.js config
- React Compiler (Babel plugin)

## Backend

**Framework**: FastAPI 0.109.0
- Uvicorn ASGI server
- Pydantic for data validation
- Python-multipart for file uploads

**ML/AI Stack**:
- PyTorch 2.1.2 + TorchVision 0.16.2
- Diffusers 0.25.0 (SDXL body generation)
- Transformers 4.36.2
- Accelerate 0.25.0
- xformers 0.23 (memory-efficient attention)
- safetensors 0.4.2

**Image Processing**:
- Pillow 10.2.0
- NumPy 1.26.3
- OpenCV 4.9.0.80

**AI Services**:
- Google Generative AI (Gemini Vision API)

**Infrastructure**:
- Docker + Docker Compose
- NVIDIA GPU support (CUDA)
- CORS middleware for cross-origin requests
- Custom logging and request middleware

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
USE_GPU=true
LOG_LEVEL=INFO
ALLOWED_ORIGINS=http://localhost:3000
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

- `GET /health` - Health check
- `POST /recommend` - Get AI outfit recommendations
- `POST /process-tryon` - Virtual try-on processing
- `POST /generate-body` - Generate body models
- `POST /generate-bodies` - Generate multiple body variations
- `POST /analyze-image` - Analyze if image is head-only or full-body
- `POST /combine-head-body` - Combine head with generated body

## GPU Requirements

Backend requires NVIDIA GPU with CUDA support for optimal performance. CPU fallback available but significantly slower.
