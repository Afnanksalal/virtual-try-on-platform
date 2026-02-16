# Virtual Try-On Platform

An AI-powered fashion technology application that enables users to virtually try on clothing using advanced machine learning models.

## Features

- **2D Virtual Try-On**: Realistic garment try-on using Leffa (supports upper body, lower body, dresses)
- **Body Generation**: Create synthetic body models with SDXL based on user parameters
- **Identity-Preserving Generation**: Generate full-body images with InstantID that preserve user's facial identity
- **AI Recommendations**: Personalized outfit suggestions using Gemini 2.5 Flash + eBay search
- **3D Reconstruction**: Generate 3D meshes from 2D images using TripoSR
- **Smart Onboarding**: Automatic head-only vs full-body detection
- **Wardrobe Management**: Store and organize garment collections
- **Try-On History**: Track and review previous try-on results

## Tech Stack

### Frontend
- Next.js 16.1.3 (React 19.2.3)
- TypeScript 5
- Tailwind CSS 4
- Three.js for 3D visualization
- Supabase for auth & storage

### Backend
- FastAPI 0.128.5
- PyTorch 2.6.0+cu124
- Leffa (2D virtual try-on)
- TripoSR (3D reconstruction)
- SDXL (body generation)
- InstantID (identity-preserving generation)
- Gemini 2.5 Flash (AI recommendations)
- Supabase (storage & database)

## Quick Start

### Prerequisites
- Node.js 18+ (for frontend)
- Python 3.10.x (for backend)
- NVIDIA GPU with 4GB+ VRAM (recommended)
- CUDA 12.4 (for GPU support)

### Frontend Setup

```bash
cd frontend
npm install
cp .env.local.example .env.local  # Configure environment variables
npm run dev  # Start dev server at http://localhost:3000
```

### Backend Setup

```bash
cd backend
pip install -r requirements.txt
cp .env.example .env  # Configure environment variables

# Clone Leffa repository at project root
cd ..
git clone https://github.com/franciszzj/Leffa

# Start backend
cd backend
python main.py  # Start dev server at http://localhost:8000
```

### Environment Variables

**Frontend (.env.local)**:
```
NEXT_PUBLIC_API_URL=http://localhost:8000
NEXT_PUBLIC_SUPABASE_URL=your-supabase-url
NEXT_PUBLIC_SUPABASE_ANON_KEY=your-supabase-anon-key
```

**Backend (.env)**:
```
SUPABASE_URL=your-supabase-url
SUPABASE_KEY=your-supabase-service-key
GEMINI_API_KEY=your-gemini-api-key
USE_GPU=true
LOG_LEVEL=INFO
ALLOWED_ORIGINS=http://localhost:3000
```

## Project Structure

```
/
├── frontend/          # Next.js application
├── backend/           # FastAPI ML service
├── Leffa/             # Leffa repository (clone here)
├── data/              # Runtime data storage
├── docs/              # Documentation
├── .kiro/             # Kiro configuration & specs
└── README.md          # This file
```

## API Documentation

Once the backend is running, visit:
- API docs: http://localhost:8000/docs
- Health check: http://localhost:8000/health

## GPU Requirements

- **Minimum**: NVIDIA GPU with 4GB VRAM (tested on RTX 3050)
- **Recommended**: 8GB+ VRAM for optimal performance
- **CUDA**: 12.4 (required for PyTorch 2.6.0+cu124)

CPU fallback available but 10-20x slower.

## 3D Reconstruction Setup

For 3D reconstruction features, see detailed setup guide:
- [backend/3d/SETUP.md](backend/3d/SETUP.md)

Includes:
- Visual Studio 2022 Build Tools installation
- CUDA 12.4 Toolkit setup
- torchmcubes compilation from source
- SAM 2.1 and Depth Anything V2 setup

## Documentation

- [Product Overview](.kiro/steering/product.md)
- [Technology Stack](.kiro/steering/tech.md)
- [Project Structure](.kiro/steering/structure.md)
- [API Documentation](docs/API.md)
- [Development Guide](docs/DEVELOPMENT.md)
- [Error Handling Guide](backend/app/core/ERROR_HANDLING_GUIDE.md)
- [3D Setup Guide](backend/3d/SETUP.md)
- [InstantID Setup Guide](docs/INSTANTID_SETUP.md)

## Development

### Start Both Services

```powershell
# Windows PowerShell
.\dev_start.ps1
```

### Run Tests

```bash
cd backend
pytest tests/
```

### Check Environment

```bash
cd backend
python scripts/verify_environment.py
```

## Security

- JWT-based authentication via Supabase
- User data isolation enforced at API level
- File upload validation (max 10MB, images only)
- CORS protection with configurable origins

## Performance

- GPU memory optimization for 4GB VRAM
- Model caching with singleton pattern
- Background temp file cleanup (15 min intervals)
- Automatic OOM detection and recovery
- Performance metrics tracking

## License

[Add your license here]

## Support

For issues and questions:
1. Check the documentation in `.kiro/steering/`
2. Review error handling guide: `backend/app/core/ERROR_HANDLING_GUIDE.md`
3. Check 3D setup guide: `backend/3d/SETUP.md`

---

**Last Updated**: February 15, 2026
