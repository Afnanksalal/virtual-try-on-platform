# Project Structure

## Root Layout

```
/
├── frontend/          # Next.js application
├── backend/           # FastAPI ML service
├── data/              # Runtime data storage
├── .kiro/             # Kiro configuration
├── docker-compose.yml # Container orchestration
└── dev_start.ps1      # Development startup script
```

## Frontend Structure

```
frontend/
├── src/
│   ├── app/                    # Next.js App Router pages
│   │   ├── auth/              # Authentication page
│   │   ├── onboard/           # User onboarding flow
│   │   ├── studio/            # Virtual try-on studio
│   │   ├── wardrobe/          # User wardrobe management
│   │   ├── shop/              # Shopping/recommendations
│   │   ├── layout.tsx         # Root layout with Navbar
│   │   ├── page.tsx           # Landing page
│   │   └── globals.css        # Global styles
│   ├── components/            # React components
│   │   ├── ErrorBoundary.tsx  # Error handling wrapper
│   │   ├── Hero.tsx           # Landing page hero
│   │   ├── ModelViewer.tsx    # 3D model viewer (Three.js)
│   │   ├── Navbar.tsx         # Navigation bar
│   │   ├── ProtectedRoute.tsx # Auth guard
│   │   ├── Recommendations.tsx # AI recommendations widget
│   │   ├── TryOnWidget.tsx    # Try-on upload interface
│   │   └── UploadWidget.tsx   # Generic file upload
│   └── lib/                   # Utilities and services
│       ├── api.ts             # Backend API client
│       ├── supabase.ts        # Supabase client setup
│       └── types.ts           # TypeScript type definitions
├── public/                    # Static assets
├── package.json
└── next.config.ts
```

## Backend Structure

```
backend/
├── app/
│   ├── api/                       # API route handlers
│   │   ├── endpoints.py           # Core endpoints (recommend, try-on, batch)
│   │   ├── body_generation.py     # Body model generation (SDXL)
│   │   ├── image_analysis.py      # Image type detection
│   │   ├── image_composition.py   # Head-body composition
│   │   ├── garment_management.py  # Wardrobe CRUD operations
│   │   ├── identity_body.py       # Identity-preserving body generation
│   │   └── reconstruction_3d.py   # 3D mesh reconstruction
│   ├── core/                      # Core utilities
│   │   ├── auth.py                # JWT authentication (Supabase)
│   │   ├── cache_manager.py       # Model cache management
│   │   ├── error_context.py       # Error logging with context
│   │   ├── error_handler.py       # Centralized error handling
│   │   ├── exceptions.py          # Custom exception classes
│   │   ├── file_validator.py      # File upload validation
│   │   ├── logging_config.py      # Structured logging setup
│   │   ├── middleware.py          # Request logging middleware
│   │   ├── oom_handler.py         # GPU OOM detection & recovery
│   │   ├── performance_metrics.py # Performance tracking
│   │   ├── utils.py               # Helper functions
│   │   └── ERROR_HANDLING_GUIDE.md # Error handling documentation
│   ├── models/                    # Data models
│   │   └── schemas.py             # Pydantic schemas
│   └── services/                  # Business logic
│       ├── body_detection.py      # Body type detection
│       ├── body_generation.py     # Body generation service
│       ├── face_analysis.py       # Face detection & analysis
│       ├── image_collage.py       # Image collage creation
│       ├── recommendation.py      # AI recommendation engine (Gemini)
│       ├── supabase_storage.py    # Supabase storage & DB operations
│       ├── temp_file_manager.py   # Temporary file cleanup
│       └── tryon_service.py       # Try-on orchestration (Leffa)
├── ml_engine/                     # ML model management
│   ├── loader.py                  # Model loading & caching (singleton)
│   └── pipelines/                 # ML pipeline implementations
│       ├── body_gen.py            # SDXL body generation
│       ├── leffa_tryon.py         # Leffa virtual try-on
│       ├── reconstruction_3d.py   # 3D reconstruction (TripoSR)
│       └── segmentation.py        # SAM 2.1 segmentation
├── 3d/                            # 3D reconstruction components
│   ├── SETUP.md                   # 3D setup guide (CUDA, torchmcubes)
│   ├── TripoSR/                   # TripoSR repository (submodule)
│   ├── models/                    # SAM 2.1 model weights
│   └── outputs/                   # Generated 3D meshes
├── database_migrations/           # SQL migration scripts
│   └── 001_add_metadata_fields.sql
├── scripts/                       # Utility scripts
│   └── verify_environment.py      # Environment verification
├── tests/                         # Test suite
├── logs/                          # Application logs
├── main.py                        # FastAPI application entry
├── requirements.txt               # Python dependencies
├── .env                           # Environment variables
├── .env.example                   # Environment template
├── database_schema.sql            # Database schema
├── database_schema_complete.sql   # Complete schema with all tables
└── verification_report.json       # Environment verification report
```

## Data Directory

```
data/
├── results/    # Generated try-on results
├── uploads/    # User uploaded images
└── users/      # User-specific data
```

## Architecture Patterns

### Frontend
- **Component Organization**: Separate components by feature (widgets, pages, utilities)
- **API Layer**: Centralized API client in `lib/api.ts` with typed responses
- **Error Handling**: ErrorBoundary components wrap feature sections
- **Auth**: ProtectedRoute HOC guards authenticated pages
- **Styling**: Utility-first with Tailwind, component-scoped styles

### Backend
- **Layered Architecture**:
  - API layer (routes/endpoints)
  - Service layer (business logic)
  - ML Engine layer (model inference)
- **Singleton Pattern**: ML models loaded once via ModelLoader
- **Async Processing**: Heavy ML operations offloaded to thread pool
- **Middleware**: Request logging, CORS, error handling
- **Logging**: Structured logging with configurable levels

### ML Pipeline Flow
1. User uploads images → API endpoint
2. Service layer validates and preprocesses
3. ML Engine pipelines execute (CatVTON with auto-masking)
4. Results saved to Supabase storage
5. Response returned with result URL

## Naming Conventions

- **Files**: snake_case for Python, kebab-case for config, PascalCase for React components
- **Components**: PascalCase (e.g., `TryOnWidget.tsx`)
- **API Routes**: kebab-case endpoints (e.g., `/process-tryon`)
- **Python Modules**: snake_case (e.g., `body_generation.py`)
- **Classes**: PascalCase (e.g., `TryOnService`)
- **Functions**: snake_case (e.g., `process_try_on`)

## Key Design Decisions

- **Monorepo**: Frontend and backend in same repository for easier development
- **Type Safety**: TypeScript frontend, Pydantic backend for end-to-end type safety
- **GPU Optimization**: Models use xformers for memory-efficient attention
- **Async/Await**: Non-blocking I/O throughout the stack
- **Error Boundaries**: Graceful degradation when features fail
- **Environment-based Config**: Different settings for dev/prod via env vars
