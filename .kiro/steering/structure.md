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
│   ├── api/                   # API route handlers
│   │   ├── endpoints.py       # Core endpoints (recommend, try-on)
│   │   ├── body_generation.py # Body model generation
│   │   ├── image_analysis.py  # Image type detection
│   │   └── image_composition.py # Head-body composition
│   ├── core/                  # Core utilities
│   │   ├── logging_config.py  # Logging setup
│   │   ├── middleware.py      # Custom middleware
│   │   └── utils.py           # Helper functions
│   ├── models/                # Data models
│   │   └── schemas.py         # Pydantic schemas
│   └── services/              # Business logic
│       ├── body_detection.py  # Body detection service
│       ├── image_collage.py   # Image collage creation
│       ├── pipeline.py        # Try-on pipeline orchestration
│       └── recommendation.py  # AI recommendation engine
├── ml_engine/                 # ML model management
│   ├── loader.py              # Model loading and caching
│   ├── pipelines/             # ML pipeline implementations
│   │   ├── body_gen.py        # SDXL body generation
│   │   ├── pose.py            # Pose estimation
│   │   ├── reconstruction.py  # 3D reconstruction (PIFuHD)
│   │   ├── segmentation.py    # Image segmentation
│   │   └── tryon.py           # Virtual try-on (IDM-VTON)
│   └── weights/               # Model weights storage
├── main.py                    # FastAPI application entry
├── requirements.txt
└── Dockerfile
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
3. ML Engine pipelines execute (segmentation → pose → try-on)
4. Results saved to data/results
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
