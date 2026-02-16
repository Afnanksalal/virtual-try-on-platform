# Development Guide

This guide covers development workflows, best practices, and common tasks for the Virtual Try-On Platform.

## Development Environment Setup

### Prerequisites
- Node.js 18+ and npm
- Python 3.10.x
- Git
- NVIDIA GPU with CUDA 12.4 (optional but recommended)
- Visual Studio Code (recommended)

### Initial Setup

1. **Clone the repository**
```bash
git clone <repository-url>
cd <project-directory>
```

2. **Clone Leffa at project root**
```bash
git clone https://github.com/franciszzj/Leffa
```

3. **Setup frontend**
```bash
cd frontend
npm install
cp .env.local.example .env.local
# Edit .env.local with your configuration
```

4. **Setup backend**
```bash
cd backend
python -m venv venv
.\venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac

pip install -r requirements.txt
cp .env.example .env
# Edit .env with your configuration
```

## Development Workflow

### Starting Development Servers

**Option 1: Use the development script (Windows)**
```powershell
.\dev_start.ps1
```

**Option 2: Start services manually**

Terminal 1 (Frontend):
```bash
cd frontend
npm run dev
```

Terminal 2 (Backend):
```bash
cd backend
python main.py
```

### Code Style

**Frontend (TypeScript/React)**:
- Use TypeScript for all new files
- Follow React hooks best practices
- Use Tailwind CSS for styling
- Component naming: PascalCase (e.g., `TryOnWidget.tsx`)
- File naming: kebab-case for config, PascalCase for components

**Backend (Python)**:
- Follow PEP 8 style guide
- Use type hints for all functions
- Module naming: snake_case (e.g., `body_generation.py`)
- Class naming: PascalCase (e.g., `TryOnService`)
- Function naming: snake_case (e.g., `process_try_on`)

### Error Handling

Always use the comprehensive error handling system:

```python
from app.core.error_context import error_context
from app.core.performance_metrics import measure_performance
from app.core.oom_handler import with_oom_handling

@error_context("my_module")
@measure_performance("my_operation")
@with_oom_handling(model_loader, max_retries=2)
async def my_function():
    # Your code here
    pass
```

See [Error Handling Guide](../backend/app/core/ERROR_HANDLING_GUIDE.md) for details.

## Common Development Tasks

### Setting Up InstantID (Optional)

InstantID enables identity-preserving body generation. To set it up:

1. **Add HuggingFace token to .env**:
```bash
HUGGINGFACE_TOKEN=your_token_here
```

2. **Download models**:
```powershell
cd backend
.\download_instantid_models.ps1
```

3. **Install dependencies**:
```bash
pip install insightface==0.7.3 albumentations==1.4.23
```

See [InstantID Setup Guide](INSTANTID_SETUP.md) for detailed instructions.

### Adding a New API Endpoint

1. **Create endpoint in appropriate router**
```python
# backend/app/api/endpoints.py
@router.post("/my-endpoint")
async def my_endpoint(
    param: str = Form(...),
    user_id: str = Depends(get_current_user)
):
    # Implementation
    pass
```

2. **Add Pydantic schema if needed**
```python
# backend/app/models/schemas.py
class MyRequest(BaseModel):
    param: str
    
class MyResponse(BaseModel):
    result: str
```

3. **Add service logic**
```python
# backend/app/services/my_service.py
class MyService:
    def process(self, param: str) -> str:
        # Business logic
        pass
```

4. **Update frontend API client**
```typescript
// frontend/src/lib/api.ts
export async function myEndpoint(param: string) {
  const response = await fetch(`${API_URL}/my-endpoint`, {
    method: 'POST',
    body: JSON.stringify({ param }),
  });
  return response.json();
}
```

### Adding a New ML Model

1. **Add model loading logic**
```python
# backend/ml_engine/loader.py
def load_my_model(self):
    if "my_model" not in self.models:
        self.models["my_model"] = MyModel.from_pretrained(...)
    return self.models["my_model"]
```

2. **Create pipeline**
```python
# backend/ml_engine/pipelines/my_pipeline.py
class MyPipeline:
    def __init__(self, model):
        self.model = model
    
    def process(self, input_data):
        # Pipeline logic
        pass
```

3. **Add to service layer**
```python
# backend/app/services/my_service.py
from ml_engine.loader import model_loader

class MyService:
    def process(self, data):
        model = model_loader.load_my_model()
        pipeline = MyPipeline(model)
        return pipeline.process(data)
```

### Database Migrations

1. **Create migration file**
```sql
-- backend/database_migrations/002_my_migration.sql
ALTER TABLE users ADD COLUMN new_field VARCHAR(255);
```

2. **Apply migration**
```bash
# Use Supabase dashboard or CLI
supabase db push
```

### Testing

**Backend Tests**:
```bash
cd backend
pytest tests/
pytest tests/test_specific.py  # Run specific test
pytest -v  # Verbose output
```

**Frontend Tests**:
```bash
cd frontend
npm test
```

### Debugging

**Backend Debugging**:
- Check logs: `backend/logs/app.log`
- Use structured logging:
```python
from app.core.logging_config import get_logger
logger = get_logger("my_module")
logger.info("Debug message", extra={"context": "value"})
```

**Frontend Debugging**:
- Use browser DevTools
- Check Network tab for API calls
- Use React DevTools extension

### Performance Optimization

**GPU Memory Management**:
```python
from ml_engine.loader import model_loader

# Unload unused models
model_loader.unload_model("model_name")

# Clear GPU cache
model_loader.clear_gpu_memory()

# Check memory usage
stats = model_loader.get_memory_usage()
```

**Frontend Performance**:
- Use React.memo for expensive components
- Lazy load images and components
- Use Next.js Image component for optimization

## Git Workflow

1. **Create feature branch**
```bash
git checkout -b feature/my-feature
```

2. **Make changes and commit**
```bash
git add .
git commit -m "feat: add my feature"
```

3. **Push and create PR**
```bash
git push origin feature/my-feature
```

### Commit Message Convention

- `feat:` New feature
- `fix:` Bug fix
- `docs:` Documentation changes
- `style:` Code style changes (formatting)
- `refactor:` Code refactoring
- `test:` Adding tests
- `chore:` Maintenance tasks

## Environment Variables

### Frontend (.env.local)
```
NEXT_PUBLIC_API_URL=http://localhost:8000
NEXT_PUBLIC_SUPABASE_URL=your-supabase-url
NEXT_PUBLIC_SUPABASE_ANON_KEY=your-supabase-anon-key
```

### Backend (.env)
```
# Supabase
SUPABASE_URL=your-supabase-url
SUPABASE_KEY=your-supabase-service-key

# AI Services
GEMINI_API_KEY=your-gemini-api-key
HUGGINGFACE_TOKEN=your-huggingface-token  # For InstantID model downloads

# GPU Settings
USE_GPU=true

# Logging
LOG_LEVEL=INFO  # DEBUG, INFO, WARNING, ERROR

# CORS
ALLOWED_ORIGINS=http://localhost:3000

# Model Settings
WARMUP_MODELS=true
PRELOAD_MODELS=tryon,segmentation
```

## Troubleshooting

### Backend Issues

**Models not loading**:
- Check if Leffa is cloned at project root
- Verify CUDA 12.4 is installed
- Check GPU availability: `python -c "import torch; print(torch.cuda.is_available())"`

**OOM errors**:
- Reduce batch size
- Enable model offloading
- Check GPU memory: `nvidia-smi`

**Import errors**:
- Verify virtual environment is activated
- Reinstall requirements: `pip install -r requirements.txt`

### Frontend Issues

**API connection failed**:
- Verify backend is running
- Check NEXT_PUBLIC_API_URL in .env.local
- Check CORS settings in backend

**Build errors**:
- Clear Next.js cache: `rm -rf .next`
- Reinstall dependencies: `rm -rf node_modules && npm install`

## Resources

- [Product Overview](../.kiro/steering/product.md)
- [Technology Stack](../.kiro/steering/tech.md)
- [Project Structure](../.kiro/steering/structure.md)
- [Error Handling Guide](../backend/app/core/ERROR_HANDLING_GUIDE.md)
- [3D Setup Guide](../backend/3d/SETUP.md)

## Getting Help

1. Check documentation in `.kiro/steering/`
2. Review error logs in `backend/logs/`
3. Check health endpoint: `http://localhost:8000/health`
4. Review verification report: `backend/verification_report.json`

---

**Last Updated**: February 15, 2026
