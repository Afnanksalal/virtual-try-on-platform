# Lightning.ai Setup Guide

## Prerequisites
- Lightning.ai Studio with GPU
- Supabase account (for authentication)
- Gemini API key (for AI recommendations)
- Python 3.10 installed via pyenv

## Step 1: Environment Setup

### Backend Environment Variables

Create `backend/.env`:

```bash
cd ~/virtual-try-on-platform/backend
cat > .env << 'EOF'
# Server Configuration
PORT=8000
LOG_LEVEL=INFO
ALLOWED_ORIGINS=*

# Redis Configuration (optional for now)
REDIS_URL=redis://localhost:6379/0

# AI/ML API Keys
GEMINI_API_KEY=your_actual_gemini_api_key_here
HUGGINGFACE_TOKEN=your_huggingface_token_here

# RapidAPI for eBay Product Search (optional)
RAPIDAPI_KEY=your_rapidapi_key_here
RAPIDAPI_HOST=ebay-search-result.p.rapidapi.com

# GPU Configuration
USE_GPU=true

# Model Configuration
PRELOAD_MODELS=tryon,segmentation
WARMUP_MODELS=true
EOF
```

**Important**: Replace `your_actual_gemini_api_key_here` with your real Gemini API key!

### Frontend Environment Variables

Create `frontend/.env.local`:

```bash
cd ~/virtual-try-on-platform/frontend
cat > .env.local << 'EOF'
# Supabase Configuration
NEXT_PUBLIC_SUPABASE_URL=https://your-project.supabase.co
NEXT_PUBLIC_SUPABASE_ANON_KEY=your_supabase_anon_key_here

# Backend API Configuration
# Lightning.ai exposes services on specific ports
NEXT_PUBLIC_API_URL=http://localhost:8000
EOF
```

**Important**: Replace Supabase values with your actual project credentials!

## Step 2: Install Dependencies

### Backend

```bash
cd ~/virtual-try-on-platform/backend

# Ensure Python 3.10 is active
pyenv local 3.10.13
python --version  # Should show Python 3.10.13

# Upgrade pip
pip install --upgrade pip

# Install dependencies
pip install -r requirements.txt
```

### Frontend

```bash
cd ~/virtual-try-on-platform/frontend

# Install Node dependencies
npm install
```

## Step 3: Create Required Directories

```bash
cd ~/virtual-try-on-platform

# Create data directories
mkdir -p data/uploads data/results data/users
mkdir -p backend/logs

# Create .gitkeep files
touch data/uploads/.gitkeep
touch data/results/.gitkeep
touch data/users/.gitkeep
```

## Step 4: Lightning.ai Port Configuration

Lightning.ai exposes services through specific ports. You need to:

1. **Backend (Port 8000)**: FastAPI will run on port 8000
2. **Frontend (Port 3000)**: Next.js will run on port 3000

### Update CORS for Lightning.ai

Since Lightning.ai uses a proxy, update backend CORS:

```bash
# In backend/.env, set:
ALLOWED_ORIGINS=*
```

**Note**: In production, replace `*` with your actual Lightning.ai domain.

## Step 5: Start Services

### Option A: Start Both Services (Recommended)

Open **two terminals** in Lightning.ai:

**Terminal 1 - Backend:**
```bash
cd ~/virtual-try-on-platform/backend
pyenv local 3.10.13
python main.py
```

**Terminal 2 - Frontend:**
```bash
cd ~/virtual-try-on-platform/frontend
npm run dev
```

### Option B: Use Background Processes

**Start Backend:**
```bash
cd ~/virtual-try-on-platform/backend
pyenv local 3.10.13
nohup python main.py > logs/backend.log 2>&1 &
```

**Start Frontend:**
```bash
cd ~/virtual-try-on-platform/frontend
nohup npm run dev > ../backend/logs/frontend.log 2>&1 &
```

**View logs:**
```bash
# Backend logs
tail -f ~/virtual-try-on-platform/backend/logs/backend.log

# Frontend logs
tail -f ~/virtual-try-on-platform/backend/logs/frontend.log
```

## Step 6: Access Your Application

Lightning.ai will provide URLs for your services:

- **Frontend**: Click the "Open" button for port 3000
- **Backend API**: Click the "Open" button for port 8000
- **API Docs**: `<backend-url>/docs` (Swagger UI)

## Step 7: Verify Setup

### Check Backend Health

```bash
curl http://localhost:8000/health
```

Expected response:
```json
{
  "status": "healthy",
  "service": "ml-api",
  "checks": {
    "compute": {
      "gpu_available": true,
      "device": "Tesla T4"
    },
    "ai_service": {
      "configured": true
    }
  }
}
```

### Check Frontend

Open the frontend URL in your browser. You should see the landing page.

## Troubleshooting

### Backend Issues

**GPU not detected:**
```bash
python -c "import torch; print(torch.cuda.is_available())"
```

**Port already in use:**
```bash
# Find process using port 8000
lsof -i :8000
# Kill it
kill -9 <PID>
```

**Missing dependencies:**
```bash
cd ~/virtual-try-on-platform/backend
pip install -r requirements.txt --force-reinstall
```

### Frontend Issues

**Port 3000 in use:**
```bash
# Kill process on port 3000
lsof -i :3000
kill -9 <PID>
```

**Module not found:**
```bash
cd ~/virtual-try-on-platform/frontend
rm -rf node_modules package-lock.json
npm install
```

**CORS errors:**
- Ensure `ALLOWED_ORIGINS=*` in `backend/.env`
- Restart backend after changing .env

### Model Loading Issues

**Models not downloading:**
```bash
# Set Hugging Face token
export HUGGINGFACE_TOKEN=your_token_here

# Manually download models
cd ~/virtual-try-on-platform/backend/ml_engine/weights/idm-vton
python download_weights.py
```

## Quick Start Commands

```bash
# Full setup from scratch
cd ~/virtual-try-on-platform

# Backend
cd backend
pyenv local 3.10.13
pip install -r requirements.txt
cp .env.example .env
# Edit .env with your API keys
python main.py

# Frontend (in new terminal)
cd ~/virtual-try-on-platform/frontend
npm install
cp .env.example .env.local
# Edit .env.local with your Supabase credentials
npm run dev
```

## Environment Variables Checklist

### Required for Backend:
- ✅ `GEMINI_API_KEY` - Get from Google AI Studio
- ✅ `ALLOWED_ORIGINS` - Set to `*` for Lightning.ai
- ✅ `USE_GPU` - Set to `true`

### Required for Frontend:
- ✅ `NEXT_PUBLIC_SUPABASE_URL` - From Supabase dashboard
- ✅ `NEXT_PUBLIC_SUPABASE_ANON_KEY` - From Supabase dashboard
- ✅ `NEXT_PUBLIC_API_URL` - Set to `http://localhost:8000`

### Optional:
- `HUGGINGFACE_TOKEN` - For downloading models
- `RAPIDAPI_KEY` - For eBay product search
- `REDIS_URL` - For caching (not required initially)

## Next Steps

1. Get your API keys:
   - Gemini: https://makersuite.google.com/app/apikey
   - Supabase: https://supabase.com/dashboard
   - Hugging Face: https://huggingface.co/settings/tokens

2. Update `.env` files with real credentials

3. Start both services

4. Test the application by uploading an image

5. Check logs for any errors

## Production Considerations

When deploying to production:

1. **Replace CORS wildcard:**
   ```bash
   ALLOWED_ORIGINS=https://your-frontend-domain.com
   ```

2. **Use environment-specific configs:**
   - Separate `.env.production` files
   - Use secrets management

3. **Enable HTTPS:**
   - Configure SSL certificates
   - Update all URLs to use `https://`

4. **Set up monitoring:**
   - Log aggregation
   - Error tracking
   - Performance monitoring
