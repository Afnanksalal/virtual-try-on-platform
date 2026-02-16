#!/bin/bash
# =============================================================================
# Virtual Try-On Platform - A100 Linux Setup Script
# =============================================================================
# 
# This script sets up the complete environment for the Virtual Try-On Platform
# on an A100 GPU server running Linux with Python 3.10 and CUDA 12.4
#
# Requirements:
# - Ubuntu 20.04+ or similar Linux distribution
# - Python 3.10.x
# - CUDA 12.4 Toolkit installed
# - NVIDIA Driver 550+ (for CUDA 12.4)
# - Git
# - 100GB+ free disk space (for models)
# - Internet connection (for downloading models)
#
# Usage:
#   chmod +x setup_a100.sh
#   ./setup_a100.sh
#
# =============================================================================

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# =============================================================================
# Step 1: System Requirements Check
# =============================================================================
log_info "Checking system requirements..."

# Check Python version
if ! command -v python3.10 &> /dev/null; then
    log_error "Python 3.10 not found. Please install Python 3.10 first."
    exit 1
fi

PYTHON_VERSION=$(python3.10 --version | cut -d' ' -f2)
log_success "Python version: $PYTHON_VERSION"

# Check CUDA
if ! command -v nvcc &> /dev/null; then
    log_error "CUDA toolkit not found. Please install CUDA 12.4 first."
    exit 1
fi

CUDA_VERSION=$(nvcc --version | grep "release" | awk '{print $5}' | cut -d',' -f1)
log_success "CUDA version: $CUDA_VERSION"

# Check NVIDIA driver
if ! command -v nvidia-smi &> /dev/null; then
    log_error "NVIDIA driver not found. Please install NVIDIA driver 550+ first."
    exit 1
fi

DRIVER_VERSION=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -n1)
log_success "NVIDIA driver version: $DRIVER_VERSION"

# Check GPU
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n1)
log_success "GPU detected: $GPU_NAME"

# Check Git
if ! command -v git &> /dev/null; then
    log_error "Git not found. Please install git first."
    exit 1
fi

# Check disk space
AVAILABLE_SPACE=$(df -BG . | tail -1 | awk '{print $4}' | sed 's/G//')
if [ "$AVAILABLE_SPACE" -lt 100 ]; then
    log_warning "Less than 100GB free disk space. You have ${AVAILABLE_SPACE}GB. Models require ~80GB."
fi

log_success "System requirements check passed!"

# =============================================================================
# Step 2: Create Virtual Environment
# =============================================================================
log_info "Creating Python virtual environment..."

if [ -d "venv" ]; then
    log_warning "Virtual environment already exists. Skipping creation."
else
    python3.10 -m venv venv
    log_success "Virtual environment created"
fi

# Activate virtual environment
source venv/bin/activate
log_success "Virtual environment activated"

# Upgrade pip
log_info "Upgrading pip..."
pip install --upgrade pip setuptools wheel
log_success "Pip upgraded"

# =============================================================================
# Step 3: Install System Dependencies
# =============================================================================
log_info "Installing system dependencies..."

# Check if running as root or with sudo
if [ "$EUID" -ne 0 ]; then
    log_warning "Not running as root. Some system packages may require sudo."
    SUDO="sudo"
else
    SUDO=""
fi

# Install build essentials
$SUDO apt-get update
$SUDO apt-get install -y \
    build-essential \
    cmake \
    git \
    wget \
    curl \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    ninja-build

log_success "System dependencies installed"

# =============================================================================
# Step 4: Install PyTorch with CUDA 12.4
# =============================================================================
log_info "Installing PyTorch 2.6.0 with CUDA 12.4..."

pip install torch==2.6.0+cu124 torchvision==0.21.0+cu124 torchaudio==2.6.0+cu124 \
    --index-url https://download.pytorch.org/whl/cu124

log_success "PyTorch installed"

# Verify PyTorch installation
python3.10 -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'CUDA Version: {torch.version.cuda}')"

# =============================================================================
# Step 5: Install Main Requirements
# =============================================================================
log_info "Installing main Python requirements..."

cd backend
pip install -r requirements.txt
cd ..

log_success "Main requirements installed"

# =============================================================================
# Step 6: Install Git-based Packages
# =============================================================================
log_info "Installing SAM-2 from GitHub..."

pip install git+https://github.com/facebookresearch/segment-anything-2.git@2b90b9f5ceec907a1c18123530e92e794ad901a4

log_success "SAM-2 installed"

log_info "Installing torchmcubes from GitHub..."

# torchmcubes requires compilation
pip install git+https://github.com/tatsy/torchmcubes.git@3381600ddc3d2e4d74222f8495866be5fafbace4

log_success "torchmcubes installed"

# =============================================================================
# Step 7: Clone Leffa Repository
# =============================================================================
log_info "Cloning Leffa repository..."

if [ -d "Leffa" ]; then
    log_warning "Leffa directory already exists. Skipping clone."
else
    git clone https://github.com/franciszzj/Leffa.git
    log_success "Leffa repository cloned"
fi

# =============================================================================
# Step 8: Clone TripoSR Repository
# =============================================================================
log_info "Setting up TripoSR..."

if [ -d "backend/3d/TripoSR" ]; then
    log_warning "TripoSR directory already exists. Skipping clone."
else
    cd backend/3d
    git clone https://github.com/VAST-AI-Research/TripoSR.git
    cd ../..
    log_success "TripoSR repository cloned"
fi

# =============================================================================
# Step 9: Create Required Directories
# =============================================================================
log_info "Creating required directories..."

mkdir -p data/uploads
mkdir -p data/results
mkdir -p data/temp
mkdir -p backend/data
mkdir -p backend/logs
mkdir -p backend/3d/models
mkdir -p backend/3d/outputs
mkdir -p backend/ml_engine/weights
mkdir -p Leffa/ckpts

# Create .gitkeep files
touch data/uploads/.gitkeep
touch data/results/.gitkeep
touch data/temp/.gitkeep
touch backend/data/.gitkeep
touch backend/logs/.gitkeep
touch backend/3d/models/.gitkeep
touch backend/3d/outputs/.gitkeep
touch backend/ml_engine/weights/.gitkeep
touch Leffa/ckpts/.gitkeep

log_success "Directories created"

# =============================================================================
# Step 10: Download Model Weights
# =============================================================================
log_info "Downloading model weights (this may take a while - ~80GB)..."

# Create Python script to download models
cat > download_models.py << 'EOF'
#!/usr/bin/env python3
"""
Download all required model weights for the Virtual Try-On Platform
"""
import os
import sys
from pathlib import Path

def download_sam2_model():
    """Download SAM 2.1 Large model"""
    print("Downloading SAM 2.1 Large model...")
    from huggingface_hub import hf_hub_download
    
    model_path = hf_hub_download(
        repo_id="facebook/sam2.1-hiera-large",
        filename="sam2.1_hiera_large.pt",
        local_dir="backend/3d/models"
    )
    print(f"✓ SAM 2.1 model downloaded to: {model_path}")

def download_triposr_model():
    """Download TripoSR model"""
    print("Downloading TripoSR model...")
    from huggingface_hub import hf_hub_download
    
    model_path = hf_hub_download(
        repo_id="stabilityai/TripoSR",
        filename="model.safetensors",
        local_dir="backend/3d/TripoSR/ckpts"
    )
    print(f"✓ TripoSR model downloaded to: {model_path}")

def download_leffa_models():
    """Download Leffa checkpoints"""
    print("Downloading Leffa checkpoints...")
    from huggingface_hub import snapshot_download
    
    snapshot_download(
        repo_id="franciszzj/Leffa",
        local_dir="Leffa/ckpts",
        allow_patterns=["*.pth", "*.safetensors", "*.bin"]
    )
    print("✓ Leffa checkpoints downloaded")

def download_depth_anything_v2():
    """Download Depth Anything V2 model"""
    print("Downloading Depth Anything V2 Large model...")
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation
    
    # This will download and cache the model
    processor = AutoImageProcessor.from_pretrained("depth-anything/Depth-Anything-V2-Large-hf")
    model = AutoModelForDepthEstimation.from_pretrained("depth-anything/Depth-Anything-V2-Large-hf")
    print("✓ Depth Anything V2 model downloaded and cached")

def download_sdxl_models():
    """Download SDXL models"""
    print("Downloading SDXL models...")
    from diffusers import StableDiffusionXLPipeline
    
    # This will download and cache the model
    pipe = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=None  # Will download FP32
    )
    print("✓ SDXL models downloaded and cached")

def download_instantid_models():
    """Download InstantID models"""
    print("Downloading InstantID models...")
    from huggingface_hub import hf_hub_download
    
    # InstantID main model
    hf_hub_download(
        repo_id="InstantX/InstantID",
        filename="ip-adapter.bin",
        local_dir="backend/ml_engine/weights/InstantID"
    )
    
    # ControlNet
    hf_hub_download(
        repo_id="InstantX/InstantID",
        filename="ControlNetModel/config.json",
        local_dir="backend/ml_engine/weights/ControlNet"
    )
    hf_hub_download(
        repo_id="InstantX/InstantID",
        filename="ControlNetModel/diffusion_pytorch_model.safetensors",
        local_dir="backend/ml_engine/weights/ControlNet"
    )
    
    print("✓ InstantID models downloaded")

def download_insightface_models():
    """Download InsightFace antelopev2 models"""
    print("Downloading InsightFace antelopev2 models...")
    import gdown
    
    # Download antelopev2 models
    os.makedirs("backend/ml_engine/weights/antelopev2", exist_ok=True)
    
    # Note: These need to be downloaded from InsightFace's official source
    print("⚠ InsightFace models need to be downloaded manually from:")
    print("  https://github.com/deepinsight/insightface/tree/master/python-package")
    print("  Place them in: backend/ml_engine/weights/antelopev2/")

if __name__ == "__main__":
    print("=" * 80)
    print("Downloading Model Weights")
    print("=" * 80)
    
    try:
        # Download models
        download_sam2_model()
        download_triposr_model()
        download_leffa_models()
        download_depth_anything_v2()
        download_sdxl_models()
        download_instantid_models()
        download_insightface_models()
        
        print("\n" + "=" * 80)
        print("✓ All models downloaded successfully!")
        print("=" * 80)
        
    except Exception as e:
        print(f"\n✗ Error downloading models: {e}")
        sys.exit(1)
EOF

# Run the download script
python3.10 download_models.py

log_success "Model weights downloaded"

# Clean up download script
rm download_models.py

# =============================================================================
# Step 11: Setup Environment Variables
# =============================================================================
log_info "Setting up environment variables..."

if [ ! -f "backend/.env" ]; then
    if [ -f "backend/.env.example" ]; then
        cp backend/.env.example backend/.env
        log_warning "Created backend/.env from .env.example. Please update with your API keys!"
    else
        log_warning "No .env.example found. Please create backend/.env manually."
    fi
else
    log_warning "backend/.env already exists. Skipping."
fi

if [ ! -f "frontend/.env.local" ]; then
    cat > frontend/.env.local << 'EOF'
NEXT_PUBLIC_API_URL=http://localhost:8000
NEXT_PUBLIC_SUPABASE_URL=your-supabase-url
NEXT_PUBLIC_SUPABASE_ANON_KEY=your-supabase-anon-key
EOF
    log_warning "Created frontend/.env.local. Please update with your Supabase credentials!"
else
    log_warning "frontend/.env.local already exists. Skipping."
fi

log_success "Environment files created"

# =============================================================================
# Step 12: Install Frontend Dependencies
# =============================================================================
log_info "Installing frontend dependencies..."

cd frontend

if command -v npm &> /dev/null; then
    npm install
    log_success "Frontend dependencies installed"
else
    log_warning "npm not found. Please install Node.js and run 'npm install' in the frontend directory."
fi

cd ..

# =============================================================================
# Step 13: Verify Installation
# =============================================================================
log_info "Verifying installation..."

python3.10 << 'EOF'
import sys
import torch

print("\n" + "=" * 80)
print("Installation Verification")
print("=" * 80)

# Check Python version
print(f"Python: {sys.version}")

# Check PyTorch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA Version: {torch.version.cuda}")
    print(f"GPU Count: {torch.cuda.device_count()}")
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

# Check key packages
packages = [
    'transformers',
    'diffusers',
    'accelerate',
    'fastapi',
    'uvicorn',
    'opencv-python',
    'trimesh',
    'open3d',
    'segment-anything',
    'insightface',
    'supabase',
]

print("\nPackage Versions:")
for pkg in packages:
    try:
        mod = __import__(pkg.replace('-', '_'))
        version = getattr(mod, '__version__', 'unknown')
        print(f"  {pkg}: {version}")
    except ImportError:
        print(f"  {pkg}: NOT INSTALLED")

print("=" * 80)
EOF

log_success "Installation verification complete"

# =============================================================================
# Step 14: Create Startup Scripts
# =============================================================================
log_info "Creating startup scripts..."

# Backend startup script
cat > start_backend.sh << 'EOF'
#!/bin/bash
# Start the FastAPI backend server

source venv/bin/activate
cd backend
python main.py
EOF

chmod +x start_backend.sh

# Frontend startup script
cat > start_frontend.sh << 'EOF'
#!/bin/bash
# Start the Next.js frontend server

cd frontend
npm run dev
EOF

chmod +x start_frontend.sh

# Combined startup script
cat > start_all.sh << 'EOF'
#!/bin/bash
# Start both backend and frontend servers

echo "Starting Virtual Try-On Platform..."

# Start backend in background
./start_backend.sh &
BACKEND_PID=$!

# Wait for backend to start
sleep 5

# Start frontend in background
./start_frontend.sh &
FRONTEND_PID=$!

echo "Backend PID: $BACKEND_PID"
echo "Frontend PID: $FRONTEND_PID"
echo ""
echo "Backend running at: http://localhost:8000"
echo "Frontend running at: http://localhost:3000"
echo ""
echo "Press Ctrl+C to stop both servers"

# Wait for Ctrl+C
trap "kill $BACKEND_PID $FRONTEND_PID; exit" INT
wait
EOF

chmod +x start_all.sh

log_success "Startup scripts created"

# =============================================================================
# Setup Complete
# =============================================================================
echo ""
echo "=" * 80
log_success "Setup Complete!"
echo "=" * 80
echo ""
echo "Next steps:"
echo "  1. Update backend/.env with your API keys (Gemini, Supabase, HuggingFace)"
echo "  2. Update frontend/.env.local with your Supabase credentials"
echo "  3. Start the backend: ./start_backend.sh"
echo "  4. Start the frontend: ./start_frontend.sh"
echo "  5. Or start both: ./start_all.sh"
echo ""
echo "Backend will be available at: http://localhost:8000"
echo "Frontend will be available at: http://localhost:3000"
echo "API docs will be available at: http://localhost:8000/docs"
echo ""
echo "For production deployment, see docs/DEVELOPMENT.md"
echo ""
echo "=" * 80
