#!/bin/bash
# =============================================================================
# InstantID Setup Script for Virtual Try-On Platform
# =============================================================================
# 
# This script downloads and configures InstantID models for identity-preserving
# body generation. It downloads:
#   - InstantID ControlNet model
#   - InstantID IP-Adapter weights  
#   - antelopev2 face analysis model (InsightFace)
#
# Usage:
#   ./scripts/setup_instantid.sh [OPTIONS]
#
# Options:
#   --force           Re-download even if files exist
#   --skip-face       Skip antelopev2 face model download
#   --help            Show this help message
#
# Requirements:
#   - Python 3.10+ with pip
#   - Virtual environment activated
#   - curl, unzip
#
# =============================================================================

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
MAGENTA='\033[0;35m'
NC='\033[0m' # No Color

# Logging functions
log_success() { echo -e "${GREEN}[OK]${NC} $1"; }
log_info() { echo -e "${CYAN}[INFO]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# Parse arguments
FORCE=false
SKIP_FACE=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --force)
            FORCE=true
            shift
            ;;
        --skip-face)
            SKIP_FACE=true
            shift
            ;;
        --help)
            head -n 25 "$0" | tail -n 22
            exit 0
            ;;
        *)
            log_error "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo ""
echo -e "${MAGENTA}============================================${NC}"
echo -e "${MAGENTA}  InstantID Setup for Virtual Try-On${NC}"
echo -e "${MAGENTA}============================================${NC}"
echo ""

# Determine paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND_DIR="$(dirname "$SCRIPT_DIR")"
WEIGHTS_DIR="$BACKEND_DIR/ml_engine/weights/instantid"
MODELS_DIR="$WEIGHTS_DIR/models"
ANTELOPE_DIR="$MODELS_DIR/antelopev2"
CHECKPOINTS_DIR="$WEIGHTS_DIR/checkpoints"

log_info "Backend directory: $BACKEND_DIR"
log_info "Weights directory: $WEIGHTS_DIR"

# Create directories
log_info "Creating directories..."
mkdir -p "$WEIGHTS_DIR"
mkdir -p "$MODELS_DIR"
mkdir -p "$ANTELOPE_DIR"
mkdir -p "$CHECKPOINTS_DIR"
log_success "Directories created"

# Check Python environment
log_info "Checking Python environment..."
if ! command -v python &> /dev/null; then
    log_error "Python not found. Please activate your virtual environment first."
    exit 1
fi

PYTHON_VERSION=$(python --version 2>&1)
log_success "Python found: $PYTHON_VERSION"

# Check if in virtual environment
if [[ -z "$VIRTUAL_ENV" ]]; then
    log_warn "Virtual environment not detected. Make sure dependencies install to the right location."
fi

# Install required packages from requirements.txt for version pinning
log_info "Installing/updating required Python packages from requirements.txt..."
pip install --quiet --upgrade -r "$BACKEND_DIR/requirements.txt" || {
    log_warn "Some packages may have failed to install. Continuing..."
}
log_success "Python packages ready"

# Download InstantID models using Python
log_info "Downloading InstantID models from HuggingFace..."

python << EOF
import os
import sys
from huggingface_hub import hf_hub_download

checkpoints_dir = '$CHECKPOINTS_DIR'
os.makedirs(checkpoints_dir, exist_ok=True)

print('Downloading InstantID ControlNet config...')
hf_hub_download(
    repo_id='InstantX/InstantID',
    filename='ControlNetModel/config.json',
    local_dir=checkpoints_dir  
)

print('Downloading InstantID ControlNet model...')
hf_hub_download(
    repo_id='InstantX/InstantID',
    filename='ControlNetModel/diffusion_pytorch_model.safetensors',
    local_dir=checkpoints_dir
)

print('Downloading InstantID IP-Adapter...')
hf_hub_download(
    repo_id='InstantX/InstantID',
    filename='ip-adapter.bin',
    local_dir=checkpoints_dir
)

print('InstantID models downloaded successfully!')
EOF

if [ $? -eq 0 ]; then
    log_success "InstantID models downloaded"
else
    log_error "Failed to download InstantID models"
    exit 1
fi

# Download antelopev2 face model
if [ "$SKIP_FACE" = false ]; then
    log_info "Downloading antelopev2 face analysis model..."
    
    ANTELOPE_URL="https://huggingface.co/MonsterMMORPG/tools/resolve/main/antelopev2.zip"
    ANTELOPE_ZIP="/tmp/antelopev2.zip"
    
    # Check if already downloaded
    if [ -f "$ANTELOPE_DIR/1k3d68.onnx" ] && [ "$FORCE" = false ]; then
        log_info "antelopev2 already exists. Use --force to re-download."
    else
        log_info "Downloading from: $ANTELOPE_URL"
        log_info "This may take a few minutes..."
        
        # Download using curl
        if curl -L -o "$ANTELOPE_ZIP" "$ANTELOPE_URL" --progress-bar; then
            log_info "Extracting antelopev2..."
            unzip -o -q "$ANTELOPE_ZIP" -d "$MODELS_DIR"
            rm -f "$ANTELOPE_ZIP"
            log_success "antelopev2 model downloaded and extracted"
        else
            log_error "Failed to download antelopev2"
            log_warn "You may need to download manually from:"
            log_warn "  https://github.com/deepinsight/insightface/issues/1896#issuecomment-1023867304"
            log_warn "Extract to: $ANTELOPE_DIR"
        fi
    fi
else
    log_warn "Skipping face model download (remove --skip-face to download)"
fi

# Verify installation
echo ""
log_info "Verifying installation..."

ALL_GOOD=true

# Check InstantID files
CONTROLNET_CONFIG="$CHECKPOINTS_DIR/ControlNetModel/config.json"
CONTROLNET_MODEL="$CHECKPOINTS_DIR/ControlNetModel/diffusion_pytorch_model.safetensors"
IP_ADAPTER="$CHECKPOINTS_DIR/ip-adapter.bin"

if [ -f "$CONTROLNET_CONFIG" ]; then
    log_success "ControlNet config found"
else
    log_error "ControlNet config missing: $CONTROLNET_CONFIG"
    ALL_GOOD=false
fi

if [ -f "$CONTROLNET_MODEL" ]; then
    SIZE=$(du -h "$CONTROLNET_MODEL" | cut -f1)
    log_success "ControlNet model found ($SIZE)"
else
    log_error "ControlNet model missing: $CONTROLNET_MODEL"
    ALL_GOOD=false
fi

if [ -f "$IP_ADAPTER" ]; then
    SIZE=$(du -h "$IP_ADAPTER" | cut -f1)
    log_success "IP-Adapter found ($SIZE)"
else
    log_error "IP-Adapter missing: $IP_ADAPTER"
    ALL_GOOD=false
fi

# Check antelopev2
ANTELOPE_FILES=(
    "1k3d68.onnx"
    "2d106det.onnx"
    "genderage.onnx"
    "glintr100.onnx"
    "scrfd_10g_bnkps.onnx"
)

ANTELOPE_FOUND=0
for file in "${ANTELOPE_FILES[@]}"; do
    if [ -f "$ANTELOPE_DIR/$file" ]; then
        ((ANTELOPE_FOUND++))
    fi
done

TOTAL_ANTELOPE=${#ANTELOPE_FILES[@]}
if [ $ANTELOPE_FOUND -eq $TOTAL_ANTELOPE ]; then
    log_success "antelopev2 model complete ($ANTELOPE_FOUND/$TOTAL_ANTELOPE files)"
elif [ $ANTELOPE_FOUND -gt 0 ]; then
    log_warn "antelopev2 model incomplete ($ANTELOPE_FOUND/$TOTAL_ANTELOPE files)"
    ALL_GOOD=false
else
    log_warn "antelopev2 model not found"
    log_warn "Identity-preserving generation will fall back to SDXL"
fi

echo ""
echo -e "${MAGENTA}============================================${NC}"

if [ "$ALL_GOOD" = true ]; then
    echo -e "${GREEN}  Setup Complete!${NC}"
    echo -e "${MAGENTA}============================================${NC}"
    echo ""
    echo -e "${GREEN}InstantID is ready to use.${NC}"
    echo ""
    echo -e "${CYAN}Model paths:${NC}"
    echo "  Checkpoints: $CHECKPOINTS_DIR"
    echo "  Face model:  $ANTELOPE_DIR"
else
    echo -e "${YELLOW}  Setup Incomplete${NC}"
    echo -e "${MAGENTA}============================================${NC}"
    echo ""
    echo -e "${YELLOW}Some files are missing. The system will fall back to SDXL.${NC}"
    echo -e "${YELLOW}Re-run this script or download files manually.${NC}"
fi

echo ""
