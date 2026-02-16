# Complete Setup Guide - 3D Virtual Try-On System

This guide covers the complete installation process including all the challenges we encountered and how to solve them.

## System Requirements

- **OS**: Windows 10/11 (64-bit)
- **GPU**: NVIDIA GPU with 4GB+ VRAM (tested on RTX 3050 4GB)
- **RAM**: 16GB+ recommended
- **Storage**: 20GB+ free space
- **CUDA**: 12.4 (specific version required)
- **Python**: 3.10.x

---

## Part 1: Prerequisites Installation

### 1.1 Visual Studio 2022 Build Tools

**Why needed**: Required to compile C++ extensions like torchmcubes

1. Download Visual Studio 2022 Build Tools:
   ```
   https://visualstudio.microsoft.com/downloads/
   ```

2. Run the installer and select:
   - ✅ Desktop development with C++
   - ✅ MSVC v143 - VS 2022 C++ x64/x86 build tools
   - ✅ Windows 10/11 SDK (10.0.22621.0 or latest)
   - ✅ C++ CMake tools for Windows

3. Installation paths (note these for later):
   ```
   Compiler: C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Tools\MSVC\14.44.35207\bin\Hostx64\x64\cl.exe
   RC.exe: C:\Program Files (x86)\Windows Kits\10\bin\10.0.22621.0\x64\rc.exe
   Libraries: C:\Program Files (x86)\Windows Kits\10\Lib\10.0.22621.0\um\x64\
   ```

### 1.2 CUDA 12.4 Toolkit

**Why this specific version**: PyTorch 2.6.0 requires CUDA 12.4, and torchmcubes must be compiled against the same CUDA version.

1. Download CUDA 12.4:
   ```
   https://developer.nvidia.com/cuda-12-4-0-download-archive
   ```

2. Install to default location:
   ```
   C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4
   ```

3. Verify installation:
   ```powershell
   nvcc --version
   ```
   Should show: `Cuda compilation tools, release 12.4`

### 1.3 Ninja Build System

**Why needed**: Fast build system for CMake projects

```powershell
pip install ninja
```

Verify:
```powershell
ninja --version
```

---

## Part 2: Python Environment Setup

### 2.1 Create Virtual Environment

```powershell
cd C:\Users\ASUS\Downloads\afnan
python -m venv .
```

### 2.2 Activate Environment

```powershell
.\Scripts\activate
```

### 2.3 Install PyTorch with CUDA 12.4

**CRITICAL**: Must install PyTorch with CUDA 12.4 support BEFORE other packages.

```powershell
pip install torch==2.6.0+cu124 torchvision==0.21.0+cu124 torchaudio==2.5.1+cu124 --index-url https://download.pytorch.org/whl/cu124
```

Verify:
```powershell
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.version.cuda}'); print(f'CUDA Available: {torch.cuda.is_available()}')"
```

Expected output:
```
PyTorch: 2.6.0+cu124
CUDA: 12.4
CUDA Available: True
```

---

## Part 3: The torchmcubes Challenge

### 3.1 The Problem

torchmcubes is required by TripoSR for marching cubes mesh extraction. The pre-built wheels don't work with PyTorch 2.6.0+cu124, so we must compile from source.

**Error you'll encounter**:
```
DLL load failed while importing torchmcubes_module: The specified module could not be found.
```

### 3.2 The nvToolsExt Issue

PyTorch's CMake configuration looks for `CUDA::nvToolsExt`, but CUDA 12.x removed nvToolsExt from the toolkit. However, PyTorch ships with its own copy.

**Error during compilation**:
```
CMake Error: The link interface of target "torch::nvtoolsext" contains:
  CUDA::nvToolsExt
but the target was not found.
```

### 3.3 Solution: Patch PyTorch's CMake Config

**Step 1**: Clone NVTX repository (for headers):
```powershell
git clone https://github.com/NVIDIA/NVTX.git
xcopy /E /I /Y "NVTX\c\include" "nvtx_local"
```

**Step 2**: Patch PyTorch's cuda.cmake to comment out nvToolsExt dependency:

```powershell
# Run this PowerShell script to patch the file
$cudaCmake = "C:\Users\ASUS\AppData\Local\Programs\Python\Python310\Lib\site-packages\torch\share\cmake\Caffe2\public\cuda.cmake"
$lines = Get-Content $cudaCmake
$newLines = @()

for($i=0; $i -lt $lines.Count; $i++) {
    if($lines[$i] -match "set_property\(TARGET torch::nvtoolsext.*CUDA::nvToolsExt\)") {
        $newLines += "  # Commented out to avoid nvToolsExt dependency issue"
        $newLines += "  # " + $lines[$i]
    } else {
        $newLines += $lines[$i]
    }
}

$newLines | Set-Content $cudaCmake
Write-Host "✓ Patched cuda.cmake - nvToolsExt dependency removed"
```

### 3.4 Compile torchmcubes

**Step 1**: Set up build environment:

```powershell
# Set compiler paths
$env:PATH = "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Tools\MSVC\14.44.35207\bin\Hostx64\x64;C:\Program Files (x86)\Windows Kits\10\bin\10.0.22621.0\x64;" + $env:PATH

# Set library paths
$env:LIB = "C:\Program Files (x86)\Windows Kits\10\Lib\10.0.22621.0\um\x64;C:\Program Files (x86)\Windows Kits\10\Lib\10.0.22621.0\ucrt\x64;C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Tools\MSVC\14.44.35207\lib\x64"

# Set include paths
$env:INCLUDE = "C:\Program Files (x86)\Windows Kits\10\Include\10.0.22621.0\um;C:\Program Files (x86)\Windows Kits\10\Include\10.0.22621.0\ucrt;C:\Program Files (x86)\Windows Kits\10\Include\10.0.22621.0\shared;C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Tools\MSVC\14.44.35207\include"

# Set CUDA path
$env:CUDA_PATH = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4"
```

**Step 2**: Install torchmcubes from source:

```powershell
pip install --no-cache-dir git+https://github.com/tatsy/torchmcubes.git
```

**Expected output**:
```
Building wheels for collected packages: torchmcubes
  Building wheel for torchmcubes (pyproject.toml) ... done
  Created wheel for torchmcubes: filename=torchmcubes-0.1.0-cp310-cp310-win_amd64.whl
Successfully built torchmcubes
Installing collected packages: torchmcubes
Successfully installed torchmcubes-0.1.0
```

**Step 3**: Verify installation:

```powershell
python -c "import torchmcubes; print('✓ torchmcubes installed successfully')"
```

---

## Part 4: Install Application Dependencies

### 4.1 Install Core Dependencies

```powershell
pip install -r requirements.txt
```

### 4.2 Install SAM 2

```powershell
pip install git+https://github.com/facebookresearch/sam2.git
```

### 4.3 Install Additional Dependencies

```powershell
pip install transformers accelerate open3d
```

---

## Part 5: Model Setup

### 5.1 Create Required Directories

```powershell
mkdir models
mkdir outputs
mkdir TripoSR
```

### 5.2 Models Will Auto-Download

The following models will download automatically on first run:
- **SAM 2.1**: `facebook/sam2.1-hiera-large` (~900MB)
- **Depth Anything V2**: `depth-anything/Depth-Anything-V2-Large-hf` (~1.3GB)
- **TripoSR**: `stabilityai/TripoSR` (~500MB)

---

## Part 6: Troubleshooting

### Issue 1: PyTorch CUDA Not Available

**Symptom**: `torch.cuda.is_available()` returns `False`

**Solution**: Reinstall PyTorch with CUDA:
```powershell
pip uninstall torch torchvision torchaudio -y
pip install torch==2.6.0+cu124 torchvision==0.21.0+cu124 torchaudio==2.5.1+cu124 --index-url https://download.pytorch.org/whl/cu124
```

### Issue 2: torchmcubes DLL Error

**Symptom**: `DLL load failed while importing torchmcubes_module`

**Solution**: 
1. Verify PyTorch CUDA version matches CUDA toolkit (12.4)
2. Re-patch cuda.cmake (see Part 3.3)
3. Recompile torchmcubes (see Part 3.4)

### Issue 3: Out of Memory (OOM) Errors

**Symptom**: `CUDA out of memory` during inference

**Solution**: The app is optimized for 4GB VRAM:
- SAM2 and Depth Anything V2 run on CPU
- TripoSR runs on GPU with FP16 precision
- Resolution is set to 128 for 4GB VRAM

If still OOM, reduce resolution in `app.py`:
```python
MESH_RESOLUTIONS = [128]  # Only use lowest resolution
```

### Issue 4: Gradio Version Warning

**Symptom**: `You are using gradio version 4.8.0, however version 4.44.1 is available`

**Solution** (optional):
```powershell
pip install --upgrade gradio
```

### Issue 5: SAM2 Segmentation Fails

**Symptom**: `arrays used as indices must be of integer (or boolean) type`

**Solution**: Already fixed in latest code. If you encounter this, update `app.py`.

---

## Part 7: Running the Application

### 7.1 Start the Server

```powershell
python app.py
```

### 7.2 Expected Output

```
🎮 GPU: NVIDIA GeForce RTX 3050 4GB Laptop GPU (4.0GB)
💡 Strategy: Preprocessing on CPU, TripoSR on GPU (4GB VRAM optimization)
✓ Shared GPU memory enabled (can use system RAM as extended VRAM)
Device: cuda:0 | Precision: torch.float16
Initializing Enhanced Production 3D Pipeline...
Strategy: SAM2 & Depth on CPU → TripoSR on GPU (4GB VRAM optimization)
Features: SAM2 segmentation + Depth Anything V2 + Multi-view optimization
Starting 3D Try-On System...
GPU: NVIDIA GeForce RTX 3050 4GB Laptop GPU (4.0GB)
Running on local URL:  http://0.0.0.0:7860
```

### 7.3 Access the Application

Open your browser and navigate to:
```
http://localhost:7860
```

---

## Part 8: Performance Optimization

### 8.1 Memory Management

The application uses aggressive memory management:
- **Nuclear memory reset**: Clears GPU memory between stages
- **Immediate offloading**: Models moved to CPU after use
- **FP16 precision**: Reduces VRAM usage by 50%

### 8.2 Expected VRAM Usage

- **Idle**: ~0.14GB
- **After SAM2/Depth**: ~0.14GB (runs on CPU)
- **During TripoSR**: ~1.7GB (GPU inference)
- **After generation**: ~0.14GB (offloaded to CPU)

### 8.3 Processing Times (RTX 3050 4GB)

- **Depth estimation**: ~7 seconds (CPU)
- **SAM2 segmentation**: ~13 seconds (CPU)
- **TripoSR inference**: ~11 seconds (GPU)
- **Total single-view**: ~35-40 seconds

---

## Part 9: Advanced Configuration

### 9.1 Increase Mesh Quality

Edit `app.py`:
```python
# For 6GB+ VRAM, use higher resolution
resolution = 256  # or 512 for 8GB+ VRAM
```

### 9.2 Disable Depth Preprocessing

In the UI, uncheck "Use Depth Preprocessing" for faster processing (lower quality).

### 9.3 Multi-View Settings

For best multi-view results:
- Use consistent lighting across views
- Keep camera distance similar
- Provide at least 3 views (Front, Back, Left/Right)

---

## Part 10: File Structure

```
afnan/
├── app.py                          # Main application
├── requirements.txt                # Python dependencies
├── SETUP.md                        # This file
├── config.json                     # Configuration
├── models/                         # Model cache (auto-created)
├── outputs/                        # Generated 3D models
├── TripoSR/                        # TripoSR repository (auto-cloned)
│   └── tsr/
│       ├── enhanced_system.py      # Enhanced TripoSR with quality improvements
│       ├── system.py               # Standard TripoSR
│       └── models/
├── nvtx_local/                     # NVTX headers (for compilation)
├── NVTX/                           # NVTX repository (for compilation)
├── Scripts/                        # Virtual environment scripts
├── Lib/                            # Virtual environment libraries
└── Include/                        # Virtual environment includes
```

---

## Part 11: Quick Setup Script

Save this as `setup.ps1` and run it after installing Visual Studio and CUDA:

```powershell
# Quick Setup Script for 3D Virtual Try-On System

Write-Host "=== 3D Virtual Try-On System - Quick Setup ===" -ForegroundColor Cyan

# Step 1: Create virtual environment
Write-Host "`n[1/6] Creating virtual environment..." -ForegroundColor Yellow
python -m venv .

# Step 2: Activate environment
Write-Host "`n[2/6] Activating environment..." -ForegroundColor Yellow
.\Scripts\activate

# Step 3: Install PyTorch with CUDA 12.4
Write-Host "`n[3/6] Installing PyTorch with CUDA 12.4..." -ForegroundColor Yellow
pip install torch==2.6.0+cu124 torchvision==0.21.0+cu124 torchaudio==2.5.1+cu124 --index-url https://download.pytorch.org/whl/cu124

# Step 4: Clone NVTX and patch PyTorch
Write-Host "`n[4/6] Setting up NVTX and patching PyTorch..." -ForegroundColor Yellow
git clone https://github.com/NVIDIA/NVTX.git
xcopy /E /I /Y "NVTX\c\include" "nvtx_local"

$cudaCmake = "C:\Users\ASUS\AppData\Local\Programs\Python\Python310\Lib\site-packages\torch\share\cmake\Caffe2\public\cuda.cmake"
$lines = Get-Content $cudaCmake
$newLines = @()
for($i=0; $i -lt $lines.Count; $i++) {
    if($lines[$i] -match "set_property\(TARGET torch::nvtoolsext.*CUDA::nvToolsExt\)") {
        $newLines += "  # Commented out to avoid nvToolsExt dependency issue"
        $newLines += "  # " + $lines[$i]
    } else {
        $newLines += $lines[$i]
    }
}
$newLines | Set-Content $cudaCmake
Write-Host "✓ Patched cuda.cmake" -ForegroundColor Green

# Step 5: Compile torchmcubes
Write-Host "`n[5/6] Compiling torchmcubes (this may take 5-10 minutes)..." -ForegroundColor Yellow
$env:PATH = "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Tools\MSVC\14.44.35207\bin\Hostx64\x64;C:\Program Files (x86)\Windows Kits\10\bin\10.0.22621.0\x64;" + $env:PATH
$env:LIB = "C:\Program Files (x86)\Windows Kits\10\Lib\10.0.22621.0\um\x64;C:\Program Files (x86)\Windows Kits\10\Lib\10.0.22621.0\ucrt\x64;C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Tools\MSVC\14.44.35207\lib\x64"
$env:INCLUDE = "C:\Program Files (x86)\Windows Kits\10\Include\10.0.22621.0\um;C:\Program Files (x86)\Windows Kits\10\Include\10.0.22621.0\ucrt;C:\Program Files (x86)\Windows Kits\10\Include\10.0.22621.0\shared;C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Tools\MSVC\14.44.35207\include"
$env:CUDA_PATH = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4"
pip install --no-cache-dir git+https://github.com/tatsy/torchmcubes.git

# Step 6: Install application dependencies
Write-Host "`n[6/6] Installing application dependencies..." -ForegroundColor Yellow
pip install -r requirements.txt
pip install git+https://github.com/facebookresearch/sam2.git
pip install transformers accelerate open3d

Write-Host "`n=== Setup Complete! ===" -ForegroundColor Green
Write-Host "Run the application with: python app.py" -ForegroundColor Cyan
```

---

## Part 12: Common Questions

### Q: Can I use a different CUDA version?

**A**: No. PyTorch 2.6.0 requires CUDA 12.4 specifically. Using a different version will cause compatibility issues.

### Q: Can I use AMD GPU or CPU only?

**A**: The app will run on CPU, but it will be extremely slow (10-20x slower). AMD GPU support via ROCm is not tested.

### Q: Why does compilation take so long?

**A**: torchmcubes compiles CUDA kernels from source, which involves:
- CMake configuration (~40 seconds)
- C++ compilation (~2 minutes)
- CUDA compilation (~3 minutes)
- Linking (~1 minute)

### Q: Can I skip the nvToolsExt patch?

**A**: No. Without the patch, torchmcubes compilation will fail with CMake errors.

### Q: What if I have CUDA 12.1 installed?

**A**: You need CUDA 12.4. Install it alongside 12.1 (they can coexist). Make sure `CUDA_PATH` points to 12.4.

---

## Part 13: Credits & References

- **TripoSR**: https://github.com/VAST-AI-Research/TripoSR
- **SAM 2**: https://github.com/facebookresearch/sam2
- **Depth Anything V2**: https://github.com/DepthAnything/Depth-Anything-V2
- **torchmcubes**: https://github.com/tatsy/torchmcubes
- **NVTX**: https://github.com/NVIDIA/NVTX

---

## Support

If you encounter issues not covered in this guide:

1. Check the error message carefully
2. Verify all paths in Part 3.4 match your system
3. Ensure CUDA 12.4 and PyTorch 2.6.0+cu124 are installed
4. Try the troubleshooting steps in Part 6

**Last Updated**: February 11, 2026
**Tested On**: Windows 11, RTX 3050 4GB, CUDA 12.4, Python 3.10
