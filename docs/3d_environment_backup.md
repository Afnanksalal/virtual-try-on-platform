# 3D Environment Backup - Complete Restoration Guide

**Created**: February 11, 2026  
**Purpose**: Complete backup of working 3D pipeline environment for rollback if unification causes issues  
**System**: Windows 11, RTX 3050 4GB, Python 3.10.0, CUDA 12.4

---

## Environment Overview

This document provides a complete backup of the working 3D virtual try-on environment. Use this to restore the environment if the unification process causes issues.

### System Information

- **Python Version**: 3.10.0
- **Python Location**: `C:\Users\ASUS\AppData\Local\Programs\Python\Python310\python.exe`
- **Environment Type**: Standard Python (not conda)
- **CUDA Version**: 12.4
- **PyTorch Version**: 2.6.0+cu124
- **GPU**: NVIDIA GeForce RTX 3050 4GB Laptop GPU

### CUDA Information

```
CUDA Toolkit: 12.4
Location: C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4
PyTorch CUDA: 12.4
CUDA Available: True
```

---

## Quick Restoration Steps

If you need to restore this environment:

1. **Create fresh Python 3.10 environment**
2. **Install exact package versions** from `3d_requirements_baseline.txt`
3. **Follow manual compilation steps** (see Part 3 below)
4. **Verify with baseline test** (see Part 4 below)

---

## Part 1: Package Versions

### Complete Package List

All installed packages with exact versions are documented in:
- **`3d_environment_baseline.txt`** - Human-readable list (`pip list` output)
- **`3d_requirements_baseline.txt`** - Pip-installable format (`pip freeze` output)

### Critical Packages (Must Match Exactly)

```
torch==2.6.0+cu124
torchvision==0.21.0+cu124
torchaudio==2.6.0+cu124
transformers==5.1.0
diffusers==0.23.0
accelerate==0.24.0
safetensors==0.7.0
einops==0.7.0
timm (via transformers)
segment-anything==1.0
SAM-2==1.0 (from git)
open3d==0.19.0
opencv-python==4.11.0.86
opencv-python-headless==4.11.0.86
Pillow==10.1.0
numpy==1.26.4
scipy==1.15.3
scikit-image==0.25.2
trimesh==4.0.5
gradio==4.8.0
fastapi==0.128.5
uvicorn==0.40.0
pydantic==2.12.5
```

### Git-Installed Packages

These packages were installed from git repositories:

```bash
# SAM-2
pip install git+https://github.com/facebookresearch/segment-anything-2.git@2b90b9f5ceec907a1c18123530e92e794ad901a4

# torchmcubes (requires compilation - see Part 3)
pip install git+https://github.com/tatsy/torchmcubes.git@3381600ddc3d2e4d74222f8495866be5fafbace4
```

---

## Part 2: Prerequisites

### Required Software

1. **Visual Studio 2022 Build Tools**
   - Desktop development with C++
   - MSVC v143 - VS 2022 C++ x64/x86 build tools
   - Windows 10/11 SDK (10.0.22621.0)
   - C++ CMake tools for Windows
   
   **Paths**:
   ```
   Compiler: C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Tools\MSVC\14.44.35207\bin\Hostx64\x64\cl.exe
   RC.exe: C:\Program Files (x86)\Windows Kits\10\bin\10.0.22621.0\x64\rc.exe
   Libraries: C:\Program Files (x86)\Windows Kits\10\Lib\10.0.22621.0\um\x64\
   ```

2. **CUDA 12.4 Toolkit**
   - Download: https://developer.nvidia.com/cuda-12-4-0-download-archive
   - Install location: `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4`
   
   **Verify**:
   ```powershell
   nvcc --version
   # Should show: Cuda compilation tools, release 12.4
   ```

3. **Ninja Build System**
   ```powershell
   pip install ninja
   ```

---

## Part 3: Manual Compilation Steps

### 3.1 The torchmcubes Challenge

**Why manual compilation is needed**: Pre-built wheels don't work with PyTorch 2.6.0+cu124. Must compile from source.

### 3.2 NVTX Setup (Required for Compilation)

```powershell
# Clone NVTX repository for headers
git clone https://github.com/NVIDIA/NVTX.git
xcopy /E /I /Y "NVTX\c\include" "nvtx_local"
```

### 3.3 Patch PyTorch's CMake Configuration

**Critical Step**: PyTorch's CMake looks for `CUDA::nvToolsExt`, but CUDA 12.x removed it. Must patch the config file.

```powershell
# PowerShell script to patch cuda.cmake
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

### 3.4 Set Build Environment Variables

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

### 3.5 Compile torchmcubes

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

**Verify**:
```powershell
python -c "import torchmcubes; print('✓ torchmcubes installed successfully')"
```

---

## Part 4: Restoration Procedure

### Step-by-Step Restoration

1. **Install Python 3.10.0**
   ```powershell
   # Download from python.org and install to default location
   # Verify: python --version
   ```

2. **Install Prerequisites**
   - Visual Studio 2022 Build Tools (see Part 2)
   - CUDA 12.4 Toolkit (see Part 2)
   - Ninja: `pip install ninja`

3. **Install PyTorch First**
   ```powershell
   pip install torch==2.6.0+cu124 torchvision==0.21.0+cu124 torchaudio==2.6.0+cu124 --index-url https://download.pytorch.org/whl/cu124
   ```

4. **Verify PyTorch CUDA**
   ```powershell
   python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.version.cuda}'); print(f'CUDA Available: {torch.cuda.is_available()}')"
   ```
   
   Expected:
   ```
   PyTorch: 2.6.0+cu124
   CUDA: 12.4
   CUDA Available: True
   ```

5. **Compile torchmcubes**
   - Follow Part 3 steps (NVTX setup, patch PyTorch, set env vars, compile)

6. **Install Remaining Packages**
   ```powershell
   pip install -r 3d_requirements_baseline.txt
   ```
   
   Note: This will skip torch/torchvision/torchaudio (already installed) and torchmcubes (already compiled)

7. **Install SAM-2**
   ```powershell
   pip install git+https://github.com/facebookresearch/segment-anything-2.git
   ```

8. **Verify Installation**
   ```powershell
   python -c "import torch, transformers, diffusers, SAM2, torchmcubes, open3d; print('✓ All critical imports successful')"
   ```

---

## Part 5: Verification

### Run Baseline Test

The working 3D pipeline was verified with a baseline test. Results are stored in:
- **`backend/3d/baseline_test_results.json`** - Test results
- **`backend/3d/BASELINE_VERIFICATION.md`** - Verification documentation

To verify the restored environment:

```powershell
cd backend/3d
python test_pipeline.py
```

**Expected output**:
```
🎮 GPU: NVIDIA GeForce RTX 3050 4GB Laptop GPU (4.0GB)
💡 Strategy: Preprocessing on CPU, TripoSR on GPU (4GB VRAM optimization)
✓ Shared GPU memory enabled
Device: cuda:0 | Precision: torch.float16
Initializing Enhanced Production 3D Pipeline...
✓ All tests passed
```

Compare results with `baseline_test_results.json` to ensure consistency.

---

## Part 6: Known Issues and Solutions

### Issue 1: torchmcubes DLL Error

**Symptom**: `DLL load failed while importing torchmcubes_module`

**Cause**: PyTorch CUDA version mismatch or missing nvToolsExt patch

**Solution**:
1. Verify PyTorch CUDA version: `python -c "import torch; print(torch.version.cuda)"`
2. Should be `12.4` - if not, reinstall PyTorch
3. Re-apply nvToolsExt patch (Part 3.3)
4. Recompile torchmcubes (Part 3.5)

### Issue 2: CUDA Not Available

**Symptom**: `torch.cuda.is_available()` returns `False`

**Solution**:
1. Verify CUDA 12.4 installed: `nvcc --version`
2. Reinstall PyTorch with correct CUDA version
3. Check GPU drivers are up to date

### Issue 3: Import Errors

**Symptom**: `ModuleNotFoundError` for various packages

**Solution**:
1. Check package versions match `3d_requirements_baseline.txt`
2. Reinstall specific package: `pip install package==version`
3. For git packages, use exact commit hash

---

## Part 7: Environment Comparison

### Before Unification (This Backup)

- **Environment**: Standard Python 3.10.0
- **PyTorch**: 2.6.0+cu124
- **CUDA**: 12.4
- **Models**: TripoSR, SAM2, Depth Anything V2
- **Interface**: Gradio (backend/3d/app.py)
- **Status**: ✅ Working and verified

### After Unification (Target)

- **Environment**: Unified Python 3.10.x
- **PyTorch**: 2.6.0+cu124 (preserved)
- **CUDA**: 12.4 (preserved)
- **Models**: TripoSR, SAM2, Depth Anything V2, Leffa, SDXL
- **Interface**: FastAPI only (no Gradio)
- **Status**: To be implemented

---

## Part 8: Rollback Instructions

If unification causes issues and you need to rollback:

1. **Backup unified environment** (if you want to debug later)
   ```powershell
   pip freeze > unified_environment_failed.txt
   ```

2. **Uninstall all packages**
   ```powershell
   pip freeze > temp_packages.txt
   pip uninstall -r temp_packages.txt -y
   ```

3. **Follow restoration procedure** (Part 4)

4. **Verify with baseline test** (Part 5)

5. **Document what went wrong** for future debugging

---

## Part 9: Critical Files Reference

### Backup Files (Created by Task 0.1)

- `backend/3d_environment_baseline.txt` - Human-readable package list
- `backend/3d_requirements_baseline.txt` - Pip-installable requirements
- `backend/3d/baseline_test_results.json` - Test results

### Documentation Files

- `backend/3d/SETUP.md` - Complete setup guide with all manual steps
- `backend/3d/BASELINE_VERIFICATION.md` - Verification documentation
- `backend/3d_environment_backup.md` - This file

### Configuration Files

- `backend/3d/requirements.txt` - Original requirements (may differ from baseline)
- `backend/3d/config.json` - Application configuration
- `backend/3d/.env` - Environment variables

### Test Files

- `backend/3d/test_pipeline.py` - Pipeline verification script
- `backend/3d/app.py` - Working Gradio application

---

## Part 10: Notes for Unification

### Packages to Preserve (DO NOT CHANGE)

These packages make the 3D pipeline work. Keep exact versions:

```
torch==2.6.0+cu124
torchvision==0.21.0+cu124
torchaudio==2.6.0+cu124
transformers==5.1.0
einops==0.7.0
timm (from transformers)
SAM-2==1.0
segment-anything==1.0
open3d==0.19.0
opencv-python==4.11.0.86
opencv-python-headless==4.11.0.86
trimesh==4.0.5
torchmcubes (compiled from git)
```

### Packages That Can Be Upgraded

These can be adjusted for Leffa/SDXL compatibility:

```
diffusers==0.23.0 (may need upgrade for Leffa)
accelerate==0.24.0 (may need upgrade for SDXL)
safetensors==0.7.0 (likely compatible)
peft==0.6.0 (may need upgrade)
```

### Packages to Remove During Unification

```
gradio==4.8.0 (remove - using FastAPI only)
gradio_client==0.7.1 (remove)
dash==4.0.0 (remove if not needed)
Flask==3.1.2 (remove if not needed)
```

### Research Needed

Before unification, research:
1. Leffa's PyTorch version requirements
2. Leffa's diffusers version requirements
3. SDXL's accelerate version requirements
4. Compatibility between Leffa and PyTorch 2.6.0+cu124

---

## Part 11: Success Criteria

The environment is successfully restored when:

1. ✅ All packages from `3d_requirements_baseline.txt` are installed
2. ✅ PyTorch CUDA is available: `torch.cuda.is_available() == True`
3. ✅ PyTorch version is `2.6.0+cu124`
4. ✅ CUDA version is `12.4`
5. ✅ torchmcubes imports without errors
6. ✅ All critical imports work (torch, transformers, SAM2, open3d)
7. ✅ Baseline test passes: `python backend/3d/test_pipeline.py`
8. ✅ Test results match `baseline_test_results.json`
9. ✅ Gradio app starts: `python backend/3d/app.py`
10. ✅ 3D generation works end-to-end

---

## Part 12: Additional Resources

### Documentation

- **Complete Setup Guide**: `backend/3d/SETUP.md`
- **Baseline Verification**: `backend/3d/BASELINE_VERIFICATION.md`
- **TripoSR Docs**: https://github.com/VAST-AI-Research/TripoSR
- **SAM2 Docs**: https://github.com/facebookresearch/sam2
- **Depth Anything V2**: https://github.com/DepthAnything/Depth-Anything-V2

### Support

If restoration fails:
1. Check error messages against Part 6 (Known Issues)
2. Verify all prerequisites are installed (Part 2)
3. Ensure exact package versions match baseline
4. Check CUDA and PyTorch versions match exactly
5. Verify torchmcubes compilation succeeded

---

## Backup Metadata

- **Created**: February 11, 2026
- **Python**: 3.10.0
- **CUDA**: 12.4
- **PyTorch**: 2.6.0+cu124
- **System**: Windows 11, RTX 3050 4GB
- **Status**: ✅ Verified working
- **Purpose**: Rollback reference for unified-2d-3d-tryon spec
- **Related Tasks**: 0.1, 0.2, 0.3

---

**END OF BACKUP DOCUMENT**
