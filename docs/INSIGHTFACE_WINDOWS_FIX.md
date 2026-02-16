# InsightFace Windows Installation Fix

**Issue**: InsightFace 0.7.3 fails to build on Windows due to missing Microsoft Visual C++ 14.0 build tools.

**Error Message**:
```
error: Microsoft Visual C++ 14.0 or greater is required. Get it with "Microsoft C++ Build Tools"
```

## Solution: Use Pre-Built Wheel

Instead of building from source, use a pre-built wheel file for Windows.

### Step 1: Download Pre-Built Wheel

Download the appropriate wheel for Python 3.10:

**Direct Download Links**:
- Python 3.10: https://github.com/Gourieff/Assets/raw/main/Insightface/insightface-0.7.3-cp310-cp310-win_amd64.whl
- Python 3.11: https://github.com/Gourieff/Assets/raw/main/Insightface/insightface-0.7.3-cp311-cp311-win_amd64.whl
- Python 3.12: https://github.com/Gourieff/Assets/raw/main/Insightface/insightface-0.7.3-cp312-cp312-win_amd64.whl

**Alternative Source**:
- https://github.com/C0untFloyd/roop-unleashed/releases/download/3.6.6/insightface-0.7.3-cp310-cp310-win_amd64.whl

### Step 2: Install the Wheel

```powershell
cd backend

# Download the wheel (use your Python version)
# For Python 3.10:
curl -L -o insightface-0.7.3-cp310-cp310-win_amd64.whl https://github.com/Gourieff/Assets/raw/main/Insightface/insightface-0.7.3-cp310-cp310-win_amd64.whl

# Install the wheel
pip install insightface-0.7.3-cp310-cp310-win_amd64.whl

# Install albumentations
pip install albumentations==1.4.23

# Clean up
del insightface-0.7.3-cp310-cp310-win_amd64.whl
```

### Step 3: Verify Installation

```powershell
python -c "import insightface; print('InsightFace version:', insightface.__version__)"
```

Expected output:
```
InsightFace version: 0.7.3
```

## Alternative: Install Visual Studio Build Tools (Not Recommended)

If you prefer to build from source, install Microsoft Visual C++ Build Tools:

1. Download: https://visualstudio.microsoft.com/visual-cpp-build-tools/
2. Install "Desktop development with C++" workload
3. Restart your terminal
4. Run: `pip install insightface==0.7.3`

**Note**: This requires ~6GB download and installation. Using pre-built wheels is much faster.

## Automated Installation Script

Create a file `install_insightface.ps1`:

```powershell
# InsightFace Windows Installation Script
# Automatically downloads and installs pre-built wheel

Write-Host "Installing InsightFace for Windows..." -ForegroundColor Cyan

# Detect Python version
$pythonVersion = python -c "import sys; print(f'{sys.version_info.major}{sys.version_info.minor}')"
Write-Host "Detected Python version: $pythonVersion" -ForegroundColor Gray

# Determine wheel URL based on Python version
$wheelUrls = @{
    "310" = "https://github.com/Gourieff/Assets/raw/main/Insightface/insightface-0.7.3-cp310-cp310-win_amd64.whl"
    "311" = "https://github.com/Gourieff/Assets/raw/main/Insightface/insightface-0.7.3-cp311-cp311-win_amd64.whl"
    "312" = "https://github.com/Gourieff/Assets/raw/main/Insightface/insightface-0.7.3-cp312-cp312-win_amd64.whl"
}

if (-not $wheelUrls.ContainsKey($pythonVersion)) {
    Write-Host "Error: Unsupported Python version. Requires Python 3.10, 3.11, or 3.12" -ForegroundColor Red
    exit 1
}

$wheelUrl = $wheelUrls[$pythonVersion]
$wheelFile = "insightface-0.7.3-cp$pythonVersion-cp$pythonVersion-win_amd64.whl"

Write-Host "Downloading pre-built wheel..." -ForegroundColor Yellow
try {
    Invoke-WebRequest -Uri $wheelUrl -OutFile $wheelFile
    Write-Host "Done: Wheel downloaded" -ForegroundColor Green
}
catch {
    Write-Host "Error: Failed to download wheel - $_" -ForegroundColor Red
    exit 1
}

Write-Host "Installing InsightFace..." -ForegroundColor Yellow
pip install $wheelFile

if ($LASTEXITCODE -eq 0) {
    Write-Host "Done: InsightFace installed successfully!" -ForegroundColor Green
    
    # Clean up
    Remove-Item $wheelFile -Force
    
    # Verify installation
    Write-Host "Verifying installation..." -ForegroundColor Yellow
    python -c "import insightface; print('InsightFace version:', insightface.__version__)"
    
    Write-Host ""
    Write-Host "Installation complete! You can now use InstantID." -ForegroundColor Cyan
}
else {
    Write-Host "Error: Installation failed" -ForegroundColor Red
    Remove-Item $wheelFile -Force -ErrorAction SilentlyContinue
    exit 1
}
```

Run the script:
```powershell
cd backend
.\install_insightface.ps1
```

## Troubleshooting

### "is not a supported wheel on this platform"

**Cause**: Python version mismatch

**Solution**: 
1. Check your Python version: `python --version`
2. Download the correct wheel for your Python version
3. Ensure you're using Python 3.10, 3.11, or 3.12

### "No module named 'insightface'"

**Cause**: Installation failed or wrong virtual environment

**Solution**:
1. Verify virtual environment is activated
2. Reinstall: `pip uninstall insightface && pip install <wheel-file>`
3. Check installation: `pip list | grep insightface`

### DLL Load Failure

**Cause**: Missing Visual C++ Redistributable

**Solution**:
1. Download: https://aka.ms/vs/17/release/vc_redist.x64.exe
2. Install Microsoft Visual C++ Redistributable
3. Restart terminal and try again

## Dependencies

InsightFace requires these packages (automatically installed):
- numpy<2.0 (InsightFace requires numpy 1.x)
- onnx>=1.16.0
- onnxruntime-gpu>=1.19.0 (or onnxruntime for CPU)
- opencv-python
- scikit-learn
- scikit-image
- easydict
- cython
- albumentations

## References

- Pre-built wheels: https://github.com/Gourieff/Assets/tree/main/Insightface
- InsightFace GitHub: https://github.com/deepinsight/insightface
- Windows installation guide: https://github.com/cobanov/insightface_windows

---

**Last Updated**: February 15, 2026
**Tested On**: Windows 11, Python 3.10
