# ============================================================================
# InstantID Model Download Script
# ============================================================================

Write-Host "============================================================================" -ForegroundColor Cyan
Write-Host "InstantID Model Download Script" -ForegroundColor Cyan
Write-Host "============================================================================" -ForegroundColor Cyan
Write-Host ""

# Set up paths
$SCRIPT_DIR = Split-Path -Parent $MyInvocation.MyCommand.Path
$MODEL_DIR = Join-Path $SCRIPT_DIR "ml_engine\weights\instantid"
$FACE_MODEL_DIR = Join-Path $MODEL_DIR "models\antelopev2"

Write-Host "[1/4] Setting up directories..." -ForegroundColor Yellow
New-Item -ItemType Directory -Force -Path $MODEL_DIR | Out-Null
New-Item -ItemType Directory -Force -Path $FACE_MODEL_DIR | Out-Null
Write-Host "Done: Directories created at $MODEL_DIR" -ForegroundColor Green
Write-Host ""

# Check Python
Write-Host "[2/4] Checking Python installation..." -ForegroundColor Yellow
try {
    $pythonVersion = python --version 2>&1
    Write-Host "Done: Python found - $pythonVersion" -ForegroundColor Green
}
catch {
    Write-Host "Error: Python not found. Please install Python 3.10+ and try again." -ForegroundColor Red
    exit 1
}
Write-Host ""

# Download InstantID models
Write-Host "[3/4] Downloading InstantID models from HuggingFace..." -ForegroundColor Yellow
Write-Host "This will download approximately 1.5GB of model files. Please be patient..." -ForegroundColor Gray
Write-Host ""

# Load HuggingFace token from .env if available
$envFile = Join-Path $SCRIPT_DIR ".env"
$hfToken = ""
if (Test-Path $envFile) {
    Get-Content $envFile | ForEach-Object {
        if ($_ -match "^HUGGINGFACE_TOKEN=(.+)$") {
            $hfToken = $matches[1].Trim()
            Write-Host "Found HuggingFace token in .env file" -ForegroundColor Gray
        }
    }
}

$downloadScript = @"
from huggingface_hub import hf_hub_download
import os

model_dir = r'$MODEL_DIR'
token = '$hfToken' if '$hfToken' else None

print('Downloading InstantID ControlNet config...')
hf_hub_download(
    repo_id='InstantX/InstantID',
    filename='ControlNetModel/config.json',
    local_dir=model_dir,
    token=token
)
print('Done: ControlNet config downloaded')

print('Downloading InstantID ControlNet weights (this may take a while)...')
hf_hub_download(
    repo_id='InstantX/InstantID',
    filename='ControlNetModel/diffusion_pytorch_model.safetensors',
    local_dir=model_dir,
    token=token
)
print('Done: ControlNet weights downloaded')

print('Downloading InstantID IP-Adapter...')
hf_hub_download(
    repo_id='InstantX/InstantID',
    filename='ip-adapter.bin',
    local_dir=model_dir,
    token=token
)
print('Done: IP-Adapter downloaded')

print('')
print('All InstantID models downloaded successfully!')
"@

$tempScript = Join-Path $env:TEMP "download_instantid.py"
$downloadScript | Out-File -FilePath $tempScript -Encoding UTF8

try {
    python $tempScript
    if ($LASTEXITCODE -ne 0) {
        throw "Python script failed with exit code $LASTEXITCODE"
    }
    Write-Host ""
    Write-Host "Done: InstantID models downloaded successfully!" -ForegroundColor Green
}
catch {
    Write-Host "Error: Failed to download InstantID models - $_" -ForegroundColor Red
    Write-Host "Please check your internet connection and try again." -ForegroundColor Red
    Remove-Item $tempScript -ErrorAction SilentlyContinue
    exit 1
}
finally {
    Remove-Item $tempScript -ErrorAction SilentlyContinue
}
Write-Host ""

# Download InsightFace model
Write-Host "[4/4] Downloading InsightFace antelopev2 face model..." -ForegroundColor Yellow
Write-Host "This model must be downloaded manually due to licensing." -ForegroundColor Gray
Write-Host ""

$antelopeUrl = "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip"
$antelopeZip = Join-Path $MODEL_DIR "buffalo_l.zip"

Write-Host "Downloading from: $antelopeUrl" -ForegroundColor Gray

try {
    $webClient = New-Object System.Net.WebClient
    
    Register-ObjectEvent -InputObject $webClient -EventName DownloadProgressChanged -SourceIdentifier WebClient.DownloadProgressChanged -Action {
        $percent = $EventArgs.ProgressPercentage
        Write-Progress -Activity "Downloading antelopev2 model" -Status "$percent% Complete" -PercentComplete $percent
    } | Out-Null
    
    $webClient.DownloadFile($antelopeUrl, $antelopeZip)
    
    Unregister-Event -SourceIdentifier WebClient.DownloadProgressChanged -ErrorAction SilentlyContinue
    Remove-Job -Name WebClient.DownloadProgressChanged -ErrorAction SilentlyContinue
    
    Write-Host "Done: Downloaded buffalo_l.zip" -ForegroundColor Green
    
    Write-Host "Extracting face model..." -ForegroundColor Gray
    Expand-Archive -Path $antelopeZip -DestinationPath $FACE_MODEL_DIR -Force
    
    Remove-Item $antelopeZip -Force
    
    Write-Host "Done: InsightFace antelopev2 model installed!" -ForegroundColor Green
}
catch {
    Write-Host "Error: Failed to download antelopev2 model - $_" -ForegroundColor Red
    Write-Host ""
    Write-Host "Please download manually:" -ForegroundColor Yellow
    Write-Host "1. Download from: https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip" -ForegroundColor Yellow
    Write-Host "2. Extract to: $FACE_MODEL_DIR" -ForegroundColor Yellow
    Write-Host ""
}

Write-Host ""
Write-Host "============================================================================" -ForegroundColor Cyan
Write-Host "Download Complete!" -ForegroundColor Green
Write-Host "============================================================================" -ForegroundColor Cyan
Write-Host ""

# Verify installation
Write-Host "Verifying installation..." -ForegroundColor Yellow
Write-Host ""

$controlnetConfig = Join-Path $MODEL_DIR "ControlNetModel\config.json"
$controlnetWeights = Join-Path $MODEL_DIR "ControlNetModel\diffusion_pytorch_model.safetensors"
$ipAdapter = Join-Path $MODEL_DIR "ip-adapter.bin"
$faceModel = Join-Path $FACE_MODEL_DIR "1k3d68.onnx"

$allFilesPresent = $true

if (Test-Path $controlnetConfig) {
    Write-Host "Done: ControlNet config found" -ForegroundColor Green
}
else {
    Write-Host "Error: ControlNet config missing" -ForegroundColor Red
    $allFilesPresent = $false
}

if (Test-Path $controlnetWeights) {
    Write-Host "Done: ControlNet weights found" -ForegroundColor Green
}
else {
    Write-Host "Error: ControlNet weights missing" -ForegroundColor Red
    $allFilesPresent = $false
}

if (Test-Path $ipAdapter) {
    Write-Host "Done: IP-Adapter found" -ForegroundColor Green
}
else {
    Write-Host "Error: IP-Adapter missing" -ForegroundColor Red
    $allFilesPresent = $false
}

if (Test-Path $faceModel) {
    Write-Host "Done: Face model (antelopev2) found" -ForegroundColor Green
}
else {
    Write-Host "Error: Face model (antelopev2) missing" -ForegroundColor Red
    $allFilesPresent = $false
}

Write-Host ""

if ($allFilesPresent) {
    Write-Host "============================================================================" -ForegroundColor Green
    Write-Host "All models installed successfully!" -ForegroundColor Green
    Write-Host "============================================================================" -ForegroundColor Green
    Write-Host ""
    Write-Host "You can now use InstantID for identity-preserving body generation." -ForegroundColor Cyan
    Write-Host ""
    Write-Host "Next steps:" -ForegroundColor Yellow
    Write-Host "1. Install dependencies: pip install insightface==0.7.3 albumentations==1.4.23" -ForegroundColor Gray
    Write-Host "2. Start the backend: python main.py" -ForegroundColor Gray
    Write-Host "3. Test the endpoint: POST /api/v1/generate-identity-body" -ForegroundColor Gray
}
else {
    Write-Host "============================================================================" -ForegroundColor Red
    Write-Host "Some models are missing!" -ForegroundColor Red
    Write-Host "============================================================================" -ForegroundColor Red
    Write-Host ""
    Write-Host "Please check the errors above and try again." -ForegroundColor Yellow
    Write-Host "You may need to download some models manually." -ForegroundColor Yellow
}

Write-Host ""
Write-Host "Model directory: $MODEL_DIR" -ForegroundColor Gray
Write-Host ""
