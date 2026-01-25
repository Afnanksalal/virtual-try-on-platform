$ErrorActionPreference = "Stop"

Write-Host "Starting Virtual Try-On Platform..." -ForegroundColor Green

# Start Backend
Write-Host "Starting Backend on Port 8000..."
Start-Process -FilePath "uvicorn" -ArgumentList "main:app --host 0.0.0.0 --port 8000 --reload" -WorkingDirectory "backend" -NoNewWindow
# Note: In a real script we might want separate windows or background jobs

# Start Frontend
Write-Host "Starting Frontend..."
Set-Location "frontend"
npm run dev
