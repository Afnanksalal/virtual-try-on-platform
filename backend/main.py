from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import os
import torch
from dotenv import load_dotenv
from app.core.logging_config import setup_logging, get_logger
from app.core.middleware import RequestLoggingMiddleware
from contextlib import asynccontextmanager

# Load environment variables from .env file
load_dotenv()

# Initialize logging
log_level = os.getenv("LOG_LEVEL", "INFO")
setup_logging(log_level=log_level, log_file="logs/app.log")
logger = get_logger("main")

# Log environment configuration on startup
logger.info(f"Environment loaded - GEMINI_API_KEY: {'✓ Set' if os.getenv('GEMINI_API_KEY') else '✗ Not set'}")

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown hooks."""
    # Startup
    logger.info("Starting Virtual Try-On ML API...")
    
    # Log GPU status
    gpu_available = torch.cuda.is_available()
    device = torch.cuda.get_device_name(0) if gpu_available else "CPU"
    logger.info(f"Compute device: {device}")
    
    # Preload ML models
    from ml_engine.loader import model_loader
    try:
        logger.info("Preloading ML models...")
        await model_loader.preload_models()
        logger.info("ML models preloaded successfully")
        
        # Warmup models if enabled
        warmup_enabled = os.getenv("WARMUP_MODELS", "true").lower() == "true"
        if warmup_enabled:
            logger.info("Warming up ML models...")
            models_to_warmup = os.getenv("PRELOAD_MODELS", "tryon,segmentation").split(",")
            for model_name in [m.strip() for m in models_to_warmup if m.strip()]:
                await model_loader.warmup_model(model_name)
            logger.info("ML models warmed up successfully")
    except Exception as e:
        logger.error(f"Failed to preload/warmup models: {e}", exc_info=True)
        logger.warning("API will start but models will be loaded on first request")
    
    yield
    
    # Shutdown
    logger.info("Shutting down Virtual Try-On ML API...")

app = FastAPI(
    title="Virtual Try-On ML API",
    description="Machine Learning API for Virtual Try-On - Recommendations, Processing, Body Generation",
    version="1.0.0",
    lifespan=lifespan
)

# CORS Configuration 
# SECURITY: Use explicit origins in production, never "*"
allowed_origins_str = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000")
origins = [origin.strip() for origin in allowed_origins_str.split(",")]

# Validate CORS configuration
if "*" in origins:
    logger.warning(
        "CORS configured to allow ALL origins (*). "
        "This is a SECURITY RISK in production! "
        "Set ALLOWED_ORIGINS environment variable to specific domains."
    )

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# Add custom middleware
app.add_middleware(RequestLoggingMiddleware)

from app.api.endpoints import router as api_router
from app.api.body_generation import router as body_gen_router
from app.api.image_analysis import router as image_analysis_router
from app.api.image_composition import router as image_composition_router

app.include_router(api_router, prefix="/api/v1")
app.include_router(body_gen_router, prefix="/api/v1")
app.include_router(image_analysis_router, prefix="/api/v1")
app.include_router(image_composition_router, prefix="/api/v1")

@app.get("/")
async def root():
    return {
        "message": "Virtual Try-On ML API Online", 
        "status": "active",
        "version": "1.0.0",
        "type": "ml-service"
    }

@app.get("/health")
async def health_check():
    """
    ML service health check - GPU, AI services, environment
    """
    health_status = {
        "status": "healthy",
        "service": "ml-api",
        "checks": {}
    }
    
    # 1. GPU/Device Check
    gpu_status = torch.cuda.is_available()
    device_name = torch.cuda.get_device_name(0) if gpu_status else "CPU"
    health_status["checks"]["compute"] = {
        "gpu_available": gpu_status,
        "device": device_name
    }
    
    # 2. Gemini API Key Check (for recommendations)
    gemini_key = os.getenv("GEMINI_API_KEY")
    health_status["checks"]["ai_service"] = {
        "configured": bool(gemini_key),
        "masked_key": f"{gemini_key[:8]}..." if gemini_key else None
    }
    if not gemini_key:
        logger.warning("GEMINI_API_KEY not configured - recommendations will fail")
        health_status["status"] = "degraded"
    
    # 3. Environment Check
    health_status["checks"]["environment"] = {
        "log_level": os.getenv("LOG_LEVEL", "INFO"),
        "cors_origins": len(origins)
    }
    
    logger.info(f"Health check performed: {health_status['status']}")
    return health_status

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
