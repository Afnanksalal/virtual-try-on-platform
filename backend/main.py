from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
import uvicorn
import os
import torch
from dotenv import load_dotenv
from app.core.logging_config import setup_logging, get_logger
from app.core.middleware import RequestLoggingMiddleware
from app.core.error_handler import (
    app_exception_handler,
    validation_exception_handler,
    http_exception_handler,
    generic_exception_handler
)
from app.core.exceptions import AppException
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
    logger.info("=" * 60)
    logger.info("Starting Virtual Try-On ML API...")
    logger.info("=" * 60)
    
    # Log GPU status
    gpu_available = torch.cuda.is_available()
    device = torch.cuda.get_device_name(0) if gpu_available else "CPU"
    logger.info(f"Compute device: {device}")
    
    # Preload ML models (includes checkpoint download on first run)
    from ml_engine.loader import model_loader
    try:
        logger.info("Preloading ML models...")
        logger.info("Note: On first run, this will download Leffa checkpoints (several GB)")
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
    
    # Start temporary file cleanup background task
    from app.services.temp_file_manager import TempFileManager
    import asyncio
    
    temp_file_manager = TempFileManager()
    cleanup_task = None
    
    async def periodic_cleanup():
        """Run cleanup every 15 minutes."""
        while True:
            try:
                await asyncio.sleep(900)  # 15 minutes = 900 seconds
                deleted_count = temp_file_manager.cleanup_expired_files()
                if deleted_count > 0:
                    logger.info(f"Background cleanup: removed {deleted_count} expired temp files")
            except asyncio.CancelledError:
                logger.info("Cleanup task cancelled")
                break
            except Exception as e:
                logger.error(f"Error in cleanup task: {e}", exc_info=True)
    
    cleanup_task = asyncio.create_task(periodic_cleanup())
    logger.info("Started background task for temporary file cleanup (runs every 15 minutes)")
    
    logger.info("=" * 60)
    logger.info("Virtual Try-On ML API is ready!")
    logger.info("=" * 60)
    
    yield
    
    # Shutdown
    logger.info("Shutting down Virtual Try-On ML API...")
    
    # Cancel cleanup task
    if cleanup_task:
        cleanup_task.cancel()
        try:
            await cleanup_task
        except asyncio.CancelledError:
            pass
        logger.info("Stopped background cleanup task")

app = FastAPI(
    title="Virtual Try-On ML API",
    description="Machine Learning API for Virtual Try-On - Recommendations, Processing, Body Generation",
    version="1.0.0",
    lifespan=lifespan
)

# CORS Configuration 
# SECURITY: Use explicit origins in production, never "*"
allowed_origins_str = os.getenv("ALLOWED_ORIGINS", "*")
origins = [origin.strip() for origin in allowed_origins_str.split(",")]

# Validate CORS configuration
allow_all_origins = "*" in origins
if allow_all_origins:
    logger.warning(
        "CORS configured to allow ALL origins (*). "
        "This is a SECURITY RISK in production! "
        "Set ALLOWED_ORIGINS environment variable to specific domains."
    )

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if allow_all_origins else origins,
    allow_credentials=not allow_all_origins,  # Can't use credentials with wildcard
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# Add custom middleware
app.add_middleware(RequestLoggingMiddleware)

# Register exception handlers
app.add_exception_handler(AppException, app_exception_handler)
app.add_exception_handler(RequestValidationError, validation_exception_handler)
app.add_exception_handler(StarletteHTTPException, http_exception_handler)
app.add_exception_handler(Exception, generic_exception_handler)

from app.api.endpoints import router as api_router
from app.api.body_generation import router as body_gen_router
from app.api.image_analysis import router as image_analysis_router
from app.api.image_composition import router as image_composition_router
from app.api.garment_management import router as garment_router
from app.api.identity_body import router as identity_body_router
from app.api.reconstruction_3d import router as reconstruction_3d_router

app.include_router(api_router, prefix="/api/v1")
app.include_router(body_gen_router, prefix="/api/v1")
app.include_router(image_analysis_router, prefix="/api/v1")
app.include_router(image_composition_router, prefix="/api/v1")
app.include_router(garment_router, prefix="/api/v1")
app.include_router(identity_body_router, prefix="/api/v1")
app.include_router(reconstruction_3d_router, prefix="/api/v1")

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
    ML service health check - GPU, AI services, Leffa, environment
    """
    from ml_engine.loader import model_loader, LEFFA_AVAILABLE, LEFFA_PATH
    
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
    
    # 2. Leffa Virtual Try-On Check
    leffa_status = {
        "available": LEFFA_AVAILABLE,
        "path": LEFFA_PATH,
        "loaded": "tryon" in model_loader.models
    }
    health_status["checks"]["leffa"] = leffa_status
    if not LEFFA_AVAILABLE:
        logger.warning("Leffa not available - virtual try-on will fail")
        health_status["status"] = "degraded"
    
    # 3. Gemini API Key Check (for recommendations)
    gemini_key = os.getenv("GEMINI_API_KEY")
    health_status["checks"]["ai_service"] = {
        "configured": bool(gemini_key),
        "masked_key": f"{gemini_key[:8]}..." if gemini_key else None
    }
    if not gemini_key:
        logger.warning("GEMINI_API_KEY not configured - recommendations will fail")
        health_status["status"] = "degraded"
    
    # 4. Environment Check
    health_status["checks"]["environment"] = {
        "log_level": os.getenv("LOG_LEVEL", "INFO"),
        "cors_origins": len(origins)
    }
    
    # 5. Memory usage
    health_status["checks"]["memory"] = model_loader.get_memory_usage()
    
    logger.info(f"Health check performed: {health_status['status']}")
    return health_status

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
