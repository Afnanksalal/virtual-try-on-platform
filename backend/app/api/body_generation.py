from fastapi import APIRouter, Form, HTTPException
from typing import List
import torch
from diffusers import AutoPipelineForText2Image
import io
import uuid
from datetime import datetime
from PIL import Image
from ..core.logging_config import get_logger
from ..services.supabase_storage import supabase_storage

router = APIRouter()
logger = get_logger("api.body_generation")

# Singleton SDXL pipeline
_sdxl_pipeline = None

def get_sdxl_pipeline():
    """Initialize SDXL Turbo pipeline as singleton for fast inference."""
    global _sdxl_pipeline
    if _sdxl_pipeline is None:
        logger.info("Loading SDXL-Turbo model (first time)...")
        
        # Use SDXL-Turbo for fast generation (1-4 steps)
        _sdxl_pipeline = AutoPipelineForText2Image.from_pretrained(
            "stabilityai/sdxl-turbo",
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            variant="fp16" if torch.cuda.is_available() else None
        )
        
        # Move to GPU if available
        if torch.cuda.is_available():
            _sdxl_pipeline = _sdxl_pipeline.to("cuda")
            logger.info("SDXL loaded on GPU")
        else:
            logger.warning("GPU not available, using CPU (will be slower)")
        
        # Enable memory optimizations
        if torch.cuda.is_available():
            _sdxl_pipeline.enable_model_cpu_offload()
        
        logger.info("SDXL-Turbo loaded successfully")
    
    return _sdxl_pipeline

@router.post("/generate-bodies")
async def generate_bodies(
    ethnicity: str = Form(...),
    skin_tone: str = Form(...),
    body_type: str = Form(...),
    height_cm: float = Form(...),
    weight_kg: float = Form(...)
):
    """
    Generate 4 full-body images using SDXL-Turbo.
    ALL images stored in Supabase ONLY.
    Returns Supabase URLs.
    """
    try:
        # Validation
        if height_cm < 50 or height_cm > 300:
            raise HTTPException(400, "Invalid height. Must be between 50-300 cm")
        if weight_kg < 20 or weight_kg > 500:
            raise HTTPException(400, "Invalid weight. Must be between 20-500 kg")
        
        logger.info(f"Generating bodies: ethnicity={ethnicity}, skin={skin_tone}, body={body_type}, height={height_cm}cm, weight={weight_kg}kg")
        
        # Get SDXL pipeline
        pipe = get_sdxl_pipeline()
        
        # Map body type to descriptive terms
        body_descriptors = {
            "athletic": "athletic build, toned muscles, fit physique",
            "slim": "slim build, lean physique, slender body",
            "muscular": "muscular build, strong physique, well-defined muscles",
            "average": "average build, normal proportions, medium physique"
        }
        
        body_desc = body_descriptors.get(body_type.lower(), "average build")
        
        # Map skin tone
        skin_descriptors = {
            "fair": "fair skin",
            "light": "light skin",
            "medium": "medium skin tone",
            "olive": "olive skin",
            "tan": "tan skin",
            "dark": "dark skin",
            "deep": "deep skin tone"
        }
        
        skin_desc = skin_descriptors.get(skin_tone.lower(), "medium skin tone")
        
        # Generate unique request ID
        request_id = str(uuid.uuid4())
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Generate 4 variations
        generated_images = []
        
        for i in range(4):
            # Create detailed prompt
            prompt = f"""full body portrait photo of a person, {ethnicity} ethnicity, {skin_desc}, {body_desc}, 
standing straight, neutral pose, arms at sides, facing camera, plain white background, 
professional photography, high quality, well-lit, realistic, 8k, ultra detailed"""
            
            # Negative prompt to avoid unwanted elements
            negative_prompt = """cropped, cut off, sitting, partial body, face only, headshot, 
multiple people, cluttered background, low quality, blurry, distorted, deformed"""
            
            logger.info(f"Generating body variant {i+1}/4...")
            
            # Generate with SDXL-Turbo (fast, 1-4 steps)
            image = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=4,  # SDXL-Turbo optimized for 1-4 steps
                guidance_scale=0.0,  # SDXL-Turbo doesn't use guidance
                height=768,
                width=512,
            ).images[0]
            
            # Upload to Supabase
            path = f"generated/{request_id}/body_{i}_{timestamp}.png"
            public_url = supabase_storage.upload_image(
                image,
                bucket=supabase_storage.GENERATED_BUCKET,
                path=path
            )
            
            generated_images.append({
                "id": f"body_{i}",
                "url": public_url
            })
            
            logger.info(f"Body variant {i+1}/4 uploaded to Supabase: {public_url}")
        
        logger.info("All 4 body variants generated and uploaded to Supabase")
        
        return {
            "message": "Body generation complete",
            "request_id": request_id,
            "count": 4,
            "images": generated_images
        }
        
    except Exception as e:
        logger.error(f"Body generation failed: {e}", exc_info=True)
        raise HTTPException(500, f"Body generation failed: {str(e)}")
