from fastapi import APIRouter, Form, HTTPException
from typing import List
import torch
from diffusers import AutoPipelineForText2Image
import io
import base64
from PIL import Image
from ..core.logging_config import get_logger

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
    Generate 4 full-body images using SDXL-Turbo based on user parameters.
    Returns base64-encoded PNG images.
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
            
            # Convert to base64
            buffered = io.BytesIO()
            image.save(buffered, format="PNG", quality=95)
            img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
            
            generated_images.append({
                "id": f"body_{i}",
                "data": f"data:image/png;base64,{img_base64}"
            })
            
            logger.info(f"Body variant {i+1}/4 generated successfully")
        
        logger.info("All 4 body variants generated successfully")
        
        return {
            "message": "Body generation complete",
            "count": 4,
            "images": generated_images
        }
        
    except Exception as e:
        logger.error(f"Body generation failed: {e}", exc_info=True)
        raise HTTPException(500, f"Body generation failed: {str(e)}")
