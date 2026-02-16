from fastapi import APIRouter, Form, HTTPException, UploadFile, File, Depends
from typing import List, Optional
import torch
from diffusers import AutoPipelineForText2Image
import io
import uuid
from datetime import datetime
from PIL import Image
import time
from ..core.logging_config import get_logger
from ..core.auth import get_current_user
from ..services.supabase_storage import supabase_storage
from ..services.body_generation import get_body_generation_service
from ..models.schemas import FullBodyGenerationRequest, FullBodyGenerationResponse
from ..core.file_validator import FileValidator

router = APIRouter()
logger = get_logger("api.body_generation")
file_validator = FileValidator(max_size_mb=10)

# Singleton SDXL pipeline
_sdxl_pipeline = None

def log_vram_usage(context=""):
    """Log current VRAM usage"""
    if not torch.cuda.is_available():
        return
    
    vram_allocated = torch.cuda.memory_allocated() / 1024**3
    vram_reserved = torch.cuda.memory_reserved() / 1024**3
    prefix = f"{context}: " if context else ""
    logger.info(f"{prefix}VRAM: {vram_allocated:.2f}GB allocated, {vram_reserved:.2f}GB reserved")

def get_sdxl_pipeline():
    """Initialize SDXL Turbo pipeline as singleton for fast inference."""
    global _sdxl_pipeline
    if _sdxl_pipeline is None:
        logger.info("Loading SDXL-Turbo model (first time)...")
        
        # CRITICAL: Clear GPU memory before loading
        if torch.cuda.is_available():
            logger.info("Clearing GPU memory before loading SDXL...")
            for _ in range(3):
                torch.cuda.empty_cache()
                import gc
                gc.collect()
            torch.cuda.synchronize()
            torch.cuda.ipc_collect()
            logger.info("✓ GPU memory cleared")
        
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
        
        # Enable aggressive memory optimizations for 4GB VRAM
        if torch.cuda.is_available():
            logger.info("Enabling memory optimizations for 4GB VRAM...")
            # Sequential CPU offloading - moves model components to CPU when not in use
            _sdxl_pipeline.enable_model_cpu_offload()
            # VAE tiling - processes images in tiles to reduce memory
            _sdxl_pipeline.enable_vae_tiling()
            # Enable attention slicing for lower memory usage
            _sdxl_pipeline.enable_attention_slicing(slice_size="auto")
            logger.info("✓ Memory optimizations enabled (CPU offload + VAE tiling + attention slicing)")
        
        logger.info("SDXL-Turbo loaded successfully")
    
    return _sdxl_pipeline

@router.post("/generate-bodies")
async def generate_bodies(
    ethnicity: str = Form(...),
    skin_tone: str = Form(...),
    body_type: str = Form(...),
    height_cm: float = Form(...),
    weight_kg: float = Form(...),
    user_id: str = Depends(get_current_user)
):
    """
    Generate 4 full-body images using SDXL-Turbo.
    ALL images stored in Supabase ONLY.
    Returns Supabase URLs.
    
    Authentication: Requires valid JWT token.
    """
    try:
        # Validation
        if height_cm < 50 or height_cm > 300:
            raise HTTPException(400, "Invalid height. Must be between 50-300 cm")
        if weight_kg < 20 or weight_kg > 500:
            raise HTTPException(400, "Invalid weight. Must be between 20-500 kg")
        
        logger.info(f"Generating bodies: ethnicity={ethnicity}, skin={skin_tone}, body={body_type}, height={height_cm}cm, weight={weight_kg}kg")
        
        # Log initial VRAM state
        log_vram_usage("Before body generation")
        
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
            
            # Cleanup after each image to prevent memory buildup
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                import gc
                gc.collect()
            
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
        
        # Final VRAM cleanup
        log_vram_usage("After body generation")
        
        return {
            "message": "Body generation complete",
            "request_id": request_id,
            "count": 4,
            "images": generated_images
        }
        
    except Exception as e:
        logger.error(f"Body generation failed: {e}", exc_info=True)
        raise HTTPException(500, f"Body generation failed: {str(e)}")

@router.post("/generate-full-body", response_model=FullBodyGenerationResponse)
async def generate_full_body_endpoint(
    partial_image: UploadFile = File(..., description="Partial body image (head/upper body)"),
    body_type: str = Form(..., description="Body type: slim, athletic, average, curvy, plus_size"),
    ethnicity: Optional[str] = Form(None, description="Ethnicity (optional, Gemini can infer)"),
    height_cm: Optional[float] = Form(None, description="Height in cm (140-220)"),
    weight_kg: Optional[float] = Form(None, description="Weight in kg (40-200)"),
    gender: Optional[str] = Form(None, description="Gender: male, female, non_binary, prefer_not_to_say"),
    pose: str = Form("standing", description="Desired pose"),
    clothing: str = Form("casual minimal clothing", description="Clothing description"),
    num_inference_steps: int = Form(4, description="Number of inference steps (1-50)"),
    guidance_scale: float = Form(0.0, description="Guidance scale (0.0-20.0)"),
    seed: int = Form(42, description="Random seed for reproducibility"),
    user_id: str = Depends(get_current_user)
):
    """
    Generate a full body image from a partial body image (head/upper body).
    
    Authentication: Requires valid JWT token.
    
    This endpoint:
    1. Analyzes the face using Gemini Vision
    2. Combines face analysis with user parameters
    3. Generates a full body image using SDXL
    4. Stores the result in Supabase storage
    5. Returns the public URL
    
    Requirements: 4.2, 4.5, 4.6, 4.7
    
    Args:
        partial_image: Image file containing head/upper body
        body_type: Body type (slim, athletic, average, curvy, plus_size)
        ethnicity: Optional ethnicity (Gemini can infer if not provided)
        height_cm: Optional height in cm (140-220)
        weight_kg: Optional weight in kg (40-200)
        gender: Optional gender
        pose: Desired pose (default: standing)
        clothing: Clothing description (default: casual minimal clothing)
        num_inference_steps: Number of diffusion steps (default: 4 for SDXL Turbo)
        guidance_scale: CFG scale (default: 0.0 for SDXL Turbo)
        seed: Random seed for reproducibility
        
    Returns:
        FullBodyGenerationResponse with:
        - result_url: Supabase URL to generated image
        - request_id: Unique request identifier
        - processing_time: Processing time in seconds
        - face_analysis: Face analysis results from Gemini
        - generation_prompt: Prompt used for generation
        - metadata: Additional metadata
    """
    start_time = time.time()
    request_id = str(uuid.uuid4())
    
    try:
        logger.info(f"Full body generation request {request_id}: body_type={body_type}")
        
        # Validate uploaded file
        await file_validator.validate_file(partial_image)
        
        # Load partial image
        partial_img_bytes = await partial_image.read()
        partial_img = Image.open(io.BytesIO(partial_img_bytes))
        
        logger.info(f"Loaded partial image: {partial_img.size}")
        
        # Build user parameters dictionary
        user_params = {
            "body_type": body_type,
            "pose": pose,
            "clothing": clothing,
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
            "seed": seed
        }
        
        # Add optional parameters if provided
        if ethnicity:
            user_params["ethnicity"] = ethnicity
        if height_cm is not None:
            user_params["height_cm"] = height_cm
        if weight_kg is not None:
            user_params["weight_kg"] = weight_kg
        if gender:
            user_params["gender"] = gender
        
        # Validate parameters using Pydantic
        try:
            validated_request = FullBodyGenerationRequest(**user_params)
        except Exception as e:
            logger.error(f"Parameter validation failed: {e}")
            raise HTTPException(400, f"Invalid parameters: {str(e)}")
        
        # Get body generation service
        body_gen_service = get_body_generation_service()
        
        # Generate full body image
        logger.info("Calling body generation service...")
        generation_result = await body_gen_service.generate_full_body(
            partial_image=partial_img,
            user_params=user_params
        )
        
        # Check if generation was successful
        if not generation_result.get("success"):
            error_msg = generation_result.get("error", "Unknown error")
            logger.error(f"Body generation failed: {error_msg}")
            raise HTTPException(500, f"Body generation failed: {error_msg}")
        
        generated_image = generation_result["generated_image"]
        face_analysis = generation_result.get("face_analysis", {})
        generation_prompt = generation_result.get("generation_prompt", "")
        
        logger.info("Full body image generated successfully")
        
        # Upload result to Supabase
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_path = f"body-generations/{request_id}/full_body_{timestamp}.png"
        
        logger.info(f"Uploading result to Supabase: {result_path}")
        result_url = supabase_storage.upload_image(
            image=generated_image,
            bucket=supabase_storage.GENERATED_BUCKET,
            path=result_path,
            max_size_mb=5.0,
            quality=85
        )
        
        logger.info(f"Result uploaded to Supabase: {result_url}")
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        # Build metadata
        metadata = {
            "body_type": body_type,
            "ethnicity": ethnicity,
            "height_cm": height_cm,
            "weight_kg": weight_kg,
            "gender": gender,
            "pose": pose,
            "clothing": clothing,
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
            "seed": seed,
            "partial_image_size": partial_img.size,
            "generated_image_size": generated_image.size
        }
        
        # Save to database (optional, for history tracking)
        try:
            # TODO: Create body_generations table if needed
            # For now, we just log the generation
            logger.info(f"Full body generation completed: {request_id}")
        except Exception as e:
            logger.warning(f"Failed to save to database: {e}")
            # Don't fail the request if database save fails
        
        logger.info(
            f"Full body generation complete in {processing_time:.2f}s: {request_id}"
        )
        
        # Return response
        return FullBodyGenerationResponse(
            result_url=result_url,
            request_id=request_id,
            processing_time=processing_time,
            face_analysis=face_analysis,
            generation_prompt=generation_prompt,
            metadata=metadata
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Full body generation failed: {e}", exc_info=True)
        raise HTTPException(500, f"Full body generation failed: {str(e)}")
