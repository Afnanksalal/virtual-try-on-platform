"""
Identity-Preserving Body Generation API

This module provides endpoints for generating full-body images that preserve
the user's facial identity using InstantID + Gemini Vision analysis.

The workflow:
1. User uploads head-only photo
2. System shows body type reference images (SDXL previews)
3. User selects preferred body type
4. Gemini analyzes user's facial features (skin tone, ethnicity, etc.)
5. InstantID generates full-body images preserving user's actual face

This replaces the problematic cut-and-paste approach with native face generation.
"""

from fastapi import APIRouter, Form, File, UploadFile, HTTPException
from typing import List, Optional
import io
import uuid
from datetime import datetime
from PIL import Image

from ..core.logging_config import get_logger
from ..core.file_validator import FileValidator
from ..services.supabase_storage import supabase_storage
from ..services.face_analysis import get_face_analysis_service

router = APIRouter()
logger = get_logger("api.identity_body")

# File validator
file_validator = FileValidator(max_size_mb=10)

# Lazy import for ML pipeline
_identity_pipeline = None


def get_identity_pipeline():
    """Lazy load identity-preserving pipeline."""
    global _identity_pipeline
    if _identity_pipeline is None:
        try:
            from ml_engine.pipelines.identity_preserving import get_identity_pipeline as get_pipeline
            _identity_pipeline = get_pipeline()
        except ImportError as e:
            logger.error(f"Failed to import identity pipeline: {e}")
            raise HTTPException(500, "ML pipeline unavailable")
    return _identity_pipeline


@router.post("/analyze-face-features")
async def analyze_face_features(image: UploadFile = File(...)):
    """
    Analyze facial features from uploaded image using Gemini Vision.
    
    This provides detailed information about skin tone, ethnicity, etc.
    that will guide the body generation for accurate results.
    
    Returns:
        Dict with facial analysis including skin_tone, ethnicity, age_range, etc.
    """
    try:
        # Validate file
        await file_validator.validate_file(image)
        
        # Read image
        image_bytes = await image.read()
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        
        # Analyze with Gemini
        service = get_face_analysis_service()
        analysis = service.analyze_face(img)
        
        logger.info(f"Face analysis completed for upload")
        
        return {
            "success": True,
            "analysis": analysis
        }
        
    except Exception as e:
        logger.error(f"Face analysis failed: {e}", exc_info=True)
        raise HTTPException(500, f"Face analysis failed: {str(e)}")


@router.post("/generate-identity-body")
async def generate_identity_body(
    face_image: UploadFile = File(...),
    body_type: str = Form("average"),
    height_cm: float = Form(170.0),
    weight_kg: float = Form(65.0),
    gender: str = Form("female"),
    ethnicity: str = Form(None),
    skin_tone: str = Form(None),
    num_images: int = Form(4),
    use_gemini_analysis: bool = Form(True),
):
    """
    Generate full-body images that preserve the user's facial identity.
    
    This uses InstantID to generate bodies with the user's actual face
    natively generated (not stitched), resulting in natural-looking images.
    
    Args:
        face_image: User's face photo (head-only or portrait)
        body_type: Desired body type (athletic, slim, muscular, average, curvy)
        height_cm: Height reference
        weight_kg: Weight reference
        gender: Gender for body generation
        ethnicity: Optional ethnicity override (Gemini will detect if not provided)
        skin_tone: Optional skin tone override (Gemini will detect if not provided)
        num_images: Number of variations to generate (1-4)
        use_gemini_analysis: Whether to use Gemini for facial analysis
    
    Returns:
        Dict with:
        - request_id: Unique identifier
        - images: List of generated image URLs
        - analysis: Gemini facial analysis (if enabled)
    """
    try:
        # Validate file
        await file_validator.validate_file(face_image)
        
        # Read image
        image_bytes = await face_image.read()
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        
        # Validate parameters
        num_images = max(1, min(4, num_images))
        if height_cm < 50 or height_cm > 300:
            raise HTTPException(400, "Invalid height. Must be between 50-300 cm")
        if weight_kg < 20 or weight_kg > 500:
            raise HTTPException(400, "Invalid weight. Must be between 20-500 kg")
        
        logger.info(f"Generating identity-preserving body: body_type={body_type}, "
                   f"gender={gender}, num_images={num_images}")
        
        # Step 1: Gemini facial analysis (optional but recommended)
        gemini_analysis = None
        if use_gemini_analysis:
            try:
                service = get_face_analysis_service()
                gemini_analysis = service.analyze_face(img)
                logger.info(f"Gemini analysis: skin={gemini_analysis.get('skin_tone_category')}, "
                           f"ethnicity={gemini_analysis.get('ethnicity')}")
            except Exception as e:
                logger.warning(f"Gemini analysis failed, continuing without it: {e}")
        
        # Build body params (user input + Gemini overrides)
        body_params = {
            "body_type": body_type,
            "height_cm": height_cm,
            "weight_kg": weight_kg,
            "gender": gender,
            "ethnicity": ethnicity,
            "skin_tone": skin_tone,
        }
        
        # Override with Gemini analysis if available and user didn't specify
        if gemini_analysis:
            if not ethnicity:
                body_params["ethnicity"] = gemini_analysis.get("ethnicity", "mixed heritage")
            if not skin_tone:
                body_params["skin_tone"] = gemini_analysis.get("skin_tone", "medium skin tone")
        
        # Step 2: Generate with InstantID (primary) or SDXL fallback
        request_id = str(uuid.uuid4())
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        use_fallback = False
        
        try:
            pipeline = get_identity_pipeline()
            result = pipeline(
                face_image=img,
                body_params=body_params,
                gemini_analysis=gemini_analysis,
                num_images=num_images,
            )
            
            generated_images = result["images"]
        except Exception as e:
            logger.warning(f"InstantID not available, using SDXL fallback: {e}")
            use_fallback = True
            
            # Fallback: Use improved SDXL generation with Gemini-analyzed prompts
            try:
                from .body_generation import get_sdxl_pipeline
                
                pipe = get_sdxl_pipeline()
                
                # Build improved prompt using Gemini analysis
                face_service = get_face_analysis_service()
                prompt = face_service.build_generation_prompt(
                    analysis=gemini_analysis or {},
                    body_params=body_params,
                    pose="standing, arms at sides, facing camera",
                    clothing="casual minimal clothing"
                )
                negative_prompt = face_service.get_negative_prompt()
                
                generated_images = []
                for i in range(num_images):
                    logger.info(f"SDXL fallback: generating image {i+1}/{num_images}")
                    image = pipe(
                        prompt=prompt,
                        negative_prompt=negative_prompt,
                        num_inference_steps=4,  # SDXL-Turbo
                        guidance_scale=0.0,
                        height=768,
                        width=512,
                    ).images[0]
                    generated_images.append(image)
                    
            except Exception as fallback_error:
                logger.error(f"SDXL fallback also failed: {fallback_error}", exc_info=True)
                raise HTTPException(500, f"Body generation failed: {str(e)}. Fallback also failed: {str(fallback_error)}")
        
        # Step 3: Upload generated images to Supabase
        uploaded_images = []
        for i, img_result in enumerate(generated_images):
            path = f"identity-body/{request_id}/body_{i}_{timestamp}.png"
            
            public_url = supabase_storage.upload_image(
                img_result,
                bucket=supabase_storage.GENERATED_BUCKET,
                path=path
            )
            
            uploaded_images.append({
                "id": f"identity_body_{i}",
                "url": public_url
            })
            
            logger.info(f"Generated body {i+1}/{num_images} uploaded: {public_url}")
        
        logger.info(f"Identity-preserving body generation complete: {len(uploaded_images)} images")
        
        return {
            "success": True,
            "request_id": request_id,
            "count": len(uploaded_images),
            "images": uploaded_images,
            "analysis": gemini_analysis,
            "params_used": result.get("params_used", body_params) if not use_fallback else body_params,
            "method": "sdxl_fallback" if use_fallback else "instantid",
            "note": "Used SDXL fallback (InstantID not available). Results may require head stitching." if use_fallback else None
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Identity body generation failed: {e}", exc_info=True)
        raise HTTPException(500, f"Body generation failed: {str(e)}")


@router.post("/preview-body-types")
async def preview_body_types():
    """
    Get reference images for different body types.
    
    These are pre-generated SDXL images showing body type options
    that the user can select from before identity-preserving generation.
    
    Returns:
        List of body type options with reference image URLs
    """
    # These are static reference images - could be pre-generated and stored
    body_types = [
        {
            "id": "athletic",
            "label": "Athletic",
            "description": "Toned muscles, fit physique",
            "preview_url": None,  # Can add pre-generated previews
        },
        {
            "id": "slim",
            "label": "Slim",
            "description": "Lean, slender build",
            "preview_url": None,
        },
        {
            "id": "muscular",
            "label": "Muscular",
            "description": "Strong, well-defined muscles",
            "preview_url": None,
        },
        {
            "id": "average",
            "label": "Average",
            "description": "Normal proportions, medium build",
            "preview_url": None,
        },
        {
            "id": "curvy",
            "label": "Curvy",
            "description": "Proportionate curves",
            "preview_url": None,
        },
    ]
    
    return {
        "body_types": body_types,
        "note": "Select your preferred body type for personalized generation"
    }
