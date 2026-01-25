from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse
from typing import List
from PIL import Image
import io
from pathlib import Path
import uuid
from datetime import datetime
from app.services.recommendation import recommendation_engine
from app.services.supabase_storage import supabase_storage
from app.core.logging_config import get_logger
from app.core.file_validator import FileValidator

logger = get_logger("api.endpoints")
router = APIRouter()

# Initialize file validator
file_validator = FileValidator(max_size_mb=10)

@router.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "service": "ml-api",
        "endpoints": ["health", "recommend", "process-tryon", "generate-body", "analyze-image", "combine-head-body", "results"]
    }

@router.get("/results/{filename}")
async def get_result_image(filename: str):
    """
    Get result image URL from Supabase storage.
    
    Args:
        filename: Name of the result file
        
    Returns:
        Public URL to the image in Supabase
    """
    try:
        # Validate filename to prevent path traversal
        if ".." in filename or "/" in filename or "\\" in filename:
            raise HTTPException(400, "Invalid filename")
        
        # Get public URL from Supabase
        public_url = supabase_storage.get_public_url(
            bucket=supabase_storage.RESULTS_BUCKET,
            path=filename
        )
        
        return {
            "url": public_url,
            "filename": filename
        }
        
    except Exception as e:
        logger.error(f"Failed to get result URL: {e}", exc_info=True)
        raise HTTPException(500, f"Failed to get result: {str(e)}")

@router.post("/recommend")
async def get_recommendations(
    user_photo: UploadFile = File(...),
    wardrobe_images: List[UploadFile] = File(default=[]),
    generated_images: List[UploadFile] = File(default=[])
):
    """
    Get AI-powered outfit recommendations using image collage + Gemini Vision + eBay search.
    
    Pipeline:
    1. Create collage from user photo, wardrobe, generated images
    2. Gemini Vision extracts keywords with color theory
    3. Search eBay via RapidAPI
    4. Return products with buy links
    """
    try:
        # Validate user photo
        await file_validator.validate_file(user_photo)
        
        # Validate wardrobe images
        for img_file in wardrobe_images:
            await file_validator.validate_file(img_file)
        
        # Validate generated images
        for img_file in generated_images:
            await file_validator.validate_file(img_file)
        
        # Load user photo
        user_img_bytes = await user_photo.read()
        user_img = Image.open(io.BytesIO(user_img_bytes))
        
        # Load wardrobe images
        wardrobe_imgs = []
        for img_file in wardrobe_images:
            img_bytes = await img_file.read()
            wardrobe_imgs.append(Image.open(io.BytesIO(img_bytes)))
        
        # Load generated images
        generated_imgs = []
        for img_file in generated_images:
            img_bytes = await img_file.read()
            generated_imgs.append(Image.open(io.BytesIO(img_bytes)))
        
        # Get recommendations
        products = await recommendation_engine.get_outfit_recommendations(
            user_photo=user_img,
            wardrobe_images=wardrobe_imgs if wardrobe_imgs else None,
            generated_images=generated_imgs if generated_imgs else None
        )
        
        return products
        
    except Exception as e:
        logger.error(f"Recommendation failed: {e}", exc_info=True)
        raise HTTPException(500, f"Recommendation failed: {str(e)}")

@router.post("/process-tryon")
async def process_virtual_tryon(
    user_image: UploadFile = File(...),
    garment_image: UploadFile = File(...),
):
    """
    Process virtual try-on using IDM-VTON model.
    ALL files stored in Supabase ONLY.
    
    Returns:
        - request_id: Unique request identifier
        - result_url: Supabase URL to result image
        - processing_time: Processing time in seconds
        - metadata: Additional processing metadata
    """
    try:
        logger.info("Virtual try-on request received")
        
        # Validate uploaded files
        await file_validator.validate_file(user_image)
        await file_validator.validate_file(garment_image)
        
        # Load images
        user_img_bytes = await user_image.read()
        garment_img_bytes = await garment_image.read()
        
        user_img = Image.open(io.BytesIO(user_img_bytes))
        garment_img = Image.open(io.BytesIO(garment_img_bytes))
        
        # Generate unique request ID
        request_id = str(uuid.uuid4())
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Upload input images to Supabase
        user_path = f"tryon/{request_id}/user_{timestamp}.png"
        garment_path = f"tryon/{request_id}/garment_{timestamp}.png"
        
        user_url = supabase_storage.upload_image(
            user_img,
            bucket=supabase_storage.UPLOADS_BUCKET,
            path=user_path
        )
        
        garment_url = supabase_storage.upload_image(
            garment_img,
            bucket=supabase_storage.UPLOADS_BUCKET,
            path=garment_path
        )
        
        logger.info(f"Uploaded input images to Supabase: {user_url}, {garment_url}")
        
        # Import try-on service
        from app.services.tryon_service import tryon_service
        
        # Process try-on
        result = tryon_service.process_tryon(
            person_image=user_img,
            garment_image=garment_img,
            request_id=request_id,
            options={
                "target_size": (512, 768),
                "num_inference_steps": 30,
                "guidance_scale": 7.5,
            }
        )
        
        logger.info(f"Try-on processed in {result['processing_time']:.2f}s")
        
        return {
            "message": "Virtual try-on processed successfully",
            "status": "success",
            **result
        }
        
    except ValueError as e:
        logger.error(f"Validation error: {e}")
        raise HTTPException(400, f"Invalid input: {str(e)}")
    except Exception as e:
        logger.error(f"Try-on failed: {e}", exc_info=True)
        raise HTTPException(500, f"Virtual try-on failed: {str(e)}")

@router.post("/generate-body")
async def generate_body_endpoint(
    ethnicity: str = Form(...),
    height_cm: float = Form(...),
    weight_kg: float = Form(...),
    body_type: str = Form(...),
    count: int = Form(default=4)
):
    """
    Generate body model variations.
    Note: This endpoint exists but /generate-bodies is the primary one used.
    """
    try:
        logger.info(f"Body generation: {body_type}, {ethnicity}")
        
        return {
            "message": "Body generation complete",
            "count": count
        }
        
    except Exception as e:
        logger.error(f"Body generation failed: {e}", exc_info=True)
        raise HTTPException(500, f"Body generation failed: {str(e)}")
