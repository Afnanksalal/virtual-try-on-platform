from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse
from typing import List, Optional
from PIL import Image
import io
from pathlib import Path
import uuid
from datetime import datetime
from app.services.recommendation import recommendation_engine
from app.services.supabase_storage import supabase_storage
from app.services.tryon_service import tryon_service
from app.core.logging_config import get_logger
from app.core.file_validator import FileValidator

logger = get_logger("api.endpoints")
router = APIRouter()

# Initialize file validator
file_validator = FileValidator(max_size_mb=10)

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
    generated_images: List[UploadFile] = File(default=[]),
    height_cm: Optional[float] = Form(None),
    weight_kg: Optional[float] = Form(None),
    body_type: Optional[str] = Form(None),
    ethnicity: Optional[str] = Form(None),
    gender: Optional[str] = Form(None),
    skin_tone: Optional[str] = Form(None),
    style_preference: Optional[str] = Form(None)
):
    """
    Get AI-powered outfit recommendations using image collage + Gemini Vision + eBay search.
    
    Pipeline:
    1. Extract skin tone from user photo (if not provided)
    2. Create collage from user photo, wardrobe, generated images
    3. Gemini Vision analyzes with user profile data and color theory
    4. Extract keywords with scientific color theory based on skin tone
    5. Search eBay via RapidAPI
    6. Return products with buy links (up to 20 items)
    
    Args:
        user_photo: User's photo (used for skin tone extraction if not provided)
        wardrobe_images: Optional wardrobe items
        generated_images: Optional generated body images
        height_cm: User height in cm (140-220)
        weight_kg: User weight in kg (40-200)
        body_type: Body type (slim, athletic, average, curvy, plus_size)
        ethnicity: Ethnicity for cultural style preferences
        gender: Gender for style preferences
        skin_tone: Skin tone category (fair_cool, fair_warm, light_cool, etc.)
        style_preference: User's style preferences (casual, formal, sporty, etc.)
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
        
        # Build user profile dictionary
        user_profile = {}
        if height_cm is not None:
            user_profile['height_cm'] = height_cm
        if weight_kg is not None:
            user_profile['weight_kg'] = weight_kg
        if body_type:
            user_profile['body_type'] = body_type
        if ethnicity:
            user_profile['ethnicity'] = ethnicity
        if gender:
            user_profile['gender'] = gender
        if skin_tone:
            user_profile['skin_tone'] = skin_tone
        if style_preference:
            user_profile['style_preference'] = style_preference
        
        logger.info(f"Recommendation request with user profile: {user_profile}")
        
        # Get recommendations
        products = await recommendation_engine.get_outfit_recommendations(
            user_photo=user_img,
            wardrobe_images=wardrobe_imgs if wardrobe_imgs else None,
            generated_images=generated_imgs if generated_imgs else None,
            user_profile=user_profile if user_profile else None
        )
        
        logger.info(f"Returning {len(products)} recommendations")
        return products
        
    except Exception as e:
        logger.error(f"Recommendation failed: {e}", exc_info=True)
        raise HTTPException(500, f"Recommendation failed: {str(e)}")

@router.post("/process-tryon")
async def process_virtual_tryon(
    user_image: UploadFile = File(...),
    garment_image: UploadFile = File(...),
    garment_type: str = Form("upper_body"),  # "upper_body", "lower_body", or "dresses"
    num_inference_steps: int = Form(30),
    guidance_scale: float = Form(2.5),
    seed: int = Form(42),
    model_type: str = Form("viton_hd"),  # "viton_hd" or "dress_code"
    ref_acceleration: bool = Form(False),  # Speed up reference UNet (slight quality loss)
    repaint: bool = Form(False),  # Enable repaint mode for better edge handling
):
    """
    Process virtual try-on using Leffa model with advanced options.
    ALL files stored in Supabase ONLY.
    
    Args:
        user_image: User/person image file
        garment_image: Garment image file
        garment_type: Type of garment - "upper_body", "lower_body", or "dresses"
        num_inference_steps: Number of diffusion steps (default: 30, range: 10-50)
        guidance_scale: CFG scale (default: 2.5, range: 1.0-5.0)
        seed: Random seed for reproducibility (default: 42)
        model_type: Model variant - "viton_hd" (recommended) or "dress_code"
        ref_acceleration: Speed up reference UNet, slight quality loss (default: False)
        repaint: Enable repaint mode for better edge handling (default: False)
    
    Returns:
        - request_id: Unique request identifier
        - result_url: Supabase URL to result image
        - processing_time: Processing time in seconds
        - metadata: Additional processing metadata including all options used
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
        
        # Process try-on with Leffa
        result = tryon_service.process_tryon(
            person_image=user_img,
            garment_image=garment_img,
            request_id=request_id,
            options={
                "garment_type": garment_type,  # "upper_body", "lower_body", or "dresses"
                "num_inference_steps": num_inference_steps,
                "guidance_scale": guidance_scale,
                "seed": seed,
                "model_type": model_type,  # "viton_hd" or "dress_code"
                "ref_acceleration": ref_acceleration,
                "repaint": repaint,
            }
        )
        
        logger.info(f"Try-on processed in {result['processing_time']:.2f}s")
        
        # Save result to database for history
        try:
            # Extract user_id from request headers or authentication
            # For now, using request_id as fallback until auth middleware is implemented
            user_id = request_id  # Will be replaced with actual user_id from JWT token
            
            db_record = supabase_storage.save_tryon_result_db(
                user_id=user_id,
                personal_image_url=user_url,
                garment_url=garment_url,
                result_url=result['result_url'],
                metadata=result.get('metadata', {})
            )
            if db_record:
                logger.info(f"Try-on result saved to database: {db_record.get('id')}")
        except Exception as e:
            logger.warning(f"Failed to save try-on result to database: {e}")
            # Don't fail the request if database save fails
        
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

@router.get("/wardrobe/{user_id}")
async def get_user_wardrobe(user_id: str):
    """
    Get all wardrobe items (garments) for a user.
    Returns both storage files and database records.
    """
    try:
        logger.info(f"Fetching wardrobe for user: {user_id}")
        
        # Get garments from storage
        storage_garments = supabase_storage.list_garments(user_id)
        
        # Get garments from database (if table exists)
        db_garments = supabase_storage.list_user_garments_db(user_id)
        
        # Merge results (prefer database records if they exist)
        garments_map = {}
        
        # Add storage garments first
        for garment in storage_garments:
            garments_map[garment['id']] = garment
        
        # Override/add database garments
        for garment in db_garments:
            garment_id = garment.get('id', '')
            garments_map[garment_id] = garment
        
        # Convert to list
        all_garments = list(garments_map.values())
        
        logger.info(f"Found {len(all_garments)} wardrobe items for user {user_id}")
        
        return {
            "items": all_garments,
            "count": len(all_garments)
        }
        
    except Exception as e:
        logger.error(f"Failed to fetch wardrobe: {e}", exc_info=True)
        raise HTTPException(500, f"Failed to fetch wardrobe: {str(e)}")

@router.get("/tryon/history/{user_id}")
async def get_tryon_history(user_id: str, limit: int = 50):
    """
    Get try-on history for a user.
    Returns previous try-on results ordered by most recent first.
    """
    try:
        logger.info(f"Fetching try-on history for user: {user_id}")
        
        # Get results from database
        results = supabase_storage.list_user_results_db(user_id)
        
        # Limit results
        results = results[:limit]
        
        logger.info(f"Found {len(results)} try-on results for user {user_id}")
        
        return {
            "results": results,
            "count": len(results)
        }
        
    except Exception as e:
        logger.error(f"Failed to fetch try-on history: {e}", exc_info=True)
        raise HTTPException(500, f"Failed to fetch history: {str(e)}")
