from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Depends
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
from app.core.auth import get_current_user, get_optional_user

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
    style_preference: Optional[str] = Form(None),
    user_id: str = Depends(get_current_user)
):
    """
    Get AI-powered outfit recommendations using image collage + Gemini Vision + eBay search.
    
    Pipeline:
    1. Extract skin tone from user photo (if not provided)
    2. Fetch user's wardrobe and try-on history from database (if user_id provided)
    3. Create collage from user photo, wardrobe, generated images, and try-on history
    4. Gemini Vision analyzes with user profile data and color theory
    5. Extract keywords with scientific color theory based on skin tone
    6. Search eBay via RapidAPI
    7. Return products with buy links (up to 20 items)
    
    Args:
        user_photo: User's photo (used for skin tone extraction if not provided)
        user_id: Optional user ID to fetch wardrobe and try-on history from database
        wardrobe_images: Optional wardrobe items (if not fetching from database)
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
        
        # Load user photo
        user_img_bytes = await user_photo.read()
        user_img = Image.open(io.BytesIO(user_img_bytes))
        
        # Fetch wardrobe from database if user_id provided
        wardrobe_imgs = []
        if user_id:
            try:
                logger.info(f"Fetching wardrobe for user: {user_id}")
                wardrobe_records = supabase_storage.list_user_garments_db(user_id)
                
                # Download wardrobe images from URLs
                for record in wardrobe_records[:10]:  # Limit to 10 wardrobe items
                    try:
                        image_url = record.get('image_url')
                        if image_url:
                            # Extract bucket and path from URL
                            # URL format: https://{project}.supabase.co/storage/v1/object/public/{bucket}/{path}
                            parts = image_url.split('/storage/v1/object/public/')
                            if len(parts) == 2:
                                bucket_path = parts[1].split('/', 1)
                                if len(bucket_path) == 2:
                                    bucket, path = bucket_path
                                    wardrobe_img = supabase_storage.download_image(bucket, path)
                                    wardrobe_imgs.append(wardrobe_img)
                    except Exception as e:
                        logger.warning(f"Failed to download wardrobe image: {e}")
                        continue
                
                logger.info(f"Loaded {len(wardrobe_imgs)} wardrobe images from database")
            except Exception as e:
                logger.warning(f"Failed to fetch wardrobe from database: {e}")
        
        # If no wardrobe from database, use uploaded images
        if not wardrobe_imgs and wardrobe_images:
            logger.info("Using uploaded wardrobe images")
            for img_file in wardrobe_images:
                await file_validator.validate_file(img_file)
                img_bytes = await img_file.read()
                wardrobe_imgs.append(Image.open(io.BytesIO(img_bytes)))
        
        # Fetch try-on history from database if user_id provided
        tryon_history_imgs = []
        if user_id:
            try:
                logger.info(f"Fetching try-on history for user: {user_id}")
                tryon_records = supabase_storage.list_user_results_db(user_id)
                
                # Download try-on result images from URLs
                for record in tryon_records[:5]:  # Limit to 5 recent try-ons
                    try:
                        result_url = record.get('result_url')
                        if result_url:
                            # Extract bucket and path from URL
                            parts = result_url.split('/storage/v1/object/public/')
                            if len(parts) == 2:
                                bucket_path = parts[1].split('/', 1)
                                if len(bucket_path) == 2:
                                    bucket, path = bucket_path
                                    tryon_img = supabase_storage.download_image(bucket, path)
                                    tryon_history_imgs.append(tryon_img)
                    except Exception as e:
                        logger.warning(f"Failed to download try-on history image: {e}")
                        continue
                
                logger.info(f"Loaded {len(tryon_history_imgs)} try-on history images from database")
            except Exception as e:
                logger.warning(f"Failed to fetch try-on history from database: {e}")
        
        # Load generated images
        generated_imgs = []
        for img_file in generated_images:
            await file_validator.validate_file(img_file)
            img_bytes = await img_file.read()
            generated_imgs.append(Image.open(io.BytesIO(img_bytes)))
        
        # Combine all images for analysis
        all_analysis_images = wardrobe_imgs + tryon_history_imgs + generated_imgs
        
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
        logger.info(f"Analysis images: {len(wardrobe_imgs)} wardrobe, {len(tryon_history_imgs)} try-on history, {len(generated_imgs)} generated")
        
        # Get recommendations
        products = await recommendation_engine.get_outfit_recommendations(
            user_photo=user_img,
            wardrobe_images=all_analysis_images if all_analysis_images else None,
            generated_images=None,  # Already included in all_analysis_images
            user_profile=user_profile if user_profile else None,
            user_id=user_id
        )
        
        # Validate response structure
        if not isinstance(products, list):
            logger.error(f"Invalid recommendation response type: {type(products)}")
            raise ValueError("Recommendation engine returned invalid response format")
        
        # Validate each product has required fields
        required_fields = ['id', 'name', 'image_url', 'price', 'ebay_url']
        validated_products = []
        for product in products:
            if all(field in product for field in required_fields):
                validated_products.append(product)
            else:
                logger.warning(f"Product missing required fields: {product}")
        
        logger.info(f"Returning {len(validated_products)} validated recommendations")
        
        return {
            "recommendations": validated_products,
            "count": len(validated_products),
            "sources": {
                "wardrobe_count": len(wardrobe_imgs),
                "tryon_history_count": len(tryon_history_imgs),
                "generated_count": len(generated_imgs)
            }
        }
        
    except ValueError as e:
        logger.error(f"Validation error: {e}")
        raise HTTPException(
            status_code=400,
            detail={
                "error": {
                    "type": "user_error",
                    "code": "INVALID_INPUT",
                    "message": f"Invalid input: {str(e)}",
                    "details": {"reason": str(e)}
                }
            }
        )
    except Exception as e:
        logger.error(f"Recommendation failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail={
                "error": {
                    "type": "system_error",
                    "code": "RECOMMENDATION_FAILED",
                    "message": "Recommendation generation failed. Please try again.",
                    "details": {"reason": str(e)}
                }
            }
        )

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
    user_id: str = Depends(get_current_user)
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
        logger.info(f"[{request_id}] Progress: Uploading input images (10%)")
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
        
        logger.info(f"[{request_id}] Progress: Input images uploaded (20%)")
        logger.info(f"Uploaded input images to Supabase: {user_url}, {garment_url}")
        
        # Process try-on with Leffa
        logger.info(f"[{request_id}] Progress: Starting ML inference (30%)")
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
        
        logger.info(f"[{request_id}] Progress: ML inference complete (90%)")
        logger.info(f"Try-on processed in {result['processing_time']:.2f}s")
        
        # Save result to database for history
        try:
            # Use authenticated user_id from JWT token
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
        
        logger.info(f"[{request_id}] Progress: Complete (100%)")
        
        return {
            "message": "Virtual try-on processed successfully",
            "status": "success",
            **result
        }
        
    except ValueError as e:
        logger.error(f"Validation error: {e}")
        raise HTTPException(
            status_code=400,
            detail={
                "error": {
                    "type": "user_error",
                    "code": "INVALID_INPUT",
                    "message": f"Invalid input: {str(e)}",
                    "details": {"reason": str(e)}
                }
            }
        )
    except Exception as e:
        logger.error(f"Try-on failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail={
                "error": {
                    "type": "system_error",
                    "code": "TRYON_FAILED",
                    "message": "Virtual try-on processing failed. Please try again.",
                    "details": {"reason": str(e)}
                }
            }
        )

@router.post("/process-tryon-batch")
async def process_virtual_tryon_batch(
    user_image: UploadFile = File(...),
    garment_images: List[UploadFile] = File(...),
    garment_type: str = Form("upper_body"),
    num_inference_steps: int = Form(30),
    guidance_scale: float = Form(2.5),
    seed: int = Form(42),
    model_type: str = Form("viton_hd"),
    ref_acceleration: bool = Form(False),
    repaint: bool = Form(False),
    user_id: str = Depends(get_current_user)
):
    """
    Process batch virtual try-on with one person image and multiple garments.
    Processes all garments sequentially and returns all results.
    
    Args:
        user_image: User/person image file
        garment_images: List of garment image files
        garment_type: Type of garment - "upper_body", "lower_body", or "dresses"
        num_inference_steps: Number of diffusion steps (default: 30, range: 10-50)
        guidance_scale: CFG scale (default: 2.5, range: 1.0-5.0)
        seed: Random seed for reproducibility (default: 42)
        model_type: Model variant - "viton_hd" (recommended) or "dress_code"
        ref_acceleration: Speed up reference UNet, slight quality loss (default: False)
        repaint: Enable repaint mode for better edge handling (default: False)
    
    Returns:
        - batch_id: Unique batch identifier
        - total_count: Total number of garments processed
        - successful_count: Number of successful try-ons
        - failed_count: Number of failed try-ons
        - results: List of try-on results
        - total_processing_time: Total processing time in seconds
    """
    try:
        batch_id = str(uuid.uuid4())
        logger.info(f"[{batch_id}] Batch try-on request received with {len(garment_images)} garments")
        
        # Validate user image
        await file_validator.validate_file(user_image)
        
        # Validate all garment images
        for garment_img in garment_images:
            await file_validator.validate_file(garment_img)
        
        # Load user image once
        user_img_bytes = await user_image.read()
        user_img = Image.open(io.BytesIO(user_img_bytes))
        
        # Upload user image once
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        user_path = f"tryon/{batch_id}/user_{timestamp}.png"
        user_url = supabase_storage.upload_image(
            user_img,
            bucket=supabase_storage.UPLOADS_BUCKET,
            path=user_path
        )
        
        logger.info(f"[{batch_id}] User image uploaded: {user_url}")
        
        # Process each garment
        results = []
        successful_count = 0
        failed_count = 0
        start_time = datetime.now()
        
        for idx, garment_file in enumerate(garment_images, 1):
            try:
                logger.info(f"[{batch_id}] Progress: Processing garment {idx}/{len(garment_images)} ({int(idx/len(garment_images)*100)}%)")
                
                # Load garment image
                garment_img_bytes = await garment_file.read()
                garment_img = Image.open(io.BytesIO(garment_img_bytes))
                
                # Generate request ID for this try-on
                request_id = f"{batch_id}_{idx}"
                
                # Upload garment image
                garment_path = f"tryon/{batch_id}/garment_{idx}_{timestamp}.png"
                garment_url = supabase_storage.upload_image(
                    garment_img,
                    bucket=supabase_storage.UPLOADS_BUCKET,
                    path=garment_path
                )
                
                # Process try-on
                result = tryon_service.process_tryon(
                    person_image=user_img,
                    garment_image=garment_img,
                    request_id=request_id,
                    options={
                        "garment_type": garment_type,
                        "num_inference_steps": num_inference_steps,
                        "guidance_scale": guidance_scale,
                        "seed": seed,
                        "model_type": model_type,
                        "ref_acceleration": ref_acceleration,
                        "repaint": repaint,
                    }
                )
                
                # Save to database
                try:
                    # Use authenticated user_id from JWT token
                    db_record = supabase_storage.save_tryon_result_db(
                        user_id=user_id,
                        personal_image_url=user_url,
                        garment_url=garment_url,
                        result_url=result['result_url'],
                        metadata={
                            **result.get('metadata', {}),
                            'batch_id': batch_id,
                            'batch_index': idx
                        }
                    )
                except Exception as e:
                    logger.warning(f"Failed to save batch result to database: {e}")
                
                results.append({
                    "status": "success",
                    "garment_index": idx,
                    "garment_url": garment_url,
                    **result
                })
                successful_count += 1
                logger.info(f"[{batch_id}] Garment {idx} processed successfully")
                
            except Exception as e:
                logger.error(f"[{batch_id}] Failed to process garment {idx}: {e}", exc_info=True)
                results.append({
                    "status": "failed",
                    "garment_index": idx,
                    "error": {
                        "type": "system_error",
                        "code": "TRYON_FAILED",
                        "message": str(e)
                    }
                })
                failed_count += 1
        
        total_processing_time = (datetime.now() - start_time).total_seconds()
        
        logger.info(f"[{batch_id}] Batch complete: {successful_count} successful, {failed_count} failed, {total_processing_time:.2f}s total")
        
        return {
            "message": "Batch try-on processing complete",
            "batch_id": batch_id,
            "total_count": len(garment_images),
            "successful_count": successful_count,
            "failed_count": failed_count,
            "results": results,
            "total_processing_time": round(total_processing_time, 2)
        }
        
    except ValueError as e:
        logger.error(f"Batch validation error: {e}")
        raise HTTPException(
            status_code=400,
            detail={
                "error": {
                    "type": "user_error",
                    "code": "INVALID_INPUT",
                    "message": f"Invalid input: {str(e)}",
                    "details": {"reason": str(e)}
                }
            }
        )
    except Exception as e:
        logger.error(f"Batch try-on failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail={
                "error": {
                    "type": "system_error",
                    "code": "BATCH_TRYON_FAILED",
                    "message": "Batch virtual try-on processing failed. Please try again.",
                    "details": {"reason": str(e)}
                }
            }
        )

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
async def get_user_wardrobe(
    user_id: str,
    authenticated_user_id: str = Depends(get_current_user)
):
    """
    Get all wardrobe items (garments) for a user.
    Returns both storage files and database records.
    
    Security: Users can only access their own wardrobe.
    """
    # Enforce user data isolation - users can only access their own wardrobe
    if user_id != authenticated_user_id:
        logger.warning(f"User {authenticated_user_id} attempted to access wardrobe of user {user_id}")
        raise HTTPException(
            status_code=403,
            detail={
                "error": {
                    "type": "user_error",
                    "code": "FORBIDDEN",
                    "message": "You do not have permission to access this wardrobe.",
                    "details": {"reason": "User data isolation enforced"}
                }
            }
        )
    
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
async def get_tryon_history(
    user_id: str,
    limit: int = 50,
    authenticated_user_id: str = Depends(get_current_user)
):
    """
    Get try-on history for a user.
    Returns previous try-on results ordered by most recent first.
    
    Security: Users can only access their own history.
    """
    # Enforce user data isolation - users can only access their own history
    if user_id != authenticated_user_id:
        logger.warning(f"User {authenticated_user_id} attempted to access history of user {user_id}")
        raise HTTPException(
            status_code=403,
            detail={
                "error": {
                    "type": "user_error",
                    "code": "FORBIDDEN",
                    "message": "You do not have permission to access this history.",
                    "details": {"reason": "User data isolation enforced"}
                }
            }
        )
    
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
