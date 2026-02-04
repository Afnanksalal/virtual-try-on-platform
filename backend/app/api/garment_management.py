"""
Garment Management API Endpoints

Provides endpoints for:
- Uploading garments to user-specific storage
- Listing user's garments
- Deleting garments

All operations use Supabase storage with proper user isolation.
"""

from fastapi import APIRouter, UploadFile, File, HTTPException, Path, Query
from typing import List
from app.services.supabase_storage import supabase_storage
from app.models.schemas import (
    GarmentUploadResponse,
    GarmentListResponse,
    GarmentDeleteResponse,
    GarmentMetadata
)
from app.core.logging_config import get_logger
from app.core.exceptions import (
    StorageErrorException,
    SupabaseUnavailableException,
    FileTooLargeException,
    InvalidMimeTypeException
)

logger = get_logger("api.garment_management")
router = APIRouter()


@router.post("/garments/upload", response_model=GarmentUploadResponse)
async def upload_garment(
    user_id: str = Query(..., min_length=1, max_length=100, description="User ID for storage isolation"),
    file: UploadFile = File(..., description="Garment image file (JPG, PNG, WEBP, max 10MB)")
):
    """
    Upload a garment image to user-specific storage.
    
    **Validation:**
    - File format: JPG, PNG, WEBP
    - File size: Maximum 10MB
    - User ID: Required for storage isolation
    
    **Storage:**
    - Bucket: user-garments
    - Path: {user_id}/garments/{unique_filename}
    - Unique filename prevents collisions
    
    **Returns:**
    - Garment metadata including ID, URL, and upload timestamp
    
    **Errors:**
    - 400: Invalid file format or size
    - 500: Storage operation failed
    - 503: Supabase unavailable
    """
    try:
        logger.info(f"Garment upload request from user {user_id}: {file.filename}")
        
        # Upload garment using storage service
        garment_metadata = await supabase_storage.upload_garment(
            user_id=user_id,
            file=file
        )
        
        # Convert to response model
        garment = GarmentMetadata(**garment_metadata)
        
        logger.info(f"Garment uploaded successfully for user {user_id}: {garment.id}")
        
        return GarmentUploadResponse(
            message="Garment uploaded successfully",
            garment=garment
        )
        
    except FileTooLargeException as e:
        logger.warning(f"File too large for user {user_id}: {e}")
        raise HTTPException(
            status_code=400,
            detail=f"File size exceeds the maximum limit of 10MB. Please upload a smaller image."
        )
    
    except InvalidMimeTypeException as e:
        logger.warning(f"Invalid file type for user {user_id}: {e}")
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file format. Please upload a JPG, PNG, or WEBP image."
        )
    
    except SupabaseUnavailableException as e:
        logger.error(f"Supabase unavailable for user {user_id}: {e}")
        raise HTTPException(
            status_code=503,
            detail="Storage service is temporarily unavailable. Please try again in a moment."
        )
    
    except StorageErrorException as e:
        logger.error(f"Storage error for user {user_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to upload garment: {str(e)}. Please try again."
        )
    
    except Exception as e:
        logger.error(f"Unexpected error uploading garment for user {user_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="An unexpected error occurred while uploading the garment. Please try again."
        )


@router.get("/garments/list", response_model=GarmentListResponse)
async def list_garments(
    user_id: str = Query(..., min_length=1, max_length=100, description="User ID for storage isolation")
):
    """
    List all garments for a specific user.
    
    **Data Isolation:**
    - Only returns garments belonging to the specified user
    - No cross-user data leakage
    
    **Returns:**
    - List of garment metadata (ID, URL, name, upload date, size)
    - Total count of garments
    
    **Errors:**
    - 500: Storage operation failed
    - 503: Supabase unavailable
    """
    try:
        logger.info(f"Listing garments for user {user_id}")
        
        # List garments using storage service
        garments_data = supabase_storage.list_garments(user_id=user_id)
        
        # Convert to response models
        garments = [GarmentMetadata(**g) for g in garments_data]
        
        logger.info(f"Found {len(garments)} garments for user {user_id}")
        
        return GarmentListResponse(
            garments=garments,
            count=len(garments),
            user_id=user_id
        )
        
    except SupabaseUnavailableException as e:
        logger.error(f"Supabase unavailable for user {user_id}: {e}")
        raise HTTPException(
            status_code=503,
            detail="Storage service is temporarily unavailable. Please try again in a moment."
        )
    
    except StorageErrorException as e:
        logger.error(f"Storage error for user {user_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to list garments: {str(e)}. Please try again."
        )
    
    except Exception as e:
        logger.error(f"Unexpected error listing garments for user {user_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="An unexpected error occurred while listing garments. Please try again."
        )


@router.delete("/garments/{garment_id}", response_model=GarmentDeleteResponse)
async def delete_garment(
    garment_id: str = Path(..., min_length=1, max_length=100, description="Garment ID to delete"),
    user_id: str = Query(..., min_length=1, max_length=100, description="User ID for storage isolation")
):
    """
    Delete a specific garment for a user.
    
    **Cleanup:**
    - Removes file from Supabase storage
    - Cleans up associated metadata
    - No orphaned data left behind
    
    **Data Isolation:**
    - Only deletes garments belonging to the specified user
    - Prevents unauthorized deletion of other users' garments
    
    **Returns:**
    - Confirmation message
    - Garment ID that was deleted
    - Deletion status
    
    **Errors:**
    - 404: Garment not found
    - 500: Storage operation failed
    """
    try:
        logger.info(f"Deleting garment {garment_id} for user {user_id}")
        
        # Delete garment using storage service
        deleted = supabase_storage.delete_garment(
            user_id=user_id,
            garment_id=garment_id
        )
        
        if not deleted:
            logger.warning(f"Garment {garment_id} not found for user {user_id}")
            raise HTTPException(
                status_code=404,
                detail=f"Garment not found. It may have already been deleted."
            )
        
        logger.info(f"Garment {garment_id} deleted successfully for user {user_id}")
        
        return GarmentDeleteResponse(
            message="Garment deleted successfully",
            garment_id=garment_id,
            deleted=True
        )
        
    except HTTPException:
        raise
    
    except StorageErrorException as e:
        logger.error(f"Storage error deleting garment {garment_id} for user {user_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to delete garment: {str(e)}. Please try again."
        )
    
    except Exception as e:
        logger.error(f"Unexpected error deleting garment {garment_id} for user {user_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="An unexpected error occurred while deleting the garment. Please try again."
        )
