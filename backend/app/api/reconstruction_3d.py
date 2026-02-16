"""
3D Reconstruction API Endpoints

Provides endpoints for generating 3D models from 2D images using the
TripoSR pipeline with SAM2 segmentation and Depth Anything V2.

Endpoints:
- POST /api/v1/generate-3d: Generate 3D mesh from uploaded image or URL
- GET /api/v1/download-3d/{token}: Download generated 3D model

Features:
- Temporary file storage with secure token-based access
- Automatic file expiration (1 hour default)
- Support for GLB, OBJ, and PLY formats
- Memory-optimized for 4GB VRAM (RTX 3050)
"""

import time
import uuid
from datetime import datetime, timedelta
from typing import Optional
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Path as FastAPIPath, Depends
from fastapi.responses import Response
from PIL import Image
import io
import httpx

from app.core.logging_config import get_logger
from app.core.auth import get_current_user
from app.models.schemas import (
    ThreeDGenerationRequest,
    ThreeDGenerationResponse,
    ErrorResponse
)
from app.services.temp_file_manager import TempFileManager
from ml_engine.pipelines.reconstruction_3d import ThreeDReconstructionPipeline

logger = get_logger("api.reconstruction_3d")
router = APIRouter(tags=["3D Reconstruction"])

# Initialize services
temp_file_manager = TempFileManager()
reconstruction_pipeline: Optional[ThreeDReconstructionPipeline] = None


def get_reconstruction_pipeline() -> ThreeDReconstructionPipeline:
    """Lazy initialization of 3D reconstruction pipeline."""
    global reconstruction_pipeline
    if reconstruction_pipeline is None:
        logger.info("Initializing 3D reconstruction pipeline...")
        reconstruction_pipeline = ThreeDReconstructionPipeline(device="cuda")
        logger.info("3D reconstruction pipeline initialized")
    return reconstruction_pipeline


@router.post(
    "/generate-3d",
    response_model=ThreeDGenerationResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid input"},
        500: {"model": ErrorResponse, "description": "Processing failed"}
    },
    summary="Generate 3D model from image",
    description="""
    Generate a 3D mesh from a 2D image using the TripoSR pipeline.
    
    **Pipeline stages:**
    1. Segmentation (SAM2) - optional, recommended for better results
    2. 3D reconstruction (TripoSR)
    
    **Output formats:**
    - GLB: Binary glTF format (recommended, smallest file size)
    - OBJ: Wavefront OBJ format (widely supported)
    - PLY: Polygon File Format (good for point clouds)
    
    **Memory optimization:**
    - Default resolution (256) optimized for 4GB VRAM
    - Higher resolutions (512) require 8GB+ VRAM
    
    **File expiration:**
    - Generated files expire after 1 hour
    - Download via /download-3d/{token} endpoint
    """
)
async def generate_3d(
    image: Optional[UploadFile] = File(None, description="Input image (JPEG, PNG, WebP)"),
    tryon_result_url: Optional[str] = Form(None, description="URL to try-on result image"),
    output_format: str = Form("glb", description="Output format (glb, obj, ply)"),
    use_segmentation: bool = Form(True, description="Use SAM2 for background removal"),
    mc_resolution: int = Form(256, description="Marching cubes resolution (128-512)"),
    user_id: str = Depends(get_current_user)
) -> ThreeDGenerationResponse:
    """
    Generate 3D model from uploaded image or URL.
    
    Authentication: Requires valid JWT token.
    
    Args:
        image: Input image file (optional if tryon_result_url provided)
        tryon_result_url: URL to try-on result image (optional if image provided)
        output_format: Output format (glb, obj, ply)
        use_segmentation: Whether to use SAM2 for background removal
        mc_resolution: Marching cubes resolution (256 for 4GB VRAM, 512 for 8GB+)
        user_id: Authenticated user ID (from JWT token)
    
    Returns:
        ThreeDGenerationResponse with download token and metadata
    
    Raises:
        HTTPException: If validation fails or processing errors occur
    """
    request_id = str(uuid.uuid4())
    logger.info(f"[{request_id}] 3D generation request received")
    logger.info(f"[{request_id}] Parameters: format={output_format}, segmentation={use_segmentation}, resolution={mc_resolution}")
    
    start_time = time.time()
    
    try:
        # Validate that either image or URL is provided
        if image is None and tryon_result_url is None:
            raise HTTPException(
                status_code=400,
                detail={
                    "error_code": "MISSING_INPUT",
                    "message": "Either 'image' file or 'tryon_result_url' must be provided",
                    "request_id": request_id
                }
            )
        
        # If URL is provided, download the image
        if tryon_result_url:
            logger.info(f"[{request_id}] Downloading image from URL: {tryon_result_url}")
            try:
                import httpx
                async with httpx.AsyncClient() as client:
                    response = await client.get(tryon_result_url, timeout=30.0)
                    response.raise_for_status()
                    image_bytes = response.content
                    
                    if len(image_bytes) > 10 * 1024 * 1024:  # 10MB limit
                        raise HTTPException(
                            status_code=400,
                            detail={
                                "error_code": "FILE_TOO_LARGE",
                                "message": f"Image size exceeds 10MB limit (got {len(image_bytes) / 1024 / 1024:.2f}MB)",
                                "request_id": request_id
                            }
                        )
                    
                    logger.info(f"[{request_id}] Downloaded {len(image_bytes)} bytes from URL")
            except httpx.HTTPError as e:
                raise HTTPException(
                    status_code=400,
                    detail={
                        "error_code": "URL_DOWNLOAD_FAILED",
                        "message": f"Failed to download image from URL: {str(e)}",
                        "request_id": request_id
                    }
                )
        else:
            # Validate file type
            if not image.content_type or not image.content_type.startswith("image/"):
                raise HTTPException(
                    status_code=400,
                    detail={
                        "error_code": "INVALID_FILE_TYPE",
                        "message": f"Invalid file type: {image.content_type}. Expected image file.",
                        "request_id": request_id
                    }
                )
            
            # Read and validate image
            logger.info(f"[{request_id}] Reading uploaded image...")
            image_bytes = await image.read()
            
            if len(image_bytes) > 10 * 1024 * 1024:  # 10MB limit
                raise HTTPException(
                    status_code=400,
                    detail={
                        "error_code": "FILE_TOO_LARGE",
                        "message": f"Image size exceeds 10MB limit (got {len(image_bytes) / 1024 / 1024:.2f}MB)",
                        "request_id": request_id
                    }
                )
        
        # Validate output format
        valid_formats = ["glb", "obj", "ply"]
        output_format_lower = output_format.lower()
        if output_format_lower not in valid_formats:
            raise HTTPException(
                status_code=400,
                detail={
                    "error_code": "INVALID_FORMAT",
                    "message": f"Invalid output format: {output_format}. Supported: {', '.join(valid_formats)}",
                    "request_id": request_id
                }
            )
        
        # Validate resolution
        if mc_resolution < 128 or mc_resolution > 512:
            raise HTTPException(
                status_code=400,
                detail={
                    "error_code": "INVALID_RESOLUTION",
                    "message": f"Invalid resolution: {mc_resolution}. Must be between 128 and 512.",
                    "request_id": request_id
                }
            )
        
        # Warn if resolution is too high for 4GB VRAM
        if mc_resolution > 256:
            logger.warning(
                f"[{request_id}] High resolution ({mc_resolution}) may exceed 4GB VRAM. "
                "Recommended: 256 for RTX 3050."
            )
        
        # Load image from bytes
        try:
            pil_image = Image.open(io.BytesIO(image_bytes))
            pil_image = pil_image.convert("RGB")  # Ensure RGB format
            logger.info(f"[{request_id}] Image loaded: {pil_image.size[0]}x{pil_image.size[1]}")
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail={
                    "error_code": "INVALID_IMAGE",
                    "message": f"Failed to load image: {str(e)}",
                    "request_id": request_id
                }
            )
        
        # Get reconstruction pipeline
        pipeline = get_reconstruction_pipeline()
        
        # Process 3D reconstruction
        logger.info(f"[{request_id}] Starting 3D reconstruction pipeline...")
        try:
            file_bytes, format_str, _ = pipeline.process_3d_reconstruction(
                image=pil_image,
                output_format=output_format_lower,
                use_segmentation=use_segmentation,
                mc_resolution=mc_resolution,
                return_intermediate=False  # No intermediate images
            )
        except Exception as e:
            logger.error(f"[{request_id}] 3D reconstruction failed: {e}", exc_info=True)
            raise HTTPException(
                status_code=500,
                detail={
                    "error_code": "RECONSTRUCTION_FAILED",
                    "message": f"3D reconstruction failed: {str(e)}",
                    "request_id": request_id
                }
            )
        
        # Store file with temporary token
        logger.info(f"[{request_id}] Storing 3D model ({len(file_bytes)} bytes)...")
        original_filename = f"model_{request_id}.{format_str}"
        
        try:
            download_token = temp_file_manager.store_temp_file(
                file_bytes=file_bytes,
                original_filename=original_filename,
                expires_in_seconds=3600  # 1 hour
            )
        except Exception as e:
            logger.error(f"[{request_id}] Failed to store temp file: {e}", exc_info=True)
            raise HTTPException(
                status_code=500,
                detail={
                    "error_code": "STORAGE_FAILED",
                    "message": f"Failed to store 3D model: {str(e)}",
                    "request_id": request_id
                }
            )
        
        # Calculate expiration time
        expires_at = datetime.utcnow() + timedelta(seconds=3600)
        
        # Build download URL
        download_url = f"/api/v1/download-3d/{download_token}"
        
        processing_time = time.time() - start_time
        
        # Offload 3D models from GPU to free VRAM
        logger.info(f"[{request_id}] Offloading 3D models from GPU...")
        try:
            pipeline.reset_cuda_memory()
            logger.info(f"[{request_id}] GPU memory freed successfully")
        except Exception as cleanup_error:
            logger.warning(f"[{request_id}] GPU cleanup warning: {cleanup_error}")
        
        logger.info(
            f"[{request_id}] 3D generation completed in {processing_time:.2f}s "
            f"(format={format_str}, size={len(file_bytes)} bytes, token={download_token})"
        )
        
        return ThreeDGenerationResponse(
            download_token=download_token,
            download_url=download_url,
            expires_at=expires_at.isoformat(),
            format=format_str,
            file_size_bytes=len(file_bytes),
            processing_time=processing_time,
            metadata={
                "request_id": request_id,
                "use_segmentation": use_segmentation,
                "mc_resolution": mc_resolution,
                "image_size": f"{pil_image.size[0]}x{pil_image.size[1]}"
            }
        )
    
    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    
    except Exception as e:
        # Catch-all for unexpected errors
        logger.error(f"[{request_id}] Unexpected error: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail={
                "error_code": "INTERNAL_ERROR",
                "message": f"An unexpected error occurred: {str(e)}",
                "request_id": request_id
            }
        )


@router.get(
    "/download-3d/{token}",
    responses={
        200: {"description": "3D model file", "content": {"application/octet-stream": {}}},
        404: {"model": ErrorResponse, "description": "Token not found or expired"},
        500: {"model": ErrorResponse, "description": "Download failed"}
    },
    summary="Download 3D model",
    description="""
    Download a generated 3D model using the token from /generate-3d.
    
    **Token expiration:**
    - Tokens expire after 1 hour
    - Expired tokens return 404
    
    **Content-Type:**
    - GLB: model/gltf-binary
    - OBJ: model/obj
    - PLY: application/ply
    """
)
async def download_3d(
    token: str = FastAPIPath(..., description="Download token from /generate-3d")
) -> Response:
    """
    Download 3D model file by token.
    
    Args:
        token: Download token returned by /generate-3d
    
    Returns:
        Binary file response with appropriate Content-Type
    
    Raises:
        HTTPException: If token is invalid, expired, or file not found
    """
    logger.info(f"Download request for token: {token}")
    
    try:
        # Retrieve file from temp storage
        result = temp_file_manager.get_temp_file(token)
        
        if result is None:
            logger.warning(f"Token not found or expired: {token}")
            raise HTTPException(
                status_code=404,
                detail={
                    "error_code": "TOKEN_NOT_FOUND",
                    "message": "Download token not found or has expired",
                    "token": token
                }
            )
        
        file_bytes, original_filename = result
        
        # Determine content type based on file extension
        extension = original_filename.split('.')[-1].lower()
        content_type_map = {
            "glb": "model/gltf-binary",
            "obj": "model/obj",
            "ply": "application/ply"
        }
        content_type = content_type_map.get(extension, "application/octet-stream")
        
        logger.info(
            f"Serving file: {original_filename} "
            f"({len(file_bytes)} bytes, type={content_type})"
        )
        
        return Response(
            content=file_bytes,
            media_type=content_type,
            headers={
                "Content-Disposition": f'attachment; filename="{original_filename}"',
                "Content-Length": str(len(file_bytes))
            }
        )
    
    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    
    except Exception as e:
        # Catch-all for unexpected errors
        logger.error(f"Download failed for token {token}: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail={
                "error_code": "DOWNLOAD_FAILED",
                "message": f"Failed to download file: {str(e)}",
                "token": token
            }
        )
