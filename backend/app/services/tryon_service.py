"""
Virtual Try-On Service

This service handles virtual try-on requests using the IDM-VTON pipeline.
It manages the complete workflow from image upload to result storage.
"""

import time
import uuid
from datetime import datetime
from typing import Dict, Optional
from PIL import Image
import io

from ml_engine.pipelines.idm_vton import IDMVTONPipeline
from app.core.logging_config import get_logger
from app.services.supabase_storage import supabase_storage

logger = get_logger("services.tryon")


class TryOnService:
    """
    Service for processing virtual try-on requests.
    
    This service:
    1. Validates input images
    2. Runs IDM-VTON pipeline
    3. Saves results to storage
    4. Returns result URLs and metadata
    """
    
    def __init__(self):
        """Initialize try-on service."""
        self.pipeline = None
        logger.info("TryOnService initialized - using Supabase storage ONLY")
    
    def _load_pipeline(self):
        """Lazy load IDM-VTON pipeline."""
        if self.pipeline is None:
            logger.info("Loading IDM-VTON pipeline...")
            self.pipeline = IDMVTONPipeline()
            
            # ALWAYS load from HuggingFace - local weights have incompatible custom UNet classes
            logger.info("Loading from HuggingFace: yisol/IDM-VTON")
            logger.warning("Note: Local weights require custom UNet classes from official repo")
            self.pipeline.load_models("yisol/IDM-VTON")
            
            logger.info("IDM-VTON pipeline loaded successfully")
    
    def _save_result(
        self,
        result_image: Image.Image,
        request_id: str
    ) -> str:
        """
        Save result image to Supabase storage ONLY.
        
        Args:
            result_image: Result PIL image
            request_id: Unique request identifier
            
        Returns:
            Public URL to uploaded image
        """
        # Generate path
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = f"tryon/{request_id}/result_{timestamp}.png"
        
        # Upload to Supabase
        public_url = supabase_storage.upload_image(
            result_image,
            bucket=supabase_storage.RESULTS_BUCKET,
            path=path
        )
        
        logger.info(f"Result uploaded to Supabase: {public_url}")
        return public_url
    
    def process_tryon(
        self,
        person_image: Image.Image,
        garment_image: Image.Image,
        request_id: str,
        options: Optional[Dict] = None
    ) -> Dict:
        """
        Process virtual try-on request.
        
        Args:
            person_image: Person image (PIL Image)
            garment_image: Garment image (PIL Image)
            options: Optional processing options:
                - target_size: Target processing size (default: (512, 768))
                - num_inference_steps: Number of diffusion steps (default: 30)
                - guidance_scale: Guidance scale (default: 7.5)
                - return_intermediate: Return intermediate results (default: False)
        
        Returns:
            Dictionary containing:
                - request_id: Unique request identifier
                - result_url: URL to result image
                - processing_time: Processing time in seconds
                - metadata: Additional metadata
        """
        try:
            # Generate request ID
            request_id = str(uuid.uuid4())
            logger.info(f"Processing try-on request {request_id}")
            
            # Start timing
            start_time = time.time()
            
            # Load pipeline
            self._load_pipeline()
            
            # Parse options
            options = options or {}
            garment_description = options.get("garment_description", "clothing")
            num_inference_steps = options.get("num_inference_steps", 30)
            guidance_scale = options.get("guidance_scale", 2.0)  # IDM-VTON default is 2.0
            seed = options.get("seed", 42)
            
            # Validate inference steps
            if not isinstance(num_inference_steps, int) or num_inference_steps < 1:
                raise ValueError("num_inference_steps must be a positive integer")
            
            # Validate guidance scale
            if not isinstance(guidance_scale, (int, float)) or guidance_scale < 0:
                raise ValueError("guidance_scale must be a non-negative number")
            
            logger.debug(f"Options: steps={num_inference_steps}, "
                        f"guidance={guidance_scale}, seed={seed}")
            
            # Run IDM-VTON pipeline
            logger.info("Running IDM-VTON pipeline...")
            pipeline_result = self.pipeline(
                person_image=person_image,
                garment_image=garment_image,
                garment_description=garment_description,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                seed=seed,
            )
            
            result_image = pipeline_result["result"]
            
            # Save result to Supabase
            logger.info("Uploading result to Supabase...")
            result_url = self._save_result(result_image, request_id)
            
            # Calculate processing time
            processing_time = time.time() - start_time
            
            # Prepare response
            response = {
                "request_id": request_id,
                "result_url": result_url,
                "processing_time": round(processing_time, 2),
                "metadata": {
                    "garment_description": garment_description,
                    "num_inference_steps": num_inference_steps,
                    "guidance_scale": guidance_scale,
                    "seed": seed,
                    "result_dimensions": {
                        "width": result_image.width,
                        "height": result_image.height,
                    }
                }
            }
            
            logger.info(f"Try-on completed in {processing_time:.2f}s")
            return response
            
        except ValueError as e:
            logger.error(f"Validation error: {e}")
            raise
        except Exception as e:
            logger.error(f"Try-on processing failed: {e}", exc_info=True)
            raise RuntimeError(f"Virtual try-on failed: {str(e)}")


# Global service instance
tryon_service = TryOnService()
