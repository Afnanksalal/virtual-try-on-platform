"""
Virtual Try-On Service

This service handles virtual try-on requests using the Leffa pipeline.
It manages the complete workflow from image upload to result storage.

Leffa features:
- Flow-based diffusion model for high-quality try-on
- Built-in pose and mask handling
- Clean pipeline interface
"""

import time
import uuid
from datetime import datetime
from typing import Dict, Optional
from PIL import Image
import io
import os

from app.core.logging_config import get_logger
from app.services.supabase_storage import supabase_storage

logger = get_logger("services.tryon")

# Lazy import for Leffa
LeffaPipeline = None


def _get_pipeline_class():
    """Lazy import of LeffaPipeline."""
    global LeffaPipeline
    if LeffaPipeline is None:
        try:
            from ml_engine.pipelines.tryon import LeffaPipeline as Pipeline
            LeffaPipeline = Pipeline
        except ImportError as e:
            logger.error(f"Failed to import LeffaPipeline: {e}")
            raise ImportError(
                "Leffa pipeline not available. Install with: pip install -r requirements.txt"
            )
    return LeffaPipeline


class TryOnService:
    """
    Service for processing virtual try-on requests.
    
    This service:
    1. Validates input images
    2. Runs Leffa pipeline
    3. Saves results to storage
    4. Returns result URLs and metadata
    """
    
    def __init__(self):
        """Initialize try-on service."""
        self.pipeline = None
        logger.info("TryOnService initialized - using Leffa with Supabase storage")
    
    def _load_pipeline(self):
        """Lazy load Leffa pipeline."""
        if self.pipeline is None:
            logger.info("Loading Leffa pipeline...")
            
            # Get the pipeline class
            PipelineClass = _get_pipeline_class()
            
            # Determine device
            device = os.getenv("DEVICE", None)  # None = auto-detect
            
            # Initialize pipeline
            self.pipeline = PipelineClass(device=device)
            
            # Load Leffa models from local repo
            logger.info("Loading Leffa from local repository")
            self.pipeline.load_models()
            
            logger.info("Leffa pipeline loaded successfully")
    
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
        Process virtual try-on request using Leffa.
        
        Args:
            person_image: Person image (PIL Image)
            garment_image: Garment image (PIL Image)
            request_id: Unique request identifier
            options: Optional processing options:
                - garment_type: Cloth type - "upper_body", "lower_body", or "dresses" (default: "upper_body")
                - num_inference_steps: Number of diffusion steps (default: 30, range: 10-50)
                - guidance_scale: CFG strength (default: 2.5, range: 1.0-5.0)
                - seed: Random seed (default: 42)
                - model_type: Model variant - "viton_hd" or "dress_code" (default: "viton_hd")
                - ref_acceleration: Speed up reference UNet (default: False)
                - repaint: Enable repaint mode (default: False)
        
        Returns:
            Dictionary containing:
                - request_id: Unique request identifier
                - result_url: URL to result image
                - processing_time: Processing time in seconds
                - metadata: Additional metadata including all options used
        """
        try:
            # Generate request ID
            request_id = str(uuid.uuid4())
            logger.info(f"Processing Leffa try-on request {request_id}")
            
            # Start timing
            start_time = time.time()
            
            # Load pipeline
            self._load_pipeline()
            
            # Parse options
            options = options or {}
            garment_type = options.get("garment_type", options.get("garment_description", "upper_body"))
            num_inference_steps = options.get("num_inference_steps", 30)
            guidance_scale = options.get("guidance_scale", 2.5)  # Leffa default
            seed = options.get("seed", 42)
            model_type = options.get("model_type", "viton_hd")  # viton_hd or dress_code
            ref_acceleration = options.get("ref_acceleration", False)  # Speed up reference UNet
            repaint = options.get("repaint", False)  # Enable repaint mode
            
            # Map legacy garment types to Leffa types
            garment_type_map = {
                "upper": "upper_body",
                "lower": "lower_body",
                "full": "dresses",
                "overall": "dresses",
            }
            if garment_type in garment_type_map:
                garment_type = garment_type_map[garment_type]
            
            # Validate inference steps (clamp to reasonable range)
            if not isinstance(num_inference_steps, int) or num_inference_steps < 1:
                raise ValueError("num_inference_steps must be a positive integer")
            num_inference_steps = max(10, min(50, num_inference_steps))  # Clamp 10-50
            
            # Validate guidance scale (clamp to reasonable range)
            if not isinstance(guidance_scale, (int, float)) or guidance_scale < 0:
                raise ValueError("guidance_scale must be a non-negative number")
            guidance_scale = max(1.0, min(5.0, float(guidance_scale)))  # Clamp 1.0-5.0
            
            # Validate garment type
            if garment_type not in ["upper_body", "lower_body", "dresses"]:
                logger.warning(f"Invalid garment_type '{garment_type}', defaulting to 'upper_body'")
                garment_type = "upper_body"
            
            # Validate model type
            if model_type not in ["viton_hd", "dress_code"]:
                logger.warning(f"Invalid model_type '{model_type}', defaulting to 'viton_hd'")
                model_type = "viton_hd"
            
            logger.debug(f"Options: garment_type={garment_type}, model_type={model_type}, "
                        f"steps={num_inference_steps}, guidance={guidance_scale}, seed={seed}, "
                        f"ref_acceleration={ref_acceleration}, repaint={repaint}")
            
            # Run Leffa pipeline
            logger.info("Running Leffa pipeline...")
            pipeline_result = self.pipeline(
                person_image=person_image,
                garment_image=garment_image,
                garment_type=garment_type,
                model_type=model_type,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                seed=seed,
                ref_acceleration=ref_acceleration,
                repaint=repaint,
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
                    "garment_type": garment_type,
                    "model_type": model_type,
                    "num_inference_steps": num_inference_steps,
                    "guidance_scale": guidance_scale,
                    "seed": seed,
                    "ref_acceleration": ref_acceleration,
                    "repaint": repaint,
                    "result_dimensions": {
                        "width": result_image.width,
                        "height": result_image.height,
                    }
                }
            }
            
            logger.info(f"Leffa try-on completed in {processing_time:.2f}s")
            return response
            
        except ValueError as e:
            logger.error(f"Validation error: {e}")
            raise
        except Exception as e:
            logger.error(f"Try-on processing failed: {e}", exc_info=True)
            raise RuntimeError(f"Virtual try-on failed: {str(e)}")


# Global service instance
tryon_service = TryOnService()
