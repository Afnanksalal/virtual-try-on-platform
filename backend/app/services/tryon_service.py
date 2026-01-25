"""
Virtual Try-On Service

This service handles virtual try-on requests using the IDM-VTON pipeline.
It manages the complete workflow from image upload to result storage.
"""

import time
import uuid
from pathlib import Path
from typing import Dict, Optional
from PIL import Image
import io

from ml_engine.pipelines.idm_vton import IDMVTONPipeline
from app.core.logging_config import get_logger

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
        self.results_dir = Path("data/results")
        self.results_dir.mkdir(parents=True, exist_ok=True)
        logger.info("TryOnService initialized")
    
    def _load_pipeline(self):
        """Lazy load IDM-VTON pipeline."""
        if self.pipeline is None:
            logger.info("Loading IDM-VTON pipeline...")
            self.pipeline = IDMVTONPipeline()
            logger.info("IDM-VTON pipeline loaded")
    
    def _save_result(
        self,
        result_image: Image.Image,
        request_id: str
    ) -> str:
        """
        Save result image to storage.
        
        Args:
            result_image: Result PIL image
            request_id: Unique request identifier
            
        Returns:
            Relative path to saved image
        """
        # Generate filename
        filename = f"tryon_{request_id}.png"
        filepath = self.results_dir / filename
        
        # Save image
        result_image.save(filepath, format="PNG", optimize=True)
        logger.debug(f"Result saved to {filepath}")
        
        # Return relative path for URL construction
        return f"results/{filename}"
    
    def process_tryon(
        self,
        person_image: Image.Image,
        garment_image: Image.Image,
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
                - use_cache: Use cached intermediate results (default: True)
                - return_intermediate: Return intermediate results (default: False)
        
        Returns:
            Dictionary containing:
                - request_id: Unique request identifier
                - result_url: URL to result image
                - processing_time: Processing time in seconds
                - cached: Whether cached results were used
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
            target_size = options.get("target_size", (512, 768))
            num_inference_steps = options.get("num_inference_steps", 30)
            guidance_scale = options.get("guidance_scale", 7.5)
            use_cache = options.get("use_cache", True)
            return_intermediate = options.get("return_intermediate", False)
            
            # Validate target size
            if not isinstance(target_size, (tuple, list)) or len(target_size) != 2:
                raise ValueError("target_size must be a tuple of (width, height)")
            
            # Validate inference steps
            if not isinstance(num_inference_steps, int) or num_inference_steps < 1:
                raise ValueError("num_inference_steps must be a positive integer")
            
            # Validate guidance scale
            if not isinstance(guidance_scale, (int, float)) or guidance_scale < 0:
                raise ValueError("guidance_scale must be a non-negative number")
            
            logger.debug(f"Options: size={target_size}, steps={num_inference_steps}, "
                        f"guidance={guidance_scale}, cache={use_cache}")
            
            # Run IDM-VTON pipeline
            logger.info("Running IDM-VTON pipeline...")
            pipeline_result = self.pipeline(
                person_image=person_image,
                garment_image=garment_image,
                target_size=tuple(target_size),
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                use_cache=use_cache,
                return_intermediate=return_intermediate,
            )
            
            result_image = pipeline_result["result"]
            
            # Save result
            logger.info("Saving result...")
            result_path = self._save_result(result_image, request_id)
            
            # Calculate processing time
            processing_time = time.time() - start_time
            
            # Prepare response
            response = {
                "request_id": request_id,
                "result_url": f"/api/v1/{result_path}",
                "processing_time": round(processing_time, 2),
                "cached": False,  # TODO: Implement result caching in task 6.4
                "metadata": {
                    "target_size": target_size,
                    "num_inference_steps": num_inference_steps,
                    "guidance_scale": guidance_scale,
                    "result_dimensions": {
                        "width": result_image.width,
                        "height": result_image.height,
                    }
                }
            }
            
            # Add intermediate results if requested
            if return_intermediate:
                response["intermediate"] = {
                    "segmentation_available": "segmentation_mask" in pipeline_result,
                    "pose_available": "pose_map" in pipeline_result,
                    "mask_available": "inpainting_mask" in pipeline_result,
                }
            
            logger.info(f"Try-on completed in {processing_time:.2f}s")
            return response
            
        except ValueError as e:
            logger.error(f"Validation error: {e}")
            raise
        except Exception as e:
            logger.error(f"Try-on processing failed: {e}", exc_info=True)
            raise RuntimeError(f"Virtual try-on failed: {str(e)}")
    
    def get_cache_stats(self) -> Dict:
        """
        Get cache statistics from pipeline.
        
        Returns:
            Dictionary with cache statistics
        """
        if self.pipeline is None:
            return {"pipeline_loaded": False}
        
        stats = self.pipeline.get_cache_stats()
        stats["pipeline_loaded"] = True
        return stats
    
    def clear_cache(self):
        """Clear cached intermediate results."""
        if self.pipeline is not None:
            self.pipeline.clear_cache()
            logger.info("Pipeline cache cleared")
    
    def cleanup_old_results(self, max_age_hours: int = 24):
        """
        Clean up old result files.
        
        Args:
            max_age_hours: Maximum age of files to keep (in hours)
        """
        import time
        
        current_time = time.time()
        max_age_seconds = max_age_hours * 3600
        deleted_count = 0
        
        for filepath in self.results_dir.glob("tryon_*.png"):
            file_age = current_time - filepath.stat().st_mtime
            if file_age > max_age_seconds:
                filepath.unlink()
                deleted_count += 1
        
        if deleted_count > 0:
            logger.info(f"Cleaned up {deleted_count} old result files")


# Global service instance
tryon_service = TryOnService()
