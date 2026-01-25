from PIL import Image
import numpy as np
from ..core.logging_config import get_logger

logger = get_logger("services.body_detection")

class BodyDetector:
    def __init__(self):
        # Model loader handles the weight loading for OpenPose.
        # Lazy loading to avoid startup delays
        self._pose_model = None

    def _get_pose_model(self):
        """Lazy load pose model."""
        if self._pose_model is None:
            try:
                from ..ml_engine.loader import model_loader
                self._pose_model = model_loader.load_pose()
                logger.info("Pose model loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load pose model: {e}", exc_info=True)
                raise RuntimeError(f"Pose model initialization failed: {e}")
        return self._pose_model

    def check_full_body(self, image: Image.Image) -> dict:
        """
        Analyzes image to determine if it shows a full body or just head/upper body.
        Returns: {
            "is_full_body": bool,
            "missing_parts": list[str] (e.g. ['legs', 'feet']),
            "confidence": float
        }
        """
        try:
            pose_model = self._get_pose_model()
            
            # 1. Generate Pose Map using ControlNet Preprocessor
            # We analyze the pixel distribution in the lower section of the pose map.
            # If the bottom 15% is predominantly black, it suggests no feet/legs were detected.
            
            pose_map = pose_model(image)
            # Convert to numpy
            pose_arr = np.array(pose_map)
            
            height, width, _ = pose_arr.shape
            bottom_cutoff = int(height * 0.85)
            
            # Check bottom strip (feet area)
            bottom_strip = pose_arr[bottom_cutoff:, :, :]
            
            # Check usage (non-black pixels)
            # OpenPose background is black (0,0,0)
            non_zero_pixels = np.count_nonzero(np.any(bottom_strip > 20, axis=2))
            total_pixels = bottom_strip.shape[0] * bottom_strip.shape[1]
            coverage = non_zero_pixels / total_pixels if total_pixels > 0 else 0
            
            # If very little coverage at bottom, legs are missing
            is_full_body = coverage > 0.02  # Threshold: 2% of bottom area has limb drawing
            
            logger.info(f"Body detection result: is_full_body={is_full_body}, coverage={coverage:.4f}")
            
            return {
                "is_full_body": is_full_body,
                "coverage_metric": coverage
            }
        except Exception as e:
            logger.error(f"Body detection failed: {e}", exc_info=True)
            # Return safe default
            return {
                "is_full_body": False,
                "coverage_metric": 0.0,
                "error": str(e)
            }

body_detector = BodyDetector()
