from ..loader import model_loader
from PIL import Image
from app.core.logging_config import get_logger

logger = get_logger("ml.pose")

class PosePipeline:
    def __init__(self):
        # Lazy load via singleton
        self.model = model_loader.load_pose()

    def __call__(self, image: Image.Image) -> Image.Image:
        """
        Extracts pose keypoints and returns the pose map image.
        ControlNetAux OpenPoseDetector call method returns the annotated image directly.
        """
        try:
            # OpenPoseDetector from controlnet_aux returns the PIL image
            pose_image = self.model(image)
            logger.debug(f"Pose extraction completed for image size {image.size}")
            return pose_image
        except Exception as e:
            logger.error(f"Pose estimation failed: {e}", exc_info=True)
            raise RuntimeError(f"Pose pipeline error: {e}")
