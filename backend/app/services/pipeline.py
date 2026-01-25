import asyncio
from typing import Dict, Any
from pathlib import Path

from ..ml_engine.pipelines.segmentation import SegmentationPipeline
from ..ml_engine.pipelines.pose import PosePipeline
from ..ml_engine.pipelines.tryon import TryOnPipeline
from ..core.logging_config import get_logger
from PIL import Image
import os
import uuid

logger = get_logger("services.pipeline")

class TryOnService:
    def __init__(self):
        # Pipelines are now singletons managed by ModelLoader called within their classes
        logger.info("Initializing TryOnService pipelines...")
        self.seg_pipe = SegmentationPipeline()
        self.pose_pipe = PosePipeline()
        self.vton_pipe = TryOnPipeline()
        logger.info("TryOnService pipelines initialized successfully")
    
    async def process_try_on(self, user_image_path: str, cloth_image_path: str) -> dict:
        loop = asyncio.get_event_loop()
        # Offload blocking ML operations to a separate thread
        return await loop.run_in_executor(None, self._process_sync, user_image_path, cloth_image_path)

    def _process_sync(self, user_image_path: str, cloth_image_path: str) -> dict:
        # This execution runs in a worker thread, safe to block
        
        logger.info(f"Processing try-on: User={user_image_path}, Cloth={cloth_image_path}")
        
        # Load Images
        try:
            user_img = Image.open(user_image_path).convert("RGB")
            cloth_img = Image.open(cloth_image_path).convert("RGB")
        except Exception as e:
            logger.error(f"Failed to load images: {e}", exc_info=True)
            raise RuntimeError(f"Image loading failed: {e}")
        
        # 1. Segmentation
        try:
            seg_result = self.seg_pipe(user_img)
            mask = Image.fromarray(seg_result["segmentation_map"].astype("uint8") * 255)
            logger.debug("Segmentation completed")
        except Exception as e:
            logger.error(f"Segmentation failed: {e}", exc_info=True)
            raise
        
        # 2. Pose
        try:
            pose_map = self.pose_pipe(user_img)
            logger.debug("Pose estimation completed")
        except Exception as e:
            logger.error(f"Pose estimation failed: {e}", exc_info=True)
            raise
        
        # 3. Try-On
        try:
            result_img = self.vton_pipe(user_img, cloth_img, mask)
            logger.debug("Virtual try-on completed")
        except Exception as e:
            logger.error(f"Virtual try-on failed: {e}", exc_info=True)
            raise
        
        # Save Result
        output_dir = Path("data") / "results"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_filename = f"result_{uuid.uuid4()}.png"
        output_path = output_dir / output_filename
        
        try:
            result_img.save(str(output_path))
            logger.info(f"Result saved to {output_path}")
        except Exception as e:
            logger.error(f"Failed to save result: {e}", exc_info=True)
            raise
        
        return {
            "status": "success",
            "result_path": str(output_path.absolute()),
            "metadata": {
                "model": "IDM-VTON",
                "resolution": f"{result_img.size}"
            }
        }

try_on_service = TryOnService()
