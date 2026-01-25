from ..loader import model_loader
from PIL import Image
import torch
import torch.nn as nn
import numpy as np
from app.core.logging_config import get_logger

logger = get_logger("ml.segmentation")

class SegmentationPipeline:
    def __init__(self):
        # Lazy load via singleton
        self.processor, self.model = model_loader.load_segmentation()
        self.device = model_loader.device

    def __call__(self, image: Image.Image) -> dict:
        """
        Returns a dictionary of masks for different body parts.
        """
        try:
            inputs = self.processor(images=image, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits.cpu()

            upsampled_logits = nn.functional.interpolate(
                logits,
                size=image.size[::-1],
                mode="bilinear",
                align_corners=False,
            )

            pred_seg = upsampled_logits.argmax(dim=1)[0].numpy()
            
            logger.debug(f"Segmentation completed for image size {image.size}")
            
            return {
                "segmentation_map": pred_seg,
                "raw_output": outputs
            }
        except Exception as e:
            logger.error(f"Segmentation failed: {e}", exc_info=True)
            raise RuntimeError(f"Segmentation pipeline error: {e}")
