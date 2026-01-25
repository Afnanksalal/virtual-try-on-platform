from ..loader import model_loader
from PIL import Image
import torch
from app.core.logging_config import get_logger

logger = get_logger("ml.tryon")

class TryOnPipeline:
    def __init__(self):
        self.pipe = model_loader.load_tryon()
        
    def __call__(self, person_image: Image.Image, cloth_image: Image.Image, mask: Image.Image) -> Image.Image:
        """
        Executes the try-on process (Simulated via Inpainting for MVP stability).
        """
        try:
            prompt = "high quality, photorealistic, person wearing the cloth"
            
            # Resize images to standard 512x768 or 1024x1024 for SDXL
            # Ensuring divisibility by 8 is good practice for VAEs
            w, h = person_image.size
            new_w = (w // 8) * 8
            new_h = (h // 8) * 8
            if new_w != w or new_h != h:
                person_image = person_image.resize((new_w, new_h))
                mask = mask.resize((new_w, new_h))
                logger.debug(f"Resized images to {new_w}x{new_h} for model compatibility")

            result = self.pipe(
                prompt=prompt,
                image=person_image,
                mask_image=mask,
                num_inference_steps=30,
                strength=0.99, # High strength to replace content in mask
                # For real VTON, we would use IP-Adapter or ControlNet with cloth_image
                # Here we rely on the prompt + mask for the "Structure" and "Inpainting"
            ).images[0]
            
            logger.debug(f"Try-on completed successfully")
            return result
        except Exception as e:
            logger.error(f"Try-on pipeline failed: {e}", exc_info=True)
            raise RuntimeError(f"Try-on pipeline error: {e}")
