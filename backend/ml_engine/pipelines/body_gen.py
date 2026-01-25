from diffusers import StableDiffusionXLPipeline
import torch
from PIL import Image
import os
from app.core.logging_config import get_logger

logger = get_logger("ml.body_gen")

class BodyGenerationPipeline:
    def __init__(self, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        self.pipe = None
        # Lazy loading handled here or by ModelLoader
    
    def load_model(self):
        if self.pipe is None:
            # Production: Load real model
            try:
                logger.info("Loading SDXL Turbo for body generation...")
                self.pipe = StableDiffusionXLPipeline.from_pretrained(
                    "stabilityai/sdxl-turbo", 
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32, 
                    variant="fp16" if self.device == "cuda" else None,
                    use_safetensors=True
                ).to(self.device)
                logger.info("SDXL Turbo loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load SDXL Turbo: {e}", exc_info=True)
                raise RuntimeError("ML Model unavailable. Ensure 'stabilityai/sdxl-turbo' is accessible.")

    def generate_bodies(self, ethnicity: str, height: float, weight: float, body_type: str, count: int = 4) -> list[Image.Image]:
        try:
            self.load_model()
            
            # Construct Prompt for Full Body generation
            prompt = f"Professional full body photo of a person with {ethnicity} skin tone, {body_type} body type, wearing casual minimal clothing, standing pose, white background, photorealistic, 8k, highly detailed"
            
            logger.debug(f"Generating {count} body images with prompt: {prompt[:100]}...")
            
            images = self.pipe(
                prompt=prompt, 
                num_inference_steps=2, # Turbo needs few steps
                guidance_scale=0.0,
                num_images_per_prompt=count
            ).images
            
            logger.info(f"Successfully generated {len(images)} body images")
            return images
        except Exception as e:
            logger.error(f"Body generation failed: {e}", exc_info=True)
            raise
