"""
IDM-VTON Pipeline for Virtual Try-On
Based on official implementation: https://github.com/yisol/IDM-VTON
HuggingFace Model: https://huggingface.co/yisol/IDM-VTON

This implementation loads IDM-VTON directly from HuggingFace with custom UNet classes.
"""

import torch
from PIL import Image
import numpy as np
from typing import Optional, Dict
import logging

logger = logging.getLogger(__name__)


class IDMVTONPipeline:
    """
    IDM-VTON pipeline for virtual try-on using diffusion models.
    
    Loads the complete pipeline from HuggingFace yisol/IDM-VTON with custom UNet classes.
    The model uses trust_remote_code=True to load custom UNet implementations.
    """
    
    def __init__(self, device: str = "cuda"):
        """
        Initialize the IDM-VTON pipeline.
        
        Args:
            device: Device to run inference on ('cuda' or 'cpu')
        """
        self.device = device
        self.pipe = None
        self.dtype = torch.float16 if device == "cuda" else torch.float32
        
    def load_models(self, model_id: str = "yisol/IDM-VTON") -> None:
        """
        Load IDM-VTON models from HuggingFace.
        
        The model contains custom UNet classes that require trust_remote_code=True.
        This will download and cache the model in ~/.cache/huggingface/hub/
        
        Args:
            model_id: HuggingFace model ID (default: yisol/IDM-VTON)
        """
        try:
            logger.info(f"Loading IDM-VTON from HuggingFace: {model_id}")
            logger.info("This will download ~20GB of model weights on first run...")
            
            from diffusers import DiffusionPipeline
            
            # Load the complete pipeline from HuggingFace with custom code
            # The yisol/IDM-VTON repo contains custom UNet classes
            logger.info("Downloading model (this may take several minutes)...")
            self.pipe = DiffusionPipeline.from_pretrained(
                model_id,
                torch_dtype=self.dtype,
                trust_remote_code=True,  # Required for custom UNet classes
            )
            
            # Move to device
            if self.device == "cuda":
                logger.info("Moving models to GPU...")
                self.pipe.to(self.device)
                
                # Enable memory optimizations
                try:
                    self.pipe.enable_xformers_memory_efficient_attention()
                    logger.info("✓ xformers memory optimization enabled")
                except Exception as e:
                    logger.warning(f"Could not enable xformers: {e}")
            
            logger.info("✓ IDM-VTON models loaded successfully")
            logger.info(f"Model cached in: ~/.cache/huggingface/hub/")
            
        except Exception as e:
            logger.error(f"Failed to load IDM-VTON: {str(e)}", exc_info=True)
            raise RuntimeError(f"IDM-VTON loading failed: {str(e)}")
    
    def _preprocess_image(
        self,
        image: Image.Image,
        target_size: tuple = (768, 1024)
    ) -> Image.Image:
        """
        Preprocess image to target size.
        IDM-VTON uses (768, 1024) as default size.
        
        Args:
            image: Input PIL image
            target_size: Target (width, height) tuple
            
        Returns:
            Preprocessed PIL image
        """
        if image.mode != "RGB":
            image = image.convert("RGB")
        
        if image.size != target_size:
            image = image.resize(target_size, Image.Resampling.LANCZOS)
        
        return image
    
    def __call__(
        self,
        person_image: Image.Image,
        garment_image: Image.Image,
        garment_description: str = "clothing",
        num_inference_steps: int = 30,
        guidance_scale: float = 2.0,
        seed: Optional[int] = 42,
    ) -> Dict:
        """
        Run IDM-VTON virtual try-on.
        
        Args:
            person_image: Person image (PIL Image)
            garment_image: Garment image (PIL Image)
            garment_description: Text description of garment
            num_inference_steps: Number of diffusion steps (default: 30)
            guidance_scale: Classifier-free guidance scale (default: 2.0)
            seed: Random seed for reproducibility
            
        Returns:
            Dictionary containing:
                - result: Final try-on result image
        """
        try:
            if self.pipe is None:
                raise RuntimeError("Models not loaded. Call load_models() first.")
            
            logger.info("Starting IDM-VTON inference...")
            
            # Preprocess images
            logger.info("Preprocessing images...")
            person_processed = self._preprocess_image(person_image, (768, 1024))
            garment_processed = self._preprocess_image(garment_image, (768, 1024))
            
            # Set random seed
            generator = torch.Generator(self.device).manual_seed(seed) if seed is not None else None
            
            # Run inference using the pipeline's __call__ method
            # The exact parameters depend on how the custom pipeline is implemented
            logger.info(f"Running inference ({num_inference_steps} steps)...")
            
            with torch.no_grad():
                # Try calling the pipeline - the custom implementation should handle the rest
                result = self.pipe(
                    person_image=person_processed,
                    garment_image=garment_processed,
                    prompt=f"model wearing {garment_description}",
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    generator=generator,
                )
                
                # Extract the result image
                if hasattr(result, 'images'):
                    result_image = result.images[0]
                elif isinstance(result, list):
                    result_image = result[0]
                else:
                    result_image = result
            
            logger.info("✓ IDM-VTON inference completed")
            
            return {"result": result_image}
            
        except Exception as e:
            logger.error(f"IDM-VTON inference failed: {str(e)}", exc_info=True)
            raise RuntimeError(f"IDM-VTON inference error: {str(e)}")
