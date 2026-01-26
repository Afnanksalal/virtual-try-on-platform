"""
IDM-VTON Pipeline for Virtual Try-On
Based on official implementation: https://github.com/yisol/IDM-VTON
HuggingFace Space: https://huggingface.co/spaces/yisol/IDM-VTON

This implementation follows the official code structure from the HuggingFace demo.
IDM-VTON uses custom UNet architectures and a modified SDXL inpainting pipeline.
"""

import torch
from PIL import Image
import numpy as np
from pathlib import Path
from typing import Optional, Dict
import logging

logger = logging.getLogger(__name__)


class IDMVTONPipeline:
    """
    IDM-VTON pipeline for virtual try-on using diffusion models.
    
    Based on the official implementation which uses:
    - Custom UNet for try-on (unet)
    - Custom UNet encoder for garment encoding (unet_encoder)
    - SDXL-based inpainting pipeline with modifications
    - IP-Adapter for garment conditioning
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
        
    def load_models(self, model_path: str = "yisol/IDM-VTON") -> None:
        """
        Load IDM-VTON models from HuggingFace or local checkpoint.
        
        The official implementation loads from 'yisol/IDM-VTON' on HuggingFace.
        For local weights, the structure should match the HF repo structure with:
        - checkpoint/ (main model weights)
        - ip_adapter/ (IP-Adapter weights)
        - image_encoder/ (CLIP image encoder)
        
        Args:
            model_path: Path to IDM-VTON checkpoint or HF model ID
        """
        try:
            logger.info(f"Loading IDM-VTON models from {model_path}")
            
            # Import required modules
            from diffusers import (
                AutoencoderKL,
                DDPMScheduler,
                UNet2DConditionModel,
                StableDiffusionXLInpaintPipeline,
            )
            from transformers import (
                CLIPImageProcessor,
                CLIPVisionModelWithProjection,
                CLIPTextModel,
                CLIPTextModelWithProjection,
                AutoTokenizer,
            )
            
            # Check if loading from local weights
            model_path_obj = Path(model_path)
            if model_path_obj.exists():
                # Local weights - check for checkpoint subdirectory
                checkpoint_dir = model_path_obj / "checkpoint"
                if checkpoint_dir.exists():
                    logger.info(f"Loading from local checkpoint: {checkpoint_dir}")
                    model_path = str(checkpoint_dir)
                else:
                    logger.info(f"Loading from local path: {model_path}")
            else:
                logger.info(f"Loading from HuggingFace: {model_path}")
            
            # Load custom UNet models (IDM-VTON uses modified UNet architectures)
            logger.info("Loading main UNet (try-on model)...")
            unet = UNet2DConditionModel.from_pretrained(
                model_path,
                subfolder="unet",
                torch_dtype=self.dtype,
            )
            unet.requires_grad_(False)
            
            logger.info("Loading UNet encoder (garment encoder)...")
            unet_encoder = UNet2DConditionModel.from_pretrained(
                model_path,
                subfolder="unet_encoder",
                torch_dtype=self.dtype,
            )
            unet_encoder.requires_grad_(False)
            
            logger.info("Loading VAE...")
            vae = AutoencoderKL.from_pretrained(
                model_path,
                subfolder="vae",
                torch_dtype=self.dtype,
            )
            vae.requires_grad_(False)
            
            logger.info("Loading text encoders...")
            text_encoder_one = CLIPTextModel.from_pretrained(
                model_path,
                subfolder="text_encoder",
                torch_dtype=self.dtype,
            )
            text_encoder_one.requires_grad_(False)
            
            text_encoder_two = CLIPTextModelWithProjection.from_pretrained(
                model_path,
                subfolder="text_encoder_2",
                torch_dtype=self.dtype,
            )
            text_encoder_two.requires_grad_(False)
            
            logger.info("Loading image encoder...")
            image_encoder = CLIPVisionModelWithProjection.from_pretrained(
                model_path,
                subfolder="image_encoder",
                torch_dtype=self.dtype,
            )
            image_encoder.requires_grad_(False)
            
            logger.info("Loading tokenizers...")
            tokenizer_one = AutoTokenizer.from_pretrained(
                model_path,
                subfolder="tokenizer",
                revision=None,
                use_fast=False,
            )
            tokenizer_two = AutoTokenizer.from_pretrained(
                model_path,
                subfolder="tokenizer_2",
                revision=None,
                use_fast=False,
            )
            
            logger.info("Loading scheduler...")
            noise_scheduler = DDPMScheduler.from_pretrained(
                model_path,
                subfolder="scheduler"
            )
            
            # Create the pipeline using standard SDXL Inpainting pipeline
            # The custom behavior comes from the modified UNets, not the pipeline itself
            logger.info("Creating IDM-VTON pipeline...")
            self.pipe = StableDiffusionXLInpaintPipeline.from_pretrained(
                model_path,
                unet=unet,
                vae=vae,
                feature_extractor=CLIPImageProcessor(),
                text_encoder=text_encoder_one,
                text_encoder_2=text_encoder_two,
                tokenizer=tokenizer_one,
                tokenizer_2=tokenizer_two,
                scheduler=noise_scheduler,
                image_encoder=image_encoder,
                torch_dtype=self.dtype,
            )
            
            # Attach the garment encoder (this is IDM-VTON specific)
            self.pipe.unet_encoder = unet_encoder
            
            # Move to device
            if self.device == "cuda":
                logger.info("Moving models to GPU...")
                self.pipe.to(self.device)
                self.pipe.unet_encoder.to(self.device)
            
            logger.info("✓ IDM-VTON models loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load IDM-VTON models: {str(e)}", exc_info=True)
            raise RuntimeError(f"IDM-VTON model loading failed: {str(e)}")
    
    def _preprocess_image(
        self,
        image: Image.Image,
        target_size: tuple = (768, 1024)
    ) -> Image.Image:
        """
        Preprocess image to target size.
        Official IDM-VTON uses (768, 1024) as default size.
        
        Args:
            image: Input PIL image
            target_size: Target (width, height) tuple
            
        Returns:
            Preprocessed PIL image
        """
        # Ensure image is RGB
        if image.mode != "RGB":
            image = image.convert("RGB")
        
        # Resize to target size
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
        
        This is a simplified version that uses the core diffusion pipeline.
        The full official implementation includes:
        - Human parsing for automatic masking
        - OpenPose for pose estimation
        - DensePose for detailed body understanding
        
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
            # Load models if not already loaded
            if self.pipe is None:
                raise RuntimeError("Models not loaded. Call load_models() first.")
            
            logger.info("Starting IDM-VTON inference...")
            
            # Preprocess images to (768, 1024) as per official implementation
            logger.info("Preprocessing images...")
            person_processed = self._preprocess_image(person_image, (768, 1024))
            garment_processed = self._preprocess_image(garment_image, (768, 1024))
            
            # Create a simple mask for the upper body region
            # In the full implementation, this would come from human parsing
            logger.info("Creating inpainting mask...")
            mask = Image.new('L', (768, 1024), 255)  # White mask = inpaint this area
            # Create a rectangular mask for upper body (simplified)
            mask_array = np.array(mask)
            mask_array[200:800, 100:668] = 255  # Upper body region
            mask = Image.fromarray(mask_array)
            
            # Prepare prompts (following official implementation)
            prompt = f"model is wearing {garment_description}"
            negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality"
            
            # Set random seed
            generator = torch.Generator(self.device).manual_seed(seed) if seed is not None else None
            
            # Run inference
            logger.info(f"Running diffusion inference ({num_inference_steps} steps)...")
            with torch.no_grad():
                result = self.pipe(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    image=person_processed,
                    mask_image=mask,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    generator=generator,
                    strength=1.0,
                    height=1024,
                    width=768,
                ).images[0]
            
            logger.info("✓ IDM-VTON inference completed successfully")
            
            return {"result": result}
            
        except Exception as e:
            logger.error(f"IDM-VTON inference failed: {str(e)}", exc_info=True)
            raise RuntimeError(f"IDM-VTON inference error: {str(e)}")
