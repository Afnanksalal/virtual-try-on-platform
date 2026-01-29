"""
Virtual Try-On Pipeline using CatVTON
Based on: https://github.com/Zheng-Chong/CatVTON
Paper: https://arxiv.org/abs/2407.15886
"""

import os
import sys
import torch
from PIL import Image
from typing import Optional, Dict
import logging
from diffusers.image_processor import VaeImageProcessor
from huggingface_hub import snapshot_download

# Add CatVTON to Python path
CATVTON_PATH = os.path.join(os.path.dirname(__file__), '..', '..', 'CatVTON')
if os.path.exists(CATVTON_PATH) and CATVTON_PATH not in sys.path:
    sys.path.insert(0, CATVTON_PATH)
    print(f"Added CatVTON to Python path: {CATVTON_PATH}")

logger = logging.getLogger(__name__)


class IDMVTONPipeline:
    """
    Virtual try-on using CatVTON (Concatenation Is All You Need for Virtual Try-On).
    
    CatVTON is a lightweight diffusion model that requires:
    - Person image
    - Garment image  
    - Mask (auto-generated using DensePose + SCHP)
    
    Features:
    - Lightweight: 899M parameters total, 49.57M trainable
    - Memory efficient: < 8GB VRAM for 1024x768 resolution
    - No text prompts or pose estimation needed
    """
    
    def __init__(self, device: str = "cuda"):
        """Initialize CatVTON pipeline."""
        self.device = device
        self.pipeline = None
        self.automasker = None
        self.mask_processor = None
        self.dtype = torch.bfloat16 if device == "cuda" else torch.float32
        self.catvton_path = None
        
    def load_models(self, model_id: str = "zhengchong/CatVTON") -> None:
        """
        Load CatVTON models from HuggingFace.
        
        Args:
            model_id: HuggingFace model ID (default: zhengchong/CatVTON)
        """
        try:
            logger.info(f"Loading CatVTON from {model_id}...")
            
            # Download CatVTON checkpoint from HuggingFace
            logger.info("Downloading CatVTON checkpoint...")
            repo_path = snapshot_download(repo_id=model_id)
            self.catvton_path = repo_path
            logger.info(f"CatVTON checkpoint downloaded to: {repo_path}")
            
            # Import CatVTON modules (requires cloned repo in Python path)
            try:
                from model.pipeline import CatVTONPipeline
                from model.cloth_masker import AutoMasker
                from utils import init_weight_dtype, resize_and_crop, resize_and_padding
            except ImportError as e:
                logger.error(
                    "Failed to import CatVTON modules. "
                    "Please clone the CatVTON repository: "
                    "git clone https://github.com/Zheng-Chong/CatVTON "
                    "and add it to your Python path or place it in the backend directory."
                )
                raise ImportError(
                    "CatVTON modules not found. Clone the repo from: "
                    "https://github.com/Zheng-Chong/CatVTON"
                ) from e
            
            # Store utility functions
            self.resize_and_crop = resize_and_crop
            self.resize_and_padding = resize_and_padding
            
            # Initialize CatVTON pipeline
            logger.info("Initializing CatVTON pipeline...")
            self.pipeline = CatVTONPipeline(
                base_ckpt="booksforcharlie/stable-diffusion-inpainting",
                attn_ckpt=repo_path,
                attn_ckpt_version="mix",
                weight_dtype=self.dtype,
                use_tf32=True,
                device=self.device
            )
            
            # Initialize mask processor
            self.mask_processor = VaeImageProcessor(
                vae_scale_factor=8,
                do_normalize=False,
                do_binarize=True,
                do_convert_grayscale=True
            )
            
            # Initialize AutoMasker for automatic mask generation
            logger.info("Initializing AutoMasker (DensePose + SCHP)...")
            self.automasker = AutoMasker(
                densepose_ckpt=os.path.join(repo_path, "DensePose"),
                schp_ckpt=os.path.join(repo_path, "SCHP"),
                device=self.device
            )
            
            logger.info("✓ CatVTON pipeline loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load CatVTON pipeline: {str(e)}", exc_info=True)
            raise RuntimeError(f"CatVTON loading failed: {str(e)}")
    
    def __call__(
        self,
        person_image: Image.Image,
        garment_image: Image.Image,
        garment_description: str = "upper",
        num_inference_steps: int = 50,
        guidance_scale: float = 2.5,
        seed: Optional[int] = 42,
        width: int = 768,
        height: int = 1024,
    ) -> Dict:
        """
        Run CatVTON virtual try-on.
        
        Args:
            person_image: Person image (PIL Image)
            garment_image: Garment image (PIL Image)
            garment_description: Cloth type - "upper", "lower", or "overall" (default: "upper")
            num_inference_steps: Number of diffusion steps (default: 50)
            guidance_scale: CFG strength (default: 2.5, CatVTON recommended)
            seed: Random seed (default: 42)
            width: Output width (default: 768)
            height: Output height (default: 1024)
            
        Returns:
            Dictionary with result image
        """
        try:
            if self.pipeline is None:
                raise RuntimeError("CatVTON not loaded. Call load_models() first.")
            
            logger.info("Starting CatVTON virtual try-on inference...")
            
            # Set random seed
            generator = None
            if seed is not None and seed != -1:
                generator = torch.Generator(device=self.device).manual_seed(seed)
                logger.info(f"Using seed: {seed}")
            
            # Resize images to target resolution
            logger.info(f"Resizing images to {width}x{height}...")
            person_image = self.resize_and_crop(person_image, (width, height))
            garment_image = self.resize_and_padding(garment_image, (width, height))
            
            # Generate mask using AutoMasker
            logger.info(f"Generating mask for cloth type: {garment_description}...")
            mask = self.automasker(person_image, garment_description)['mask']
            
            # Apply blur to mask for smoother transitions
            mask = self.mask_processor.blur(mask, blur_factor=9)
            
            # Run CatVTON inference
            logger.info(f"Running CatVTON inference ({num_inference_steps} steps, CFG={guidance_scale})...")
            result_image = self.pipeline(
                image=person_image,
                condition_image=garment_image,
                mask=mask,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=generator
            )[0]
            
            logger.info("✓ CatVTON virtual try-on completed")
            
            return {"result": result_image}
            
        except Exception as e:
            logger.error(f"CatVTON inference failed: {str(e)}", exc_info=True)
            raise RuntimeError(f"Virtual try-on error: {str(e)}")
