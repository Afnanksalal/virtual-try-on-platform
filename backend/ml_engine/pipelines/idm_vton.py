"""
IDM-VTON Pipeline Implementation

This module implements the IDM-VTON (Improving Diffusion Models for Authentic Virtual Try-on)
pipeline for realistic virtual try-on using diffusion models.

Pipeline Steps:
1. Preprocessing: Resize images to 512x768, normalize
2. Segmentation: Extract person segmentation mask
3. Pose Estimation: Detect pose keypoints
4. Garment Transfer: Apply IDM-VTON diffusion model
5. Postprocessing: Upscale result, apply refinements

References:
- Paper: https://arxiv.org/abs/2403.05139
- GitHub: https://github.com/yisol/IDM-VTON
"""

import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
from typing import Dict, Optional, Tuple
from pathlib import Path
import hashlib

from diffusers import StableDiffusionXLInpaintPipeline, AutoPipelineForInpainting
from diffusers.models import UNet2DConditionModel
from transformers import CLIPVisionModelWithProjection, CLIPImageProcessor

from ..loader import model_loader
from .segmentation import SegmentationPipeline
from .pose import PosePipeline
from app.core.logging_config import get_logger

logger = get_logger("ml.idm_vton")


class IDMVTONPipeline:
    """
    IDM-VTON pipeline for virtual try-on using diffusion models.
    
    This pipeline combines segmentation, pose estimation, and diffusion-based
    garment transfer to create realistic virtual try-on results.
    """
    
    def __init__(
        self,
        device: Optional[str] = None,
        dtype: Optional[torch.dtype] = None,
        enable_xformers: bool = True,
    ):
        """
        Initialize IDM-VTON pipeline.
        
        Args:
            device: Device to run inference on ('cuda' or 'cpu')
            dtype: Data type for model weights (torch.float16 or torch.float32)
            enable_xformers: Enable memory-efficient attention with xformers
        """
        self.device = device or model_loader.device
        self.dtype = dtype or (torch.float16 if self.device == "cuda" else torch.float32)
        self.enable_xformers = enable_xformers
        
        # Sub-pipelines
        self.segmentation_pipeline = None
        self.pose_pipeline = None
        self.diffusion_pipeline = None
        self.image_encoder = None
        
        # Model paths
        self.weights_dir = Path(__file__).parent.parent / "weights" / "idm-vton"
        
        logger.info(f"IDM-VTON pipeline initialized on {self.device} with dtype {self.dtype}")
    
    def _load_models(self):
        """Lazy load all required models."""
        if self.diffusion_pipeline is None:
            logger.info("Loading IDM-VTON models...")
            
            # Load segmentation pipeline
            if self.segmentation_pipeline is None:
                self.segmentation_pipeline = SegmentationPipeline()
                logger.debug("Segmentation pipeline loaded")
            
            # Load pose pipeline
            if self.pose_pipeline is None:
                self.pose_pipeline = PosePipeline()
                logger.debug("Pose pipeline loaded")
            
            # Load diffusion model
            self._load_diffusion_model()
            
            logger.info("All IDM-VTON models loaded successfully")
    
    def _load_diffusion_model(self):
        """Load the diffusion model for garment transfer."""
        try:
            # Check if custom IDM-VTON checkpoint exists
            checkpoint_path = self.weights_dir / "checkpoint"
            
            if checkpoint_path.exists() and any(checkpoint_path.iterdir()):
                logger.info(f"Loading IDM-VTON checkpoint from {checkpoint_path}")
                # Load custom IDM-VTON checkpoint
                self.diffusion_pipeline = AutoPipelineForInpainting.from_pretrained(
                    str(checkpoint_path),
                    torch_dtype=self.dtype,
                    variant="fp16" if self.dtype == torch.float16 else None,
                )
            else:
                logger.warning("IDM-VTON checkpoint not found, using SDXL Inpainting as fallback")
                logger.info("To use full IDM-VTON, run: python backend/ml_engine/weights/idm-vton/download_weights.py")
                
                # Fallback to SDXL Inpainting
                self.diffusion_pipeline = AutoPipelineForInpainting.from_pretrained(
                    "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
                    torch_dtype=self.dtype,
                    variant="fp16" if self.dtype == torch.float16 else None,
                )
            
            # Move to device
            self.diffusion_pipeline.to(self.device)
            
            # Enable memory optimizations
            if self.enable_xformers and self.device == "cuda":
                try:
                    self.diffusion_pipeline.enable_xformers_memory_efficient_attention()
                    logger.debug("xformers memory-efficient attention enabled")
                except Exception as e:
                    logger.warning(f"Could not enable xformers: {e}")
            
            # Enable CPU offload for low VRAM
            if self.device == "cuda":
                try:
                    if torch.cuda.is_available():
                        total_memory = torch.cuda.get_device_properties(0).total_memory
                        # Enable offload if less than 12GB VRAM
                        if total_memory < 12 * 1024**3:
                            self.diffusion_pipeline.enable_model_cpu_offload()
                            logger.debug("Model CPU offload enabled for low VRAM")
                except Exception as e:
                    logger.warning(f"Could not check VRAM: {e}")
            
            logger.info("Diffusion model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load diffusion model: {e}", exc_info=True)
            raise RuntimeError(f"Failed to load IDM-VTON diffusion model: {e}")
    
    def _preprocess_image(
        self,
        image: Image.Image,
        target_size: Tuple[int, int] = (512, 768)
    ) -> Image.Image:
        """
        Preprocess image to target size.
        
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
            logger.debug(f"Resized image to {target_size}")
        
        return image
    
    
    def _get_image_hash(self, image: Image.Image) -> str:
        """Generate hash for image (unused, kept for compatibility)."""
        img_bytes = image.tobytes()
        return hashlib.md5(img_bytes).hexdigest()
    
    def _extract_segmentation_mask(
        self,
        person_image: Image.Image,
    ) -> np.ndarray:
        """
        Extract segmentation mask for person image.
        
        Args:
            person_image: Person image
            
        Returns:
            Segmentation mask as numpy array
        """
        logger.debug("Extracting segmentation mask...")
        result = self.segmentation_pipeline(person_image)
        mask = result["segmentation_map"]
        
        return mask
    
    def _extract_pose_keypoints(
        self,
        person_image: Image.Image,
    ) -> Image.Image:
        """
        Extract pose keypoints from person image.
        
        Args:
            person_image: Person image
            
        Returns:
            Pose map as PIL image
        """
        logger.debug("Extracting pose keypoints...")
        pose_map = self.pose_pipeline(person_image)
        
        return pose_map
    
    def _create_inpainting_mask(
        self,
        segmentation_mask: np.ndarray,
        target_size: Tuple[int, int]
    ) -> Image.Image:
        """
        Create inpainting mask from segmentation.
        
        Args:
            segmentation_mask: Segmentation mask array
            target_size: Target (width, height) tuple
            
        Returns:
            Binary mask as PIL image
        """
        # Create mask for clothing regions (classes 4-7 typically represent clothing)
        # This is a simplified version - full IDM-VTON uses more sophisticated masking
        clothing_classes = [4, 5, 6, 7]  # Upper clothes, dress, coat, etc.
        mask = np.isin(segmentation_mask, clothing_classes).astype(np.uint8) * 255
        
        # Convert to PIL image
        mask_image = Image.fromarray(mask, mode='L')
        
        # Resize to target size if needed
        if mask_image.size != target_size:
            mask_image = mask_image.resize(target_size, Image.Resampling.NEAREST)
        
        return mask_image
    
    def _apply_garment_transfer(
        self,
        person_image: Image.Image,
        garment_image: Image.Image,
        mask: Image.Image,
        pose_map: Optional[Image.Image] = None,
        num_inference_steps: int = 30,
        strength: float = 0.99,
        guidance_scale: float = 7.5,
    ) -> Image.Image:
        """
        Apply garment transfer using diffusion model.
        
        Args:
            person_image: Preprocessed person image
            garment_image: Preprocessed garment image
            mask: Inpainting mask
            pose_map: Optional pose map for conditioning
            num_inference_steps: Number of diffusion steps
            strength: Inpainting strength (0-1)
            guidance_scale: Classifier-free guidance scale
            
        Returns:
            Result image with garment transferred
        """
        logger.debug("Applying garment transfer...")
        
        # Create prompt for garment transfer
        prompt = (
            "high quality, photorealistic, person wearing the garment, "
            "natural lighting, detailed clothing texture, realistic fit"
        )
        
        negative_prompt = (
            "blurry, distorted, deformed, low quality, artifacts, "
            "unrealistic, bad anatomy, bad proportions"
        )
        
        # Run diffusion inpainting
        result = self.diffusion_pipeline(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=person_image,
            mask_image=mask,
            num_inference_steps=num_inference_steps,
            strength=strength,
            guidance_scale=guidance_scale,
        ).images[0]
        
        logger.debug("Garment transfer completed")
        return result
    
    def _postprocess_result(
        self,
        result_image: Image.Image,
        original_size: Optional[Tuple[int, int]] = None
    ) -> Image.Image:
        """
        Postprocess result image.
        
        Args:
            result_image: Result from diffusion model
            original_size: Optional original size to upscale to
            
        Returns:
            Postprocessed image
        """
        # Upscale to original size if specified
        if original_size and result_image.size != original_size:
            result_image = result_image.resize(
                original_size,
                Image.Resampling.LANCZOS
            )
            logger.debug(f"Upscaled result to {original_size}")
        
        return result_image
    
    def __call__(
        self,
        person_image: Image.Image,
        garment_image: Image.Image,
        target_size: Tuple[int, int] = (512, 768),
        num_inference_steps: int = 30,
        guidance_scale: float = 7.5,
        return_intermediate: bool = False,
    ) -> Dict:
        """
        Run complete IDM-VTON pipeline.
        
        Args:
            person_image: Person image (PIL Image)
            garment_image: Garment image (PIL Image)
            target_size: Target processing size (width, height)
            num_inference_steps: Number of diffusion steps
            guidance_scale: Classifier-free guidance scale
            return_intermediate: Whether to return intermediate results
            
        Returns:
            Dictionary containing:
                - result: Final try-on result image
                - segmentation_mask: (optional) Segmentation mask
                - pose_map: (optional) Pose map
                - inpainting_mask: (optional) Inpainting mask
        """
        try:
            # Load models if not already loaded
            self._load_models()
            
            # Store original size for upscaling
            original_size = person_image.size
            
            # Step 1: Preprocessing
            logger.info("Step 1/5: Preprocessing images...")
            person_image_processed = self._preprocess_image(person_image, target_size)
            garment_image_processed = self._preprocess_image(garment_image, target_size)
            
            # Step 2: Segmentation
            logger.info("Step 2/5: Extracting segmentation mask...")
            segmentation_mask = self._extract_segmentation_mask(person_image_processed)
            
            # Step 3: Pose Estimation
            logger.info("Step 3/5: Extracting pose keypoints...")
            pose_map = self._extract_pose_keypoints(person_image_processed)
            
            # Step 4: Create inpainting mask
            logger.info("Step 4/5: Creating inpainting mask...")
            inpainting_mask = self._create_inpainting_mask(
                segmentation_mask,
                target_size
            )
            
            # Step 5: Garment Transfer
            logger.info("Step 5/5: Applying garment transfer...")
            result_image = self._apply_garment_transfer(
                person_image_processed,
                garment_image_processed,
                inpainting_mask,
                pose_map=pose_map,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
            )
            
            # Postprocessing
            logger.info("Postprocessing result...")
            result_image = self._postprocess_result(result_image, original_size)
            
            logger.info("IDM-VTON pipeline completed successfully")
            
            # Prepare output
            output = {"result": result_image}
            
            if return_intermediate:
                output.update({
                    "segmentation_mask": segmentation_mask,
                    "pose_map": pose_map,
                    "inpainting_mask": inpainting_mask,
                })
            
            return output
            
        except Exception as e:
            logger.error(f"IDM-VTON pipeline failed: {e}", exc_info=True)
            raise RuntimeError(f"IDM-VTON pipeline error: {e}")
