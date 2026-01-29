"""
StableVITON Pipeline for Virtual Try-On
Using rlawjdghek/StableVITON - CVPR 2024
"""

import torch
from PIL import Image
import sys
import os
from pathlib import Path
from typing import Optional, Dict
import logging

logger = logging.getLogger(__name__)


class IDMVTONPipeline:
    """
    StableVITON pipeline for virtual try-on.
    Uses the cloned StableVITON repository.
    """
    
    def __init__(self, device: str = "cuda"):
        """Initialize StableVITON pipeline."""
        self.device = device
        self.model = None
        self.dtype = torch.float16 if device == "cuda" else torch.float32
        
        # Find StableVITON repo
        if os.path.exists("/teamspace/studios/this_studio/StableVITON"):
            self.repo_path = Path("/teamspace/studios/this_studio/StableVITON")
        else:
            backend_dir = Path(__file__).parent.parent.parent
            self.repo_path = backend_dir / "StableVITON"
        
        logger.info(f"StableVITON repo path: {self.repo_path}")
        
    def load_models(self, model_id: str = "rlawjdghek/StableVITON") -> None:
        """
        Load StableVITON model from cloned repository.
        
        Args:
            model_id: HuggingFace model ID (default: rlawjdghek/StableVITON)
        """
        try:
            if not self.repo_path.exists():
                raise RuntimeError(
                    f"StableVITON repository not found at {self.repo_path}\n"
                    f"Clone it: git clone https://github.com/rlawjdghek/StableVITON"
                )
            
            logger.info(f"Found StableVITON at: {self.repo_path}")
            
            # Add repo to path
            if str(self.repo_path) not in sys.path:
                sys.path.insert(0, str(self.repo_path))
            
            logger.info("Importing StableVITON modules...")
            
            # Import StableVITON's model classes
            from model.networks import ConditionGenerator, SPADEGenerator
            from model.afwm import AFWM
            
            logger.info(f"Loading model weights from {model_id}...")
            
            # Download weights from HuggingFace
            from huggingface_hub import hf_hub_download
            
            # Download checkpoint
            ckpt_path = hf_hub_download(
                repo_id=model_id,
                filename="model.ckpt",
                cache_dir=str(self.repo_path / "ckpts")
            )
            
            logger.info(f"Loading checkpoint from {ckpt_path}...")
            
            # Load checkpoint
            checkpoint = torch.load(ckpt_path, map_location=self.device)
            
            # Initialize models
            self.condition_generator = ConditionGenerator().to(self.device)
            self.generator = SPADEGenerator().to(self.device)
            self.afwm = AFWM().to(self.device)
            
            # Load weights
            self.condition_generator.load_state_dict(checkpoint['condition_generator'])
            self.generator.load_state_dict(checkpoint['generator'])
            self.afwm.load_state_dict(checkpoint['afwm'])
            
            # Set to eval mode
            self.condition_generator.eval()
            self.generator.eval()
            self.afwm.eval()
            
            logger.info("✓ StableVITON loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load StableVITON: {str(e)}", exc_info=True)
            raise RuntimeError(f"StableVITON loading failed: {str(e)}")
    
    def __call__(
        self,
        person_image: Image.Image,
        garment_image: Image.Image,
        garment_description: str = "clothing",
        num_inference_steps: int = 50,
        guidance_scale: float = 2.5,
        seed: Optional[int] = 42,
    ) -> Dict:
        """
        Run StableVITON virtual try-on.
        
        Args:
            person_image: Person image (PIL Image)
            garment_image: Garment image (PIL Image)
            garment_description: Text description
            num_inference_steps: Number of steps
            guidance_scale: Guidance scale
            seed: Random seed
            
        Returns:
            Dictionary with result image
        """
        try:
            if self.model is None:
                raise RuntimeError("Model not loaded. Call load_models() first.")
            
            logger.info("Starting StableVITON inference...")
            
            # Set seed
            if seed is not None:
                torch.manual_seed(seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed(seed)
            
            # Preprocess images
            from utils.preprocessing import preprocess_image
            
            person_tensor = preprocess_image(person_image).to(self.device)
            garment_tensor = preprocess_image(garment_image).to(self.device)
            
            # Run inference
            with torch.no_grad():
                # Generate conditions
                conditions = self.condition_generator(person_tensor, garment_tensor)
                
                # Warp garment
                warped_garment = self.afwm(garment_tensor, conditions)
                
                # Generate final result
                result_tensor = self.generator(person_tensor, warped_garment, conditions)
            
            # Convert to PIL
            from utils.postprocessing import tensor_to_pil
            result_image = tensor_to_pil(result_tensor)
            
            logger.info("✓ StableVITON inference completed")
            
            return {"result": result_image}
            
        except Exception as e:
            logger.error(f"StableVITON inference failed: {str(e)}", exc_info=True)
            raise RuntimeError(f"StableVITON inference error: {str(e)}")
