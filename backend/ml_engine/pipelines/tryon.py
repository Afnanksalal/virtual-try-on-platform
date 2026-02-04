"""
Virtual Try-On Pipeline using Leffa
Based on: https://github.com/franciszzj/Leffa
HuggingFace: https://huggingface.co/spaces/franciszzj/Leffa

Leffa is a flow-based virtual try-on model that provides:
- Built-in pose and mask handling via preprocessing
- Clean pipeline interface
- High-quality results with efficient inference

Setup:
    1. Clone repo at project root: git clone https://github.com/franciszzj/Leffa
       The Leffa folder should be at the same level as backend/ and frontend/
    2. Checkpoints auto-download on first run via huggingface_hub

Usage:
    pipeline = LeffaPipeline()
    pipeline.load_models()
    result = pipeline(person_image, garment_image, garment_type="upper_body")
"""

import os
import sys
import torch
import numpy as np
from PIL import Image
from typing import Optional, Dict, Tuple
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def setup_leffa_path() -> Optional[str]:
    """
    Setup Leffa path by finding and adding it to sys.path.
    
    The Leffa repository should be cloned at the project root level,
    at the same level as backend/ and frontend/ folders.
    
    Returns:
        Path to Leffa directory if found, None otherwise
    """
    # Priority order for Leffa location
    # 1. Project root (same level as backend/frontend) - PREFERRED
    # 2. Inside backend folder (legacy support)
    possible_paths = [
        os.path.join(os.path.dirname(__file__), '..', '..', '..', 'Leffa'),  # project_root/Leffa (preferred)
        os.path.join(os.path.dirname(__file__), '..', '..', 'Leffa'),  # backend/Leffa (legacy)
    ]
    
    for path in possible_paths:
        abs_path = os.path.abspath(path)
        # Check for leffa subfolder to confirm it's the right repo
        leffa_subfolder = os.path.join(abs_path, 'leffa')
        if os.path.exists(abs_path) and os.path.isdir(abs_path) and os.path.exists(leffa_subfolder):
            if abs_path not in sys.path:
                sys.path.insert(0, abs_path)
            logger.info(f"Leffa repository found at: {abs_path}")
            return abs_path
    
    return None


# Initialize Leffa path on module load
LEFFA_PATH = setup_leffa_path()

if LEFFA_PATH:
    logger.info(f"Leffa initialized from: {LEFFA_PATH}")
else:
    logger.warning(
        "Leffa repository not found. Clone it at project root with:\n"
        "  git clone https://github.com/franciszzj/Leffa\n"
        "The Leffa folder should be at the same level as backend/ and frontend/"
    )


def download_leffa_checkpoints(ckpts_path: str = None) -> str:
    """
    Download Leffa checkpoints from HuggingFace on initial run.
    
    This mirrors the snapshot_download call from the original Leffa app.py.
    Downloads all model weights, preprocessor models, and examples.
    
    Args:
        ckpts_path: Optional custom path for checkpoints.
                   Defaults to LEFFA_PATH/ckpts
    
    Returns:
        Path to the checkpoints directory
    """
    try:
        from huggingface_hub import snapshot_download
        
        if ckpts_path is None:
            if LEFFA_PATH:
                ckpts_path = os.path.join(LEFFA_PATH, "ckpts")
            else:
                raise RuntimeError("LEFFA_PATH not set and no ckpts_path provided")
        
        # Check if checkpoints already exist and have content
        required_files = [
            "stable-diffusion-inpainting",
            "virtual_tryon.pth",
            "densepose",
            "humanparsing",
            "openpose",
            "schp",
        ]
        
        all_exist = all(
            os.path.exists(os.path.join(ckpts_path, f)) 
            for f in required_files
        )
        
        if all_exist:
            logger.info(f"Leffa checkpoints already exist at {ckpts_path}")
            return ckpts_path
        
        logger.info("=" * 60)
        logger.info("Downloading Leffa checkpoints from HuggingFace...")
        logger.info("Repository: franciszzj/Leffa")
        logger.info("This may take 10-30 minutes on first run (several GB)")
        logger.info("=" * 60)
        
        snapshot_download(
            repo_id="franciszzj/Leffa",
            local_dir=ckpts_path,
            # Optionally exclude large files not needed for inference
            # ignore_patterns=["*.md", "examples/*"]  
        )
        
        logger.info(f"Checkpoints downloaded successfully to: {ckpts_path}")
        return ckpts_path
        
    except ImportError:
        logger.error("huggingface_hub not installed. Run: pip install huggingface_hub")
        raise
    except Exception as e:
        logger.error(f"Failed to download Leffa checkpoints: {e}")
        raise


class LeffaPipeline:
    """
    Virtual try-on pipeline using Leffa.
    
    Based on the LeffaPredictor from the original Leffa app.py.
    Provides both virtual try-on and pose transfer capabilities.
    
    Leffa is a flow-based diffusion model for virtual try-on that handles:
    - Person image processing with DensePose
    - Garment-agnostic mask generation (via AutoMasker)
    - Human parsing and pose detection
    - Garment transfer with reference UNet
    
    Features:
    - High quality: State-of-the-art results
    - Multiple garment types: upper_body, lower_body, dresses
    - Two model variants: viton_hd (default) and dress_code
    - Pose transfer support (SDXL-based)
    - Auto-downloads checkpoints on first run
    """
    
    # HuggingFace repo for auto-downloading checkpoints
    HUGGINGFACE_REPO = "franciszzj/Leffa"
    
    # Default inference parameters
    DEFAULT_STEPS = 30
    DEFAULT_CFG = 2.5
    DEFAULT_WIDTH = 768
    DEFAULT_HEIGHT = 1024
    
    def __init__(self, device: str = None, auto_download: bool = True):
        """
        Initialize Leffa pipeline.
        
        Args:
            device: Target device ("cuda", "cpu", or None for auto-detect)
            auto_download: Whether to auto-download checkpoints if missing
        """
        # Auto-detect device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        # Set dtype based on device (float16 for GPU, float32 for CPU)
        self.dtype = "float16" if self.device == "cuda" else "float32"
        
        # Checkpoints path (relative to Leffa repo)
        self.ckpts_path = os.path.join(LEFFA_PATH, "ckpts") if LEFFA_PATH else "./ckpts"
        
        # Auto-download checkpoints if enabled
        if auto_download:
            download_leffa_checkpoints(self.ckpts_path)
        
        # Model components (lazy loaded in load_models)
        # Virtual Try-On models
        self.vt_model_hd = None       # VITON-HD model
        self.vt_model_dc = None       # DressCode model
        self.vt_inference_hd = None   # VITON-HD inference
        self.vt_inference_dc = None   # DressCode inference
        
        # Pose Transfer model
        self.pt_model = None          # Pose transfer model (SDXL-based)
        self.pt_inference = None      # Pose transfer inference
        
        # Transform
        self.transform = None
        
        # Preprocessors (from LeffaPredictor)
        self.mask_predictor = None    # AutoMasker for garment-agnostic masks
        self.densepose_predictor = None  # DensePose predictor
        self.parsing = None           # Human parsing (ATR/LIP)
        self.openpose = None          # OpenPose for keypoints
        
        # Status
        self._loaded = False
        
        logger.info(f"LeffaPipeline initialized (device={self.device}, dtype={self.dtype})")
    
    def _import_leffa_modules(self):
        """
        Import Leffa modules from cloned repository.
        
        Imports all required modules as specified in the original app.py:
        - leffa.transform, leffa.model, leffa.inference
        - leffa_utils (AutoMasker, DensePosePredictor, utils)
        - preprocess (humanparsing, openpose)
        """
        try:
            # Core Leffa modules
            from leffa.transform import LeffaTransform
            from leffa.model import LeffaModel
            from leffa.inference import LeffaInference
            
            # Leffa utilities
            from leffa_utils.garment_agnostic_mask_predictor import AutoMasker
            from leffa_utils.densepose_predictor import DensePosePredictor
            from leffa_utils.utils import (
                resize_and_center, 
                list_dir, 
                get_agnostic_mask_hd, 
                get_agnostic_mask_dc,
                preprocess_garment_image
            )
            
            # Preprocessors
            from preprocess.humanparsing.run_parsing import Parsing
            from preprocess.openpose.run_openpose import OpenPose
            
            # Store all imported modules as instance attributes
            self.LeffaTransform = LeffaTransform
            self.LeffaModel = LeffaModel
            self.LeffaInference = LeffaInference
            self.AutoMasker = AutoMasker
            self.DensePosePredictor = DensePosePredictor
            self.resize_and_center = resize_and_center
            self.list_dir = list_dir
            self.get_agnostic_mask_hd = get_agnostic_mask_hd
            self.get_agnostic_mask_dc = get_agnostic_mask_dc
            self.preprocess_garment_image = preprocess_garment_image
            self.Parsing = Parsing
            self.OpenPose = OpenPose
            
            logger.info("Leffa modules imported successfully")
            return True
            
        except ImportError as e:
            logger.error(f"Failed to import Leffa modules: {e}")
            logger.error(
                "Please clone the Leffa repository at project root:\n"
                "  git clone https://github.com/franciszzj/Leffa\n"
                "The Leffa folder should be at the same level as backend/ and frontend/"
            )
            return False
    
    def load_models(self, load_pose_transfer: bool = False) -> None:
        """
        Load Leffa models and preprocessors.
        
        This mirrors the LeffaPredictor.__init__ from the original app.py.
        Loads all required models for virtual try-on and optionally pose transfer.
        
        Args:
            load_pose_transfer: Whether to also load pose transfer model (uses more VRAM)
        """
        try:
            logger.info("=" * 60)
            logger.info("Loading Leffa Virtual Try-On Pipeline")
            logger.info("=" * 60)
            
            # Import modules
            if not self._import_leffa_modules():
                raise ImportError("Leffa modules not available")
            
            # Initialize transform
            self.transform = self.LeffaTransform()
            
            # =========================================================
            # Initialize preprocessors (as in LeffaPredictor.__init__)
            # =========================================================
            
            logger.info("Loading AutoMasker (garment-agnostic mask predictor)...")
            self.mask_predictor = self.AutoMasker(
                densepose_path=os.path.join(self.ckpts_path, "densepose"),
                schp_path=os.path.join(self.ckpts_path, "schp"),
            )
            
            logger.info("Loading DensePose predictor...")
            self.densepose_predictor = self.DensePosePredictor(
                config_path=os.path.join(self.ckpts_path, "densepose", "densepose_rcnn_R_50_FPN_s1x.yaml"),
                weights_path=os.path.join(self.ckpts_path, "densepose", "model_final_162be9.pkl"),
            )
            
            logger.info("Loading Human Parsing model (ATR + LIP)...")
            self.parsing = self.Parsing(
                atr_path=os.path.join(self.ckpts_path, "humanparsing", "parsing_atr.onnx"),
                lip_path=os.path.join(self.ckpts_path, "humanparsing", "parsing_lip.onnx"),
            )
            
            logger.info("Loading OpenPose model...")
            self.openpose = self.OpenPose(
                body_model_path=os.path.join(self.ckpts_path, "openpose", "body_pose_model.pth"),
            )
            
            # =========================================================
            # Load Virtual Try-On models
            # =========================================================
            
            sd_inpainting_path = os.path.join(self.ckpts_path, "stable-diffusion-inpainting")
            
            # VITON-HD model (default, recommended)
            logger.info("Loading VITON-HD virtual try-on model...")
            self.vt_model_hd = self.LeffaModel(
                pretrained_model_name_or_path=sd_inpainting_path,
                pretrained_model=os.path.join(self.ckpts_path, "virtual_tryon.pth"),
                dtype=self.dtype,
            )
            self.vt_inference_hd = self.LeffaInference(model=self.vt_model_hd)
            
            # DressCode model (for dress_code dataset compatibility)
            dc_model_path = os.path.join(self.ckpts_path, "virtual_tryon_dc.pth")
            if os.path.exists(dc_model_path):
                logger.info("Loading DressCode virtual try-on model...")
                self.vt_model_dc = self.LeffaModel(
                    pretrained_model_name_or_path=sd_inpainting_path,
                    pretrained_model=dc_model_path,
                    dtype=self.dtype,
                )
                self.vt_inference_dc = self.LeffaInference(model=self.vt_model_dc)
            else:
                logger.info("DressCode model not found, skipping (optional)")
            
            # =========================================================
            # Load Pose Transfer model (optional, uses SDXL)
            # =========================================================
            
            if load_pose_transfer:
                pt_model_path = os.path.join(self.ckpts_path, "pose_transfer.pth")
                sdxl_inpainting_path = os.path.join(self.ckpts_path, "stable-diffusion-xl-1.0-inpainting-0.1")
                
                if os.path.exists(pt_model_path) and os.path.exists(sdxl_inpainting_path):
                    logger.info("Loading Pose Transfer model (SDXL-based)...")
                    self.pt_model = self.LeffaModel(
                        pretrained_model_name_or_path=sdxl_inpainting_path,
                        pretrained_model=pt_model_path,
                        dtype=self.dtype,
                    )
                    self.pt_inference = self.LeffaInference(model=self.pt_model)
                else:
                    logger.warning("Pose transfer model files not found, skipping")
            
            self._loaded = True
            
            logger.info("=" * 60)
            logger.info("Leffa Pipeline loaded successfully!")
            logger.info("=" * 60)
            
            # Log memory usage
            if self.device == "cuda":
                allocated = torch.cuda.memory_allocated() / (1024**3)
                reserved = torch.cuda.memory_reserved() / (1024**3)
                logger.info(f"GPU Memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")
            
        except Exception as e:
            logger.error(f"Failed to load Leffa pipeline: {e}", exc_info=True)
            raise RuntimeError(f"Leffa loading failed: {e}")
    
    def _preprocess(
        self, 
        src_image: Image.Image, 
        garment_type: str,
        model_type: str = "viton_hd",
        control_type: str = "virtual_tryon"
    ) -> Tuple[Image.Image, Image.Image]:
        """
        Generate mask and densepose for the source image.
        
        This mirrors the preprocessing in LeffaPredictor.leffa_predict from app.py.
        
        Args:
            src_image: Person image (already resized to 768x1024)
            garment_type: "upper_body", "lower_body", or "dresses"
            model_type: "viton_hd" or "dress_code"
            control_type: "virtual_tryon" or "pose_transfer"
            
        Returns:
            Tuple of (mask, densepose) PIL Images
        """
        src_image_array = np.array(src_image)
        
        if control_type == "virtual_tryon":
            # Generate parsing and keypoints for virtual try-on
            src_image_rgb = src_image.convert("RGB")
            src_image_small = src_image_rgb.resize((384, 512))
            
            model_parse, _ = self.parsing(src_image_small)
            keypoints = self.openpose(src_image_small)
            
            # Generate mask based on model type
            if model_type == "viton_hd":
                mask = self.get_agnostic_mask_hd(model_parse, keypoints, garment_type)
            elif model_type == "dress_code":
                mask = self.get_agnostic_mask_dc(model_parse, keypoints, garment_type)
            
            mask = mask.resize((768, 1024))
            
        elif control_type == "pose_transfer":
            # For pose transfer, use full white mask
            mask = Image.fromarray(np.ones_like(src_image_array) * 255)
        
        # Generate densepose
        if control_type == "virtual_tryon":
            if model_type == "viton_hd":
                src_image_seg_array = self.densepose_predictor.predict_seg(src_image_array)[:, :, ::-1]
                densepose = Image.fromarray(src_image_seg_array)
            elif model_type == "dress_code":
                # dress_code uses IUV format
                src_image_iuv_array = self.densepose_predictor.predict_iuv(src_image_array)
                src_image_seg_array = src_image_iuv_array[:, :, 0:1]
                src_image_seg_array = np.concatenate([src_image_seg_array] * 3, axis=-1)
                densepose = Image.fromarray(src_image_seg_array)
        elif control_type == "pose_transfer":
            # Pose transfer uses full IUV
            src_image_iuv_array = self.densepose_predictor.predict_iuv(src_image_array)[:, :, ::-1]
            densepose = Image.fromarray(src_image_iuv_array)
        
        return mask, densepose
    
    def leffa_predict(
        self,
        src_image_path: str,
        ref_image_path: str,
        control_type: str = "virtual_tryon",
        ref_acceleration: bool = False,
        step: int = 50,
        scale: float = 2.5,
        seed: int = 42,
        vt_model_type: str = "viton_hd",
        vt_garment_type: str = "upper_body",
        vt_repaint: bool = False,
        preprocess_garment: bool = False
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Main Leffa prediction method - mirrors LeffaPredictor.leffa_predict from app.py.
        
        Args:
            src_image_path: Path to source/person image
            ref_image_path: Path to reference image (garment for VT, target pose for PT)
            control_type: "virtual_tryon" or "pose_transfer"
            ref_acceleration: Speed up reference UNet (may slightly reduce quality)
            step: Number of inference steps (default: 50)
            scale: Guidance scale (default: 2.5)
            seed: Random seed
            vt_model_type: For VT - "viton_hd" or "dress_code"
            vt_garment_type: For VT - "upper_body", "lower_body", or "dresses"
            vt_repaint: For VT - enable repaint mode
            preprocess_garment: For VT - preprocess garment image (PNG only)
            
        Returns:
            Tuple of (generated_image, mask, densepose) as numpy arrays
        """
        if not self._loaded:
            raise RuntimeError("Leffa not loaded. Call load_models() first.")
        
        # Open and resize the source image
        src_image = Image.open(src_image_path)
        src_image = self.resize_and_center(src_image, self.DEFAULT_WIDTH, self.DEFAULT_HEIGHT)
        
        # Handle garment preprocessing for virtual try-on
        if control_type == "virtual_tryon" and preprocess_garment:
            if isinstance(ref_image_path, str) and ref_image_path.lower().endswith('.png'):
                ref_image = self.preprocess_garment_image(ref_image_path)
            else:
                raise ValueError("Reference garment image must be a PNG file when preprocessing is enabled.")
        else:
            ref_image = Image.open(ref_image_path)
        
        ref_image = self.resize_and_center(ref_image, self.DEFAULT_WIDTH, self.DEFAULT_HEIGHT)
        
        # Preprocess: generate mask and densepose
        mask, densepose = self._preprocess(
            src_image, 
            vt_garment_type, 
            vt_model_type, 
            control_type
        )
        
        # Transform data
        data = {
            "src_image": [src_image],
            "ref_image": [ref_image],
            "mask": [mask],
            "densepose": [densepose],
        }
        data = self.transform(data)
        
        # Select inference model
        if control_type == "virtual_tryon":
            if vt_model_type == "viton_hd":
                inference = self.vt_inference_hd
            elif vt_model_type == "dress_code":
                inference = self.vt_inference_dc
        elif control_type == "pose_transfer":
            if self.pt_inference is None:
                raise RuntimeError("Pose transfer model not loaded. Call load_models(load_pose_transfer=True)")
            inference = self.pt_inference
        
        # Run inference
        output = inference(
            data,
            ref_acceleration=ref_acceleration,
            num_inference_steps=step,
            guidance_scale=scale,
            seed=seed,
            repaint=vt_repaint,
        )
        
        gen_image = output["generated_image"][0]
        return np.array(gen_image), np.array(mask), np.array(densepose)
    
    def leffa_predict_vt(
        self,
        src_image_path: str,
        ref_image_path: str,
        ref_acceleration: bool = False,
        step: int = 30,
        scale: float = 2.5,
        seed: int = 42,
        vt_model_type: str = "viton_hd",
        vt_garment_type: str = "upper_body",
        vt_repaint: bool = False,
        preprocess_garment: bool = False
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Virtual try-on prediction shortcut.
        
        Args:
            src_image_path: Path to person image
            ref_image_path: Path to garment image
            ref_acceleration: Speed up reference UNet
            step: Number of inference steps
            scale: Guidance scale
            seed: Random seed
            vt_model_type: "viton_hd" or "dress_code"
            vt_garment_type: "upper_body", "lower_body", or "dresses"
            vt_repaint: Enable repaint mode
            preprocess_garment: Preprocess garment image (PNG only)
            
        Returns:
            Tuple of (generated_image, mask, densepose) as numpy arrays
        """
        return self.leffa_predict(
            src_image_path,
            ref_image_path,
            "virtual_tryon",
            ref_acceleration,
            step,
            scale,
            seed,
            vt_model_type,
            vt_garment_type,
            vt_repaint,
            preprocess_garment,
        )
    
    def leffa_predict_pt(
        self,
        src_image_path: str,
        ref_image_path: str,
        ref_acceleration: bool = False,
        step: int = 30,
        scale: float = 2.5,
        seed: int = 42
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Pose transfer prediction shortcut.
        
        Args:
            src_image_path: Path to target pose person image
            ref_image_path: Path to source person image
            ref_acceleration: Speed up reference UNet
            step: Number of inference steps
            scale: Guidance scale
            seed: Random seed
            
        Returns:
            Tuple of (generated_image, mask, densepose) as numpy arrays
        """
        return self.leffa_predict(
            src_image_path,
            ref_image_path,
            "pose_transfer",
            ref_acceleration,
            step,
            scale,
            seed,
        )
    
    def __call__(
        self,
        person_image: Image.Image,
        garment_image: Image.Image,
        garment_type: str = "upper_body",
        model_type: str = "viton_hd",
        num_inference_steps: int = None,
        guidance_scale: float = None,
        seed: Optional[int] = 42,
        ref_acceleration: bool = False,
        repaint: bool = False,
    ) -> Dict:
        """
        Run Leffa virtual try-on (Pythonic interface with PIL Images).
        
        For file-path based interface matching app.py, use leffa_predict_vt().
        
        Args:
            person_image: Person image (PIL Image)
            garment_image: Garment image (PIL Image)
            garment_type: Garment type - "upper_body", "lower_body", or "dresses"
            model_type: Model variant - "viton_hd" or "dress_code"
            num_inference_steps: Number of diffusion steps (default: 30)
            guidance_scale: CFG strength (default: 2.5)
            seed: Random seed (default: 42)
            ref_acceleration: Speed up reference UNet (slight quality loss)
            repaint: Enable repaint mode
            
        Returns:
            Dictionary with keys: "result", "mask", "densepose"
        """
        try:
            if not self._loaded:
                raise RuntimeError("Leffa not loaded. Call load_models() first.")
            
            # Use defaults if not specified
            num_inference_steps = num_inference_steps or self.DEFAULT_STEPS
            guidance_scale = guidance_scale or self.DEFAULT_CFG
            
            logger.info(f"Starting Leffa inference...")
            logger.info(f"  Model: {model_type}, Garment type: {garment_type}")
            logger.info(f"  Steps: {num_inference_steps}, CFG: {guidance_scale}")
            
            # Resize images to 768x1024
            src_image = self.resize_and_center(person_image, self.DEFAULT_WIDTH, self.DEFAULT_HEIGHT)
            ref_image = self.resize_and_center(garment_image, self.DEFAULT_WIDTH, self.DEFAULT_HEIGHT)
            
            # Preprocess: generate mask and densepose
            logger.info("Generating mask and densepose...")
            mask, densepose = self._preprocess(src_image, garment_type, model_type, "virtual_tryon")
            
            # Transform data
            data = {
                "src_image": [src_image],
                "ref_image": [ref_image],
                "mask": [mask],
                "densepose": [densepose],
            }
            data = self.transform(data)
            
            # Select inference model
            if model_type == "dress_code" and self.vt_inference_dc:
                inference = self.vt_inference_dc
            else:
                inference = self.vt_inference_hd
            
            # Run inference
            logger.info("Running Leffa inference...")
            output = inference(
                data,
                ref_acceleration=ref_acceleration,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                seed=seed,
                repaint=repaint,
            )
            
            # Get result
            result_image = output["generated_image"][0]
            
            logger.info("Leffa inference completed successfully")
            
            return {
                "result": result_image,
                "mask": mask,
                "densepose": densepose
            }
            
        except Exception as e:
            logger.error(f"Leffa inference failed: {e}", exc_info=True)
            raise RuntimeError(f"Virtual try-on error: {e}")
    
    def pose_transfer(
        self,
        person_image: Image.Image,
        target_pose_image: Image.Image,
        num_inference_steps: int = None,
        guidance_scale: float = None,
        seed: Optional[int] = 42,
        ref_acceleration: bool = False,
    ) -> Dict:
        """
        Run Leffa pose transfer (Pythonic interface with PIL Images).
        
        Transfers the appearance of person_image to the pose in target_pose_image.
        
        Args:
            person_image: Source person image (PIL Image)
            target_pose_image: Target pose person image (PIL Image)
            num_inference_steps: Number of diffusion steps (default: 30)
            guidance_scale: CFG strength (default: 2.5)
            seed: Random seed (default: 42)
            ref_acceleration: Speed up reference UNet (slight quality loss)
            
        Returns:
            Dictionary with keys: "result", "mask", "densepose"
        """
        try:
            if not self._loaded:
                raise RuntimeError("Leffa not loaded. Call load_models() first.")
            
            if self.pt_inference is None:
                raise RuntimeError(
                    "Pose transfer model not loaded. "
                    "Call load_models(load_pose_transfer=True) first."
                )
            
            # Use defaults if not specified
            num_inference_steps = num_inference_steps or self.DEFAULT_STEPS
            guidance_scale = guidance_scale or self.DEFAULT_CFG
            
            logger.info(f"Starting Leffa pose transfer...")
            logger.info(f"  Steps: {num_inference_steps}, CFG: {guidance_scale}")
            
            # Resize images to 768x1024
            # Note: For pose transfer, src is target pose, ref is source person
            src_image = self.resize_and_center(target_pose_image, self.DEFAULT_WIDTH, self.DEFAULT_HEIGHT)
            ref_image = self.resize_and_center(person_image, self.DEFAULT_WIDTH, self.DEFAULT_HEIGHT)
            
            # Preprocess: generate mask and densepose for pose transfer
            logger.info("Generating mask and densepose...")
            mask, densepose = self._preprocess(src_image, "upper_body", "viton_hd", "pose_transfer")
            
            # Transform data
            data = {
                "src_image": [src_image],
                "ref_image": [ref_image],
                "mask": [mask],
                "densepose": [densepose],
            }
            data = self.transform(data)
            
            # Run inference
            logger.info("Running Leffa pose transfer inference...")
            output = self.pt_inference(
                data,
                ref_acceleration=ref_acceleration,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                seed=seed,
                repaint=False,
            )
            
            # Get result
            result_image = output["generated_image"][0]
            
            logger.info("Leffa pose transfer completed successfully")
            
            return {
                "result": result_image,
                "mask": mask,
                "densepose": densepose
            }
            
        except Exception as e:
            logger.error(f"Leffa pose transfer failed: {e}", exc_info=True)
            raise RuntimeError(f"Pose transfer error: {e}")
    
    @property
    def is_loaded(self) -> bool:
        """Check if models are loaded."""
        return self._loaded


class TryOnPipeline:
    """
    Wrapper class for backward compatibility.
    
    This wraps LeffaPipeline to provide the same interface as before.
    """
    
    def __init__(self, device: str = None):
        """
        Initialize the try-on pipeline.
        
        Args:
            device: Target device ("cuda", "cpu", or None for auto-detect)
        """
        self.device = device
        self._pipeline = None
        self._loaded = False
        
    def _ensure_loaded(self):
        """Ensure the pipeline is loaded."""
        if self._loaded:
            return
        
        logger.info("Loading Leffa pipeline...")
        self._pipeline = LeffaPipeline(device=self.device)
        self._pipeline.load_models()
        self._loaded = True
        logger.info("Leffa pipeline loaded successfully")
        
    def __call__(
        self, 
        person_image: Image.Image, 
        cloth_image: Image.Image, 
        mask: Image.Image = None,
        garment_type: str = "upper_body",
        num_inference_steps: int = 30,
        guidance_scale: float = 2.5,
        seed: int = 42,
    ) -> Image.Image:
        """
        Execute virtual try-on.
        
        Args:
            person_image: Person image (PIL Image)
            cloth_image: Garment image (PIL Image)
            mask: Optional mask image (ignored - Leffa handles internally)
            garment_type: Type of garment - "upper_body", "lower_body", or "dresses"
            num_inference_steps: Number of diffusion steps (default: 30)
            guidance_scale: CFG strength (default: 2.5)
            seed: Random seed (default: 42)
            
        Returns:
            Result image (PIL Image) with the person wearing the garment
        """
        try:
            self._ensure_loaded()
            
            logger.debug(f"Running try-on: garment_type={garment_type}, steps={num_inference_steps}")
            
            # Note: mask is ignored as Leffa handles masking internally
            if mask is not None:
                logger.debug("Mask provided but ignored - Leffa handles masking internally")
            
            result = self._pipeline(
                person_image=person_image,
                garment_image=cloth_image,
                garment_type=garment_type,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                seed=seed,
            )
            
            logger.debug("Try-on completed successfully")
            return result["result"]
            
        except Exception as e:
            logger.error(f"Try-on pipeline failed: {e}", exc_info=True)
            raise RuntimeError(f"Try-on pipeline error: {e}")
    
    @property
    def is_loaded(self) -> bool:
        """Check if the pipeline is loaded."""
        return self._loaded
