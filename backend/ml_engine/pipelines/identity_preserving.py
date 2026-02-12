"""
Identity-Preserving Body Generation Pipeline using InstantID

This pipeline generates full-body images while preserving the user's facial identity.
Unlike simple cut-and-paste, InstantID embeds the face features into the generation
process, resulting in natural-looking images where the face is natively generated.

Key features:
- Face embedding extraction using InsightFace
- Identity-preserving generation with InstantID + SDXL
- ControlNet for pose guidance
- LCM-LoRA for fast inference
"""

import torch
import numpy as np
from PIL import Image
import cv2
from typing import Optional, Dict, List, Tuple
import os

from app.core.logging_config import get_logger

logger = get_logger("ml.identity_preserving")

# Lazy imports for heavy dependencies
_instantid_pipeline = None
_face_analyzer = None
_controlnet = None


class IdentityPreservingPipeline:
    """
    Pipeline for generating full-body images that preserve user's facial identity.
    
    Uses InstantID which combines:
    - Face embedding from InsightFace antelopev2
    - ControlNet for facial keypoints/pose
    - IP-Adapter for identity injection
    """
    
    def __init__(self, device: Optional[str] = None):
        """
        Initialize the identity-preserving pipeline.
        
        Args:
            device: Device to run on ('cuda', 'cpu', or None for auto-detect)
        """
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        self.pipe = None
        self.face_analyzer = None
        self.controlnet = None
        self._loaded = False
        
        logger.info(f"IdentityPreservingPipeline initialized on device: {self.device}")
    
    def load_models(self):
        """Load InstantID models and dependencies."""
        if self._loaded:
            return
            
        logger.info("Loading InstantID models...")
        
        try:
            from huggingface_hub import hf_hub_download
            from diffusers import ControlNetModel, StableDiffusionXLPipeline, LCMScheduler
            from diffusers.utils import load_image
            from insightface.app import FaceAnalysis
            
            # Set up model paths
            model_dir = os.path.join(os.path.dirname(__file__), "..", "weights", "instantid")
            os.makedirs(model_dir, exist_ok=True)
            
            # Download InstantID checkpoints if not present
            logger.info("Checking/downloading InstantID checkpoints...")
            
            # Download ControlNet model
            controlnet_path = os.path.join(model_dir, "ControlNetModel")
            if not os.path.exists(os.path.join(controlnet_path, "config.json")):
                logger.info("Downloading InstantID ControlNet...")
                hf_hub_download(
                    repo_id="InstantX/InstantID",
                    filename="ControlNetModel/config.json",
                    local_dir=model_dir
                )
                hf_hub_download(
                    repo_id="InstantX/InstantID",
                    filename="ControlNetModel/diffusion_pytorch_model.safetensors",
                    local_dir=model_dir
                )
            
            # Download IP-Adapter
            ip_adapter_path = os.path.join(model_dir, "ip-adapter.bin")
            if not os.path.exists(ip_adapter_path):
                logger.info("Downloading InstantID IP-Adapter...")
                hf_hub_download(
                    repo_id="InstantX/InstantID",
                    filename="ip-adapter.bin",
                    local_dir=model_dir
                )
            
            # Initialize FaceAnalysis (antelopev2)
            logger.info("Initializing face analyzer (antelopev2)...")
            face_model_dir = os.path.join(model_dir, "models")
            os.makedirs(face_model_dir, exist_ok=True)
            
            self.face_analyzer = FaceAnalysis(
                name='antelopev2',
                root=model_dir,
                providers=['CUDAExecutionProvider', 'CPUExecutionProvider'] if self.device == "cuda" else ['CPUExecutionProvider']
            )
            self.face_analyzer.prepare(ctx_id=0 if self.device == "cuda" else -1, det_size=(640, 640))
            
            # Load ControlNet
            logger.info("Loading ControlNet for InstantID...")
            self.controlnet = ControlNetModel.from_pretrained(
                controlnet_path,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
            )
            
            # Load base SDXL model with ControlNet
            logger.info("Loading SDXL base model...")
            
            # Try to import and use the InstantID pipeline
            try:
                from .instantid_pipeline import StableDiffusionXLInstantIDPipeline, draw_kps
                self._draw_kps = draw_kps
                
                self.pipe = StableDiffusionXLInstantIDPipeline.from_pretrained(
                    "stabilityai/stable-diffusion-xl-base-1.0",
                    controlnet=self.controlnet,
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
                ).to(self.device)
            except ImportError:
                # Fallback: use community pipeline from diffusers
                logger.info("Using community InstantID pipeline...")
                self.pipe = StableDiffusionXLPipeline.from_pretrained(
                    "stabilityai/stable-diffusion-xl-base-1.0",
                    controlnet=self.controlnet,
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                    custom_pipeline="pipeline_stable_diffusion_xl_instantid"
                ).to(self.device)
                self._draw_kps = self._simple_draw_kps
            
            # Load IP-Adapter for InstantID
            logger.info("Loading InstantID IP-Adapter...")
            self.pipe.load_ip_adapter_instantid(ip_adapter_path)
            
            # Enable memory optimizations
            if self.device == "cuda":
                self.pipe.enable_model_cpu_offload()
                self.pipe.enable_vae_tiling()
            
            # Optional: Load LCM-LoRA for faster inference
            try:
                logger.info("Loading LCM-LoRA for faster inference...")
                self.pipe.scheduler = LCMScheduler.from_config(self.pipe.scheduler.config)
            except Exception as e:
                logger.warning(f"LCM-LoRA not loaded: {e}")
            
            self._loaded = True
            logger.info("InstantID models loaded successfully!")
            
        except Exception as e:
            logger.error(f"Failed to load InstantID models: {e}", exc_info=True)
            raise RuntimeError(f"InstantID model loading failed: {e}")
    
    def _simple_draw_kps(self, image: Image.Image, kps: np.ndarray) -> Image.Image:
        """Simple keypoint visualization fallback."""
        img = np.array(image)
        kps_colors = [
            (255, 0, 0),    # left eye
            (0, 255, 0),    # right eye
            (0, 0, 255),    # nose
            (255, 255, 0),  # left mouth
            (255, 0, 255),  # right mouth
        ]
        for i, (x, y) in enumerate(kps[:5]):
            cv2.circle(img, (int(x), int(y)), 3, kps_colors[i % len(kps_colors)], -1)
        return Image.fromarray(img)
    
    def extract_face_info(self, face_image: Image.Image) -> Tuple[np.ndarray, Image.Image]:
        """
        Extract face embedding and keypoints from an image.
        
        Args:
            face_image: PIL Image containing a face
            
        Returns:
            Tuple of (face_embedding, keypoints_image)
        """
        if not self._loaded:
            self.load_models()
        
        # Convert to BGR for OpenCV
        img_array = np.array(face_image)
        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        
        # Get face info
        faces = self.face_analyzer.get(img_bgr)
        
        if len(faces) == 0:
            raise ValueError("No face detected in the image")
        
        # Use largest face
        face = max(faces, key=lambda x: (x['bbox'][2] - x['bbox'][0]) * (x['bbox'][3] - x['bbox'][1]))
        
        face_emb = face['embedding']
        face_kps = self._draw_kps(face_image, face['kps'])
        
        return face_emb, face_kps
    
    def generate(
        self,
        face_image: Image.Image,
        body_type: str = "average",
        height_cm: float = 170.0,
        weight_kg: float = 65.0,
        gender: str = "female",
        skin_tone: str = "medium",
        ethnicity: str = "mixed",
        pose: str = "standing",
        clothing: str = "casual minimal",
        num_images: int = 4,
        num_inference_steps: int = 20,
        guidance_scale: float = 0.0,  # InstantID with LCM uses 0
        controlnet_conditioning_scale: float = 0.8,
        ip_adapter_scale: float = 0.8,
        seed: Optional[int] = None,
    ) -> List[Image.Image]:
        """
        Generate full-body images preserving the user's facial identity.
        
        Args:
            face_image: PIL Image of user's face
            body_type: Body type ('slim', 'athletic', 'muscular', 'average', etc.)
            height_cm: Height in cm (for proportion reference)
            weight_kg: Weight in kg (for body shape reference)
            gender: Gender for body generation
            skin_tone: Skin tone description
            ethnicity: Ethnicity for accurate representation
            pose: Desired pose ('standing', 'walking', 'sitting', etc.)
            clothing: Clothing description
            num_images: Number of variations to generate
            num_inference_steps: Diffusion steps (20 recommended with LCM)
            guidance_scale: CFG scale (0.0 with LCM, higher without)
            controlnet_conditioning_scale: ControlNet strength (0.8 default)
            ip_adapter_scale: Identity preservation strength (0.8 default)
            seed: Random seed for reproducibility
            
        Returns:
            List of generated PIL Images
        """
        if not self._loaded:
            self.load_models()
        
        logger.info(f"Generating {num_images} identity-preserving full-body images...")
        
        # Extract face embedding and keypoints
        face_emb, face_kps = self.extract_face_info(face_image)
        
        # Build body descriptor from parameters
        body_descriptors = {
            "athletic": "athletic build, toned muscles, fit physique",
            "slim": "slim build, lean physique, slender body",
            "muscular": "muscular build, strong physique, well-defined muscles",
            "average": "average build, normal proportions, medium physique",
            "curvy": "curvy figure, proportionate body",
            "plus": "plus size, full figure"
        }
        body_desc = body_descriptors.get(body_type.lower(), "average build")
        
        # Build prompt
        prompt = f"""professional full body photograph of a {gender}, {ethnicity} ethnicity, 
{skin_tone} skin tone, {body_desc}, {pose} pose, wearing {clothing}, 
plain white studio background, high quality, professional photography, 
well-lit, realistic, 8k, detailed"""
        
        negative_prompt = """cropped, cut off, partial body, close-up face only, headshot only,
multiple people, cluttered background, low quality, blurry, distorted, deformed,
bad anatomy, extra limbs, missing limbs, floating limbs, disconnected limbs,
mutation, ugly, disgusting, disfigured, watermark, text"""
        
        logger.debug(f"Generation prompt: {prompt[:200]}...")
        
        # Set seed for reproducibility
        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)
        
        # Generate images
        images = []
        for i in range(num_images):
            logger.info(f"Generating image {i+1}/{num_images}...")
            
            # Vary seed slightly for each image
            img_generator = generator
            if seed is not None:
                img_generator = torch.Generator(device=self.device).manual_seed(seed + i)
            
            result = self.pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image_embeds=face_emb,
                image=face_kps,
                controlnet_conditioning_scale=controlnet_conditioning_scale,
                ip_adapter_scale=ip_adapter_scale,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                height=1024,
                width=768,
                generator=img_generator,
            ).images[0]
            
            images.append(result)
        
        logger.info(f"Generated {len(images)} identity-preserving images")
        return images
    
    def __call__(
        self,
        face_image: Image.Image,
        body_params: Dict,
        gemini_analysis: Optional[Dict] = None,
        num_images: int = 4,
        **kwargs
    ) -> Dict:
        """
        Main entry point for identity-preserving generation.
        
        Args:
            face_image: User's face photo
            body_params: Dict with body type preferences from user
            gemini_analysis: Optional Gemini Vision analysis of user's features
            num_images: Number of variations to generate
            **kwargs: Additional generation parameters
            
        Returns:
            Dict with generated images and metadata
        """
        # Merge Gemini analysis with user params (Gemini fills gaps)
        params = {
            "body_type": body_params.get("body_type", "average"),
            "height_cm": body_params.get("height_cm", 170),
            "weight_kg": body_params.get("weight_kg", 65),
            "gender": body_params.get("gender", "female"),
            "ethnicity": body_params.get("ethnicity", "mixed"),
            "skin_tone": body_params.get("skin_tone", "medium"),
        }
        
        # Override with Gemini analysis if available
        if gemini_analysis:
            # Gemini provides more accurate skin tone, ethnicity detection
            if "skin_tone" in gemini_analysis:
                params["skin_tone"] = gemini_analysis["skin_tone"]
            if "ethnicity" in gemini_analysis:
                params["ethnicity"] = gemini_analysis["ethnicity"]
            if "facial_features" in gemini_analysis:
                # Can use this for additional prompt refinement
                pass
        
        # Generate images
        images = self.generate(
            face_image=face_image,
            num_images=num_images,
            **params,
            **kwargs
        )
        
        return {
            "images": images,
            "params_used": params,
            "count": len(images)
        }


# Singleton instance for efficiency
_pipeline_instance: Optional[IdentityPreservingPipeline] = None


def get_identity_pipeline() -> IdentityPreservingPipeline:
    """Get or create the singleton pipeline instance."""
    global _pipeline_instance
    if _pipeline_instance is None:
        _pipeline_instance = IdentityPreservingPipeline()
    return _pipeline_instance
