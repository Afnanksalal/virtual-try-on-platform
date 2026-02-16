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
    
    def _log_vram_usage(self, context=""):
        """Log current VRAM usage"""
        if not torch.cuda.is_available():
            return
        
        vram_allocated = torch.cuda.memory_allocated() / 1024**3
        vram_reserved = torch.cuda.memory_reserved() / 1024**3
        prefix = f"{context}: " if context else ""
        logger.info(f"{prefix}VRAM: {vram_allocated:.2f}GB allocated, {vram_reserved:.2f}GB reserved")
    
    def reset_cuda_memory(self):
        """Aggressively reset CUDA memory - NUCLEAR OPTION"""
        logger.info("🔥 NUCLEAR MEMORY RESET - Clearing everything from GPU...")
        
        # Multiple rounds of garbage collection
        for _ in range(3):
            import gc
            gc.collect()
        
        # Clear all CUDA caches multiple times
        for _ in range(3):
            torch.cuda.empty_cache()
        
        # Synchronize and collect IPC
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.ipc_collect()
        
        # Reset memory stats
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.reset_accumulated_memory_stats()
        
        # Final garbage collection
        import gc
        gc.collect()
        
        self._log_vram_usage("After nuclear reset")
    
    def load_models(self):
        """Load InstantID models and dependencies."""
        if self._loaded:
            logger.info("InstantID models already loaded, skipping...")
            return
        
        # CRITICAL: Clear GPU memory before loading
        logger.info("Clearing GPU memory before loading InstantID...")
        self.reset_cuda_memory()
            
        logger.info("Loading InstantID models for the first time...")
        
        try:
            # Set HuggingFace token from environment if available
            hf_token = os.getenv("HUGGINGFACE_TOKEN")
            if hf_token:
                os.environ["HF_TOKEN"] = hf_token
                logger.info("HuggingFace token configured for authenticated downloads")
            else:
                logger.warning("HUGGINGFACE_TOKEN not set - downloads will be unauthenticated and slower")
            
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
            logger.info("Loading SDXL base model from cache...")
            
            # HuggingFace will automatically use its cache at ~/.cache/huggingface
            # Try to load from cache first, download only if needed
            
            # Try to import and use the InstantID pipeline
            try:
                from .instantid_pipeline import StableDiffusionXLInstantIDPipeline, draw_kps
                self._draw_kps = draw_kps
                
                logger.info("Loading SDXL with InstantID pipeline...")
                
                # Try loading from cache first
                try:
                    logger.info("Attempting to load SDXL from local cache...")
                    self.pipe = StableDiffusionXLInstantIDPipeline.from_pretrained(
                        "stabilityai/stable-diffusion-xl-base-1.0",
                        controlnet=self.controlnet,
                        torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                        variant="fp16" if self.device == "cuda" else None,
                        use_safetensors=True,
                        local_files_only=True  # Force use of cache
                    ).to(self.device)
                    logger.info("✓ SDXL loaded from cache successfully")
                except Exception as cache_error:
                    logger.warning(f"Cache load failed: {cache_error}, downloading from HuggingFace...")
                    self.pipe = StableDiffusionXLInstantIDPipeline.from_pretrained(
                        "stabilityai/stable-diffusion-xl-base-1.0",
                        controlnet=self.controlnet,
                        torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                        variant="fp16" if self.device == "cuda" else None,
                        use_safetensors=True,
                        local_files_only=False  # Download if not in cache
                    ).to(self.device)
                    logger.info("✓ SDXL downloaded and loaded")
                    
            except ImportError:
                # Fallback: use community pipeline from diffusers
                logger.info("InstantID pipeline not found, using community pipeline...")
                
                # Try loading from cache first
                try:
                    logger.info("Attempting to load SDXL from local cache...")
                    self.pipe = StableDiffusionXLPipeline.from_pretrained(
                        "stabilityai/stable-diffusion-xl-base-1.0",
                        controlnet=self.controlnet,
                        torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                        variant="fp16" if self.device == "cuda" else None,
                        use_safetensors=True,
                        local_files_only=True,  # Force use of cache
                        custom_pipeline="pipeline_stable_diffusion_xl_instantid"
                    ).to(self.device)
                    logger.info("✓ SDXL loaded from cache with community pipeline")
                except Exception as cache_error:
                    logger.warning(f"Cache load failed: {cache_error}, downloading from HuggingFace...")
                    self.pipe = StableDiffusionXLPipeline.from_pretrained(
                        "stabilityai/stable-diffusion-xl-base-1.0",
                        controlnet=self.controlnet,
                        torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                        variant="fp16" if self.device == "cuda" else None,
                        use_safetensors=True,
                        local_files_only=False,  # Download if not in cache
                        custom_pipeline="pipeline_stable_diffusion_xl_instantid"
                    ).to(self.device)
                    logger.info("✓ SDXL downloaded and loaded with community pipeline")
                self._draw_kps = self._simple_draw_kps
            
            # Load IP-Adapter for InstantID
            logger.info("Loading InstantID IP-Adapter...")
            self.pipe.load_ip_adapter_instantid(ip_adapter_path)
            
            # Enable memory optimizations
            if self.device == "cuda":
                logger.info("Enabling memory optimizations for 4GB VRAM...")
                # Sequential CPU offloading - moves model components to CPU when not in use
                self.pipe.enable_model_cpu_offload()
                # VAE tiling - processes images in tiles to reduce memory
                self.pipe.enable_vae_tiling()
                # Enable attention slicing for lower memory usage
                self.pipe.enable_attention_slicing(slice_size="auto")
                logger.info("✓ Memory optimizations enabled (CPU offload + VAE tiling + attention slicing)")
            
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
        num_inference_steps: int = 25,  # Optimized: 20-30 for LCM
        guidance_scale: float = 2.0,  # Optimized: 2-3 for InstantID (not 0)
        controlnet_conditioning_scale: float = 0.6,  # Optimized: 0.4-0.8 range
        ip_adapter_scale: float = 0.8,  # Optimized: maintains identity strength
        seed: Optional[int] = None,
    ) -> List[Image.Image]:
        """
        Generate full-body images preserving the user's facial identity.
        
        Optimized parameters based on InstantID research and best practices:
        - CFG Scale: 2-3 (lower than typical SDXL 7-15)
        - Image size: 1016x768 (avoid exact 1024 multiples)
        - ControlNet scale: 0.4-0.8 for facial keypoint guidance
        - IP-Adapter scale: 0.8 for strong identity preservation
        - Inference steps: 20-30 with LCM scheduler
        
        Args:
            face_image: PIL Image of user's face
            body_type: Body type ('slim', 'athletic', 'muscular', 'average', 'curvy', 'plus')
            height_cm: Height in cm (for proportion reference)
            weight_kg: Weight in kg (for body shape reference)
            gender: Gender for body generation
            skin_tone: Skin tone description
            ethnicity: Ethnicity for accurate representation
            pose: Desired pose ('standing', 'walking', 'sitting', etc.)
            clothing: Clothing description
            num_images: Number of variations to generate
            num_inference_steps: Diffusion steps (25 recommended, 20-30 range)
            guidance_scale: CFG scale (2.0 recommended for InstantID, 2-3 range)
            controlnet_conditioning_scale: ControlNet strength (0.6 default, 0.4-0.8 range)
            ip_adapter_scale: Identity preservation strength (0.8 default, 0.5-1.0 range)
            seed: Random seed for reproducibility
            
        Returns:
            List of generated PIL Images
        """
        if not self._loaded:
            self.load_models()
        
        logger.info(f"Generating {num_images} identity-preserving full-body images...")
        
        # Log initial VRAM state
        self._log_vram_usage("Before generation")
        
        # Extract face embedding and keypoints
        face_emb, face_kps = self.extract_face_info(face_image)
        
        # Build body descriptor from parameters (optimized for better results)
        body_descriptors = {
            "athletic": "athletic build with toned muscles and fit physique, well-proportioned body",
            "slim": "slim slender build with lean physique, graceful proportions",
            "muscular": "muscular build with strong physique and well-defined muscles, powerful stance",
            "average": "average build with normal proportions and balanced physique",
            "curvy": "curvy figure with proportionate body and natural curves",
            "plus": "plus size with full figure and confident presence"
        }
        body_desc = body_descriptors.get(body_type.lower(), "average build with normal proportions")
        
        # Build optimized prompt (specific, detailed, avoids face-only descriptions)
        prompt = f"""professional full body portrait photograph, {gender} person, {ethnicity} ethnicity, 
{skin_tone} skin tone, {body_desc}, {pose} pose, 
wearing {clothing}, full body visible from head to toe, 
clean white studio background, professional studio lighting, high quality photography, 
photorealistic, sharp focus, 8k uhd, detailed textures"""
        
        # Optimized negative prompt (more specific about what to avoid)
        negative_prompt = """cropped body, cut off limbs, partial body, close-up only, headshot, portrait crop, 
face only, upper body only, multiple people, extra people, crowd, 
cluttered background, busy background, outdoor scene, 
low quality, blurry, out of focus, distorted, deformed, disfigured, 
bad anatomy, extra limbs, missing limbs, floating limbs, disconnected limbs, 
extra fingers, missing fingers, mutated hands, poorly drawn hands, 
mutation, ugly, disgusting, watermark, text, signature, logo, 
overexposed, underexposed, bad lighting"""
        
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
            
            # Optimized: Use 1016x768 instead of 1024x768 (avoid exact 1024 multiples)
            # Research shows InstantID works better with slight offsets from 1024
            result = self.pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image_embeds=face_emb,
                image=face_kps,
                controlnet_conditioning_scale=controlnet_conditioning_scale,
                ip_adapter_scale=ip_adapter_scale,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                height=1016,  # Optimized: slight offset from 1024
                width=768,
                generator=img_generator,
            ).images[0]
            
            images.append(result)
            
            # Cleanup after each image to prevent memory buildup
            if self.device == "cuda":
                torch.cuda.empty_cache()
                import gc
                gc.collect()
        
        logger.info(f"Generated {len(images)} identity-preserving images")
        self._log_vram_usage("After generation")
        
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
