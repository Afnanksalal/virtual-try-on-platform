import torch
from transformers import SegformerImageProcessor, AutoModelForSemanticSegmentation
import os
import threading
import time
from typing import Tuple, Dict, Any, Optional
from app.core.logging_config import get_logger

logger = get_logger("ml.loader")

# Set HuggingFace token from environment if available
hf_token = os.getenv("HUGGINGFACE_TOKEN")
if hf_token:
    os.environ["HF_TOKEN"] = hf_token
    logger.info("HuggingFace token configured for authenticated downloads")
else:
    logger.warning("HUGGINGFACE_TOKEN not set in .env - downloads will be unauthenticated and slower")

# Try importing optional dependencies
try:
    from controlnet_aux import OpenposeDetector
    OPENPOSE_AVAILABLE = True
except ImportError:
    OpenposeDetector = None
    OPENPOSE_AVAILABLE = False
    logger.warning("controlnet_aux not installed. OpenPose will not be available.")

# Leffa pipeline import
try:
    from ml_engine.pipelines.tryon import LeffaPipeline, download_leffa_checkpoints, LEFFA_PATH
    LEFFA_AVAILABLE = True if LEFFA_PATH else False
except ImportError:
    LeffaPipeline = None
    download_leffa_checkpoints = None
    LEFFA_PATH = None
    LEFFA_AVAILABLE = False
    logger.warning("Leffa pipeline not available.")

# TripoSR import
try:
    import sys
    triposr_path = os.path.join(os.path.dirname(__file__), "..", "3d", "TripoSR")
    if os.path.exists(triposr_path) and triposr_path not in sys.path:
        sys.path.insert(0, triposr_path)
    from tsr.system import TSR
    TRIPOSR_AVAILABLE = True
except ImportError:
    TSR = None
    TRIPOSR_AVAILABLE = False
    logger.warning("TripoSR not available.")


class ModelLoader:
    """
    Singleton model loader for managing ML model lifecycle.
    
    Handles loading, caching, and memory management for:
    - Leffa virtual try-on pipeline
    - TripoSR 3D reconstruction model
    - Segmentation models
    - Pose estimation models
    """
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        # Thread-safe singleton implementation using double-checked locking
        if cls._instance is None:
            with cls._lock:
                # Double-check inside the lock to prevent race conditions
                if cls._instance is None:
                    cls._instance = super(ModelLoader, cls).__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        # Ensure initialization happens only once
        if self._initialized:
            return
            
        with self._lock:
            if self._initialized:
                return
                
            self.models = {}  # Simple model storage
            self.device = self._detect_device()
            self._initialized = True
            logger.info(f"ModelLoader initialized on device: {self.device}")
    
    def _detect_device(self) -> str:
        """Detect the best available device."""
        use_gpu = os.getenv("USE_GPU", "true").lower() == "true"
        
        if use_gpu and torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            logger.info(f"CUDA device detected: {device_name}")
            return "cuda"
        else:
            if use_gpu and not torch.cuda.is_available():
                logger.warning("USE_GPU=true but CUDA not available. Falling back to CPU.")
            return "cpu"
    
    @classmethod
    def get_instance(cls) -> 'ModelLoader':
        """Thread-safe method to get the singleton instance."""
        return cls()
    
    def get_memory_usage(self) -> Dict[str, Any]:
        """Get current memory usage statistics."""
        stats = {
            "device": self.device,
            "loaded_models": list(self.models.keys()),
            "model_count": len(self.models)
        }
        
        if self.device == "cuda" and torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated(0)
            reserved = torch.cuda.memory_reserved(0)
            total = torch.cuda.get_device_properties(0).total_memory
            
            stats.update({
                "gpu_memory_allocated_mb": allocated / (1024 ** 2),
                "gpu_memory_reserved_mb": reserved / (1024 ** 2),
                "gpu_memory_total_mb": total / (1024 ** 2),
                "gpu_memory_usage_percent": (allocated / total * 100) if total > 0 else 0.0
            })
        
        return stats

    def load_segmentation(self) -> Tuple[SegformerImageProcessor, AutoModelForSemanticSegmentation]:
        """Load segmentation model (Segformer for clothes parsing)."""
        model_name = "segmentation"
        
        if model_name not in self.models:
            logger.info("Loading Segmentation Model (Segformer)...")
            start_time = time.time()
            
            processor = SegformerImageProcessor.from_pretrained("mattmdjaga/segformer_b2_clothes")
            model = AutoModelForSemanticSegmentation.from_pretrained("mattmdjaga/segformer_b2_clothes")
            model.to(self.device)
            
            self.models[model_name] = (processor, model)
            load_time = time.time() - start_time
            logger.info(f"Segmentation model loaded successfully in {load_time:.2f}s")
        
        return self.models[model_name]

    def load_pose(self):
        """Load pose estimation model (OpenPose)."""
        model_name = "pose"
        
        if model_name not in self.models:
            if not OPENPOSE_AVAILABLE:
                raise ImportError(
                    "OpenPose not available. Install with: pip install controlnet-aux"
                )
            
            logger.info("Loading Pose Model (OpenPose)...")
            start_time = time.time()
            
            model = OpenposeDetector.from_pretrained("lllyasviel/ControlNet")
            
            self.models[model_name] = model
            load_time = time.time() - start_time
            logger.info(f"Pose model loaded successfully in {load_time:.2f}s")
        
        return self.models[model_name]
    
    def ensure_leffa_checkpoints(self) -> bool:
        """
        Ensure Leffa checkpoints are downloaded.
        
        This should be called on startup to download checkpoints before loading models.
        Downloads are cached, so subsequent calls are fast.
        
        Returns:
            True if checkpoints are available, False otherwise
        """
        if not LEFFA_AVAILABLE:
            logger.warning(
                "Leffa repository not found. Clone it at project root:\n"
                "  git clone https://github.com/franciszzj/Leffa\n"
                "The Leffa folder should be at the same level as backend/ and frontend/"
            )
            return False
        
        try:
            logger.info("Ensuring Leffa checkpoints are downloaded...")
            download_leffa_checkpoints()
            logger.info("Leffa checkpoints are ready")
            return True
        except Exception as e:
            logger.error(f"Failed to download Leffa checkpoints: {e}", exc_info=True)
            return False

    def load_tryon(self, auto_download: bool = True, load_pose_transfer: bool = False) -> 'LeffaPipeline':
        """
        Load Leffa virtual try-on pipeline.
        
        On first run, this will download checkpoints from HuggingFace (several GB).
        
        Args:
            auto_download: Whether to auto-download checkpoints if missing (default: True)
            load_pose_transfer: Whether to also load pose transfer model (default: False)
        
        Returns:
            LeffaPipeline: Loaded Leffa pipeline
        """
        model_name = "tryon"
        
        if model_name not in self.models:
            if not LEFFA_AVAILABLE:
                raise ImportError(
                    "Leffa not available. Clone the repository at project root:\n"
                    "  git clone https://github.com/franciszzj/Leffa\n"
                    "The Leffa folder should be at the same level as backend/ and frontend/"
                )
            
            logger.info("Loading Leffa Virtual Try-On Pipeline...")
            start_time = time.time()
            
            try:
                # Initialize Leffa pipeline (auto-downloads checkpoints if needed)
                pipe = LeffaPipeline(device=self.device, auto_download=auto_download)
                
                # Load models (preprocessors + virtual try-on + optionally pose transfer)
                pipe.load_models(load_pose_transfer=load_pose_transfer)
                
                self.models[model_name] = pipe
                
                load_time = time.time() - start_time
                logger.info(f"Leffa pipeline loaded successfully in {load_time:.2f}s")
                
            except Exception as e:
                logger.error(f"Error loading Leffa model: {e}", exc_info=True)
                raise
        
        return self.models[model_name]
    def load_triposr(
        self,
        pretrained_model_name_or_path: str = "stabilityai/TripoSR",
        chunk_size: int = 8192,
        use_fp16: bool = True,
        use_enhanced: bool = True
    ) -> 'TSR':
        """
        Load TripoSR 3D reconstruction model (Enhanced version with quality improvements).

        TripoSR is a single-image 3D reconstruction model that generates 3D meshes
        from 2D images. The enhanced version includes adaptive thresholding, multi-resolution
        extraction, and mesh post-processing for better quality.

        Args:
            pretrained_model_name_or_path: HuggingFace model ID or local path
                                          (default: "stabilityai/TripoSR")
            chunk_size: Evaluation chunk size for surface extraction and rendering.
                       Smaller values reduce VRAM usage but increase computation time.
                       0 for no chunking. (default: 8192)
            use_fp16: Use half precision (FP16) for memory optimization (default: True)
            use_enhanced: Use enhanced TripoSR with quality improvements (default: True)

        Returns:
            TSR or EnhancedTSR: Loaded TripoSR model

        Raises:
            ImportError: If TripoSR is not available
            RuntimeError: If model loading fails
        """
        model_name = "triposr"

        if model_name not in self.models:
            if not TRIPOSR_AVAILABLE:
                raise ImportError(
                    "TripoSR not available. The TripoSR repository should be at:\n"
                    "  backend/3d/TripoSR/\n"
                    "Clone it with:\n"
                    "  cd backend/3d\n"
                    "  git clone https://github.com/VAST-AI-Research/TripoSR.git"
                )

            logger.info("Loading TripoSR 3D Reconstruction Model...")
            start_time = time.time()

            try:
                # Try to use enhanced version first
                if use_enhanced:
                    try:
                        from tsr.enhanced_system import create_enhanced_triposr
                        logger.info("Loading Enhanced TripoSR with quality improvements...")
                        model = create_enhanced_triposr(pretrained_model_name_or_path)
                        logger.info("✓ Using Enhanced TripoSR (adaptive threshold, multi-res, post-processing)")
                    except ImportError as e:
                        logger.warning(f"Enhanced TripoSR not available: {e}, falling back to standard")
                        from tsr.system import TSR
                        model = TSR.from_pretrained(
                            pretrained_model_name_or_path,
                            config_name="config.yaml",
                            weight_name="model.ckpt"
                        )
                        logger.info("✓ Using Standard TripoSR")
                else:
                    # Use standard TripoSR
                    from tsr.system import TSR
                    model = TSR.from_pretrained(
                        pretrained_model_name_or_path,
                        config_name="config.yaml",
                        weight_name="model.ckpt"
                    )
                    logger.info("✓ Using Standard TripoSR")

                # Configure chunk size for memory management
                model.renderer.set_chunk_size(chunk_size)
                logger.info(f"TripoSR chunk size set to: {chunk_size}")

                # Move to device
                model.to(self.device)

                # Convert to FP16 if requested and on CUDA
                # Note: FP16 can cause dtype mismatches with inputs, so we keep it as FP32
                # The memory savings are minimal compared to the compatibility issues
                if use_fp16 and self.device == "cuda" and False:  # Disabled for now
                    model = model.half()
                    logger.info("TripoSR converted to FP16 for memory optimization")
                else:
                    logger.info("TripoSR kept in FP32 for compatibility")

                self.models[model_name] = model

                load_time = time.time() - start_time
                logger.info(f"TripoSR loaded successfully in {load_time:.2f}s")

                # Log memory usage
                if self.device == "cuda":
                    memory_stats = self.get_memory_usage()
                    logger.info(
                        f"GPU memory after loading TripoSR: "
                        f"{memory_stats['gpu_memory_allocated_mb']:.2f}MB / "
                        f"{memory_stats['gpu_memory_total_mb']:.2f}MB "
                        f"({memory_stats['gpu_memory_usage_percent']:.1f}%)"
                    )

            except Exception as e:
                logger.error(f"Error loading TripoSR model: {e}", exc_info=True)
                raise RuntimeError(f"Failed to load TripoSR: {e}") from e

        return self.models[model_name]

    def load_sam2(
        self,
        checkpoint_path: Optional[str] = None,
        model_cfg: str = "sam2/sam2_hiera_l",
        use_fp16: bool = True
    ) -> 'SAM2ImagePredictor':
        """
        Load SAM2 (Segment Anything Model 2) for image segmentation.

        SAM2 is a foundation model for promptable visual segmentation that can
        segment any object in an image. It's used for precise background removal
        and object extraction in the 3D reconstruction pipeline.

        Args:
            checkpoint_path: Path to SAM2 checkpoint file. If None, uses default
                           location: backend/3d/models/sam2_hiera_large.pt
            model_cfg: Config file name for SAM2 model architecture
                      (default: "sam2/sam2_hiera_l" - relative to configs directory)
            use_fp16: Use half precision (FP16) for memory optimization on CUDA
                     (default: True)

        Returns:
            SAM2ImagePredictor: Loaded SAM2 image predictor

        Raises:
            ImportError: If SAM2 is not installed
            FileNotFoundError: If checkpoint file doesn't exist
            RuntimeError: If model loading fails
        """
        model_name = "sam2"

        if model_name not in self.models:
            logger.info("Loading SAM2 Segmentation Model...")
            start_time = time.time()

            try:
                # Import SAM2 modules
                try:
                    import sam2
                    from sam2.sam2_image_predictor import SAM2ImagePredictor
                except ImportError as e:
                    raise ImportError(
                        "SAM2 not installed. Install with:\n"
                        "  pip install git+https://github.com/facebookresearch/segment-anything-2.git"
                    ) from e

                # Determine checkpoint path
                if checkpoint_path is None:
                    # Default to local checkpoint
                    checkpoint_path = os.path.join(
                        os.path.dirname(__file__),
                        "..",
                        "3d",
                        "models",
                        "sam2_hiera_large.pt"
                    )
                    checkpoint_path = os.path.abspath(checkpoint_path)

                # Verify checkpoint exists
                if not os.path.exists(checkpoint_path):
                    raise FileNotFoundError(
                        f"SAM2 checkpoint not found at: {checkpoint_path}\n"
                        f"Download it with:\n"
                        f"  from huggingface_hub import hf_hub_download\n"
                        f"  hf_hub_download(repo_id='facebook/sam2-hiera-large', "
                        f"filename='sam2_hiera_large.pt', local_dir='./backend/3d/models')"
                    )

                logger.info(f"Loading SAM2 from checkpoint: {checkpoint_path}")

                # Build SAM2 model using Hydra config
                # Hydra needs to be initialized with the correct config path
                from hydra import initialize_config_dir, compose
                from hydra.core.global_hydra import GlobalHydra
                
                # Get SAM2 configs directory
                sam2_configs_dir = os.path.join(
                    os.path.dirname(sam2.__file__),
                    "configs"
                )
                sam2_configs_dir = os.path.abspath(sam2_configs_dir)
                
                # Initialize Hydra with SAM2 configs directory
                # Clear any existing Hydra instance first
                GlobalHydra.instance().clear()
                
                with initialize_config_dir(config_dir=sam2_configs_dir, version_base="1.2"):
                    # Compose config
                    cfg = compose(config_name=model_cfg)
                    from omegaconf import OmegaConf
                    OmegaConf.resolve(cfg)
                    
                    # Instantiate model
                    from hydra.utils import instantiate
                    sam2_model = instantiate(cfg.model, _recursive_=True)
                    
                    # Load checkpoint
                    from sam2.build_sam import _load_checkpoint
                    _load_checkpoint(sam2_model, checkpoint_path)
                    
                    # FORCE CPU for 4GB VRAM optimization (used once, doesn't need GPU speed)
                    sam2_model = sam2_model.to('cpu')
                    logger.info("✓ SAM2 forced to CPU for memory optimization")

                # Wrap in predictor for easier inference
                predictor = SAM2ImagePredictor(sam2_model)

                self.models[model_name] = predictor

                load_time = time.time() - start_time
                logger.info(f"SAM2 loaded successfully in {load_time:.2f}s")

                # Log memory usage
                if self.device == "cuda":
                    memory_stats = self.get_memory_usage()
                    logger.info(
                        f"GPU memory after loading SAM2: "
                        f"{memory_stats['gpu_memory_allocated_mb']:.2f}MB / "
                        f"{memory_stats['gpu_memory_total_mb']:.2f}MB "
                        f"({memory_stats['gpu_memory_usage_percent']:.1f}%)"
                    )

            except Exception as e:
                logger.error(f"Error loading SAM2 model: {e}", exc_info=True)
                raise RuntimeError(f"Failed to load SAM2: {e}") from e

        return self.models[model_name]

    def load_depth_anything_v2(
        self,
        model_name: str = "depth-anything/Depth-Anything-V2-Large-hf",
        use_fp16: bool = True
    ) -> 'DepthAnythingV2':
        """
        Load Depth Anything V2 for monocular depth estimation.

        Depth Anything V2 is a foundation model for monocular depth estimation that
        generates depth maps from single RGB images. It's used in the 3D reconstruction
        pipeline to provide depth information for mesh generation.

        Args:
            model_name: HuggingFace model ID for Depth Anything V2
                       (default: "depth-anything/Depth-Anything-V2-Large-hf")
            use_fp16: Use half precision (FP16) for memory optimization on CUDA
                     (default: True)

        Returns:
            DepthAnythingV2: Loaded depth estimation pipeline

        Raises:
            ImportError: If required dependencies are not installed
            RuntimeError: If model loading fails
        """
        model_key = "depth_anything_v2"

        if model_key not in self.models:
            logger.info("=" * 60)
            logger.info("Loading Depth Anything V2 Model...")
            logger.info("=" * 60)
            start_time = time.time()

            try:
                # Import required modules
                try:
                    from transformers import AutoImageProcessor, AutoModelForDepthEstimation
                except ImportError as e:
                    raise ImportError(
                        "transformers not installed. Install with:\n"
                        "  pip install transformers"
                    ) from e

                logger.info(f"Model ID: {model_name}")
                logger.info("NOTE: Using -hf variant with proper config.json for transformers compatibility")

                # Load model and processor from HuggingFace
                # Use the -hf variant which has proper config.json
                logger.info("Loading processor...")
                image_processor = AutoImageProcessor.from_pretrained(model_name, local_files_only=False)
                logger.info(f"✓ Processor loaded: {type(image_processor).__name__}")
                
                logger.info("Loading model...")
                model = AutoModelForDepthEstimation.from_pretrained(model_name, local_files_only=False)
                logger.info(f"✓ Model loaded: {type(model).__name__}")
                
                # FORCE CPU for 4GB VRAM optimization (preprocessing only, doesn't need GPU)
                model = model.to('cpu')
                logger.info("✓ Depth Anything V2 forced to CPU for memory optimization")
                
                # Store both processor and model
                self.models[model_key] = {
                    'processor': image_processor,
                    'model': model
                }

                load_time = time.time() - start_time
                logger.info("=" * 60)
                logger.info(f"✓ Depth Anything V2 loaded successfully in {load_time:.2f}s")
                logger.info("=" * 60)

                # Log memory usage
                if self.device == "cuda":
                    memory_stats = self.get_memory_usage()
                    logger.info(
                        f"GPU memory after loading Depth Anything V2: "
                        f"{memory_stats['gpu_memory_allocated_mb']:.2f}MB / "
                        f"{memory_stats['gpu_memory_total_mb']:.2f}MB "
                        f"({memory_stats['gpu_memory_usage_percent']:.1f}%)"
                    )

            except Exception as e:
                logger.error(f"Error loading Depth Anything V2 model: {e}", exc_info=True)
                raise RuntimeError(f"Failed to load Depth Anything V2: {e}") from e

        return self.models[model_key]

    def load_sdxl(
        self,
        model_name: str = "stabilityai/sdxl-turbo",
        use_fp16: bool = True,
        enable_cpu_offload: bool = True
    ) -> 'StableDiffusionXLPipeline':
        """
        Load Stable Diffusion XL (SDXL) for body generation.

        SDXL is a large diffusion model used for generating full body images based on
        text prompts. Due to its size (~7GB), it requires aggressive memory optimization
        including CPU offloading to fit in 4GB VRAM.

        Args:
            model_name: HuggingFace model ID for SDXL
                       (default: "stabilityai/sdxl-turbo" - faster variant)
            use_fp16: Use half precision (FP16) for memory optimization on CUDA
                     (default: True)
            enable_cpu_offload: Enable model CPU offloading to reduce VRAM usage
                               (default: True, CRITICAL for 4GB VRAM)

        Returns:
            StableDiffusionXLPipeline: Loaded SDXL pipeline

        Raises:
            ImportError: If diffusers is not installed
            RuntimeError: If model loading fails
        """
        model_key = "sdxl"

        if model_key not in self.models:
            logger.info("Loading SDXL Model...")
            start_time = time.time()

            try:
                # Import required modules
                try:
                    from diffusers import StableDiffusionXLPipeline
                except ImportError as e:
                    raise ImportError(
                        "diffusers not installed. Install with:\n"
                        "  pip install diffusers"
                    ) from e

                logger.info(f"Loading SDXL from: {model_name}")

                # Determine dtype based on device and fp16 setting
                torch_dtype = torch.float16 if (use_fp16 and self.device == "cuda") else torch.float32
                variant = "fp16" if (use_fp16 and self.device == "cuda") else None

                # Load SDXL pipeline from HuggingFace
                # This automatically downloads and caches the model
                sdxl_pipeline = StableDiffusionXLPipeline.from_pretrained(
                    model_name,
                    torch_dtype=torch_dtype,
                    variant=variant,
                    use_safetensors=True
                )

                # Enable CPU offloading for memory optimization (CRITICAL for 4GB VRAM)
                if enable_cpu_offload and self.device == "cuda":
                    sdxl_pipeline.enable_model_cpu_offload()
                    logger.info("SDXL CPU offloading enabled for memory optimization")
                else:
                    # Move to device if not using offloading
                    sdxl_pipeline = sdxl_pipeline.to(self.device)

                # Log FP16 conversion
                if use_fp16 and self.device == "cuda":
                    logger.info("SDXL using FP16 for memory optimization")

                self.models[model_key] = sdxl_pipeline

                load_time = time.time() - start_time
                logger.info(f"SDXL loaded successfully in {load_time:.2f}s")

                # Log memory usage
                if self.device == "cuda":
                    memory_stats = self.get_memory_usage()
                    logger.info(
                        f"GPU memory after loading SDXL: "
                        f"{memory_stats['gpu_memory_allocated_mb']:.2f}MB / "
                        f"{memory_stats['gpu_memory_total_mb']:.2f}MB "
                        f"({memory_stats['gpu_memory_usage_percent']:.1f}%)"
                    )

            except Exception as e:
                logger.error(f"Error loading SDXL model: {e}", exc_info=True)
                raise RuntimeError(f"Failed to load SDXL: {e}") from e

        return self.models[model_key]




    
    async def preload_models(self) -> None:
        """
        Preload critical models on startup.
        
        On first run, this will download Leffa checkpoints from HuggingFace.
        """
        logger.info("Starting model preloading...")
        preload_start = time.time()
        
        models_to_preload = os.getenv("PRELOAD_MODELS", "tryon,segmentation").split(",")
        models_to_preload = [m.strip() for m in models_to_preload if m.strip()]
        
        # First, ensure Leffa checkpoints are downloaded (if tryon is in preload list)
        if "tryon" in models_to_preload:
            logger.info("=" * 60)
            logger.info("Ensuring Leffa checkpoints are downloaded...")
            logger.info("=" * 60)
            if not self.ensure_leffa_checkpoints():
                logger.warning("Leffa checkpoints not available. Try-on will fail until resolved.")
                models_to_preload.remove("tryon")
        
        loaded_models = []
        failed_models = []
        
        for model_name in models_to_preload:
            try:
                logger.info(f"Preloading model: {model_name}")
                model_start = time.time()
                
                if model_name == "tryon":
                    self.load_tryon()
                elif model_name == "segmentation":
                    self.load_segmentation()
                elif model_name == "pose":
                    self.load_pose()
                else:
                    logger.warning(f"Unknown model name for preloading: {model_name}")
                    continue
                
                model_time = time.time() - model_start
                loaded_models.append(model_name)
                logger.info(f"Model '{model_name}' preloaded in {model_time:.2f}s")
                
            except Exception as e:
                logger.error(f"Failed to preload model '{model_name}': {e}", exc_info=True)
                failed_models.append(model_name)
        
        preload_time = time.time() - preload_start
        logger.info(f"Model preloading completed in {preload_time:.2f}s")
        logger.info(f"Loaded models: {loaded_models}")
        
        if failed_models:
            logger.warning(f"Failed to load models: {failed_models}")
        
        # Log memory usage after preloading
        memory_stats = self.get_memory_usage()
        logger.info(f"Memory usage after preloading: {memory_stats}")
    
    async def warmup_model(self, model_name: str) -> None:
        """Warmup a specific model with sample inputs."""
        logger.info(f"Warming up model: {model_name}")
        warmup_start = time.time()
        
        try:
            if model_name == "tryon":
                # Leffa warmup - skip for now as it requires valid images
                logger.info("Leffa warmup skipped (requires valid input images)")
                
            elif model_name == "segmentation":
                # Warmup segmentation model
                processor, model = self.load_segmentation()
                logger.info("Running warmup inference for segmentation model...")
                
                import numpy as np
                from PIL import Image
                
                dummy_image = Image.fromarray(np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8))
                
                with torch.no_grad():
                    inputs = processor(images=dummy_image, return_tensors="pt")
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}
                    _ = model(**inputs)
                
                logger.info("Segmentation model warmup completed")
                
            elif model_name == "pose":
                if not OPENPOSE_AVAILABLE:
                    logger.warning("OpenPose not available for warmup")
                    return
                    
                # Warmup pose model
                model = self.load_pose()
                logger.info("Running warmup inference for pose model...")
                
                import numpy as np
                from PIL import Image
                
                dummy_image = Image.fromarray(np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8))
                _ = model(dummy_image)
                
                logger.info("Pose model warmup completed")
            
            else:
                logger.warning(f"Unknown model name for warmup: {model_name}")
                return
            
            warmup_time = time.time() - warmup_start
            logger.info(f"Model '{model_name}' warmup completed in {warmup_time:.2f}s")
            
        except Exception as e:
            logger.error(f"Failed to warmup model '{model_name}': {e}", exc_info=True)
    
    def unload_model(self, model_name: str) -> bool:
        """
        Unload a specific model to free memory.
        
        Args:
            model_name: Name of the model to unload
            
        Returns:
            True if model was unloaded, False if not found
        """
        if model_name in self.models:
            del self.models[model_name]
            
            # Force garbage collection
            import gc
            gc.collect()
            
            if self.device == "cuda":
                torch.cuda.empty_cache()
            
            logger.info(f"Model '{model_name}' unloaded")
            return True
        
        return False
    
    def unload_all(self) -> None:
        """Unload all models to free memory."""
        model_names = list(self.models.keys())
        for name in model_names:
            self.unload_model(name)
        
        logger.info("All models unloaded")


model_loader = ModelLoader()
