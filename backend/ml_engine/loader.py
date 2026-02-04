import torch
from transformers import SegformerImageProcessor, AutoModelForSemanticSegmentation
import os
import threading
import time
from typing import Tuple, Dict, Any, Optional
from app.core.logging_config import get_logger

logger = get_logger("ml.loader")

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


class ModelLoader:
    """
    Singleton model loader for managing ML model lifecycle.
    
    Handles loading, caching, and memory management for:
    - Leffa virtual try-on pipeline
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
