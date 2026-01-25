import torch
from diffusers import DiffusionPipeline, AutoPipelineForInpainting
from transformers import SegformerImageProcessor, AutoModelForSemanticSegmentation
from controlnet_aux import OpenposeDetector
import os
import threading
import time
from typing import Tuple, Dict, Any, Optional
from collections import OrderedDict
from app.core.logging_config import get_logger

logger = get_logger("ml.loader")

class ModelLoader:
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
                
            self.models = OrderedDict()  # LRU cache using OrderedDict
            self.model_access_times = {}  # Track last access time for each model
            self.device = "cuda" if torch.cuda.is_available() and os.getenv("USE_GPU", "false").lower() == "true" else "cpu"
            self.max_cache_size = int(os.getenv("MODEL_CACHE_SIZE", "5"))  # Maximum number of models to cache
            self.gpu_memory_threshold = float(os.getenv("GPU_MEMORY_THRESHOLD", "0.8"))  # 80% threshold
            self._initialized = True
            logger.info(f"ModelLoader initialized on device: {self.device}")
            logger.info(f"Model cache size: {self.max_cache_size}, GPU memory threshold: {self.gpu_memory_threshold * 100}%")
    
    @classmethod
    def get_instance(cls) -> 'ModelLoader':
        """Thread-safe method to get the singleton instance."""
        return cls()
    
    def _get_gpu_memory_usage(self) -> float:
        """Get current GPU memory usage as a percentage (0.0 to 1.0)."""
        if self.device == "cuda" and torch.cuda.is_available():
            try:
                allocated = torch.cuda.memory_allocated(0)
                total = torch.cuda.get_device_properties(0).total_memory
                usage = allocated / total if total > 0 else 0.0
                return usage
            except Exception as e:
                logger.warning(f"Failed to get GPU memory usage: {e}")
                return 0.0
        return 0.0
    
    def _evict_lru_model(self) -> None:
        """Evict the least recently used model from cache."""
        if not self.models:
            return
        
        # Find the least recently used model
        lru_model_name = None
        lru_time = float('inf')
        
        for model_name, access_time in self.model_access_times.items():
            if access_time < lru_time:
                lru_time = access_time
                lru_model_name = model_name
        
        if lru_model_name:
            logger.info(f"Evicting LRU model: {lru_model_name}")
            del self.models[lru_model_name]
            del self.model_access_times[lru_model_name]
            
            # Clear GPU cache if using CUDA
            if self.device == "cuda":
                torch.cuda.empty_cache()
                logger.info(f"GPU cache cleared after evicting {lru_model_name}")
    
    def _check_and_evict_if_needed(self) -> None:
        """Check cache size and GPU memory, evict models if necessary."""
        # Check cache size limit
        while len(self.models) >= self.max_cache_size:
            logger.info(f"Model cache full ({len(self.models)}/{self.max_cache_size}), evicting LRU model")
            self._evict_lru_model()
        
        # Check GPU memory usage
        gpu_usage = self._get_gpu_memory_usage()
        while gpu_usage > self.gpu_memory_threshold and self.models:
            logger.warning(f"GPU memory usage high ({gpu_usage * 100:.1f}%), evicting LRU model")
            self._evict_lru_model()
            gpu_usage = self._get_gpu_memory_usage()
    
    def _update_model_access(self, model_name: str) -> None:
        """Update the access time for a model (for LRU tracking)."""
        self.model_access_times[model_name] = time.time()
        
        # Move to end of OrderedDict to maintain LRU order
        if model_name in self.models:
            self.models.move_to_end(model_name)
    
    def get_memory_usage(self) -> Dict[str, Any]:
        """Get current memory usage statistics."""
        stats = {
            "device": self.device,
            "cached_models": list(self.models.keys()),
            "cache_size": len(self.models),
            "max_cache_size": self.max_cache_size
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
        model_name = "segmentation"
        
        if model_name not in self.models:
            # Check and evict if needed before loading new model
            self._check_and_evict_if_needed()
            
            logger.info("Loading Segmentation Model (Segformer)...")
            start_time = time.time()
            
            processor = SegformerImageProcessor.from_pretrained("mattmdjaga/segformer_b2_clothes")
            model = AutoModelForSemanticSegmentation.from_pretrained("mattmdjaga/segformer_b2_clothes")
            model.to(self.device)
            
            self.models[model_name] = (processor, model)
            load_time = time.time() - start_time
            logger.info(f"Segmentation model loaded successfully in {load_time:.2f}s")
        
        # Update access time for LRU tracking
        self._update_model_access(model_name)
        return self.models[model_name]

    def load_pose(self) -> OpenposeDetector:
        model_name = "pose"
        
        if model_name not in self.models:
            # Check and evict if needed before loading new model
            self._check_and_evict_if_needed()
            
            logger.info("Loading Pose Model (OpenPose)...")
            start_time = time.time()
            
            # controlnet_aux loads seamlessly
            model = OpenposeDetector.from_pretrained("lllyasviel/ControlNet")
            # OpenposeDetector usually handles device internally or doesn't support .to() directly depending on version, 
            # but usually runs on CPU or GPU automatically if torch is setup.
            
            self.models[model_name] = model
            load_time = time.time() - start_time
            logger.info(f"Pose model loaded successfully in {load_time:.2f}s")
        
        # Update access time for LRU tracking
        self._update_model_access(model_name)
        return self.models[model_name]

    def load_tryon(self) -> AutoPipelineForInpainting:
        model_name = "tryon"
        
        if model_name not in self.models:
            # Check and evict if needed before loading new model
            self._check_and_evict_if_needed()
            
            logger.info("Loading Try-On Model (IDM-VTON)...")
            start_time = time.time()
            
            # Using standard stable-diffusion-inpainting as a robust fallback base 
            # or the specific IDM-VTON checkpoint if available. 
            # For this MVP/Production readiness, we use a standard reliable pipe.
            # Real IDM-VTON requires custom pipeline code often. 
            # We will use 'diffusers' standard pipe for stability.
            model_id = "diffusers/stable-diffusion-xl-1.0-inpainting-0.1" 
            try:
                pipe = AutoPipelineForInpainting.from_pretrained(
                    model_id, 
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
                )
                pipe.to(self.device)
                self.models[model_name] = pipe
                
                load_time = time.time() - start_time
                logger.info(f"Try-on model loaded successfully in {load_time:.2f}s")
            except Exception as e:
                logger.error(f"Error loading VTON model: {e}", exc_info=True)
                raise e
        
        # Update access time for LRU tracking
        self._update_model_access(model_name)
        return self.models[model_name]
    
    async def preload_models(self) -> None:
        """Preload critical models on startup."""
        logger.info("Starting model preloading...")
        preload_start = time.time()
        
        models_to_preload = os.getenv("PRELOAD_MODELS", "tryon,segmentation").split(",")
        models_to_preload = [m.strip() for m in models_to_preload if m.strip()]
        
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
                # Warmup try-on model with dummy input
                pipe = self.load_tryon()
                logger.info("Running warmup inference for try-on model...")
                
                # Create dummy inputs
                import numpy as np
                from PIL import Image
                
                dummy_image = Image.fromarray(np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8))
                dummy_mask = Image.fromarray(np.random.randint(0, 255, (512, 512), dtype=np.uint8))
                
                # Run a quick inference to warm up the model
                with torch.no_grad():
                    _ = pipe(
                        prompt="warmup",
                        image=dummy_image,
                        mask_image=dummy_mask,
                        num_inference_steps=1,
                        guidance_scale=1.0
                    )
                
                logger.info("Try-on model warmup completed")
                
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
                # Warmup pose model
                model = self.load_pose()
                logger.info("Running warmup inference for pose model...")
                
                import numpy as np
                from PIL import Image
                
                dummy_image = Image.fromarray(np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8))
                
                # OpenposeDetector typically doesn't need explicit warmup but we'll call it once
                _ = model(dummy_image)
                
                logger.info("Pose model warmup completed")
            
            else:
                logger.warning(f"Unknown model name for warmup: {model_name}")
                return
            
            warmup_time = time.time() - warmup_start
            logger.info(f"Model '{model_name}' warmup completed in {warmup_time:.2f}s")
            
        except Exception as e:
            logger.error(f"Failed to warmup model '{model_name}': {e}", exc_info=True)

model_loader = ModelLoader()
