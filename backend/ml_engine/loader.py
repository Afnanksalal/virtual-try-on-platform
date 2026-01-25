import torch
from diffusers import DiffusionPipeline, AutoPipelineForInpainting
from transformers import SegformerImageProcessor, AutoModelForSemanticSegmentation
from controlnet_aux import OpenposeDetector
import os
import threading
import time
from typing import Tuple, Dict, Any, Optional
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
                
            self.models = {}  # Simple model storage
            self.device = "cuda" if torch.cuda.is_available() and os.getenv("USE_GPU", "false").lower() == "true" else "cpu"
            self._initialized = True
            logger.info(f"ModelLoader initialized on device: {self.device}")
    
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

    def load_pose(self) -> OpenposeDetector:
        model_name = "pose"
        
        if model_name not in self.models:
            logger.info("Loading Pose Model (OpenPose)...")
            start_time = time.time()
            
            model = OpenposeDetector.from_pretrained("lllyasviel/ControlNet")
            
            self.models[model_name] = model
            load_time = time.time() - start_time
            logger.info(f"Pose model loaded successfully in {load_time:.2f}s")
        
        return self.models[model_name]

    def load_tryon(self) -> AutoPipelineForInpainting:
        model_name = "tryon"
        
        if model_name not in self.models:
            logger.info("Loading Try-On Model (IDM-VTON)...")
            start_time = time.time()
            
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
