"""
Production-Grade 3D Reconstruction Pipeline

Features:
- SAM2 automatic segmentation for person extraction
- Enhanced TripoSR with adaptive thresholds
- Aggressive memory management for 4GB VRAM
- CPU offloading strategy

Optimized for RTX 3050 4GB VRAM.
"""

import torch
import numpy as np
from PIL import Image
from typing import Tuple, Optional, Dict, Any, List
import io
import time
import gc
from pathlib import Path
import cv2
from app.core.logging_config import get_logger

logger = get_logger("ml.reconstruction_3d")


class ThreeDReconstructionPipeline:
    """
    Production-grade 3D reconstruction pipeline
    
    Strategy for 4GB VRAM:
    - SAM2 on CPU (offloaded immediately after use)
    - TripoSR on GPU with FP16 for fast inference
    - Nuclear memory cleanup between stages
    - Aggressive garbage collection
    """
    
    def __init__(self, device: str = "cuda"):
        """
        Initialize the 3D reconstruction pipeline.
        
        Args:
            device: Device to use for inference ("cuda" or "cpu")
        """
        self.device = device if torch.cuda.is_available() else "cpu"
        self.cpu_device = torch.device("cpu")  # For CPU offloading
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32
        
        # Model references
        self.models = {}  # Active models on GPU
        self.models_cpu = {}  # Offloaded models in RAM
        
        # Offloading strategy for 4GB VRAM:
        # - SAM2: Always on CPU (large model, used once)
        # - TripoSR: On GPU with FP16 (needs GPU for fast inference)
        
        logger.info(f"ThreeDReconstructionPipeline initialized on device: {self.device}")
        logger.info("Offloading Strategy: SAM2 on CPU → TripoSR on GPU (4GB VRAM)")
    
    def _log_vram_usage(self, context=""):
        """Log current VRAM usage"""
        if not torch.cuda.is_available():
            return
        
        vram_allocated = torch.cuda.memory_allocated() / 1024**3
        vram_reserved = torch.cuda.memory_reserved() / 1024**3
        prefix = f"{context}: " if context else ""
        logger.info(f"{prefix}VRAM: {vram_allocated:.2f}GB allocated, {vram_reserved:.2f}GB reserved")
    
    def offload_to_cpu(self, model_name: str) -> bool:
        """Offload model from GPU to CPU RAM"""
        if model_name in self.models:
            try:
                model = self.models[model_name]
                if hasattr(model, 'to'):
                    model.to('cpu')
                    self.models_cpu[model_name] = model
                    del self.models[model_name]
                    torch.cuda.empty_cache()
                    gc.collect()
                    logger.info(f"✓ Offloaded {model_name} to CPU RAM")
                    return True
            except Exception as e:
                logger.warning(f"Failed to offload {model_name}: {e}")
        return False
    
    def reset_cuda_memory(self):
        """Aggressively reset CUDA memory - NUCLEAR OPTION"""
        logger.info("🔥 NUCLEAR MEMORY RESET - Clearing everything from GPU...")
        
        # Move all models to CPU
        for model_name in list(self.models.keys()):
            self.offload_to_cpu(model_name)
        
        # Multiple rounds of garbage collection
        for _ in range(3):
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
        gc.collect()
        
        self._log_vram_usage("After nuclear reset")
    
    def _cleanup_segmentation_models(self):
        """AGGRESSIVELY clean up segmentation models to free memory"""
        logger.info("🔥 Aggressive cleanup of segmentation models...")
        
        if 'sam2' in self.models:
            self.offload_to_cpu('sam2')
        
        # Multiple rounds of cleanup
        for _ in range(3):
            torch.cuda.empty_cache()
            gc.collect()
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.ipc_collect()
        
        self._log_vram_usage("After segmentation cleanup")
    
    # ==================== MODEL LOADING ====================
    
    def load_sam2(self) -> bool:
        """Load SAM2 on CPU (FORCED for 4GB VRAM optimization)"""
        if 'sam2' in self.models:
            return True
            
        logger.info("Loading SAM2 on CPU (forced for memory optimization)...")
        try:
            from ml_engine.loader import model_loader
            
            # Load via ModelLoader - FORCE CPU device
            predictor = model_loader.load_sam2()
            
            # FORCE SAM2 model to CPU if it's on GPU
            if hasattr(predictor, 'model'):
                predictor.model = predictor.model.to('cpu')
                logger.info("✓ SAM2 model forced to CPU")
            
            # Store reference
            self.models['sam2'] = predictor
            logger.info("✓ SAM2 loaded on CPU")
            return True
            
        except Exception as e:
            logger.error(f"SAM2 loading failed: {e}")
            return False
    
    def load_triposr(self) -> bool:
        """Load TripoSR on GPU"""
        if 'triposr' in self.models:
            logger.info("✓ TripoSR already loaded on GPU")
            return True
            
        logger.info("Loading TripoSR on GPU...")
        try:
            # Free VRAM before loading
            self.reset_cuda_memory()
            
            from ml_engine.loader import model_loader
            
            # Load via ModelLoader (with FP16 and chunk size for 4GB VRAM)
            model = model_loader.load_triposr(chunk_size=8192, use_fp16=True)
            
            # Store reference
            self.models['triposr'] = model
            logger.info(f"✓ TripoSR loaded on {self.device} with FP16")
            return True
            
        except Exception as e:
            logger.warning(f"TripoSR unavailable: {e}")
            return False
    
    # ==================== SEGMENTATION ====================
    
    def segment_with_sam2(self, image: Image.Image) -> Image.Image:
        """
        Segment person using SAM2 automatic segmentation (no prompts)
        
        Args:
            image: Input RGB image
        
        Returns:
            Segmented RGBA image with transparent background
            
        Strategy:
        - Use SAM2's automatic mask generation to find all objects
        - Select the largest mask (typically the person in portrait photos)
        - No depth-based prompting to avoid interfering with 3D reconstruction
        """
        logger.info("Segmenting with SAM2 automatic mask generation...")
        
        if not self.load_sam2():
            raise RuntimeError("SAM2 not available")
        
        try:
            from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
            
            img_array = np.array(image.convert('RGB'))
            
            # Create automatic mask generator
            mask_generator = SAM2AutomaticMaskGenerator(
                model=self.models['sam2'].model,
                points_per_side=32,
                pred_iou_thresh=0.7,
                stability_score_thresh=0.85,
                crop_n_layers=1,
                crop_n_points_downscale_factor=2,
            )
            
            # Generate masks
            logger.info("Generating automatic masks...")
            masks = mask_generator.generate(img_array)
            
            if not masks:
                logger.warning("No masks generated, returning original image")
                # Return original image as RGBA
                result = np.zeros((img_array.shape[0], img_array.shape[1], 4), dtype=np.uint8)
                result[:, :, :3] = img_array
                result[:, :, 3] = 255  # Full opacity
                return Image.fromarray(result, 'RGBA')
            
            # Sort masks by area (largest first)
            masks = sorted(masks, key=lambda x: x['area'], reverse=True)
            
            # Select the largest mask (typically the person)
            best_mask = masks[0]['segmentation']
            
            logger.info(f"Selected largest mask: area={masks[0]['area']}, "
                       f"stability_score={masks[0]['stability_score']:.3f}, "
                       f"predicted_iou={masks[0]['predicted_iou']:.3f}")
            
            # Create RGBA output with transparency
            result = np.zeros((img_array.shape[0], img_array.shape[1], 4), dtype=np.uint8)
            result[:, :, :3] = img_array
            result[:, :, 3] = (best_mask * 255).astype(np.uint8)  # Alpha channel
            
            coverage = 100 * np.sum(best_mask) / best_mask.size
            logger.info(f"✓ SAM2 segmentation complete (coverage: {coverage:.1f}%)")
            
            # IMMEDIATELY offload SAM2 after use
            logger.info("Offloading SAM2 from memory...")
            self.offload_to_cpu('sam2')
            torch.cuda.empty_cache()
            gc.collect()
            
            return Image.fromarray(result, 'RGBA')
            
        except Exception as e:
            logger.error(f"SAM2 segmentation failed: {e}", exc_info=True)
            raise RuntimeError(f"SAM2 segmentation failed: {e}") from e
    
    # ==================== 3D GENERATION ====================
    
    def generate_3d_mesh(
        self,
        image: Image.Image,
        mc_resolution: int = 256,
        chunk_size: int = 8192,
        use_adaptive_threshold: bool = True,
        post_process: bool = True,
        adaptive_resolution: bool = True
    ) -> 'trimesh.Trimesh':
        """
        Generate 3D mesh using Enhanced TripoSR with adaptive resolution fallback
        
        Args:
            image: Input RGB image (should be segmented with transparent background)
            mc_resolution: Starting marching cubes resolution (256 for 4GB VRAM, 512 for 8GB+)
            chunk_size: Chunk size for surface extraction (8192 for 4GB VRAM)
            use_adaptive_threshold: Use adaptive threshold based on density statistics
            post_process: Apply mesh post-processing (smoothing, hole filling)
            adaptive_resolution: Try higher resolutions and fallback on OOM (default: True)
            
        Returns:
            trimesh.Trimesh object containing the 3D mesh
            
        Note:
            TripoSR texture quality is tied to mesh resolution (no separate texture parameter).
            Max resolution is 1024. Adaptive mode tries: 1024 → 512 → 256 → 128 on OOM.
            Production (A100): Will use 1024 for best quality.
            Development (RTX 3050 4GB): Will fallback to 256 or 128.
        """
        # Adaptive resolution strategy for better texture quality
        # Try higher resolutions first, fallback on OOM
        # Production (A100): 1024 → 512 → 256 → 128
        # Development (RTX 3050 4GB): Will fallback to 256 or 128
        if adaptive_resolution:
            resolution_cascade = [1024, 512, 256, 128]
            # Start from requested resolution or higher
            if mc_resolution not in resolution_cascade:
                resolution_cascade = [mc_resolution] + resolution_cascade
            resolution_cascade = sorted(set(resolution_cascade), reverse=True)
        else:
            resolution_cascade = [mc_resolution]
        
        logger.info("=" * 60)
        logger.info("3D MESH GENERATION")
        logger.info("=" * 60)
        logger.info(f"Adaptive resolution: {adaptive_resolution}")
        logger.info(f"Resolution cascade: {resolution_cascade}")
        logger.info(f"Chunk size: {chunk_size}")
        
        # STEP 1: NUCLEAR MEMORY CLEANUP
        logger.info("🔥 STEP 1: Nuclear memory cleanup...")
        self.reset_cuda_memory()
        
        # STEP 2: Load TripoSR (Enhanced version)
        if not self.load_triposr():
            raise RuntimeError("TripoSR not available")
        
        model = self.models['triposr']
        
        # Check if model has enhanced features
        has_enhanced = hasattr(model, 'extract_mesh_enhanced')
        if has_enhanced:
            logger.info("✓ Using Enhanced TripoSR with adaptive threshold and post-processing")
        else:
            logger.info("Using Standard TripoSR")
        
        self._log_vram_usage("Before TripoSR inference")
        
        # TripoSR preprocessing
        import sys
        triposr_path = Path(__file__).parent.parent.parent / "3d" / "TripoSR"
        sys.path.insert(0, str(triposr_path.absolute()))
        
        try:
            from tsr.utils import resize_foreground
            
            # Resize foreground
            if image.mode == "RGBA":
                image_resized = resize_foreground(image, 0.85)
            else:
                image_resized = image.resize((512, 512), Image.Resampling.LANCZOS)
            
            # Handle RGBA - composite on gray background
            if image_resized.mode == "RGBA":
                img_array = np.array(image_resized).astype(np.float32) / 255.0
                img_array = img_array[:, :, :3] * img_array[:, :, 3:4] + (1 - img_array[:, :, 3:4]) * 0.5
                image_resized = Image.fromarray((img_array * 255.0).astype(np.uint8))
            
        except ImportError:
            # Fallback if TripoSR utils not available
            if image.mode == "RGBA":
                # Simple composite on gray
                rgb_image = Image.new("RGB", image.size, (127, 127, 127))
                rgb_image.paste(image, mask=image.split()[3])
                image_resized = rgb_image.resize((512, 512), Image.Resampling.LANCZOS)
            else:
                image_resized = image.resize((512, 512), Image.Resampling.LANCZOS)
        
        mesh = None
        last_error = None
        
        # Try resolutions in cascade (highest to lowest)
        for attempt, current_resolution in enumerate(resolution_cascade, 1):
            try:
                logger.info("=" * 60)
                logger.info(f"ATTEMPT {attempt}/{len(resolution_cascade)}: Resolution {current_resolution}")
                logger.info("=" * 60)
                
                # Clear cache before each attempt
                for _ in range(3):
                    torch.cuda.empty_cache()
                    gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                
                self._log_vram_usage(f"Before encoding at {current_resolution}")
                
                logger.info(f"Encoding image at resolution={current_resolution} on GPU...")
                
                # Run inference on GPU
                with torch.no_grad():
                    scene_codes = model([image_resized], device=self.device)
                
                # Cleanup
                for _ in range(2):
                    torch.cuda.empty_cache()
                    gc.collect()
                
                self._log_vram_usage(f"After encoding at {current_resolution}")
                
                logger.info(f"Extracting mesh (resolution={current_resolution}) on GPU...")
                
                # Extract mesh using enhanced features if available
                if has_enhanced and (use_adaptive_threshold or post_process):
                    logger.info("Using enhanced extraction with adaptive threshold and post-processing...")
                    meshes = model.extract_mesh_enhanced(
                        scene_codes,
                        has_vertex_color=True,
                        resolution=current_resolution,
                        use_adaptive_threshold=use_adaptive_threshold,
                        use_multires=False,  # Disable for 4GB VRAM
                        post_process=post_process
                    )
                else:
                    # Standard extraction
                    meshes = model.extract_mesh(
                        scene_codes,
                        resolution=current_resolution
                    )
                
                mesh = meshes[0]
                
                # Apply coordinate transformation and rotation
                mesh = self._transform_mesh_coordinates(mesh)
                
                # Cleanup
                del scene_codes
                for _ in range(3):
                    torch.cuda.empty_cache()
                    gc.collect()
                
                logger.info("=" * 60)
                logger.info(f"✓ SUCCESS at resolution {current_resolution}")
                logger.info(f"  Vertices: {len(mesh.vertices):,}")
                logger.info(f"  Faces: {len(mesh.faces):,}")
                logger.info("=" * 60)
                
                self._log_vram_usage("After mesh extraction")
                
                # Success! Break out of cascade
                break
                
            except RuntimeError as e:
                error_msg = str(e).lower()
                is_oom = "out of memory" in error_msg or "cuda" in error_msg
                
                if is_oom and attempt < len(resolution_cascade):
                    logger.warning("=" * 60)
                    logger.warning(f"⚠️ OOM at resolution {current_resolution}")
                    logger.warning(f"Falling back to resolution {resolution_cascade[attempt]}")
                    logger.warning("=" * 60)
                    
                    # Aggressive cleanup before retry
                    if 'scene_codes' in locals():
                        del scene_codes
                    for _ in range(5):
                        torch.cuda.empty_cache()
                        gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                        torch.cuda.ipc_collect()
                    
                    last_error = e
                    continue
                else:
                    # Not OOM or last attempt failed
                    logger.error(f"Error at resolution={current_resolution}: {e}")
                    raise RuntimeError(f"3D reconstruction failed at all resolutions: {e}") from e
            
            except Exception as e:
                logger.error(f"Unexpected error at resolution={current_resolution}: {e}")
                raise RuntimeError(f"3D reconstruction failed: {e}") from e
        
        if mesh is None:
            raise RuntimeError(f"Failed to generate mesh at all resolutions. Last error: {last_error}")
        
        # STEP 3: FINAL CLEANUP
        logger.info("🔥 STEP 3: Final cleanup...")
        for _ in range(3):
            torch.cuda.empty_cache()
            gc.collect()
        
        self._log_vram_usage("After TripoSR generation")
        
        return mesh
    
    def _transform_mesh_coordinates(self, mesh, rotation_degrees: int = 60):
        """Apply coordinate transformation and rotation to mesh"""
        import math
        import trimesh
        
        vertices = mesh.vertices.copy()
        
        # Apply coordinate fix
        vertices = np.stack([
            vertices[:, 0],   # X stays the same
            vertices[:, 2],   # Z becomes Y
            -vertices[:, 1]   # Y becomes -Z
        ], axis=1)
        
        # Rotate around Y-axis
        angle = math.radians(rotation_degrees)
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)
        
        x = vertices[:, 0] * cos_a + vertices[:, 2] * sin_a
        y = vertices[:, 1]
        z = -vertices[:, 0] * sin_a + vertices[:, 2] * cos_a
        
        mesh.vertices = np.stack([x, y, z], axis=1)
        return mesh
    
    def export_mesh(
        self,
        mesh: 'trimesh.Trimesh',
        output_format: str = "glb"
    ) -> Tuple[bytes, str]:
        """
        Export mesh to specified format.
        
        Args:
            mesh: trimesh.Trimesh object to export
            output_format: Output format ("glb", "obj", or "ply")
            
        Returns:
            Tuple of (file_bytes, format)
        """
        logger.info(f"Exporting mesh to {output_format.upper()} format...")
        
        try:
            # Validate format
            supported_formats = ["glb", "obj", "ply"]
            if output_format.lower() not in supported_formats:
                raise ValueError(
                    f"Unsupported format: {output_format}. "
                    f"Supported formats: {', '.join(supported_formats)}"
                )
            
            # Export to bytes
            file_obj = io.BytesIO()
            mesh.export(file_obj, file_type=output_format.lower())
            file_bytes = file_obj.getvalue()
            
            logger.info(f"Mesh exported successfully ({len(file_bytes)} bytes)")
            
            return file_bytes, output_format.lower()
        
        except Exception as e:
            logger.error(f"Mesh export failed: {e}")
            raise RuntimeError(f"Export failed: {e}") from e
    
    # ==================== MAIN PIPELINE ====================
    
    def process_3d_reconstruction(
        self,
        image: Image.Image,
        output_format: str = "glb",
        use_segmentation: bool = True,
        mc_resolution: int = 256,
        chunk_size: int = 8192,
        return_intermediate: bool = False
    ) -> Tuple[bytes, str, Optional[Dict[str, Image.Image]]]:
        """
        Complete 3D reconstruction pipeline.
        
        Args:
            image: Input RGB image
            output_format: Output format ("glb", "obj", or "ply")
            use_segmentation: Whether to use SAM2 for background removal
            mc_resolution: Marching cubes resolution (256 for 4GB VRAM)
            chunk_size: Chunk size for TripoSR (8192 for 4GB VRAM)
            return_intermediate: Whether to return intermediate processing images
            
        Returns:
            Tuple of (file_bytes, format, intermediate_images)
            intermediate_images is None (no intermediate steps shown)
        """
        logger.info("=" * 60)
        logger.info("Starting 3D Reconstruction Pipeline")
        logger.info("=" * 60)
        pipeline_start = time.time()
        
        try:
            # Stage 1: Segmentation (optional, no depth estimation)
            if use_segmentation:
                logger.info("Stage 1/2: SAM2 Automatic Segmentation")
                segmented_image = self.segment_with_sam2(image)
            else:
                logger.info("Stage 1/2: Segmentation (skipped)")
                segmented_image = image
            
            # Cleanup segmentation models
            self._cleanup_segmentation_models()
            
            # Stage 2: 3D Mesh Generation
            logger.info("Stage 2/2: 3D Mesh Generation")
            mesh = self.generate_3d_mesh(
                segmented_image,
                mc_resolution=mc_resolution,
                chunk_size=chunk_size
            )
            
            # Export mesh to requested format
            file_bytes, format_str = self.export_mesh(mesh, output_format)
            
            pipeline_time = time.time() - pipeline_start
            logger.info("=" * 60)
            logger.info(f"3D Reconstruction Pipeline completed in {pipeline_time:.2f}s")
            logger.info("=" * 60)
            
            return file_bytes, format_str, None
        
        except Exception as e:
            logger.error(f"3D reconstruction pipeline failed: {e}")
            raise RuntimeError(f"Pipeline failed: {e}") from e
        
        finally:
            # Clean up GPU memory
            if self.device == "cuda" and torch.cuda.is_available():
                torch.cuda.empty_cache()
                logger.debug("GPU memory cache cleared")
