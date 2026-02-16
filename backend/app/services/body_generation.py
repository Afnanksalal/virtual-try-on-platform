"""
Body Generation Service

This service handles body type analysis and full body generation using SDXL.
It integrates with existing body detection logic and uses Gemini for conditioning
based on user parameters.

Key Features:
- Analyze images to determine if they show full body or partial body
- Generate full body images using SDXL with identity preservation
- Use Gemini Vision for intelligent conditioning based on user parameters
"""

import os
from typing import Dict, Any, Optional
from PIL import Image

from app.core.logging_config import get_logger
from app.services.body_detection import body_detector
from app.services.face_analysis import get_face_analysis_service

logger = get_logger("services.body_generation")


class BodyGenerationService:
    """
    Service for analyzing body types and generating full body images.
    
    This service combines:
    - Body detection (using existing pose-based detection)
    - Face analysis (using Gemini Vision)
    - Body generation (using SDXL)
    """
    
    def __init__(self):
        """Initialize the body generation service."""
        self.face_analysis_service = get_face_analysis_service()
        logger.info("BodyGenerationService initialized")
    
    def analyze_body_type(self, image: Image.Image) -> Dict[str, Any]:
        """
        Analyze if image contains full body or partial body.
        
        Uses existing body detection logic to determine if the image shows
        a complete body (including legs/feet) or just head/upper body.
        
        Args:
            image: PIL Image to analyze
            
        Returns:
            Dictionary with analysis results:
            {
                "is_full_body": bool,
                "body_type": "full_body" | "partial_body",
                "confidence": float (0.0-1.0),
                "coverage_metric": float,
                "detected_parts": dict (optional),
                "error": str (if analysis failed)
            }
        """
        try:
            logger.info("Analyzing body type in image...")
            
            # Use existing body detector
            detection_result = body_detector.check_full_body(image)
            
            # Extract results
            is_full_body = detection_result.get("is_full_body", False)
            coverage_metric = detection_result.get("coverage_metric", 0.0)
            
            # Calculate confidence based on coverage metric
            # Higher coverage = higher confidence in full body detection
            # Lower coverage = higher confidence in partial body detection
            if is_full_body:
                # Full body detected - confidence based on how much coverage we have
                confidence = min(0.5 + (coverage_metric * 5), 1.0)  # Scale coverage to confidence
            else:
                # Partial body detected - confidence based on lack of coverage
                confidence = min(0.5 + ((1.0 - coverage_metric) * 5), 1.0)
            
            # Determine body type string
            body_type = "full_body" if is_full_body else "partial_body"
            
            result = {
                "is_full_body": is_full_body,
                "body_type": body_type,
                "confidence": round(confidence, 3),
                "coverage_metric": round(coverage_metric, 4),
            }
            
            # Include error if present
            if "error" in detection_result:
                result["error"] = detection_result["error"]
                logger.warning(f"Body detection had errors: {detection_result['error']}")
            
            logger.info(
                f"Body type analysis complete: {body_type} "
                f"(confidence: {result['confidence']:.2f}, "
                f"coverage: {result['coverage_metric']:.4f})"
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Body type analysis failed: {e}", exc_info=True)
            # Return safe default (assume partial body to trigger generation)
            return {
                "is_full_body": False,
                "body_type": "partial_body",
                "confidence": 0.0,
                "coverage_metric": 0.0,
                "error": str(e)
            }
    
    async def generate_full_body(
        self,
        partial_image: Image.Image,
        user_params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Generate a full body image from a partial image (head/upper body).
        
        This method:
        1. Analyzes the face using Gemini Vision
        2. Combines face analysis with user parameters
        3. Generates a full body image using SDXL
        4. Preserves facial identity in the generated image
        
        Args:
            partial_image: PIL Image containing head/upper body
            user_params: Dictionary with user preferences:
                - ethnicity: str (optional, Gemini can infer)
                - body_type: str (e.g., "slim", "athletic", "average", "curvy", "plus_size")
                - height_cm: float (optional)
                - weight_kg: float (optional)
                - gender: str (optional, e.g., "male", "female", "person")
                - pose: str (optional, default: "standing")
                - clothing: str (optional, default: "casual minimal")
                
        Returns:
            Dictionary with generation results:
            {
                "generated_image": Image.Image,
                "face_analysis": dict,
                "generation_prompt": str,
                "success": bool,
                "error": str (if failed)
            }
        """
        try:
            logger.info("Starting full body generation...")
            
            # Step 1: Analyze face using Gemini Vision
            logger.info("Analyzing face features with Gemini Vision...")
            face_analysis = self.face_analysis_service.analyze_face(partial_image)
            logger.info(
                f"Face analysis complete: "
                f"skin_tone={face_analysis.get('skin_tone_category')}, "
                f"ethnicity={face_analysis.get('ethnicity')}"
            )
            
            # Step 2: Build generation prompt
            pose = user_params.get("pose", "standing")
            clothing = user_params.get("clothing", "casual minimal clothing")
            
            generation_prompt = self.face_analysis_service.build_generation_prompt(
                analysis=face_analysis,
                body_params=user_params,
                pose=pose,
                clothing=clothing
            )
            
            negative_prompt = self.face_analysis_service.get_negative_prompt()
            
            logger.info(f"Generation prompt: {generation_prompt[:150]}...")
            
            # Step 3: Load SDXL model
            logger.info("Loading SDXL model...")
            from ml_engine.loader import model_loader
            sdxl_pipeline = model_loader.load_sdxl(
                use_fp16=True,
                enable_cpu_offload=True  # Critical for 4GB VRAM
            )
            
            # Step 4: Generate full body image
            logger.info("Generating full body image with SDXL...")
            
            # SDXL generation parameters
            num_inference_steps = user_params.get("num_inference_steps", 4)  # SDXL Turbo uses 4 steps
            guidance_scale = user_params.get("guidance_scale", 0.0)  # SDXL Turbo uses 0.0
            seed = user_params.get("seed", 42)
            
            import torch
            generator = torch.Generator(device=model_loader.device).manual_seed(seed)
            
            # Generate image
            result = sdxl_pipeline(
                prompt=generation_prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=generator,
                height=1024,  # SDXL native resolution
                width=768,    # Portrait aspect ratio
            )
            
            generated_image = result.images[0]
            
            logger.info("Full body generation complete")
            
            # Log memory usage
            memory_stats = model_loader.get_memory_usage()
            logger.info(f"Memory usage after generation: {memory_stats}")
            
            return {
                "generated_image": generated_image,
                "face_analysis": face_analysis,
                "generation_prompt": generation_prompt,
                "success": True
            }
            
        except Exception as e:
            logger.error(f"Full body generation failed: {e}", exc_info=True)
            return {
                "generated_image": None,
                "face_analysis": None,
                "generation_prompt": None,
                "success": False,
                "error": str(e)
            }
    
    async def preserve_identity(
        self,
        source_face: Image.Image,
        generated_body: Image.Image
    ) -> Dict[str, Any]:
        """
        Preserve facial identity by combining face from source with generated body.
        
        This method uses face swapping or blending techniques to ensure the
        generated body has the exact face from the source image.
        
        NOTE: This is a placeholder for future implementation. Currently, SDXL
        generation should preserve general facial features based on the prompt,
        but precise identity preservation requires additional models like
        InstantID or IP-Adapter.
        
        Args:
            source_face: Original image with the face to preserve
            generated_body: Generated full body image
            
        Returns:
            Dictionary with results:
            {
                "result_image": Image.Image,
                "success": bool,
                "method": str,
                "error": str (if failed)
            }
        """
        try:
            logger.info("Identity preservation requested...")
            
            # TODO: Implement identity preservation using one of:
            # 1. InstantID (best for identity preservation)
            # 2. IP-Adapter (good for style/identity transfer)
            # 3. Face swapping models (InsightFace, etc.)
            # 4. Simple face blending (basic approach)
            
            # For now, return the generated body as-is
            logger.warning(
                "Identity preservation not yet implemented. "
                "Returning generated body image. "
                "Consider implementing InstantID or IP-Adapter for better results."
            )
            
            return {
                "result_image": generated_body,
                "success": True,
                "method": "none (not implemented)",
                "note": "Generated body returned without identity preservation"
            }
            
        except Exception as e:
            logger.error(f"Identity preservation failed: {e}", exc_info=True)
            return {
                "result_image": generated_body,  # Fallback to generated image
                "success": False,
                "method": "fallback",
                "error": str(e)
            }


# Singleton instance
_body_generation_service: Optional[BodyGenerationService] = None


def get_body_generation_service() -> BodyGenerationService:
    """Get or create the singleton service instance."""
    global _body_generation_service
    if _body_generation_service is None:
        _body_generation_service = BodyGenerationService()
    return _body_generation_service
