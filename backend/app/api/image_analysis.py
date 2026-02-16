from fastapi import APIRouter, UploadFile, File, HTTPException
from PIL import Image
import io
from google import genai
from google.genai import types
import os
import json
from ..core.logging_config import get_logger
from ..core.file_validator import FileValidator
from ..services.body_generation import get_body_generation_service
from ..models.schemas import BodyAnalysisResponse

router = APIRouter()
logger = get_logger("api.image_analysis")

# Initialize file validator
file_validator = FileValidator(max_size_mb=10)

# Gemini client singleton
_gemini_client = None

def get_gemini_client():
    """Get Gemini client, initializing if needed."""
    global _gemini_client
    gemini_key = os.getenv("GEMINI_API_KEY")
    if not gemini_key:
        return None
    if _gemini_client is None:
        _gemini_client = genai.Client(api_key=gemini_key)
    return _gemini_client

@router.post("/analyze-body", response_model=BodyAnalysisResponse)
async def analyze_body(image: UploadFile = File(...)):
    """
    Analyze if uploaded image contains full body or partial body using pose detection.
    
    This endpoint uses the BodyGenerationService to detect body coverage based on
    pose estimation and keypoint detection. It determines whether the image shows
    a complete body (including legs/feet) or just head/upper body.
    
    Args:
        image: Uploaded image file (JPEG, PNG, or WebP)
        
    Returns:
        BodyAnalysisResponse with:
        - body_type: "full_body" or "partial_body"
        - is_full_body: Boolean flag
        - confidence: Confidence score (0.0-1.0)
        - coverage_metric: Body coverage metric
        - error: Optional error message if analysis had issues
        
    Raises:
        HTTPException 400: Invalid file type or size
        HTTPException 500: Analysis failed
    """
    try:
        logger.info("Body analysis request received")
        
        # Validate uploaded file
        await file_validator.validate_file(image)
        
        # Read and load image
        image_bytes = await image.read()
        img = Image.open(io.BytesIO(image_bytes))
        
        logger.info(f"Image loaded: {img.size[0]}x{img.size[1]} pixels")
        
        # Get body generation service
        body_service = get_body_generation_service()
        
        # Analyze body type
        analysis_result = body_service.analyze_body_type(img)
        
        logger.info(
            f"Body analysis complete: {analysis_result['body_type']} "
            f"(confidence: {analysis_result['confidence']:.2f})"
        )
        
        # Return structured response
        return BodyAnalysisResponse(**analysis_result)
        
    except ValueError as e:
        logger.error(f"Validation error: {e}")
        raise HTTPException(400, f"Invalid input: {str(e)}")
    except Exception as e:
        logger.error(f"Body analysis failed: {e}", exc_info=True)
        raise HTTPException(500, f"Body analysis failed: {str(e)}")


@router.post("/analyze-image")
async def analyze_image(image: UploadFile = File(...)):
    """
    Use Gemini Vision API to detect if uploaded image is head-only or full-body.
    Returns: {"type": "head_only" | "full_body", "confidence": 0-1}
    
    NOTE: This endpoint uses Gemini Vision for analysis. For pose-based detection,
    use /analyze-body instead.
    """
    try:
        client = get_gemini_client()
        if not client:
            logger.error("GEMINI_API_KEY not configured")
            raise HTTPException(500, "GEMINI_API_KEY not configured")
        
        # Validate uploaded file
        await file_validator.validate_file(image)
        
        # Read image
        image_bytes = await image.read()
        img = Image.open(io.BytesIO(image_bytes))
        
        prompt = """Analyze this image carefully. Determine if this is:
1. A headshot/portrait (showing only head and shoulders)
2. A full-body photo (showing at least torso and legs)

Respond ONLY with valid JSON in this exact format:
{"type": "head_only", "confidence": 0.95}
OR
{"type": "full_body", "confidence": 0.85}

Be strict: if you can't see at least the torso and upper legs, it's "head_only".
"""
        
        # Convert image to bytes for Gemini
        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format='PNG')
        img_byte_arr.seek(0)
        
        # Use google-genai SDK with inline image data
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=[
                types.Part.from_text(text=prompt),
                types.Part.from_bytes(
                    data=img_byte_arr.getvalue(),
                    mime_type='image/png'
                )
            ]
        )
        
        # Parse response
        response_text = response.text.strip()
        
        # Extract JSON from response (sometimes Gemini wraps it in markdown)
        if "```json" in response_text:
            json_str = response_text.split("```json")[1].split("```")[0].strip()
        elif "```" in response_text:
            json_str = response_text.split("```")[1].split("```")[0].strip()
        else:
            json_str = response_text
        
        result = json.loads(json_str)
        
        # Validate response
        if result.get("type") not in ["head_only", "full_body"]:
            raise ValueError("Invalid response type from Gemini")
        
        logger.info(f"Image analysis result: {result}")
        return result
        
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse Gemini response: {e}, response: {response_text}")
        raise HTTPException(500, "Failed to parse AI response")
    except Exception as e:
        logger.error(f"Image analysis failed: {e}", exc_info=True)
        raise HTTPException(500, f"Image analysis failed: {str(e)}")
