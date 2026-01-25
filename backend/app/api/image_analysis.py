from fastapi import APIRouter, UploadFile, File, HTTPException
from PIL import Image
import io
from google import genai
from google.genai import types
import os
import json
from ..core.logging_config import get_logger
from ..core.file_validator import FileValidator

router = APIRouter()
logger = get_logger("api.image_analysis")

# Initialize file validator
file_validator = FileValidator(max_size_mb=10)

# Configure Gemini
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
gemini_client = None
if GEMINI_API_KEY:
    gemini_client = genai.Client(api_key=GEMINI_API_KEY)

@router.post("/analyze-image")
async def analyze_image(image: UploadFile = File(...)):
    """
    Use Gemini Vision API to detect if uploaded image is head-only or full-body.
    Returns: {"type": "head_only" | "full_body", "confidence": 0-1}
    """
    try:
        if not gemini_client:
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
        
        # Save image temporarily for upload
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            img.save(tmp.name, format='PNG')
            tmp_path = tmp.name
        
        try:
            # Use new Gemini SDK
            response = gemini_client.models.generate_content(
                model='gemini-2.0-flash-exp',
                contents=[
                    types.Part.from_text(text=prompt),
                    types.Part.from_uri(
                        file_uri=tmp_path,
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
        finally:
            # Clean up temp file
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
        
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse Gemini response: {e}, response: {response_text}")
        raise HTTPException(500, "Failed to parse AI response")
    except Exception as e:
        logger.error(f"Image analysis failed: {e}", exc_info=True)
        raise HTTPException(500, f"Image analysis failed: {str(e)}")
