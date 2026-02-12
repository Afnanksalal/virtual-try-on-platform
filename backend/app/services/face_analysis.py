"""
Gemini Vision Face Analysis Service

This service uses Google's Gemini Vision API to analyze facial features
from a user's photo. This information is used to guide the identity-preserving
body generation to create more accurate and personalized results.

Features extracted:
- Skin tone (detailed description)
- Ethnicity estimation
- Facial structure hints
- Unique characteristics for better generation prompts
"""

import os
import io
import json
from typing import Dict, Optional
from PIL import Image
from google import genai
from google.genai import types

from app.core.logging_config import get_logger

logger = get_logger("services.face_analysis")

# Gemini client singleton
_gemini_client = None


def get_gemini_client():
    """Get or create Gemini client."""
    global _gemini_client
    gemini_key = os.getenv("GEMINI_API_KEY")
    if not gemini_key:
        logger.warning("GEMINI_API_KEY not configured")
        return None
    if _gemini_client is None:
        _gemini_client = genai.Client(api_key=gemini_key)
    return _gemini_client


class FaceAnalysisService:
    """
    Service for analyzing facial features using Gemini Vision.
    
    This provides detailed information about the user's appearance
    to guide body generation for more realistic results.
    """
    
    def __init__(self):
        """Initialize the face analysis service."""
        self.client = None
        logger.info("FaceAnalysisService initialized")
    
    def _ensure_client(self):
        """Ensure Gemini client is available."""
        if self.client is None:
            self.client = get_gemini_client()
        if self.client is None:
            raise RuntimeError("GEMINI_API_KEY not configured")
        return self.client
    
    def analyze_face(self, image: Image.Image) -> Dict:
        """
        Analyze facial features from an image.
        
        Args:
            image: PIL Image containing a face
            
        Returns:
            Dictionary with analyzed features:
            - skin_tone: Detailed skin tone description
            - ethnicity: Estimated ethnicity
            - age_range: Estimated age range
            - gender_presentation: How the person presents
            - facial_features: Unique characteristics
            - hair: Hair color and style
            - generation_prompt_hints: Suggestions for SDXL prompting
        """
        client = self._ensure_client()
        
        prompt = """Analyze this face photo carefully for generating a realistic full-body image.

Provide a detailed JSON response with these fields:
{
    "skin_tone": "detailed description (e.g., 'warm ivory with peachy undertones', 'rich dark brown with golden undertones', 'olive with warm undertones')",
    "skin_tone_category": "one of: fair, light, light-medium, medium, medium-tan, tan, olive, dark, deep",
    "ethnicity": "estimated ethnicity/heritage (e.g., 'East Asian', 'South Asian', 'African', 'European', 'Hispanic/Latino', 'Middle Eastern', 'Mixed heritage', 'Southeast Asian')",
    "age_range": "estimated age range (e.g., '20-25', '30-35')",
    "gender_presentation": "how they present (e.g., 'feminine', 'masculine', 'androgynous')",
    "facial_features": {
        "face_shape": "oval, round, square, heart, oblong, diamond",
        "distinctive_features": ["list of notable features that should be preserved"],
        "expression": "neutral, smiling, etc."
    },
    "hair": {
        "color": "hair color description",
        "style": "hairstyle description",
        "length": "short, medium, long"
    },
    "body_proportion_hints": "any visible hints about body proportions from visible shoulders/neck",
    "generation_prompt_additions": "specific descriptive terms to add to SDXL prompt for accuracy"
}

Be respectful and accurate. Focus on visual descriptors that will help AI generate a realistic full-body version.
Return ONLY valid JSON, no additional text."""

        try:
            # Convert image to bytes
            img_byte_arr = io.BytesIO()
            image.save(img_byte_arr, format='PNG')
            img_byte_arr.seek(0)
            
            # Call Gemini Vision
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
            
            # Extract JSON from response
            if "```json" in response_text:
                json_str = response_text.split("```json")[1].split("```")[0].strip()
            elif "```" in response_text:
                json_str = response_text.split("```")[1].split("```")[0].strip()
            else:
                json_str = response_text
            
            result = json.loads(json_str)
            
            logger.info(f"Face analysis completed: skin_tone={result.get('skin_tone_category')}, "
                       f"ethnicity={result.get('ethnicity')}")
            
            return result
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse Gemini response: {e}")
            # Return defaults
            return self._get_default_analysis()
        except Exception as e:
            logger.error(f"Face analysis failed: {e}", exc_info=True)
            return self._get_default_analysis()
    
    def _get_default_analysis(self) -> Dict:
        """Return default analysis when Gemini fails."""
        return {
            "skin_tone": "medium skin tone",
            "skin_tone_category": "medium",
            "ethnicity": "mixed heritage",
            "age_range": "25-35",
            "gender_presentation": "neutral",
            "facial_features": {
                "face_shape": "oval",
                "distinctive_features": [],
                "expression": "neutral"
            },
            "hair": {
                "color": "dark",
                "style": "natural",
                "length": "medium"
            },
            "body_proportion_hints": "",
            "generation_prompt_additions": ""
        }
    
    def build_generation_prompt(
        self,
        analysis: Dict,
        body_params: Dict,
        pose: str = "standing",
        clothing: str = "casual minimal"
    ) -> str:
        """
        Build an optimized prompt for body generation based on analysis.
        
        Args:
            analysis: Result from analyze_face()
            body_params: User's body type preferences
            pose: Desired pose
            clothing: Clothing description
            
        Returns:
            Optimized prompt string for SDXL/InstantID
        """
        # Get values with defaults
        skin_tone = analysis.get("skin_tone", "medium skin tone")
        ethnicity = analysis.get("ethnicity", "mixed heritage")
        gender = analysis.get("gender_presentation", body_params.get("gender", "person"))
        age_range = analysis.get("age_range", "25-35")
        hair = analysis.get("hair", {})
        hair_desc = f"{hair.get('color', 'dark')} {hair.get('length', '')} hair" if hair else ""
        
        # Body descriptors
        body_type = body_params.get("body_type", "average")
        body_descriptors = {
            "athletic": "athletic build, toned muscles, fit physique",
            "slim": "slim build, lean physique, slender body",
            "muscular": "muscular build, strong physique, well-defined muscles",
            "average": "average build, normal proportions, medium physique",
            "curvy": "curvy figure, proportionate curves",
            "plus": "plus size, full figure, voluptuous"
        }
        body_desc = body_descriptors.get(body_type.lower(), "average build")
        
        # Additional hints from Gemini
        prompt_additions = analysis.get("generation_prompt_additions", "")
        
        # Build comprehensive prompt
        prompt_parts = [
            "professional full body photograph",
            f"of a {gender}",
            f"in their {age_range}s",
            f"{ethnicity} heritage",
            f"{skin_tone}",
            hair_desc,
            body_desc,
            f"{pose} pose",
            f"wearing {clothing}",
            "plain white studio background",
            "professional photography lighting",
            "high quality",
            "realistic",
            "8k resolution",
            "highly detailed",
        ]
        
        if prompt_additions:
            prompt_parts.append(prompt_additions)
        
        prompt = ", ".join(filter(None, prompt_parts))
        
        logger.debug(f"Built generation prompt: {prompt[:200]}...")
        
        return prompt
    
    def get_negative_prompt(self) -> str:
        """Get standard negative prompt for body generation."""
        return """cropped, cut off, partial body, close-up, headshot only,
multiple people, cluttered background, low quality, blurry, distorted, deformed,
bad anatomy, extra limbs, missing limbs, floating limbs, disconnected limbs,
mutation, ugly, disgusting, disfigured, watermark, text, logo,
oversaturated, undersaturated, overexposed, underexposed"""


# Singleton instance
_face_analysis_service: Optional[FaceAnalysisService] = None


def get_face_analysis_service() -> FaceAnalysisService:
    """Get or create the singleton service instance."""
    global _face_analysis_service
    if _face_analysis_service is None:
        _face_analysis_service = FaceAnalysisService()
    return _face_analysis_service
