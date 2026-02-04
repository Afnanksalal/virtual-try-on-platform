import os
from google import genai
from google.genai import types
from typing import List, Dict, Optional
from PIL import Image
import json
import re
import asyncio
import hashlib
import httpx
from datetime import datetime, timedelta
from .image_collage import collage_service
from ..core.logging_config import get_logger

logger = get_logger("services.recommendation")


class CircuitBreaker:
    """
    Circuit breaker pattern implementation for external API calls.
    
    States:
    - CLOSED: Normal operation, requests pass through
    - OPEN: Too many failures, requests fail fast
    - HALF_OPEN: Testing if service recovered
    """
    
    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: int = 60,
        name: str = "circuit_breaker"
    ):
        """
        Initialize circuit breaker.
        
        Args:
            failure_threshold: Number of failures before opening circuit
            recovery_timeout: Seconds to wait before attempting recovery
            name: Name for logging
        """
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.name = name
        
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
    
    def call(self, func, *args, **kwargs):
        """Execute function with circuit breaker protection."""
        if self.state == "OPEN":
            # Check if recovery timeout has passed
            if self.last_failure_time and \
               (datetime.now() - self.last_failure_time).seconds >= self.recovery_timeout:
                logger.info(f"{self.name}: Attempting recovery (HALF_OPEN)")
                self.state = "HALF_OPEN"
            else:
                logger.warning(f"{self.name}: Circuit OPEN, failing fast")
                raise Exception(f"Circuit breaker {self.name} is OPEN")
        
        try:
            result = func(*args, **kwargs)
            self.on_success()
            return result
        except Exception as e:
            self.on_failure()
            raise e
    
    def on_success(self):
        """Handle successful call."""
        if self.state == "HALF_OPEN":
            logger.info(f"{self.name}: Recovery successful, closing circuit")
            self.state = "CLOSED"
        
        self.failure_count = 0
        self.last_failure_time = None
    
    def on_failure(self):
        """Handle failed call."""
        self.failure_count += 1
        self.last_failure_time = datetime.now()
        
        if self.failure_count >= self.failure_threshold:
            logger.error(f"{self.name}: Failure threshold reached, opening circuit")
            self.state = "OPEN"
    
    def is_open(self) -> bool:
        """Check if circuit is open."""
        return self.state == "OPEN"
    
    def reset(self):
        """Reset circuit breaker."""
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "CLOSED"
        logger.info(f"{self.name}: Circuit breaker reset")


class RecommendationEngine:
    """
    Advanced recommendation pipeline:
    1. Extract skin tone from user photo using Gemini Vision
    2. Create image collage from user photos, wardrobe, generated bodies
    3. Use Gemini Vision to extract keywords with color theory based on skin tone
    4. Search eBay via RapidAPI using extracted keywords
    5. Return product listings with direct eBay links
    """
    
    # Skin Tone Color Theory Mapping (Fitzpatrick Scale + Undertones)
    SKIN_TONE_PALETTES = {
        "fair_cool": {
            "best_colors": ["navy blue", "emerald green", "burgundy", "plum", "soft pink", "lavender", "charcoal", "silver gray"],
            "avoid_colors": ["orange", "mustard yellow", "camel", "warm browns"],
            "metals": ["silver", "platinum", "white gold"],
            "neutrals": ["pure white", "black", "cool gray", "slate"]
        },
        "fair_warm": {
            "best_colors": ["coral", "peach", "warm red", "olive green", "camel", "golden brown", "cream", "warm teal"],
            "avoid_colors": ["cool pink", "fuchsia", "pure white", "electric blue"],
            "metals": ["gold", "rose gold", "brass"],
            "neutrals": ["ivory", "cream", "warm brown", "tan"]
        },
        "light_cool": {
            "best_colors": ["royal blue", "deep purple", "magenta", "forest green", "raspberry", "cool gray", "soft pink"],
            "avoid_colors": ["orange", "mustard", "warm yellows"],
            "metals": ["silver", "platinum"],
            "neutrals": ["charcoal", "navy", "cool beige"]
        },
        "light_warm": {
            "best_colors": ["salmon", "apricot", "turquoise", "olive", "rust", "terracotta", "golden yellow"],
            "avoid_colors": ["hot pink", "electric blue", "stark white"],
            "metals": ["gold", "copper", "bronze"],
            "neutrals": ["beige", "camel", "tan", "chocolate brown"]
        },
        "medium_cool": {
            "best_colors": ["teal", "deep blue", "burgundy", "purple", "emerald", "raspberry", "cool pink", "pewter"],
            "avoid_colors": ["orange", "coral", "warm yellows"],
            "metals": ["silver", "white gold", "gunmetal"],
            "neutrals": ["navy", "charcoal", "cool gray", "eggplant"]
        },
        "medium_warm": {
            "best_colors": ["burnt orange", "mustard", "olive green", "warm red", "coral", "bronze", "warm turquoise"],
            "avoid_colors": ["cool pinks", "lavender", "stark white"],
            "metals": ["gold", "brass", "copper"],
            "neutrals": ["khaki", "caramel", "chocolate", "warm gray"]
        },
        "olive": {
            "best_colors": ["deep red", "burgundy", "forest green", "royal purple", "cobalt blue", "burnt sienna", "warm pink"],
            "avoid_colors": ["pastels", "pale yellow", "pale blue", "orange"],
            "metals": ["gold", "bronze", "antique brass"],
            "neutrals": ["olive", "chocolate", "navy", "deep brown"]
        },
        "tan_warm": {
            "best_colors": ["coral", "turquoise", "orange", "golden yellow", "warm red", "teal", "fuchsia"],
            "avoid_colors": ["pastels", "muted colors", "cool grays"],
            "metals": ["gold", "rose gold", "copper"],
            "neutrals": ["tan", "caramel", "warm brown", "chocolate"]
        },
        "tan_cool": {
            "best_colors": ["hot pink", "royal blue", "deep purple", "emerald", "bright teal", "berry"],
            "avoid_colors": ["orange", "warm yellows", "rust"],
            "metals": ["silver", "platinum", "white gold"],
            "neutrals": ["navy", "black", "charcoal", "cool brown"]
        },
        "deep_warm": {
            "best_colors": ["bright white", "orange", "coral", "tomato red", "fuchsia", "gold", "turquoise", "bright yellow"],
            "avoid_colors": ["muted tones", "pastels", "cool grays"],
            "metals": ["gold", "brass", "copper", "bronze"],
            "neutrals": ["camel", "deep brown", "rich tan", "orange-brown"]
        },
        "deep_cool": {
            "best_colors": ["pure white", "hot pink", "cobalt blue", "bright purple", "emerald green", "silver", "icy blue"],
            "avoid_colors": ["muted oranges", "rust", "warm browns", "olive"],
            "metals": ["silver", "platinum", "white gold"],
            "neutrals": ["black", "charcoal", "navy", "deep purple"]
        },
        "deep_neutral": {
            "best_colors": ["bright white", "red", "cobalt", "purple", "emerald", "orange", "fuchsia", "teal"],
            "avoid_colors": ["muted browns", "dusty colors"],
            "metals": ["gold", "silver", "copper"],
            "neutrals": ["black", "white", "deep brown", "navy"]
        }
    }
    
    def __init__(self):
        # API Keys
        self.gemini_key = os.getenv("GEMINI_API_KEY")
        self.rapidapi_key = os.getenv("RAPIDAPI_KEY")
        self.rapidapi_host = os.getenv("RAPIDAPI_HOST", "ebay-search-result.p.rapidapi.com")
        
        # Retry configuration
        self.max_retries = 3
        self.retry_delay = 1.0  # seconds
        self.retry_backoff = 2.0  # exponential backoff multiplier
        
        # HTTP client with connection pooling
        self.http_client = httpx.AsyncClient(
            timeout=httpx.Timeout(10.0, connect=5.0),
            limits=httpx.Limits(
                max_connections=100,
                max_keepalive_connections=20,
                keepalive_expiry=30.0
            ),
            http2=True,  # Enable HTTP/2 for better performance
        )
        
        # Circuit breakers
        self.gemini_circuit_breaker = CircuitBreaker(
            failure_threshold=5,
            recovery_timeout=60,
            name="Gemini API"
        )
        self.ebay_circuit_breaker = CircuitBreaker(
            failure_threshold=5,
            recovery_timeout=60,
            name="eBay API"
        )
        
        # Configure Gemini
        if self.gemini_key:
            self.gemini_client = genai.Client(api_key=self.gemini_key)
        else:
            logger.warning("GEMINI_API_KEY not set")
            self.gemini_client = None
        
        if not self.rapidapi_key:
            logger.warning("RAPIDAPI_KEY not set")
    
    async def close(self):
        """Close HTTP client connections."""
        await self.http_client.aclose()
        logger.info("HTTP client connections closed")
    
    async def extract_skin_tone(self, user_photo: Image.Image) -> Dict:
        """
        Extract skin tone from user photo using Gemini Vision.
        
        Returns:
            Dict with skin_tone_category, undertone, fitzpatrick_scale, and color recommendations
        """
        if not self.gemini_client:
            logger.warning("Gemini not configured, using default skin tone")
            return {
                "skin_tone_category": "medium_neutral",
                "undertone": "neutral",
                "fitzpatrick_scale": 3,
                "confidence": 0.0
            }
        
        prompt = """Analyze this person's photo and determine their skin tone for fashion color matching.

Provide ONLY a JSON object with the following fields:
1. "skin_tone_category": One of: fair_cool, fair_warm, light_cool, light_warm, medium_cool, medium_warm, olive, tan_warm, tan_cool, deep_warm, deep_cool, deep_neutral
2. "undertone": One of: cool (pink/blue undertones), warm (yellow/golden undertones), neutral (mix of both)
3. "fitzpatrick_scale": Number 1-6 (1=very fair, 6=very deep)
4. "observed_features": Brief description of what you observed (hair color, eye color if visible)
5. "confidence": Your confidence level 0.0-1.0

Example output:
{"skin_tone_category": "medium_warm", "undertone": "warm", "fitzpatrick_scale": 4, "observed_features": "golden undertones, dark brown hair", "confidence": 0.85}

CRITICAL: Output ONLY the JSON object, no markdown or other text."""

        try:
            import io
            img_byte_arr = io.BytesIO()
            user_photo.save(img_byte_arr, format='PNG')
            img_byte_arr.seek(0)
            
            response = self.gemini_client.models.generate_content(
                model='gemini-2.5-flash',
                contents=[
                    types.Part.from_text(text=prompt),
                    types.Part.from_bytes(
                        data=img_byte_arr.getvalue(),
                        mime_type='image/png'
                    )
                ]
            )
            
            text = response.text.strip()
            logger.debug(f"Skin tone analysis response: {text[:200]}...")
            
            # Parse JSON
            if "```json" in text:
                json_str = text.split("```json")[1].split("```")[0].strip()
            elif "```" in text:
                json_str = text.split("```")[1].split("```")[0].strip()
            else:
                json_str = text
            
            result = json.loads(json_str)
            logger.info(f"Extracted skin tone: {result.get('skin_tone_category')} (confidence: {result.get('confidence')})")
            return result
            
        except Exception as e:
            logger.error(f"Skin tone extraction failed: {e}")
            return {
                "skin_tone_category": "medium_neutral",
                "undertone": "neutral",
                "fitzpatrick_scale": 3,
                "confidence": 0.0
            }
    
    def get_color_recommendations(self, skin_tone_category: str) -> Dict:
        """Get color recommendations based on skin tone category."""
        # Map to closest category if not exact match
        if skin_tone_category in self.SKIN_TONE_PALETTES:
            return self.SKIN_TONE_PALETTES[skin_tone_category]
        
        # Try to find a close match
        if "fair" in skin_tone_category.lower():
            return self.SKIN_TONE_PALETTES.get("fair_cool", self.SKIN_TONE_PALETTES["medium_cool"])
        elif "light" in skin_tone_category.lower():
            return self.SKIN_TONE_PALETTES.get("light_warm", self.SKIN_TONE_PALETTES["medium_warm"])
        elif "olive" in skin_tone_category.lower():
            return self.SKIN_TONE_PALETTES["olive"]
        elif "deep" in skin_tone_category.lower() or "dark" in skin_tone_category.lower():
            return self.SKIN_TONE_PALETTES.get("deep_neutral", self.SKIN_TONE_PALETTES["deep_warm"])
        elif "tan" in skin_tone_category.lower():
            return self.SKIN_TONE_PALETTES.get("tan_warm", self.SKIN_TONE_PALETTES["medium_warm"])
        
        # Default to medium warm
        return self.SKIN_TONE_PALETTES["medium_warm"]
    
    async def close(self):
        """Close HTTP client connections."""
        await self.http_client.aclose()
        logger.info("HTTP client connections closed")
    
    async def get_outfit_recommendations(
        self,
        user_photo: Image.Image,
        wardrobe_images: List[Image.Image] = None,
        generated_images: List[Image.Image] = None,
        user_profile: Optional[Dict] = None,
        user_id: Optional[str] = None
    ) -> List[Dict]:
        """
        Main recommendation pipeline with skin tone extraction and color theory.
        
        Args:
            user_photo: User's profile photo
            wardrobe_images: List of wardrobe items
            generated_images: Generated body/outfit images
            user_profile: User profile data (height, weight, body_type, ethnicity, gender, style_preference, skin_tone)
            user_id: Optional user ID for per-user caching
            
        Returns:
            List of eBay products with links and color-coordinated recommendations
        """
        try:
            # Step 1: Extract skin tone from user photo (if not provided in profile)
            skin_tone_info = None
            color_recs = None
            
            if user_profile and user_profile.get('skin_tone'):
                # Use provided skin tone
                skin_tone_category = user_profile['skin_tone'].lower().replace(' ', '_')
                color_recs = self.get_color_recommendations(skin_tone_category)
                skin_tone_info = {"skin_tone_category": skin_tone_category, "confidence": 1.0}
                logger.info(f"Using provided skin tone: {skin_tone_category}")
            else:
                # Extract from image
                logger.info("Extracting skin tone from user photo...")
                skin_tone_info = await self.extract_skin_tone(user_photo)
                color_recs = self.get_color_recommendations(skin_tone_info.get('skin_tone_category', 'medium_warm'))
                logger.info(f"Extracted skin tone: {skin_tone_info}")
            
            # Step 2: Create collage with labels
            all_images = [user_photo]
            labels = ["User Photo"]
            
            if wardrobe_images:
                all_images.extend(wardrobe_images)
                labels.extend([f"Wardrobe Item {i+1}" for i in range(len(wardrobe_images))])
            
            if generated_images:
                all_images.extend(generated_images)
                labels.extend([f"Generated Look {i+1}" for i in range(len(generated_images))])
            
            collage = collage_service.create_collage(
                all_images, 
                labels=labels,
                add_borders=True
            )
            logger.info(f"Created collage from {len(all_images)} images with labels")
            
            # Step 3: Extract keywords with Gemini Vision (with color theory and user profile)
            keywords = await self._extract_keywords_with_circuit_breaker(
                collage, 
                user_profile, 
                skin_tone_info, 
                color_recs
            )
            logger.info(f"Extracted keywords: {keywords}")
            
            # Step 4: Search eBay for each keyword (with circuit breaker)
            # Increase results: use top 8 keywords, get 4 items per keyword
            products = []
            for keyword in keywords[:8]:  # Top 8 keywords
                ebay_products = await self._search_ebay_with_circuit_breaker(keyword)
                products.extend(ebay_products[:4])  # Top 4 per keyword
            
            # Deduplicate products by ID
            seen_ids = set()
            unique_products = []
            for product in products:
                if product['id'] not in seen_ids:
                    seen_ids.add(product['id'])
                    unique_products.append(product)
            
            # Return top 20 products (increased from 10)
            final_products = unique_products[:20]
            
            logger.info(f"Returning {len(final_products)} unique product recommendations")
            return final_products
            
        except Exception as e:
            logger.error(f"Recommendation pipeline failed: {e}", exc_info=True)
            
            # Return graceful error message with fallback products
            logger.warning("Returning fallback recommendations due to error")
            return self._get_fallback_recommendations()
    
    async def _extract_keywords_with_circuit_breaker(
        self, 
        collage: Image.Image, 
        user_profile: Optional[Dict] = None,
        skin_tone_info: Optional[Dict] = None,
        color_recs: Optional[Dict] = None
    ) -> List[str]:
        """Extract keywords with circuit breaker protection."""
        if self.gemini_circuit_breaker.is_open():
            logger.warning("Gemini circuit breaker is OPEN, using fallback keywords")
            return self._get_fallback_keywords(color_recs)
        
        try:
            return self.gemini_circuit_breaker.call(
                self._extract_keywords_with_color_theory_sync,
                collage,
                user_profile,
                skin_tone_info,
                color_recs
            )
        except Exception as e:
            logger.error(f"Gemini API call failed: {e}")
            return self._get_fallback_keywords(color_recs)
    
    def _extract_keywords_with_color_theory_sync(
        self, 
        collage: Image.Image, 
        user_profile: Optional[Dict] = None,
        skin_tone_info: Optional[Dict] = None,
        color_recs: Optional[Dict] = None
    ) -> List[str]:
        """Synchronous version for circuit breaker."""
        # This is a wrapper to make the async method work with circuit breaker
        import asyncio
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(self._extract_keywords_with_color_theory(
                collage, user_profile, skin_tone_info, color_recs
            ))
        finally:
            loop.close()
    
    async def _search_ebay_with_circuit_breaker(self, query: str) -> List[Dict]:
        """Search eBay with circuit breaker protection."""
        if self.ebay_circuit_breaker.is_open():
            logger.warning("eBay circuit breaker is OPEN, using fallback")
            return self._get_fallback_product(query)
        
        try:
            return await self._search_ebay(query)
        except Exception as e:
            logger.error(f"eBay API call failed: {e}")
            self.ebay_circuit_breaker.on_failure()
            return self._get_fallback_product(query)
    
    async def _extract_keywords_with_color_theory(
        self, 
        collage: Image.Image, 
        user_profile: Optional[Dict] = None,
        skin_tone_info: Optional[Dict] = None,
        color_recs: Optional[Dict] = None
    ) -> List[str]:
        """
        Use Gemini Vision to extract clothing keywords with proper color theory based on skin tone.
        Implements retry logic with exponential backoff.
        """
        if not self.gemini_client:
            logger.error("Gemini API not configured, using fallback keywords")
            return self._get_fallback_keywords(color_recs)
        
        # Build user profile context
        profile_context = ""
        if user_profile:
            profile_parts = []
            
            if 'height_cm' in user_profile and 'weight_kg' in user_profile:
                height = user_profile['height_cm']
                weight = user_profile['weight_kg']
                # Calculate approximate size suggestion
                bmi = weight / ((height/100) ** 2)
                if bmi < 18.5:
                    size_note = "likely XS-S sizes"
                elif bmi < 25:
                    size_note = "likely S-M sizes"
                elif bmi < 30:
                    size_note = "likely M-L sizes"
                else:
                    size_note = "likely L-XL or plus sizes"
                profile_parts.append(f"Height: {height}cm, Weight: {weight}kg ({size_note})")
            
            if 'body_type' in user_profile:
                body_type = user_profile['body_type'].replace('_', ' ').title()
                profile_parts.append(f"Body Type: {body_type}")
            
            if 'ethnicity' in user_profile:
                ethnicity = user_profile['ethnicity'].replace('_', ' ').title()
                profile_parts.append(f"Ethnicity: {ethnicity}")
            
            if 'gender' in user_profile:
                gender = user_profile['gender'].replace('_', ' ').title()
                profile_parts.append(f"Gender: {gender}")
            
            if 'style_preference' in user_profile:
                style = user_profile['style_preference']
                profile_parts.append(f"Style Preference: {style}")
            
            if profile_parts:
                profile_context = f"\n\n**User Profile:**\n" + "\n".join(f"- {part}" for part in profile_parts)
        
        # Build skin tone color theory context
        color_theory_context = ""
        if skin_tone_info and color_recs:
            skin_cat = skin_tone_info.get('skin_tone_category', 'unknown')
            undertone = skin_tone_info.get('undertone', 'neutral')
            
            best_colors = color_recs.get('best_colors', [])
            avoid_colors = color_recs.get('avoid_colors', [])
            metals = color_recs.get('metals', [])
            neutrals = color_recs.get('neutrals', [])
            
            color_theory_context = f"""

**Skin Tone Analysis Result:**
- Category: {skin_cat.replace('_', ' ').title()}
- Undertone: {undertone.title()}

**SCIENTIFIC COLOR THEORY RECOMMENDATIONS for this skin tone:**
- BEST Colors (will make them look vibrant and healthy): {', '.join(best_colors)}
- AVOID Colors (will wash them out or clash): {', '.join(avoid_colors)}
- Best Metal Tones for accessories: {', '.join(metals)}
- Best Neutral Colors: {', '.join(neutrals)}

IMPORTANT: Generate keywords using ONLY the recommended "BEST Colors" and "Best Neutral Colors" above.
Avoid suggesting items in the "AVOID Colors" at all costs."""

        prompt = f"""You are an expert fashion stylist and color theory specialist.
Analyze this fashion collage image and the provided user data carefully.{profile_context}{color_theory_context}

Generate eBay search keywords for clothing items that would:
1. PERFECTLY complement the user's skin tone using the color recommendations above
2. Flatter their body type and proportions
3. Match their style preference (if provided)
4. Fill gaps in their current wardrobe (if wardrobe items are shown)
5. Be appropriate for their gender presentation

**KEYWORD GENERATION RULES:**
1. Each keyword must include a specific COLOR from the "BEST Colors" list
2. Include garment type (shirt, pants, dress, blazer, etc.)
3. Add size/fit descriptors when relevant (e.g., "plus size", "slim fit", "petite")
4. Make keywords specific enough for eBay search (3-5 words each)
5. Include a mix of:
   - Tops (shirts, blouses, sweaters)
   - Bottoms (pants, skirts, shorts)
   - Outerwear (jackets, blazers, coats)
   - Dresses/jumpsuits (if applicable)
   - Accessories matching recommended metals

**Example good keywords:**
- "navy blue slim fit blazer"
- "emerald green silk blouse"
- "burgundy wool sweater women"
- "charcoal dress pants men"
- "gold tone statement necklace"

Output ONLY a JSON array of 8-10 specific, color-coordinated search keywords.
Do NOT include any explanation or markdown formatting.

Example output:
["navy blue slim fit blazer", "emerald green silk blouse", "burgundy wool sweater", "charcoal dress pants", "silver hoop earrings"]"""

        # Retry loop with exponential backoff
        last_error = None
        delay = self.retry_delay
        
        for attempt in range(self.max_retries):
            try:
                logger.info(f"Attempting Gemini Vision API call (attempt {attempt + 1}/{self.max_retries})")
                
                # Convert image to bytes
                import io
                img_byte_arr = io.BytesIO()
                collage.save(img_byte_arr, format='PNG')
                img_byte_arr.seek(0)
                
                # Generate content with inline image data
                response = self.gemini_client.models.generate_content(
                    model='gemini-2.5-flash',
                    contents=[
                        types.Part.from_text(text=prompt),
                        types.Part.from_bytes(
                            data=img_byte_arr.getvalue(),
                            mime_type='image/png'
                        )
                    ]
                )
                
                text = response.text.strip()
                logger.debug(f"Gemini response: {text[:200]}...")
                
                # Extract JSON from response
                keywords = self._parse_keywords_from_response(text)
                
                if keywords and len(keywords) > 0:
                    logger.info(f"Successfully extracted {len(keywords)} color-coordinated keywords")
                    return keywords
                else:
                    logger.warning("Gemini returned empty keywords, retrying...")
                    raise ValueError("Empty keywords returned")
                    
            except Exception as e:
                last_error = e
                logger.warning(f"Gemini API call failed (attempt {attempt + 1}/{self.max_retries}): {e}")
                
                if attempt < self.max_retries - 1:
                    logger.info(f"Retrying in {delay} seconds...")
                    await asyncio.sleep(delay)
                    delay *= self.retry_backoff  # Exponential backoff
        
        # All retries failed, use fallback
        logger.error(f"All Gemini API retries failed. Last error: {last_error}")
        return self._get_fallback_keywords(color_recs)
    
    def _parse_keywords_from_response(self, text: str) -> List[str]:
        """Parse keywords from Gemini response, handling various formats."""
        try:
            # Try to extract JSON from markdown code blocks
            if "```json" in text:
                json_str = text.split("```json")[1].split("```")[0].strip()
            elif "```" in text:
                json_str = text.split("```")[1].split("```")[0].strip()
            else:
                json_str = text
            
            # Remove any markdown or extra characters before/after JSON
            json_str = re.sub(r'^[^[]*', '', json_str)  # Remove before [
            json_str = re.sub(r'[^\]]*$', '', json_str)  # Remove after ]
            
            # Parse JSON
            keywords = json.loads(json_str)
            
            # Validate it's a list of strings
            if isinstance(keywords, list) and all(isinstance(k, str) for k in keywords):
                # Filter out empty strings and limit to 8 keywords
                keywords = [k.strip() for k in keywords if k.strip()][:8]
                return keywords
            else:
                logger.warning(f"Invalid keywords format: {type(keywords)}")
                return []
                
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON from Gemini response: {e}")
            logger.debug(f"Response text: {text}")
            return []
        except Exception as e:
            logger.error(f"Unexpected error parsing keywords: {e}")
            return []
    
    def _get_fallback_keywords(self, color_recs: Optional[Dict] = None) -> List[str]:
        """Return fallback keywords when Gemini API fails, using color recommendations if available."""
        if color_recs and color_recs.get('best_colors'):
            best_colors = color_recs['best_colors'][:4]
            neutrals = color_recs.get('neutrals', ['black', 'white'])[:2]
            
            fallback = [
                f"{best_colors[0]} dress shirt" if len(best_colors) > 0 else "navy dress shirt",
                f"{best_colors[1]} blazer" if len(best_colors) > 1 else "black blazer",
                f"{neutrals[0]} dress pants" if neutrals else "black dress pants",
                f"{best_colors[2]} sweater" if len(best_colors) > 2 else "gray sweater",
                "casual denim jeans",
                f"{best_colors[3]} blouse" if len(best_colors) > 3 else "white blouse",
                f"{neutrals[1]} casual pants" if len(neutrals) > 1 else "khaki pants",
                "leather dress shoes"
            ]
        else:
            fallback = [
                "black dress pants",
                "white button shirt", 
                "navy blue blazer",
                "casual denim jeans",
                "leather dress shoes",
                "casual sneakers",
                "cotton t-shirt",
                "summer dress"
            ]
        logger.info(f"Using fallback keywords: {fallback}")
        return fallback
    
    async def _search_ebay(self, query: str) -> List[Dict]:
        """
        Search eBay using RapidAPI with retry logic and ranking.
        Uses httpx AsyncClient with connection pooling.
        
        Args:
            query: Search query string
            
        Returns:
            List of product dictionaries with ranking by relevance
        """
        if not self.rapidapi_key:
            logger.warning("RapidAPI key not configured, using fallback")
            return self._get_fallback_product(query)
        
        url = f"https://{self.rapidapi_host}/search/{query}"
        
        headers = {
            "X-RapidAPI-Key": self.rapidapi_key,
            "X-RapidAPI-Host": self.rapidapi_host
        }
        
        # Retry loop with exponential backoff
        last_error = None
        delay = self.retry_delay
        
        for attempt in range(self.max_retries):
            try:
                logger.info(f"Searching eBay for '{query}' (attempt {attempt + 1}/{self.max_retries})")
                
                # Use httpx AsyncClient with connection pooling
                response = await self.http_client.get(url, headers=headers)
                response.raise_for_status()
                data = response.json()
                
                # Parse and rank products
                products = self._parse_ebay_results(data, query)
                
                if products:
                    logger.info(f"Found {len(products)} products for '{query}'")
                    self.ebay_circuit_breaker.on_success()
                    return products
                else:
                    logger.warning(f"No products found for '{query}'")
                    return self._get_fallback_product(query)
                    
            except httpx.TimeoutException:
                last_error = "Request timeout"
                logger.warning(f"eBay API timeout (attempt {attempt + 1}/{self.max_retries})")
                
            except httpx.HTTPStatusError as e:
                last_error = str(e)
                logger.warning(f"eBay API HTTP error: {e.response.status_code} (attempt {attempt + 1}/{self.max_retries})")
                
                # Don't retry on 4xx errors (client errors)
                if 400 <= e.response.status_code < 500:
                    logger.error(f"Client error, not retrying: {e}")
                    break
                    
            except Exception as e:
                last_error = str(e)
                logger.warning(f"eBay API error: {e} (attempt {attempt + 1}/{self.max_retries})")
            
            # Wait before retry
            if attempt < self.max_retries - 1:
                logger.info(f"Retrying in {delay} seconds...")
                await asyncio.sleep(delay)
                delay *= self.retry_backoff
        
        # All retries failed
        logger.error(f"All eBay API retries failed for '{query}'. Last error: {last_error}")
        self.ebay_circuit_breaker.on_failure()
        return self._get_fallback_product(query)
    
    def _parse_ebay_results(self, data: Dict, query: str) -> List[Dict]:
        """
        Parse eBay API results and rank by relevance.
        Enhanced to extract more data and better images.
        
        Args:
            data: Raw eBay API response
            query: Original search query for relevance scoring
            
        Returns:
            List of parsed and ranked products
        """
        products = []
        results = data.get('results', [])
        
        # Also check alternative response structures
        if not results:
            results = data.get('items', [])
        if not results:
            results = data.get('searchResults', [])
        if not results:
            results = data.get('findItemsAdvancedResponse', [{}])[0].get('searchResult', [{}])[0].get('item', [])
        
        logger.info(f"Parsing {len(results)} eBay results for query '{query}'")
        
        for item in results[:15]:  # Increased to top 15 results for better selection
            try:
                # Extract image URL - try multiple possible paths (comprehensive)
                image_url = None
                
                # Try different image field locations
                image_paths = [
                    lambda i: i.get('image', {}).get('imageUrl'),
                    lambda i: i.get('image', {}).get('url'),
                    lambda i: i.get('image') if isinstance(i.get('image'), str) else None,
                    lambda i: i.get('imageUrl'),
                    lambda i: i.get('galleryURL'),
                    lambda i: i.get('pictureURLLarge'),
                    lambda i: i.get('pictureURLSuperSize'),
                    lambda i: i.get('thumbnailUrl'),
                    lambda i: (i.get('thumbnailImages', [{}])[0].get('imageUrl') if i.get('thumbnailImages') else None),
                    lambda i: (i.get('additionalImages', [{}])[0].get('imageUrl') if i.get('additionalImages') else None),
                    lambda i: i.get('primaryImage', {}).get('imageUrl'),
                    lambda i: (i.get('galleryInfo', {}).get('galleryURL', [None])[0] if i.get('galleryInfo') else None),
                ]
                
                for path_fn in image_paths:
                    try:
                        url = path_fn(item)
                        if url and isinstance(url, str) and url.startswith('http'):
                            image_url = url
                            break
                    except:
                        continue
                
                # Fallback to placeholder if no image found
                if not image_url:
                    image_url = f'https://via.placeholder.com/400x500?text={query.replace(" ", "+")}'
                
                # Upgrade image quality if possible (eBay allows size modification)
                if image_url and 's-l' in image_url:
                    # Change thumbnail to larger image (s-l140 -> s-l500/s-l1600)
                    image_url = image_url.replace('s-l140', 's-l500').replace('s-l225', 's-l500')
                
                # Extract price - try multiple structures
                price_value = 0.0
                price_currency = 'USD'
                
                if isinstance(item.get('price'), dict):
                    price_value = float(item['price'].get('value', 0))
                    price_currency = item['price'].get('currency', 'USD')
                elif isinstance(item.get('price'), (int, float)):
                    price_value = float(item['price'])
                elif item.get('sellingStatus'):
                    price_value = float(item['sellingStatus'][0].get('currentPrice', [{}])[0].get('__value__', 0))
                    price_currency = item['sellingStatus'][0].get('currentPrice', [{}])[0].get('@currencyId', 'USD')
                
                # Extract title
                title = item.get('title', query)
                if isinstance(title, list):
                    title = title[0] if title else query
                
                # Extract item ID
                item_id = item.get('itemId', '') or item.get('id', '') or f"item_{hash(title)}"
                if isinstance(item_id, list):
                    item_id = item_id[0] if item_id else f"item_{hash(title)}"
                
                # Extract URL
                item_url = item.get('itemWebUrl', '') or item.get('viewItemURL', '')
                if isinstance(item_url, list):
                    item_url = item_url[0] if item_url else ''
                if not item_url:
                    item_url = f"https://www.ebay.com/sch/i.html?_nkw={query.replace(' ', '+')}"
                
                # Extract condition
                condition = 'Unknown'
                if isinstance(item.get('condition'), str):
                    condition = item['condition']
                elif isinstance(item.get('condition'), dict):
                    condition = item['condition'].get('conditionDisplayName', 'Unknown')
                elif item.get('conditionDisplayName'):
                    condition = item['conditionDisplayName']
                
                # Extract shipping info
                shipping_cost = 0.0
                shipping_info = item.get('shippingOptions', []) or item.get('shippingInfo', [])
                if shipping_info:
                    if isinstance(shipping_info, list) and len(shipping_info) > 0:
                        ship_item = shipping_info[0]
                        if isinstance(ship_item, dict):
                            ship_cost = ship_item.get('shippingCost', {})
                            if isinstance(ship_cost, dict):
                                shipping_cost = float(ship_cost.get('value', 0))
                            elif isinstance(ship_cost, list) and len(ship_cost) > 0:
                                shipping_cost = float(ship_cost[0].get('__value__', 0))
                
                # Build product dictionary
                product = {
                    "id": str(item_id),
                    "name": str(title)[:200],  # Limit title length
                    "image_url": image_url,
                    "price": price_value,
                    "currency": price_currency,
                    "category": self._extract_category(item, query),
                    "ebay_url": item_url,
                    "condition": condition,
                    "shipping": shipping_cost,
                    "search_query": query,  # Added for reference
                }
                
                # Calculate relevance score
                product['relevance_score'] = self._calculate_relevance(product, query)
                
                products.append(product)
                
            except Exception as e:
                logger.warning(f"Failed to parse eBay item: {e}")
                continue
        
        # Sort by relevance score (highest first)
        products.sort(key=lambda x: x['relevance_score'], reverse=True)
        
        logger.info(f"Parsed {len(products)} valid products from eBay results")
        return products[:5]  # Return top 5 most relevant (increased from 3)
    
    def _extract_category(self, item: Dict, query: str) -> str:
        """Extract or infer product category."""
        # Try to get category from item
        category = item.get('categories', [{}])[0].get('categoryName', '')
        
        if not category:
            # Infer from query
            category = query.split()[0].capitalize()
        
        return category
    
    def _calculate_relevance(self, product: Dict, query: str) -> float:
        """
        Calculate relevance score for ranking products.
        
        Scoring factors:
        - Title match with query keywords (0-50 points)
        - Price reasonableness (0-20 points)
        - Condition (0-15 points)
        - Shipping cost (0-15 points)
        
        Returns:
            Relevance score (0-100)
        """
        score = 0.0
        
        # Title match (0-50 points)
        query_words = set(query.lower().split())
        title_words = set(product['name'].lower().split())
        match_ratio = len(query_words & title_words) / len(query_words) if query_words else 0
        score += match_ratio * 50
        
        # Price reasonableness (0-20 points)
        # Prefer items in $20-$200 range
        price = product['price']
        if 20 <= price <= 200:
            score += 20
        elif 10 <= price < 20 or 200 < price <= 500:
            score += 10
        elif price < 10 or price > 500:
            score += 5
        
        # Condition (0-15 points)
        condition = product.get('condition', '').lower()
        if 'new' in condition:
            score += 15
        elif 'like new' in condition or 'excellent' in condition:
            score += 12
        elif 'good' in condition:
            score += 8
        else:
            score += 5
        
        # Shipping cost (0-15 points)
        shipping = product.get('shipping', 0)
        if shipping == 0:
            score += 15  # Free shipping
        elif shipping < 5:
            score += 10
        elif shipping < 10:
            score += 5
        
        return score
    
    def _get_fallback_product(self, query: str) -> List[Dict]:
        """Generate fallback product when API fails."""
        return [{
            "id": f"fallback_{hash(query)}",
            "name": query.title(),
            "image_url": "https://via.placeholder.com/400x600?text=" + query.replace(" ", "+"),
            "price": 29.99,
            "currency": "USD",
            "category": query.split()[0].capitalize(),
            "ebay_url": f"https://www.ebay.com/sch/i.html?_nkw={query.replace(' ', '+')}",
            "condition": "New",
            "shipping": 0,
            "relevance_score": 50.0
        }]
    
    def _get_fallback_recommendations(self) -> List[Dict]:
        """Generate fallback recommendations when entire pipeline fails."""
        fallback_queries = [
            "black dress pants",
            "white button shirt",
            "navy blue blazer",
            "casual denim jeans",
            "leather dress shoes"
        ]
        
        products = []
        for query in fallback_queries:
            products.extend(self._get_fallback_product(query))
        
        return products


recommendation_engine = RecommendationEngine()
