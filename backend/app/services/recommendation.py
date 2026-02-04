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
    
    async def async_call(self, coro_func, *args, **kwargs):
        """Execute async function with circuit breaker protection."""
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
            result = await coro_func(*args, **kwargs)
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
        self.rapidapi_host = os.getenv("RAPIDAPI_HOST", "ebay32.p.rapidapi.com")
        
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
            logger.info("="*60)
            logger.info("RECOMMENDATION PIPELINE STARTED")
            logger.info("="*60)
            
            # Log detailed user profile
            if user_profile:
                logger.info("[USER PROFILE]")
                logger.info(f"  Gender: {user_profile.get('gender', 'Not specified')}")
                logger.info(f"  Ethnicity: {user_profile.get('ethnicity', 'Not specified')}")
                logger.info(f"  Style Preference: {user_profile.get('style_preference', 'Not specified')}")
                logger.info(f"  Body Type: {user_profile.get('body_type', 'Not specified')}")
                if user_profile.get('height_cm') and user_profile.get('weight_kg'):
                    logger.info(f"  Dimensions: {user_profile.get('height_cm')}cm, {user_profile.get('weight_kg')}kg")
            else:
                logger.warning("[USER PROFILE] No user profile provided - recommendations may be generic")
            
            # Step 1: Extract skin tone from user photo (if not provided in profile)
            skin_tone_info = None
            color_recs = None
            
            if user_profile and user_profile.get('skin_tone'):
                # Use provided skin tone
                skin_tone_category = user_profile['skin_tone'].lower().replace(' ', '_')
                color_recs = self.get_color_recommendations(skin_tone_category)
                skin_tone_info = {"skin_tone_category": skin_tone_category, "confidence": 1.0}
                logger.info(f"[SKIN TONE] Using provided: {skin_tone_category}")
            else:
                # Extract from image
                logger.info("[SKIN TONE] Extracting from user photo...")
                skin_tone_info = await self.extract_skin_tone(user_photo)
                color_recs = self.get_color_recommendations(skin_tone_info.get('skin_tone_category', 'medium_warm'))
                logger.info(f"[SKIN TONE] Extracted: {skin_tone_info}")
            
            # Log color recommendations
            logger.info(f"[COLOR THEORY] Best colors: {color_recs.get('best_colors', [])}")
            logger.info(f"[COLOR THEORY] Avoid colors: {color_recs.get('avoid_colors', [])}")
            logger.info(f"[COLOR THEORY] Metals: {color_recs.get('metals', [])}")
            logger.info(f"[COLOR THEORY] Neutrals: {color_recs.get('neutrals', [])}")
            
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
            logger.info(f"[COLLAGE] Created from {len(all_images)} images")
            
            # Step 3: Extract keywords with Gemini Vision (with color theory and user profile)
            keywords = await self._extract_keywords_with_circuit_breaker(
                collage, 
                user_profile, 
                skin_tone_info, 
                color_recs
            )
            logger.info("[KEYWORDS] Extracted keywords for eBay search:")
            for i, kw in enumerate(keywords, 1):
                logger.info(f"  {i}. {kw}")
            
            # Step 4: Search eBay for each keyword (with circuit breaker)
            # Increase results: use top 8 keywords, get 4 items per keyword
            logger.info("[EBAY SEARCH] Starting eBay product search...")
            products = []
            for keyword in keywords[:8]:  # Top 8 keywords
                ebay_products = await self._search_ebay_with_circuit_breaker(keyword)
                if ebay_products:
                    logger.info(f"[EBAY SEARCH] '{keyword}' -> {len(ebay_products)} products found")
                    for p in ebay_products[:2]:  # Log first 2 for brevity
                        logger.info(f"    - {p.get('name', 'N/A')[:50]}... | ${p.get('price', 0):.2f} | img: {p.get('image_url', 'N/A')[:60]}...")
                else:
                    logger.warning(f"[EBAY SEARCH] '{keyword}' -> No products found")
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
            
            logger.info("="*60)
            logger.info(f"[FINAL] Returning {len(final_products)} unique recommendations")
            for i, p in enumerate(final_products, 1):
                logger.info(f"  {i}. {p.get('name', 'N/A')[:40]}... | ${p.get('price', 0):.2f}")
            logger.info("="*60)
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
        gender = user_profile.get('gender', '') if user_profile else ''
        ethnicity = user_profile.get('ethnicity', '') if user_profile else ''
        style = user_profile.get('style_preference', '') if user_profile else ''
        
        if self.gemini_circuit_breaker.is_open():
            logger.warning("[CIRCUIT BREAKER] Gemini circuit breaker is OPEN, using fallback keywords")
            return self._get_fallback_keywords(color_recs, gender, ethnicity, style)
        
        try:
            return await self.gemini_circuit_breaker.async_call(
                self._extract_keywords_with_color_theory,
                collage,
                user_profile,
                skin_tone_info,
                color_recs
            )
        except Exception as e:
            logger.error(f"[GEMINI ERROR] API call failed: {e}")
            return self._get_fallback_keywords(color_recs, gender, ethnicity, style)
    
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
        # Extract user profile fields for fallback
        gender = user_profile.get('gender', '') if user_profile else ''
        ethnicity = user_profile.get('ethnicity', '') if user_profile else ''
        style = user_profile.get('style_preference', '') if user_profile else ''
        
        if not self.gemini_client:
            logger.error("Gemini API not configured, using fallback keywords")
            return self._get_fallback_keywords(color_recs, gender, ethnicity, style)
        
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

**COLOR PALETTE that will look BEST on this person:**
- Flattering Colors: {', '.join(best_colors)}
- Colors to AVOID: {', '.join(avoid_colors)}
- Best Metal Tones: {', '.join(metals)}
- Safe Neutrals: {', '.join(neutrals)}"""

        # Build gender context (minimal, just for eBay search accuracy)
        gender = user_profile.get('gender', '').lower() if user_profile else ''
        gender_suffix = ""
        if gender in ['male', 'man', 'men', 'masculine']:
            gender_suffix = "mens/men"
        elif gender in ['female', 'woman', 'women', 'feminine']:
            gender_suffix = "womens/women"

        # Optional context from profile (secondary, not primary)
        optional_context = ""
        ethnicity = user_profile.get('ethnicity', '') if user_profile else ''
        style_pref = user_profile.get('style_preference', '') if user_profile else ''
        
        if ethnicity or style_pref:
            optional_context = "\n\n**Optional Profile Hints (use only if relevant to what you observe):**"
            if ethnicity:
                optional_context += f"\n- Background: {ethnicity} (consider if their style reflects this)"
            if style_pref:
                optional_context += f"\n- Stated preference: {style_pref} (but trust what you SEE over what's stated)"

        prompt = f"""You are an expert fashion stylist with exceptional visual analysis skills.

**YOUR PRIMARY TASK:** Analyze the person in this image and recommend CLOTHING they would ACTUALLY like based on what you SEE.

{color_theory_context}

**CRITICAL CONSTRAINT - VIRTUAL TRY-ON COMPATIBLE ITEMS ONLY:**
This is for a virtual try-on system. You may ONLY recommend:
✅ TOPS: t-shirts, shirts, blouses, polos, hoodies, sweaters, cardigans, jackets, coats, blazers, crop tops, tank tops, kurtas, tunics
✅ BOTTOMS: pants, jeans, trousers, chinos, joggers, shorts, skirts, leggings, cargo pants, dress pants
✅ DRESSES/FULL: dresses, jumpsuits, rompers, overalls, sarees, gowns

❌ ABSOLUTELY DO NOT RECOMMEND:
- Shoes, sneakers, boots, heels, sandals, loafers (NO FOOTWEAR)
- Caps, hats, beanies (NO HEADWEAR)
- Watches, jewelry, sunglasses, glasses (NO ACCESSORIES)
- Bags, purses, backpacks (NO BAGS)
- Belts, scarves, gloves (NO ACCESSORIES)
- Any non-clothing items

**STEP 1: VISUAL ANALYSIS (Most Important - Study the image carefully)**

Look at the person and analyze:
1. **Current Outfit Style**: What are they wearing RIGHT NOW? Is it casual, formal, streetwear, traditional, athleisure, minimalist, bold, preppy, etc.?
2. **Fashion Vibe**: Do they look trendy, classic, edgy, relaxed, polished, sporty?
3. **Body Language & Presentation**: Confident, casual, professional setting?
4. **Age Bracket & Energy**: Young trendy, mature professional, relaxed adult?
5. **Color Patterns**: What colors are they currently wearing? Do they seem to prefer bold or muted?
6. **Fit Preferences**: Are their clothes fitted, relaxed, oversized?{optional_context}

**STEP 2: DEDUCE THEIR PERSONAL STYLE**

Based on your visual analysis, determine:
- What type of CLOTHING (tops/bottoms/dresses) would this person buy?
- What fits their apparent lifestyle and aesthetic?
- What colors from the recommended palette suit their vibe?

**STEP 3: GENERATE SEARCH KEYWORDS**

Create 8-10 search keywords for CLOTHING ONLY:
1. Match THEIR style (not generic fashion advice)
2. Use colors from the flattering palette above
3. Include "{gender_suffix}" suffix for accurate search results
4. ONLY tops, bottoms, or dresses - nothing else!

**CRITICAL RULES:**
- If they're wearing a casual t-shirt and jeans → suggest casual tops and pants, NOT shoes
- If they look like they prefer minimal/clean aesthetics → don't suggest loud prints
- If they're dressed in traditional/ethnic wear → suggest similar CLOTHING items
- If they look sporty/athletic → suggest athleisure CLOTHING
- MATCH their energy and style - don't impose YOUR taste
- NEVER suggest shoes, caps, watches, or accessories

Output ONLY a JSON array of 8-10 search keywords. No explanation.

Example - Person wearing casual streetwear:
["black oversized hoodie mens", "navy cargo joggers men", "gray graphic tee men", "olive bomber jacket mens", "black jogger pants mens", "white cotton t-shirt men"]

Example - Person in smart casual:
["navy slim fit polo mens", "beige cotton chinos men", "white oxford shirt men", "olive green sweater mens", "gray dress pants mens", "burgundy cardigan men"]

Example - Person in ethnic/traditional wear:
["embroidered cotton kurta mens", "white linen kurta men", "beige churidar pants mens", "navy nehru jacket mens", "silk kurta men", "cotton pajama pants mens"]"""

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
        return self._get_fallback_keywords(color_recs, gender, ethnicity, style)
    
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
    
    def _get_fallback_keywords(self, color_recs: Optional[Dict] = None, gender: str = None, ethnicity: str = None, style: str = None) -> List[str]:
        """
        Return simple fallback keywords when Gemini API fails.
        ONLY returns try-on-able clothing: tops, bottoms, dresses.
        NO shoes, caps, watches, accessories.
        """
        gender = (gender or '').lower()
        
        is_male = gender in ['male', 'man', 'men', 'masculine']
        is_female = gender in ['female', 'woman', 'women', 'feminine']
        
        # Get colors from recommendations or use safe defaults
        if color_recs and color_recs.get('best_colors'):
            colors = color_recs['best_colors'][:4]
            neutrals = color_recs.get('neutrals', ['black', 'white', 'gray'])[:3]
        else:
            colors = ['navy', 'gray', 'olive', 'burgundy']
            neutrals = ['black', 'white', 'beige']
        
        # ONLY clothing that can be virtually tried on - NO shoes, caps, accessories
        if is_male:
            fallback = [
                f"{colors[0]} t-shirt mens",
                f"{colors[1]} casual shirt men",
                f"{neutrals[0]} chinos mens",
                f"{colors[2]} hoodie men",
                "denim jeans mens",
                f"{colors[3]} sweater mens" if len(colors) > 3 else "gray sweater mens",
                "casual jacket mens",
                f"{neutrals[1]} jogger pants men"
            ]
        elif is_female:
            fallback = [
                f"{colors[0]} top womens",
                f"{colors[1]} blouse women",
                f"{neutrals[0]} pants womens",
                f"{colors[2]} cardigan women",
                "denim jeans womens",
                f"{colors[3]} dress womens" if len(colors) > 3 else "casual dress womens",
                "casual jacket womens",
                f"{neutrals[1]} skirt women"
            ]
        else:
            # Gender neutral - still only try-on-able clothing
            fallback = [
                f"{colors[0]} t-shirt",
                f"{colors[1]} hoodie",
                f"{neutrals[0]} joggers",
                f"{colors[2]} sweater",
                "denim jeans",
                "casual jacket",
                f"{neutrals[1]} pants",
                f"{colors[3]} cardigan" if len(colors) > 3 else "gray cardigan"
            ]
        
        logger.info(f"[FALLBACK KEYWORDS] Using try-on compatible fallback: {fallback}")
        return fallback
    
    async def _search_ebay(self, query: str) -> List[Dict]:
        """
        Search for products using Platzi Fake Store API (free, no rate limits, great fashion selection).
        Returns products with real images and generates eBay search links for purchase.
        
        API: https://api.escuelajs.co/api/v1/products
        Categories: 1=Clothes, 4=Shoes, 5=Miscellaneous (accessories)
        
        Args:
            query: Search query string
            
        Returns:
            List of product dictionaries with real images
        """
        import urllib.parse
        
        # Extract key search terms for Platzi API
        search_term = self._extract_search_term(query)
        
        # Platzi API supports title search
        encoded_term = urllib.parse.quote(search_term)
        url = f"https://api.escuelajs.co/api/v1/products?title={encoded_term}&limit=10"
        
        # Retry loop with exponential backoff
        last_error = None
        delay = self.retry_delay
        
        for attempt in range(self.max_retries):
            try:
                logger.info(f"Searching Platzi API for '{search_term}' (attempt {attempt + 1}/{self.max_retries})")
                
                response = await self.http_client.get(url)
                response.raise_for_status()
                products_data = response.json()  # Platzi returns array directly
                
                # Filter to ONLY clothes (category 1) - NO shoes (4) or accessories (5)
                # LEFFA can only try on tops, bottoms, and dresses - not footwear or accessories
                products_data = [
                    p for p in products_data 
                    if p.get('category', {}).get('id') == 1  # Only clothes category
                ]
                
                if not products_data:
                    # Try category-based search as fallback
                    category_id = self._guess_category(query)
                    if category_id:
                        category_url = f"https://api.escuelajs.co/api/v1/products?categoryId={category_id}&limit=10"
                        logger.info(f"Trying category fallback: categoryId={category_id}")
                        response = await self.http_client.get(category_url)
                        if response.status_code == 200:
                            products_data = response.json()
                
                if not products_data:
                    logger.warning(f"No products found for '{search_term}'")
                    return self._get_fallback_product(query)
                
                # CRITICAL: Filter out non-try-on-able items by title keywords
                # Even in clothes category, some items can't be tried on
                excluded_keywords = [
                    'shoe', 'sneaker', 'boot', 'heel', 'loafer', 'sandal', 'slipper', 'pump', 'oxford', 'moccasin',
                    'cap', 'hat', 'beanie', 'visor', 'headband', 'bandana',
                    'watch', 'jewelry', 'necklace', 'bracelet', 'ring', 'earring', 'anklet',
                    'bag', 'purse', 'backpack', 'wallet', 'clutch', 'tote', 'luggage', 'suitcase',
                    'sunglasses', 'glasses', 'eyewear', 'goggles',
                    'belt', 'scarf', 'glove', 'sock', 'tie', 'bow tie', 'cufflink',
                    'mask', 'umbrella', 'keychain', 'phone case',
                    'gokart', 'go-kart', 'vehicle', 'toy', 'game', 'electronic'
                ]
                
                def is_tryonable(item):
                    title = item.get('title', '').lower()
                    return not any(excluded in title for excluded in excluded_keywords)
                
                products_data = [p for p in products_data if is_tryonable(p)]
                
                if not products_data:
                    logger.warning(f"All products filtered out for '{search_term}' (non-try-on-able)")
                    return self._get_fallback_product(query)
                
                # Parse and return products with real images
                products = []
                for item in products_data[:4]:  # Top 4 per keyword
                    # Get the best image (first from images array)
                    images = item.get('images', [])
                    image_url = None
                    
                    # Find a valid imgur image (skip escuelajs.co uploads which may be junk)
                    for img in images:
                        if img and 'imgur.com' in img:
                            image_url = img
                            break
                    
                    # Fallback to first image if no imgur found
                    if not image_url and images:
                        image_url = images[0]
                    
                    if not image_url or 'escuelajs.co' in image_url:
                        continue  # Skip items without proper images
                    
                    # Build eBay search URL for "Buy" button
                    ebay_search_url = f"https://www.ebay.com/sch/i.html?_nkw={query.replace(' ', '+')}"
                    
                    # Get category name from item
                    item_category = item.get('category', {}).get('name', 'Fashion')
                    
                    product = {
                        "id": str(item.get('id', hash(item.get('title', '')))),
                        "name": item.get('title', query),
                        "image_url": image_url,
                        "price": float(item.get('price', 0)),
                        "currency": "USD",
                        "category": item_category,
                        "ebay_url": ebay_search_url,
                        "condition": "New",
                        "shipping": 0.0,
                        "search_query": query,
                        "brand": "Fashion Brand",
                        "rating": 4.5,
                        "description": item.get('description', '')[:100],
                    }
                    product['relevance_score'] = self._calculate_relevance(product, query)
                    products.append(product)
                    logger.info(f"[PRODUCT] {product['name'][:50]}... | ${product['price']} | img: {image_url[:60]}...")
                
                if products:
                    logger.info(f"Returning {len(products)} fashion products for '{query}'")
                    self.ebay_circuit_breaker.on_success()
                    return products
                else:
                    return self._get_fallback_product(query)
                    
            except httpx.TimeoutException:
                last_error = "Request timeout"
                logger.warning(f"Platzi API timeout (attempt {attempt + 1}/{self.max_retries})")
                
            except httpx.HTTPStatusError as e:
                last_error = str(e)
                logger.warning(f"Platzi API HTTP error: {e.response.status_code} (attempt {attempt + 1}/{self.max_retries})")
                    
            except Exception as e:
                last_error = str(e)
                logger.warning(f"Platzi API error: {e} (attempt {attempt + 1}/{self.max_retries})")
            
            # Wait before retry
            if attempt < self.max_retries - 1:
                logger.info(f"Retrying in {delay} seconds...")
                await asyncio.sleep(delay)
                delay *= self.retry_backoff
        
        # All retries failed
        logger.error(f"All Platzi API retries failed for '{query}'. Last error: {last_error}")
        self.ebay_circuit_breaker.on_failure()
        return self._get_fallback_product(query)
    
    def _extract_search_term(self, query: str) -> str:
        """Extract clean search term from the query for Platzi API.
        
        ONLY extracts terms for try-on-able clothing:
        - Tops: shirts, hoodies, jackets, sweaters, etc.
        - Bottoms: pants, jeans, shorts, joggers, etc.
        - Dresses: dresses, jumpsuits, etc.
        
        Ignores shoes, caps, accessories completely.
        """
        query_lower = query.lower()
        
        # Remove gender terms and common modifiers
        remove_terms = ['mens', 'men', 'womens', 'women', 'male', 'female', 
                        'casual', 'formal', 'stylish', 'trendy', 'modern', 'elegant']
        for term in remove_terms:
            query_lower = query_lower.replace(term, '').strip()
        
        # ONLY try-on-able clothing terms - NO shoes, caps, accessories
        priority_matches = {
            # TOPS
            'hoodie': 'hoodie',
            't-shirt': 't-shirt',
            'tshirt': 't-shirt',
            'shirt': 'shirt',
            'blouse': 'shirt',
            'polo': 'shirt',
            'jacket': 'jacket',
            'blazer': 'jacket',
            'coat': 'jacket',
            'sweater': 'sweater',
            'cardigan': 'sweater',
            'top': 'shirt',
            'kurta': 'shirt',
            'tunic': 'shirt',
            # BOTTOMS
            'jogger': 'jogger',
            'joggers': 'jogger', 
            'pants': 'pants',
            'trousers': 'pants',
            'chinos': 'pants',
            'jean': 'jeans',
            'jeans': 'jeans',
            'shorts': 'shorts',
            'skirt': 'shorts',  # Map to shorts for search
            'leggings': 'jogger',
            # DRESSES/FULL
            'dress': 'dress',
            'jumpsuit': 'dress',
            'romper': 'dress',
            'overall': 'dress',
        }
        
        # Check for priority matches (try-on-able items only)
        for keyword, search_term in priority_matches.items():
            if keyword in query_lower:
                return search_term
        
        # Try to extract a color + default to shirt/pants
        colors = ['black', 'white', 'red', 'blue', 'navy', 'green', 'pink', 
                  'gray', 'grey', 'brown', 'purple', 'orange', 'teal']
        
        for color in colors:
            if color in query_lower:
                return color
        
        # Default to 'shirt' for tops or 'pants' for bottoms - always try-on-able
        if any(top_word in query_lower for top_word in ['top', 'upper', 'chest']):
            return 'shirt'
        if any(bottom_word in query_lower for bottom_word in ['bottom', 'lower', 'leg']):
            return 'pants'
        
        # Generic default - shirt is most common try-on item
        return 'shirt'
    
    def _guess_category(self, query: str) -> Optional[int]:
        """Always return clothes category (1) for virtual try-on.
        
        LEFFA can only try on:
        - Tops (shirts, hoodies, jackets, etc.)
        - Bottoms (pants, jeans, shorts, etc.)
        - Dresses/full body items
        
        We NEVER want shoes (4) or accessories (5) since they can't be tried on.
        """
        # Always return clothes category - we filter everything else out
        return 1
    
    def _extract_category_from_query(self, query: str) -> str:
        """Extract category from search query."""
        # Common clothing categories
        categories = {
            'shirt': 'Shirts', 'tshirt': 'T-Shirts', 't-shirt': 'T-Shirts',
            'polo': 'Polos', 'jacket': 'Jackets', 'hoodie': 'Hoodies',
            'sweater': 'Sweaters', 'jeans': 'Jeans', 'pants': 'Pants',
            'chinos': 'Pants', 'shorts': 'Shorts', 'sneakers': 'Shoes',
            'shoes': 'Shoes', 'boots': 'Shoes', 'blazer': 'Blazers',
            'coat': 'Coats', 'dress': 'Dresses', 'skirt': 'Skirts',
            'blouse': 'Tops', 'cardigan': 'Knitwear'
        }
        
        query_lower = query.lower()
        for keyword, category in categories.items():
            if keyword in query_lower:
                return category
        
        return query.split()[0].capitalize() if query else 'Clothing'
    
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
        """Generate fallback product when API fails - returns eBay search link without placeholder image."""
        logger.warning(f"[FALLBACK] Using fallback product for query: {query}")
        # Use a real eBay search URL - the frontend will handle the missing image gracefully
        ebay_search_url = f"https://www.ebay.com/sch/i.html?_nkw={query.replace(' ', '+')}"
        return [{
            "id": f"search_{hash(query)}",
            "name": f"Search eBay: {query.title()}",
            # Use eBay logo as fallback image - always available
            "image_url": "https://ir.ebaystatic.com/pictures/aw/pics/s_1x2.gif",
            "price": 0.0,
            "currency": "USD",
            "category": query.split()[0].capitalize() if query.split() else "Fashion",
            "ebay_url": ebay_search_url,
            "condition": "Various",
            "shipping": 0,
            "relevance_score": 25.0,
            "is_search_fallback": True
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
