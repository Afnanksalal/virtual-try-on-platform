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
from ..core.cache_manager import cache_manager

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
    1. Create image collage from user photos, wardrobe, generated bodies
    2. Use Gemini Vision to extract keywords with color theory
    3. Search eBay via RapidAPI using extracted keywords
    4. Return product listings with direct eBay links
    """
    
    def __init__(self):
        # API Keys
        self.gemini_key = os.getenv("GEMINI_API_KEY")
        self.rapidapi_key = os.getenv("RAPIDAPI_KEY")
        self.rapidapi_host = os.getenv("RAPIDAPI_HOST", "ebay-search-result.p.rapidapi.com")
        
        # Retry configuration
        self.max_retries = 3
        self.retry_delay = 1.0  # seconds
        self.retry_backoff = 2.0  # exponential backoff multiplier
        
        # Cache configuration
        self.cache_ttl = 3600  # 1 hour
        
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
    
    async def get_outfit_recommendations(
        self,
        user_photo: Image.Image,
        wardrobe_images: List[Image.Image] = None,
        generated_images: List[Image.Image] = None,
        user_id: Optional[str] = None
    ) -> List[Dict]:
        """
        Main recommendation pipeline with caching and fallback.
        
        Args:
            user_photo: User's profile photo
            wardrobe_images: List of wardrobe items
            generated_images: Generated body/outfit images
            user_id: Optional user ID for per-user caching
            
        Returns:
            List of eBay products with links
        """
        # Generate cache key from images
        cache_key = self._generate_cache_key(user_photo, wardrobe_images, generated_images, user_id)
        
        # Try to get cached recommendations
        cached_recommendations = cache_manager.get("recommendations", cache_key)
        if cached_recommendations:
            logger.info(f"Returning cached recommendations for key: {cache_key}")
            return cached_recommendations
        
        try:
            # Step 1: Create collage with labels
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
            
            # Step 2: Extract keywords with Gemini Vision (with circuit breaker)
            keywords = await self._extract_keywords_with_circuit_breaker(collage)
            logger.info(f"Extracted keywords: {keywords}")
            
            # Step 3: Search eBay for each keyword (with circuit breaker)
            products = []
            for keyword in keywords[:5]:  # Top 5 keywords
                ebay_products = await self._search_ebay_with_circuit_breaker(keyword)
                products.extend(ebay_products[:2])  # Top 2 per keyword
            
            # Deduplicate products by ID
            seen_ids = set()
            unique_products = []
            for product in products:
                if product['id'] not in seen_ids:
                    seen_ids.add(product['id'])
                    unique_products.append(product)
            
            # Return top 10 products
            final_products = unique_products[:10]
            
            # Cache the results
            cache_manager.set("recommendations", cache_key, final_products, ttl=self.cache_ttl)
            logger.info(f"Cached {len(final_products)} recommendations for key: {cache_key}")
            
            return final_products
            
        except Exception as e:
            logger.error(f"Recommendation pipeline failed: {e}", exc_info=True)
            
            # Try to return cached results even if expired
            cached_recommendations = cache_manager.get("recommendations", cache_key)
            if cached_recommendations:
                logger.info("Returning stale cached recommendations due to error")
                return cached_recommendations
            
            # Return graceful error message with fallback products
            logger.warning("No cached recommendations available, returning fallback")
            return self._get_fallback_recommendations()
    
    def _generate_cache_key(
        self,
        user_photo: Image.Image,
        wardrobe_images: Optional[List[Image.Image]],
        generated_images: Optional[List[Image.Image]],
        user_id: Optional[str]
    ) -> str:
        """Generate cache key from images and user ID."""
        # Create hash from image data
        hasher = hashlib.sha256()
        
        # Add user ID if provided
        if user_id:
            hasher.update(user_id.encode())
        
        # Add user photo hash
        hasher.update(user_photo.tobytes())
        
        # Add wardrobe images
        if wardrobe_images:
            for img in wardrobe_images:
                hasher.update(img.tobytes())
        
        # Add generated images
        if generated_images:
            for img in generated_images:
                hasher.update(img.tobytes())
        
        return hasher.hexdigest()
    
    async def _extract_keywords_with_circuit_breaker(self, collage: Image.Image) -> List[str]:
        """Extract keywords with circuit breaker protection."""
        if self.gemini_circuit_breaker.is_open():
            logger.warning("Gemini circuit breaker is OPEN, using fallback keywords")
            return self._get_fallback_keywords()
        
        try:
            return self.gemini_circuit_breaker.call(
                self._extract_keywords_with_color_theory_sync,
                collage
            )
        except Exception as e:
            logger.error(f"Gemini API call failed: {e}")
            return self._get_fallback_keywords()
    
    def _extract_keywords_with_color_theory_sync(self, collage: Image.Image) -> List[str]:
        """Synchronous version for circuit breaker."""
        # This is a wrapper to make the async method work with circuit breaker
        import asyncio
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(self._extract_keywords_with_color_theory(collage))
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
    
    async def _extract_keywords_with_color_theory(self, collage: Image.Image) -> List[str]:
        """
        Use Gemini Vision to extract clothing keywords with color theory.
        Implements retry logic with exponential backoff.
        """
        if not self.gemini_client:
            logger.error("Gemini API not configured, using fallback keywords")
            return self._get_fallback_keywords()
        
        prompt = """Analyze this fashion collage image carefully.

Extract clothing recommendation keywords based on:
1. **Color Theory**: Identify dominant colors and suggest complementary/analogous colors
   - Complementary: Opposite on color wheel (e.g., blue → orange)
   - Analogous: Adjacent colors (e.g., blue → blue-green → green)
   - Triadic: Evenly spaced colors
   
2. **Style Analysis**: Identify clothing style (casual, formal, sporty, elegant)

3. **Body Type Considerations**: Suggest items that would flatter the body type shown

4. **Items Needed**: Suggest specific clothing items that would complete the wardrobe

Output ONLY a JSON array of search keywords (strings), 5-8 keywords maximum.
Each keyword should be specific and eBay-searchable (e.g., "navy blue blazer", "white dress shirt").

Example output:
["navy blue blazer", "white dress shirt", "burgundy tie", "black dress pants", "brown leather shoes"]

CRITICAL: Output ONLY the JSON array, no other text."""

        # Retry loop with exponential backoff
        last_error = None
        delay = self.retry_delay
        
        for attempt in range(self.max_retries):
            try:
                logger.info(f"Attempting Gemini Vision API call (attempt {attempt + 1}/{self.max_retries})")
                
                # Save image temporarily for upload
                import tempfile
                with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
                    collage.save(tmp.name, format='PNG')
                    tmp_path = tmp.name
                
                try:
                    # Upload image and generate content
                    response = self.gemini_client.models.generate_content(
                        model='gemini-2.0-flash-exp',
                        contents=[
                            types.Part.from_text(text=prompt),
                            types.Part.from_uri(
                                file_uri=tmp_path,
                                mime_type='image/png'
                            )
                        ]
                    )
                    
                    text = response.text.strip()
                    logger.debug(f"Gemini response: {text[:200]}...")
                    
                    # Extract JSON from response
                    keywords = self._parse_keywords_from_response(text)
                    
                    if keywords and len(keywords) > 0:
                        logger.info(f"Successfully extracted {len(keywords)} keywords from Gemini")
                        return keywords
                    else:
                        logger.warning("Gemini returned empty keywords, retrying...")
                        raise ValueError("Empty keywords returned")
                finally:
                    # Clean up temp file
                    import os
                    if os.path.exists(tmp_path):
                        os.unlink(tmp_path)
                    
            except Exception as e:
                last_error = e
                logger.warning(f"Gemini API call failed (attempt {attempt + 1}/{self.max_retries}): {e}")
                
                if attempt < self.max_retries - 1:
                    logger.info(f"Retrying in {delay} seconds...")
                    await asyncio.sleep(delay)
                    delay *= self.retry_backoff  # Exponential backoff
        
        # All retries failed, use fallback
        logger.error(f"All Gemini API retries failed. Last error: {last_error}")
        return self._get_fallback_keywords()
    
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
    
    def _get_fallback_keywords(self) -> List[str]:
        """Return fallback keywords when Gemini API fails."""
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
        
        Args:
            data: Raw eBay API response
            query: Original search query for relevance scoring
            
        Returns:
            List of parsed and ranked products
        """
        products = []
        results = data.get('results', [])
        
        for item in results[:10]:  # Limit to top 10 results
            try:
                # Extract product data
                product = {
                    "id": item.get('itemId', ''),
                    "name": item.get('title', query),
                    "image_url": item.get('image', {}).get('imageUrl', 'https://via.placeholder.com/400'),
                    "price": float(item.get('price', {}).get('value', 0)),
                    "currency": item.get('price', {}).get('currency', 'USD'),
                    "category": self._extract_category(item, query),
                    "ebay_url": item.get('itemWebUrl', f"https://www.ebay.com/sch/i.html?_nkw={query.replace(' ', '+')}"),
                    "condition": item.get('condition', 'Unknown'),
                    "shipping": item.get('shippingOptions', [{}])[0].get('shippingCost', {}).get('value', 0),
                }
                
                # Calculate relevance score
                product['relevance_score'] = self._calculate_relevance(product, query)
                
                products.append(product)
                
            except Exception as e:
                logger.warning(f"Failed to parse eBay item: {e}")
                continue
        
        # Sort by relevance score (highest first)
        products.sort(key=lambda x: x['relevance_score'], reverse=True)
        
        return products[:3]  # Return top 3 most relevant
    
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
