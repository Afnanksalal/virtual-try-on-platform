"""
Cache Manager for Intermediate Results

This module provides caching functionality for ML pipeline intermediate results
using Redis with fallback to in-memory caching.
"""

import os
import pickle
import hashlib
import json
from typing import Any, Optional
from datetime import timedelta
import logging

logger = logging.getLogger("cache_manager")


class CacheManager:
    """
    Cache manager with Redis backend and in-memory fallback.
    
    Features:
    - Redis-based distributed caching
    - In-memory fallback when Redis unavailable
    - Automatic serialization/deserialization
    - TTL support
    - Cache statistics
    """
    
    def __init__(
        self,
        redis_url: Optional[str] = None,
        default_ttl: int = 86400,  # 24 hours
        use_redis: bool = True,
    ):
        """
        Initialize cache manager.
        
        Args:
            redis_url: Redis connection URL (e.g., "redis://localhost:6379/0")
            default_ttl: Default TTL in seconds (24 hours)
            use_redis: Whether to use Redis (falls back to in-memory if False)
        """
        self.default_ttl = default_ttl
        self.redis_client = None
        self.in_memory_cache = {}
        self.cache_stats = {
            "hits": 0,
            "misses": 0,
            "sets": 0,
            "deletes": 0,
        }
        
        # Try to connect to Redis if enabled
        if use_redis:
            self._connect_redis(redis_url)
        
        if self.redis_client is None:
            logger.warning("Redis not available, using in-memory cache")
    
    def _connect_redis(self, redis_url: Optional[str]):
        """Attempt to connect to Redis."""
        try:
            import redis
            
            if redis_url:
                self.redis_client = redis.from_url(
                    redis_url,
                    decode_responses=False,  # We'll handle binary data
                    socket_connect_timeout=2,
                    socket_timeout=2,
                )
            else:
                # Try default connection
                self.redis_client = redis.Redis(
                    host="localhost",
                    port=6379,
                    db=0,
                    decode_responses=False,
                    socket_connect_timeout=2,
                    socket_timeout=2,
                )
            
            # Test connection
            self.redis_client.ping()
            logger.info("Connected to Redis successfully")
            
        except ImportError:
            logger.warning("redis-py not installed, using in-memory cache")
            self.redis_client = None
        except Exception as e:
            logger.warning(f"Failed to connect to Redis: {e}, using in-memory cache")
            self.redis_client = None
    
    def _generate_key(self, namespace: str, key: str) -> str:
        """
        Generate cache key with namespace.
        
        Args:
            namespace: Cache namespace (e.g., "segmentation", "pose")
            key: Cache key (typically image hash)
            
        Returns:
            Full cache key
        """
        return f"idm_vton:{namespace}:{key}"
    
    def _serialize(self, value: Any) -> bytes:
        """Serialize value for storage."""
        return pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
    
    def _deserialize(self, data: bytes) -> Any:
        """Deserialize value from storage."""
        return pickle.loads(data)
    
    def get(self, namespace: str, key: str) -> Optional[Any]:
        """
        Get value from cache.
        
        Args:
            namespace: Cache namespace
            key: Cache key
            
        Returns:
            Cached value or None if not found
        """
        cache_key = self._generate_key(namespace, key)
        
        try:
            # Try Redis first
            if self.redis_client is not None:
                data = self.redis_client.get(cache_key)
                if data is not None:
                    self.cache_stats["hits"] += 1
                    return self._deserialize(data)
            
            # Fallback to in-memory cache
            if cache_key in self.in_memory_cache:
                self.cache_stats["hits"] += 1
                return self.in_memory_cache[cache_key]
            
            self.cache_stats["misses"] += 1
            return None
            
        except Exception as e:
            logger.error(f"Cache get error: {e}")
            self.cache_stats["misses"] += 1
            return None
    
    def set(
        self,
        namespace: str,
        key: str,
        value: Any,
        ttl: Optional[int] = None
    ) -> bool:
        """
        Set value in cache.
        
        Args:
            namespace: Cache namespace
            key: Cache key
            value: Value to cache
            ttl: Time-to-live in seconds (uses default if None)
            
        Returns:
            True if successful, False otherwise
        """
        cache_key = self._generate_key(namespace, key)
        ttl = ttl or self.default_ttl
        
        try:
            serialized = self._serialize(value)
            
            # Try Redis first
            if self.redis_client is not None:
                self.redis_client.setex(
                    cache_key,
                    ttl,
                    serialized
                )
            else:
                # Fallback to in-memory cache
                self.in_memory_cache[cache_key] = value
            
            self.cache_stats["sets"] += 1
            return True
            
        except Exception as e:
            logger.error(f"Cache set error: {e}")
            return False
    
    def delete(self, namespace: str, key: str) -> bool:
        """
        Delete value from cache.
        
        Args:
            namespace: Cache namespace
            key: Cache key
            
        Returns:
            True if successful, False otherwise
        """
        cache_key = self._generate_key(namespace, key)
        
        try:
            # Try Redis first
            if self.redis_client is not None:
                self.redis_client.delete(cache_key)
            
            # Also delete from in-memory cache
            if cache_key in self.in_memory_cache:
                del self.in_memory_cache[cache_key]
            
            self.cache_stats["deletes"] += 1
            return True
            
        except Exception as e:
            logger.error(f"Cache delete error: {e}")
            return False
    
    def clear_namespace(self, namespace: str) -> int:
        """
        Clear all keys in a namespace.
        
        Args:
            namespace: Cache namespace to clear
            
        Returns:
            Number of keys deleted
        """
        pattern = self._generate_key(namespace, "*")
        deleted = 0
        
        try:
            # Try Redis first
            if self.redis_client is not None:
                keys = self.redis_client.keys(pattern)
                if keys:
                    deleted = self.redis_client.delete(*keys)
            
            # Also clear from in-memory cache
            keys_to_delete = [
                k for k in self.in_memory_cache.keys()
                if k.startswith(f"idm_vton:{namespace}:")
            ]
            for key in keys_to_delete:
                del self.in_memory_cache[key]
                deleted += 1
            
            return deleted
            
        except Exception as e:
            logger.error(f"Cache clear error: {e}")
            return 0
    
    def get_stats(self) -> dict:
        """
        Get cache statistics.
        
        Returns:
            Dictionary with cache statistics
        """
        stats = self.cache_stats.copy()
        
        # Add hit rate
        total_requests = stats["hits"] + stats["misses"]
        if total_requests > 0:
            stats["hit_rate"] = stats["hits"] / total_requests
        else:
            stats["hit_rate"] = 0.0
        
        # Add backend info
        stats["backend"] = "redis" if self.redis_client is not None else "memory"
        
        # Add memory cache size
        stats["memory_cache_size"] = len(self.in_memory_cache)
        
        return stats
    
    def reset_stats(self):
        """Reset cache statistics."""
        self.cache_stats = {
            "hits": 0,
            "misses": 0,
            "sets": 0,
            "deletes": 0,
        }


# Global cache manager instance
cache_manager = CacheManager(
    redis_url=os.getenv("REDIS_URL"),  # Read from environment
    default_ttl=86400,  # 24 hours
    use_redis=True,
)
