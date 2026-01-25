"""
Simple placeholder for removed cache functionality.
"""

import logging

logger = logging.getLogger("cache_manager")


class CacheManager:
    """Disabled cache manager - no caching."""
    
    def __init__(self, default_ttl: int = 86400):
        """Initialize disabled cache manager."""
        logger.info("Caching disabled")
    
    def _generate_key(self, namespace: str, key: str) -> str:
        """Generate cache key (unused)."""
        return f"{namespace}:{key}"
    
    def get(self, namespace: str, key: str):
        """Always return None (no cache)."""
        return None
    
    def set(self, namespace: str, key: str, value, ttl=None) -> bool:
        """Do nothing (no cache)."""
        return True
    
    def delete(self, namespace: str, key: str) -> bool:
        """Do nothing (no cache)."""
        return True
    
    def clear_namespace(self, namespace: str) -> int:
        """Do nothing (no cache)."""
        return 0
    
    def get_stats(self) -> dict:
        """Return empty stats."""
        return {
            "hits": 0,
            "misses": 0,
            "sets": 0,
            "deletes": 0,
            "hit_rate": 0.0,
            "backend": "disabled",
            "memory_cache_size": 0,
        }
    
    def reset_stats(self):
        """Do nothing."""
        pass


# Global cache manager instance (disabled)
cache_manager = CacheManager(default_ttl=86400)
