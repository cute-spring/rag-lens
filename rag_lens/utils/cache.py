"""
Caching Utilities for RAG Lens

This module provides caching utilities to improve performance and reduce
repeated API calls and expensive computations.
"""

import hashlib
import json
import pickle
import time
from typing import Any, Dict, Optional, Union, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
import threading
from functools import wraps

from .logger import get_logger
from .errors import CacheError
from ..config.settings import config

logger = get_logger(__name__)


@dataclass
class CacheEntry:
    """Represents a single cache entry"""
    key: str
    value: Any
    created_at: float = field(default_factory=time.time)
    ttl: Optional[float] = None
    access_count: int = 0
    last_accessed: float = field(default_factory=time.time)

    def is_expired(self) -> bool:
        """Check if cache entry is expired"""
        if self.ttl is None:
            return False
        return (time.time() - self.created_at) > self.ttl

    def update_access(self):
        """Update access information"""
        self.access_count += 1
        self.last_accessed = time.time()


class CacheManager:
    """Thread-safe cache manager with multiple backends"""

    def __init__(self, max_size: int = 1000, default_ttl: int = 3600):
        """
        Initialize cache manager

        Args:
            max_size: Maximum number of items in cache
            default_ttl: Default time-to-live in seconds
        """
        self.max_size = max_size
        self.default_ttl = default_ttl
        self._memory_cache: Dict[str, CacheEntry] = {}
        self._lock = threading.RLock()
        self._hit_count = 0
        self._miss_count = 0
        self._eviction_count = 0

        # Initialize file cache
        self._file_cache_dir = Path("cache")
        self._file_cache_dir.mkdir(exist_ok=True)

    def _generate_key(self, func: Callable, *args, **kwargs) -> str:
        """Generate cache key from function and arguments"""
        # Create a string representation of the call
        key_data = {
            "function": func.__name__,
            "module": func.__module__,
            "args": str(args),
            "kwargs": str(sorted(kwargs.items()))
        }

        # Hash the key data for consistency
        key_string = json.dumps(key_data, sort_keys=True)
        return hashlib.md5(key_string.encode()).hexdigest()

    def _evict_if_needed(self):
        """Evict entries if cache is full using LRU strategy"""
        if len(self._memory_cache) < self.max_size:
            return

        # Sort by last access time (LRU)
        entries = sorted(
            self._memory_cache.values(),
            key=lambda x: x.last_accessed
        )

        # Evict 10% of entries
        evict_count = max(1, int(self.max_size * 0.1))
        for entry in entries[:evict_count]:
            del self._memory_cache[entry.key]

        self._eviction_count += evict_count
        logger.info(f"Evicted {evict_count} entries from cache")

    def _clean_expired_entries(self):
        """Clean expired entries from cache"""
        expired_keys = [
            key for key, entry in self._memory_cache.items()
            if entry.is_expired()
        ]

        for key in expired_keys:
            del self._memory_cache[key]

        if expired_keys:
            logger.debug(f"Cleaned {len(expired_keys)} expired entries from cache")

    def get(self, key: str, default: Any = None) -> Any:
        """Get value from cache"""
        with self._lock:
            # Clean expired entries periodically
            if time.time() % 100 < 1:  # Roughly every 100 seconds
                self._clean_expired_entries()

            entry = self._memory_cache.get(key)
            if entry is None:
                self._miss_count += 1
                # Try file cache as fallback
                return self._get_from_file_cache(key, default)

            if entry.is_expired():
                del self._memory_cache[key]
                self._miss_count += 1
                return default

            entry.update_access()
            self._hit_count += 1
            return entry.value

    def set(self, key: str, value: Any, ttl: Optional[float] = None):
        """Set value in cache"""
        with self._lock:
            self._evict_if_needed()

            if ttl is None:
                ttl = self.default_ttl

            entry = CacheEntry(
                key=key,
                value=value,
                ttl=ttl
            )

            self._memory_cache[key] = entry

            # Also cache to file for persistence
            self._set_to_file_cache(key, value, ttl)

    def delete(self, key: str):
        """Delete value from cache"""
        with self._lock:
            if key in self._memory_cache:
                del self._memory_cache[key]

            # Also remove from file cache
            self._delete_from_file_cache(key)

    def clear(self):
        """Clear all cache entries"""
        with self._lock:
            self._memory_cache.clear()
            self._clear_file_cache()
            self._hit_count = 0
            self._miss_count = 0
            self._eviction_count = 0

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        with self._lock:
            total_requests = self._hit_count + self._miss_count
            hit_rate = self._hit_count / total_requests if total_requests > 0 else 0

            return {
                "size": len(self._memory_cache),
                "max_size": self.max_size,
                "hit_count": self._hit_count,
                "miss_count": self._miss_count,
                "hit_rate": hit_rate,
                "eviction_count": self._eviction_count,
                "memory_usage_mb": self._get_memory_usage()
            }

    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB"""
        try:
            import sys
            return sys.getsizeof(self._memory_cache) / (1024 * 1024)
        except ImportError:
            return 0

    # File cache methods
    def _get_file_path(self, key: str) -> Path:
        """Get file path for cache key"""
        return self._file_cache_dir / f"{key}.cache"

    def _get_from_file_cache(self, key: str, default: Any) -> Any:
        """Get value from file cache"""
        file_path = self._get_file_path(key)
        if not file_path.exists():
            return default

        try:
            with open(file_path, 'rb') as f:
                entry = pickle.load(f)

            if entry.is_expired():
                file_path.unlink()
                return default

            entry.update_access()
            # Update memory cache
            self._memory_cache[key] = entry
            return entry.value

        except (pickle.PickleError, EOFError, OSError) as e:
            logger.warning(f"Failed to read from file cache: {e}")
            if file_path.exists():
                file_path.unlink()
            return default

    def _set_to_file_cache(self, key: str, value: Any, ttl: Optional[float]):
        """Set value in file cache"""
        file_path = self._get_file_path(key)

        try:
            entry = CacheEntry(key=key, value=value, ttl=ttl)
            with open(file_path, 'wb') as f:
                pickle.dump(entry, f)
        except (pickle.PickleError, OSError) as e:
            logger.warning(f"Failed to write to file cache: {e}")

    def _delete_from_file_cache(self, key: str):
        """Delete value from file cache"""
        file_path = self._get_file_path(key)
        if file_path.exists():
            try:
                file_path.unlink()
            except OSError as e:
                logger.warning(f"Failed to delete from file cache: {e}")

    def _clear_file_cache(self):
        """Clear all file cache entries"""
        try:
            for file_path in self._file_cache_dir.glob("*.cache"):
                file_path.unlink()
        except OSError as e:
            logger.warning(f"Failed to clear file cache: {e}")


def cached(
    ttl: Optional[int] = None,
    key_func: Optional[Callable] = None,
    ignore_args: bool = False
):
    """
    Decorator for caching function results

    Args:
        ttl: Time-to-live in seconds
        key_func: Custom function to generate cache keys
        ignore_args: If True, cache key won't include function arguments
    """
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Get cache instance
            cache = get_cache()

            # Generate cache key
            if key_func:
                key = key_func(func, *args, **kwargs)
            elif ignore_args:
                key = f"{func.__module__}.{func.__name__}"
            else:
                key = cache._generate_key(func, *args, **kwargs)

            # Try to get from cache
            cached_result = cache.get(key)
            if cached_result is not None:
                logger.debug(f"Cache hit for {func.__name__}")
                return cached_result

            # Call function and cache result
            logger.debug(f"Cache miss for {func.__name__}")
            result = func(*args, **kwargs)
            cache.set(key, result, ttl)

            return result

        return wrapper
    return decorator


def cached_async(ttl: Optional[int] = None):
    """
    Decorator for caching async function results
    """
    def decorator(func: Callable):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Get cache instance
            cache = get_cache()

            # Generate cache key
            key = cache._generate_key(func, *args, **kwargs)

            # Try to get from cache
            cached_result = cache.get(key)
            if cached_result is not None:
                logger.debug(f"Cache hit for async {func.__name__}")
                return cached_result

            # Call function and cache result
            logger.debug(f"Cache miss for async {func.__name__}")
            result = await func(*args, **kwargs)
            cache.set(key, result, ttl)

            return result

        return wrapper
    return decorator


class APICache:
    """Specialized cache for API responses"""

    def __init__(self, cache_manager: CacheManager):
        self.cache = cache_manager

    def get_cached_response(self, url: str, method: str = "GET", **kwargs) -> Optional[Dict]:
        """Get cached API response"""
        key = self._generate_api_key(url, method, **kwargs)
        return self.cache.get(key)

    def cache_response(
        self,
        url: str,
        response: Dict,
        method: str = "GET",
        ttl: int = 300,  # 5 minutes default for API responses
        **kwargs
    ):
        """Cache API response"""
        key = self._generate_api_key(url, method, **kwargs)
        self.cache.set(key, response, ttl)

    def _generate_api_key(self, url: str, method: str, **kwargs) -> str:
        """Generate key for API request"""
        # Normalize URL and parameters
        normalized_url = url.rstrip('/')
        normalized_params = sorted(kwargs.items())

        key_data = {
            "url": normalized_url,
            "method": method.upper(),
            "params": str(normalized_params)
        }

        key_string = json.dumps(key_data, sort_keys=True)
        return hashlib.md5(key_string.encode()).hexdigest()


class QueryResultCache:
    """Specialized cache for RAG query results"""

    def __init__(self, cache_manager: CacheManager):
        self.cache = cache_manager

    def get_cached_query_result(self, query: str, context: Dict) -> Optional[Dict]:
        """Get cached query result"""
        key = self._generate_query_key(query, context)
        return self.cache.get(key)

    def cache_query_result(
        self,
        query: str,
        context: Dict,
        result: Dict,
        ttl: int = 1800  # 30 minutes default for query results
    ):
        """Cache query result"""
        key = self._generate_query_key(query, context)
        self.cache.set(key, result, ttl)

    def _generate_query_key(self, query: str, context: Dict) -> str:
        """Generate key for query"""
        # Normalize query and context
        normalized_query = query.strip().lower()
        normalized_context = {k: str(v) for k, v in sorted(context.items())}

        key_data = {
            "query": normalized_query,
            "context": normalized_context
        }

        key_string = json.dumps(key_data, sort_keys=True)
        return hashlib.md5(key_string.encode()).hexdigest()


class EmbeddingCache:
    """Specialized cache for text embeddings"""

    def __init__(self, cache_manager: CacheManager):
        self.cache = cache_manager

    def get_cached_embedding(self, text: str, model: str) -> Optional[list]:
        """Get cached embedding"""
        key = self._generate_embedding_key(text, model)
        return self.cache.get(key)

    def cache_embedding(
        self,
        text: str,
        model: str,
        embedding: list,
        ttl: int = 86400  # 24 hours default for embeddings
    ):
        """Cache embedding"""
        key = self._generate_embedding_key(text, model)
        self.cache.set(key, embedding, ttl)

    def _generate_embedding_key(self, text: str, model: str) -> str:
        """Generate key for embedding"""
        # Normalize text
        normalized_text = text.strip().lower()

        key_data = {
            "text": normalized_text,
            "model": model
        }

        key_string = json.dumps(key_data, sort_keys=True)
        return hashlib.md5(key_string.encode()).hexdigest()


# Global cache instance
_cache_instance: Optional[CacheManager] = None
_cache_lock = threading.Lock()


def get_cache() -> CacheManager:
    """Get global cache instance (thread-safe)"""
    global _cache_instance

    if _cache_instance is None:
        with _cache_lock:
            if _cache_instance is None:
                cache_size = getattr(config, 'cache_size', 1000)
                default_ttl = getattr(config, 'cache_ttl', 3600)
                _cache_instance = CacheManager(max_size=cache_size, default_ttl=default_ttl)

    return _cache_instance


def get_api_cache() -> APICache:
    """Get API cache instance"""
    return APICache(get_cache())


def get_query_cache() -> QueryResultCache:
    """Get query result cache instance"""
    return QueryResultCache(get_cache())


def get_embedding_cache() -> EmbeddingCache:
    """Get embedding cache instance"""
    return EmbeddingCache(get_cache())


def get_cache_stats() -> Dict[str, Any]:
    """Get comprehensive cache statistics"""
    cache = get_cache()
    base_stats = cache.get_stats()

    # Add specialized cache stats
    return {
        **base_stats,
        "file_cache_dir": str(cache._file_cache_dir),
        "file_cache_exists": cache._file_cache_dir.exists(),
        "specialized_caches": {
            "api": "APICache",
            "query": "QueryResultCache",
            "embedding": "EmbeddingCache"
        }
    }


def clear_all_caches():
    """Clear all caches"""
    cache = get_cache()
    cache.clear()
    logger.info("All caches cleared")


def optimize_cache_settings():
    """Optimize cache settings based on current usage patterns"""
    cache = get_cache()
    stats = cache.get_stats()

    # Adjust cache size based on hit rate
    if stats["hit_rate"] > 0.8 and stats["size"] > stats["max_size"] * 0.9:
        logger.info("High hit rate detected, consider increasing cache size")
    elif stats["hit_rate"] < 0.3:
        logger.info("Low hit rate detected, consider reducing TTL or cache size")

    # Log recommendations
    logger.info(f"Cache optimization recommendations based on current stats: {stats}")