"""
Utility modules for RAG Lens
"""

from .logger import get_logger, setup_logging, log_performance
from .errors import (
    RAGLensError, ErrorHandler, RetryHandler, CircuitBreaker,
    ConfigurationError, TestCaseManagerError, PipelineError,
    APIError, SecurityError, ValidationError, PerformanceError, CacheError
)
from .security import security_manager
from .cache import (
    get_cache, get_api_cache, get_query_cache, get_embedding_cache,
    cached, cached_async, get_cache_stats, clear_all_caches
)
from .performance import (
    get_performance_monitor, get_response_optimizer, get_connection_pool,
    get_memory_optimizer, get_performance_dashboard_data,
    performance_monitor, timed_cache, rate_limit
)

__all__ = [
    # Logging
    'get_logger', 'setup_logging', 'log_performance',
    # Error handling
    'RAGLensError', 'ErrorHandler', 'RetryHandler', 'CircuitBreaker',
    'ConfigurationError', 'TestCaseManagerError', 'PipelineError',
    'APIError', 'SecurityError', 'ValidationError', 'PerformanceError', 'CacheError',
    # Security
    'security_manager',
    # Caching
    'get_cache', 'get_api_cache', 'get_query_cache', 'get_embedding_cache',
    'cached', 'cached_async', 'get_cache_stats', 'clear_all_caches',
    # Performance
    'get_performance_monitor', 'get_response_optimizer', 'get_connection_pool',
    'get_memory_optimizer', 'get_performance_dashboard_data',
    'performance_monitor', 'timed_cache', 'rate_limit'
]