"""
Performance Optimization Utilities for RAG Lens

This module provides performance monitoring and optimization utilities
to improve application responsiveness and resource usage.
"""

import time
import psutil
import threading
from typing import Dict, List, Any, Optional, Callable, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from contextlib import contextmanager
import functools

from .logger import get_logger
from .cache import get_cache, cached
from .errors import PerformanceError
from ..config.settings import config

logger = get_logger(__name__)


@dataclass
class PerformanceMetrics:
    """Performance metrics data"""
    operation: str
    duration: float
    memory_usage: float
    cpu_usage: float
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "operation": self.operation,
            "duration": self.duration,
            "memory_usage": self.memory_usage,
            "cpu_usage": self.cpu_usage,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata
        }


class PerformanceMonitor:
    """Monitor application performance metrics"""

    def __init__(self, max_metrics: int = 1000):
        self.max_metrics = max_metrics
        self.metrics: List[PerformanceMetrics] = []
        self._lock = threading.Lock()
        self._process = psutil.Process()

    def record_metric(
        self,
        operation: str,
        duration: float,
        memory_usage: Optional[float] = None,
        cpu_usage: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Record a performance metric"""
        with self._lock:
            # Get current resource usage if not provided
            if memory_usage is None:
                memory_usage = self._get_memory_usage()
            if cpu_usage is None:
                cpu_usage = self._get_cpu_usage()

            metric = PerformanceMetrics(
                operation=operation,
                duration=duration,
                memory_usage=memory_usage,
                cpu_usage=cpu_usage,
                metadata=metadata or {}
            )

            self.metrics.append(metric)

            # Maintain max size
            if len(self.metrics) > self.max_metrics:
                self.metrics = self.metrics[-self.max_metrics:]

    def get_metrics(
        self,
        operation: Optional[str] = None,
        since: Optional[datetime] = None,
        limit: Optional[int] = None
    ) -> List[PerformanceMetrics]:
        """Get metrics with optional filtering"""
        with self._lock:
            filtered_metrics = self.metrics

            if operation:
                filtered_metrics = [m for m in filtered_metrics if m.operation == operation]

            if since:
                filtered_metrics = [m for m in filtered_metrics if m.timestamp >= since]

            if limit:
                filtered_metrics = filtered_metrics[-limit:]

            return filtered_metrics

    def get_average_metrics(self, operation: str) -> Dict[str, float]:
        """Get average metrics for an operation"""
        metrics = self.get_metrics(operation=operation)
        if not metrics:
            return {}

        return {
            "average_duration": sum(m.duration for m in metrics) / len(metrics),
            "average_memory_usage": sum(m.memory_usage for m in metrics) / len(metrics),
            "average_cpu_usage": sum(m.cpu_usage for m in metrics) / len(metrics),
            "total_calls": len(metrics),
            "min_duration": min(m.duration for m in metrics),
            "max_duration": max(m.duration for m in metrics)
        }

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get overall performance summary"""
        with self._lock:
            if not self.metrics:
                return {"message": "No metrics recorded yet"}

            total_operations = len(self.metrics)
            operations = list(set(m.operation for m in self.metrics))

            summary = {
                "total_operations": total_operations,
                "unique_operations": len(operations),
                "operations_breakdown": {}
            }

            for operation in operations:
                summary["operations_breakdown"][operation] = self.get_average_metrics(operation)

            return summary

    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB"""
        try:
            return self._process.memory_info().rss / (1024 * 1024)
        except Exception:
            return 0.0

    def _get_cpu_usage(self) -> float:
        """Get current CPU usage percentage"""
        try:
            return self._process.cpu_percent(interval=0.1)
        except Exception:
            return 0.0


class ResponseTimeOptimizer:
    """Optimize response times through various strategies"""

    def __init__(self):
        self.cache = get_cache()
        self.monitor = PerformanceMonitor()

    @contextmanager
    def measure_performance(self, operation: str, metadata: Optional[Dict[str, Any]] = None):
        """Context manager for measuring performance"""
        start_time = time.time()
        start_memory = self._get_memory_usage()
        start_cpu = self._get_cpu_usage()

        try:
            yield
        finally:
            duration = time.time() - start_time
            end_memory = self._get_memory_usage()
            end_cpu = self._get_cpu_usage()

            self.monitor.record_metric(
                operation=operation,
                duration=duration,
                memory_usage=end_memory - start_memory,
                cpu_usage=end_cpu - start_cpu,
                metadata=metadata
            )

    def optimize_function_call(
        self,
        func: Callable,
        cache_key: Optional[str] = None,
        cache_ttl: Optional[int] = None,
        **kwargs
    ) -> Any:
        """Optimize function call with caching and performance monitoring"""
        operation_name = f"{func.__module__}.{func.__name__}"

        with self.measure_performance(operation_name, {"cached": cache_key is not None}):
            if cache_key:
                cached_result = self.cache.get(cache_key)
                if cached_result is not None:
                    logger.debug(f"Cache hit for {operation_name}")
                    return cached_result

            # Call function
            result = func(**kwargs)

            # Cache result if key provided
            if cache_key:
                self.cache.set(cache_key, result, cache_ttl)

            return result

    def batch_process(
        self,
        items: List[Any],
        process_func: Callable,
        batch_size: int = 10,
        parallel: bool = False
    ) -> List[Any]:
        """Process items in batches for better performance"""
        operation_name = f"{process_func.__module__}.{process_func.__name__}"

        with self.measure_performance(
            operation_name,
            {"items_count": len(items), "batch_size": batch_size, "parallel": parallel}
        ):
            if parallel and len(items) > batch_size:
                return self._parallel_batch_process(items, process_func, batch_size)
            else:
                return self._sequential_batch_process(items, process_func, batch_size)

    def _sequential_batch_process(self, items: List[Any], process_func: Callable, batch_size: int) -> List[Any]:
        """Process items sequentially in batches"""
        results = []
        for i in range(0, len(items), batch_size):
            batch = items[i:i + batch_size]
            batch_results = [process_func(item) for item in batch]
            results.extend(batch_results)
        return results

    def _parallel_batch_process(self, items: List[Any], process_func: Callable, batch_size: int) -> List[Any]:
        """Process items in parallel using threads"""
        import concurrent.futures

        results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            # Submit batches
            futures = []
            for i in range(0, len(items), batch_size):
                batch = items[i:i + batch_size]
                future = executor.submit(self._sequential_batch_process, batch, process_func, len(batch))
                futures.append(future)

            # Collect results
            for future in concurrent.futures.as_completed(futures):
                batch_results = future.result()
                results.extend(batch_results)

        return results


class ConnectionPool:
    """Manage connection pools for better performance"""

    def __init__(self, max_connections: int = 10):
        self.max_connections = max_connections
        self.pools: Dict[str, List[Any]] = {}
        self.locks: Dict[str, threading.Lock] = {}

    def get_pool_key(self, connection_params: Dict[str, Any]) -> str:
        """Generate pool key from connection parameters"""
        import hashlib
        key_string = json.dumps(connection_params, sort_keys=True)
        return hashlib.md5(key_string.encode()).hexdigest()

    def get_connection(self, connection_params: Dict[str, Any], create_func: Callable) -> Any:
        """Get connection from pool"""
        pool_key = self.get_pool_key(connection_params)

        if pool_key not in self.pools:
            self.pools[pool_key] = []
            self.locks[pool_key] = threading.Lock()

        with self.locks[pool_key]:
            if self.pools[pool_key]:
                return self.pools[pool_key].pop()

            # Create new connection if pool is empty
            if len(self.pools[pool_key]) < self.max_connections:
                return create_func(connection_params)
            else:
                raise PerformanceError("Maximum connections reached")

    def return_connection(self, connection_params: Dict[str, Any], connection: Any):
        """Return connection to pool"""
        pool_key = self.get_pool_key(connection_params)

        if pool_key in self.pools:
            with self.locks[pool_key]:
                if len(self.pools[pool_key]) < self.max_connections:
                    self.pools[pool_key].append(connection)

    def clear_pool(self, connection_params: Dict[str, Any]):
        """Clear connection pool"""
        pool_key = self.get_pool_key(connection_params)

        if pool_key in self.pools:
            with self.locks[pool_key]:
                self.pools[pool_key].clear()


def performance_monitor(operation: Optional[str] = None):
    """Decorator for monitoring function performance"""
    def decorator(func: Callable):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            monitor = get_performance_monitor()
            op_name = operation or f"{func.__module__}.{func.__name__}"

            with monitor.measure_performance(op_name):
                return func(*args, **kwargs)

        return wrapper
    return decorator


def timed_cache(ttl: int = 3600):
    """Decorator that combines timing and caching"""
    def decorator(func: Callable):
        cached_func = cached(ttl=ttl)(func)
        monitored_func = performance_monitor()(cached_func)
        return monitored_func
    return decorator


def rate_limit(calls_per_second: float):
    """Decorator for rate limiting function calls"""
    def decorator(func: Callable):
        last_called = [0.0]
        min_interval = 1.0 / calls_per_second

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            elapsed = time.time() - last_called[0]
            left_to_wait = min_interval - elapsed

            if left_to_wait > 0:
                time.sleep(left_to_wait)

            ret = func(*args, **kwargs)
            last_called[0] = time.time()
            return ret

        return wrapper
    return decorator


def async_rate_limit(calls_per_second: float):
    """Decorator for rate limiting async function calls"""
    def decorator(func: Callable):
        last_called = [0.0]
        min_interval = 1.0 / calls_per_second

        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            elapsed = time.time() - last_called[0]
            left_to_wait = min_interval - elapsed

            if left_to_wait > 0:
                await asyncio.sleep(left_to_wait)

            ret = await func(*args, **kwargs)
            last_called[0] = time.time()
            return ret

        return wrapper
    return decorator


class MemoryOptimizer:
    """Optimize memory usage"""

    def __init__(self):
        self.monitor = PerformanceMonitor()

    def optimize_large_data_structure(self, data_structure: Any) -> Any:
        """Optimize large data structures for memory usage"""
        if isinstance(data_structure, dict):
            return self._optimize_dict(data_structure)
        elif isinstance(data_structure, list):
            return self._optimize_list(data_structure)
        else:
            return data_structure

    def _optimize_dict(self, data_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize dictionary memory usage"""
        optimized = {}
        for key, value in data_dict.items():
            # Use __slots__ for small objects
            if isinstance(value, dict) and len(value) < 10:
                optimized[key] = self._dict_to_obj(value)
            else:
                optimized[key] = value
        return optimized

    def _optimize_list(self, data_list: List[Any]) -> List[Any]:
        """Optimize list memory usage"""
        # Convert to tuple if list is small and not modified often
        if len(data_list) < 100:
            return tuple(data_list)
        return data_list

    def _dict_to_obj(self, data_dict: Dict[str, Any]) -> Any:
        """Convert dictionary to object with __slots__"""
        class OptimizedObj:
            __slots__ = list(data_dict.keys())

            def __init__(self, **kwargs):
                for key, value in kwargs.items():
                    setattr(self, key, value)

        return OptimizedObj(**data_dict)

    def cleanup_unused_memory(self):
        """Force garbage collection and cleanup"""
        import gc
        collected = gc.collect()
        logger.info(f"Garbage collection cleaned up {collected} objects")


# Global instances
_performance_monitor: Optional[PerformanceMonitor] = None
_response_optimizer: Optional[ResponseTimeOptimizer] = None
_connection_pool: Optional[ConnectionPool] = None
_memory_optimizer: Optional[MemoryOptimizer] = None


def get_performance_monitor() -> PerformanceMonitor:
    """Get global performance monitor instance"""
    global _performance_monitor
    if _performance_monitor is None:
        _performance_monitor = PerformanceMonitor()
    return _performance_monitor


def get_response_optimizer() -> ResponseTimeOptimizer:
    """Get global response optimizer instance"""
    global _response_optimizer
    if _response_optimizer is None:
        _response_optimizer = ResponseTimeOptimizer()
    return _response_optimizer


def get_connection_pool() -> ConnectionPool:
    """Get global connection pool instance"""
    global _connection_pool
    if _connection_pool is None:
        _connection_pool = ConnectionPool()
    return _connection_pool


def get_memory_optimizer() -> MemoryOptimizer:
    """Get global memory optimizer instance"""
    global _memory_optimizer
    if _memory_optimizer is None:
        _memory_optimizer = MemoryOptimizer()
    return _memory_optimizer


def get_performance_dashboard_data() -> Dict[str, Any]:
    """Get comprehensive performance dashboard data"""
    monitor = get_performance_monitor()
    cache_stats = get_cache_stats()

    return {
        "performance_summary": monitor.get_performance_summary(),
        "cache_stats": cache_stats,
        "system_resources": {
            "memory_usage_mb": monitor._get_memory_usage(),
            "cpu_usage_percent": monitor._get_cpu_usage(),
            "uptime_seconds": time.time() - monitor.metrics[0].timestamp.timestamp() if monitor.metrics else 0
        }
    }


def optimize_application_performance():
    """Run comprehensive performance optimization"""
    optimizer = get_response_optimizer()
    memory_optimizer = get_memory_optimizer()

    # Optimize cache settings
    optimize_cache_settings()

    # Clean up memory
    memory_optimizer.cleanup_unused_memory()

    # Log performance summary
    summary = get_performance_monitor().get_performance_summary()
    logger.info(f"Performance optimization completed. Summary: {summary}")

    return summary