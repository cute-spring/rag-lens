"""
Custom exceptions and error handling utilities for RAG Lens
"""

import traceback
from typing import Any, Dict, Optional
from datetime import datetime

from .logger import get_logger

logger = get_logger(__name__)


class RAGLensError(Exception):
    """Base exception for RAG Lens application"""
    pass


class ConfigurationError(RAGLensError):
    """Configuration related errors"""
    pass


class TestCaseManagerError(RAGLensError):
    """Test case management errors"""
    pass


class PipelineError(RAGLensError):
    """Pipeline processing errors"""
    pass


class APIError(RAGLensError):
    """API integration errors"""
    pass


class SecurityError(RAGLensError):
    """Security related errors"""
    pass


class ValidationError(RAGLensError):
    """Data validation errors"""
    pass


class PerformanceError(RAGLensError):
    """Performance related errors"""
    pass


class CacheError(RAGLensError):
    """Cache related errors"""
    pass


class ErrorHandler:
    """Centralized error handler"""

    @staticmethod
    def handle_error(error: Exception, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Handle error and return structured error response"""
        error_info = {
            "error_type": error.__class__.__name__,
            "error_message": str(error),
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "context": context or {},
            "traceback": traceback.format_exc() if logger.isEnabledFor(logging.DEBUG) else None
        }

        # Log the error
        logger.error(
            f"Error occurred: {error_info['error_type']} - {error_info['error_message']}",
            extra={"error": error_info}
        )

        return error_info

    @staticmethod
    def should_retry(error: Exception) -> bool:
        """Determine if an error should trigger a retry"""
        retryable_errors = (
            APIError,
            ConnectionError,
            TimeoutError,
            OSError
        )
        return isinstance(error, retryable_errors)

    @staticmethod
    def get_user_friendly_message(error: Exception) -> str:
        """Get user-friendly error message"""
        error_messages = {
            ConfigurationError: "Configuration error. Please check your settings.",
            TestCaseManagerError: "Unable to manage test cases. Please try again.",
            PipelineError: "Pipeline processing failed. Please check your input.",
            APIError: "API integration error. Please check your connection.",
            SecurityError: "Security validation failed. Please check your credentials.",
            ValidationError: "Invalid data provided. Please check your input.",
            PerformanceError: "Performance issue detected. Please try again later.",
            FileNotFoundError: "Required file not found.",
            PermissionError: "Permission denied.",
            ValueError: "Invalid value provided.",
            KeyError: "Required key not found.",
        }

        return error_messages.get(type(error), "An unexpected error occurred. Please try again.")


class RetryHandler:
    """Handle retry logic for operations"""

    def __init__(self, max_retries: int = 3, delay: float = 1.0, backoff_factor: float = 2.0):
        self.max_retries = max_retries
        self.delay = delay
        self.backoff_factor = backoff_factor

    def retry(self, func, *args, **kwargs):
        """Retry function with exponential backoff"""
        import time

        last_error = None
        for attempt in range(self.max_retries + 1):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                last_error = e
                if attempt < self.max_retries and ErrorHandler.should_retry(e):
                    delay = self.delay * (self.backoff_factor ** attempt)
                    logger.warning(
                        f"Attempt {attempt + 1} failed, retrying in {delay}s: {e}",
                        extra={"retry_attempt": attempt + 1, "max_retries": self.max_retries}
                    )
                    time.sleep(delay)
                else:
                    logger.error(
                        f"All {self.max_retries + 1} attempts failed: {e}",
                        extra={"retry_attempts": attempt + 1, "final_error": str(e)}
                    )
                    break

        raise last_error


class CircuitBreaker:
    """Circuit breaker for fault tolerance"""

    def __init__(self, failure_threshold: int = 5, recovery_timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN

    def call(self, func, *args, **kwargs):
        """Call function with circuit breaker protection"""
        if self.state == "OPEN":
            if self._should_attempt_reset():
                self.state = "HALF_OPEN"
                logger.info("Circuit breaker transitioning to HALF_OPEN")
            else:
                raise PipelineError("Circuit breaker is OPEN")

        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure()
            raise

    def _should_attempt_reset(self) -> bool:
        """Check if circuit breaker should attempt reset"""
        if self.last_failure_time is None:
            return True

        import time
        return (time.time() - self.last_failure_time) >= self.recovery_timeout

    def _on_success(self):
        """Handle successful call"""
        self.failure_count = 0
        if self.state == "HALF_OPEN":
            self.state = "CLOSED"
            logger.info("Circuit breaker reset to CLOSED")

    def _on_failure(self):
        """Handle failed call"""
        self.failure_count += 1
        self.last_failure_time = time.time()

        if self.failure_count >= self.failure_threshold:
            self.state = "OPEN"
            logger.error(
                f"Circuit breaker opened after {self.failure_count} failures",
                extra={"failure_count": self.failure_count}
            )