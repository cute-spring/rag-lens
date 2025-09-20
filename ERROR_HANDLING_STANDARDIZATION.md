# Standardized Error Handling for API Integration

This document defines a comprehensive error handling system that ensures consistency across all API integrations and provides clear, actionable feedback for developers.

## 1. Error Classification System

### 1.1 Error Categories

```python
class ErrorCategory(Enum):
    """High-level error categories"""
    AUTHENTICATION = "authentication"
    AUTHORIZATION = "authorization"
    VALIDATION = "validation"
    NETWORK = "network"
    TIMEOUT = "timeout"
    RATE_LIMIT = "rate_limit"
    SERVICE_UNAVAILABLE = "service_unavailable"
    PARSE_ERROR = "parse_error"
    BUSINESS_LOGIC = "business_logic"
    UNKNOWN = "unknown"

class ErrorSeverity(Enum):
    """Error severity levels"""
    LOW = "low"           # Non-critical, can continue
    MEDIUM = "medium"     # May impact functionality
    HIGH = "high"         # Critical, cannot continue
    CRITICAL = "critical"  # System failure
```

### 1.2 Standard Error Codes

```python
class ErrorCode:
    """Standardized error codes for all integrations"""

    # Authentication Errors (AUTH_XXX)
    AUTH_MISSING_KEY = "AUTH_001"
    AUTH_INVALID_KEY = "AUTH_002"
    AUTH_EXPIRED_KEY = "AUTH_003"
    AUTH_INSUFFICIENT_PERMISSIONS = "AUTH_004"

    # Authorization Errors (AUTHZ_XXX)
    AUTHZ_ACCESS_DENIED = "AUTHZ_001"
    AUTHZ_QUOTA_EXCEEDED = "AUTHZ_002"
    AUTHZ_FEATURE_DISABLED = "AUTHZ_003"

    # Validation Errors (VAL_XXX)
    VAL_MISSING_REQUIRED_FIELD = "VAL_001"
    VAL_INVALID_FORMAT = "VAL_002"
    VAL_VALUE_OUT_OF_RANGE = "VAL_003"
    VAL_INVALID_ENUM = "VAL_004"

    # Network Errors (NET_XXX)
    NET_CONNECTION_FAILED = "NET_001"
    NET_DNS_RESOLUTION_FAILED = "NET_002"
    NET_SSL_ERROR = "NET_003"
    NET_PROXY_ERROR = "NET_004"

    # Timeout Errors (TIMEOUT_XXX)
    TIMEOUT_REQUEST = "TIMEOUT_001"
    TIMEOUT_CONNECT = "TIMEOUT_002"
    TIMEOUT_READ = "TIMEOUT_003"
    TIMEOUT_WRITE = "TIMEOUT_004"

    # Rate Limit Errors (RATE_XXX)
    RATE_LIMIT_EXCEEDED = "RATE_001"
    RATE_LIMIT_RETRY_AFTER = "RATE_002"
    RATE_LIMIT_DAILY_CAP = "RATE_003"
    RATE_LIMIT_CONCURRENT = "RATE_004"

    # Service Errors (SVC_XXX)
    SVC_UNAVAILABLE = "SVC_001"
    SVC_MAINTENANCE = "SVC_002"
    SVC_OVERLOADED = "SVC_003"
    SVC_DEGRADED = "SVC_004"

    # Parse Errors (PARSE_XXX)
    PARSE_INVALID_JSON = "PARSE_001"
    PARSE_INVALID_XML = "PARSE_002"
    PARSE_RESPONSE_FORMAT = "PARSE_003"
    PARSE_ENCODING_ERROR = "PARSE_004"

    # Business Logic Errors (BIZ_XXX)
    BIZ_INVALID_STATE = "BIZ_001"
    BIZ_CONFLICT = "BIZ_002"
    BIZ_RESOURCE_NOT_FOUND = "BIZ_003"
    BIZ_OPERATION_FAILED = "BIZ_004"
```

## 2. Standard Error Response Format

### 2.1 API Error Response Structure

```python
{
    "success": false,
    "error": {
        "code": "RATE_LIMIT_EXCEEDED",        # Standard error code
        "category": "rate_limit",            # Error category
        "severity": "medium",                  # Error severity
        "message": "API rate limit exceeded",  # Human-readable message
        "detailed_message": "Exceeded rate limit of 60 requests per minute",  # Detailed explanation
        "retry_after": 60,                    # Seconds to wait before retry (if applicable)
        "suggestions": [                      # Specific suggestions for fixing
            "Wait 60 seconds before retrying",
            "Implement request batching",
            "Use exponential backoff"
        ],
        "context": {                          # Additional error context
            "provider": "openai",
            "endpoint": "/v1/chat/completions",
            "method": "POST",
            "timestamp": "2024-01-01T12:00:00Z",
            "request_id": "req_123456789"
        },
        "stack_trace": "Limited stack trace for debugging (in development mode)"
    }
}
```

### 2.2 Error Response Builder

```python
import traceback
from datetime import datetime
from typing import Dict, Any, List, Optional
import json

class ErrorResponseBuilder:
    """Build standardized error responses"""

    @staticmethod
    def build_error(
        error_code: str,
        message: str,
        category: str = "unknown",
        severity: str = "medium",
        detailed_message: str = None,
        retry_after: int = None,
        suggestions: List[str] = None,
        context: Dict[str, Any] = None,
        include_stack_trace: bool = False
    ) -> Dict[str, Any]:
        """Build a standardized error response"""

        # Get default suggestions based on error code
        if suggestions is None:
            suggestions = ErrorResponseBuilder._get_default_suggestions(error_code)

        # Build context if not provided
        if context is None:
            context = {
                "timestamp": datetime.now().isoformat(),
                "error_code": error_code
            }

        error_response = {
            "success": False,
            "error": {
                "code": error_code,
                "category": category,
                "severity": severity,
                "message": message,
                "detailed_message": detailed_message or message,
                "suggestions": suggestions,
                "context": context
            }
        }

        # Add optional fields
        if retry_after is not None:
            error_response["error"]["retry_after"] = retry_after

        if include_stack_trace:
            error_response["error"]["stack_trace"] = traceback.format_exc()

        return error_response

    @staticmethod
    def _get_default_suggestions(error_code: str) -> List[str]:
        """Get default suggestions based on error code"""
        suggestion_map = {
            # Authentication errors
            "AUTH_MISSING_KEY": [
                "Check if API key is configured in environment variables",
                "Verify your .env file exists and is properly formatted",
                "Ensure the API key is not empty or None"
            ],
            "AUTH_INVALID_KEY": [
                "Verify your API key is correct",
                "Check for typos or extra spaces in the API key",
                "Regenerate your API key if necessary"
            ],

            # Rate limit errors
            "RATE_LIMIT_EXCEEDED": [
                "Wait the specified time before retrying",
                "Implement exponential backoff",
                "Consider upgrading your API plan",
                "Use request batching to reduce API calls"
            ],

            # Network errors
            "NET_CONNECTION_FAILED": [
                "Check your internet connection",
                "Verify the API endpoint is accessible",
                "Check firewall settings",
                "Try again with exponential backoff"
            ],

            # Timeout errors
            "TIMEOUT_REQUEST": [
                "Increase timeout settings",
                "Optimize your request size",
                "Check API service status",
                "Consider using smaller batches"
            ],

            # Validation errors
            "VAL_MISSING_REQUIRED_FIELD": [
                "Check all required fields are included in your request",
                "Refer to the API documentation for required fields",
                "Validate your request before sending"
            ],

            # Service errors
            "SVC_UNAVAILABLE": [
                "Check API service status page",
                "Wait for service to be restored",
                "Implement retry logic with exponential backoff",
                "Consider using fallback provider"
            ]
        }

        return suggestion_map.get(error_code, [
            "Check the error message for details",
            "Refer to the API documentation",
            "Contact support if the issue persists"
        ])

    @staticmethod
    def from_exception(exception: Exception, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Convert exception to standardized error response"""

        # Determine error code and category based on exception type
        error_code, category, severity = ErrorResponseBuilder._classify_exception(exception)

        # Build error message
        message = str(exception)
        detailed_message = ErrorResponseBuilder._get_detailed_message(exception)

        # Extract retry_after if present
        retry_after = ErrorResponseBuilder._extract_retry_after(exception)

        # Build context
        error_context = {
            "exception_type": type(exception).__name__,
            "timestamp": datetime.now().isoformat()
        }
        if context:
            error_context.update(context)

        return ErrorResponseBuilder.build_error(
            error_code=error_code,
            message=message,
            category=category,
            severity=severity,
            detailed_message=detailed_message,
            retry_after=retry_after,
            context=error_context,
            include_stack_trace=True  # Include stack trace for exceptions
        )

    @staticmethod
    def _classify_exception(exception: Exception) -> tuple:
        """Classify exception to determine error code, category, and severity"""

        exception_type = type(exception).__name__

        if exception_type in ["ConnectionError", "ConnectTimeout"]:
            return "NET_CONNECTION_FAILED", "network", "high"
        elif exception_type in ["ReadTimeout", "WriteTimeout"]:
            return "TIMEOUT_REQUEST", "timeout", "medium"
        elif exception_type in ["SSLError", "ProxyError"]:
            return "NET_SSL_ERROR", "network", "high"
        elif "401" in str(exception) or "Unauthorized" in str(exception):
            return "AUTH_INVALID_KEY", "authentication", "high"
        elif "403" in str(exception) or "Forbidden" in str(exception):
            return "AUTHZ_ACCESS_DENIED", "authorization", "high"
        elif "429" in str(exception) or "TooManyRequests" in str(exception):
            return "RATE_LIMIT_EXCEEDED", "rate_limit", "medium"
        elif "404" in str(exception) or "NotFound" in str(exception):
            return "BIZ_RESOURCE_NOT_FOUND", "business_logic", "medium"
        elif "5" in str(exception)[:1]:  # 5xx server errors
            return "SVC_UNAVAILABLE", "service_unavailable", "high"
        elif "JSONDecodeError" in exception_type:
            return "PARSE_INVALID_JSON", "parse_error", "medium"
        else:
            return "UNKNOWN_ERROR", "unknown", "medium"

    @staticmethod
    def _get_detailed_message(exception: Exception) -> str:
        """Get detailed error message from exception"""
        try:
            if hasattr(exception, 'response'):
                # For HTTP errors, try to extract more details
                response = getattr(exception, 'response', None)
                if response and hasattr(response, 'json'):
                    try:
                        error_data = response.json()
                        return error_data.get('error', {}).get('message', str(exception))
                    except:
                        pass
        except:
            pass

        return str(exception)

    @staticmethod
    def _extract_retry_after(exception: Exception) -> Optional[int]:
        """Extract retry_after value from exception"""
        try:
            if hasattr(exception, 'response'):
                response = getattr(exception, 'response', None)
                if response:
                    # Check Retry-After header
                    retry_after = response.headers.get('Retry-After')
                    if retry_after:
                        try:
                            return int(retry_after)
                        except ValueError:
                            pass

                    # Check for retry_after in response body
                    try:
                        error_data = response.json()
                        retry_after = error_data.get('retry_after') or error_data.get('error', {}).get('retry_after')
                        if retry_after:
                            return int(retry_after)
                    except:
                        pass
        except:
            pass

        return None
```

## 3. Retry Mechanisms

### 3.1 Retry Strategy Classes

```python
import asyncio
import random
from abc import ABC, abstractmethod
from typing import Callable, Dict, Any, Optional

class RetryStrategy(ABC):
    """Base class for retry strategies"""

    @abstractmethod
    async def should_retry(self, attempt: int, error: Dict[str, Any]) -> bool:
        """Determine if request should be retried"""
        pass

    @abstractmethod
    def get_delay(self, attempt: int, error: Dict[str, Any]) -> float:
        """Get delay before next retry"""
        pass

class ExponentialBackoffStrategy(RetryStrategy):
    """Exponential backoff with jitter"""

    def __init__(self, base_delay: float = 1.0, max_delay: float = 60.0, jitter: bool = True):
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.jitter = jitter

    async def should_retry(self, attempt: int, error: Dict[str, Any]) -> bool:
        """Retry on network, timeout, and rate limit errors"""
        if attempt >= 5:  # Max retries
            return False

        error_category = error.get("error", {}).get("category")
        error_code = error.get("error", {}).get("code")

        # Don't retry on authentication or validation errors
        if error_category in ["authentication", "validation"]:
            return False

        # Retry on network, timeout, rate limit, and service errors
        if error_category in ["network", "timeout", "rate_limit", "service_unavailable"]:
            return True

        # Specific error codes that are worth retrying
        retryable_codes = [
            "RATE_LIMIT_EXCEEDED",
            "SVC_UNAVAILABLE",
            "SVC_OVERLOADED",
            "NET_CONNECTION_FAILED",
            "TIMEOUT_REQUEST"
        ]

        return error_code in retryable_codes

    def get_delay(self, attempt: int, error: Dict[str, Any]) -> float:
        """Calculate delay with exponential backoff and jitter"""
        # Exponential backoff: base_delay * 2^(attempt-1)
        delay = self.base_delay * (2 ** (attempt - 1))

        # Cap at maximum delay
        delay = min(delay, self.max_delay)

        # Add jitter to prevent thundering herd
        if self.jitter:
            delay = delay * (0.5 + random.random() * 0.5)

        # For rate limit errors, use the suggested retry_after if available
        retry_after = error.get("error", {}).get("retry_after")
        if retry_after:
            delay = max(delay, float(retry_after))

        return delay

class LinearBackoffStrategy(RetryStrategy):
    """Linear backoff strategy"""

    def __init__(self, base_delay: float = 2.0, max_delay: float = 30.0):
        self.base_delay = base_delay
        self.max_delay = max_delay

    async def should_retry(self, attempt: int, error: Dict[str, Any]) -> bool:
        """Retry up to 3 times on retryable errors"""
        return attempt < 3 and error.get("error", {}).get("category") in ["network", "timeout"]

    def get_delay(self, attempt: int, error: Dict[str, Any]) -> float:
        """Calculate linear delay"""
        delay = self.base_delay * attempt
        return min(delay, self.max_delay)
```

### 3.2 Retry Handler

```python
class RetryHandler:
    """Handle retry logic with configurable strategies"""

    def __init__(self, strategy: RetryStrategy = None):
        self.strategy = strategy or ExponentialBackoffStrategy()
        self.retry_logger = RetryLogger()

    async def execute_with_retry(
        self,
        func: Callable,
        *args,
        max_retries: int = 3,
        **kwargs
    ) -> Dict[str, Any]:
        """Execute function with retry logic"""

        last_error = None

        for attempt in range(1, max_retries + 1):
            try:
                result = await func(*args, **kwargs)

                # Check if result contains an error
                if isinstance(result, dict) and not result.get("success", True):
                    # This is an application-level error
                    error_response = result

                    # Check if we should retry
                    if await self.strategy.should_retry(attempt, error_response):
                        delay = self.strategy.get_delay(attempt, error_response)
                        await self.retry_logger.log_retry_attempt(
                            func.__name__, attempt, error_response, delay
                        )
                        await asyncio.sleep(delay)
                        continue
                    else:
                        # Don't retry this error
                        return error_response

                # Success
                if attempt > 1:
                    await self.retry_logger.log_retry_success(func.__name__, attempt)
                return result

            except Exception as e:
                # Convert exception to error response
                error_response = ErrorResponseBuilder.from_exception(e)

                # Check if we should retry
                if await self.strategy.should_retry(attempt, error_response):
                    delay = self.strategy.get_delay(attempt, error_response)
                    await self.retry_logger.log_retry_attempt(
                        func.__name__, attempt, error_response, delay
                    )
                    await asyncio.sleep(delay)
                    last_error = error_response
                    continue
                else:
                    # Don't retry this exception
                    return error_response

        # All retries failed
        await self.retry_logger.log_retry_failure(func.__name__, max_retries, last_error)
        return last_error or ErrorResponseBuilder.build_error(
            error_code="UNKNOWN_ERROR",
            message=f"Failed after {max_retries} retries",
            severity="high"
        )

class RetryLogger:
    """Log retry attempts for debugging and monitoring"""

    async def log_retry_attempt(self, function_name: str, attempt: int, error: Dict[str, Any], delay: float):
        """Log retry attempt"""
        print(f"[RETRY] {function_name} - Attempt {attempt}/5 failed. "
              f"Retrying in {delay:.2f}s. Error: {error['error']['code']} - {error['error']['message']}")

    async def log_retry_success(self, function_name: str, attempt: int):
        """Log successful retry"""
        print(f"[RETRY] {function_name} - Succeeded on attempt {attempt}")

    async def log_retry_failure(self, function_name: str, max_retries: int, error: Dict[str, Any]):
        """Log retry failure"""
        print(f"[RETRY] {function_name} - Failed after {max_retries} attempts. "
              f"Final error: {error['error']['code']} - {error['error']['message']}")
```

## 4. Circuit Breaker Pattern

### 4.1 Circuit Breaker Implementation

```python
import time
from enum import Enum
from typing import Callable, Dict, Any, Optional
from collections import defaultdict

class CircuitState(Enum):
    """Circuit breaker states"""
    CLOSED = "closed"      # Normal operation, requests flow through
    OPEN = "open"         # Circuit is open, requests fail fast
    HALF_OPEN = "half_open"  # Testing if service has recovered

class CircuitBreaker:
    """Circuit breaker to prevent cascade failures"""

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: int = 60,
        expected_exception: tuple = (Exception,)
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception

        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.last_failure_time = None
        self.success_count = 0

        # Metrics
        self.metrics = defaultdict(int)

    async def execute(self, func: Callable, *args, **kwargs) -> Dict[str, Any]:
        """Execute function with circuit breaker protection"""

        if self.state == CircuitState.OPEN:
            if self._should_attempt_reset():
                self.state = CircuitState.HALF_OPEN
                self.metrics["half_open_attempts"] += 1
            else:
                self.metrics["open_circuit_rejections"] += 1
                return ErrorResponseBuilder.build_error(
                    error_code="CIRCUIT_OPEN",
                    message="Service temporarily unavailable due to failures",
                    category="service_unavailable",
                    severity="high",
                    suggestions=[
                        "Wait for service to recover",
                        "Check service status",
                        "Implement fallback mechanisms"
                    ],
                    context={
                        "circuit_state": "open",
                        "failure_count": self.failure_count,
                        "cooldown_remaining": self._get_cooldown_remaining()
                    }
                )

        try:
            result = await func(*args, **kwargs)

            # Check for application-level errors
            if isinstance(result, dict) and not result.get("success", True):
                self._handle_failure(result.get("error", {}))
            else:
                self._handle_success()

            return result

        except self.expected_exception as e:
            error_response = ErrorResponseBuilder.from_exception(e)
            self._handle_failure(error_response.get("error", {}))
            return error_response

    def _should_attempt_reset(self) -> bool:
        """Check if circuit breaker should attempt reset"""
        if self.last_failure_time is None:
            return True

        return time.time() - self.last_failure_time >= self.recovery_timeout

    def _handle_failure(self, error: Dict[str, Any]):
        """Handle a failure"""
        self.failure_count += 1
        self.last_failure_time = time.time()
        self.metrics["failures"] += 1

        error_category = error.get("category")

        # Only open circuit for certain types of errors
        if error_category in ["network", "timeout", "service_unavailable"]:
            if self.failure_count >= self.failure_threshold:
                self.state = CircuitState.OPEN
                self.metrics["circuit_opened"] += 1

    def _handle_success(self):
        """Handle a success"""
        self.failure_count = 0
        self.metrics["successes"] += 1

        if self.state == CircuitState.HALF_OPEN:
            self.success_count += 1
            if self.success_count >= 3:  # Need 3 consecutive successes to close
                self.state = CircuitState.CLOSED
                self.success_count = 0
                self.metrics["circuit_closed"] += 1

    def _get_cooldown_remaining(self) -> int:
        """Get remaining cooldown time in seconds"""
        if self.last_failure_time is None:
            return 0

        elapsed = time.time() - self.last_failure_time
        return max(0, self.recovery_timeout - elapsed)

    def get_state(self) -> Dict[str, Any]:
        """Get current circuit breaker state and metrics"""
        return {
            "state": self.state.value,
            "failure_count": self.failure_count,
            "last_failure_time": self.last_failure_time,
            "cooldown_remaining": self._get_cooldown_remaining(),
            "metrics": dict(self.metrics)
        }
```

## 5. Error Handling Middleware

### 5.1 API Client with Error Handling

```python
class APIClientWithErrorHandling:
    """API client with comprehensive error handling"""

    def __init__(self, base_url: str, api_key: str = None, timeout: int = 30):
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.timeout = aiohttp.ClientTimeout(total=timeout)

        # Initialize error handling components
        self.retry_handler = RetryHandler(ExponentialBackoffStrategy())
        self.circuit_breaker = CircuitBreaker(failure_threshold=5, recovery_timeout=60)

        # Prepare headers
        self.headers = {"Content-Type": "application/json"}
        if api_key:
            self.headers["Authorization"] = f"Bearer {api_key}"

    async def request(self, method: str, endpoint: str, data: Dict[str, Any] = None,
                    params: Dict[str, Any] = None, enable_retry: bool = True) -> Dict[str, Any]:
        """Make HTTP request with comprehensive error handling"""

        url = f"{self.base_url}/{endpoint.lstrip('/')}"

        if enable_retry:
            return await self.retry_handler.execute_with_retry(
                self._make_request,
                method, url, data, params
            )
        else:
            return await self.circuit_breaker.execute(
                self._make_request,
                method, url, data, params
            )

    async def _make_request(self, method: str, url: str, data: Dict[str, Any] = None,
                           params: Dict[str, Any] = None) -> Dict[str, Any]:
        """Make actual HTTP request"""

        context = {
            "method": method,
            "url": url,
            "endpoint": url.split('/')[-1],
            "timestamp": datetime.now().isoformat()
        }

        try:
            async with aiohttp.ClientSession(headers=self.headers, timeout=self.timeout) as session:
                if method.upper() == "GET":
                    async with session.get(url, params=params) as response:
                        return await self._handle_response(response, context)
                elif method.upper() == "POST":
                    async with session.post(url, json=data, params=params) as response:
                        return await self._handle_response(response, context)
                elif method.upper() == "PUT":
                    async with session.put(url, json=data, params=params) as response:
                        return await self._handle_response(response, context)
                elif method.upper() == "DELETE":
                    async with session.delete(url, params=params) as response:
                        return await self._handle_response(response, context)
                else:
                    return ErrorResponseBuilder.build_error(
                        error_code="VAL_INVALID_METHOD",
                        message=f"Unsupported HTTP method: {method}",
                        context=context
                    )

        except Exception as e:
            return ErrorResponseBuilder.from_exception(e, context)

    async def _handle_response(self, response: aiohttp.ClientResponse, context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle HTTP response"""

        # Add response info to context
        context.update({
            "status_code": response.status,
            "response_headers": dict(response.headers)
        })

        try:
            if response.status == 200:
                response_data = await response.json()
                return {"success": True, "data": response_data, "context": context}

            elif response.status == 201:
                response_data = await response.json()
                return {"success": True, "data": response_data, "created": True, "context": context}

            elif response.status == 400:
                error_data = await response.json()
                return ErrorResponseBuilder.build_error(
                    error_code="VAL_INVALID_REQUEST",
                    message="Invalid request parameters",
                    detailed_message=error_data.get("error", {}).get("message", "Bad request"),
                    context=context
                )

            elif response.status == 401:
                return ErrorResponseBuilder.build_error(
                    error_code="AUTH_UNAUTHORIZED",
                    message="Authentication required",
                    category="authentication",
                    severity="high",
                    context=context
                )

            elif response.status == 403:
                return ErrorResponseBuilder.build_error(
                    error_code="AUTHZ_FORBIDDEN",
                    message="Access denied",
                    category="authorization",
                    severity="high",
                    context=context
                )

            elif response.status == 404:
                return ErrorResponseBuilder.build_error(
                    error_code="BIZ_NOT_FOUND",
                    message="Resource not found",
                    category="business_logic",
                    severity="medium",
                    context=context
                )

            elif response.status == 429:
                # Extract retry_after from headers
                retry_after = None
                if "Retry-After" in response.headers:
                    try:
                        retry_after = int(response.headers["Retry-After"])
                    except ValueError:
                        pass

                return ErrorResponseBuilder.build_error(
                    error_code="RATE_LIMIT_EXCEEDED",
                    message="API rate limit exceeded",
                    category="rate_limit",
                    severity="medium",
                    retry_after=retry_after,
                    context=context
                )

            elif 500 <= response.status < 600:
                return ErrorResponseBuilder.build_error(
                    error_code="SVC_SERVER_ERROR",
                    message=f"Server error: {response.status}",
                    category="service_unavailable",
                    severity="high",
                    context=context
                )

            else:
                return ErrorResponseBuilder.build_error(
                    error_code="UNKNOWN_HTTP_ERROR",
                    message=f"Unexpected HTTP status: {response.status}",
                    category="unknown",
                    severity="medium",
                    context=context
                )

        except Exception as e:
            return ErrorResponseBuilder.from_exception(e, context)
```

## 6. Error Aggregation and Monitoring

### 6.1 Error Aggregator

```python
class ErrorAggregator:
    """Aggregate and analyze errors across API calls"""

    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self.errors = []
        self.error_counts = defaultdict(int)
        self.error_by_category = defaultdict(list)
        self.error_by_provider = defaultdict(list)

    def record_error(self, error_response: Dict[str, Any], provider: str = "unknown"):
        """Record an error for analysis"""

        # Add timestamp
        error_record = {
            "timestamp": datetime.now().isoformat(),
            "provider": provider,
            "error": error_response
        }

        # Maintain error list size
        self.errors.append(error_record)
        if len(self.errors) > self.max_size:
            self.errors.pop(0)

        # Update counts
        error_code = error_response.get("error", {}).get("code", "UNKNOWN")
        self.error_counts[error_code] += 1

        category = error_response.get("error", {}).get("category", "unknown")
        self.error_by_category[category].append(error_record)

        self.error_by_provider[provider].append(error_record)

    def get_error_summary(self, time_window_minutes: int = 60) -> Dict[str, Any]:
        """Get error summary for specified time window"""
        cutoff_time = datetime.now().timestamp() - (time_window_minutes * 60)

        recent_errors = [
            error for error in self.errors
            if datetime.fromisoformat(error["timestamp"]).timestamp() > cutoff_time
        ]

        summary = {
            "time_window_minutes": time_window_minutes,
            "total_errors": len(recent_errors),
            "unique_error_codes": len(set(error["error"]["error"]["code"] for error in recent_errors)),
            "top_errors": self._get_top_errors(recent_errors, 5),
            "errors_by_category": self._get_errors_by_category(recent_errors),
            "errors_by_provider": self._get_errors_by_provider(recent_errors),
            "error_rate_trend": self._calculate_error_rate_trend(recent_errors)
        }

        return summary

    def _get_top_errors(self, errors: List[Dict], limit: int) -> List[Dict[str, Any]]:
        """Get most frequent errors"""
        error_freq = defaultdict(int)
        for error in errors:
            error_code = error["error"]["error"]["code"]
            error_freq[error_code] += 1

        top_errors = sorted(error_freq.items(), key=lambda x: x[1], reverse=True)[:limit]

        return [
            {
                "error_code": error_code,
                "count": count,
                "percentage": (count / len(errors)) * 100 if errors else 0
            }
            for error_code, count in top_errors
        ]

    def _get_errors_by_category(self, errors: List[Dict]) -> Dict[str, Any]:
        """Get errors grouped by category"""
        category_counts = defaultdict(int)
        for error in errors:
            category = error["error"]["error"]["category"]
            category_counts[category] += 1

        return dict(category_counts)

    def _get_errors_by_provider(self, errors: List[Dict]) -> Dict[str, Any]:
        """Get errors grouped by provider"""
        provider_counts = defaultdict(int)
        for error in errors:
            provider = error["provider"]
            provider_counts[provider] += 1

        return dict(provider_counts)

    def _calculate_error_rate_trend(self, errors: List[Dict]) -> Dict[str, float]:
        """Calculate error rate trend over time"""
        if not errors:
            return {"trend": "stable", "change_percentage": 0.0}

        # Group errors by 10-minute intervals
        interval_counts = defaultdict(int)
        for error in errors:
            timestamp = datetime.fromisoformat(error["timestamp"])
            interval_key = timestamp.replace(minute=(timestamp.minute // 10) * 10, second=0, microsecond=0)
            interval_counts[interval_key] += 1

        if len(interval_counts) < 2:
            return {"trend": "insufficient_data", "change_percentage": 0.0}

        # Calculate trend
        sorted_intervals = sorted(interval_counts.items())
        recent_avg = sum(count for _, count in sorted_intervals[-2:]) / 2
        earlier_avg = sum(count for _, count in sorted_intervals[:-2]) / max(1, len(sorted_intervals) - 2)

        if earlier_avg == 0:
            change_percentage = 100.0 if recent_avg > 0 else 0.0
        else:
            change_percentage = ((recent_avg - earlier_avg) / earlier_avg) * 100

        if change_percentage > 20:
            trend = "increasing"
        elif change_percentage < -20:
            trend = "decreasing"
        else:
            trend = "stable"

        return {
            "trend": trend,
            "change_percentage": change_percentage,
            "recent_average": recent_avg,
            "earlier_average": earlier_avg
        }
```

## 7. Usage Examples

### 7.1 Basic Error Handling

```python
# Example 1: Basic API client with error handling
async def basic_error_handling_example():
    client = APIClientWithErrorHandling(
        base_url="https://api.openai.com/v1",
        api_key="your-api-key"
    )

    result = await client.request("POST", "chat/completions", {
        "model": "gpt-4",
        "messages": [{"role": "user", "content": "Hello"}]
    })

    if result["success"]:
        print("Success:", result["data"])
    else:
        error = result["error"]
        print(f"Error ({error['code']}): {error['message']}")
        print("Suggestions:")
        for suggestion in error["suggestions"]:
            print(f"  - {suggestion}")
```

### 7.2 Retry with Custom Strategy

```python
# Example 2: Custom retry strategy
async def custom_retry_example():
    class CustomRetryStrategy(RetryStrategy):
        async def should_retry(self, attempt: int, error: Dict[str, Any]) -> bool:
            return attempt < 3 and error["error"]["code"] == "RATE_LIMIT_EXCEEDED"

        def get_delay(self, attempt: int, error: Dict[str, Any]) -> float:
            return 30.0  # Fixed 30-second delay

    client = APIClientWithErrorHandling("https://api.example.com")
    client.retry_handler = RetryHandler(CustomRetryStrategy())

    result = await client.request("GET", "/data")
```

### 7.3 Error Monitoring

```python
# Example 3: Error monitoring
async def error_monitoring_example():
    aggregator = ErrorAggregator()

    # Simulate API calls with errors
    client = APIClientWithErrorHandling("https://api.example.com")

    for i in range(10):
        result = await client.request("GET", "/endpoint")
        if not result["success"]:
            aggregator.record_error(result, "example_api")

    # Get error summary
    summary = aggregator.get_error_summary(time_window_minutes=60)
    print(f"Total errors in last hour: {summary['total_errors']}")
    print(f"Top error: {summary['top_errors'][0] if summary['top_errors'] else 'None'}")
    print(f"Error trend: {summary['error_rate_trend']['trend']}")
```

This standardized error handling system provides:
- **Consistent error formats** across all API integrations
- **Actionable error messages** with specific suggestions
- **Robust retry mechanisms** with configurable strategies
- **Circuit breaker protection** to prevent cascade failures
- **Comprehensive error monitoring** and analytics
- **Clear error classification** and severity levels

By implementing this system, integrators will have much easier time debugging issues and implementing proper error handling in their applications.