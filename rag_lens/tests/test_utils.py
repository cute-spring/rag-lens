"""
Utilities Tests for RAG Lens

Test suite for utility modules (logger, errors, security).
"""

import pytest
import logging
import tempfile
import os
import json
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta
import secrets

from rag_lens.utils.logger import get_logger, setup_logging, JSONFormatter, ColoredFormatter, log_performance
from rag_lens.utils.errors import (
    RAGLensError, ConfigurationError, TestCaseManagerError, PipelineError,
    APIError, SecurityError, ValidationError, PerformanceError,
    ErrorHandler, RetryHandler, CircuitBreaker
)
from rag_lens.utils.security import SecurityManager


class TestLogger:
    """Test cases for logging utilities"""

    def setup_method(self):
        """Setup test environment"""
        # Reset logging configuration
        logging.getLogger().handlers = []

    def test_get_logger(self):
        """Test logger creation"""
        logger = get_logger("test_logger")
        assert isinstance(logger, logging.Logger)
        assert logger.name == "test_logger"

    def test_json_formatter(self):
        """Test JSON formatter"""
        formatter = JSONFormatter()

        # Create a log record
        record = logging.LogRecord(
            name="test_logger",
            level=logging.INFO,
            pathname="/test/path.py",
            lineno=42,
            msg="Test message",
            args=(),
            exc_info=None
        )

        formatted = formatter.format(record)
        parsed = json.loads(formatted)

        assert parsed["level"] == "INFO"
        assert parsed["logger"] == "test_logger"
        assert parsed["message"] == "Test message"
        assert parsed["line"] == 42
        assert "timestamp" in parsed

    def test_json_formatter_with_exception(self):
        """Test JSON formatter with exception"""
        formatter = JSONFormatter()

        # Create a log record with exception
        try:
            raise ValueError("Test exception")
        except ValueError:
            record = logging.LogRecord(
                name="test_logger",
                level=logging.ERROR,
                pathname="/test/path.py",
                lineno=42,
                msg="Error message",
                args=(),
                exc_info=True
            )
            record.exc_info = (ValueError, ValueError("Test exception"), None)

        formatted = formatter.format(record)
        parsed = json.loads(formatted)

        assert "exception" in parsed
        assert "Test exception" in parsed["exception"]

    def test_json_formatter_with_extra_fields(self):
        """Test JSON formatter with extra fields"""
        formatter = JSONFormatter()

        record = logging.LogRecord(
            name="test_logger",
            level=logging.INFO,
            pathname="/test/path.py",
            lineno=42,
            msg="Test message",
            args=(),
            exc_info=None
        )
        record.extra = {"user_id": "123", "action": "test"}

        formatted = formatter.format(record)
        parsed = json.loads(formatted)

        assert parsed["user_id"] == "123"
        assert parsed["action"] == "test"

    def test_colored_formatter(self):
        """Test colored formatter"""
        formatter = ColoredFormatter('%(levelname)s - %(message)s')

        record = logging.LogRecord(
            name="test_logger",
            level=logging.INFO,
            pathname="/test/path.py",
            lineno=42,
            msg="Test message",
            args=(),
            exc_info=None
        )

        formatted = formatter.format(record)
        assert '\033[32m' in formatted  # Green color for INFO
        assert 'Test message' in formatted
        assert '\033[0m' in formatted  # Reset color

    def test_colored_formatter_colors(self):
        """Test colored formatter for different levels"""
        formatter = ColoredFormatter('%(levelname)s - %(message)s')

        levels_colors = [
            (logging.DEBUG, '\033[36m'),    # Cyan
            (logging.INFO, '\033[32m'),     # Green
            (logging.WARNING, '\033[33m'),  # Yellow
            (logging.ERROR, '\033[31m'),    # Red
            (logging.CRITICAL, '\033[35m'), # Magenta
        ]

        for level, color in levels_colors:
            record = logging.LogRecord(
                name="test_logger",
                level=level,
                pathname="/test/path.py",
                lineno=42,
                msg="Test message",
                args=(),
                exc_info=None
            )

            formatted = formatter.format(record)
            assert color in formatted

    def test_setup_logging(self):
        """Test logging setup"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Change to temp directory
            original_cwd = os.getcwd()
            os.chdir(temp_dir)

            try:
                setup_logging()

                # Check that logs directory was created
                assert os.path.exists("logs")

                # Check that root logger has handlers
                root_logger = logging.getLogger()
                assert len(root_logger.handlers) >= 2  # Console and file handlers

                # Check that file handlers exist
                log_files = [f for f in os.listdir("logs") if f.startswith("rag_lens_")]
                assert len(log_files) >= 1

            finally:
                os.chdir(original_cwd)

    def test_log_performance_decorator(self):
        """Test performance logging decorator"""
        @log_performance(threshold_seconds=0.1)
        def slow_function():
            import time
            time.sleep(0.2)
            return "result"

        # Mock logger to capture calls
        with patch('rag_lens.utils.logger.get_logger') as mock_get_logger:
            mock_logger = Mock()
            mock_get_logger.return_value = mock_logger

            result = slow_function()

            assert result == "result"
            # Should log a warning since execution time > threshold
            mock_logger.warning.assert_called_once()

    def test_log_performance_decorator_fast(self):
        """Test performance logging decorator with fast function"""
        @log_performance(threshold_seconds=1.0)
        def fast_function():
            return "quick result"

        # Mock logger to capture calls
        with patch('rag_lens.utils.logger.get_logger') as mock_get_logger:
            mock_logger = Mock()
            mock_get_logger.return_value = mock_logger

            result = fast_function()

            assert result == "quick result"
            # Should log debug, not warning, since execution time < threshold
            mock_logger.debug.assert_called_once()
            mock_logger.warning.assert_not_called()

    def test_log_performance_decorator_with_exception(self):
        """Test performance logging decorator with exception"""
        @log_performance(threshold_seconds=0.1)
        def error_function():
            import time
            time.sleep(0.2)
            raise ValueError("Test error")

        # Mock logger to capture calls
        with patch('rag_lens.utils.logger.get_logger') as mock_get_logger:
            mock_logger = Mock()
            mock_get_logger.return_value = mock_logger

            with pytest.raises(ValueError):
                error_function()

            # Should log error
            mock_logger.error.assert_called_once()


class TestErrors:
    """Test cases for error handling utilities"""

    def test_custom_exceptions_inheritance(self):
        """Test custom exception inheritance"""
        assert issubclass(ConfigurationError, RAGLensError)
        assert issubclass(TestCaseManagerError, RAGLensError)
        assert issubclass(PipelineError, RAGLensError)
        assert issubclass(APIError, RAGLensError)
        assert issubclass(SecurityError, RAGLensError)
        assert issubclass(ValidationError, RAGLensError)
        assert issubclass(PerformanceError, RAGLensError)

    def test_custom_exceptions_creation(self):
        """Test custom exception creation"""
        exc = ConfigurationError("Configuration error")
        assert str(exc) == "Configuration error"
        assert isinstance(exc, RAGLensError)

        exc = APIError("API error", error_code="API_TIMEOUT")
        assert str(exc) == "API error"
        assert isinstance(exc, RAGLensError)

    def test_error_handler_handle_error(self):
        """Test error handler"""
        error = ValueError("Test error")
        context = {"operation": "test", "user_id": "123"}

        error_info = ErrorHandler.handle_error(error, context)

        assert error_info["error_type"] == "ValueError"
        assert error_info["error_message"] == "Test error"
        assert error_info["context"] == context
        assert "timestamp" in error_info
        assert "traceback" in error_info

    def test_error_handler_should_retry(self):
        """Test retry logic"""
        # Test retryable errors
        assert ErrorHandler.should_retry(APIError("API timeout"))
        assert ErrorHandler.should_retry(ConnectionError("Connection failed"))
        assert ErrorHandler.should_retry(TimeoutError("Timeout"))
        assert ErrorHandler.should_retry(OSError("OS error"))

        # Test non-retryable errors
        assert not ErrorHandler.should_retry(ValidationError("Invalid data"))
        assert not ErrorHandler.should_retry(SecurityError("Security violation"))
        assert not ErrorHandler.should_retry(ValueError("Invalid value"))

    def test_error_handler_get_user_friendly_message(self):
        """Test user-friendly error messages"""
        test_cases = [
            (ConfigurationError("Config error"), "Configuration error. Please check your settings."),
            (TestCaseManagerError("Test case error"), "Unable to manage test cases. Please try again."),
            (PipelineError("Pipeline error"), "Pipeline processing failed. Please check your input."),
            (APIError("API error"), "API integration error. Please check your connection."),
            (SecurityError("Security error"), "Security validation failed. Please check your credentials."),
            (ValidationError("Validation error"), "Invalid data provided. Please check your input."),
            (PerformanceError("Performance error"), "Performance issue detected. Please try again later."),
            (FileNotFoundError("File not found"), "Required file not found."),
            (PermissionError("Permission denied"), "Permission denied."),
            (ValueError("Invalid value"), "Invalid value provided."),
            (KeyError("Key not found"), "Required key not found."),
            (RuntimeError("Generic error"), "An unexpected error occurred. Please try again."),
        ]

        for error, expected_message in test_cases:
            message = ErrorHandler.get_user_friendly_message(error)
            assert message == expected_message, f"Failed for {error.__class__.__name__}"

    def test_retry_handler_success(self):
        """Test successful retry"""
        mock_func = Mock(return_value="success")
        retry_handler = RetryHandler(max_retries=3, delay=0.1)

        result = retry_handler.retry(mock_func, "arg1", kwarg1="kwarg1")

        assert result == "success"
        mock_func.assert_called_once_with("arg1", kwarg1="kwarg1")

    def test_retry_handler_failure_then_success(self):
        """Test retry with initial failure then success"""
        mock_func = Mock(side_effect=[APIError("First failure"), "success"])
        retry_handler = RetryHandler(max_retries=3, delay=0.01)

        result = retry_handler.retry(mock_func)

        assert result == "success"
        assert mock_func.call_count == 2

    def test_retry_handler_all_failures(self):
        """Test retry with all failures"""
        mock_func = Mock(side_effect=APIError("Persistent failure"))
        retry_handler = RetryHandler(max_retries=2, delay=0.01)

        with pytest.raises(APIError):
            retry_handler.retry(mock_func)

        assert mock_func.call_count == 3  # Initial + 2 retries

    def test_retry_handler_non_retryable_error(self):
        """Test retry with non-retryable error"""
        mock_func = Mock(side_effect=ValidationError("Invalid data"))
        retry_handler = RetryHandler(max_retries=3, delay=0.01)

        with pytest.raises(ValidationError):
            retry_handler.retry(mock_func)

        mock_func.assert_called_once()  # Should not retry

    def test_circuit_breaker_initial_state(self):
        """Test circuit breaker initial state"""
        breaker = CircuitBreaker(failure_threshold=3, recovery_timeout=60)
        assert breaker.state == "CLOSED"
        assert breaker.failure_count == 0

    def test_circuit_breaker_success(self):
        """Test circuit breaker with successful call"""
        breaker = CircuitBreaker(failure_threshold=3, recovery_timeout=60)
        mock_func = Mock(return_value="success")

        result = breaker.call(mock_func)

        assert result == "success"
        assert breaker.state == "CLOSED"
        assert breaker.failure_count == 0

    def test_circuit_breaker_failure_under_threshold(self):
        """Test circuit breaker with failures under threshold"""
        breaker = CircuitBreaker(failure_threshold=3, recovery_timeout=60)
        mock_func = Mock(side_effect=APIError("API error"))

        with pytest.raises(APIError):
            breaker.call(mock_func)

        assert breaker.state == "CLOSED"
        assert breaker.failure_count == 1

    def test_circuit_breaker_failure_over_threshold(self):
        """Test circuit breaker opening after threshold exceeded"""
        breaker = CircuitBreaker(failure_threshold=2, recovery_timeout=60)
        mock_func = Mock(side_effect=APIError("API error"))

        # First failure
        with pytest.raises(APIError):
            breaker.call(mock_func)
        assert breaker.failure_count == 1
        assert breaker.state == "CLOSED"

        # Second failure - should open circuit
        with pytest.raises(PipelineError):
            breaker.call(mock_func)
        assert breaker.failure_count == 2
        assert breaker.state == "OPEN"

        # Third call - should fail immediately
        with pytest.raises(PipelineError):
            breaker.call(mock_func)
        mock_func.assert_called_count = 2  # Should not attempt third call

    def test_circuit_breaker_recovery(self):
        """Test circuit breaker recovery"""
        breaker = CircuitBreaker(failure_threshold=2, recovery_timeout=0.1)
        mock_func = Mock(side_effect=[APIError("API error"), APIError("API error"), "success"])

        # Cause circuit to open
        with pytest.raises(APIError):
            breaker.call(mock_func)
        with pytest.raises(PipelineError):
            breaker.call(mock_func)

        assert breaker.state == "OPEN"

        # Wait for recovery timeout
        import time
        time.sleep(0.2)

        # Next call should succeed and reset circuit
        result = breaker.call(mock_func)

        assert result == "success"
        assert breaker.state == "CLOSED"
        assert breaker.failure_count == 0


class TestSecurity:
    """Test cases for security utilities"""

    def setup_method(self):
        """Setup test environment"""
        self.security_manager = SecurityManager()

    def test_security_manager_init(self):
        """Test security manager initialization"""
        assert self.security_manager.secret_key is not None
        assert isinstance(self.security_manager.session_tokens, dict)
        assert len(self.security_manager.secret_key) > 0

    def test_generate_api_key(self):
        """Test API key generation"""
        api_key = self.security_manager.generate_api_key()
        assert isinstance(api_key, str)
        assert api_key.startswith("ragl_")
        assert len(api_key) > 10  # Should be reasonably long

    def test_hash_and_validate_api_key(self):
        """Test API key hashing and validation"""
        api_key = self.security_manager.generate_api_key()
        stored_hash = self.security_manager._hash_api_key(api_key)

        # Should validate correctly
        assert self.security_manager.validate_api_key(api_key, stored_hash)

        # Should reject invalid key
        assert not self.security_manager.validate_api_key("invalid_key", stored_hash)

    def test_generate_and_validate_session_token(self):
        """Test session token generation and validation"""
        user_id = "test_user"
        token = self.security_manager.generate_session_token(user_id)

        # Token should be valid immediately
        assert self.security_manager.validate_session_token(token)

        # Should reject invalid token
        assert not self.security_manager.validate_session_token("invalid_token")

    def test_session_token_expiry(self):
        """Test session token expiry"""
        # Create token with very short expiry
        user_id = "test_user"
        token = self.security_manager.generate_session_token(user_id, expiry_hours=0)

        # Should be valid initially
        assert self.security_manager.validate_session_token(token)

        # Manually set expiry time in the past
        if token in self.security_manager.session_tokens:
            self.security_manager.session_tokens[token]["expires_at"] = (
                datetime.utcnow() - timedelta(hours=1)
            ).isoformat()

        # Should now be invalid
        assert not self.security_manager.validate_session_token(token)

    def test_sanitize_input(self):
        """Test input sanitization"""
        # Normal input
        clean_input = self.security_manager.sanitize_input("Normal text")
        assert clean_input == "Normal text"

        # Input with null bytes
        dirty_input = "Text with\x00null bytes"
        clean_input = self.security_manager.sanitize_input(dirty_input)
        assert "\x00" not in clean_input

        # Input with whitespace
        whitespace_input = "  spaced text  "
        clean_input = self.security_manager.sanitize_input(whitespace_input)
        assert clean_input == "spaced text"

    def test_sanitize_input_too_long(self):
        """Test input sanitization with too long input"""
        long_input = "x" * 10001  # 10KB + 1 byte
        with pytest.raises(SecurityError):
            self.security_manager.sanitize_input(long_input)

    def test_sanitize_input_invalid_type(self):
        """Test input sanitization with invalid type"""
        with pytest.raises(SecurityError):
            self.security_manager.sanitize_input(123)  # Not a string

    def test_validate_file_type(self):
        """Test file type validation"""
        # Valid extensions
        assert self.security_manager.validate_file_type("test.pdf", [".pdf", ".doc"])
        assert self.security_manager.validate_file_type("document.txt", [".txt"])

        # Invalid extensions
        assert not self.security_manager.validate_file_type("test.exe", [".pdf", ".doc"])
        assert not self.security_manager.validate_file_type("test.pdf", [".txt"])

        # Empty filename
        assert not self.security_manager.validate_file_type("", [".pdf"])

    def test_generate_and_validate_csrf_token(self):
        """Test CSRF token generation and validation"""
        token = self.security_manager.generate_csrf_token()

        # Basic validation (simple implementation)
        assert self.security_manager.validate_csrf_token(token, "session_token")

        # Invalid tokens
        assert not self.security_manager.validate_csrf_token("short", "session_token")
        assert not self.security_manager.validate_csrf_token("invalid_token_with_spaces", "session_token")

    def test_password_strength_check(self):
        """Test password strength checking"""
        # Strong password
        strong_result = self.security_manager.check_password_strength("StrongPass123!")
        assert strong_result["strong"] is True
        assert strong_result["score"] >= 4

        # Weak password (too short)
        weak_result = self.security_manager.check_password_strength("weak")
        assert weak_result["strong"] is False
        assert "at least 8 characters" in weak_result["reason"]

        # Medium password
        medium_result = self.security_manager.check_password_strength("medium123")
        assert medium_result["strong"] is False
        assert medium_result["score"] < 4

    def test_generate_secure_password(self):
        """Test secure password generation"""
        password = self.security_manager.generate_secure_password(length=12)

        assert len(password) == 12
        assert any(c.isupper() for c in password)  # Has uppercase
        assert any(c.islower() for c in password)  # Has lowercase
        assert any(c.isdigit() for c in password)  # Has digit
        assert any(c in "!@#$%^&*()_+-=[]{}|;:,.<>?" for c in password)  # Has special char

    def test_generate_secure_password_too_short(self):
        """Test secure password generation with too short length"""
        with pytest.raises(SecurityError):
            self.security_manager.generate_secure_password(length=4)

    def test_sanitize_html(self):
        """Test HTML sanitization"""
        # Safe HTML
        safe_html = "<p>Safe content</p><strong>Bold text</strong>"
        clean_html = self.security_manager.sanitize_html(safe_html)
        assert "<p>" in clean_html
        assert "<strong>" in clean_html

        # Dangerous HTML
        dangerous_html = "<script>alert('xss')</script><p>Content</p>"
        clean_html = self.security_manager.sanitize_html(dangerous_html)
        assert "<script>" not in clean_html
        assert "<p>" in clean_html

    def test_validate_email(self):
        """Test email validation"""
        # Valid emails
        assert self.security_manager.validate_email("test@example.com")
        assert self.security_manager.validate_email("user.name+tag@domain.co.uk")

        # Invalid emails
        assert not self.security_manager.validate_email("invalid-email")
        assert not self.security_manager.validate_email("@domain.com")
        assert not self.security_manager.validate_email("user@")
        assert not self.security_manager.validate_email("user.domain.com")

    def test_rate_limit_check(self):
        """Test rate limiting"""
        identifier = "test_user"

        # Should allow initial requests
        assert self.security_manager.rate_limit_check(identifier, max_requests=5, window_minutes=1)

        # Fill up the limit
        for _ in range(4):  # 1 + 4 = 5 total
            assert self.security_manager.rate_limit_check(identifier, max_requests=5, window_minutes=1)

        # Next request should be denied
        assert not self.security_manager.rate_limit_check(identifier, max_requests=5, window_minutes=1)

    def test_rate_limit_window_expiry(self):
        """Test rate limit window expiry"""
        identifier = "test_user"

        # Set a very short window (1 second) and low limit
        assert self.security_manager.rate_limit_check(identifier, max_requests=1, window_minutes=1/60)

        # Second request should be denied
        assert not self.security_manager.rate_limit_check(identifier, max_requests=1, window_minutes=1/60)

        # Wait for window to expire
        import time
        time.sleep(1.1)

        # Should allow request again
        assert self.security_manager.rate_limit_check(identifier, max_requests=1, window_minutes=1/60)


if __name__ == "__main__":
    pytest.main([__file__])