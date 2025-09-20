"""
Structured logging utilities for RAG Lens
"""

import logging
import sys
from datetime import datetime
from typing import Any, Dict, Optional
from pathlib import Path
import json

from ..config.settings import config


class JSONFormatter(logging.Formatter):
    """JSON formatter for structured logging"""

    def format(self, record):
        log_entry = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno
        }

        # Add exception info if present
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)

        # Add extra fields
        if hasattr(record, 'extra') and record.extra:
            log_entry.update(record.extra)

        return json.dumps(log_entry)


class ColoredFormatter(logging.Formatter):
    """Colored formatter for console output"""

    COLORS = {
        'DEBUG': '\033[36m',    # Cyan
        'INFO': '\033[32m',     # Green
        'WARNING': '\033[33m',  # Yellow
        'ERROR': '\033[31m',    # Red
        'CRITICAL': '\033[35m', # Magenta
    }
    RESET = '\033[0m'

    def format(self, record):
        log_color = self.COLORS.get(record.levelname, self.RESET)
        message = super().format(record)
        return f"{log_color}{message}{self.RESET}"


def setup_logging():
    """Setup logging configuration"""
    # Create logs directory if it doesn't exist
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, config.monitoring.log_level))

    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    if config.is_development():
        console_handler.setFormatter(ColoredFormatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        ))
    else:
        console_handler.setFormatter(JSONFormatter())
    root_logger.addHandler(console_handler)

    # File handler
    file_handler = logging.FileHandler(
        log_dir / f"rag_lens_{datetime.now().strftime('%Y%m%d')}.log"
    )
    file_handler.setFormatter(JSONFormatter())
    root_logger.addHandler(file_handler)

    # Error file handler
    error_handler = logging.FileHandler(
        log_dir / f"rag_lens_errors_{datetime.now().strftime('%Y%m%d')}.log"
    )
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(JSONFormatter())
    root_logger.addHandler(error_handler)


def get_logger(name: str) -> logging.Logger:
    """Get logger with specified name"""
    return logging.getLogger(name)


class LoggerMixin:
    """Mixin class to add logging capability to other classes"""

    @property
    def logger(self) -> logging.Logger:
        """Get logger for this class"""
        return get_logger(self.__class__.__name__)


def log_function_call(func):
    """Decorator to log function calls"""
    def wrapper(*args, **kwargs):
        logger = get_logger(func.__module__)
        logger.debug(f"Calling {func.__name__} with args={args}, kwargs={kwargs}")
        try:
            result = func(*args, **kwargs)
            logger.debug(f"Function {func.__name__} returned successfully")
            return result
        except Exception as e:
            logger.error(f"Function {func.__name__} failed with error: {e}")
            raise
    return wrapper


def log_performance(threshold_seconds: float = 1.0):
    """Decorator to log function performance"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            logger = get_logger(func.__module__)
            import time
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                end_time = time.time()
                duration = end_time - start_time
                if duration > threshold_seconds:
                    logger.warning(
                        f"Function {func.__name__} took {duration:.2f}s (threshold: {threshold_seconds}s)",
                        extra={"performance": {"duration": duration, "threshold": threshold_seconds}}
                    )
                else:
                    logger.debug(
                        f"Function {func.__name__} completed in {duration:.2f}s"
                    )
                return result
            except Exception as e:
                end_time = time.time()
                duration = end_time - start_time
                logger.error(
                    f"Function {func.__name__} failed after {duration:.2f}s: {e}",
                    extra={"performance": {"duration": duration, "error": str(e)}}
                )
                raise
        return wrapper
    return decorator


# Initialize logging
setup_logging()