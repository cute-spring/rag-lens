"""
RAG Lens - Enterprise-grade RAG Pipeline Testing Platform

A comprehensive platform for testing and evaluating Retrieval-Augmented Generation (RAG)
pipelines with modular architecture, standardized API integration, and comprehensive monitoring.

Author: RAG Lens Team
Version: 1.0.0
"""

__version__ = "1.0.0"
__author__ = "RAG Lens Team"
__email__ = "support@raglens.com"
__description__ = "Enterprise-grade RAG Pipeline Testing Platform"

# Core imports
from .config.settings import config
from .utils.logger import get_logger
from .utils.errors import RAGLensError, ErrorHandler
from .utils.security import security_manager
from .core.test_case_manager import TestCaseManager
from .core.pipeline_simulator import PipelineSimulator
from .api.providers import api_manager
from .api.integrations import PipelineOrchestrator, TestCaseIntegration, HealthMonitor

# Export main classes and functions
__all__ = [
    # Configuration
    'config',

    # Utilities
    'get_logger',
    'RAGLensError',
    'ErrorHandler',
    'security_manager',

    # Core components
    'TestCaseManager',
    'PipelineSimulator',

    # API components
    'api_manager',
    'PipelineOrchestrator',
    'TestCaseIntegration',
    'HealthMonitor'
]

# Initialize logging
from .utils.logger import setup_logging
setup_logging()

logger = get_logger(__name__)
logger.info(f"RAG Lens v{__version__} initialized")