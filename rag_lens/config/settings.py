"""
Centralized configuration management for RAG Lens
"""

import os
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from enum import Enum


class Environment(Enum):
    """Application environment types"""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"


@dataclass
class PageConfig:
    """Streamlit page configuration"""
    page_title: str = "RAG Lens - RAG Pipeline Testing & Performance Tuning Tool"
    page_icon: str = "🔍"
    layout: str = "wide"
    initial_sidebar_state: str = "expanded"


@dataclass
class TestSourceConfig:
    """Test case source configuration"""
    sources: Dict[str, str] = field(default_factory=lambda: {
        "Local Test Cases (Default)": "test_cases_local.json",
        "Real Test Cases Collection": "real_test_cases_collection.json",
        "Complete Enhanced Test Suite": "COMPLETE_TEST_SUITE.json",
        "Sample Reference": "sample_test_case_reference.json"
    })
    default_source: str = "Local Test Cases (Default)"


@dataclass
class PipelineConfig:
    """RAG pipeline configuration"""
    steps: List[str] = field(default_factory=lambda: [
        "Query Processing",
        "Retrieval",
        "Initial Filtering",
        "Re-ranking",
        "Final Selection",
        "Context Assembly",
        "Response Generation"
    ])

    # Default parameters
    default_semantic_weight: float = 0.5
    default_freshness_weight: float = 0.2
    default_quality_weight: float = 0.3
    default_relevance_threshold: float = 0.6
    default_top_n: int = 10


@dataclass
class APIConfig:
    """API integration configuration"""
    timeout: int = 30
    max_retries: int = 3
    retry_delay: float = 1.0

    # Supported authentication methods
    auth_methods: List[str] = field(default_factory=lambda: [
        "api_key", "oauth2", "jwt", "aws_signature", "azure_ad"
    ])


@dataclass
class MonitoringConfig:
    """Monitoring and logging configuration"""
    log_level: str = "INFO"
    enable_performance_tracking: bool = True
    health_check_interval: int = 30
    metrics_retention_days: int = 30


@dataclass
class SecurityConfig:
    """Security configuration"""
    encrypt_sensitive_data: bool = True
    session_timeout_minutes: int = 120
    max_file_size_mb: int = 10
    allowed_file_extensions: List[str] = field(default_factory=lambda: [
        ".json", ".csv", ".txt", ".md"
    ])


@dataclass
class DatabaseConfig:
    """Database configuration"""
    # BigQuery configuration
    enable_bigquery: bool = False
    bigquery_project_id: Optional[str] = None
    bigquery_dataset_id: Optional[str] = None

    # Local storage configuration
    local_data_dir: str = "data"
    backup_enabled: bool = True
    backup_retention_days: int = 7


@dataclass
class CacheConfig:
    """Cache configuration"""
    enabled: bool = True
    max_size: int = 1000
    default_ttl: int = 3600
    file_cache_enabled: bool = True
    memory_limit_mb: int = 512


@dataclass
class PerformanceConfig:
    """Performance optimization configuration"""
    enable_monitoring: bool = True
    enable_caching: bool = True
    enable_connection_pooling: bool = True
    max_concurrent_requests: int = 10
    request_timeout: int = 30
    cache_ttl: int = 3600


class Config:
    """Main configuration class"""

    def __init__(self):
        # Environment detection
        self.environment = self._detect_environment()

        # Core configurations
        self.page = PageConfig()
        self.test_sources = TestSourceConfig()
        self.pipeline = PipelineConfig()
        self.api = APIConfig()
        self.monitoring = MonitoringConfig()
        self.security = SecurityConfig()
        self.database = DatabaseConfig()
        self.cache = CacheConfig()
        self.performance = PerformanceConfig()

        # Load environment-specific settings
        self._load_environment_config()
        
        # Load API configuration from environment variables
        self._load_api_config()

    def _detect_environment(self) -> Environment:
        """Detect current environment"""
        env = os.getenv("RAG_LENS_ENV", "development").lower()
        try:
            return Environment(env)
        except ValueError:
            return Environment.DEVELOPMENT

    def _load_environment_config(self):
        """Load environment-specific configuration"""
        if self.environment == Environment.PRODUCTION:
            self.monitoring.log_level = "WARNING"
            self.security.encrypt_sensitive_data = True
            self.api.timeout = 60
        elif self.environment == Environment.STAGING:
            self.monitoring.log_level = "INFO"
            self.security.encrypt_sensitive_data = True
        else:  # DEVELOPMENT
            self.monitoring.log_level = "DEBUG"
            self.security.encrypt_sensitive_data = False
    
    def _load_api_config(self):
        """Load API configuration from environment variables."""
        import os
        
        # Ollama configuration
        self.ollama_base_url = os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434')
        self.ollama_model = os.getenv('OLLAMA_MODEL', 'llama2')
        self.ollama_api_key = os.getenv('OLLAMA_API_KEY', '')
        
        # Other API configurations can be added here as needed
        self.openai_api_key = os.getenv('OPENAI_API_KEY', '')
        self.azure_api_key = os.getenv('AZURE_API_KEY', '')
        self.elasticsearch_host = os.getenv('ELASTICSEARCH_HOST', 'localhost:9200')
        self.cross_encoder_model = os.getenv('CROSS_ENCODER_MODEL', 'cross-encoder/ms-marco-MiniLM-L-6-v2')

    def get_test_source_path(self, display_name: str) -> str:
        """Get file path for test source"""
        return self.test_sources.sources.get(display_name,
                                           self.test_sources.sources[self.test_sources.default_source])

    def is_production(self) -> bool:
        """Check if running in production"""
        return self.environment == Environment.PRODUCTION

    def is_development(self) -> bool:
        """Check if running in development"""
        return self.environment == Environment.DEVELOPMENT

    def get_cors_origins(self) -> List[str]:
        """Get allowed CORS origins based on environment"""
        if self.is_production():
            return ["https://yourdomain.com"]
        elif self.environment == Environment.STAGING:
            return ["https://staging.yourdomain.com"]
        else:
            return ["localhost", "127.0.0.1"]

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return {
            "environment": self.environment.value,
            "page": self.page.__dict__,
            "test_sources": self.test_sources.__dict__,
            "pipeline": self.pipeline.__dict__,
            "api": self.api.__dict__,
            "monitoring": self.monitoring.__dict__,
            "security": self.security.__dict__,
            "database": self.database.__dict__,
            "cache": self.cache.__dict__,
            "performance": self.performance.__dict__
        }


# Global configuration instance
config = Config()


def get_config() -> Config:
    """Get global configuration instance"""
    return config


def reload_config():
    """Reload configuration (useful for testing)"""
    global config
    config = Config()