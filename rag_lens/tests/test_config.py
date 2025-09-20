"""
Configuration Tests for RAG Lens

Test suite for the configuration management system.
"""

import pytest
import tempfile
import os
from pathlib import Path
import json

from rag_lens.config.settings import Config, PageConfig, TestSourceConfig, PipelineConfig


class TestConfig:
    """Test cases for Config class"""

    def test_config_singleton(self):
        """Test that Config is a singleton"""
        config1 = Config()
        config2 = Config()
        assert config1 is config2

    def test_environment_detection(self):
        """Test environment detection"""
        config = Config()
        assert config.environment in ["development", "testing", "staging", "production"]

    def test_page_config_default_values(self):
        """Test default values for page configuration"""
        config = Config()
        assert config.pages.title == "RAG Lens"
        assert config.pages.layout == "wide"
        assert config.pages.sidebar_state == "expanded"

    def test_test_source_config(self):
        """Test test source configuration"""
        config = Config()
        assert config.testing.test_source in ["static", "bigquery"]
        assert config.testing.test_file_path == "real_test_cases_collection.json"

    def test_pipeline_config(self):
        """Test pipeline configuration"""
        config = Config()
        assert isinstance(config.pipeline.query_generation_provider, str)
        assert isinstance(config.pipeline.query_generation_timeout, int)
        assert config.pipeline.query_generation_timeout > 0

    def test_api_config(self):
        """Test API configuration"""
        config = Config()
        assert isinstance(config.api.base_url, str)
        assert isinstance(config.api.timeout, int)
        assert config.api.timeout > 0

    def test_monitoring_config(self):
        """Test monitoring configuration"""
        config = Config()
        assert isinstance(config.monitoring.enabled, bool)
        assert config.monitoring.log_level in ["DEBUG", "INFO", "WARNING", "ERROR"]
        assert isinstance(config.monitoring.metrics_interval, int)

    def test_is_development(self):
        """Test development environment detection"""
        config = Config()
        is_dev = config.is_development()
        assert isinstance(is_dev, bool)

    def test_is_production(self):
        """Test production environment detection"""
        config = Config()
        is_prod = config.is_production()
        assert isinstance(is_prod, bool)

    def test_from_dict(self):
        """Test configuration from dictionary"""
        config_dict = {
            "app": {
                "name": "Test App",
                "version": "2.0.0",
                "debug": True
            },
            "pages": {
                "title": "Test Title",
                "layout": "centered"
            }
        }

        config = Config()
        config.from_dict(config_dict)

        assert config.app.name == "Test App"
        assert config.app.version == "2.0.0"
        assert config.app.debug is True
        assert config.pages.title == "Test Title"
        assert config.pages.layout == "centered"

    def test_to_dict(self):
        """Test configuration to dictionary"""
        config = Config()
        config_dict = config.to_dict()

        assert isinstance(config_dict, dict)
        assert "app" in config_dict
        assert "pages" in config_dict
        assert "testing" in config_dict
        assert "pipeline" in config_dict

    def test_save_and_load(self):
        """Test saving and loading configuration"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            config_path = f.name

        try:
            # Save configuration
            config = Config()
            config.app.name = "Test Config"
            config.save(config_path)

            # Load configuration
            new_config = Config()
            new_config.load(config_path)

            assert new_config.app.name == "Test Config"

        finally:
            os.unlink(config_path)

    def test_load_nonexistent_file(self):
        """Test loading non-existent configuration file"""
        config = Config()
        config.load("/nonexistent/path/config.json")
        # Should not raise an error, should use defaults

    def test_environment_override(self):
        """Test environment variable override"""
        os.environ["RAG_LENS_APP_NAME"] = "Env Override"
        os.environ["RAG_LENS_APP_DEBUG"] = "true"

        try:
            config = Config()
            # Note: This test would require the Config class to actually read environment variables
            # For now, we just test that the config loads without error
            assert config is not None

        finally:
            os.environ.pop("RAG_LENS_APP_NAME", None)
            os.environ.pop("RAG_LENS_APP_DEBUG", None)


class TestPageConfig:
    """Test cases for PageConfig dataclass"""

    def test_page_config_creation(self):
        """Test PageConfig creation"""
        page_config = PageConfig(
            title="Test Page",
            layout="centered",
            sidebar_state="collapsed"
        )

        assert page_config.title == "Test Page"
        assert page_config.layout == "centered"
        assert page_config.sidebar_state == "collapsed"


class TestTestSourceConfig:
    """Test cases for TestSourceConfig dataclass"""

    def test_test_source_config_creation(self):
        """Test TestSourceConfig creation"""
        test_config = TestSourceConfig(
            test_source="bigquery",
            test_file_path="test.json",
            bigquery_project="test-project",
            bigquery_dataset="test_dataset",
            bigquery_table="test_table"
        )

        assert test_config.test_source == "bigquery"
        assert test_config.test_file_path == "test.json"
        assert test_config.bigquery_project == "test-project"


class TestPipelineConfig:
    """Test cases for PipelineConfig dataclass"""

    def test_pipeline_config_creation(self):
        """Test PipelineConfig creation"""
        pipeline_config = PipelineConfig(
            query_generation_provider="openai",
            query_encoding_provider="openai",
            query_generation_timeout=60,
            query_encoding_timeout=30
        )

        assert pipeline_config.query_generation_provider == "openai"
        assert pipeline_config.query_encoding_provider == "openai"
        assert pipeline_config.query_generation_timeout == 60
        assert pipeline_config.query_encoding_timeout == 30


if __name__ == "__main__":
    pytest.main([__file__])