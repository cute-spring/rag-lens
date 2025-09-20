"""
Test Suite for RAG Lens

This package contains comprehensive unit tests for all RAG Lens modules.
"""

import pytest
import sys
from pathlib import Path

# Add the rag_lens directory to Python path for testing
test_dir = Path(__file__).parent
root_dir = test_dir.parent
sys.path.insert(0, str(root_dir))


def pytest_configure(config):
    """Configure pytest with custom settings"""
    # Add custom markers
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "unit: marks tests as unit tests"
    )
    config.addinivalue_line(
        "markers", "api: marks tests that require external API calls"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add default markers"""
    for item in items:
        # Mark tests that use external APIs
        if "api" in item.nodeid.lower() or "mock" not in item.nodeid.lower():
            if "test_api" in item.nodeid:
                item.add_marker(pytest.mark.api)
                item.add_marker(pytest.mark.slow)

        # Mark integration tests
        if "integration" in item.nodeid.lower():
            item.add_marker(pytest.mark.integration)

        # Mark unit tests (default)
        if not any(marker.name in ["integration", "api"] for marker in item.iter_markers()):
            item.add_marker(pytest.mark.unit)


# Test configuration
TEST_CONFIG = {
    "timeout": 30,
    "mock_api_calls": True,
    "use_test_database": False,
    "log_level": "DEBUG"
}


__all__ = ["TEST_CONFIG"]