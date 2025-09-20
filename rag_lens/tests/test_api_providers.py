"""
API Providers Tests for RAG Lens

Test suite for the API provider system.
"""

import pytest
import json
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime

from rag_lens.api.providers import (
    APIResponse, BaseAPIProvider, OpenAIProvider, AzureOpenAIProvider,
    ElasticsearchProvider, CrossEncoderProvider, APIManager
)
from rag_lens.utils.errors import APIError


class TestAPIResponse:
    """Test cases for APIResponse dataclass"""

    def test_api_response_creation(self):
        """Test APIResponse creation"""
        response = APIResponse(
            success=True,
            data={"result": "test"},
            response_time=0.5,
            metadata={"provider": "openai"}
        )

        assert response.success is True
        assert response.data == {"result": "test"}
        assert response.response_time == 0.5
        assert response.metadata == {"provider": "openai"}

    def test_api_response_to_dict(self):
        """Test APIResponse to_dict conversion"""
        response = APIResponse(
            success=True,
            data={"result": "test"},
            error_message="No error",
            error_code="SUCCESS",
            status_code=200,
            response_time=0.5,
            metadata={"provider": "openai"}
        )

        result = response.to_dict()

        assert result["success"] is True
        assert result["data"] == {"result": "test"}
        assert result["error_message"] == "No error"
        assert result["error_code"] == "SUCCESS"
        assert result["status_code"] == 200
        assert result["response_time"] == 0.5
        assert result["metadata"] == {"provider": "openai"}

    def test_api_response_to_dict_defaults(self):
        """Test APIResponse to_dict with default values"""
        response = APIResponse(success=True)

        result = response.to_dict()

        assert result["success"] is True
        assert result["data"] is None
        assert result["error_message"] is None
        assert result["error_code"] is None
        assert result["status_code"] is None
        assert result["response_time"] is None
        assert result["metadata"] == {}


class MockBaseAPIProvider(BaseAPIProvider):
    """Mock implementation of BaseAPIProvider for testing"""

    def health_check(self) -> APIResponse:
        return APIResponse(success=True, data={"status": "healthy"})

    def execute_pipeline_step(self, step_number: int, data: dict) -> APIResponse:
        return APIResponse(success=True, data={"step": step_number, "processed": data})


class TestBaseAPIProvider:
    """Test cases for BaseAPIProvider"""

    def setup_method(self):
        """Setup test environment"""
        self.provider = MockBaseAPIProvider(
            base_url="https://api.example.com",
            api_key="test_key",
            timeout=30
        )

    def test_init(self):
        """Test BaseAPIProvider initialization"""
        assert self.provider.base_url == "https://api.example.com"
        assert self.provider.api_key == "test_key"
        assert self.provider.timeout == 30
        assert self.provider.session is not None

    def test_setup_session(self):
        """Test session setup"""
        provider = MockBaseAPIProvider(
            base_url="https://api.example.com",
            api_key="test_key"
        )

        assert 'Content-Type' in provider.session.headers
        assert provider.session.headers['Content-Type'] == 'application/json'
        assert 'Authorization' in provider.session.headers
        assert provider.session.headers['Authorization'] == 'Bearer test_key'

    def test_make_request_success(self):
        """Test successful HTTP request"""
        with patch('requests.Session.request') as mock_request:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {"result": "success"}
            mock_response.headers = {"content-type": "application/json"}
            mock_request.return_value = mock_response

            response = self.provider.make_request("GET", "/test")

            assert response.success is True
            assert response.data == {"result": "success"}
            assert response.status_code == 200

    def test_make_request_failure(self):
        """Test failed HTTP request"""
        with patch('requests.Session.request') as mock_request:
            mock_response = Mock()
            mock_response.status_code = 404
            mock_response.text = "Not found"
            mock_request.return_value = mock_response

            response = self.provider.make_request("GET", "/nonexistent")

            assert response.success is False
            assert response.status_code == 404
            assert "Not found" in response.error_message

    def test_make_request_exception(self):
        """Test HTTP request with exception"""
        with patch('requests.Session.request') as mock_request:
            mock_request.side_effect = Exception("Connection error")

            response = self.provider.make_request("GET", "/test")

            assert response.success is False
            assert "Connection error" in response.error_message

    @patch('aiohttp.ClientSession')
    def test_make_async_request_success(self, mock_session_class):
        """Test successful async HTTP request"""
        async def test_async():
            mock_session = MagicMock()
            mock_response = MagicMock()
            mock_response.status = 200
            mock_response.json.return_value = {"result": "success"}
            mock_session.request.return_value.__aenter__.return_value = mock_response
            mock_session_class.return_value.__aenter__.return_value = mock_session

            response = await self.provider.make_async_request("GET", "/test")

            assert response.success is True
            assert response.data == {"result": "success"}
            assert response.status_code == 200

        # Run async test
        import asyncio
        asyncio.run(test_async())

    def test_health_check(self):
        """Test health check implementation"""
        response = self.provider.health_check()
        assert response.success is True
        assert response.data == {"status": "healthy"}

    def test_execute_pipeline_step(self):
        """Test pipeline step execution"""
        data = {"query": "test query", "test": "data"}
        response = self.provider.execute_pipeline_step(1, data)

        assert response.success is True
        assert response.data["step"] == 1
        assert response.data["processed"] == data


class TestOpenAIProvider:
    """Test cases for OpenAIProvider"""

    def setup_method(self):
        """Setup test environment"""
        self.provider = OpenAIProvider(api_key="test_key", model="gpt-4")

    def test_init(self):
        """Test OpenAIProvider initialization"""
        assert self.provider.base_url == "https://api.openai.com/v1"
        assert self.provider.api_key == "test_key"
        assert self.provider.model == "gpt-4"

    def test_health_check(self):
        """Test OpenAI health check"""
        with patch.object(self.provider, 'make_request') as mock_request:
            mock_response = APIResponse(success=True, data={"object": "list"})
            mock_request.return_value = mock_response

            response = self.provider.health_check()

            assert response.success is True
            mock_request.assert_called_once_with("GET", "/models")

    def test_generate_queries(self):
        """Test query generation"""
        with patch.object(self.provider, 'make_request') as mock_request:
            mock_response = APIResponse(
                success=True,
                data={
                    "choices": [{
                        "message": {
                            "content": "Query 1\nQuery 2\nQuery 3"
                        }
                    }]
                }
            )
            mock_request.return_value = mock_response

            data = {
                "system_prompt": "You are a helpful assistant",
                "query": "What is AI?"
            }

            response = self.provider._generate_queries(data)

            assert response.success is True
            assert len(response.data["queries"]) == 3
            assert "Query 1" in response.data["queries"]

    def test_generate_queries_parse_error(self):
        """Test query generation with parse error"""
        with patch.object(self.provider, 'make_request') as mock_request:
            mock_response = APIResponse(
                success=True,
                data={"invalid": "response"}  # Missing expected structure
            )
            mock_request.return_value = mock_response

            data = {
                "system_prompt": "You are a helpful assistant",
                "query": "What is AI?"
            }

            response = self.provider._generate_queries(data)

            assert response.success is False
            assert "parse error" in response.error_message

    def test_encode_queries(self):
        """Test query encoding"""
        with patch.object(self.provider, 'make_request') as mock_request:
            mock_response = APIResponse(
                success=True,
                data={
                    "data": [
                        {"embedding": [0.1, 0.2, 0.3]},
                        {"embedding": [0.4, 0.5, 0.6]}
                    ]
                }
            )
            mock_request.return_value = mock_response

            data = {"queries": ["Query 1", "Query 2"]}
            response = self.provider._encode_queries(data)

            assert response.success is True
            assert len(response.data["embeddings"]) == 2
            assert response.data["embeddings"][0] == [0.1, 0.2, 0.3]

    def test_generate_final_answer(self):
        """Test final answer generation"""
        with patch.object(self.provider, 'make_request') as mock_request:
            mock_response = APIResponse(
                success=True,
                data={
                    "choices": [{
                        "message": {
                            "content": "AI is artificial intelligence"
                        }
                    }]
                }
            )
            mock_request.return_value = mock_response

            data = {
                "context": "AI is a field of computer science",
                "query": "What is AI?"
            }

            response = self.provider._generate_final_answer(data)

            assert response.success is True
            assert response.data["answer"] == "AI is artificial intelligence"

    def test_execute_supported_step(self):
        """Test executing supported pipeline step"""
        data = {"query": "What is AI?", "system_prompt": "You are an assistant"}
        response = self.provider.execute_pipeline_step(0, data)  # Query generation

        assert response.success is True

    def test_execute_unsupported_step(self):
        """Test executing unsupported pipeline step"""
        data = {"query": "test"}
        response = self.provider.execute_pipeline_step(2, data)  # Candidate generation

        assert response.success is False
        assert "Unsupported pipeline step" in response.error_message


class TestAzureOpenAIProvider:
    """Test cases for AzureOpenAIProvider"""

    def setup_method(self):
        """Setup test environment"""
        self.provider = AzureOpenAIProvider(
            api_key="test_key",
            endpoint="https://test.openai.azure.com/",
            deployment_name="test-deployment"
        )

    def test_init(self):
        """Test AzureOpenAIProvider initialization"""
        assert self.provider.base_url == "https://test.openai.azure.com"
        assert self.provider.api_key == "test_key"
        assert self.provider.deployment_name == "test-deployment"

    def test_health_check(self):
        """Test Azure OpenAI health check"""
        with patch.object(self.provider, 'make_request') as mock_request:
            mock_response = APIResponse(success=True, data={})
            mock_request.return_value = mock_response

            response = self.provider.health_check()

            assert response.success is True
            expected_url = "/openai/deployments?api-version=2023-12-01-preview"
            mock_request.assert_called_once_with("GET", expected_url)

    def test_generate_final_answer(self):
        """Test final answer generation"""
        with patch.object(self.provider, 'make_request') as mock_request:
            mock_response = APIResponse(
                success=True,
                data={
                    "choices": [{
                        "message": {
                            "content": "Azure AI response"
                        }
                    }]
                }
            )
            mock_request.return_value = mock_response

            data = {
                "context": "Context about AI",
                "query": "What is AI?"
            }

            response = self.provider._generate_final_answer(data)

            assert response.success is True
            assert response.data["answer"] == "Azure AI response"

            # Check the request was made to correct endpoint
            expected_url = f"/openai/deployments/{self.provider.deployment_name}/chat/completions?api-version=2023-12-01-preview"
            mock_request.assert_called_once_with("POST", expected_url, data=mock_request.call_args[1]['data'])


class TestElasticsearchProvider:
    """Test cases for ElasticsearchProvider"""

    def setup_method(self):
        """Setup test environment"""
        self.provider = ElasticsearchProvider(
            host="https://localhost:9200",
            username="elastic",
            password="password"
        )

    def test_init(self):
        """Test ElasticsearchProvider initialization"""
        assert self.provider.base_url == "https://localhost:9200"
        assert self.provider.session.auth == ("elastic", "password")

    def test_health_check(self):
        """Test Elasticsearch health check"""
        with patch.object(self.provider, 'make_request') as mock_request:
            mock_response = APIResponse(success=True, data={"cluster_name": "test"})
            mock_request.return_value = mock_response

            response = self.provider.health_check()

            assert response.success is True
            mock_request.assert_called_once_with("GET", "/_cluster/health")

    def test_search_candidates(self):
        """Test candidate search"""
        with patch.object(self.provider, 'make_request') as mock_request:
            mock_response = APIResponse(
                success=True,
                data={
                    "hits": {
                        "hits": [
                            {
                                "_source": {
                                    "title": "Document 1",
                                    "content": "Content about AI"
                                }
                            },
                            {
                                "_source": {
                                    "title": "Document 2",
                                    "content": "Content about ML"
                                }
                            }
                        ]
                    }
                }
            )
            mock_request.return_value = mock_response

            data = {
                "index_name": "documents",
                "query": {"match": {"content": "AI"}},
                "size": 10
            }

            response = self.provider._search_candidates(data)

            assert response.success is True
            assert len(response.data["candidates"]) == 2
            assert response.data["candidates"][0]["title"] == "Document 1"


class TestCrossEncoderProvider:
    """Test cases for CrossEncoderProvider"""

    def setup_method(self):
        """Setup test environment"""
        self.provider = CrossEncoderProvider(
            model_url="https://model.example.com",
            api_key="test_key"
        )

    def test_init(self):
        """Test CrossEncoderProvider initialization"""
        assert self.provider.base_url == "https://model.example.com"
        assert self.provider.api_key == "test_key"

    def test_health_check(self):
        """Test cross-encoder health check"""
        with patch.object(self.provider, 'make_request') as mock_request:
            mock_response = APIResponse(success=True, data={"status": "ready"})
            mock_request.return_value = mock_response

            response = self.provider.health_check()

            assert response.success is True
            mock_request.assert_called_once_with("GET", "/health")

    def test_rerank_candidates(self):
        """Test candidate re-ranking"""
        with patch.object(self.provider, 'make_request') as mock_request:
            mock_response = APIResponse(
                success=True,
                data={
                    "reranked_results": [
                        {"candidate": "Candidate 1", "score": 0.9},
                        {"candidate": "Candidate 2", "score": 0.7}
                    ]
                }
            )
            mock_request.return_value = mock_response

            data = {
                "query": "What is AI?",
                "candidates": ["Candidate 1", "Candidate 2"]
            }

            response = self.provider._rerank_candidates(data)

            assert response.success is True
            assert len(response.data["reranked_candidates"]) == 2
            assert response.data["reranked_candidates"][0]["score"] == 0.9


class TestAPIManager:
    """Test cases for APIManager"""

    def setup_method(self):
        """Setup test environment"""
        with patch('rag_lens.api.providers.config'):
            self.api_manager = APIManager()

    def test_init(self):
        """Test APIManager initialization"""
        assert self.api_manager.providers == {}
        assert self.api_manager.provider_configs is not None

    def test_register_provider(self):
        """Test registering a provider"""
        provider = MockBaseAPIProvider("https://test.com", "test_key")
        self.api_manager.register_provider("test", provider)

        assert "test" in self.api_manager.providers
        assert self.api_manager.providers["test"] is provider

    def test_get_existing_provider(self):
        """Test getting existing provider"""
        provider = MockBaseAPIProvider("https://test.com", "test_key")
        self.api_manager.register_provider("test", provider)

        retrieved = self.api_manager.get_provider("test")
        assert retrieved is provider

    def test_create_openai_provider(self):
        """Test creating OpenAI provider"""
        with patch.dict(self.api_manager.provider_configs, {
            "openai": {"api_key": "test_key", "model": "gpt-4"}
        }):
            provider = self.api_manager._create_provider("openai")
            assert provider is not None
            assert isinstance(provider, OpenAIProvider)
            assert provider.api_key == "test_key"

    def test_create_provider_missing_config(self):
        """Test creating provider with missing configuration"""
        provider = self.api_manager._create_provider("openai")
        assert provider is None

    def test_get_provider_auto_create(self):
        """Test getting provider that doesn't exist but can be created"""
        with patch.dict(self.api_manager.provider_configs, {
            "openai": {"api_key": "test_key", "model": "gpt-4"}
        }):
            provider = self.api_manager.get_provider("openai")
            assert provider is not None
            assert isinstance(provider, OpenAIProvider)

    def test_get_provider_nonexistent(self):
        """Test getting non-existent provider that can't be created"""
        with pytest.raises(APIError):
            self.api_manager.get_provider("nonexistent")

    def test_execute_pipeline_step_success(self):
        """Test successful pipeline step execution"""
        provider = MockBaseAPIProvider("https://test.com", "test_key")
        self.api_manager.register_provider("test", provider)

        data = {"query": "test query"}
        response = self.api_manager.execute_pipeline_step("test", 1, data)

        assert response.success is True

    def test_execute_pipeline_step_provider_error(self):
        """Test pipeline step execution with provider error"""
        with patch.object(self.api_manager, 'get_provider') as mock_get_provider:
            mock_get_provider.side_effect = APIError("Provider error")

            data = {"query": "test query"}
            response = self.api_manager.execute_pipeline_step("test", 1, data)

            assert response.success is False
            assert "Provider error" in response.error_message

    def test_health_check_all(self):
        """Test health check for all providers"""
        provider1 = MockBaseAPIProvider("https://test1.com", "test_key")
        provider2 = MockBaseAPIProvider("https://test2.com", "test_key")

        self.api_manager.register_provider("provider1", provider1)
        self.api_manager.register_provider("provider2", provider2)

        results = self.api_manager.health_check_all()

        assert "provider1" in results
        assert "provider2" in results
        assert results["provider1"].success is True
        assert results["provider2"].success is True

    def test_health_check_all_with_error(self):
        """Test health check with provider error"""
        provider1 = MockBaseAPIProvider("https://test1.com", "test_key")
        provider2 = MockBaseAPIProvider("https://test2.com", "test_key")

        # Make provider2 health check fail
        def failing_health_check():
            return APIResponse(success=False, error_message="Health check failed")

        provider2.health_check = failing_health_check

        self.api_manager.register_provider("provider1", provider1)
        self.api_manager.register_provider("provider2", provider2)

        results = self.api_manager.health_check_all()

        assert results["provider1"].success is True
        assert results["provider2"].success is False
        assert "Health check failed" in results["provider2"].error_message

    def test_get_available_providers(self):
        """Test getting available providers"""
        provider1 = MockBaseAPIProvider("https://test1.com", "test_key")
        provider2 = MockBaseAPIProvider("https://test2.com", "test_key")

        self.api_manager.register_provider("provider1", provider1)
        self.api_manager.register_provider("provider2", provider2)

        providers = self.api_manager.get_available_providers()

        assert len(providers) == 2
        assert "provider1" in providers
        assert "provider2" in providers


if __name__ == "__main__":
    pytest.main([__file__])