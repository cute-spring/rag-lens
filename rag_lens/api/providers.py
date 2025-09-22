"""
API Providers Module for RAG Lens

This module contains standardized API provider implementations following
the interface contracts defined in the integration guide.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass
import requests
import json
import time
from datetime import datetime, timedelta
import asyncio
import aiohttp
from concurrent.futures import ThreadPoolExecutor

from ..config.settings import config
from ..utils.logger import get_logger
from ..utils.errors import APIError, RetryHandler, CircuitBreaker, ErrorHandler
from ..utils.security import security_manager

logger = get_logger(__name__)


@dataclass
class APIResponse:
    """Standardized API response format"""
    success: bool
    data: Any = None
    error_message: str = None
    error_code: str = None
    status_code: int = None
    response_time: float = None
    metadata: Dict[str, Any] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "success": self.success,
            "data": self.data,
            "error_message": self.error_message,
            "error_code": self.error_code,
            "status_code": self.status_code,
            "response_time": self.response_time,
            "metadata": self.metadata or {}
        }


class BaseAPIProvider(ABC):
    """Abstract base class for all API providers"""

    def __init__(self, base_url: str, api_key: str = None, timeout: int = 30):
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.timeout = timeout
        self.session = requests.Session()
        self.retry_handler = RetryHandler(max_retries=3, delay=1.0)
        self.circuit_breaker = CircuitBreaker(failure_threshold=5, recovery_timeout=60)
        self._setup_session()

    def _setup_session(self):
        """Setup session with default headers and authentication"""
        self.session.headers.update({
            'Content-Type': 'application/json',
            'User-Agent': f'RAG-Lens/{config.app.version}'
        })

        if self.api_key:
            self.session.headers.update({
                'Authorization': f'Bearer {self.api_key}'
            })

    @abstractmethod
    def health_check(self) -> APIResponse:
        """Check API health status"""
        pass

    @abstractmethod
    def execute_pipeline_step(self, step_number: int, data: Dict[str, Any]) -> APIResponse:
        """Execute a specific pipeline step"""
        pass

    def make_request(
        self,
        method: str,
        endpoint: str,
        data: Dict[str, Any] = None,
        params: Dict[str, Any] = None,
        headers: Dict[str, Any] = None
    ) -> APIResponse:
        """Make HTTP request with error handling and retries"""
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        start_time = time.time()

        try:
            response = self.retry_handler.retry(
                self._make_single_request,
                method, url, data, params, headers
            )

            response_time = time.time() - start_time

            if response.status_code == 200:
                return APIResponse(
                    success=True,
                    data=response.json(),
                    status_code=response.status_code,
                    response_time=response_time,
                    metadata={'headers': dict(response.headers)}
                )
            else:
                return APIResponse(
                    success=False,
                    error_message=response.text,
                    error_code=str(response.status_code),
                    status_code=response.status_code,
                    response_time=response_time
                )

        except Exception as e:
            response_time = time.time() - start_time
            error_info = ErrorHandler.handle_error(e, {'endpoint': endpoint, 'method': method})
            return APIResponse(
                success=False,
                error_message=error_info['error_message'],
                error_code=error_info.get('error_code', 'UNKNOWN_ERROR'),
                response_time=response_time
            )

    def _make_single_request(self, method: str, url: str, data: Dict[str, Any], params: Dict[str, Any], headers: Dict[str, Any]):
        """Make a single HTTP request"""
        request_headers = self.session.headers.copy()
        if headers:
            request_headers.update(headers)

        return self.session.request(
            method=method,
            url=url,
            json=data,
            params=params,
            headers=request_headers,
            timeout=self.timeout
        )

    async def make_async_request(self, method: str, endpoint: str, data: Dict[str, Any] = None) -> APIResponse:
        """Make async HTTP request"""
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        start_time = time.time()

        try:
            async with aiohttp.ClientSession(
                headers=self.session.headers,
                timeout=aiohttp.ClientTimeout(total=self.timeout)
            ) as session:
                async with session.request(method, url, json=data) as response:
                    response_time = time.time() - start_time
                    response_data = await response.json()

                    if response.status == 200:
                        return APIResponse(
                            success=True,
                            data=response_data,
                            status_code=response.status,
                            response_time=response_time
                        )
                    else:
                        return APIResponse(
                            success=False,
                            error_message=str(response_data),
                            status_code=response.status,
                            response_time=response_time
                        )

        except Exception as e:
            response_time = time.time() - start_time
            return APIResponse(
                success=False,
                error_message=str(e),
                response_time=response_time
            )


class OpenAIProvider(BaseAPIProvider):
    """OpenAI API provider for language model operations"""

    def __init__(self, api_key: str, model: str = "gpt-4"):
        super().__init__(base_url="https://api.openai.com/v1", api_key=api_key)
        self.model = model

    def health_check(self) -> APIResponse:
        """Check OpenAI API health"""
        try:
            response = self.make_request("GET", "/models")
            return response
        except Exception as e:
            return APIResponse(
                success=False,
                error_message=str(e),
                error_code="OPENAI_HEALTH_CHECK_FAILED"
            )

    def execute_pipeline_step(self, step_number: int, data: Dict[str, Any]) -> APIResponse:
        """Execute pipeline step using OpenAI"""
        step_handlers = {
            0: self._generate_queries,
            1: self._encode_queries,
            6: self._generate_final_answer
        }

        handler = step_handlers.get(step_number)
        if not handler:
            return APIResponse(
                success=False,
                error_message=f"Unsupported pipeline step: {step_number}",
                error_code="UNSUPPORTED_STEP"
            )

        return handler(data)

    def _generate_queries(self, data: Dict[str, Any]) -> APIResponse:
        """Generate search queries using OpenAI"""
        system_prompt = data.get("system_prompt", "You are a helpful assistant.")
        user_query = data.get("query", "")

        request_data = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Generate search queries for: {user_query}"}
            ],
            "max_tokens": 150,
            "temperature": 0.7
        }

        response = self.make_request("POST", "/chat/completions", data=request_data)
        if response.success:
            try:
                queries = response.data["choices"][0]["message"]["content"].strip().split('\n')
                response.data = {"queries": [q.strip() for q in queries if q.strip()]}
            except (KeyError, IndexError) as e:
                response.success = False
                response.error_message = f"Failed to parse OpenAI response: {e}"
                response.error_code = "OPENAI_PARSE_ERROR"

        return response

    def _encode_queries(self, data: Dict[str, Any]) -> APIResponse:
        """Encode queries using OpenAI embeddings"""
        queries = data.get("queries", [])

        request_data = {
            "model": "text-embedding-ada-002",
            "input": queries
        }

        response = self.make_request("POST", "/embeddings", data=request_data)
        if response.success:
            try:
                embeddings = [item["embedding"] for item in response.data["data"]]
                response.data = {"embeddings": embeddings}
            except (KeyError, IndexError) as e:
                response.success = False
                response.error_message = f"Failed to parse embeddings: {e}"
                response.error_code = "OPENAI_EMBEDDING_PARSE_ERROR"

        return response

    def _generate_final_answer(self, data: Dict[str, Any]) -> APIResponse:
        """Generate final answer using OpenAI"""
        context = data.get("context", "")
        query = data.get("query", "")

        system_prompt = f"""You are a helpful assistant. Use the following context to answer the user's question.

Context:
{context}

Question: {query}

Provide a comprehensive answer based on the context."""

        request_data = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query}
            ],
            "max_tokens": 1000,
            "temperature": 0.3
        }

        response = self.make_request("POST", "/chat/completions", data=request_data)
        if response.success:
            try:
                answer = response.data["choices"][0]["message"]["content"]
                response.data = {"answer": answer}
            except (KeyError, IndexError) as e:
                response.success = False
                response.error_message = f"Failed to parse answer: {e}"
                response.error_code = "OPENAI_ANSWER_PARSE_ERROR"

        return response


class AzureOpenAIProvider(BaseAPIProvider):
    """Azure OpenAI API provider"""

    def __init__(self, api_key: str, endpoint: str, deployment_name: str):
        super().__init__(base_url=endpoint, api_key=api_key)
        self.deployment_name = deployment_name
        self.api_version = "2023-12-01-preview"

    def health_check(self) -> APIResponse:
        """Check Azure OpenAI API health"""
        try:
            response = self.make_request("GET", f"/openai/deployments?api-version={self.api_version}")
            return response
        except Exception as e:
            return APIResponse(
                success=False,
                error_message=str(e),
                error_code="AZURE_OPENAI_HEALTH_CHECK_FAILED"
            )

    def execute_pipeline_step(self, step_number: int, data: Dict[str, Any]) -> APIResponse:
        """Execute pipeline step using Azure OpenAI"""
        if step_number == 6:  # Final answer generation
            return self._generate_final_answer(data)
        else:
            return APIResponse(
                success=False,
                error_message=f"Azure OpenAI provider only supports final answer generation (step 6)",
                error_code="UNSUPPORTED_STEP"
            )

    def _generate_final_answer(self, data: Dict[str, Any]) -> APIResponse:
        """Generate final answer using Azure OpenAI"""
        context = data.get("context", "")
        query = data.get("query", "")

        request_data = {
            "messages": [
                {"role": "system", "content": f"Use this context to answer: {context}"},
                {"role": "user", "content": query}
            ],
            "max_tokens": 1000,
            "temperature": 0.3
        }

        response = self.make_request(
            "POST",
            f"/openai/deployments/{self.deployment_name}/chat/completions?api-version={self.api_version}",
            data=request_data
        )

        if response.success:
            try:
                answer = response.data["choices"][0]["message"]["content"]
                response.data = {"answer": answer}
            except (KeyError, IndexError) as e:
                response.success = False
                response.error_message = f"Failed to parse answer: {e}"
                response.error_code = "AZURE_OPENAI_ANSWER_PARSE_ERROR"

        return response


class ElasticsearchProvider(BaseAPIProvider):
    """Elasticsearch provider for search operations"""

    def __init__(self, host: str, api_key: str = None, username: str = None, password: str = None):
        super().__init__(base_url=host, api_key=api_key)
        if username and password:
            self.session.auth = (username, password)

    def health_check(self) -> APIResponse:
        """Check Elasticsearch cluster health"""
        try:
            response = self.make_request("GET", "/_cluster/health")
            return response
        except Exception as e:
            return APIResponse(
                success=False,
                error_message=str(e),
                error_code="ELASTICSEARCH_HEALTH_CHECK_FAILED"
            )

    def execute_pipeline_step(self, step_number: int, data: Dict[str, Any]) -> APIResponse:
        """Execute pipeline step using Elasticsearch"""
        if step_number in [2, 3]:  # Candidate generation and filtering
            return self._search_candidates(data)
        else:
            return APIResponse(
                success=False,
                error_message=f"Elasticsearch provider only supports search operations (steps 2-3)",
                error_code="UNSUPPORTED_STEP"
            )

    def _search_candidates(self, data: Dict[str, Any]) -> APIResponse:
        """Search for candidates using Elasticsearch"""
        index_name = data.get("index_name", "documents")
        query = data.get("query", {})
        size = data.get("size", 10)

        request_data = {
            "query": query,
            "size": size
        }

        response = self.make_request("POST", f"/{index_name}/_search", data=request_data)
        if response.success:
            try:
                hits = response.data["hits"]["hits"]
                candidates = [hit["_source"] for hit in hits]
                response.data = {"candidates": candidates}
            except (KeyError, IndexError) as e:
                response.success = False
                response.error_message = f"Failed to parse search results: {e}"
                response.error_code = "ELASTICSEARCH_PARSE_ERROR"

        return response


class CrossEncoderProvider(BaseAPIProvider):
    """Cross-encoder provider for re-ranking"""

    def __init__(self, model_url: str, api_key: str = None):
        super().__init__(base_url=model_url, api_key=api_key)

    def health_check(self) -> APIResponse:
        """Check cross-encoder model health"""
        try:
            response = self.make_request("GET", "/health")
            return response
        except Exception as e:
            return APIResponse(
                success=False,
                error_message=str(e),
                error_code="CROSS_ENCODER_HEALTH_CHECK_FAILED"
            )

    def execute_pipeline_step(self, step_number: int, data: Dict[str, Any]) -> APIResponse:
        """Execute pipeline step using cross-encoder"""
        if step_number == 5:  # Re-ranking
            return self._rerank_candidates(data)
        else:
            return APIResponse(
                success=False,
                error_message=f"Cross-encoder provider only supports re-ranking (step 5)",
                error_code="UNSUPPORTED_STEP"
            )

    def _rerank_candidates(self, data: Dict[str, Any]) -> APIResponse:
        """Re-rank candidates using cross-encoder"""
        query = data.get("query", "")
        candidates = data.get("candidates", [])

        request_data = {
            "query": query,
            "candidates": candidates
        }

        response = self.make_request("POST", "/rerank", data=request_data)
        if response.success:
            try:
                reranked_results = response.data.get("reranked_results", [])
                response.data = {"reranked_candidates": reranked_results}
            except (KeyError, IndexError) as e:
                response.success = False
                response.error_message = f"Failed to parse rerank results: {e}"
                response.error_code = "CROSS_ENCODER_PARSE_ERROR"

        return response


class OllamaRerankProvider(BaseAPIProvider):
    """Ollama provider for re-ranking using local models"""

    def __init__(self, base_url: str = "http://localhost:11434", model: str = "nomic-embed-text", api_key: str = None):
        super().__init__(base_url=base_url, api_key=api_key)
        self.model = model
        # Ollama doesn't require API key authentication by default
        if not api_key:
            # Remove Authorization header for Ollama
            if 'Authorization' in self.session.headers:
                del self.session.headers['Authorization']

    def health_check(self) -> APIResponse:
        """Check Ollama server health"""
        try:
            # Check if Ollama is running by listing models
            response = self.make_request("GET", "/api/tags")
            if response.success:
                models = response.data.get("models", [])
                model_names = [model.get("name", "") for model in models]
                if self.model in model_names or any(self.model in name for name in model_names):
                    return APIResponse(
                        success=True,
                        data={"status": "healthy", "model_available": True, "available_models": model_names},
                        metadata={"model": self.model}
                    )
                else:
                    return APIResponse(
                        success=False,
                        error_message=f"Model '{self.model}' not found in Ollama. Available models: {model_names}",
                        error_code="OLLAMA_MODEL_NOT_FOUND"
                    )
            return response
        except Exception as e:
            return APIResponse(
                success=False,
                error_message=f"Ollama health check failed: {str(e)}",
                error_code="OLLAMA_HEALTH_CHECK_FAILED"
            )

    def execute_pipeline_step(self, step_number: int, data: Dict[str, Any]) -> APIResponse:
        """Execute pipeline step using Ollama"""
        if step_number == 5:  # Re-ranking
            return self._rerank_candidates(data)
        else:
            return APIResponse(
                success=False,
                error_message=f"Ollama provider only supports re-ranking (step 5)",
                error_code="UNSUPPORTED_STEP"
            )

    def _rerank_candidates(self, data: Dict[str, Any]) -> APIResponse:
        """Re-rank candidates using Ollama BGE reranker model"""
        query = data.get("query", "")
        candidates = data.get("candidates", [])
        
        if not query or not candidates:
            return APIResponse(
                success=False,
                error_message="Query and candidates are required for re-ranking",
                error_code="MISSING_RERANK_DATA"
            )

        try:
            # Check if this is a dedicated reranker model (like BGE reranker)
            if "reranker" in self.model.lower() or "bge-reranker" in self.model.lower():
                return self._rerank_with_dedicated_model(query, candidates)
            else:
                # Fallback to embedding-based reranking for other models
                return self._rerank_with_embeddings(query, candidates)
            
        except Exception as e:
            return APIResponse(
                success=False,
                error_message=f"Ollama re-ranking failed: {str(e)}",
                error_code="OLLAMA_RERANK_ERROR"
            )
    
    def _rerank_with_dedicated_model(self, query: str, candidates: List[Dict[str, Any]]) -> APIResponse:
        """Re-rank using dedicated reranker model via Ollama generate API"""
        scored_candidates = []
        
        for i, candidate in enumerate(candidates):
            candidate_text = candidate.get("text", "") if isinstance(candidate, dict) else str(candidate)
            
            # Create a prompt for the reranker model to score relevance
            prompt = f"Query: {query}\nDocument: {candidate_text}\nRelevance score (0-1):"
            
            request_data = {
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.1,  # Low temperature for consistent scoring
                    "num_predict": 10    # Short response for score
                }
            }
            
            response = self.make_request("POST", "/api/generate", data=request_data)
            
            if response.success:
                response_text = response.data.get("response", "0.0").strip()
                try:
                    # Extract numerical score from response
                    import re
                    score_match = re.search(r'(\d+\.?\d*)', response_text)
                    score = float(score_match.group(1)) if score_match else 0.0
                    # Ensure score is between 0 and 1
                    score = max(0.0, min(1.0, score))
                except (ValueError, AttributeError):
                    score = 0.0
            else:
                score = 0.0
            
            scored_candidate = {
                "text": candidate_text,
                "score": score,
                "original_index": i
            }
            
            # Preserve original candidate structure if it's a dict
            if isinstance(candidate, dict):
                scored_candidate.update({k: v for k, v in candidate.items() if k != "text"})
            
            scored_candidates.append(scored_candidate)
        
        # Sort by score (descending)
        reranked_candidates = sorted(scored_candidates, key=lambda x: x["score"], reverse=True)
        
        return APIResponse(
            success=True,
            data={"reranked_candidates": reranked_candidates},
            metadata={
                "model": self.model,
                "reranking_method": "dedicated_reranker",
                "total_candidates": len(candidates),
                "reranked_count": len(reranked_candidates)
            }
        )
    
    def _rerank_with_embeddings(self, query: str, candidates: List[Dict[str, Any]]) -> APIResponse:
        """Re-rank using embedding similarity (fallback method)"""
        # Get embedding for the query
        query_embedding_response = self._get_embedding(query)
        if not query_embedding_response.success:
            return query_embedding_response
        
        query_embedding = query_embedding_response.data.get("embedding", [])
        
        # Get embeddings for all candidates and calculate similarity scores
        scored_candidates = []
        for i, candidate in enumerate(candidates):
            candidate_text = candidate.get("text", "") if isinstance(candidate, dict) else str(candidate)
            
            candidate_embedding_response = self._get_embedding(candidate_text)
            if candidate_embedding_response.success:
                candidate_embedding = candidate_embedding_response.data.get("embedding", [])
                
                # Calculate cosine similarity
                similarity_score = self._cosine_similarity(query_embedding, candidate_embedding)
                
                scored_candidate = {
                    "text": candidate_text,
                    "score": similarity_score,
                    "original_index": i
                }
                
                # Preserve original candidate structure if it's a dict
                if isinstance(candidate, dict):
                    scored_candidate.update({k: v for k, v in candidate.items() if k != "text"})
                
                scored_candidates.append(scored_candidate)
            else:
                # If embedding fails, assign a low score but keep the candidate
                scored_candidate = {
                    "text": candidate_text,
                    "score": 0.0,
                    "original_index": i,
                    "embedding_error": candidate_embedding_response.error_message
                }
                if isinstance(candidate, dict):
                    scored_candidate.update({k: v for k, v in candidate.items() if k != "text"})
                scored_candidates.append(scored_candidate)
        
        # Sort by similarity score (descending)
        reranked_candidates = sorted(scored_candidates, key=lambda x: x["score"], reverse=True)
        
        return APIResponse(
            success=True,
            data={"reranked_candidates": reranked_candidates},
            metadata={
                "model": self.model,
                "reranking_method": "embedding_similarity",
                "total_candidates": len(candidates),
                "reranked_count": len(reranked_candidates)
            }
        )

    def _get_embedding(self, text: str) -> APIResponse:
        """Get embedding for text using Ollama"""
        request_data = {
            "model": self.model,
            "prompt": text
        }
        
        return self.make_request("POST", "/api/embeddings", data=request_data)
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors"""
        if not vec1 or not vec2 or len(vec1) != len(vec2):
            return 0.0
        
        try:
            import math
            
            # Calculate dot product
            dot_product = sum(a * b for a, b in zip(vec1, vec2))
            
            # Calculate magnitudes
            magnitude1 = math.sqrt(sum(a * a for a in vec1))
            magnitude2 = math.sqrt(sum(a * a for a in vec2))
            
            # Avoid division by zero
            if magnitude1 == 0.0 or magnitude2 == 0.0:
                return 0.0
            
            return dot_product / (magnitude1 * magnitude2)
        except Exception:
            return 0.0


class APIManager:
    """Manager for coordinating multiple API providers"""

    def __init__(self):
        self.providers = {}
        self.provider_configs = self._load_provider_configs()

    def _load_provider_configs(self) -> Dict[str, Dict[str, Any]]:
        """Load provider configurations from settings"""
        return {
            "openai": {
                "api_key": config.api.openai_api_key,
                "model": config.api.openai_model
            },
            "azure_openai": {
                "api_key": config.api.azure_api_key,
                "endpoint": config.api.azure_endpoint,
                "deployment_name": config.api.azure_deployment_name
            },
            "elasticsearch": {
                "host": config.api.elasticsearch_host,
                "api_key": config.api.elasticsearch_api_key,
                "username": config.api.elasticsearch_username,
                "password": config.api.elasticsearch_password
            },
            "cross_encoder": {
                "model_url": config.api.cross_encoder_url,
                "api_key": config.api.cross_encoder_api_key
            },
            "ollama_rerank": {
                "base_url": getattr(config.api, 'ollama_base_url', 'http://localhost:11434'),
                "model": getattr(config.api, 'ollama_model', 'nomic-embed-text'),
                "api_key": getattr(config.api, 'ollama_api_key', None)
            }
        }

    def register_provider(self, name: str, provider: BaseAPIProvider):
        """Register an API provider"""
        self.providers[name] = provider
        logger.info(f"Registered API provider: {name}")

    def get_provider(self, name: str) -> BaseAPIProvider:
        """Get a registered API provider"""
        if name not in self.providers:
            provider = self._create_provider(name)
            if provider:
                self.register_provider(name, provider)
            else:
                raise APIError(f"Provider {name} not found and could not be created")

        return self.providers[name]

    def _create_provider(self, name: str) -> Optional[BaseAPIProvider]:
        """Create a provider instance based on configuration"""
        if name == "openai":
            config_data = self.provider_configs.get("openai", {})
            if config_data.get("api_key"):
                return OpenAIProvider(
                    api_key=config_data["api_key"],
                    model=config_data.get("model", "gpt-4")
                )

        elif name == "azure_openai":
            config_data = self.provider_configs.get("azure_openai", {})
            required_fields = ["api_key", "endpoint", "deployment_name"]
            if all(config_data.get(field) for field in required_fields):
                return AzureOpenAIProvider(**{k: v for k, v in config_data.items() if k in required_fields})

        elif name == "elasticsearch":
            config_data = self.provider_configs.get("elasticsearch", {})
            if config_data.get("host"):
                return ElasticsearchProvider(
                    host=config_data["host"],
                    api_key=config_data.get("api_key"),
                    username=config_data.get("username"),
                    password=config_data.get("password")
                )

        elif name == "cross_encoder":
            config_data = self.provider_configs.get("cross_encoder", {})
            if config_data.get("model_url"):
                return CrossEncoderProvider(
                    model_url=config_data["model_url"],
                    api_key=config_data.get("api_key")
                )

        elif name == "ollama_rerank":
            config_data = self.provider_configs.get("ollama_rerank", {})
            return OllamaRerankProvider(
                base_url=config_data.get("base_url", "http://localhost:11434"),
                model=config_data.get("model", "nomic-embed-text"),
                api_key=config_data.get("api_key")
            )

        return None

    def execute_pipeline_step(self, provider_name: str, step_number: int, data: Dict[str, Any]) -> APIResponse:
        """Execute a pipeline step using the specified provider"""
        try:
            provider = self.get_provider(provider_name)
            return provider.execute_pipeline_step(step_number, data)
        except Exception as e:
            error_info = ErrorHandler.handle_error(e, {
                "provider": provider_name,
                "step": step_number,
                "data": data
            })
            return APIResponse(
                success=False,
                error_message=error_info['error_message'],
                error_code=error_info.get('error_code', 'EXECUTION_ERROR')
            )

    def health_check_all(self) -> Dict[str, APIResponse]:
        """Check health of all registered providers"""
        results = {}
        for name, provider in self.providers.items():
            try:
                results[name] = provider.health_check()
            except Exception as e:
                results[name] = APIResponse(
                    success=False,
                    error_message=str(e),
                    error_code="HEALTH_CHECK_ERROR"
                )
        return results

    def get_available_providers(self) -> List[str]:
        """Get list of available provider names"""
        return list(self.providers.keys())


# Global API manager instance
api_manager = APIManager()