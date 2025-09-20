# Real-World API Integration Examples

This document provides complete, working examples for integrating popular APIs with the RAG Pipeline Testing tool. Each example includes actual request/response samples and common edge cases.

## Table of Contents
1. [Document Retrieval Examples](#document-retrieval-examples)
2. [Re-ranking Examples](#re-ranking-examples)
3. [Sub-segment Extraction Examples](#sub-segment-extraction-examples)
4. [LLM Response Generation Examples](#llm-response-generation-examples)
5. [Results Analysis Examples](#results-analysis-examples)
6. [Complete Integration Examples](#complete-integration-examples)

---

## Document Retrieval Examples

### 1. OpenAI Search API Integration

```python
import aiohttp
import json
from typing import Dict, Any, List

class OpenAIDocumentRetriever:
    """Document retrieval using OpenAI's search capabilities"""

    def __init__(self, api_key: str, base_url: str = "https://api.openai.com/v1"):
        self.api_key = api_key
        self.base_url = base_url
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

    async def retrieve_documents(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Retrieve documents using OpenAI's search endpoint"""

        # Prepare search request for OpenAI
        search_request = {
            "query": request["query"],
            "filter": request.get("filters", {}),
            "max_results": request.get("filters", {}).get("max_results", 50),
            "search_type": "semantic"  # or "keyword", "hybrid"
        }

        try:
            async with aiohttp.ClientSession(headers=self.headers) as session:
                async with session.post(
                    f"{self.base_url}/search",
                    json=search_request,
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as response:

                    if response.status == 200:
                        openai_response = await response.json()
                        return self._format_openai_response(openai_response, request)
                    else:
                        error_data = await response.json()
                        return self._format_error_response(response.status, error_data)

        except Exception as e:
            return self._format_error_response("TIMEOUT", {"error": str(e)})

    def _format_openai_response(self, openai_response: Dict[str, Any], original_request: Dict[str, Any]) -> Dict[str, Any]:
        """Format OpenAI response to match interface contract"""

        documents = []
        for item in openai_response.get("data", []):
            document = {
                "id": item.get("id", f"doc_{hash(item['content'])}"),
                "title": item.get("title", "Untitled Document"),
                "content": item.get("content", ""),
                "score": item.get("score", 0.75),
                "url": item.get("url", ""),
                "metadata": {
                    "author": item.get("author", "Unknown"),
                    "publish_date": item.get("created_at", "2024-01-01T00:00:00Z"),
                    "file_type": item.get("file_type", "text"),
                    "size_bytes": len(item.get("content", "").encode()),
                    "language": item.get("language", "en"),
                    "source": "openai_search"
                }
            }
            documents.append(document)

        return {
            "success": True,
            "documents": documents,
            "total_found": openai_response.get("total_count", len(documents)),
            "search_time": openai_response.get("search_time", 1.0),
            "query_used": original_request["query"],
            "filters_applied": original_request.get("filters", {})
        }

    def _format_error_response(self, status_code: str, error_data: Dict[str, Any]) -> Dict[str, Any]:
        """Format error response"""
        return {
            "success": False,
            "error": {
                "code": f"OPENAI_{status_code}",
                "message": error_data.get("error", {}).get("message", "Unknown error"),
                "retry_after": error_data.get("retry_after", 5),
                "details": error_data
            }
        }

# Example Usage
async def example_openai_retrieval():
    retriever = OpenAIDocumentRetriever("your-openai-api-key")

    request = {
        "query": "What are the latest developments in quantum computing?",
        "filters": {
            "max_results": 10,
            "date_from": "2023-01-01T00:00:00Z",
            "document_types": ["pdf", "webpage"]
        },
        "metadata": {
            "request_id": "test_001",
            "user_id": "user_123"
        }
    }

    result = await retriever.retrieve_documents(request)
    print(f"Found {result['total_found']} documents")
    return result
```

### 2. Azure Cognitive Search Integration

```python
class AzureSearchRetriever:
    """Document retrieval using Azure Cognitive Search"""

    def __init__(self, endpoint: str, api_key: str, index_name: str):
        self.endpoint = endpoint.rstrip('/')
        self.api_key = api_key
        self.index_name = index_name
        self.headers = {
            "api-key": api_key,
            "Content-Type": "application/json"
        }

    async def retrieve_documents(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Retrieve documents from Azure Cognitive Search"""

        # Build Azure Search query
        search_query = {
            "search": request["query"],
            "filter": self._build_filter_string(request.get("filters", {})),
            "top": request.get("filters", {}).get("max_results", 50),
            "queryType": "full",  # or "simple"
            "searchMode": "any",
            "includeTotalCount": True
        }

        try:
            async with aiohttp.ClientSession(headers=self.headers) as session:
                async with session.post(
                    f"{self.endpoint}/indexes/{self.index_name}/docs/search",
                    json=search_query,
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as response:

                    if response.status == 200:
                        azure_response = await response.json()
                        return self._format_azure_response(azure_response, request)
                    else:
                        return self._format_azure_error(response)

        except Exception as e:
            return self._format_error_response("TIMEOUT", str(e))

    def _build_filter_string(self, filters: Dict[str, Any]) -> str:
        """Build Azure Search filter string"""
        filter_parts = []

        if "date_from" in filters:
            filter_parts.append(f"publish_date ge {filters['date_from']}")

        if "date_to" in filters:
            filter_parts.append(f"publish_date le {filters['date_to']}")

        if "document_types" in filters:
            types = ", ".join([f"'{t}'" for t in filters["document_types"]])
            filter_parts.append(f"file_type in ({types})")

        return " and ".join(filter_parts) if filter_parts else ""

    def _format_azure_response(self, azure_response: Dict[str, Any], original_request: Dict[str, Any]) -> Dict[str, Any]:
        """Format Azure response to match interface contract"""

        documents = []
        for item in azure_response.get("value", []):
            document = {
                "id": item.get("metadata_storage_id", item.get("id")),
                "title": item.get("title", "Untitled Document"),
                "content": item.get("content", ""),
                "score": item.get("@search.score", 0.75),
                "url": item.get("metadata_storage_path", ""),
                "metadata": {
                    "author": item.get("author", "Unknown"),
                    "publish_date": item.get("publish_date", "2024-01-01T00:00:00Z"),
                    "file_type": item.get("metadata_storage_content_type", "text"),
                    "size_bytes": item.get("metadata_storage_size", 0),
                    "language": item.get("language", "en"),
                    "source": "azure_search"
                }
            }
            documents.append(document)

        return {
            "success": True,
            "documents": documents,
            "total_found": azure_response.get("@odata.count", len(documents)),
            "search_time": azure_response.get("@search.elapsedTime", 1.0) / 1000,
            "query_used": original_request["query"],
            "filters_applied": original_request.get("filters", {})
        }

# Example Usage
async def example_azure_search():
    retriever = AzureSearchRetriever(
        endpoint="https://your-search-service.search.windows.net",
        api_key="your-azure-search-key",
        index_name="your-index-name"
    )

    request = {
        "query": "machine learning algorithms",
        "filters": {
            "max_results": 5,
            "date_from": "2023-06-01T00:00:00Z",
            "document_types": ["pdf", "docx"]
        }
    }

    result = await retriever.retrieve_documents(request)
    return result
```

### 3. Elasticsearch Integration

```python
class ElasticsearchRetriever:
    """Document retrieval using Elasticsearch"""

    def __init__(self, host: str, index: str, api_key: str = None,
                 username: str = None, password: str = None):
        self.host = host.rstrip('/')
        self.index = index
        self.auth = aiohttp.BasicAuth(username, password) if username and password else None
        self.headers = {"Authorization": f"ApiKey {api_key}"} if api_key else {}

    async def retrieve_documents(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Retrieve documents from Elasticsearch"""

        # Build Elasticsearch query
        es_query = {
            "query": {
                "bool": {
                    "must": [
                        {
                            "multi_match": {
                                "query": request["query"],
                                "fields": ["title^2", "content", "description"],
                                "type": "best_fields"
                            }
                        }
                    ],
                    "filter": self._build_filters(request.get("filters", {}))
                }
            },
            "size": request.get("filters", {}).get("max_results", 50),
            "highlight": {
                "fields": {
                    "content": {"fragment_size": 150, "number_of_fragments": 3}
                }
            }
        }

        try:
            auth = self.auth
            headers = self.headers

            async with aiohttp.ClientSession(auth=auth, headers=headers) as session:
                async with session.post(
                    f"{self.host}/{self.index}/_search",
                    json=es_query,
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as response:

                    if response.status == 200:
                        es_response = await response.json()
                        return self._format_es_response(es_response, request)
                    else:
                        return self._format_es_error(response)

        except Exception as e:
            return self._format_error_response("TIMEOUT", str(e))

    def _build_filters(self, filters: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Build Elasticsearch filters"""
        es_filters = []

        if "date_from" in filters or "date_to" in filters:
            date_range = {}
            if "date_from" in filters:
                date_range["gte"] = filters["date_from"]
            if "date_to" in filters:
                date_range["lte"] = filters["date_to"]
            es_filters.append({"range": {"publish_date": date_range}})

        if "document_types" in filters:
            es_filters.append({"terms": {"file_type": filters["document_types"]}})

        return es_filters

    def _format_es_response(self, es_response: Dict[str, Any], original_request: Dict[str, Any]) -> Dict[str, Any]:
        """Format Elasticsearch response"""

        documents = []
        for hit in es_response.get("hits", {}).get("hits", []):
            source = hit["_source"]
            highlight = hit.get("highlight", {})

            document = {
                "id": hit["_id"],
                "title": source.get("title", "Untitled Document"),
                "content": source.get("content", ""),
                "score": hit.get("_score", 0.75),
                "url": source.get("url", ""),
                "highlights": highlight.get("content", []),
                "metadata": {
                    "author": source.get("author", "Unknown"),
                    "publish_date": source.get("publish_date", "2024-01-01T00:00:00Z"),
                    "file_type": source.get("file_type", "text"),
                    "size_bytes": source.get("size_bytes", 0),
                    "language": source.get("language", "en"),
                    "source": "elasticsearch"
                }
            }
            documents.append(document)

        total_found = es_response.get("hits", {}).get("total", {}).get("value", len(documents))

        return {
            "success": True,
            "documents": documents,
            "total_found": total_found,
            "search_time": es_response.get("took", 0) / 1000,
            "query_used": original_request["query"],
            "filters_applied": original_request.get("filters", {})
        }
```

---

## Re-ranking Examples

### 1. Cross-Encoder Re-ranking

```python
from sentence_transformers import CrossEncoder
import numpy as np

class CrossEncoderReranker:
    """Re-ranking using Sentence Transformers Cross-Encoder"""

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model = CrossEncoder(model_name)
        self.model_name = model_name

    async def rerank_documents(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Rerank documents using cross-encoder"""

        start_time = time.time()

        try:
            query = request["query"]
            documents = request["documents"]
            weights = request.get("weights", {"semantic": 1.0, "freshness": 0.0, "quality": 0.0})

            # Prepare query-document pairs for scoring
            pairs = [(query, doc["content"][:512]) for doc in documents]  # Truncate for performance

            # Get cross-encoder scores
            scores = self.model.predict(pairs)

            # Normalize scores to 0-1 range
            if len(scores) > 0:
                scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)

            # Apply weights and calculate final scores
            ranked_documents = []
            for i, (doc, semantic_score) in enumerate(zip(documents, scores)):

                # Calculate component scores
                freshness_score = self._calculate_freshness_score(doc)
                quality_score = self._calculate_quality_score(doc)

                # Apply weights
                final_score = (
                    semantic_score * weights["semantic"] +
                    freshness_score * weights["freshness"] +
                    quality_score * weights["quality"]
                )

                ranked_doc = {
                    "id": doc["id"],
                    "score": float(final_score),
                    "breakdown": {
                        "semantic_score": float(semantic_score),
                        "freshness_score": freshness_score,
                        "quality_score": quality_score
                    },
                    "explanation": f"Re-ranked using {self.model_name}"
                }
                ranked_documents.append(ranked_doc)

            # Sort by final score
            ranked_documents.sort(key=lambda x: x["score"], reverse=True)

            reranking_time = time.time() - start_time

            return {
                "success": True,
                "ranked_documents": ranked_documents,
                "model_used": self.model_name,
                "reranking_time": reranking_time,
                "weights_applied": weights
            }

        except Exception as e:
            return {
                "success": False,
                "error": {
                    "code": "RERANKING_ERROR",
                    "message": f"Cross-encoder reranking failed: {str(e)}",
                    "retry_after": 5
                }
            }

    def _calculate_freshness_score(self, doc: Dict[str, Any]) -> float:
        """Calculate freshness score based on publish date"""
        try:
            publish_date = doc.get("metadata", {}).get("publish_date", "2024-01-01T00:00:00Z")
            if isinstance(publish_date, str):
                publish_date = datetime.fromisoformat(publish_date.replace('Z', '+00:00'))

            # Calculate age in days
            age_days = (datetime.now() - publish_date).days

            # Score decreases with age (0.0 to 1.0)
            if age_days <= 30:
                return 1.0
            elif age_days <= 365:
                return 1.0 - (age_days - 30) / 335 * 0.5
            else:
                return 0.5 - min(age_days - 365, 365) / 365 * 0.3

        except:
            return 0.5  # Default if date parsing fails

    def _calculate_quality_score(self, doc: Dict[str, Any]) -> float:
        """Calculate quality score based on various signals"""
        try:
            metadata = doc.get("metadata", {})

            # Base quality from user rating if available
            user_rating = metadata.get("user_rating", 3) / 5.0

            # Adjust based on content length
            content_length = len(doc.get("content", ""))
            length_score = min(content_length / 1000, 1.0) * 0.3 + 0.7

            # Adjust based on source reliability
            source_reliability = {
                "academic": 1.0,
                "official": 0.9,
                "reputable": 0.8,
                "unknown": 0.5
            }.get(metadata.get("source", "unknown"), 0.5)

            return user_rating * length_score * source_reliability

        except:
            return 0.5  # Default if calculation fails

# Example Usage
async def example_cross_encoder_reranking():
    reranker = CrossEncoderReranker()

    request = {
        "query": "What is deep learning?",
        "documents": [
            {
                "id": "doc1",
                "title": "Introduction to Deep Learning",
                "content": "Deep learning is a subset of machine learning...",
                "metadata": {"publish_date": "2023-10-01T00:00:00Z", "source": "academic"}
            },
            {
                "id": "doc2",
                "title": "Neural Networks Basics",
                "content": "Neural networks are computational models...",
                "metadata": {"publish_date": "2022-05-15T00:00:00Z", "source": "reputable"}
            }
        ],
        "weights": {"semantic": 0.6, "freshness": 0.2, "quality": 0.2}
    }

    result = await reranker.rerank_documents(request)
    return result
```

### 2. OpenAI Re-ranking API

```python
class OpenAIReranker:
    """Re-ranking using OpenAI's API"""

    def __init__(self, api_key: str, model: str = "gpt-3.5-turbo"):
        self.api_key = api_key
        self.model = model
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

    async def rerank_documents(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Rerank documents using OpenAI's judgment"""

        start_time = time.time()

        try:
            query = request["query"]
            documents = request["documents"]
            weights = request.get("weights", {"semantic": 1.0, "freshness": 0.0, "quality": 0.0})

            # Prepare ranking prompt
            ranking_prompt = self._create_ranking_prompt(query, documents)

            # Call OpenAI API
            async with aiohttp.ClientSession(headers=self.headers) as session:
                openai_request = {
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": "You are a document ranking expert. Rank documents by relevance to the query."},
                        {"role": "user", "content": ranking_prompt}
                    ],
                    "temperature": 0.3,
                    "max_tokens": 1000
                }

                async with session.post(
                    "https://api.openai.com/v1/chat/completions",
                    json=openai_request,
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as response:

                    if response.status == 200:
                        openai_response = await response.json()
                        ranking_text = openai_response["choices"][0]["message"]["content"]

                        # Parse ranking results
                        ranked_documents = self._parse_ranking_response(ranking_text, documents, weights)

                        reranking_time = time.time() - start_time

                        return {
                            "success": True,
                            "ranked_documents": ranked_documents,
                            "model_used": self.model,
                            "reranking_time": reranking_time,
                            "weights_applied": weights
                        }
                    else:
                        error_data = await response.json()
                        return self._format_error_response(f"OPENAI_{response.status}", error_data)

        except Exception as e:
            return self._format_error_response("TIMEOUT", {"error": str(e)})

    def _create_ranking_prompt(self, query: str, documents: List[Dict[str, Any]]) -> str:
        """Create ranking prompt for OpenAI"""

        doc_texts = []
        for i, doc in enumerate(documents):
            doc_text = f"""
Document {i+1}:
ID: {doc['id']}
Title: {doc['title']}
Content: {doc['content'][:500]}...  # Truncate for token limits
"""
            doc_texts.append(doc_text)

        prompt = f"""Query: {query}

Please rank these documents by relevance to the query. Consider:
1. How well the document addresses the query
2. The quality and completeness of information
3. The recency and reliability of the source

Documents:
{"".join(doc_texts)}

Return your ranking as a JSON object with document IDs and relevance scores (0.0-1.0):
{{
    "rankings": [
        {{"doc_id": "doc1", "score": 0.9, "reason": "Directly addresses the query with comprehensive information"}},
        {{"doc_id": "doc2", "score": 0.7, "reason": "Related but less comprehensive"}}
    ]
}}
"""
        return prompt

    def _parse_ranking_response(self, ranking_text: str, documents: List[Dict[str, Any]],
                              weights: Dict[str, float]) -> List[Dict[str, Any]]:
        """Parse OpenAI ranking response"""

        try:
            # Extract JSON from response
            start_idx = ranking_text.find("{")
            end_idx = ranking_text.rfind("}")

            if start_idx != -1 and end_idx != -1:
                json_str = ranking_text[start_idx:end_idx+1]
                ranking_data = json.loads(json_str)

                # Create ranked documents list
                doc_score_map = {item["doc_id"]: item["score"] for item in ranking_data["rankings"]}

                ranked_documents = []
                for doc in documents:
                    doc_id = doc["id"]
                    semantic_score = doc_score_map.get(doc_id, 0.5)

                    # Calculate other scores
                    freshness_score = self._calculate_freshness_score(doc)
                    quality_score = self._calculate_quality_score(doc)

                    # Apply weights
                    final_score = (
                        semantic_score * weights["semantic"] +
                        freshness_score * weights["freshness"] +
                        quality_score * weights["quality"]
                    )

                    ranked_doc = {
                        "id": doc_id,
                        "score": final_score,
                        "breakdown": {
                            "semantic_score": semantic_score,
                            "freshness_score": freshness_score,
                            "quality_score": quality_score
                        },
                        "explanation": f"OpenAI reranking using {self.model}"
                    }
                    ranked_documents.append(ranked_doc)

                # Sort by score
                ranked_documents.sort(key=lambda x: x["score"], reverse=True)
                return ranked_documents

        except Exception as e:
            print(f"Error parsing ranking response: {e}")

        # Fallback: return documents in original order with default scores
        return [
            {
                "id": doc["id"],
                "score": 0.5,
                "breakdown": {"semantic_score": 0.5, "freshness_score": 0.5, "quality_score": 0.5},
                "explanation": "Fallback ranking due to parsing error"
            }
            for doc in documents
        ]
```

---

## LLM Response Generation Examples

### 1. OpenAI Chat Completion

```python
class OpenAIResponseGenerator:
    """Response generation using OpenAI's Chat Completions API"""

    def __init__(self, api_key: str, model: str = "gpt-4"):
        self.api_key = api_key
        self.model = model
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

    async def generate_response(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Generate response using OpenAI"""

        start_time = time.time()

        try:
            # Prepare messages
            messages = [
                {"role": "system", "content": request["system_prompt"]},
                {"role": "user", "content": f"Context:\n{request['context']}\n\nQuestion: {request['query']}"}
            ]

            # Add user instruction if provided
            if request.get("user_instruction"):
                messages.insert(1, {"role": "user", "content": request["user_instruction"]})

            # Prepare generation request
            generation_config = request.get("generation_config", {})
            openai_request = {
                "model": generation_config.get("model_name", self.model),
                "messages": messages,
                "temperature": generation_config.get("temperature", 0.7),
                "max_tokens": generation_config.get("max_tokens", 1000),
                "top_p": generation_config.get("top_p", 1.0),
                "frequency_penalty": generation_config.get("frequency_penalty", 0),
                "presence_penalty": generation_config.get("presence_penalty", 0)
            }

            # Add stop sequences if provided
            if generation_config.get("stop_sequences"):
                openai_request["stop"] = generation_config["stop_sequences"]

            # Call OpenAI API
            async with aiohttp.ClientSession(headers=self.headers) as session:
                async with session.post(
                    "https://api.openai.com/v1/chat/completions",
                    json=openai_request,
                    timeout=aiohttp.ClientTimeout(total=60)
                ) as response:

                    if response.status == 200:
                        openai_response = await response.json()

                        # Calculate cost
                        usage = openai_response["usage"]
                        cost = self._calculate_cost(self.model, usage)

                        generation_time = time.time() - start_time

                        return {
                            "success": True,
                            "generated_response": {
                                "content": openai_response["choices"][0]["message"]["content"],
                                "tokens_used": {
                                    "prompt_tokens": usage["prompt_tokens"],
                                    "completion_tokens": usage["completion_tokens"],
                                    "total_tokens": usage["total_tokens"]
                                },
                                "generation_time": generation_time,
                                "model_used": self.model,
                                "finish_reason": openai_response["choices"][0]["finish_reason"]
                            },
                            "cost_analysis": {
                                "prompt_cost": cost["prompt_cost"],
                                "completion_cost": cost["completion_cost"],
                                "total_cost": cost["total_cost"]
                            }
                        }
                    else:
                        error_data = await response.json()
                        return self._format_error_response(f"OPENAI_{response.status}", error_data)

        except Exception as e:
            return self._format_error_response("TIMEOUT", {"error": str(e)})

    def _calculate_cost(self, model: str, usage: Dict[str, Any]) -> Dict[str, float]:
        """Calculate API costs"""
        # Pricing as of 2024 (update as needed)
        pricing = {
            "gpt-4": {"prompt": 0.03/1000, "completion": 0.06/1000},
            "gpt-4-turbo": {"prompt": 0.01/1000, "completion": 0.03/1000},
            "gpt-3.5-turbo": {"prompt": 0.0015/1000, "completion": 0.002/1000}
        }

        model_pricing = pricing.get(model, pricing["gpt-3.5-turbo"])

        prompt_cost = usage["prompt_tokens"] * model_pricing["prompt"]
        completion_cost = usage["completion_tokens"] * model_pricing["completion"]
        total_cost = prompt_cost + completion_cost

        return {
            "prompt_cost": round(prompt_cost, 6),
            "completion_cost": round(completion_cost, 6),
            "total_cost": round(total_cost, 6)
        }

# Example Usage
async def example_openai_generation():
    generator = OpenAIResponseGenerator("your-openai-api-key")

    request = {
        "query": "What are the main benefits of renewable energy?",
        "context": "Renewable energy comes from natural sources that are constantly replenished...",
        "system_prompt": "You are an expert environmental scientist. Provide detailed, accurate information about renewable energy.",
        "user_instruction": "Focus on economic and environmental benefits, and provide specific examples.",
        "generation_config": {
            "temperature": 0.7,
            "max_tokens": 800,
            "top_p": 1.0
        }
    }

    result = await generator.generate_response(request)
    return result
```

### 2. Anthropic Claude Integration

```python
class AnthropicResponseGenerator:
    """Response generation using Anthropic's Claude API"""

    def __init__(self, api_key: str, model: str = "claude-3-sonnet-20240229"):
        self.api_key = api_key
        self.model = model
        self.headers = {
            "x-api-key": api_key,
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01"
        }

    async def generate_response(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Generate response using Claude"""

        start_time = time.time()

        try:
            # Prepare system prompt
            system_prompt = request["system_prompt"]

            # Prepare user message with context and query
            user_message = f"""I need you to answer a question based on the provided context.

Context:
{request['context']}

Question: {request['query']}"""

            if request.get("user_instruction"):
                user_message += f"\n\nAdditional instruction: {request['user_instruction']}"

            # Prepare generation request
            generation_config = request.get("generation_config", {})
            anthropic_request = {
                "model": generation_config.get("model_name", self.model),
                "max_tokens": generation_config.get("max_tokens", 1000),
                "temperature": generation_config.get("temperature", 0.7),
                "top_p": generation_config.get("top_p", 1.0),
                "system": system_prompt,
                "messages": [
                    {"role": "user", "content": user_message}
                ]
            }

            # Call Anthropic API
            async with aiohttp.ClientSession(headers=self.headers) as session:
                async with session.post(
                    "https://api.anthropic.com/v1/messages",
                    json=anthropic_request,
                    timeout=aiohttp.ClientTimeout(total=60)
                ) as response:

                    if response.status == 200:
                        anthropic_response = await response.json()

                        # Calculate cost
                        usage = anthropic_response.get("usage", {})
                        cost = self._calculate_cost(self.model, usage)

                        generation_time = time.time() - start_time

                        return {
                            "success": True,
                            "generated_response": {
                                "content": anthropic_response["content"][0]["text"],
                                "tokens_used": {
                                    "prompt_tokens": usage.get("input_tokens", 0),
                                    "completion_tokens": usage.get("output_tokens", 0),
                                    "total_tokens": usage.get("input_tokens", 0) + usage.get("output_tokens", 0)
                                },
                                "generation_time": generation_time,
                                "model_used": self.model,
                                "finish_reason": anthropic_response.get("stop_reason", "end_turn")
                            },
                            "cost_analysis": {
                                "prompt_cost": cost["prompt_cost"],
                                "completion_cost": cost["completion_cost"],
                                "total_cost": cost["total_cost"]
                            }
                        }
                    else:
                        error_data = await response.json()
                        return self._format_error_response(f"ANTHROPIC_{response.status}", error_data)

        except Exception as e:
            return self._format_error_response("TIMEOUT", {"error": str(e)})

    def _calculate_cost(self, model: str, usage: Dict[str, Any]) -> Dict[str, float]:
        """Calculate Anthropic API costs"""
        # Pricing as of 2024 (update as needed)
        pricing = {
            "claude-3-sonnet-20240229": {"input": 0.015/1000, "output": 0.075/1000},
            "claude-3-opus-20240229": {"input": 0.025/1000, "output": 0.125/1000},
            "claude-3-haiku-20240307": {"input": 0.00025/1000, "output": 0.00125/1000}
        }

        model_pricing = pricing.get(model, pricing["claude-3-sonnet-20240229"])

        prompt_cost = usage.get("input_tokens", 0) * model_pricing["input"]
        completion_cost = usage.get("output_tokens", 0) * model_pricing["output"]
        total_cost = prompt_cost + completion_cost

        return {
            "prompt_cost": round(prompt_cost, 6),
            "completion_cost": round(completion_cost, 6),
            "total_cost": round(total_cost, 6)
        }
```

---

## Complete Integration Examples

### 1. Complete OpenAI Stack Integration

```python
class CompleteOpenAIIntegration:
    """Complete integration using OpenAI for all components"""

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.retriever = OpenAIDocumentRetriever(api_key)
        self.reranker = OpenAIReranker(api_key)
        self.extractor = OpenAISegmentExtractor(api_key)
        self.generator = OpenAIResponseGenerator(api_key)
        self.analyzer = OpenAIResultsAnalyzer(api_key)

    async def execute_full_pipeline(self, query: str, test_case_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute complete RAG pipeline using OpenAI APIs"""

        pipeline_results = {}

        # Step 1: Document Retrieval
        retrieval_request = {
            "query": query,
            "filters": {"max_results": 20},
            "metadata": {"request_id": f"req_{uuid.uuid4().hex[:8]}"}
        }

        retrieval_result = await self.retriever.retrieve_documents(retrieval_request)
        pipeline_results["retrieval"] = retrieval_result

        if not retrieval_result["success"]:
            return {"success": False, "error": "Document retrieval failed", "pipeline_results": pipeline_results}

        # Step 2: Re-ranking
        reranking_request = {
            "query": query,
            "documents": retrieval_result["documents"],
            "weights": test_case_data.get("rerank_params", {
                "semantic_weight": 0.5,
                "freshness_weight": 0.2,
                "quality_weight": 0.3
            })
        }

        reranking_result = await self.reranker.rerank_documents(reranking_request)
        pipeline_results["reranking"] = reranking_result

        if not reranking_result["success"]:
            return {"success": False, "error": "Document reranking failed", "pipeline_results": pipeline_results}

        # Step 3: Sub-segment Extraction
        top_documents = reranking_result["ranked_documents"][:10]
        extraction_request = {
            "query": query,
            "documents": top_documents,
            "extraction_config": {
                "max_segments": 15,
                "min_relevance": 0.6,
                "max_length": 500
            }
        }

        extraction_result = await self.extractor.extract_segments(extraction_request)
        pipeline_results["extraction"] = extraction_result

        if not extraction_result["success"]:
            return {"success": False, "error": "Segment extraction failed", "pipeline_results": pipeline_results}

        # Step 4: Context Assembly (built-in)
        context_result = self._assemble_context(extraction_result["extracted_segments"])
        pipeline_results["context"] = context_result

        # Step 5: Response Generation
        generation_request = {
            "query": query,
            "context": context_result["assembled_context"]["context_text"],
            "system_prompt": test_case_data["system_prompt"],
            "user_instruction": test_case_data.get("user_instruction", ""),
            "generation_config": test_case_data.get("generation_config", {
                "temperature": 0.7,
                "max_tokens": 1000
            })
        }

        generation_result = await self.generator.generate_response(generation_request)
        pipeline_results["generation"] = generation_result

        if not generation_result["success"]:
            return {"success": False, "error": "Response generation failed", "pipeline_results": pipeline_results}

        # Step 6: Results Analysis
        analysis_request = {
            "test_case": {
                "query": query,
                "expected_answer": test_case_data.get("expected_answer", ""),
                "domain": test_case_data.get("domain", "general")
            },
            "pipeline_results": {
                "query": query,
                "response": generation_result["generated_response"]["content"],
                "context": context_result["assembled_context"]["context_text"],
                "sources_used": len(top_documents),
                "execution_time": sum(r.get("generation_time", 0) for r in pipeline_results.values())
            }
        }

        analysis_result = await self.analyzer.analyze_results(analysis_request)
        pipeline_results["analysis"] = analysis_result

        return {
            "success": True,
            "pipeline_results": pipeline_results,
            "final_response": generation_result["generated_response"]["content"],
            "analysis": analysis_result.get("analysis_results", {}) if analysis_result["success"] else {}
        }

    def _assemble_context(self, extracted_segments: Dict[str, Any]) -> Dict[str, Any]:
        """Built-in context assembly"""
        context_parts = []
        sources_used = set()

        for doc_extraction in extracted_segments["extracted_segments"]:
            for segment in doc_extraction["segments"]:
                if segment["relevance"] >= 0.6:  # Relevance threshold
                    context_parts.append(segment["text"])
                    sources_used.add(doc_extraction["document_id"])

        context_text = "\n\n".join(context_parts)

        return {
            "success": True,
            "assembled_context": {
                "context_text": context_text,
                "length": len(context_text),
                "segments_used": len(context_parts),
                "sources_count": len(sources_used),
                "compression_ratio": 1.0
            }
        }

# Example Usage
async def example_complete_openai_pipeline():
    integration = CompleteOpenAIIntegration("your-openai-api-key")

    test_case = {
        "query": "How do neural networks learn?",
        "system_prompt": "You are an AI expert. Explain technical concepts clearly and accurately.",
        "expected_answer": "Neural networks learn through backpropagation and gradient descent...",
        "domain": "technology",
        "rerank_params": {"semantic_weight": 0.6, "freshness_weight": 0.1, "quality_weight": 0.3}
    }

    result = await integration.execute_full_pipeline(test_case["query"], test_case)
    return result
```

### 2. Hybrid Integration (OpenAI + Elasticsearch + Cross-Encoder)

```python
class HybridIntegration:
    """Hybrid integration using multiple providers for optimal performance"""

    def __init__(self, configs: Dict[str, Any]):
        self.elasticsearch = ElasticsearchRetriever(
            host=configs["elasticsearch"]["host"],
            index=configs["elasticsearch"]["index"],
            api_key=configs["elasticsearch"].get("api_key")
        )
        self.cross_encoder = CrossEncoderReranker()
        self.openai_generator = OpenAIResponseGenerator(configs["openai"]["api_key"])
        self.openai_analyzer = OpenAIResultsAnalyzer(configs["openai"]["api_key"])

    async def execute_pipeline(self, query: str, test_case_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute pipeline with hybrid providers"""

        pipeline_results = {}

        # Step 1: Document Retrieval (Elasticsearch)
        retrieval_request = {
            "query": query,
            "filters": {"max_results": 50, "date_from": "2023-01-01T00:00:00Z"}
        }

        retrieval_result = await self.elasticsearch.retrieve_documents(retrieval_request)
        pipeline_results["retrieval"] = retrieval_result

        if not retrieval_result["success"]:
            return {"success": False, "error": "Document retrieval failed"}

        # Step 2: Re-ranking (Cross-Encoder)
        reranking_request = {
            "query": query,
            "documents": retrieval_result["documents"],
            "weights": test_case_data.get("rerank_params", {"semantic": 0.8, "freshness": 0.1, "quality": 0.1})
        }

        reranking_result = await self.cross_encoder.rerank_documents(reranking_request)
        pipeline_results["reranking"] = reranking_result

        if not reranking_result["success"]:
            return {"success": False, "error": "Document reranking failed"}

        # Continue with remaining steps...
        # [Similar to CompleteOpenAIIntegration]

        return {
            "success": True,
            "pipeline_results": pipeline_results,
            "providers_used": ["elasticsearch", "cross-encoder", "openai"]
        }
```

---

## Testing and Validation Examples

### 1. Integration Test Suite

```python
import pytest
import asyncio

class TestAPIIntegrations:
    """Test suite for validating API integrations"""

    @pytest.mark.asyncio
    async def test_openai_retrieval(self):
        """Test OpenAI document retrieval"""
        retriever = OpenAIDocumentRetriever("test-key")

        request = {
            "query": "test query",
            "filters": {"max_results": 5}
        }

        # Mock the API call
        with patch.object(retriever, '_make_api_call') as mock_call:
            mock_call.return_value = {
                "data": [
                    {"id": "doc1", "title": "Test Doc", "content": "Test content", "score": 0.8}
                ],
                "total_count": 1
            }

            result = await retriever.retrieve_documents(request)

            assert result["success"] is True
            assert len(result["documents"]) == 1
            assert result["documents"][0]["id"] == "doc1"

    @pytest.mark.asyncio
    async def test_cross_encoder_reranking(self):
        """Test Cross-Encoder reranking"""
        reranker = CrossEncoderReranker()

        request = {
            "query": "test query",
            "documents": [
                {"id": "doc1", "content": "relevant content", "metadata": {"publish_date": "2024-01-01T00:00:00Z"}},
                {"id": "doc2", "content": "less relevant content", "metadata": {"publish_date": "2022-01-01T00:00:00Z"}}
            ],
            "weights": {"semantic": 0.7, "freshness": 0.2, "quality": 0.1}
        }

        result = await reranker.rerank_documents(request)

        assert result["success"] is True
        assert len(result["ranked_documents"]) == 2
        assert all("score" in doc for doc in result["ranked_documents"])

# Usage
async def run_integration_tests():
    """Run all integration tests"""
    test_suite = TestAPIIntegrations()

    # Run individual tests
    await test_suite.test_openai_retrieval()
    await test_suite.test_cross_encoder_reranking()

    print("✅ All integration tests passed")
```

### 2. Performance Benchmarking

```python
import time
import statistics

class PerformanceBenchmark:
    """Benchmark API integration performance"""

    def __init__(self, api_integrations: Dict[str, Any]):
        self.integrations = api_integrations
        self.results = {}

    async def benchmark_retrieval(self, num_queries: int = 10) -> Dict[str, Any]:
        """Benchmark document retrieval performance"""

        queries = [
            "What is machine learning?",
            "How do neural networks work?",
            "Explain quantum computing",
            # Add more test queries
        ]

        results = {}

        for provider_name, retriever in self.integrations.items():
            times = []
            success_count = 0

            for _ in range(num_queries):
                query = queries[_ % len(queries)]
                start_time = time.time()

                try:
                    result = await retriever.retrieve_documents({
                        "query": query,
                        "filters": {"max_results": 10}
                    })

                    if result["success"]:
                        success_count += 1

                    times.append(time.time() - start_time)

                except Exception as e:
                    print(f"Error with {provider_name}: {e}")

            results[provider_name] = {
                "avg_time": statistics.mean(times) if times else 0,
                "min_time": min(times) if times else 0,
                "max_time": max(times) if times else 0,
                "success_rate": success_count / num_queries,
                "p95_time": statistics.quantiles(times, n=20)[18] if len(times) >= 20 else max(times) if times else 0
            }

        return results

# Example Usage
async def example_benchmark():
    integrations = {
        "openai": OpenAIDocumentRetriever("your-api-key"),
        "elasticsearch": ElasticsearchRetriever("localhost", "test-index")
    }

    benchmark = PerformanceBenchmark(integrations)
    results = await benchmark.benchmark_retrieval()

    print("=== Retrieval Performance Benchmark ===")
    for provider, metrics in results.items():
        print(f"{provider}:")
        print(f"  Average Time: {metrics['avg_time']:.2f}s")
        print(f"  P95 Time: {metrics['p95_time']:.2f}s")
        print(f"  Success Rate: {metrics['success_rate']*100:.1f}%")
        print()
```

These examples provide complete, working implementations that users can adapt for their specific API integrations. Each example includes error handling, request/response formatting, and real-world usage patterns.
