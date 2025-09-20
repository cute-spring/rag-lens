# API Interface Contracts for RAG Pipeline Integration

## Overview
This document defines the standardized contracts for integrating external APIs with the RAG Pipeline Testing tool. Each pipeline step has a clearly defined interface that API implementations must follow.

## 1. Document Retrieval Interface Contract

### Request Format
```python
{
    "query": str,           # User's search query
    "filters": {
        "date_from": str,   # ISO format date (optional)
        "date_to": str,     # ISO format date (optional)
        "document_types": List[str],  # e.g., ["pdf", "doc", "webpage"]
        "sources": List[str],        # e.g., ["internal", "external"]
        "max_results": int    # Default: 50
    },
    "metadata": {
        "user_id": str,      # Optional user identifier
        "session_id": str,   # Optional session identifier
        "request_id": str    # Unique request identifier
    }
}
```

### Required Response Format
```python
{
    "success": bool,         # Must be true if successful
    "documents": [
        {
            "id": str,       # Unique document identifier
            "title": str,    # Document title
            "content": str,  # Full document content
            "url": str,      # Source URL (optional)
            "score": float,  # Initial relevance score (0.0-1.0)
            "metadata": {
                "author": str,           # Document author
                "publish_date": str,     # ISO format date
                "file_type": str,        # pdf, doc, txt, etc.
                "size_bytes": int,       # Document size
                "language": str,         # en, es, fr, etc.
                "source": str           # internal, external, web
            }
        }
    ],
    "total_found": int,       # Total documents matching query
    "search_time": float,     # Search execution time in seconds
    "query_used": str,       # Echo back the query
    "filters_applied": dict   # Echo back applied filters
}
```

### Error Response Format
```python
{
    "success": false,
    "error": {
        "code": str,         # INVALID_QUERY, TIMEOUT, RATE_LIMITED, etc.
        "message": str,      # Human-readable error message
        "retry_after": int,  # Seconds to wait before retry (for rate limits)
        "details": dict      # Additional error context
    }
}
```

## 2. Re-ranking Interface Contract

### Request Format
```python
{
    "query": str,            # Original user query
    "documents": List[dict], # Documents from retrieval step
    "weights": {
        "semantic": float,   # Weight for semantic similarity (0.0-1.0)
        "freshness": float,  # Weight for document freshness (0.0-1.0)
        "quality": float    # Weight for document quality (0.0-1.0)
    },
    "model_config": {
        "model_name": str,   # Model to use for reranking
        "threshold": float  # Minimum score threshold (0.0-1.0)
    }
}
```

### Required Response Format
```python
{
    "success": bool,
    "ranked_documents": [
        {
            "id": str,       # Document ID (must match input)
            "score": float,   # Final reranking score (0.0-1.0)
            "breakdown": {    # Optional score breakdown
                "semantic_score": float,
                "freshness_score": float,
                "quality_score": float
            },
            "explanation": str # Optional explanation of scoring
        }
    ],
    "model_used": str,       # Model that was used
    "reranking_time": float,# Time taken for reranking
    "weights_applied": dict # Weights that were applied
}
```

## 3. Sub-segment Extraction Interface Contract

### Request Format
```python
{
    "query": str,            # Original user query
    "documents": List[dict], # Top-ranked documents
    "extraction_config": {
        "max_segments": int,     # Maximum segments per document
        "min_relevance": float,  # Minimum relevance threshold
        "max_length": int,       # Maximum characters per segment
        "overlap": int,          # Character overlap between segments
        "include_context": bool  # Include surrounding context
    },
    "model_config": {
        "model_name": str,      # Model to use for extraction
        "temperature": float    # Creativity parameter (0.0-1.0)
    }
}
```

### Required Response Format
```python
{
    "success": bool,
    "extracted_segments": [
        {
            "document_id": str,
            "segments": [
                {
                    "id": str,            # Unique segment ID
                    "text": str,          # Extracted segment text
                    "start_pos": int,     # Start position in original
                    "end_pos": int,       # End position in original
                    "relevance": float,    # Segment relevance score (0.0-1.0)
                    "context": str,       # Surrounding context (if requested)
                    "confidence": float   # Extraction confidence (0.0-1.0)
                }
            ]
        }
    ],
    "extraction_stats": {
        "total_segments": int,
        "average_relevance": float,
        "extraction_time": float
    }
}
```

## 4. Context Assembly Interface Contract

### Request Format
```python
{
    "query": str,            # Original user query
    "segments": List[dict],   # Extracted segments
    "assembly_config": {
        "max_length": int,       # Maximum context length
        "compression": bool,     # Enable context compression
        "deduplication": bool,   # Remove duplicate content
        "ordering": str,         # "relevance", "chronological", "source"
        "include_metadata": bool # Include source metadata
    }
}
```

### Required Response Format
```python
{
    "success": bool,
    "assembled_context": {
        "context_text": str,    # Final assembled context
        "length": int,          # Character count
        "segments_used": int,   # Number of segments included
        "sources_count": int,  # Number of unique sources
        "compression_ratio": float # If compression enabled
    },
    "assembly_metadata": {
        "included_segments": List[str],  # Segment IDs that were included
        "excluded_segments": List[str],  # Segment IDs that were excluded
        "ordering_method": str,
        "assembly_time": float
    }
}
```

## 5. Response Generation Interface Contract

### Request Format
```python
{
    "query": str,            # Original user query
    "context": str,         # Assembled context
    "system_prompt": str,    # System prompt for the LLM
    "user_instruction": str, # Optional user instruction
    "generation_config": {
        "model_name": str,      # LLM model to use
        "temperature": float,   # Temperature (0.0-2.0)
        "max_tokens": int,      # Maximum tokens to generate
        "top_p": float,        # Nucleus sampling (0.0-1.0)
        "stop_sequences": List[str]  # Stop sequences
    },
    "metadata": {
        "session_id": str,     # Optional session identifier
        "user_id": str         # Optional user identifier
    }
}
```

### Required Response Format
```python
{
    "success": bool,
    "generated_response": {
        "content": str,        # Generated response text
        "tokens_used": {
            "prompt_tokens": int,
            "completion_tokens": int,
            "total_tokens": int
        },
        "generation_time": float,
        "model_used": str,
        "finish_reason": str   # "stop", "length", "content_filter"
    },
    "cost_analysis": {
        "prompt_cost": float,    # Cost in USD
        "completion_cost": float, # Cost in USD
        "total_cost": float      # Total cost in USD
    }
}
```

## 6. Results Analysis Interface Contract

### Request Format
```python
{
    "test_case": {
        "query": str,           # Original query
        "expected_answer": str, # Expected answer (if available)
        "domain": str,          # Domain of the query
        "complexity": str       # simple, medium, complex
    },
    "pipeline_results": {
        "query": str,           # Original query
        "response": str,        # Generated response
        "context": str,         # Context used
        "sources_used": int,    # Number of sources
        "execution_time": float # Total execution time
    },
    "analysis_config": {
        "calculate_relevance": bool,
        "calculate_quality": bool,
        "generate_insights": bool,
        "include_suggestions": bool
    }
}
```

### Required Response Format
```python
{
    "success": bool,
    "analysis_results": {
        "relevance_metrics": {
            "context_relevance": float,    # 0.0-1.0
            "query_coverage": float,       # 0.0-1.0
            "source_utilization": float    # 0.0-1.0
        },
        "quality_metrics": {
            "accuracy": float,            # 0.0-1.0
            "completeness": float,        # 0.0-1.0
            "coherence": float,           # 0.0-1.0
            "helpfulness": float          # 0.0-1.0
        },
        "performance_metrics": {
            "response_time": float,       # in seconds
            "context_efficiency": float,  # 0.0-1.0
            "noise_reduction": float      # 0.0-1.0
        }
    },
    "insights": [
        {
            "type": str,         # "strength", "weakness", "suggestion"
            "category": str,     # "relevance", "quality", "performance"
            "description": str,  # Insight description
            "priority": str,     # "low", "medium", "high"
            "actionable": bool   # Whether this is actionable
        }
    ],
    "suggestions": [
        {
            "aspect": str,       # What to improve
            "current_value": float,
            "target_value": float,
            "recommendation": str # Specific recommendation
        }
    ]
}
```

## Implementation Base Class

```python
from abc import ABC, abstractmethod
from typing import Dict, Any, List

class BaseAPIProvider(ABC):
    """Base class that all API providers must implement"""

    @abstractmethod
    async def retrieve_documents(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Implement document retrieval"""
        pass

    @abstractmethod
    async def rerank_documents(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Implement document reranking"""
        pass

    @abstractmethod
    async def extract_segments(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Implement sub-segment extraction"""
        pass

    @abstractmethod
    async def assemble_context(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Implement context assembly"""
        pass

    @abstractmethod
    async def generate_response(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Implement response generation"""
        pass

    @abstractmethod
    async def analyze_results(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Implement results analysis"""
        pass

    def validate_request(self, request: Dict[str, Any], required_fields: List[str]) -> bool:
        """Validate request has all required fields"""
        return all(field in request for field in required_fields)

    def format_success_response(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Format a successful response"""
        return {"success": True, **data}

    def format_error_response(self, error_code: str, message: str, details: Dict[str, Any] = None) -> Dict[str, Any]:
        """Format an error response"""
        return {
            "success": False,
            "error": {
                "code": error_code,
                "message": message,
                "details": details or {}
            }
        }
```

## Integration Testing Template

```python
class APIIntegrationTester:
    """Test template for validating API implementations"""

    def __init__(self, api_provider: BaseAPIProvider):
        self.api_provider = api_provider
        self.test_results = []

    async def test_document_retrieval(self) -> Dict[str, Any]:
        """Test document retrieval integration"""
        test_request = {
            "query": "What is machine learning?",
            "filters": {"max_results": 5},
            "metadata": {"request_id": "test_001"}
        }

        try:
            response = await self.api_provider.retrieve_documents(test_request)

            # Validate response structure
            required_fields = ["success", "documents", "total_found"]
            if not self.api_provider.validate_request(response, required_fields):
                return {"passed": False, "error": "Missing required response fields"}

            if not response["success"]:
                return {"passed": False, "error": "API returned unsuccessful response"}

            if len(response["documents"]) == 0:
                return {"passed": False, "error": "No documents returned"}

            # Validate document structure
            doc = response["documents"][0]
            doc_fields = ["id", "title", "content", "score"]
            if not all(field in doc for field in doc_fields):
                return {"passed": False, "error": "Document missing required fields"}

            return {"passed": True, "documents_found": len(response["documents"])}

        except Exception as e:
            return {"passed": False, "error": str(e)}

    async def run_all_tests(self) -> Dict[str, Any]:
        """Run all integration tests"""
        test_methods = [
            ("document_retrieval", self.test_document_retrieval),
            # Add other test methods...
        ]

        results = {}
        for test_name, test_method in test_methods:
            results[test_name] = await test_method()

        return results
```

## Quick Start Checklist

For API implementers:

1. **Read the interface contract** for your specific API type
2. **Extend the BaseAPIProvider** class
3. **Implement required methods** with exact request/response formats
4. **Run integration tests** to validate compliance
5. **Test with sample data** provided in the examples
6. **Configure environment variables** using the templates

This contract ensures that any API implementation will work seamlessly with the RAG Pipeline Testing tool.