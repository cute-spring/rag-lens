#!/usr/bin/env python3
"""
Test script for Ollama reranking integration.

This script demonstrates how to use the OllamaRerankProvider
to rerank documents using Ollama's RESTful API.
"""

import asyncio
import sys
import os
from typing import List, Dict, Any

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rag_lens.api.providers import OllamaRerankProvider
from rag_lens.config.settings import config


def create_sample_candidates() -> List[Dict[str, Any]]:
    """Create sample candidates for testing."""
    return [
        {
            "id": "doc1",
            "content": "Machine learning is a subset of artificial intelligence that focuses on algorithms.",
            "metadata": {"source": "textbook", "relevance_score": 0.8}
        },
        {
            "id": "doc2", 
            "content": "Deep learning uses neural networks with multiple layers to process data.",
            "metadata": {"source": "research_paper", "relevance_score": 0.7}
        },
        {
            "id": "doc3",
            "content": "Natural language processing enables computers to understand human language.",
            "metadata": {"source": "article", "relevance_score": 0.6}
        },
        {
            "id": "doc4",
            "content": "Computer vision allows machines to interpret and analyze visual information.",
            "metadata": {"source": "blog", "relevance_score": 0.5}
        }
    ]


async def test_ollama_reranking():
    """Test the Ollama reranking functionality."""
    print("🚀 Testing Ollama Reranking Integration")
    print("=" * 50)
    
    # Initialize the provider
    provider_config = {
        "base_url": getattr(config, 'ollama_base_url', 'http://localhost:11434'),
        "model": getattr(config, 'ollama_model', 'llama2'),
        "api_key": getattr(config, 'ollama_api_key', '')
    }
    
    print(f"📡 Connecting to Ollama at: {provider_config['base_url']}")
    print(f"🤖 Using model: {provider_config['model']}")
    
    try:
        provider = OllamaRerankProvider(provider_config)
        
        # Test health check
        print("\n🏥 Performing health check...")
        health_result = await provider.health_check()
        
        if not health_result.success:
            print(f"❌ Health check failed: {health_result.error}")
            return False
            
        print("✅ Health check passed!")
        
        # Test reranking
        print("\n🔄 Testing reranking functionality...")
        
        query = "What is machine learning and how does it work?"
        candidates = create_sample_candidates()
        
        print(f"📝 Query: {query}")
        print(f"📚 Number of candidates: {len(candidates)}")
        
        # Create pipeline step data
        step_data = {
            "query": query,
            "candidates": candidates,
            "parameters": {
                "top_k": 3,
                "threshold": 0.1
            }
        }
        
        result = await provider.execute_pipeline_step(step_data)
        
        if not result.success:
            print(f"❌ Reranking failed: {result.error}")
            return False
            
        print("✅ Reranking completed successfully!")
        
        # Display results
        reranked_candidates = result.data.get("reranked_candidates", [])
        print(f"\n📊 Reranked {len(reranked_candidates)} candidates:")
        
        for i, candidate in enumerate(reranked_candidates, 1):
            score = candidate.get("rerank_score", 0)
            content_preview = candidate.get("content", "")[:80] + "..."
            print(f"  {i}. Score: {score:.4f} - {content_preview}")
            
        return True
        
    except Exception as e:
        print(f"❌ Test failed with exception: {str(e)}")
        return False


def main():
    """Main test function."""
    print("Ollama Reranking Integration Test")
    print("Make sure Ollama is running on your system before running this test.")
    print("You can start Ollama with: ollama serve")
    print()
    
    # Run the async test
    success = asyncio.run(test_ollama_reranking())
    
    if success:
        print("\n🎉 All tests passed! Ollama integration is working correctly.")
        sys.exit(0)
    else:
        print("\n💥 Tests failed. Please check your Ollama setup and configuration.")
        sys.exit(1)


if __name__ == "__main__":
    main()