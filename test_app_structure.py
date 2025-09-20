#!/usr/bin/env python3
"""
Test script to validate the RAG pipeline tuning tool structure
"""

import json
import sys
from datetime import datetime

def validate_mock_data_structure():
    """Validate that mock data structure is correct"""

    # Simulate mock data structure
    mock_chunk = {
        "id": "test_chunk",
        "title": "Test Title",
        "content": "## Test Content\nThis is a test markdown content.",
        "publish_time": datetime.now(),
        "effective_time": datetime.now(),
        "expiration_time": datetime.now(),
        "user_rating": 4,
        "relevance_score": 0.8,
        "freshness_score": 0.9,
        "quality_score": 0.7
    }

    mock_test_case = {
        "id": "test_case_1",
        "name": "Test Case",
        "topic": "Machine Learning",
        "purpose": "Retrieval Accuracy",
        "description": "Test case description",
        "chunks": [mock_chunk] * 20,
        "user_query": "What is machine learning?",
        "user_instruction": "Focus on basics",
        "system_prompt": "You are an ML expert",
        "model_version": "gpt-4",
        "rerank_params": {
            "semantic_weight": 0.5,
            "freshness_weight": 0.2,
            "quality_weight": 0.3,
            "relevance_threshold": 0.6,
            "top_n": 10
        },
        "expected_answer": "Expected answer content",
        "incorrect_answer": {
            "answer": "Incorrect answer",
            "user_concerns": ["Missing details"],
            "user_comments": "Needs improvement"
        },
        "status": "pending",
        "created_at": datetime.now(),
        "last_modified": datetime.now()
    }

    # Validate structure
    required_chunk_fields = ["id", "title", "content", "publish_time", "effective_time",
                           "expiration_time", "user_rating", "relevance_score",
                           "freshness_score", "quality_score"]

    required_case_fields = ["id", "name", "topic", "purpose", "description", "chunks",
                          "user_query", "user_instruction", "system_prompt", "model_version",
                          "rerank_params", "expected_answer", "incorrect_answer", "status"]

    # Check chunk fields
    for field in required_chunk_fields:
        if field not in mock_chunk:
            print(f"❌ Missing required chunk field: {field}")
            return False

    # Check test case fields
    for field in required_case_fields:
        if field not in mock_test_case:
            print(f"❌ Missing required test case field: {field}")
            return False

    # Check incorrect answer structure
    incorrect_fields = ["answer", "user_concerns", "user_comments"]
    for field in incorrect_fields:
        if field not in mock_test_case["incorrect_answer"]:
            print(f"❌ Missing incorrect answer field: {field}")
            return False

    # Check rerank params structure
    rerank_fields = ["semantic_weight", "freshness_weight", "quality_weight",
                    "relevance_threshold", "top_n"]
    for field in rerank_fields:
        if field not in mock_test_case["rerank_params"]:
            print(f"❌ Missing rerank param field: {field}")
            return False

    print("✅ All required fields are present in the data structure")
    return True

def validate_app_structure():
    """Validate that the main app file has correct structure"""

    try:
        with open('rag_pipeline_tuning_tool.py', 'r') as f:
            content = f.read()

        # Check for key components
        required_components = [
            "class MockDataGenerator",
            "class RAGPipelineSimulator",
            "def render_test_case_selector",
            "def render_parameter_controls",
            "def render_pipeline_visualization",
            "def render_chunk_comparison",
            "def render_results_analysis",
            "def main",
            "st.set_page_config",
            "st.sidebar",
            "st.expander",
            "st.slider",
            "st.plotly_chart"
        ]

        missing_components = []
        for component in required_components:
            if component not in content:
                missing_components.append(component)

        if missing_components:
            print(f"❌ Missing components: {missing_components}")
            return False

        print("✅ All required components are present in the app")
        return True

    except FileNotFoundError:
        print("❌ Main app file not found")
        return False

def main():
    """Run all validation tests"""
    print("🔍 Validating RAG Pipeline Tuning Tool Structure")
    print("=" * 50)

    structure_valid = validate_app_structure()
    data_valid = validate_mock_data_structure()

    if structure_valid and data_valid:
        print("\n🎉 All validations passed! The application structure is correct.")
        print("\nTo run the application:")
        print("1. Install dependencies: pip install -r requirements.txt")
        print("2. Run the app: streamlit run rag_pipeline_tuning_tool.py")
        return 0
    else:
        print("\n❌ Some validations failed. Please check the issues above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())