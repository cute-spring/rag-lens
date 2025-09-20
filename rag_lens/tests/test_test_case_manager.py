"""
Test Case Manager Tests for RAG Lens

Test suite for the test case management system.
"""

import pytest
import tempfile
import os
import json
from datetime import datetime
from unittest.mock import Mock, patch

from rag_lens.core.test_case_manager import TestCaseManager, TestCase, Chunk
from rag_lens.utils.errors import TestCaseManagerError


class TestTestCase:
    """Test cases for TestCase dataclass"""

    def test_test_case_creation(self):
        """Test TestCase creation"""
        test_case = TestCase(
            id="test_001",
            name="Test Case",
            description="Description",
            query="What is AI?",
            system_prompt="You are an AI assistant",
            user_instruction="Answer the question",
            expected_answer="AI is artificial intelligence",
            domain="technology",
            difficulty="medium",
            user_rating=5,
            publish_time=datetime.now(),
            effective_time=datetime.now(),
            expiration_time=datetime.now(),
            chunks=[]
        )

        assert test_case.id == "test_001"
        assert test_case.name == "Test Case"
        assert test_case.user_rating == 5
        assert len(test_case.chunks) == 0

    def test_test_case_to_dict(self):
        """Test TestCase to_dict conversion"""
        test_time = datetime.now()
        test_case = TestCase(
            id="test_001",
            name="Test Case",
            description="Description",
            query="What is AI?",
            system_prompt="You are an AI assistant",
            user_instruction="Answer the question",
            expected_answer="AI is artificial intelligence",
            domain="technology",
            difficulty="medium",
            user_rating=5,
            publish_time=test_time,
            effective_time=test_time,
            expiration_time=test_time,
            chunks=[]
        )

        result = test_case.to_dict()

        assert result["id"] == "test_001"
        assert result["name"] == "Test Case"
        assert result["query"] == "What is AI?"
        assert result["user_rating"] == 5
        assert "chunks" in result


class TestChunk:
    """Test cases for Chunk dataclass"""

    def test_chunk_creation(self):
        """Test Chunk creation"""
        chunk = Chunk(
            id="chunk_001",
            title="Test Chunk",
            content="This is test content",
            user_rating=4,
            publish_time=datetime.now(),
            effective_time=datetime.now(),
            expiration_time=datetime.now()
        )

        assert chunk.id == "chunk_001"
        assert chunk.title == "Test Chunk"
        assert chunk.content == "This is test content"
        assert chunk.user_rating == 4

    def test_chunk_to_dict(self):
        """Test Chunk to_dict conversion"""
        test_time = datetime.now()
        chunk = Chunk(
            id="chunk_001",
            title="Test Chunk",
            content="This is test content",
            user_rating=4,
            publish_time=test_time,
            effective_time=test_time,
            expiration_time=test_time
        )

        result = chunk.to_dict()

        assert result["id"] == "chunk_001"
        assert result["title"] == "Test Chunk"
        assert result["content"] == "This is test content"
        assert result["user_rating"] == 4


class TestTestCaseManager:
    """Test cases for TestCaseManager"""

    def setup_method(self):
        """Setup test environment"""
        self.test_case_manager = TestCaseManager()

    def test_init(self):
        """Test TestCaseManager initialization"""
        manager = TestCaseManager()
        assert manager.test_cases == []
        assert manager.current_source == "static"

    def test_create_test_case(self):
        """Test creating a test case"""
        test_data = {
            "id": "test_001",
            "name": "Test Case",
            "description": "Test description",
            "query": "What is AI?",
            "system_prompt": "You are an AI assistant",
            "user_instruction": "Answer about AI",
            "expected_answer": "Artificial Intelligence",
            "domain": "technology",
            "difficulty": "medium",
            "user_rating": 5
        }

        test_case = self.test_case_manager.create_test_case(test_data)

        assert test_case.id == "test_001"
        assert test_case.name == "Test Case"
        assert test_case.query == "What is AI?"
        assert test_case.user_rating == 5

    def test_create_test_case_missing_required_fields(self):
        """Test creating test case with missing required fields"""
        test_data = {
            "name": "Test Case"
            # Missing required 'id' field
        }

        with pytest.raises(TestCaseManagerError):
            self.test_case_manager.create_test_case(test_data)

    def test_create_chunk(self):
        """Test creating a chunk"""
        chunk_data = {
            "id": "chunk_001",
            "title": "Test Chunk",
            "content": "Test content",
            "user_rating": 4
        }

        chunk = self.test_case_manager.create_chunk(chunk_data)

        assert chunk.id == "chunk_001"
        assert chunk.title == "Test Chunk"
        assert chunk.content == "Test content"
        assert chunk.user_rating == 4

    def test_create_chunk_missing_required_fields(self):
        """Test creating chunk with missing required fields"""
        chunk_data = {
            "title": "Test Chunk"
            # Missing required 'id' field
        }

        with pytest.raises(TestCaseManagerError):
            self.test_case_manager.create_chunk(chunk_data)

    def test_add_test_case(self):
        """Test adding a test case to the manager"""
        test_case = TestCase(
            id="test_001",
            name="Test Case",
            description="Description",
            query="What is AI?",
            system_prompt="You are an AI assistant",
            user_instruction="Answer the question",
            expected_answer="AI is artificial intelligence"
        )

        self.test_case_manager.add_test_case(test_case)
        assert len(self.test_case_manager.test_cases) == 1
        assert self.test_case_manager.test_cases[0].id == "test_001"

    def test_add_duplicate_test_case(self):
        """Test adding duplicate test case"""
        test_case1 = TestCase(
            id="test_001",
            name="Test Case 1",
            description="Description 1",
            query="What is AI?",
            system_prompt="You are an AI assistant",
            user_instruction="Answer the question",
            expected_answer="AI is artificial intelligence"
        )

        test_case2 = TestCase(
            id="test_001",  # Same ID
            name="Test Case 2",
            description="Description 2",
            query="What is ML?",
            system_prompt="You are an ML assistant",
            user_instruction="Answer about ML",
            expected_answer="Machine Learning"
        )

        self.test_case_manager.add_test_case(test_case1)
        with pytest.raises(TestCaseManagerError):
            self.test_case_manager.add_test_case(test_case2)

    def test_get_test_case(self):
        """Test getting a test case by ID"""
        test_case = TestCase(
            id="test_001",
            name="Test Case",
            description="Description",
            query="What is AI?",
            system_prompt="You are an AI assistant",
            user_instruction="Answer the question",
            expected_answer="AI is artificial intelligence"
        )

        self.test_case_manager.add_test_case(test_case)
        retrieved = self.test_case_manager.get_test_case("test_001")

        assert retrieved is not None
        assert retrieved.id == "test_001"
        assert retrieved.name == "Test Case"

    def test_get_nonexistent_test_case(self):
        """Test getting non-existent test case"""
        retrieved = self.test_case_manager.get_test_case("nonexistent")
        assert retrieved is None

    def test_update_test_case(self):
        """Test updating a test case"""
        test_case = TestCase(
            id="test_001",
            name="Test Case",
            description="Description",
            query="What is AI?",
            system_prompt="You are an AI assistant",
            user_instruction="Answer the question",
            expected_answer="AI is artificial intelligence"
        )

        self.test_case_manager.add_test_case(test_case)

        update_data = {
            "name": "Updated Test Case",
            "query": "What is Machine Learning?",
            "user_rating": 4
        }

        updated = self.test_case_manager.update_test_case("test_001", update_data)

        assert updated.name == "Updated Test Case"
        assert updated.query == "What is Machine Learning?"
        assert updated.user_rating == 4

    def test_update_nonexistent_test_case(self):
        """Test updating non-existent test case"""
        update_data = {"name": "Updated Test Case"}

        with pytest.raises(TestCaseManagerError):
            self.test_case_manager.update_test_case("nonexistent", update_data)

    def test_delete_test_case(self):
        """Test deleting a test case"""
        test_case = TestCase(
            id="test_001",
            name="Test Case",
            description="Description",
            query="What is AI?",
            system_prompt="You are an AI assistant",
            user_instruction="Answer the question",
            expected_answer="AI is artificial intelligence"
        )

        self.test_case_manager.add_test_case(test_case)
        assert len(self.test_case_manager.test_cases) == 1

        self.test_case_manager.delete_test_case("test_001")
        assert len(self.test_case_manager.test_cases) == 0

    def test_delete_nonexistent_test_case(self):
        """Test deleting non-existent test case"""
        with pytest.raises(TestCaseManagerError):
            self.test_case_manager.delete_test_case("nonexistent")

    def test_list_test_cases(self):
        """Test listing test cases"""
        test_cases = [
            TestCase(
                id="test_001",
                name="Test Case 1",
                description="Description 1",
                query="What is AI?",
                system_prompt="You are an AI assistant",
                user_instruction="Answer the question",
                expected_answer="AI is artificial intelligence"
            ),
            TestCase(
                id="test_002",
                name="Test Case 2",
                description="Description 2",
                query="What is ML?",
                system_prompt="You are an ML assistant",
                user_instruction="Answer about ML",
                expected_answer="Machine Learning"
            )
        ]

        for test_case in test_cases:
            self.test_case_manager.add_test_case(test_case)

        result = self.test_case_manager.list_test_cases()
        assert len(result) == 2
        assert result[0].id == "test_001"
        assert result[1].id == "test_002"

    def test_search_test_cases(self):
        """Test searching test cases"""
        test_cases = [
            TestCase(
                id="test_001",
                name="AI Test Case",
                description="Description about AI",
                query="What is AI?",
                system_prompt="You are an AI assistant",
                user_instruction="Answer about AI",
                expected_answer="AI is artificial intelligence",
                domain="technology"
            ),
            TestCase(
                id="test_002",
                name="Healthcare Test Case",
                description="Description about healthcare",
                query="What is healthcare?",
                system_prompt="You are a healthcare assistant",
                user_instruction="Answer about healthcare",
                expected_answer="Healthcare is medical care",
                domain="healthcare"
            )
        ]

        for test_case in test_cases:
            self.test_case_manager.add_test_case(test_case)

        # Search by name
        results = self.test_case_manager.search_test_cases("AI")
        assert len(results) == 1
        assert results[0].id == "test_001"

        # Search by domain
        results = self.test_case_manager.search_test_cases("healthcare")
        assert len(results) == 1
        assert results[0].id == "test_002"

        # Search with no results
        results = self.test_case_manager.search_test_cases("nonexistent")
        assert len(results) == 0

    def test_load_test_cases_from_json(self):
        """Test loading test cases from JSON file"""
        test_data = {
            "test_cases_collection": [
                {
                    "id": "test_001",
                    "name": "Test Case 1",
                    "description": "Description 1",
                    "query": "What is AI?",
                    "system_prompt": "You are an AI assistant",
                    "user_instruction": "Answer the question",
                    "expected_answer": "AI is artificial intelligence",
                    "chunks": [
                        {
                            "id": "chunk_001",
                            "title": "Chunk 1",
                            "content": "Content 1",
                            "user_rating": 5
                        }
                    ]
                }
            ]
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(test_data, f)
            temp_file = f.name

        try:
            self.test_case_manager.load_test_cases_from_json(temp_file)
            assert len(self.test_case_manager.test_cases) == 1
            assert self.test_case_manager.test_cases[0].id == "test_001"
            assert len(self.test_case_manager.test_cases[0].chunks) == 1
        finally:
            os.unlink(temp_file)

    def test_load_test_cases_from_invalid_json(self):
        """Test loading test cases from invalid JSON file"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            f.write("invalid json content")
            temp_file = f.name

        try:
            with pytest.raises(TestCaseManagerError):
                self.test_case_manager.load_test_cases_from_json(temp_file)
        finally:
            os.unlink(temp_file)

    def test_save_test_cases_to_json(self):
        """Test saving test cases to JSON file"""
        test_case = TestCase(
            id="test_001",
            name="Test Case",
            description="Description",
            query="What is AI?",
            system_prompt="You are an AI assistant",
            user_instruction="Answer the question",
            expected_answer="AI is artificial intelligence"
        )

        chunk = Chunk(
            id="chunk_001",
            title="Chunk",
            content="Content",
            user_rating=5
        )

        test_case.chunks.append(chunk)
        self.test_case_manager.add_test_case(test_case)

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_file = f.name

        try:
            self.test_case_manager.save_test_cases_to_json(temp_file)

            with open(temp_file, 'r') as f:
                data = json.load(f)

            assert "test_cases_collection" in data
            assert len(data["test_cases_collection"]) == 1
            assert data["test_cases_collection"][0]["id"] == "test_001"
        finally:
            os.unlink(temp_file)

    @patch('rag_lens.core.test_case_manager.bigquery.Client')
    def test_load_test_cases_from_bigquery(self, mock_client):
        """Test loading test cases from BigQuery"""
        # Mock BigQuery response
        mock_query_result = Mock()
        mock_query_result.total_rows = 1

        mock_row = Mock()
        mock_row.__getitem__ = lambda self, key: {
            "id": "test_001",
            "name": "Test Case",
            "description": "Description",
            "query": "What is AI?",
            "system_prompt": "You are an AI assistant",
            "user_instruction": "Answer the question",
            "expected_answer": "AI is artificial intelligence",
            "chunks_data": '[]'
        }[key]

        mock_query_result.__iter__ = Mock(return_value=iter([mock_row]))
        mock_client.return_value.query.return_value = mock_query_result

        # Configure manager for BigQuery
        self.test_case_manager.bigquery_project = "test-project"
        self.test_case_manager.bigquery_dataset = "test_dataset"
        self.test_case_manager.bigquery_table = "test_table"

        test_cases = self.test_case_manager.load_test_cases_from_bigquery()

        assert len(test_cases) == 1
        assert test_cases[0].id == "test_001"

    @patch('rag_lens.core.test_case_manager.bigquery.Client')
    def test_load_test_cases_from_bigquery_error(self, mock_client):
        """Test BigQuery error handling"""
        mock_client.side_effect = Exception("BigQuery error")

        self.test_case_manager.bigquery_project = "test-project"
        self.test_case_manager.bigquery_dataset = "test_dataset"
        self.test_case_manager.bigquery_table = "test_table"

        with pytest.raises(TestCaseManagerError):
            self.test_case_manager.load_test_cases_from_bigquery()

    def test_validate_test_case_data(self):
        """Test test case data validation"""
        # Valid data
        valid_data = {
            "id": "test_001",
            "name": "Test Case",
            "query": "What is AI?",
            "system_prompt": "You are an AI assistant",
            "user_instruction": "Answer the question",
            "expected_answer": "AI is artificial intelligence"
        }

        # Should not raise an error
        self.test_case_manager._validate_test_case_data(valid_data)

        # Missing required field
        invalid_data = {
            "name": "Test Case",
            "query": "What is AI?"
            # Missing 'id'
        }

        with pytest.raises(TestCaseManagerError):
            self.test_case_manager._validate_test_case_data(invalid_data)

        # Invalid rating
        invalid_data = {
            "id": "test_001",
            "name": "Test Case",
            "query": "What is AI?",
            "system_prompt": "You are an AI assistant",
            "user_instruction": "Answer the question",
            "expected_answer": "AI is artificial intelligence",
            "user_rating": 10  # Invalid rating
        }

        with pytest.raises(TestCaseManagerError):
            self.test_case_manager._validate_test_case_data(invalid_data)


if __name__ == "__main__":
    pytest.main([__file__])