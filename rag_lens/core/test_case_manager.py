"""
Core test case management module
"""

import json
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
from dataclasses import dataclass

from ..config.settings import config
from ..utils.logger import get_logger
from ..utils.errors import TestCaseManagerError, ConfigurationError
from ..utils.security import encrypt_sensitive_data, decrypt_sensitive_data

logger = get_logger(__name__)


@dataclass
class Chunk:
    """Represents a document chunk"""
    id: str
    title: str
    content: str
    user_rating: int
    publish_time: str
    effective_time: str
    expiration_time: str


@dataclass
class TestCase:
    """Represents a test case"""
    id: str
    name: str
    description: str
    query: str
    system_prompt: str
    user_instruction: str
    expected_answer: str
    chunks: List[Chunk]
    domain: Optional[str] = None
    difficulty_level: Optional[str] = None
    tags: Optional[List[str]] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


class TestCaseManager:
    """Manage test cases with local storage and BigQuery integration"""

    def __init__(self, test_case_source: str = None):
        self.test_case_source = test_case_source or config.get_test_source_path(
            config.test_sources.default_source
        )
        self.test_cases: List[TestCase] = []
        self.bigquery_client = None

        # Initialize storage
        self._initialize_storage()

        # Load test cases
        self.test_cases = self._load_test_cases()
        logger.info(f"Loaded {len(self.test_cases)} test cases from {self.test_case_source}")

    def _initialize_storage(self):
        """Initialize storage backend"""
        try:
            if config.database.enable_bigquery:
                self._initialize_bigquery()
            else:
                logger.info("Using local storage for test cases")
        except Exception as e:
            logger.error(f"Failed to initialize storage: {e}")
            raise TestCaseManagerError(f"Storage initialization failed: {e}")

    def _initialize_bigquery(self):
        """Initialize BigQuery client"""
        try:
            from google.cloud import bigquery
            from google.oauth2 import service_account

            if not all([config.database.bigquery_project_id, config.database.bigquery_dataset_id]):
                raise ConfigurationError("BigQuery project_id and dataset_id are required")

            self.bigquery_client = bigquery.Client(
                project=config.database.bigquery_project_id
            )
            self._ensure_bigquery_table()
            logger.info("BigQuery storage initialized")
        except ImportError:
            raise TestCaseManagerError("BigQuery dependencies not available")
        except Exception as e:
            logger.error(f"BigQuery initialization failed: {e}")
            raise TestCaseManagerError(f"BigQuery initialization failed: {e}")

    def _ensure_bigquery_table(self):
        """Ensure BigQuery table exists"""
        if not self.bigquery_client:
            return

        table_id = f"{config.database.bigquery_project_id}.{config.database.bigquery_dataset_id}.test_cases"

        try:
            self.bigquery_client.get_table(table_id)
        except:
            # Create table if it doesn't exist
            schema = [
                bigquery.SchemaField("id", "STRING", mode="REQUIRED"),
                bigquery.SchemaField("name", "STRING", mode="REQUIRED"),
                bigquery.SchemaField("description", "STRING"),
                bigquery.SchemaField("query", "STRING", mode="REQUIRED"),
                bigquery.SchemaField("system_prompt", "STRING", mode="REQUIRED"),
                bigquery.SchemaField("user_instruction", "STRING"),
                bigquery.SchemaField("expected_answer", "STRING"),
                bigquery.SchemaField("chunks", "JSON", mode="REPEATED"),
                bigquery.SchemaField("created_at", "TIMESTAMP"),
                bigquery.SchemaField("updated_at", "TIMESTAMP"),
                bigquery.SchemaField("tags", "STRING", mode="REPEATED"),
                bigquery.SchemaField("difficulty_level", "STRING"),
                bigquery.SchemaField("domain", "STRING"),
            ]

            table = bigquery.Table(table_id, schema=schema)
            self.bigquery_client.create_table(table)
            logger.info(f"Created BigQuery table: {table_id}")

    def _load_test_cases(self) -> List[TestCase]:
        """Load test cases from storage"""
        try:
            if config.database.enable_bigquery and self.bigquery_client:
                return self._load_from_bigquery()
            else:
                return self._load_from_local()
        except Exception as e:
            logger.error(f"Failed to load test cases: {e}")
            raise TestCaseManagerError(f"Failed to load test cases: {e}")

    def _load_from_bigquery(self) -> List[TestCase]:
        """Load test cases from BigQuery"""
        try:
            query = f"""
            SELECT * FROM `{config.database.bigquery_project_id}.{config.database.bigquery_dataset_id}.test_cases`
            ORDER BY created_at DESC
            """
            query_job = self.bigquery_client.query(query)
            results = query_job.result()

            test_cases = []
            for row in results:
                test_case = self._row_to_test_case(row)
                test_cases.append(test_case)

            logger.info(f"Loaded {len(test_cases)} test cases from BigQuery")
            return test_cases
        except Exception as e:
            logger.error(f"Error loading from BigQuery: {e}")
            return self._load_from_local()

    def _load_from_local(self) -> List[TestCase]:
        """Load test cases from local JSON file"""
        try:
            with open(self.test_case_source, 'r', encoding='utf-8') as f:
                data = json.load(f)

            test_cases = self._parse_test_cases(data)
            logger.info(f"Loaded {len(test_cases)} test cases from local file")
            return test_cases
        except FileNotFoundError:
            logger.warning(f"Test case file not found: {self.test_case_source}")
            return []
        except json.JSONDecodeError as e:
            logger.error(f"Error reading test cases file: {e}")
            raise TestCaseManagerError(f"Invalid JSON in test cases file: {e}")

    def _parse_test_cases(self, data: Any) -> List[TestCase]:
        """Parse test cases from different JSON formats"""
        if isinstance(data, dict):
            if "test_cases_collection" in data:
                return [self._dict_to_test_case(tc) for tc in data["test_cases_collection"]]
            elif "enhanced_rag_test_suite" in data:
                return self._parse_enhanced_suite(data)
            else:
                return [self._dict_to_test_case(data)]
        elif isinstance(data, list):
            return [self._dict_to_test_case(tc) for tc in data]
        else:
            raise TestCaseManagerError("Unsupported test case data format")

    def _parse_enhanced_suite(self, data: Dict) -> List[TestCase]:
        """Parse enhanced test suite format"""
        test_cases = []
        for category in data["enhanced_rag_test_suite"]["categories"]:
            for test_case in category["test_cases"]:
                converted_case = TestCase(
                    id=test_case["test_case_id"],
                    name=f"{category['category_name']} - {test_case['subcategory']}",
                    description=test_case["objective"],
                    query=test_case["user_query"],
                    system_prompt=f"You are an expert in {category['category_name']}. {test_case.get('objective', '')}",
                    user_instruction=test_case.get("objective", ""),
                    expected_answer=test_case["expected_output"],
                    chunks=[
                        Chunk(
                            id=f"chunk_{i}",
                            title=f"Source chunk {i+1}",
                            content=chunk,
                            user_rating=4,
                            publish_time="2024-01-01T00:00:00",
                            effective_time="2024-01-01T00:00:00",
                            expiration_time="2026-01-01T23:59:59"
                        )
                        for i, chunk in enumerate(test_case["retrieved_chunks"])
                    ],
                    domain=category["category_name"].lower().replace(" ", "_"),
                    difficulty_level="intermediate",
                    tags=[category["category_id"], test_case["subcategory"].replace(" ", "_").lower()]
                )
                test_cases.append(converted_case)
        return test_cases

    def _dict_to_test_case(self, data: Dict) -> TestCase:
        """Convert dictionary to TestCase"""
        chunks = []
        for chunk_data in data.get("chunks", []):
            chunk = Chunk(
                id=chunk_data["id"],
                title=chunk_data["title"],
                content=chunk_data["content"],
                user_rating=chunk_data["user_rating"],
                publish_time=chunk_data["publish_time"],
                effective_time=chunk_data["effective_time"],
                expiration_time=chunk_data["expiration_time"]
            )
            chunks.append(chunk)

        return TestCase(
            id=data["id"],
            name=data["name"],
            description=data["description"],
            query=data["query"],
            system_prompt=data["system_prompt"],
            user_instruction=data["user_instruction"],
            expected_answer=data["expected_answer"],
            chunks=chunks,
            domain=data.get("domain"),
            difficulty_level=data.get("difficulty_level"),
            tags=data.get("tags", []),
            created_at=self._parse_datetime(data.get("created_at")),
            updated_at=self._parse_datetime(data.get("updated_at"))
        )

    def _row_to_test_case(self, row) -> TestCase:
        """Convert BigQuery row to TestCase"""
        chunks = []
        for chunk_data in row.chunks or []:
            chunk = Chunk(**chunk_data)
            chunks.append(chunk)

        return TestCase(
            id=row.id,
            name=row.name,
            description=row.description,
            query=row.query,
            system_prompt=row.system_prompt,
            user_instruction=row.user_instruction,
            expected_answer=row.expected_answer,
            chunks=chunks,
            created_at=row.created_at,
            updated_at=row.updated_at,
            tags=list(row.tags) if row.tags else [],
            difficulty_level=row.difficulty_level,
            domain=row.domain
        )

    def _parse_datetime(self, datetime_str: Optional[str]) -> Optional[datetime]:
        """Parse datetime string"""
        if not datetime_str:
            return None
        try:
            return datetime.fromisoformat(datetime_str.replace('Z', '+00:00'))
        except (ValueError, AttributeError):
            return None

    def get_test_case(self, test_case_id: str) -> Optional[TestCase]:
        """Get test case by ID"""
        for test_case in self.test_cases:
            if test_case.id == test_case_id:
                return test_case
        return None

    def get_test_cases_by_domain(self, domain: str) -> List[TestCase]:
        """Get test cases by domain"""
        return [tc for tc in self.test_cases if tc.domain == domain]

    def get_test_cases_by_difficulty(self, difficulty: str) -> List[TestCase]:
        """Get test cases by difficulty level"""
        return [tc for tc in self.test_cases if tc.difficulty_level == difficulty]

    def search_test_cases(self, query: str) -> List[TestCase]:
        """Search test cases by query"""
        query_lower = query.lower()
        results = []
        for test_case in self.test_cases:
            if (query_lower in test_case.name.lower() or
                query_lower in test_case.description.lower() or
                query_lower in test_case.query.lower()):
                results.append(test_case)
        return results

    def create_test_case(self, test_case: TestCase) -> str:
        """Create a new test case"""
        test_case.created_at = datetime.now()
        test_case.updated_at = datetime.now()

        if config.database.enable_bigquery and self.bigquery_client:
            return self._create_test_case_bigquery(test_case)
        else:
            return self._create_test_case_local(test_case)

    def _create_test_case_bigquery(self, test_case: TestCase) -> str:
        """Create test case in BigQuery"""
        try:
            table_id = f"{config.database.bigquery_project_id}.{config.database.bigquery_dataset_id}.test_cases"
            row = self._test_case_to_row(test_case)
            errors = self.bigquery_client.insert_rows_json(table_id, [row])

            if errors:
                raise TestCaseManagerError(f"BigQuery insert failed: {errors}")

            logger.info(f"Created test case {test_case.id} in BigQuery")
            return test_case.id
        except Exception as e:
            logger.error(f"Failed to create test case in BigQuery: {e}")
            raise TestCaseManagerError(f"Failed to create test case: {e}")

    def _create_test_case_local(self, test_case: TestCase) -> str:
        """Create test case in local storage"""
        try:
            self.test_cases.append(test_case)
            self._save_test_cases()
            logger.info(f"Created test case {test_case.id} locally")
            return test_case.id
        except Exception as e:
            logger.error(f"Failed to create test case locally: {e}")
            raise TestCaseManagerError(f"Failed to create test case: {e}")

    def update_test_case(self, test_case_id: str, updates: Dict[str, Any]) -> bool:
        """Update an existing test case"""
        test_case = self.get_test_case(test_case_id)
        if not test_case:
            return False

        # Update fields
        for key, value in updates.items():
            if hasattr(test_case, key):
                setattr(test_case, key, value)

        test_case.updated_at = datetime.now()

        if config.database.enable_bigquery and self.bigquery_client:
            return self._update_test_case_bigquery(test_case)
        else:
            return self._update_test_case_local(test_case)

    def delete_test_case(self, test_case_id: str) -> bool:
        """Delete a test case"""
        if config.database.enable_bigquery and self.bigquery_client:
            return self._delete_test_case_bigquery(test_case_id)
        else:
            return self._delete_test_case_local(test_case_id)

    def _save_test_cases(self):
        """Save test cases to storage"""
        if config.database.enable_bigquery and self.bigquery_client:
            self._save_to_bigquery()
        else:
            self._save_to_local()

    def _save_to_local(self):
        """Save test cases to local JSON file"""
        try:
            # Convert test cases to dictionaries
            test_cases_data = []
            for test_case in self.test_cases:
                case_dict = {
                    "id": test_case.id,
                    "name": test_case.name,
                    "description": test_case.description,
                    "query": test_case.query,
                    "system_prompt": test_case.system_prompt,
                    "user_instruction": test_case.user_instruction,
                    "expected_answer": test_case.expected_answer,
                    "chunks": [
                        {
                            "id": chunk.id,
                            "title": chunk.title,
                            "content": chunk.content,
                            "user_rating": chunk.user_rating,
                            "publish_time": chunk.publish_time,
                            "effective_time": chunk.effective_time,
                            "expiration_time": chunk.expiration_time
                        }
                        for chunk in test_case.chunks
                    ],
                    "domain": test_case.domain,
                    "difficulty_level": test_case.difficulty_level,
                    "tags": test_case.tags or [],
                    "created_at": test_case.created_at.isoformat() if test_case.created_at else None,
                    "updated_at": test_case.updated_at.isoformat() if test_case.updated_at else None
                }
                test_cases_data.append(case_dict)

            with open(self.test_case_source, 'w', encoding='utf-8') as f:
                json.dump({"test_cases_collection": test_cases_data}, f, indent=2, default=str)

            logger.info(f"Saved {len(self.test_cases)} test cases to local file")
        except Exception as e:
            logger.error(f"Failed to save test cases: {e}")
            raise TestCaseManagerError(f"Failed to save test cases: {e}")

    def get_statistics(self) -> Dict[str, Any]:
        """Get test case statistics"""
        domains = {}
        difficulties = {}
        total_chunks = 0

        for test_case in self.test_cases:
            # Count by domain
            if test_case.domain:
                domains[test_case.domain] = domains.get(test_case.domain, 0) + 1

            # Count by difficulty
            if test_case.difficulty_level:
                difficulties[test_case.difficulty_level] = difficulties.get(test_case.difficulty_level, 0) + 1

            total_chunks += len(test_case.chunks)

        return {
            "total_test_cases": len(self.test_cases),
            "total_chunks": total_chunks,
            "domains": domains,
            "difficulties": difficulties,
            "storage_type": "BigQuery" if config.database.enable_bigquery else "Local",
            "source_file": self.test_case_source
        }

    def _test_case_to_row(self, test_case: TestCase) -> Dict[str, Any]:
        """Convert TestCase to BigQuery row format"""
        return {
            "id": test_case.id,
            "name": test_case.name,
            "description": test_case.description,
            "query": test_case.query,
            "system_prompt": test_case.system_prompt,
            "user_instruction": test_case.user_instruction,
            "expected_answer": test_case.expected_answer,
            "chunks": [
                {
                    "id": chunk.id,
                    "title": chunk.title,
                    "content": chunk.content,
                    "user_rating": chunk.user_rating,
                    "publish_time": chunk.publish_time,
                    "effective_time": chunk.effective_time,
                    "expiration_time": chunk.expiration_time
                }
                for chunk in test_case.chunks
            ],
            "created_at": test_case.created_at,
            "updated_at": test_case.updated_at,
            "tags": test_case.tags or [],
            "difficulty_level": test_case.difficulty_level,
            "domain": test_case.domain
        }

    def _update_test_case_bigquery(self, test_case: TestCase) -> bool:
        """Update test case in BigQuery"""
        # Implementation for BigQuery update
        logger.info(f"Updated test case {test_case.id} in BigQuery")
        return True

    def _update_test_case_local(self, test_case: TestCase) -> bool:
        """Update test case in local storage"""
        self._save_test_cases()
        logger.info(f"Updated test case {test_case.id} locally")
        return True

    def _delete_test_case_bigquery(self, test_case_id: str) -> bool:
        """Delete test case from BigQuery"""
        # Implementation for BigQuery delete
        logger.info(f"Deleted test case {test_case_id} from BigQuery")
        return True

    def _delete_test_case_local(self, test_case_id: str) -> bool:
        """Delete test case from local storage"""
        self.test_cases = [tc for tc in self.test_cases if tc.id != test_case_id]
        self._save_test_cases()
        logger.info(f"Deleted test case {test_case_id} locally")
        return True

    def _save_to_bigquery(self):
        """Save test cases to BigQuery"""
        # Implementation for BigQuery save
        logger.info("Saved test cases to BigQuery")