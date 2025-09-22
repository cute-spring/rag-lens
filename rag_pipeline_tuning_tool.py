import streamlit as st
import pandas as pd
import json
import random
import uuid
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# BigQuery integration (optional - requires google-cloud-bigquery)
try:
    from google.cloud import bigquery
    from google.oauth2 import service_account
    BIGQUERY_AVAILABLE = True
except ImportError:
    BIGQUERY_AVAILABLE = False

# Configure page
st.set_page_config(
    page_title="RAG Pipeline Testing & Tuning Tool",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .metric-card {
        background: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #e1e5e9;
    }
    .pipeline-step {
        background: #ffffff;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        border-left: 4px solid #4CAF50;
    }
    .pipeline-step.processed {
        border-left-color: #2196F3;
    }
    .pipeline-step.current {
        border-left-color: #FF9800;
        background: #fff3e0;
    }
    .chunk-card {
        background: #fafafa;
        padding: 0.75rem;
        border-radius: 0.25rem;
        margin: 0.5rem 0;
        border: 1px solid #e0e0e0;
    }
    .score-indicator {
        display: inline-block;
        padding: 0.2rem 0.5rem;
        border-radius: 0.25rem;
        font-weight: bold;
        color: white;
    }
    .score-high { background: #4CAF50; }
    .score-medium { background: #FF9800; }
    .score-low { background: #f44336; }
    .status-pass { background: #4CAF50; color: white; padding: 0.2rem 0.5rem; border-radius: 0.25rem; }
    .status-fail { background: #f44336; color: white; padding: 0.2rem 0.5rem; border-radius: 0.25rem; }
    .status-pending { background: #FF9800; color: white; padding: 0.2rem 0.5rem; border-radius: 0.25rem; }
    .step-button {
        margin: 0.25rem;
        min-width: 120px;
    }
    .step-progress {
        background: #e0e0e0;
        border-radius: 1rem;
        height: 0.5rem;
        margin: 1rem 0;
    }
    .step-progress-bar {
        background: #4CAF50;
        height: 100%;
        border-radius: 1rem;
        transition: width 0.3s ease;
    }
</style>
""", unsafe_allow_html=True)


class TestCaseManager:
    """Manage test cases with local storage and BigQuery integration"""

    def __init__(self, use_bigquery: bool = False, project_id: str = None, dataset_id: str = None,
                 test_case_source: str = "test_cases_local.json"):
        self.use_bigquery = use_bigquery and BIGQUERY_AVAILABLE
        self.project_id = project_id
        self.dataset_id = dataset_id
        self.test_case_source = test_case_source
        self.bigquery_client = None

        # Initialize BigQuery client if enabled
        if self.use_bigquery:
            try:
                self.bigquery_client = bigquery.Client(project=project_id)
                self._ensure_bigquery_setup()
            except Exception as e:
                st.warning(f"BigQuery initialization failed: {e}. Falling back to local storage.")
                self.use_bigquery = False

        # Load existing test cases
        self.test_cases = self._load_test_cases()

    def _ensure_bigquery_setup(self):
        """Ensure BigQuery table exists"""
        table_id = f"{self.project_id}.{self.dataset_id}.test_cases"

        # Check if table exists
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
            table = self.bigquery_client.create_table(table)

    def _load_test_cases(self) -> List[Dict[str, Any]]:
        """Load test cases from storage"""
        if self.use_bigquery:
            return self._load_from_bigquery()
        else:
            return self._load_from_local()

    def _load_from_bigquery(self) -> List[Dict[str, Any]]:
        """Load test cases from BigQuery"""
        try:
            query = f"""
            SELECT * FROM `{self.project_id}.{self.dataset_id}.test_cases`
            ORDER BY created_at DESC
            """
            query_job = self.bigquery_client.query(query)
            results = query_job.result()

            test_cases = []
            for row in results:
                test_case = {
                    "id": row.id,
                    "name": row.name,
                    "description": row.description,
                    "query": row.query,
                    "system_prompt": row.system_prompt,
                    "user_instruction": row.user_instruction,
                    "expected_answer": row.expected_answer,
                    "chunks": json.loads(row.chunks) if row.chunks else [],
                    "created_at": row.created_at.isoformat() if row.created_at else None,
                    "updated_at": row.updated_at.isoformat() if row.updated_at else None,
                    "tags": list(row.tags) if row.tags else [],
                    "difficulty_level": row.difficulty_level,
                    "domain": row.domain,
                }
                test_cases.append(test_case)

            return test_cases
        except Exception as e:
            st.error(f"Error loading from BigQuery: {e}")
            return self._load_from_local()

    def _load_from_local(self) -> List[Dict[str, Any]]:
        """Load test cases from local JSON file"""
        try:
            with open(self.test_case_source, 'r') as f:
                data = json.load(f)

            # Handle different JSON formats
            if isinstance(data, dict):
                if "test_cases_collection" in data:
                    return data["test_cases_collection"]
                elif "enhanced_rag_test_suite" in data:
                    # Convert COMPLETE_TEST_SUITE.json format
                    test_cases = []
                    for category in data["enhanced_rag_test_suite"]["categories"]:
                        for test_case in category["test_cases"]:
                            # Convert the format
                            converted_case = {
                                "id": test_case["test_case_id"],
                                "name": f"{category['category_name']} - {test_case['subcategory']}",
                                "description": test_case["objective"],
                                "query": test_case["user_query"],
                                "system_prompt": f"You are an expert in {category['category_name']}. {test_case.get('objective', '')}",
                                "user_instruction": test_case.get("objective", ""),
                                "expected_answer": test_case["expected_output"],
                                "chunks": [
                                    {
                                        "id": f"chunk_{i}",
                                        "title": f"Source chunk {i+1}",
                                        "content": chunk,
                                        "user_rating": 4,
                                        "publish_time": "2024-01-01T00:00:00",
                                        "effective_time": "2024-01-01T00:00:00",
                                        "expiration_time": "2026-01-01T23:59:59"
                                    }
                                    for i, chunk in enumerate(test_case["retrieved_chunks"])
                                ]
                            }
                            test_cases.append(converted_case)
                    return test_cases
                else:
                    return [data]
            else:
                return data
        except FileNotFoundError:
            return []
        except json.JSONDecodeError:
            st.error(f"Error reading test cases file '{self.test_case_source}'. Starting with empty collection.")
            return []

    def _save_test_cases(self):
        """Save test cases to storage"""
        if self.use_bigquery:
            self._save_to_bigquery()
        else:
            self._save_to_local()

    def _save_to_bigquery(self):
        """Save test cases to BigQuery"""
        try:
            table_id = f"{self.project_id}.{self.dataset_id}.test_cases"

            # Prepare data for BigQuery
            rows_to_insert = []
            for test_case in self.test_cases:
                row = {
                    "id": test_case["id"],
                    "name": test_case["name"],
                    "description": test_case.get("description", ""),
                    "query": test_case["query"],
                    "system_prompt": test_case["system_prompt"],
                    "user_instruction": test_case.get("user_instruction", ""),
                    "expected_answer": test_case.get("expected_answer", ""),
                    "chunks": json.dumps(test_case.get("chunks", [])),
                    "created_at": test_case.get("created_at", datetime.now().isoformat()),
                    "updated_at": datetime.now().isoformat(),
                    "tags": test_case.get("tags", []),
                    "difficulty_level": test_case.get("difficulty_level", "medium"),
                    "domain": test_case.get("domain", "general"),
                }
                rows_to_insert.append(row)

            # Delete existing data and insert new
            self.bigquery_client.query(f"DELETE FROM `{table_id}` WHERE TRUE").result()
            errors = self.bigquery_client.insert_rows_json(table_id, rows_to_insert)

            if errors:
                st.error(f"BigQuery insert errors: {errors}")
        except Exception as e:
            st.error(f"Error saving to BigQuery: {e}")

    def _save_to_local(self):
        """Save test cases to local JSON file"""
        try:
            with open(self.local_storage_file, 'w') as f:
                json.dump(self.test_cases, f, indent=2, default=str)
        except Exception as e:
            st.error(f"Error saving test cases: {e}")

    def create_test_case(self, test_case_data: Dict[str, Any]) -> str:
        """Create a new test case"""
        # Generate ID if not provided
        if "id" not in test_case_data:
            test_case_data["id"] = str(uuid.uuid4())

        # Add timestamps
        now = datetime.now().isoformat()
        test_case_data["created_at"] = now
        test_case_data["updated_at"] = now

        # Add default values
        test_case_data.setdefault("tags", [])
        test_case_data.setdefault("difficulty_level", "medium")
        test_case_data.setdefault("domain", "general")

        # Validate required fields
        required_fields = ["id", "name", "query", "system_prompt", "chunks"]
        for field in required_fields:
            if field not in test_case_data:
                raise ValueError(f"Missing required field: {field}")

        # Add to collection
        self.test_cases.append(test_case_data)
        self._save_test_cases()

        return test_case_data["id"]

    def update_test_case(self, test_case_id: str, updates: Dict[str, Any]) -> bool:
        """Update an existing test case"""
        test_case = self.get_test_case(test_case_id)
        if not test_case:
            return False

        # Update fields
        for key, value in updates.items():
            if key in ["id", "created_at"]:  # Don't update these
                continue
            test_case[key] = value

        # Update timestamp
        test_case["updated_at"] = datetime.now().isoformat()

        self._save_test_cases()
        return True

    def delete_test_case(self, test_case_id: str) -> bool:
        """Delete a test case"""
        self.test_cases = [tc for tc in self.test_cases if tc["id"] != test_case_id]
        self._save_test_cases()
        return True

    def get_test_case(self, test_case_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific test case by ID"""
        for test_case in self.test_cases:
            if test_case["id"] == test_case_id:
                return test_case
        return None

    def search_test_cases(self, query: str = "", tags: List[str] = None,
                         domain: str = None, difficulty: str = None) -> List[Dict[str, Any]]:
        """Search test cases with filters"""
        results = self.test_cases

        # Text search
        if query:
            query_lower = query.lower()
            results = [tc for tc in results if
                      query_lower in tc.get("name", "").lower() or
                      query_lower in tc.get("description", "").lower() or
                      query_lower in tc.get("query", "").lower() or
                      query_lower in tc.get("expected_answer", "").lower()]

        # Tag filtering
        if tags:
            results = [tc for tc in results if
                      any(tag in tc.get("tags", []) for tag in tags)]

        # Domain filtering
        if domain:
            results = [tc for tc in results if tc.get("domain") == domain]

        # Difficulty filtering
        if difficulty:
            results = [tc for tc in results if tc.get("difficulty_level") == difficulty]

        return results

    def get_statistics(self) -> Dict[str, Any]:
        """Get test case statistics"""
        if not self.test_cases:
            return {"total": 0, "domains": {}, "difficulties": {}, "avg_chunks": 0}

        domains = {}
        difficulties = {}
        total_chunks = 0

        for test_case in self.test_cases:
            # Domain statistics
            domain = test_case.get("domain", "general")
            domains[domain] = domains.get(domain, 0) + 1

            # Difficulty statistics
            difficulty = test_case.get("difficulty_level", "medium")
            difficulties[difficulty] = difficulties.get(difficulty, 0) + 1

            # Chunk count
            total_chunks += len(test_case.get("chunks", []))

        return {
            "total": len(self.test_cases),
            "domains": domains,
            "difficulties": difficulties,
            "avg_chunks": total_chunks / len(self.test_cases),
            "storage_type": "BigQuery" if self.use_bigquery else "Local"
        }

    def import_test_cases(self, file_path: str) -> int:
        """Import test cases from JSON file"""
        try:
            with open(file_path, 'r') as f:
                imported_data = json.load(f)

            # Handle different import formats
            if isinstance(imported_data, dict):
                if "test_cases_collection" in imported_data:
                    test_cases = imported_data["test_cases_collection"]
                elif "test_case_reference" in imported_data:
                    test_cases = [imported_data["test_case_reference"]]
                else:
                    test_cases = [imported_data]
            else:
                test_cases = imported_data

            # Import test cases
            imported_count = 0
            for test_case in test_cases:
                try:
                    # Generate new ID to avoid conflicts
                    test_case["id"] = str(uuid.uuid4())
                    self.create_test_case(test_case)
                    imported_count += 1
                except Exception as e:
                    st.warning(f"Failed to import test case: {e}")

            return imported_count
        except Exception as e:
            st.error(f"Error importing test cases: {e}")
            return 0

    def export_test_cases(self, format_type: str = "json") -> str:
        """Export test cases in specified format"""
        if format_type == "json":
            return json.dumps(self.test_cases, indent=2, default=str)
        elif format_type == "csv":
            # Flatten test cases for CSV export
            flattened = []
            for test_case in self.test_cases:
                flattened.append({
                    "id": test_case["id"],
                    "name": test_case["name"],
                    "description": test_case.get("description", ""),
                    "query": test_case["query"],
                    "domain": test_case.get("domain", ""),
                    "difficulty_level": test_case.get("difficulty_level", ""),
                    "tags": ", ".join(test_case.get("tags", [])),
                    "chunk_count": len(test_case.get("chunks", [])),
                    "created_at": test_case.get("created_at", ""),
                    "updated_at": test_case.get("updated_at", ""),
                })

            df = pd.DataFrame(flattened)
            return df.to_csv(index=False)
        else:
            raise ValueError(f"Unsupported export format: {format_type}")


class MockDataGenerator:
    """Generate mock test case data for demonstration"""

    def __init__(self):
        self.topics = ["Machine Learning", "Healthcare", "Finance", "Education", "Technology"]
        self.purposes = ["Retrieval Accuracy", "Relevance Ranking", "Query Understanding", "Response Quality", "Parameter Sensitivity"]

        # System prompt templates for different domains
        self.system_prompts = {
            "Machine Learning": """You are an expert machine learning engineer and researcher. Your responses should:
- Provide accurate, comprehensive explanations of ML concepts
- Include mathematical foundations where relevant
- Discuss practical implementations and real-world applications
- Consider current state-of-the-art approaches and limitations
- Structure answers with clear sections and examples""",

            "Healthcare": """You are a medical professional with expertise in healthcare technology. Your responses should:
- Prioritize patient safety and evidence-based information
- Cite current medical guidelines and research
- Clearly distinguish between established facts and emerging research
- Consider ethical implications and patient perspectives
- Use precise medical terminology while remaining accessible""",

            "Finance": """You are a financial analyst and investment expert. Your responses should:
- Provide accurate financial information and analysis
- Include relevant market data and economic context
- Discuss risk factors and regulatory considerations
- Present balanced perspectives on financial strategies
- Use appropriate financial terminology with clear explanations""",

            "Education": """You are an educational technology specialist and learning scientist. Your responses should:
- Follow evidence-based teaching principles
- Adapt explanations to different learning styles
- Include practical examples and applications
- Consider accessibility and inclusivity
- Promote critical thinking and problem-solving skills""",

            "Technology": """You are a software architect and technology consultant. Your responses should:
- Provide current, accurate technical information
- Include code examples and implementation details
- Consider scalability, security, and performance implications
- Compare different approaches and trade-offs
- Reference industry best practices and standards"""
        }

        # Optional user instruction templates
        self.user_instructions = [
            "Focus on practical applications with real-world examples",
            "Include step-by-step explanations and visual descriptions",
            "Compare different approaches and highlight advantages/disadvantages",
            "Provide both technical depth and accessible explanations",
            "Include current research trends and future directions",
            "Focus on implementation challenges and solutions",
            "Provide beginner-friendly explanations with analogies",
            "Include mathematical foundations and theoretical concepts",
            "Discuss ethical considerations and societal impact",
            "Provide actionable insights and recommendations"
        ]

    def generate_chunk(self, idx: int, topic: str) -> Dict[str, Any]:
        """Generate a single content chunk with metadata"""
        publish_time = datetime.now() - timedelta(days=random.randint(1, 365))
        effective_time = publish_time + timedelta(days=random.randint(0, 30))
        expiration_time = effective_time + timedelta(days=random.randint(365, 730))

        # Generate individual scores
        relevance_score = random.uniform(0.3, 1.0)
        freshness_score = random.uniform(0.5, 1.0)
        quality_score = random.uniform(0.4, 1.0)

        # Calculate composite score using the same weights as the pipeline
        composite_score = (relevance_score * 0.5 + freshness_score * 0.2 + quality_score * 0.3)

        return {
            "id": f"chunk_{idx}",
            "title": f"{topic} Concept {idx + 1}",
            "content": f"""
## {topic} Concept {idx + 1}

This is a comprehensive overview of **key concepts** in {topic}.

### Key Points:
- **Fundamental Principle**: Understanding core concepts is essential
- **Advanced Applications**: Real-world implementation strategies
- **Best Practices**: Industry-standard approaches

### Technical Details:
The implementation involves multiple components working together:
1. Data preprocessing and normalization
2. Feature extraction and engineering
3. Model training and validation
4. Performance optimization

### Conclusion:
This concept serves as a foundation for more advanced topics in {topic}.
            """,
            "publish_time": publish_time.isoformat(),
            "effective_time": effective_time.isoformat(),
            "expiration_time": expiration_time.isoformat(),
            "user_rating": random.randint(0, 5),
            "relevance_score": relevance_score,
            "freshness_score": freshness_score,
            "quality_score": quality_score,
            "composite_score": composite_score
        }

    def generate_test_case(self, case_id: int) -> Dict[str, Any]:
        """Generate a complete test case"""
        topic = random.choice(self.topics)
        purpose = random.choice(self.purposes)

        # Generate 20 chunks
        chunks = [self.generate_chunk(i, topic) for i in range(20)]

        # Generate query and instruction
        query_types = ["What are the key concepts", "How to implement", "Explain the process", "Best practices for", "Compare and contrast"]
        query = f"{random.choice(query_types)} in {topic}?"

        # Generate model and parameters
        models = ["gpt-4", "gpt-3.5-turbo", "claude-3", "gemini-pro"]

        return {
            "id": f"test_case_{case_id}",
            "name": f"{topic} - {purpose} Test",
            "topic": topic,
            "purpose": purpose,
            "description": f"Test case for evaluating {purpose.lower()} in {topic} domain with pre-collected chunks",
            "chunks": chunks,
            "user_query": query,
            "user_instruction": random.choice(self.user_instructions),
            "system_prompt": self.system_prompts[topic],
            "model_version": random.choice(models),
            "rerank_params": {
                "semantic_weight": random.uniform(0.3, 0.7),
                "freshness_weight": random.uniform(0.1, 0.3),
                "quality_weight": random.uniform(0.2, 0.5),
                "relevance_threshold": random.uniform(0.5, 0.8),
                "top_n": random.randint(5, 15)
            },
            "expected_answer": f"This would be the ideal answer for: {query} It should cover all key aspects of {topic} with proper structure and examples.",
            "incorrect_answer": {
                "answer": f"This is an incorrect response that misses key points about {topic} and provides incomplete information.",
                "user_concerns": [
                    "Missing key concepts",
                    "No practical examples",
                    "Outdated information",
                    "Poor structure"
                ],
                "user_comments": f"The answer needs improvement in explaining {topic} concepts more clearly with better examples."
            },
            "status": random.choice(["pass", "fail", "pending"]),
            "created_at": datetime.now() - timedelta(days=random.randint(1, 30)),
            "last_modified": datetime.now() - timedelta(days=random.randint(0, 7))
        }

class RAGPipelineSimulator:
    """Simulate the RAG pipeline processing with step-by-step control"""

    def __init__(self, test_case: Dict[str, Any]):
        self.test_case = test_case
        self.pipeline_steps = [
            "filtering",
            "reranking",
            "selection",
            "context",
            "response",
            "analysis"
        ]
        self.step_names = [
            "Load Init Chunks",
            "Re-ranking",
            "Sub-segment Extraction",
            "Context Assembly",
            "Response Generation",
            "Results Analysis"
        ]
        self.reset()

    def reset(self):
        """Reset the pipeline state"""
        self.current_step = 0
        self.step_results = {}

        # Process chunks to ensure scores are present without modifying original
        processed_chunks = []
        for chunk in self.test_case["chunks"]:
            new_chunk = chunk.copy()
            if "relevance_score" not in new_chunk or new_chunk["relevance_score"] is None:
                new_chunk["relevance_score"] = 0.0
            if "freshness_score" not in new_chunk or new_chunk["freshness_score"] is None:
                new_chunk["freshness_score"] = 0.0
            if "quality_score" not in new_chunk or new_chunk["quality_score"] is None:
                new_chunk["quality_score"] = 0.0
            processed_chunks.append(new_chunk)

        self.intermediate_data = {
            "retrieved_chunks": processed_chunks,  # Start with processed chunks
            "filtered_chunks": None,
            "reranked_chunks": None,
            "selected_chunks": None,
            "context": None,
            "response": None
        }

    def execute_step(self, step_index: int, params: Dict[str, Any], system_prompt: str = None, user_instruction: str = None) -> Dict[str, Any]:
        """Execute a specific step and return its result"""
        if step_index >= len(self.pipeline_steps):
            raise ValueError(f"Invalid step index: {step_index}")

        step_name = self.pipeline_steps[step_index]

        if step_index == 0:  # Load Init Chunks
            if self.intermediate_data["retrieved_chunks"] is None:
                raise ValueError("Retrieved chunks must be available")
            result = {
                "count": len(self.intermediate_data["retrieved_chunks"]),
                "chunks": self._initial_filter(self.intermediate_data["retrieved_chunks"], params["relevance_threshold"])
            }
            self.intermediate_data["filtered_chunks"] = result["chunks"]

        elif step_index == 1:  # Re-ranking
            if self.intermediate_data["filtered_chunks"] is None:
                raise ValueError("Filtering must be completed first")
            result = {
                "count": len(self.intermediate_data["filtered_chunks"]),
                "chunks": self._rerank_chunks(self.intermediate_data["filtered_chunks"], params)
            }
            self.intermediate_data["reranked_chunks"] = result["chunks"]

        elif step_index == 2:  # Final Selection
            if self.intermediate_data["reranked_chunks"] is None:
                raise ValueError("Re-ranking must be completed first")
            result = {
                "count": len(self.intermediate_data["reranked_chunks"]),
                "chunks": self._select_top_chunks(self.intermediate_data["reranked_chunks"], params["top_n"])
            }
            self.intermediate_data["selected_chunks"] = result["chunks"]

        elif step_index == 3:  # Context Assembly
            if self.intermediate_data["selected_chunks"] is None:
                raise ValueError("Selection must be completed first")
            result = self._assemble_context(self.intermediate_data["selected_chunks"])
            self.intermediate_data["context"] = result

        elif step_index == 4:  # Response Generation
            if self.intermediate_data["context"] is None:
                raise ValueError("Context assembly must be completed first")
            result = self._generate_response(
                self.intermediate_data["context"],
                self.test_case["query"],
                system_prompt,
                user_instruction
            )
            self.intermediate_data["response"] = result
        elif step_index == 5:  # Results Analysis
            if self.intermediate_data["response"] is None:
                raise ValueError("Response generation must be completed first")
            result = self._analyze_results()
            self.intermediate_data["analysis"] = result

        self.step_results[step_name] = result
        self.current_step = step_index + 1
        return result

    def get_all_results(self) -> Dict[str, Any]:
        """Get all step results and calculate metrics"""
        # Calculate metrics only if we have enough data
        if all(data is not None for data in [
            self.intermediate_data["retrieved_chunks"],
            self.intermediate_data["filtered_chunks"],
            self.intermediate_data["selected_chunks"]
        ]):
            metrics = self._calculate_metrics(
                self.intermediate_data["retrieved_chunks"],
                self.intermediate_data["filtered_chunks"],
                self.intermediate_data["selected_chunks"]
            )
        else:
            metrics = {
                "retrieval_rate": 0,
                "filter_rate": 0,
                "selection_rate": 0,
                "avg_relevance": 0,
                "avg_composite": 0
            }

        return {
            "steps": self.step_results,
            "metrics": metrics
        }

    def process_pipeline(self, params: Dict[str, Any], system_prompt: str = None, user_instruction: str = None) -> Dict[str, Any]:
        """Simulate the full 7-step pipeline processing (legacy method)"""
        self.reset()
        for i in range(len(self.pipeline_steps)):
            self.execute_step(i, params, system_prompt, user_instruction)
        return self.get_all_results()

    def _process_query(self, query: str) -> str:
        """Simulate query processing"""
        return f"Processed: {query.lower().replace('?', '')}"

    def _retrieve_chunks(self, chunks: List[Dict], query: str) -> List[Dict]:
        """Simulate initial retrieval"""
        # Scores are now added in reset(), so we just sort
        return sorted(chunks, key=lambda x: x.get("relevance_score", 0), reverse=True)

    def _initial_filter(self, chunks: List[Dict], threshold: float) -> List[Dict]:
        """Simulate initial filtering"""
        # If all scores are 0, bypass filtering as they are not yet calculated
        if all(c.get('relevance_score', 0) == 0 for c in chunks):
            return chunks
        return [chunk for chunk in chunks if chunk.get("relevance_score", 0) >= threshold]

    def _rerank_chunks(self, chunks: List[Dict], params: Dict[str, Any]) -> List[Dict]:
        """Simulate re-ranking with custom weights"""
        for chunk in chunks:
            # Re-calculate scores for re-ranking
            chunk["relevance_score"] = random.uniform(0.7, 1.0)
            chunk["freshness_score"] = random.uniform(0.7, 1.0)
            chunk["quality_score"] = random.uniform(0.7, 1.0)

            # Calculate composite score
            composite_score = (
                chunk["relevance_score"] * params["semantic_weight"] +
                chunk["freshness_score"] * params["freshness_weight"] +
                chunk["quality_score"] * params["quality_weight"]
            )
            chunk["composite_score"] = composite_score

        return sorted(chunks, key=lambda x: x["composite_score"], reverse=True)

    def _select_top_chunks(self, chunks: List[Dict], top_n: int) -> List[Dict]:
        """Extract relevant sub-segments from chunks to reduce noise and improve focus"""
        extracted_sub_segments = []

        for chunk in chunks[:min(top_n * 2, len(chunks))]:  # Consider more chunks to find best sub-segments
            # Split content into meaningful sub-segments based on markdown headers
            content = chunk['content']
            lines = content.split('\n')

            sub_segments = []
            current_segment = []
            current_header = None

            for line in lines:
                stripped_line = line.strip()
                if stripped_line.startswith('###'):  # Third level header
                    if current_segment:  # Save previous segment
                        sub_segments.append({
                            'header': current_header,
                            'content': '\n'.join(current_segment).strip(),
                            'line_count': len(current_segment)
                        })
                    current_header = stripped_line
                    current_segment = [line]
                elif stripped_line.startswith('##') or stripped_line.startswith('#'):
                    # Skip higher level headers as they indicate major sections
                    if current_segment:
                        sub_segments.append({
                            'header': current_header,
                            'content': '\n'.join(current_segment).strip(),
                            'line_count': len(current_segment)
                        })
                        current_segment = []
                    current_header = stripped_line
                else:
                    current_segment.append(line)

            # Add the last segment
            if current_segment:
                sub_segments.append({
                    'header': current_header,
                    'content': '\n'.join(current_segment).strip(),
                    'line_count': len(current_segment)
                })

            # If no sub-segments found, treat whole chunk as one segment
            if not sub_segments:
                sub_segments = [{
                    'header': chunk['title'],
                    'content': content,
                    'line_count': len(lines)
                }]

            # Score and select best sub-segments from this chunk
            for segment in sub_segments:
                # Calculate relevance score for sub-segment based on:
                # 1. Original chunk's relevance score
                # 2. Content density (meaningful content vs noise)
                # 3. Structural importance (having a header)
                content_density = len([line for line in segment['content'].split('\n') if line.strip() and not line.strip().startswith('#')]) / max(segment['line_count'], 1)

                has_header = segment['header'] is not None
                structural_bonus = 1.2 if has_header else 1.0

                sub_segment_score = chunk['composite_score'] * content_density * structural_bonus

                extracted_sub_segments.append({
                    'id': f"{chunk['id']}_sub_{len(extracted_sub_segments)}",
                    'title': segment['header'] if segment['header'] else chunk['title'],
                    'content': segment['content'],
                    'original_chunk_id': chunk['id'],
                    'original_title': chunk['title'],
                    'relevance_score': chunk['relevance_score'],
                    'freshness_score': chunk['freshness_score'],
                    'quality_score': chunk['quality_score'],
                    'composite_score': sub_segment_score,
                    'content_density': content_density,
                    'line_count': segment['line_count'],
                    'publish_time': chunk['publish_time'],
                    'effective_time': chunk['effective_time'],
                    'user_rating': chunk['user_rating'],
                    'is_sub_segment': True
                })

        # Sort by composite score and return top N sub-segments
        extracted_sub_segments.sort(key=lambda x: x['composite_score'], reverse=True)
        return extracted_sub_segments[:top_n]

    def _assemble_context(self, chunks: List[Dict]) -> str:
        """Assemble context from selected chunks"""
        context_parts = []
        for chunk in chunks:
            context_parts.append(f"### {chunk['title']}\n{chunk['content']}")
        return "\n\n".join(context_parts)

    def _generate_response(self, context: str, query: str, system_prompt: str = None, user_instruction: str = None) -> str:
        """Simulate response generation using both system prompt and user instruction"""
        # Use the provided prompts or fall back to test case defaults
        final_system_prompt = system_prompt or self.test_case["system_prompt"]
        final_user_instruction = user_instruction or self.test_case["user_instruction"]

        # Construct the full prompt
        prompt_parts = [final_system_prompt]

        if final_user_instruction:
            prompt_parts.append(f"\n\nAdditional Instructions: {final_user_instruction}")

        prompt_parts.append(f"\n\nContext:\n{context}")
        prompt_parts.append(f"\n\nUser Query: {query}")

        full_prompt = "".join(prompt_parts)

        # Simulate different response quality based on prompt completeness
        if final_user_instruction:
            response_quality = "high-quality, detailed"
        else:
            response_quality = "standard"

        return f"""[{response_quality} response using {self.test_case['model_version']}]

Query: {query}

Response:
Based on the provided context and system guidelines, I can provide a comprehensive response to your query about "{query}". The context contains relevant information that helps address your specific needs.

Key points covered:
- Analysis of the main concepts related to your query
- Practical applications and considerations
- Current best practices and approaches

The response has been generated using the specified system prompt and follows the guidelines for providing accurate, helpful information.

[Simulated response - Full prompt would be sent to {self.test_case['model_version']}]"""

    def _analyze_results(self) -> Dict[str, Any]:
        """Analyze pipeline results and generate comprehensive reports"""
        # Get all intermediate data
        filtered_chunks = self.intermediate_data["filtered_chunks"]
        reranked_chunks = self.intermediate_data["reranked_chunks"]
        selected_chunks = self.intermediate_data["selected_chunks"]
        context = self.intermediate_data["context"]
        response = self.intermediate_data["response"]

        # Calculate quality metrics
        context_relevance = self._calculate_context_relevance(selected_chunks, context)
        response_quality = self._calculate_response_quality(response, context, self.test_case["expected_answer"])
        noise_reduction = self._calculate_noise_reduction(filtered_chunks, selected_chunks)

        # Generate insights
        insights = []
        if noise_reduction > 0.7:
            insights.append("✅ Excellent noise reduction - sub-segment extraction effectively filtered out irrelevant content")
        elif noise_reduction > 0.5:
            insights.append("✅ Good noise reduction - most irrelevant content was filtered out")
        else:
            insights.append("⚠️  Limited noise reduction - consider adjusting extraction parameters")

        if context_relevance > 0.8:
            insights.append("✅ High context relevance - assembled context is well-focused on the query")
        elif context_relevance > 0.6:
            insights.append("✅ Moderate context relevance - context contains relevant information")
        else:
            insights.append("⚠️  Low context relevance - consider improving filtering and extraction")

        if response_quality > 0.8:
            insights.append("✅ High response quality - generated response addresses the query effectively")
        elif response_quality > 0.6:
            insights.append("✅ Moderate response quality - response is somewhat relevant")
        else:
            insights.append("⚠️  Low response quality - response may not adequately address the query")

        # Calculate improvement suggestions
        suggestions = self._generate_improvement_suggestions(filtered_chunks, reranked_chunks, selected_chunks, response_quality)

        # Generate summary report
        summary_report = self._generate_summary_report(
            filtered_chunks, reranked_chunks, selected_chunks,
            context_relevance, response_quality, noise_reduction
        )

        return {
            "context_relevance": context_relevance,
            "response_quality": response_quality,
            "noise_reduction": noise_reduction,
            "insights": insights,
            "suggestions": suggestions,
            "summary_report": summary_report,
            "pipeline_efficiency": self._calculate_pipeline_efficiency(filtered_chunks, selected_chunks),
            "content_utilization": self._calculate_content_utilization(selected_chunks, context)
        }

    def _calculate_context_relevance(self, selected_chunks: List[Dict], context: str) -> float:
        """Calculate how relevant the assembled context is to the query"""
        if not selected_chunks or not context:
            return 0.0

        # Calculate based on average chunk scores and context density
        avg_relevance = sum(c.get('relevance_score', 0) for c in selected_chunks) / len(selected_chunks)
        context_density = len(context.split()) / max(len(selected_chunks) * 50, 1)  # Words per chunk ratio

        return min(avg_relevance * min(context_density / 10, 1.0), 1.0)

    def _calculate_response_quality(self, response: str, context: str, expected_answer: str) -> float:
        """Calculate the quality of the generated response"""
        if not response:
            return 0.0

        # Simple simulation based on response length and content
        response_length = len(response.split())
        context_words = set(context.lower().split())
        response_words = set(response.lower().split())
        expected_words = set(expected_answer.lower().split())

        # Calculate overlap with expected answer
        expected_overlap = len(response_words.intersection(expected_words)) / max(len(expected_words), 1)
        context_usage = len(response_words.intersection(context_words)) / max(len(response_words), 1)

        # Quality score based on multiple factors
        quality_score = (expected_overlap * 0.5 + context_usage * 0.3 + min(response_length / 200, 1.0) * 0.2)

        return min(quality_score, 1.0)

    def _calculate_noise_reduction(self, filtered_chunks: List[Dict], selected_chunks: List[Dict]) -> float:
        """Calculate how much noise was reduced through sub-segment extraction"""
        if not filtered_chunks:
            return 0.0

        original_total_length = sum(len(c['content']) for c in filtered_chunks)
        extracted_total_length = sum(len(c['content']) for c in selected_chunks)

        if original_total_length == 0:
            return 0.0

        # Noise reduction is the ratio of content removed while preserving quality
        noise_removed = 1 - (extracted_total_length / original_total_length)

        # Adjust based on average quality of extracted content
        avg_quality = sum(c.get('composite_score', 0) for c in selected_chunks) / max(len(selected_chunks), 1)
        adjusted_noise_reduction = noise_removed * avg_quality

        return min(adjusted_noise_reduction, 1.0)

    def _generate_improvement_suggestions(self, filtered_chunks: List[Dict], reranked_chunks: List[Dict],
                                        selected_chunks: List[Dict], response_quality: float) -> List[str]:
        """Generate actionable suggestions for pipeline improvement"""
        suggestions = []

        # Analyze filtering effectiveness
        filter_rate = len(reranked_chunks) / max(len(filtered_chunks), 1)
        if filter_rate > 0.8:
            suggestions.append("Consider tightening relevance thresholds to reduce the number of chunks processed")
        elif filter_rate < 0.3:
            suggestions.append("Consider relaxing relevance thresholds to avoid missing relevant content")

        # Analyze extraction effectiveness
        if len(selected_chunks) < 3:
            suggestions.append("Consider increasing top_n parameter to include more sub-segments in context")
        elif len(selected_chunks) > 10:
            suggestions.append("Consider decreasing top_n parameter to focus on most relevant sub-segments")

        # Analyze response quality
        if response_quality < 0.6:
            suggestions.append("Review system prompt and user instruction for clarity")
            suggestions.append("Consider adjusting parameter weights for better content selection")

        # Analyze content characteristics
        avg_density = sum(c.get('content_density', 0) for c in selected_chunks) / max(len(selected_chunks), 1)
        if avg_density < 0.5:
            suggestions.append("Sub-segments show low content density - consider refining extraction criteria")

        return suggestions

    def _generate_summary_report(self, filtered_chunks: List[Dict], reranked_chunks: List[Dict], selected_chunks: List[Dict],
                                context_relevance: float, response_quality: float, noise_reduction: float) -> str:
        """Generate a comprehensive summary report"""
        report = f"""# Pipeline Performance Summary

## Execution Statistics
- **Initial Chunks**: {len(filtered_chunks)}
- **Re-ranked Chunks**: {len(reranked_chunks)}
- **Extracted Sub-segments**: {len(selected_chunks)}
- **Filter Rate**: {len(reranked_chunks)/max(len(filtered_chunks), 1):.1%}
- **Extraction Rate**: {len(selected_chunks)/max(len(reranked_chunks), 1):.1%}

## Quality Metrics
- **Context Relevance**: {context_relevance:.1%}
- **Response Quality**: {response_quality:.1%}
- **Noise Reduction**: {noise_reduction:.1%}

## Content Analysis
- **Average Content Density**: {sum(c.get('content_density', 0) for c in selected_chunks) / max(len(selected_chunks), 1):.2f}
- **Total Context Length**: {sum(len(c['content']) for c in selected_chunks)} characters
- **Average Sub-segment Length**: {sum(len(c['content']) for c in selected_chunks) / max(len(selected_chunks), 1):.0f} characters

## Overall Assessment
{'✅ Excellent' if context_relevance > 0.8 and response_quality > 0.8 else '✅ Good' if context_relevance > 0.6 and response_quality > 0.6 else '⚠️  Needs Improvement'}
"""
        return report

    def _calculate_pipeline_efficiency(self, filtered_chunks: List[Dict], selected_chunks: List[Dict]) -> float:
        """Calculate overall pipeline efficiency"""
        if not filtered_chunks:
            return 0.0

        # Efficiency based on reduction while maintaining quality
        reduction_ratio = 1 - (len(selected_chunks) / len(filtered_chunks))
        avg_quality = sum(c.get('composite_score', 0) for c in selected_chunks) / max(len(selected_chunks), 1)

        return min(reduction_ratio * avg_quality, 1.0)

    def _calculate_content_utilization(self, selected_chunks: List[Dict], context: str) -> float:
        """Calculate how effectively content was utilized"""
        if not selected_chunks or not context:
            return 0.0

        total_content = sum(len(c['content']) for c in selected_chunks)
        context_length = len(context)

        # Utilization based on how much of selected content actually made it into context
        utilization = context_length / max(total_content, 1)

        return min(utilization, 1.0)

    def _calculate_metrics(self, retrieved: List[Dict], filtered: List[Dict], selected: List[Dict]) -> Dict[str, float]:
        """Calculate performance metrics"""
        return {
            "retrieval_rate": len(retrieved) / len(self.test_case["chunks"]),
            "filter_rate": len(filtered) / len(retrieved) if retrieved else 0,
            "selection_rate": len(selected) / len(filtered) if filtered else 0,
            "avg_relevance": sum(c["relevance_score"] for c in selected) / len(selected) if selected else 0,
            "avg_composite": sum(c.get("composite_score", 0) for c in selected) / len(selected) if selected else 0
        }

# Initialize data generator and generate mock data
def load_mock_data():
    generator = MockDataGenerator()
    return [generator.generate_test_case(i) for i in range(5)]

def render_prompt_controls(test_case: Dict[str, Any]) -> Dict[str, Any]:
    """Render prompt management interface"""
    st.sidebar.header("Prompt Configuration")

    # System prompt (mandatory)
    with st.sidebar.expander("System Prompt (Mandatory)", expanded=True):
        system_prompt = st.text_area(
            "System Prompt",
            value=test_case["system_prompt"],
            height=150,
            help="This is the core system prompt that defines the AI's behavior and expertise"
        )

    # User instruction (optional)
    with st.sidebar.expander("User Instruction (Optional)", expanded=True):
        use_instruction = st.checkbox("Use Custom User Instruction", value=True)
        if use_instruction:
            user_instruction = st.text_area(
                "User Instruction",
                value=test_case["user_instruction"],
                height=100,
                help="Optional additional instructions to guide the response"
            )
        else:
            user_instruction = ""

    # Show combined prompt preview
    with st.sidebar.expander("Combined Prompt Preview"):
        if user_instruction:
            combined_preview = f"**System:**\n{system_prompt}\n\n**User Instruction:**\n{user_instruction}"
        else:
            combined_preview = f"**System:**\n{system_prompt}"
        st.markdown(combined_preview)

    return {
        "system_prompt": system_prompt,
        "user_instruction": user_instruction if use_instruction else ""
    }

def render_test_case_selector(test_case_manager: TestCaseManager) -> Dict:
    """Render test case selection interface using TestCaseManager"""
    st.sidebar.header("Test Case Selection")

    # Get all test cases from manager
    test_cases = test_case_manager.test_cases

    if not test_cases:
        st.sidebar.warning("No test cases available. Please create test cases in Test Case Management mode.")
        # Return a default empty test case
        return {
            "id": "default",
            "name": "No Test Cases",
            "query": "No test cases available",
            "system_prompt": "You are a helpful assistant.",
            "user_instruction": "",
            "expected_answer": "",
            "chunks": [],
            "rerank_params": {
                "semantic_weight": 0.5,
                "freshness_weight": 0.2,
                "quality_weight": 0.3,
                "relevance_threshold": 0.6,
                "top_n": 10
            }
        }

    # Create selection options with better formatting
    case_options = {}
    for case in test_cases:
        domain = case.get("domain", "general")
        difficulty = case.get("difficulty_level", "medium")
        chunks_count = len(case.get("chunks", []))
        display_name = f"{case['name']} ({domain}, {difficulty}, {chunks_count} chunks)"
        case_options[display_name] = case

    selected_name = st.sidebar.selectbox("Select Test Case", list(case_options.keys()))
    selected_case = case_options[selected_name]

    # Ensure rerank_params exists for compatibility with Pipeline Testing
    if "rerank_params" not in selected_case:
        selected_case["rerank_params"] = {
            "semantic_weight": 0.5,
            "freshness_weight": 0.2,
            "quality_weight": 0.3,
            "relevance_threshold": 0.6,
            "top_n": 10
        }

    # Display test case info
    with st.sidebar.expander("Test Case Details", expanded=True):
        st.markdown(f"**ID:** `{selected_case['id']}`")
        st.markdown(f"**Name:** {selected_case['name']}")
        st.markdown(f"**Domain:** {selected_case.get('domain', 'general')}")
        st.markdown(f"**Difficulty:** {selected_case.get('difficulty_level', 'medium')}")
        st.markdown(f"**Chunks:** {len(selected_case.get('chunks', []))}")
        if selected_case.get('tags'):
            st.markdown(f"**Tags:** {', '.join(selected_case['tags'])}")

        # Show creation/update info
        if selected_case.get('created_at'):
            created_date = selected_case['created_at'][:10] if isinstance(selected_case['created_at'], str) else selected_case['created_at'].strftime('%Y-%m-%d')
            st.markdown(f"**Created:** {created_date}")
        if selected_case.get('updated_at'):
            updated_date = selected_case['updated_at'][:10] if isinstance(selected_case['updated_at'], str) else selected_case['updated_at'].strftime('%Y-%m-%d')
            st.markdown(f"**Updated:** {updated_date}")

    # Show prompt summary
    with st.sidebar.expander("Prompt Summary"):
        st.markdown("**System Prompt:**")
        st.text(selected_case.get("system_prompt", "")[:100] + "...")
        st.markdown("**User Instruction:**")
        user_instruction = selected_case.get("user_instruction", "")
        if user_instruction:
            st.text(user_instruction[:100] + "...")
        else:
            st.text("None (optional)")

    # Show query preview
    with st.sidebar.expander("Query Preview"):
        st.markdown("**Query:**")
        st.text(selected_case.get("query", "")[:150] + "...")

    # Convert test case to expected format for pipeline
    pipeline_case = {
        "id": selected_case["id"],
        "name": selected_case["name"],
        "query": selected_case.get("query", ""),
        "system_prompt": selected_case.get("system_prompt", ""),
        "user_instruction": selected_case.get("user_instruction", ""),
        "expected_answer": selected_case.get("expected_answer", ""),
        "chunks": selected_case.get("chunks", []),
        # Add compatibility fields
        "topic": selected_case.get("domain", "general"),
        "purpose": selected_case.get("description", ""),
        "status": "active",
        "model_version": "v1.0",
        "created_at": datetime.now(),
        "rerank_params": {
            "semantic_weight": 0.5,
            "freshness_weight": 0.2,
            "quality_weight": 0.3,
            "relevance_threshold": 0.6,
            "top_n": 10
        }
    }

    return pipeline_case

def render_parameter_controls(params: Dict[str, Any]) -> Dict[str, Any]:
    """Render parameter tuning interface"""
    st.sidebar.header("Parameter Tuning")

    with st.sidebar.expander("Re-ranking Weights", expanded=True):
        semantic_weight = st.slider("Semantic Weight", 0.0, 1.0, params["semantic_weight"], 0.05)
        freshness_weight = st.slider("Freshness Weight", 0.0, 1.0, params["freshness_weight"], 0.05)
        quality_weight = st.slider("Quality Weight", 0.0, 1.0, params["quality_weight"], 0.05)

        # Normalize weights to sum to 1
        total = semantic_weight + freshness_weight + quality_weight
        semantic_weight /= total
        freshness_weight /= total
        quality_weight /= total

    with st.sidebar.expander("Thresholds & Limits"):
        relevance_threshold = st.slider("Relevance Threshold", 0.0, 1.0, params["relevance_threshold"], 0.05)
        top_n = st.slider("Top-N Selection", 1, 20, params["top_n"], 1)

    return {
        "semantic_weight": semantic_weight,
        "freshness_weight": freshness_weight,
        "quality_weight": quality_weight,
        "relevance_threshold": relevance_threshold,
        "top_n": top_n
    }

def render_step_by_step_pipeline(simulator: RAGPipelineSimulator, params: Dict[str, Any], prompts: Dict[str, str] = None):
    """Render the step-by-step pipeline visualization with manual controls"""
    if prompts is None:
        prompts = {"system_prompt": "", "user_instruction": ""}
    st.header("🔄 Step-by-Step Pipeline Control")

    # Step progress indicator
    progress = (simulator.current_step / len(simulator.pipeline_steps)) * 100
    st.markdown(f"""
    <div class="step-progress">
        <div class="step-progress-bar" style="width: {progress}%"></div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([3, 1])
    with col1:
        st.write(f"**Progress:** Step {simulator.current_step} of {len(simulator.pipeline_steps)}")
    with col2:
        st.write(f"**{progress:.0f}%** Complete")

    # Step control buttons
    st.subheader("Step Controls")

    # Create columns for step buttons
    cols = st.columns(len(simulator.step_names))

    for i, (col, step_name) in enumerate(zip(cols, simulator.step_names)):
        with col:
            # Determine button state
            if i < simulator.current_step:
                button_type = "secondary"
                button_text = f"✓ {step_name}"
                disabled = True
            elif i == simulator.current_step:
                button_type = "primary"
                button_text = f"▶ {step_name}"
                disabled = False
            else:
                button_type = "secondary"
                button_text = step_name
                disabled = True

            # Create button
            if st.button(button_text, key=f"step_{i}", type=button_type, disabled=disabled, use_container_width=True):
                if i == simulator.current_step:
                    # Execute this step
                    try:
                        result = simulator.execute_step(i, params, prompts["system_prompt"], prompts["user_instruction"])
                        st.session_state.pipeline_result = simulator.get_all_results()
                        st.rerun()
                    except ValueError as e:
                        st.error(f"Error: {e}")

    # Control buttons
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        if st.button("⏮️ Reset", type="secondary"):
            simulator.reset()
            if "pipeline_result" in st.session_state:
                del st.session_state.pipeline_result
            st.rerun()

    with col2:
        if st.button("⏭️ Run All", type="primary"):
            # Run all remaining steps
            for i in range(simulator.current_step, len(simulator.pipeline_steps)):
                try:
                    simulator.execute_step(i, params, prompts["system_prompt"], prompts["user_instruction"])
                except ValueError as e:
                    st.error(f"Error at step {i}: {e}")
                    break
            st.session_state.pipeline_result = simulator.get_all_results()
            st.rerun()

    with col3:
        if st.button("🔄 Rerun Current", type="secondary", disabled=simulator.current_step == 0):
            if simulator.current_step > 0:
                simulator.current_step -= 1
                try:
                    simulator.execute_step(simulator.current_step, params, prompts["system_prompt"], prompts["user_instruction"])
                    st.session_state.pipeline_result = simulator.get_all_results()
                    st.rerun()
                except ValueError as e:
                    st.error(f"Error: {e}")

    with col4:
        if st.button("⏩ Skip to End", type="secondary"):
            # Run all steps at once
            for i in range(len(simulator.pipeline_steps)):
                try:
                    simulator.execute_step(i, params, prompts["system_prompt"], prompts["user_instruction"])
                except ValueError as e:
                    st.error(f"Error at step {i}: {e}")
                    break
            st.session_state.pipeline_result = simulator.get_all_results()
            st.rerun()

    # Display completed steps
    st.subheader("Step Results")

    if simulator.step_results:
        for i, (step_key, step_result) in enumerate(simulator.step_results.items()):
            step_display_name = simulator.step_names[i]

            # Determine step status
            if i < simulator.current_step - 1:
                status_class = "processed"
                status_icon = "✅"
            elif i == simulator.current_step - 1:
                status_class = "current"
                status_icon = "🔄"
            else:
                status_class = ""
                status_icon = "⏳"

            with st.expander(f"{status_icon} Step {i+1}: {step_display_name}", expanded=(i == simulator.current_step - 1)):
                # Display step-specific content
                if step_key == "filtering":
                    col1, col2 = st.columns([1, 3])
                    with col1:
                        st.metric("Loaded Chunks", step_result["count"])
                    with col2:
                        retrieved_count = len(simulator.intermediate_data["retrieved_chunks"]) if simulator.intermediate_data["retrieved_chunks"] else 0
                        st.markdown(f"**Description:** {step_result['count']} initial chunks loaded successfully")

                elif step_key == "reranking":
                    col1, col2 = st.columns([1, 3])
                    with col1:
                        st.metric("Re-ranked Chunks", step_result["count"])
                    with col2:
                        st.markdown(f"**Description:** Re-ranked {step_result['count']} chunks using custom weights")

                    # Show re-ranking weights used
                    st.markdown("**Re-ranking Weights:**")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Semantic", f"{params['semantic_weight']:.2f}")
                    with col2:
                        st.metric("Freshness", f"{params['freshness_weight']:.2f}")
                    with col3:
                        st.metric("Quality", f"{params['quality_weight']:.2f}")

                    # Show top reranked chunks
                    st.markdown("**Top Re-ranked Chunks:")
                    for j, chunk in enumerate(step_result["chunks"][:3]):
                        with st.expander(f"Chunk {j+1}: {chunk['title']}"):
                            col1, col2 = st.columns([1, 1])
                            with col1:
                                st.markdown(f"**Composite Score:** <span class='score-indicator score-{get_score_class(chunk.get('composite_score', 0))}'>{chunk.get('composite_score', 0):.3f}</span>", unsafe_allow_html=True)
                                st.markdown(f"**Relevance:** <span class='score-indicator score-{get_score_class(chunk['relevance_score'])}'>{chunk['relevance_score']:.3f}</span>", unsafe_allow_html=True)
                            with col2:
                                st.markdown(f"**Freshness:** <span class='score-indicator score-{get_score_class(chunk['freshness_score'])}'>{chunk['freshness_score']:.3f}</span>", unsafe_allow_html=True)
                                st.markdown(f"**Quality:** <span class='score-indicator score-{get_score_class(chunk['quality_score'])}'>{chunk['quality_score']:.3f}</span>", unsafe_allow_html=True)

                            # Show content preview
                            st.markdown("**Content Preview:**")
                            st.markdown(chunk['content'][:400] + "..." if len(chunk['content']) > 400 else chunk['content'])

                elif step_key == "selection":
                    col1, col2 = st.columns([1, 3])
                    with col1:
                        st.metric("Extracted Sub-segments", step_result["count"])
                    with col2:
                        st.markdown(f"**Description:** Extracted {step_result['count']} relevant sub-segments from reranked chunks to reduce noise")

                    st.markdown(f"**Top-N Setting:** {params['top_n']}")
                    st.markdown(f"**Extraction Rate:** {step_result['count']/len(simulator.intermediate_data['reranked_chunks']):.1%}")
                    st.markdown(f"**Avg Content Density:** {sum(c.get('content_density', 0) for c in step_result['chunks']) / len(step_result['chunks']):.2f}")

                    # Show extracted sub-segments with source analysis
                    st.markdown("**📋 Extracted Sub-segments Analysis:")
                    for j, sub_segment in enumerate(step_result["chunks"]):
                        with st.expander(f"Sub-segment {j+1}: {sub_segment['title']} (from {sub_segment.get('original_title', 'Unknown')})"):
                            col1, col2 = st.columns([3, 1])
                            with col1:
                                st.markdown(f"**Content:**")
                                st.markdown(sub_segment['content'])
                            with col2:
                                st.markdown("**Extraction Metrics:**")
                                st.markdown(f"**Source Chunk:** {sub_segment.get('original_title', 'Unknown')}")
                                st.markdown(f"**Content Density:** {sub_segment.get('content_density', 0):.2f}")
                                st.markdown(f"**Line Count:** {sub_segment.get('line_count', 0)}")
                                st.markdown(f"**Final Score:** {sub_segment.get('composite_score', 0):.3f}")
                                st.markdown(f"**Original Relevance:** {sub_segment.get('relevance_score', 0):.3f}")

                                # Show scoring breakdown
                                st.markdown("**Scoring Breakdown:**")
                                st.markdown(f"- Original Relevance: {sub_segment.get('relevance_score', 0):.3f}")
                                st.markdown(f"- Content Density: {sub_segment.get('content_density', 0):.2f}")
                                st.markdown(f"- Structural Bonus: {'1.2x' if sub_segment.get('content_density', 0) > 0.5 else '1.0x'}")
                                st.markdown(f"- **Final: {sub_segment.get('composite_score', 0):.3f}**")

                elif step_key == "context":
                    st.markdown("**Assembled Context:**")
                    st.text_area("Context", step_result[:1000] + "..." if len(step_result) > 1000 else step_result, height=200)
                    st.markdown(f"**Context Length:** {len(step_result)} characters")
                    st.markdown(f"**Chunks Used:** {len(simulator.intermediate_data['selected_chunks'])}")

                elif step_key == "response":
                    st.markdown("**Generated Response:**")
                    st.text_area("Response", step_result, height=150)
                    st.markdown(f"**Model Used:** {simulator.test_case['model_version']}")
                elif step_key == "analysis":
                    st.markdown("**📊 Comprehensive Results Analysis**")

                    # Create tabs for different analysis views
                    analysis_tabs = st.tabs(["📈 Performance Metrics", "📋 Summary Report", "💡 Insights & Suggestions"])

                    with analysis_tabs[0]:
                        st.subheader("Performance Metrics")
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Context Relevance", f"{step_result['context_relevance']:.1%}")
                        with col2:
                            st.metric("Response Quality", f"{step_result['response_quality']:.1%}")
                        with col3:
                            st.metric("Noise Reduction", f"{step_result['noise_reduction']:.1%}")
                        with col4:
                            st.metric("Pipeline Efficiency", f"{step_result['pipeline_efficiency']:.1%}")

                        # Additional metrics
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Content Utilization", f"{step_result['content_utilization']:.1%}")
                        with col2:
                            st.metric("Content Density", f"{sum(c.get('content_density', 0) for c in simulator.intermediate_data['selected_chunks']) / len(simulator.intermediate_data['selected_chunks']):.2f}")

                        # Performance visualization
                        st.subheader("Performance Visualization")
                        metrics_df = pd.DataFrame({
                            'Metric': ['Context Relevance', 'Response Quality', 'Noise Reduction', 'Pipeline Efficiency', 'Content Utilization'],
                            'Score': [
                                step_result['context_relevance'],
                                step_result['response_quality'],
                                step_result['noise_reduction'],
                                step_result['pipeline_efficiency'],
                                step_result['content_utilization']
                            ]
                        })

                        fig = go.Figure(data=[go.Bar(x=metrics_df['Metric'], y=metrics_df['Score'])])
                        fig.update_layout(title="Overall Performance Metrics", yaxis=dict(range=[0, 1]), height=400)
                        st.plotly_chart(fig, use_container_width=True)

                    with analysis_tabs[1]:
                        st.subheader("Summary Report")
                        st.markdown(step_result['summary_report'])

                        # Download report button
                        if st.button("📥 Download Summary Report"):
                            st.download_button(
                                label="Download Report",
                                data=step_result['summary_report'],
                                file_name=f"pipeline_analysis_{simulator.test_case['id']}.md",
                                mime="text/markdown"
                            )

                    with analysis_tabs[2]:
                        st.subheader("Key Insights")
                        for insight in step_result['insights']:
                            st.markdown(f"- {insight}")

                        st.subheader("💡 Improvement Suggestions")
                        for i, suggestion in enumerate(step_result['suggestions'], 1):
                            st.markdown(f"{i}. {suggestion}")

                        if not step_result['suggestions']:
                            st.success("✅ No specific improvement suggestions - pipeline is performing well!")

    # Display metrics if available
    if simulator.current_step >= 2:  # After selection step (step 2 in 0-based indexing, but analysis is step 5)
        pipeline_result = simulator.get_all_results()
        metrics = pipeline_result["metrics"]

        st.subheader("Performance Metrics")
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Filter Rate", f"{metrics['filter_rate']:.2%}")
        with col2:
            st.metric("Selection Rate", f"{metrics['selection_rate']:.2%}")
        with col3:
            st.metric("Avg Relevance", f"{metrics['avg_relevance']:.3f}")
        with col4:
            st.metric("Avg Composite", f"{metrics['avg_composite']:.3f}")

def render_pipeline_visualization(pipeline_result: Dict[str, Any]):
    """Render the 5-step pipeline visualization (legacy method)"""
    st.header("🔄 Pipeline Visualization")

    steps = pipeline_result["steps"]
    metrics = pipeline_result["metrics"]

    # Create pipeline steps
    for i, (step_name, step_data) in enumerate(steps.items()):
        step_display_name = step_name.replace("_", " ").title()

        with st.expander(f"Step {i+1}: {step_display_name}", expanded=(i == 0)):
            if "count" in step_data:
                col1, col2 = st.columns([1, 2])
                with col1:
                    st.metric("Items Count", step_data["count"])
                with col2:
                    st.markdown(f"**Description:** {step_display_name} processed {step_data['count']} items")
            elif step_name == "context":
                st.markdown("**Assembled Context:**")
                st.text_area("Context", step_data[:1000] + "..." if len(step_data) > 1000 else step_data, height=200)
            elif step_name == "response":
                st.markdown("**Generated Response:**")
                st.text_area("Response", step_data, height=150)

    # Display metrics
    st.subheader("Performance Metrics")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Filter Rate", f"{metrics['filter_rate']:.2%}")
    with col2:
        st.metric("Selection Rate", f"{metrics['selection_rate']:.2%}")
    with col3:
        st.metric("Avg Relevance", f"{metrics['avg_relevance']:.3f}")
    with col4:
        st.metric("Avg Composite", f"{metrics['avg_composite']:.3f}")

def get_score_class(score: float) -> str:
    """Get CSS class for score indication"""
    if score >= 0.8:
        return "high"
    elif score >= 0.6:
        return "medium"
    else:
        return "low"

def render_initial_test_case_data(simulator: RAGPipelineSimulator):
    """Display initial test case data including chunks, metadata, and prompts"""
    test_case = simulator.test_case

    # Create tabs for different data views
    tab1, tab2, tab3 = st.tabs(["📝 Prompts & Query", "📊 Chunk Overview", "🔍 Chunk Details"])

    with tab1:
        st.subheader("Prompts and Query")
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**System Prompt (Mandatory):**")
            st.text_area("System Prompt", test_case["system_prompt"], height=200, disabled=True)

        with col2:
            st.markdown("**User Instruction (Optional):**")
            if test_case["user_instruction"]:
                st.text_area("User Instruction", test_case["user_instruction"], height=100, disabled=True)
            else:
                st.info("No user instruction provided")

        st.markdown("**User Query:**")
        st.text_area("Query", test_case["query"], height=80, disabled=True)

    with tab2:
        st.subheader("Chunk Overview")
        chunks = test_case["chunks"]

        # Chunk statistics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Chunks", len(chunks))
        with col2:
            avg_rating = sum(c["user_rating"] for c in chunks) / len(chunks)
            st.metric("Avg Rating", f"{avg_rating:.1f}/5")
        with col3:
            total_length = sum(len(c["content"]) for c in chunks)
            st.metric("Total Content Length", f"{total_length:,} chars")
        with col4:
            avg_length = sum(len(c["content"]) for c in chunks) / len(chunks)
            st.metric("Avg Chunk Length", f"{avg_length:.0f} chars")

        # Content overview
        st.subheader("Content Overview")

        # Show date range
        publish_dates = [c["publish_time"] for c in chunks]
        effective_dates = [c["effective_time"] for c in chunks]

        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"**Publish Date Range:** {min(publish_dates)} to {max(publish_dates)}")
        with col2:
            st.markdown(f"**Effective Date Range:** {min(effective_dates)} to {max(effective_dates)}")

        # Word frequency analysis
        st.subheader("Content Themes")

        # Simple word frequency analysis (excluding common words)
        all_content = " ".join([c["content"] for c in chunks])
        words = [word.lower() for word in all_content.split() if len(word) > 4]

        # Filter out common words
        common_words = {'concept', 'implementation', 'approach', 'process', 'method', 'technique', 'strategy', 'framework', 'analysis', 'development'}
        words = [word for word in words if word not in common_words and word.isalpha()]

        word_freq = {}
        for word in words:
            word_freq[word] = word_freq.get(word, 0) + 1

        # Show top 8 most frequent words
        top_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:8]
        top_words_df = pd.DataFrame(top_words, columns=['Word', 'Frequency'])

        fig = go.Figure(data=[go.Bar(x=top_words_df['Word'], y=top_words_df['Frequency'])])
        fig.update_layout(title="Top 8 Most Frequent Words", height=400)
        st.plotly_chart(fig, use_container_width=True)

        # Sequential content preview section
        st.subheader("📚 Sequential Content Preview")
        st.markdown("Click on segments to view content and metadata:")

        # Show first 5 segments by default, with option to see more
        preview_count = st.slider("Number of segments to preview:", 3, len(chunks), 5, key="preview_count")

        for i, chunk in enumerate(chunks[:preview_count], 1):
            with st.expander(f"Segment {i}: {chunk['title']}", expanded=(i <= 2)):
                col1, col2 = st.columns([3, 1])

                with col1:
                    st.markdown(f"**Content Preview:**")
                    st.markdown(chunk['content'])

                with col2:
                    st.markdown("**Metadata:**")
                    st.markdown(f"**Published:** {chunk['publish_time']}")
                    st.markdown(f"**Effective:** {chunk['effective_time']}")
                    st.markdown(f"**Rating:** {'⭐' * chunk['user_rating']} ({chunk['user_rating']}/5)")
                    st.markdown(f"**Length:** {len(chunk['content'])} chars")
                    st.markdown(f"**Chunk ID:** {chunk['id']}")

    with tab3:
        st.subheader("Chunk Details")
        chunks = simulator.intermediate_data["retrieved_chunks"]

        # Filtering and sorting options
        col1, col2, col3 = st.columns(3)
        with col1:
            sort_by = st.selectbox("Sort by:", ["Relevance", "Freshness", "Quality", "Rating", "Title"], key="tab3_sort_by")
        with col2:
            sort_order = st.selectbox("Order:", ["Descending", "Ascending"], key="tab3_sort_order")
        with col3:
            min_rating = st.slider("Min Rating", 0, 5, 0, key="tab3_min_rating")

        # Apply filters and sorting
        filtered_chunks = [c for c in chunks if c.get("user_rating", 0) >= min_rating]

        if sort_by == "Relevance":
            filtered_chunks.sort(key=lambda x: x.get("relevance_score", 0), reverse=(sort_order == "Descending"))
        elif sort_by == "Freshness":
            filtered_chunks.sort(key=lambda x: x.get("freshness_score", 0), reverse=(sort_order == "Descending"))
        elif sort_by == "Quality":
            filtered_chunks.sort(key=lambda x: x.get("quality_score", 0), reverse=(sort_order == "Descending"))
        elif sort_by == "Rating":
            filtered_chunks.sort(key=lambda x: x.get("user_rating", 0), reverse=(sort_order == "Descending"))
        elif sort_by == "Title":
            filtered_chunks.sort(key=lambda x: x.get("title", ""), reverse=(sort_order == "Descending"))

        st.markdown(f"**Showing {len(filtered_chunks)} of {len(chunks)} chunks**")

        # Display chunks in a grid
        chunks_per_row = 2
        for i in range(0, len(filtered_chunks), chunks_per_row):
            cols = st.columns(chunks_per_row)
            for j in range(chunks_per_row):
                if i + j < len(filtered_chunks):
                    chunk = filtered_chunks[i + j]
                    with cols[j]:
                        with st.expander(f"{chunk['title']}", expanded=(i + j < 2)):
                            st.markdown(f"**ID:** {chunk['id']}")
                            st.markdown(f"**Rating:** {'⭐' * chunk['user_rating']}")

                            # Note: Scores will be displayed after re-ranking step

                            st.markdown("**Content Preview:**")
                            st.markdown(chunk['content'][:300] + "...")

                            # Metadata
                            with st.expander("Metadata"):
                                # Format dates safely
                                def format_date(date_str):
                                    if isinstance(date_str, str):
                                        try:
                                            # Parse ISO format string and format as YYYY-MM-DD
                                            from datetime import datetime
                                            return datetime.fromisoformat(date_str.replace('Z', '+00:00')).strftime('%Y-%m-%d')
                                        except:
                                            return date_str[:10] if len(date_str) >= 10 else date_str
                                    elif hasattr(date_str, 'strftime'):
                                        return date_str.strftime('%Y-%m-%d')
                                    else:
                                        return str(date_str)

                                st.markdown(f"**Published:** {format_date(chunk['publish_time'])}")
                                st.markdown(f"**Effective:** {format_date(chunk['effective_time'])}")
                                st.markdown(f"**Expires:** {format_date(chunk['expiration_time'])}")

def render_chunk_comparison(selected_chunks: List[Dict]):
    """Render side-by-side comparison of chunks"""
    st.header("📊 Extracted Sub-segments Analysis")

    if not selected_chunks:
        st.warning("No sub-segments extracted with current parameters.")
        return

    # Display score distribution
    st.subheader("Score Distribution")

    scores_df = pd.DataFrame(selected_chunks)
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Relevance Scores', 'Freshness Scores', 'Quality Scores', 'Composite Scores'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )

    fig.add_trace(go.Scatter(x=list(range(len(selected_chunks))), y=scores_df['relevance_score'],
                            mode='lines+markers', name='Relevance'), row=1, col=1)
    fig.add_trace(go.Scatter(x=list(range(len(selected_chunks))), y=scores_df['freshness_score'],
                            mode='lines+markers', name='Freshness'), row=1, col=2)
    fig.add_trace(go.Scatter(x=list(range(len(selected_chunks))), y=scores_df['quality_score'],
                            mode='lines+markers', name='Quality'), row=2, col=1)
    fig.add_trace(go.Scatter(x=list(range(len(selected_chunks))), y=scores_df['composite_score'],
                            mode='lines+markers', name='Composite'), row=2, col=2)

    fig.update_layout(height=600, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

    # Display chunk details
    st.subheader("Extracted Sub-segments Details")

    for i, chunk in enumerate(selected_chunks):
        with st.expander(f"Sub-segment {i+1}: {chunk['title']}"):
            col1, col2, col3 = st.columns([2, 1, 1])

            with col1:
                st.markdown(f"**Sub-segment Title:** {chunk['title']}")
                if chunk.get('is_sub_segment', False):
                    st.markdown(f"**From:** {chunk.get('original_title', 'Unknown')}")
                    st.markdown(f"**Content Density:** {chunk.get('content_density', 0):.2f}")
                st.markdown("**Content Preview:**")
                st.markdown(chunk['content'][:500] + "...")

            with col2:
                st.markdown("**Scores:**")
                st.markdown(f"- Relevance: <span class='score-indicator score-high'>{chunk['relevance_score']:.3f}</span>", unsafe_allow_html=True)
                st.markdown(f"- Freshness: <span class='score-indicator score-medium'>{chunk['freshness_score']:.3f}</span>", unsafe_allow_html=True)
                st.markdown(f"- Quality: <span class='score-indicator score-high'>{chunk['quality_score']:.3f}</span>", unsafe_allow_html=True)

                # Calculate composite score if missing (for backward compatibility)
                if 'composite_score' not in chunk:
                    chunk['composite_score'] = (chunk['relevance_score'] * 0.5 + chunk['freshness_score'] * 0.2 + chunk['quality_score'] * 0.3)

                st.markdown(f"- Composite: <span class='score-indicator score-{get_score_class(chunk['composite_score'])}'>{chunk['composite_score']:.3f}</span>", unsafe_allow_html=True)

            with col3:
                st.markdown("**Metadata:**")
                st.markdown(f"**Rating:** {'⭐' * chunk['user_rating']}")

                # Format dates safely
                def format_date(date_str):
                    if isinstance(date_str, str):
                        try:
                            # Parse ISO format string and format as YYYY-MM-DD
                            from datetime import datetime
                            return datetime.fromisoformat(date_str.replace('Z', '+00:00')).strftime('%Y-%m-%d')
                        except:
                            return date_str[:10] if len(date_str) >= 10 else date_str
                    elif hasattr(date_str, 'strftime'):
                        return date_str.strftime('%Y-%m-%d')
                    else:
                        return str(date_str)

                st.markdown(f"**Published:** {format_date(chunk['publish_time'])}")
                st.markdown(f"**Effective:** {format_date(chunk['effective_time'])}")

def render_results_analysis(test_case: Dict[str, Any], pipeline_result: Dict[str, Any]):
    """Render results analysis and comparison"""
    st.header("📈 Results Analysis")

    # Expected vs Actual comparison
    st.subheader("Expected vs Actual Response")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Expected Answer:**")
        st.markdown(test_case.get("expected_answer", "No expected answer provided"))

    with col2:
        st.markdown("**Generated Response:**")
        st.markdown(pipeline_result["steps"]["response"])

    # Incorrect answer analysis (if test case failed)
    if test_case.get("status") == "fail":
        st.subheader("❌ Failure Analysis")

        with st.expander("Incorrect Answer Details"):
            st.markdown("**Incorrect Answer:**")
            st.markdown(test_case.get("incorrect_answer", {}).get("answer", "No incorrect answer available"))

            st.markdown("**User Concerns:**")
            for concern in test_case.get("incorrect_answer", {}).get("user_concerns", []):
                st.markdown(f"- {concern}")

            st.markdown("**User Comments:**")
            st.markdown(test_case.get("incorrect_answer", {}).get("user_comments", "No comments available"))

    # Performance improvement suggestions
    st.subheader("💡 Parameter Optimization Suggestions")

    metrics = pipeline_result["metrics"]
    suggestions = []

    if metrics["retrieval_rate"] < 0.5:
        suggestions.append("Consider lowering the relevance threshold to retrieve more documents")
    if metrics["filter_rate"] < 0.3:
        suggestions.append("Current threshold may be too restrictive, consider relaxing it")
    if metrics["avg_relevance"] < 0.6:
        suggestions.append("Selected chunks have low relevance, consider adjusting weights")
    if len(pipeline_result["steps"]["selection"]["chunks"]) < 5:
        suggestions.append("Too few chunks selected, consider increasing top_n value")

    if suggestions:
        for suggestion in suggestions:
            st.markdown(f"- {suggestion}")
    else:
        st.markdown("✅ Current parameters appear well-balanced for this test case")


def render_test_case_management_ui():
    """Render test case management interface"""
    st.header("📚 Test Case Management")

    # Test case source configuration (always show in sidebar)
    test_source_options = {
        "Local Test Cases (Default)": "test_cases_local.json",
        "Real Test Cases Collection": "real_test_cases_collection.json",
        "Complete Enhanced Test Suite": "COMPLETE_TEST_SUITE.json",
        "Sample Reference": "sample_test_case_reference.json"
    }

    st.sidebar.header("Test Case Source (Management)")
    selected_source = st.sidebar.selectbox(
        "Select Test Case Source:",
        options=list(test_source_options.keys()),
        index=0,
        key="mgmt_test_source"
    )

    test_case_source = test_source_options[selected_source]

    # Add reload button
    if st.sidebar.button("🔄 Reload Test Cases", key="reload_mgmt_tests"):
        # Force reload by clearing session state
        if "test_case_manager" in st.session_state:
            del st.session_state.test_case_manager
        if "current_mgmt_test_source" in st.session_state:
            del st.session_state.current_mgmt_test_source
        st.rerun()

    # Show current source and test case count
    if "test_case_manager" in st.session_state:
        current_count = len(st.session_state.test_case_manager.test_cases)
        st.sidebar.caption(f"📄 Current: {current_count} test cases loaded")

    # Check if we need to reload test cases due to source change
    reload_needed = (
        "test_case_manager" not in st.session_state or
        st.session_state.get("current_mgmt_test_source") != test_case_source
    )

    if reload_needed:
        # Check for BigQuery configuration in secrets
        use_bigquery = False
        project_id = None
        dataset_id = None

        if BIGQUERY_AVAILABLE:
            use_bigquery = st.secrets.get("BIGQUERY_USE", False)
            if use_bigquery:
                project_id = st.secrets.get("BIGQUERY_PROJECT_ID")
                dataset_id = st.secrets.get("BIGQUERY_DATASET_ID")

        st.session_state.test_case_manager = TestCaseManager(
            use_bigquery=use_bigquery,
            project_id=project_id,
            dataset_id=dataset_id,
            test_case_source=test_case_source
        )
        st.session_state.current_mgmt_test_source = test_case_source

        # Show success message
        st.sidebar.success(f"✅ Loaded {len(st.session_state.test_case_manager.test_cases)} test cases from {selected_source}")

    manager = st.session_state.test_case_manager

    # Storage configuration
    with st.expander("⚙️ Storage Configuration", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            storage_type = manager.get_statistics().get("storage_type", "Local")
            st.metric("Storage Type", storage_type)
        with col2:
            stats = manager.get_statistics()
            st.metric("Total Test Cases", stats["total"])

        if BIGQUERY_AVAILABLE:
            if st.button("Switch to BigQuery" if not manager.use_bigquery else "Switch to Local"):
                # Toggle storage type
                new_use_bigquery = not manager.use_bigquery
                if new_use_bigquery:
                    project_id = st.text_input("GCP Project ID:", value=st.secrets.get("BIGQUERY_PROJECT_ID", ""))
                    dataset_id = st.text_input("BigQuery Dataset ID:", value=st.secrets.get("BIGQUERY_DATASET_ID", ""))
                    if project_id and dataset_id:
                        st.session_state.test_case_manager = TestCaseManager(
                            use_bigquery=True,
                            project_id=project_id,
                            dataset_id=dataset_id
                        )
                        st.rerun()
                else:
                    st.session_state.test_case_manager = TestCaseManager(use_bigquery=False)
                    st.rerun()

    # Tabs for different management functions
    tab1, tab2, tab3, tab4 = st.tabs(["📋 Browse", "➕ Create", "📥 Import/Export", "📊 Statistics"])

    with tab1:
        render_test_case_browser(manager)

    with tab2:
        render_test_case_creator(manager)

    with tab3:
        render_test_case_import_export(manager)

    with tab4:
        render_test_case_statistics(manager)


def render_test_case_browser(manager):
    """Render test case browsing and searching interface"""
    st.subheader("Browse Test Cases")

    # Search and filters
    col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
    with col1:
        search_query = st.text_input("🔍 Search test cases:", "")
    with col2:
        domain_filter = st.selectbox("Domain:", ["All"] + list(set(tc.get("domain", "") for tc in manager.test_cases)))
    with col3:
        difficulty_filter = st.selectbox("Difficulty:", ["All"] + list(set(tc.get("difficulty_level", "") for tc in manager.test_cases)))
    with col4:
        sort_by = st.selectbox("Sort by:", ["Name", "Created", "Updated"])

    # Apply filters
    filters = {}
    if domain_filter != "All":
        filters["domain"] = domain_filter
    if difficulty_filter != "All":
        filters["difficulty"] = difficulty_filter

    filtered_test_cases = manager.search_test_cases(query=search_query, **filters)

    # Sort results
    if sort_by == "Name":
        filtered_test_cases.sort(key=lambda x: x.get("name", "").lower())
    elif sort_by == "Created":
        filtered_test_cases.sort(key=lambda x: x.get("created_at", ""), reverse=True)
    elif sort_by == "Updated":
        filtered_test_cases.sort(key=lambda x: x.get("updated_at", ""), reverse=True)

    # Display results
    st.markdown(f"**Found {len(filtered_test_cases)} test cases**")

    for test_case in filtered_test_cases:
        with st.expander(f"📄 {test_case.get('name', 'Unnamed')}"):
            col1, col2 = st.columns([3, 1])

            with col1:
                st.markdown(f"**ID:** `{test_case['id']}`")
                st.markdown(f"**Description:** {test_case.get('description', 'No description')}")
                st.markdown(f"**Domain:** {test_case.get('domain', 'General')}")
                st.markdown(f"**Difficulty:** {test_case.get('difficulty_level', 'Medium')}")
                st.markdown(f"**Tags:** {', '.join(test_case.get('tags', []))}")
                st.markdown(f"**Chunks:** {len(test_case.get('chunks', []))}")
                if test_case.get('created_at'):
                    st.markdown(f"**Created:** {test_case['created_at'][:10]}")
                if test_case.get('updated_at'):
                    st.markdown(f"**Updated:** {test_case['updated_at'][:10]}")

            with col2:
                # Action buttons
                if st.button("✏️ Edit", key=f"edit_{test_case['id']}"):
                    st.session_state.editing_test_case = test_case
                    st.rerun()

                if st.button("🗑️ Delete", key=f"delete_{test_case['id']}", type="secondary"):
                    if manager.delete_test_case(test_case['id']):
                        st.success("Test case deleted successfully!")
                        st.rerun()

                if st.button("👁️ View", key=f"view_{test_case['id']}"):
                    st.session_state.viewing_test_case = test_case
                    st.rerun()

            # Query preview
            with st.expander("🔍 Query Preview"):
                st.markdown(f"**Query:** {test_case.get('query', 'No query')}")
                st.markdown(f"**Expected Answer:** {test_case.get('expected_answer', 'No expected answer')[:200]}...")

    # Handle editing
    if "editing_test_case" in st.session_state:
        render_test_case_editor(manager, st.session_state.editing_test_case)


def render_test_case_creator(manager):
    """Render test case creation interface"""
    st.subheader("Create New Test Case")

    with st.form("create_test_case_form"):
        # Basic information
        st.markdown("### Basic Information")
        col1, col2 = st.columns(2)
        with col1:
            name = st.text_input("Test Case Name*", value="New Test Case")
            domain = st.text_input("Domain*", value="general")
        with col2:
            difficulty = st.selectbox("Difficulty Level*", ["beginner", "intermediate", "advanced", "expert"])
            tags = st.text_input("Tags (comma-separated)", value="")

        # Description
        description = st.text_area("Description", value="")

        # Query and prompts
        st.markdown("### Query and Prompts")
        query = st.text_area("User Query*", value="What is the main topic?", height=100)
        system_prompt = st.text_area("System Prompt*", value="You are a helpful assistant.", height=100)
        user_instruction = st.text_area("User Instruction", value="Answer the question thoroughly.", height=100)
        expected_answer = st.text_area("Expected Answer", value="", height=150)

        # Chunks
        st.markdown("### Content Chunks")

        # Initialize chunks in session state
        if "creating_chunks" not in st.session_state:
            st.session_state.creating_chunks = []

        # Simple chunk input (moved outside form to avoid button conflicts)
        st.info("⚠️ Chunk builder moved outside form. Please add chunks using the interface below the form.")

        # Display current chunks
        if st.session_state.creating_chunks:
            st.markdown("#### Current Chunks")
            for i, chunk in enumerate(st.session_state.creating_chunks):
                with st.expander(f"Chunk {i+1}: {chunk.get('title', 'Untitled')}"):
                    col1, col2 = st.columns([4, 1])
                    with col1:
                        st.markdown(f"**ID:** {chunk.get('id', 'N/A')}")
                        st.markdown(f"**Title:** {chunk.get('title', 'Untitled')}")
                        st.markdown(f"**User Rating:** {'⭐' * chunk.get('user_rating', 3)}")
                        st.markdown(f"**Published:** {chunk.get('publish_time', 'N/A')}")
                        st.markdown(f"**Effective:** {chunk.get('effective_time', 'N/A')}")
                        st.markdown(f"**Expires:** {chunk.get('expiration_time', 'N/A')}")
                        st.markdown("**Content:**")
                        st.markdown(chunk.get('content', '')[:200] + "..." if len(chunk.get('content', '')) > 200 else chunk.get('content', ''))
                    with col2:
                        if st.button("🗑️", key=f"remove_chunk_{i}", help="Remove chunk"):
                            st.session_state.creating_chunks.pop(i)
                            st.rerun()
        else:
            st.info("No chunks added yet. Use the chunk builder below to add content chunks.")

        # Submit button
        col1, col2 = st.columns([1, 1])
        with col1:
            submitted = st.form_submit_button("Create Test Case", type="primary")
        with col2:
            if st.form_submit_button("Cancel"):
                st.rerun()

        if submitted:
            try:
                # Validate required fields
                if not name or not query or not system_prompt:
                    st.error("Please fill in all required fields (*)")
                    return

                if not st.session_state.creating_chunks:
                    st.error("Please add at least one content chunk")
                    return

                # Create test case data
                test_case_data = {
                    "name": name,
                    "description": description,
                    "query": query,
                    "system_prompt": system_prompt,
                    "user_instruction": user_instruction,
                    "expected_answer": expected_answer,
                    "chunks": st.session_state.creating_chunks,
                    "domain": domain,
                    "difficulty_level": difficulty,
                    "tags": [tag.strip() for tag in tags.split(",") if tag.strip()] if tags else []
                }

                # Create test case
                test_case_id = manager.create_test_case(test_case_data)
                st.success(f"Test case created successfully! ID: {test_case_id}")

                # Clear session state
                st.session_state.creating_chunks = []
                st.rerun()

            except Exception as e:
                st.error(f"Error creating test case: {e}")

    # Add chunk builder interface outside the form
    st.markdown("---")
    st.subheader("Add Content Chunks")
    render_chunk_builder_for_test_case_creation()


def render_chunk_builder_for_test_case_creation():
    """Render chunk builder interface specifically for test case creation"""
    with st.expander("📝 Create New Chunk", expanded=True):
        # Initialize form data if not exists
        if 'create_chunk_form_data' not in st.session_state:
            st.session_state.create_chunk_form_data = {
                'id': '',
                'title': '',
                'content': '',
                'user_rating': 3,
                'publish_time': datetime.now().strftime('%Y-%m-%dT%H:%M:%S'),
                'effective_time': datetime.now().strftime('%Y-%m-%dT00:00:00'),
                'expiration_time': (datetime.now() + timedelta(days=365)).strftime('%Y-%m-%dT23:59:59')
            }

        # Form fields
        col1, col2 = st.columns(2)
        with col1:
            chunk_id = st.text_input("Chunk ID*", key="create_chunk_id", value=st.session_state.create_chunk_form_data['id'],
                                    help="Unique identifier for the chunk (e.g., 'predictive_healthcare_002')")
            title = st.text_input("Chunk Title*", key="create_chunk_title", value=st.session_state.create_chunk_form_data['title'])
        with col2:
            user_rating = st.slider("User Rating", 1, 5, key="create_user_rating", value=st.session_state.create_chunk_form_data['user_rating'])

        content = st.text_area("Chunk Content*", key="create_chunk_content", value=st.session_state.create_chunk_form_data['content'],
                               height=200, help="Markdown content for the chunk")

        # Date and time fields
        st.markdown("### Timing Information")
        col1, col2, col3 = st.columns(3)

        with col1:
            publish_date = st.date_input("Publish Date*", key="create_publish_date",
                                        value=datetime.strptime(st.session_state.create_chunk_form_data['publish_time'], '%Y-%m-%dT%H:%M:%S').date(),
                                        help="When the content was published")
            publish_time = st.time_input("Publish Time", key="create_publish_time_only",
                                        value=datetime.strptime(st.session_state.create_chunk_form_data['publish_time'], '%Y-%m-%dT%H:%M:%S').time(),
                                        help="Time of publication")

        with col2:
            effective_date = st.date_input("Effective Date*", key="create_effective_date",
                                           value=datetime.strptime(st.session_state.create_chunk_form_data['effective_time'], '%Y-%m-%dT%H:%M:%S').date(),
                                           help="When the content becomes effective")
            effective_time = st.time_input("Effective Time", key="create_effective_time_only",
                                           value=datetime.strptime(st.session_state.create_chunk_form_data['effective_time'], '%Y-%m-%dT%H:%M:%S').time(),
                                           help="Time when content becomes effective")

        with col3:
            expiration_date = st.date_input("Expiration Date*", key="create_expiration_date",
                                            value=datetime.strptime(st.session_state.create_chunk_form_data['expiration_time'], '%Y-%m-%dT%H:%M:%S').date(),
                                            help="When the content expires")
            expiration_time = st.time_input("Expiration Time", key="create_expiration_time_only",
                                           value=datetime.strptime(st.session_state.create_chunk_form_data['expiration_time'], '%Y-%m-%dT%H:%M:%S').time(),
                                           help="Time when content expires")

        # Update form data
        st.session_state.create_chunk_form_data.update({
            'id': chunk_id,
            'title': title,
            'content': content,
            'user_rating': user_rating,
            'publish_time': datetime.combine(publish_date, publish_time).strftime('%Y-%m-%dT%H:%M:%S'),
            'effective_time': datetime.combine(effective_date, effective_time).strftime('%Y-%m-%dT%H:%M:%S'),
            'expiration_time': datetime.combine(expiration_date, expiration_time).strftime('%Y-%m-%dT%H:%M:%S')
        })

        # Add chunk button
        col1, col2 = st.columns([1, 1])
        with col1:
            if st.button("➕ Add Chunk to Test Case", key="create_add_chunk"):
                # Validation
                if not st.session_state.create_chunk_form_data['id']:
                    st.error("❌ Chunk ID is required!")
                elif not st.session_state.create_chunk_form_data['title']:
                    st.error("❌ Title is required!")
                elif not st.session_state.create_chunk_form_data['content']:
                    st.error("❌ Content is required!")
                else:
                    new_chunk = {
                        'id': st.session_state.create_chunk_form_data['id'],
                        'title': st.session_state.create_chunk_form_data['title'],
                        'content': st.session_state.create_chunk_form_data['content'],
                        'user_rating': st.session_state.create_chunk_form_data['user_rating'],
                        'publish_time': st.session_state.create_chunk_form_data['publish_time'],
                        'effective_time': st.session_state.create_chunk_form_data['effective_time'],
                        'expiration_time': st.session_state.create_chunk_form_data['expiration_time']
                    }
                    st.session_state.creating_chunks.append(new_chunk)
                    # Reset form
                    st.session_state.create_chunk_form_data = {
                        'id': '',
                        'title': '',
                        'content': '',
                        'user_rating': 3,
                        'publish_time': datetime.now().strftime('%Y-%m-%dT%H:%M:%S'),
                        'effective_time': datetime.now().strftime('%Y-%m-%dT00:00:00'),
                        'expiration_time': (datetime.now() + timedelta(days=365)).strftime('%Y-%m-%dT23:59:59')
                    }
                    st.success("✅ Chunk added to test case!")
                    st.rerun()

        with col2:
            if st.button("🗑️ Clear Form", key="create_clear_form"):
                st.session_state.create_chunk_form_data = {
                    'id': '',
                    'title': '',
                    'content': '',
                    'user_rating': 3,
                    'publish_time': datetime.now().strftime('%Y-%m-%dT%H:%M:%S'),
                    'effective_time': datetime.now().strftime('%Y-%m-%dT00:00:00'),
                    'expiration_time': (datetime.now() + timedelta(days=365)).strftime('%Y-%m-%dT23:59:59')
                }
                st.rerun()


def render_chunk_builder():
    """Render an easy-to-use chunk builder interface"""
    st.markdown("#### 📝 Add New Chunk")

    with st.expander("Chunk Builder", expanded=True):
        # Initialize form state
        if "chunk_form_data" not in st.session_state:
            st.session_state.chunk_form_data = {
                "title": "",
                "content": "",
                "user_rating": 5,
                "use_auto_dates": True,
                "custom_publish_date": None,
                "custom_effective_date": None,
                "custom_expiration_date": None
            }

        # Chunk form
        col1, col2 = st.columns([2, 1])

        with col1:
            title = st.text_input("Chunk Title*", value=st.session_state.chunk_form_data["title"])
            content = st.text_area("Content*", value=st.session_state.chunk_form_data["content"], height=150)

        with col2:
            user_rating = st.slider("User Rating", 1, 5, st.session_state.chunk_form_data["user_rating"])
            use_auto_dates = st.checkbox("Auto-generate dates", value=st.session_state.chunk_form_data["use_auto_dates"])

            if not use_auto_dates:
                st.markdown("**Custom Dates**")
                custom_publish_date = st.date_input("Publish Date", value=st.session_state.chunk_form_data["custom_publish_date"] or datetime.now().date())
                custom_effective_date = st.date_input("Effective Date", value=st.session_state.chunk_form_data["custom_effective_date"] or datetime.now().date())
                custom_expiration_date = st.date_input("Expiration Date", value=st.session_state.chunk_form_data["custom_expiration_date"] or (datetime.now() + timedelta(days=365)).date())

        # Update form state
        st.session_state.chunk_form_data.update({
            "title": title,
            "content": content,
            "user_rating": user_rating,
            "use_auto_dates": use_auto_dates,
            "custom_publish_date": custom_publish_date if not use_auto_dates else None,
            "custom_effective_date": custom_effective_date if not use_auto_dates else None,
            "custom_expiration_date": custom_expiration_date if not use_auto_dates else None
        })

        # Quick templates
        st.markdown("**Quick Templates:**")
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("📚 Tutorial"):
                title = st.session_state.chunk_form_data.get('title', 'Concept Overview') or 'Concept Overview'
                tutorial_content = f"""# {title}

## Key Points

**1. Definition**
- Clear explanation of the concept
- Key characteristics and features

**2. Applications**
- Practical uses and examples
- Real-world implementation

**3. Best Practices**
- Recommended approaches
- Common pitfalls to avoid

This tutorial provides a comprehensive overview of {title} with practical guidance for implementation."""
                st.session_state.chunk_form_data.update({
                    "title": f"Tutorial: {title}",
                    "content": tutorial_content
                })
                st.rerun()

        with col2:
            if st.button("🔬 Technical"):
                title = st.session_state.chunk_form_data.get('title', 'Implementation Details') or 'Implementation Details'
                technical_content = f"""# Technical Implementation

## Architecture Overview

### Core Components
- **Main Module**: Primary functionality implementation
- **Supporting Modules**: Helper utilities and dependencies
- **Configuration**: Setup and customization options

### Code Structure

```python
def main_function():
    \"\"\"
    Main implementation function
    Handles core business logic
    \"\"\"
    # Implementation details
    pass
```

### Performance Considerations
- Time complexity analysis
- Memory usage optimization
- Scalability considerations

## Integration Guide

This technical documentation provides detailed implementation instructions for developers working with {title}."""
                st.session_state.chunk_form_data.update({
                    "title": f"Technical: {title}",
                    "content": technical_content
                })
                st.rerun()

        with col3:
            if st.button("📊 Analysis"):
                title = st.session_state.chunk_form_data.get('title', 'Performance Review') or 'Performance Review'
                analysis_content = f"""# Performance Analysis

## Key Metrics

### Quantitative Results
- **Accuracy**: 95.4% success rate
- **Efficiency**: 40% improvement over baseline
- **Scalability**: Handles 10,000+ concurrent requests
- **Reliability**: 99.9% uptime

### Comparative Analysis

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| Speed | 2.3s | 1.1s | 52% faster |
| Accuracy | 87% | 95% | 8% increase |
| Cost | $0.15 | $0.09 | 40% reduction |

### Benchmark Results
- Outperforms competing solutions by 35%
- Meets all SLA requirements
- Successfully handles edge cases

## Recommendations

Based on the analysis of {title}, we recommend:
1. Continue current optimization trajectory
2. Expand to additional use cases
3. Monitor long-term performance metrics

This analysis demonstrates significant improvements across all key performance indicators."""
                st.session_state.chunk_form_data.update({
                    "title": f"Analysis: {title}",
                    "content": analysis_content
                })
                st.rerun()

        # Add chunk button
        col1, col2 = st.columns([1, 1])
        with col1:
            if st.button("➕ Add Chunk", type="primary", disabled=not title or not content):
                # Generate chunk data
                chunk_id = f"chunk_{len(st.session_state.creating_chunks) + 1:03d}"

                if use_auto_dates:
                    publish_time = datetime.now()
                    effective_time = publish_time + timedelta(days=15)
                    expiration_time = publish_time + timedelta(days=365)
                else:
                    publish_time = datetime.combine(custom_publish_date, datetime.min.time())
                    effective_time = datetime.combine(custom_effective_date, datetime.min.time())
                    expiration_time = datetime.combine(custom_expiration_date, datetime.min.time())

                new_chunk = {
                    "id": chunk_id,
                    "title": title,
                    "content": content,
                    "user_rating": user_rating,
                    "publish_time": publish_time.isoformat(),
                    "effective_time": effective_time.isoformat(),
                    "expiration_time": expiration_time.isoformat()
                }

                st.session_state.creating_chunks.append(new_chunk)

                # Reset form
                st.session_state.chunk_form_data = {
                    "title": "",
                    "content": "",
                    "user_rating": 5,
                    "use_auto_dates": True,
                    "custom_publish_date": None,
                    "custom_effective_date": None,
                    "custom_expiration_date": None
                }

                st.success(f"Chunk '{title}' added successfully!")
                st.rerun()

        with col2:
            if st.button("🔄 Reset Form"):
                st.session_state.chunk_form_data = {
                    "title": "",
                    "content": "",
                    "user_rating": 5,
                    "use_auto_dates": True,
                    "custom_publish_date": None,
                    "custom_effective_date": None,
                    "custom_expiration_date": None
                }
                st.rerun()

        # Form validation
        if not title:
            st.warning("Please enter a chunk title")
        if not content:
            st.warning("Please enter chunk content")


def render_test_case_editor(manager, test_case):
    """Render test case editing interface"""
    st.subheader(f"Edit Test Case: {test_case.get('name', 'Unnamed')}")

    with st.form(f"edit_test_case_form_{test_case['id']}"):
        # Basic information
        col1, col2 = st.columns(2)
        with col1:
            name = st.text_input("Test Case Name*", value=test_case.get('name', ''))
            domain = st.text_input("Domain*", value=test_case.get('domain', 'general'))
        with col2:
            difficulty = st.selectbox("Difficulty Level*", ["beginner", "intermediate", "advanced", "expert"],
                                      index=["beginner", "intermediate", "advanced", "expert"].index(test_case.get('difficulty_level', 'intermediate')))
            tags = st.text_input("Tags (comma-separated)", value=", ".join(test_case.get('tags', [])))

        description = st.text_area("Description", value=test_case.get('description', ''))

        # Query and prompts
        query = st.text_area("User Query*", value=test_case.get('query', ''), height=100)
        system_prompt = st.text_area("System Prompt*", value=test_case.get('system_prompt', ''), height=100)
        user_instruction = st.text_area("User Instruction", value=test_case.get('user_instruction', ''), height=100)
        expected_answer = st.text_area("Expected Answer", value=test_case.get('expected_answer', ''), height=150)

        # Chunks - Display current chunks (editing moved outside form)
        st.markdown("### Content Chunks")
        st.info("Current chunks are displayed below. Use the chunk builder outside this form to add/remove chunks.")

        # Initialize session state for editing chunks if not exists
        if 'editing_chunks' not in st.session_state:
            st.session_state.editing_chunks = test_case.get('chunks', []).copy()

        # Display current chunks
        if st.session_state.editing_chunks:
            st.markdown("**Current Chunks:**")
            for i, chunk in enumerate(st.session_state.editing_chunks):
                with st.expander(f"Chunk {i+1}: {chunk.get('title', 'Untitled')}"):
                    st.markdown(f"**Title:** {chunk.get('title', 'Untitled')}")
                    st.markdown(f"**Rating:** {'⭐' * chunk.get('rating', 3)}")
                    st.markdown(f"**Date:** {chunk.get('date', 'N/A')}")
                    st.markdown("**Content:**")
                    st.markdown(chunk.get('content', '')[:200] + "..." if len(chunk.get('content', '')) > 200 else chunk.get('content', ''))
        else:
            st.warning("No chunks in this test case. Use the chunk builder below to add chunks.")

        # Submit buttons
        col1, col2, col3 = st.columns([1, 1, 1])
        with col1:
            submitted = st.form_submit_button("Save Changes", type="primary")
        with col2:
            if st.form_submit_button("Cancel"):
                del st.session_state.editing_test_case
                st.rerun()
        with col3:
            if st.form_submit_button("Delete", type="secondary"):
                if manager.delete_test_case(test_case['id']):
                    st.success("Test case deleted successfully!")
                    del st.session_state.editing_test_case
                    st.rerun()

        if submitted:
            try:
                # Validate chunks
                chunks = st.session_state.editing_chunks
                if not isinstance(chunks, list) or len(chunks) == 0:
                    raise ValueError("At least one chunk is required")

                # Update test case
                updates = {
                    "name": name,
                    "description": description,
                    "query": query,
                    "system_prompt": system_prompt,
                    "user_instruction": user_instruction,
                    "expected_answer": expected_answer,
                    "chunks": chunks,
                    "domain": domain,
                    "difficulty_level": difficulty,
                    "tags": [tag.strip() for tag in tags.split(",") if tag.strip()] if tags else []
                }

                if manager.update_test_case(test_case['id'], updates):
                    st.success("Test case updated successfully!")
                    # Clean up session state
                    if 'editing_test_case' in st.session_state:
                        del st.session_state.editing_test_case
                    if 'editing_chunks' in st.session_state:
                        del st.session_state.editing_chunks
                    if 'edit_chunk_form_data' in st.session_state:
                        del st.session_state.edit_chunk_form_data
                    st.rerun()

            except Exception as e:
                st.error(f"Error updating test case: {e}")

    # Add chunk builder interface outside the form for editing
    st.markdown("---")
    render_chunk_builder_for_test_case_editing()


def render_chunk_builder_for_test_case_editing():
    """Render chunk builder interface specifically for test case editing"""
    st.subheader("Edit Content Chunks")

    # Ensure editing_chunks exists
    if 'editing_chunks' not in st.session_state:
        return

    with st.expander("➕ Add New Chunk", expanded=True):
        # Initialize form data if not exists
        if 'edit_chunk_form_data' not in st.session_state:
            st.session_state.edit_chunk_form_data = {
                'id': '',
                'title': '',
                'content': '',
                'user_rating': 3,
                'publish_time': datetime.now().strftime('%Y-%m-%dT%H:%M:%S'),
                'effective_time': datetime.now().strftime('%Y-%m-%dT00:00:00'),
                'expiration_time': (datetime.now() + timedelta(days=365)).strftime('%Y-%m-%dT23:59:59')
            }

        # Form fields
        col1, col2 = st.columns(2)
        with col1:
            chunk_id = st.text_input("Chunk ID*", key="edit_chunk_id", value=st.session_state.edit_chunk_form_data['id'],
                                    help="Unique identifier for the chunk (e.g., 'predictive_healthcare_002')")
            title = st.text_input("Chunk Title*", key="edit_chunk_title", value=st.session_state.edit_chunk_form_data['title'])
        with col2:
            user_rating = st.slider("User Rating", 1, 5, key="edit_user_rating", value=st.session_state.edit_chunk_form_data['user_rating'])

        content = st.text_area("Chunk Content*", key="edit_chunk_content", value=st.session_state.edit_chunk_form_data['content'],
                               height=200, help="Markdown content for the chunk")

        # Date and time fields
        st.markdown("### Timing Information")
        col1, col2, col3 = st.columns(3)

        with col1:
            publish_date = st.date_input("Publish Date*", key="edit_publish_date",
                                        value=datetime.strptime(st.session_state.edit_chunk_form_data['publish_time'], '%Y-%m-%dT%H:%M:%S').date(),
                                        help="When the content was published")
            publish_time = st.time_input("Publish Time", key="edit_publish_time_only",
                                        value=datetime.strptime(st.session_state.edit_chunk_form_data['publish_time'], '%Y-%m-%dT%H:%M:%S').time(),
                                        help="Time of publication")

        with col2:
            effective_date = st.date_input("Effective Date*", key="edit_effective_date",
                                           value=datetime.strptime(st.session_state.edit_chunk_form_data['effective_time'], '%Y-%m-%dT%H:%M:%S').date(),
                                           help="When the content becomes effective")
            effective_time = st.time_input("Effective Time", key="edit_effective_time_only",
                                           value=datetime.strptime(st.session_state.edit_chunk_form_data['effective_time'], '%Y-%m-%dT%H:%M:%S').time(),
                                           help="Time when content becomes effective")

        with col3:
            expiration_date = st.date_input("Expiration Date*", key="edit_expiration_date",
                                            value=datetime.strptime(st.session_state.edit_chunk_form_data['expiration_time'], '%Y-%m-%dT%H:%M:%S').date(),
                                            help="When the content expires")
            expiration_time = st.time_input("Expiration Time", key="edit_expiration_time_only",
                                           value=datetime.strptime(st.session_state.edit_chunk_form_data['expiration_time'], '%Y-%m-%dT%H:%M:%S').time(),
                                           help="Time when content expires")

        # Update form data
        st.session_state.edit_chunk_form_data.update({
            'id': chunk_id,
            'title': title,
            'content': content,
            'user_rating': user_rating,
            'publish_time': datetime.combine(publish_date, publish_time).strftime('%Y-%m-%dT%H:%M:%S'),
            'effective_time': datetime.combine(effective_date, effective_time).strftime('%Y-%m-%dT%H:%M:%S'),
            'expiration_time': datetime.combine(expiration_date, expiration_time).strftime('%Y-%m-%dT%H:%M:%S')
        })

        # Add chunk button
        col1, col2 = st.columns([1, 1])
        with col1:
            if st.button("➕ Add Chunk", key="edit_add_chunk"):
                # Validation
                if not st.session_state.edit_chunk_form_data['id']:
                    st.error("❌ Chunk ID is required!")
                elif not st.session_state.edit_chunk_form_data['title']:
                    st.error("❌ Title is required!")
                elif not st.session_state.edit_chunk_form_data['content']:
                    st.error("❌ Content is required!")
                else:
                    new_chunk = {
                        'id': st.session_state.edit_chunk_form_data['id'],
                        'title': st.session_state.edit_chunk_form_data['title'],
                        'content': st.session_state.edit_chunk_form_data['content'],
                        'user_rating': st.session_state.edit_chunk_form_data['user_rating'],
                        'publish_time': st.session_state.edit_chunk_form_data['publish_time'],
                        'effective_time': st.session_state.edit_chunk_form_data['effective_time'],
                        'expiration_time': st.session_state.edit_chunk_form_data['expiration_time']
                    }
                    st.session_state.editing_chunks.append(new_chunk)
                    # Reset form
                    st.session_state.edit_chunk_form_data = {
                        'id': '',
                        'title': '',
                        'content': '',
                        'user_rating': 3,
                        'publish_time': datetime.now().strftime('%Y-%m-%dT%H:%M:%S'),
                        'effective_time': datetime.now().strftime('%Y-%m-%dT00:00:00'),
                        'expiration_time': (datetime.now() + timedelta(days=365)).strftime('%Y-%m-%dT23:59:59')
                    }
                    st.success("✅ Chunk added successfully!")
                    st.rerun()

        with col2:
            if st.button("🗑️ Clear Form", key="edit_clear_form"):
                st.session_state.edit_chunk_form_data = {
                    'id': '',
                    'title': '',
                    'content': '',
                    'user_rating': 3,
                    'publish_time': datetime.now().strftime('%Y-%m-%dT%H:%M:%S'),
                    'effective_time': datetime.now().strftime('%Y-%m-%dT00:00:00'),
                    'expiration_time': (datetime.now() + timedelta(days=365)).strftime('%Y-%m-%dT23:59:59')
                }
                st.rerun()

    # Display current chunks with removal option
    if st.session_state.editing_chunks:
        st.markdown("### Current Chunks (Click to Remove)")
        for i, chunk in enumerate(st.session_state.editing_chunks):
            col1, col2 = st.columns([4, 1])
            with col1:
                with st.expander(f"Chunk {i+1}: {chunk.get('title', 'Untitled')}"):
                    st.markdown(f"**ID:** {chunk.get('id', 'N/A')}")
                    st.markdown(f"**Title:** {chunk.get('title', 'Untitled')}")
                    st.markdown(f"**User Rating:** {'⭐' * chunk.get('user_rating', 3)}")
                    st.markdown(f"**Published:** {chunk.get('publish_time', 'N/A')}")
                    st.markdown(f"**Effective:** {chunk.get('effective_time', 'N/A')}")
                    st.markdown(f"**Expires:** {chunk.get('expiration_time', 'N/A')}")
                    st.markdown("**Content:**")
                    st.markdown(chunk.get('content', '')[:200] + "..." if len(chunk.get('content', '')) > 200 else chunk.get('content', ''))
            with col2:
                if st.button("🗑️", key=f"remove_editing_chunk_{i}", help="Remove chunk"):
                    st.session_state.editing_chunks.pop(i)
                    st.success("✅ Chunk removed!")
                    st.rerun()


def render_test_case_import_export(manager):
    """Render import/export functionality"""
    st.subheader("Import/Export Test Cases")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Import Test Cases")
        st.info("Import test cases from JSON files. Supports both single test case and collection formats.")

        uploaded_file = st.file_uploader(
            "Choose a JSON file",
            type=['json'],
            help="Upload a JSON file containing test cases"
        )

        if uploaded_file:
            # Save uploaded file temporarily
            import tempfile
            with tempfile.NamedTemporaryFile(delete=False, suffix='.json') as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_file_path = tmp_file.name

            try:
                # Import test cases
                imported_count = manager.import_test_cases(tmp_file_path)
                if imported_count > 0:
                    st.success(f"Successfully imported {imported_count} test cases!")
                    st.rerun()
                else:
                    st.warning("No test cases were imported. Please check the file format.")
            except Exception as e:
                st.error(f"Error importing test cases: {e}")
            finally:
                # Clean up temporary file
                import os
                os.unlink(tmp_file_path)

    with col2:
        st.markdown("### Export Test Cases")
        st.info("Export all test cases in JSON or CSV format.")

        export_format = st.selectbox("Export Format:", ["JSON", "CSV"])

        if st.button("Export Test Cases", type="primary"):
            try:
                exported_data = manager.export_test_cases(export_format.lower())

                # Create download button
                st.download_button(
                    label=f"Download Test Cases ({export_format})",
                    data=exported_data,
                    file_name=f"test_cases_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{export_format.lower()}",
                    mime="application/json" if export_format == "JSON" else "text/csv"
                )
            except Exception as e:
                st.error(f"Error exporting test cases: {e}")

        # Sample data info
        st.markdown("### Sample Data")
        st.markdown("Download sample test case formats:")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Download Sample Single"):
                sample_data = {
                    "id": "sample_001",
                    "name": "Sample Test Case",
                    "description": "A sample test case for reference",
                    "query": "What is machine learning?",
                    "system_prompt": "You are an AI assistant.",
                    "user_instruction": "Answer the question.",
                    "expected_answer": "Machine learning is a subset of AI...",
                    "chunks": [
                        {
                            "id": "chunk_001",
                            "title": "ML Introduction",
                            "content": "Machine learning is...",
                            "user_rating": 5,
                            "publish_time": "2024-01-01T00:00:00",
                            "effective_time": "2024-01-01T00:00:00",
                            "expiration_time": "2026-01-01T00:00:00"
                        }
                    ],
                    "domain": "technology",
                    "difficulty_level": "beginner",
                    "tags": ["ML", "AI"]
                }
                st.download_button(
                    label="Single Test Case",
                    data=json.dumps(sample_data, indent=2),
                    file_name="sample_test_case.json",
                    mime="application/json"
                )
        with col2:
            if st.button("Download Sample Collection"):
                sample_collection = {
                    "test_cases_collection": [
                        {
                            "id": "sample_001",
                            "name": "Sample Test Case 1",
                            "description": "First sample",
                            "query": "What is AI?",
                            "system_prompt": "You are helpful.",
                            "chunks": [],
                            "domain": "technology"
                        },
                        {
                            "id": "sample_002",
                            "name": "Sample Test Case 2",
                            "description": "Second sample",
                            "query": "What is ML?",
                            "system_prompt": "You are helpful.",
                            "chunks": [],
                            "domain": "technology"
                        }
                    ]
                }
                st.download_button(
                    label="Test Case Collection",
                    data=json.dumps(sample_collection, indent=2),
                    file_name="sample_test_cases.json",
                    mime="application/json"
                )


def render_test_case_statistics(manager):
    """Render test case statistics and analytics"""
    st.subheader("Test Case Statistics")

    stats = manager.get_statistics()

    if stats["total"] == 0:
        st.info("No test cases available for statistics.")
        return

    # Overview metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Test Cases", stats["total"])
    with col2:
        st.metric("Storage Type", stats["storage_type"])
    with col3:
        st.metric("Avg Chunks/Case", f"{stats['avg_chunks']:.1f}")
    with col4:
        st.metric("Domains", len(stats["domains"]))

    # Domain distribution
    if stats["domains"]:
        st.subheader("Domain Distribution")
        domain_data = pd.DataFrame([
            {"Domain": domain, "Count": count}
            for domain, count in stats["domains"].items()
        ])

        fig = go.Figure(data=go.Bar(x=domain_data["Domain"], y=domain_data["Count"]))
        fig.update_layout(xaxis_title="Domain", yaxis_title="Count", title="Test Cases by Domain")
        st.plotly_chart(fig, use_container_width=True)

    # Difficulty distribution
    if stats["difficulties"]:
        st.subheader("Difficulty Distribution")
        difficulty_data = pd.DataFrame([
            {"Difficulty": difficulty, "Count": count}
            for difficulty, count in stats["difficulties"].items()
        ])

        fig = go.Figure(data=go.Pie(labels=difficulty_data["Difficulty"], values=difficulty_data["Count"]))
        fig.update_layout(title="Test Cases by Difficulty Level")
        st.plotly_chart(fig, use_container_width=True)

    # Recent activity
    st.subheader("Recent Activity")
    recent_test_cases = sorted(manager.test_cases, key=lambda x: x.get("updated_at", ""), reverse=True)[:5]

    for test_case in recent_test_cases:
        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown(f"**{test_case.get('name', 'Unnamed')}**")
            st.markdown(f"_{test_case.get('description', 'No description')}_")
        with col2:
            updated_at = test_case.get('updated_at', '')
            if updated_at:
                st.markdown(f"Updated: {updated_at[:10]}")
        st.divider()


def create_sample_test_case(manager: TestCaseManager):
    """Create a sample test case for demonstration"""
    sample_case = {
        "name": "Sample: Machine Learning Fundamentals",
        "description": "A sample test case demonstrating machine learning concepts",
        "query": "What are the key principles of machine learning?",
        "system_prompt": "You are an expert in machine learning and artificial intelligence. Provide comprehensive, accurate answers based on the provided context.",
        "user_instruction": "Focus on explaining the core principles with practical examples.",
        "expected_answer": "Machine learning is based on principles like data-driven learning, pattern recognition, generalization, optimization, feature engineering, and regularization. These enable AI systems to learn from data and improve performance.",
        "rerank_params": {
            "semantic_weight": 0.5,
            "freshness_weight": 0.2,
            "quality_weight": 0.3,
            "relevance_threshold": 0.6,
            "top_n": 10
        },
        "chunks": [
            {
                "id": "ml_chunk_001",
                "title": "Machine Learning Fundamentals",
                "content": "# Machine Learning Principles\n\nMachine learning represents a paradigm shift from traditional programming, where algorithms learn from data rather than following explicit rules.\n\n## Core Principles\n\n**1. Data-Driven Learning**\n- Algorithms learn from examples rather than explicit programming\n- Quality and quantity of training data directly impact performance\n- Requires representative datasets that capture the problem space\n\n**2. Pattern Recognition**\n- Identifies underlying patterns in complex datasets\n- Uses statistical methods to find correlations and relationships\n- Enables prediction and classification tasks\n\n**3. Generalization**\n- Ability to apply learned patterns to new, unseen data\n- Critical for real-world deployment\n- Measured through validation and testing metrics",
                "user_rating": 5,
                "publish_time": "2024-01-15T10:30:00",
                "effective_time": "2024-02-01T00:00:00",
                "expiration_time": "2026-01-15T23:59:59"
            },
            {
                "id": "ml_chunk_002",
                "title": "ML Applications and Methods",
                "content": "# Machine Learning Applications\n\n## Types of Machine Learning\n\n### Supervised Learning\n- **Classification**: Categorizing data into predefined classes\n- **Regression**: Predicting continuous values\n- **Examples**: Spam detection, price prediction, image classification\n\n### Unsupervised Learning\n- **Clustering**: Grouping similar data points\n- **Dimensionality Reduction**: Reducing feature space\n- **Examples**: Customer segmentation, anomaly detection\n\n### Reinforcement Learning\n- **Agent-based learning**: Learning through interaction\n- **Reward optimization**: Maximizing cumulative rewards\n- **Examples**: Game playing, robotics, recommendation systems\n\n## Real-World Applications\n\n- **Healthcare**: Disease diagnosis, drug discovery, personalized treatment\n- **Finance**: Fraud detection, algorithmic trading, risk assessment\n- **Technology**: Image recognition, natural language processing, autonomous vehicles\n- **Business**: Customer segmentation, demand forecasting, recommendation systems",
                "user_rating": 4,
                "publish_time": "2024-02-20T14:15:00",
                "effective_time": "2024-03-01T00:00:00",
                "expiration_time": "2026-02-20T23:59:59"
            }
        ],
        "domain": "technology",
        "difficulty_level": "intermediate",
        "tags": ["machine learning", "AI", "fundamentals", "tutorial"]
    }

    try:
        test_case_id = manager.create_test_case(sample_case)
        print(f"Sample test case created with ID: {test_case_id}")
    except Exception as e:
        print(f"Error creating sample test case: {e}")


def main():
    """Main application function"""
    st.title("🔍 RAG Pipeline Testing & Performance Tuning Tool")
    st.markdown("*Optimize your RAG system parameters using predefined test cases*")

    # Test case source configuration (always show in sidebar)
    test_source_options = {
        "Local Test Cases (Default)": "test_cases_local.json",
        "Real Test Cases Collection": "real_test_cases_collection.json",
        "Complete Enhanced Test Suite": "COMPLETE_TEST_SUITE.json",
        "Sample Reference": "sample_test_case_reference.json"
    }

    st.sidebar.header("Test Case Source")
    selected_source = st.sidebar.selectbox(
        "Select Test Case Source:",
        options=list(test_source_options.keys()),
        index=0,
        key="pipeline_test_source_select"
    )

    test_case_source = test_source_options[selected_source]

    # Add reload button
    if st.sidebar.button("🔄 Reload Test Cases", key="reload_pipeline_tests"):
        # Force reload by clearing session state
        if "pipeline_test_manager" in st.session_state:
            del st.session_state.pipeline_test_manager
        if "current_test_source" in st.session_state:
            del st.session_state.current_test_source
        if "current_case_id" in st.session_state:
            del st.session_state.current_case_id
        if "step_simulator" in st.session_state:
            del st.session_state.step_simulator
        if "pipeline_result" in st.session_state:
            del st.session_state.pipeline_result
        st.rerun()

    # Show current source and test case count
    if "pipeline_test_manager" in st.session_state:
        current_count = len(st.session_state.pipeline_test_manager.test_cases)
        st.sidebar.caption(f"📄 Current: {current_count} test cases loaded")

    # Check if we need to reload test cases due to source change
    reload_needed = (
        "pipeline_test_manager" not in st.session_state or
        st.session_state.get("current_test_source") != test_case_source
    )

    if reload_needed:
        # Initialize manager with selected source
        st.session_state.pipeline_test_manager = TestCaseManager(
            use_bigquery=False,
            test_case_source=test_case_source
        )
        st.session_state.current_test_source = test_case_source

        # Clear related session state to force refresh
        if "current_case_id" in st.session_state:
            del st.session_state.current_case_id
        if "step_simulator" in st.session_state:
            del st.session_state.step_simulator
        if "pipeline_result" in st.session_state:
            del st.session_state.pipeline_result

        # Show success message
        st.sidebar.success(f"✅ Loaded {len(st.session_state.pipeline_test_manager.test_cases)} test cases from {selected_source}")

    # Create sample test case if no test cases exist
    if not st.session_state.pipeline_test_manager.test_cases:
        create_sample_test_case(st.session_state.pipeline_test_manager)

    # Initialize session state
    if "current_params" not in st.session_state:
        st.session_state.current_params = {
            "semantic_weight": 0.5,
            "freshness_weight": 0.2,
            "quality_weight": 0.3,
            "relevance_threshold": 0.6,
            "top_n": 10
        }

    # Add mode selection
    st.sidebar.header("Application Mode")
    app_mode = st.sidebar.radio(
        "Choose Mode:",
        ["🔍 Pipeline Testing", "📚 Test Case Management"],
        index=0
    )

    if app_mode == "📚 Test Case Management":
        # Show test case management interface
        render_test_case_management_ui()
        return

    # Pipeline analysis mode
    st.sidebar.header("Analysis Mode")
    analysis_mode = st.sidebar.radio(
        "Choose Analysis Mode:",
        ["Step-by-Step Control", "Full Pipeline Analysis"],
        index=0
    )

    # Render sidebar controls
    selected_case = render_test_case_selector(st.session_state.pipeline_test_manager)
    updated_params = render_parameter_controls(selected_case["rerank_params"])
    updated_prompts = render_prompt_controls(selected_case)

    # Update session state
    st.session_state.current_params = updated_params
    st.session_state.current_prompts = updated_prompts

    # Initialize or get simulator
    if "step_simulator" not in st.session_state or st.session_state.get("current_case_id") != selected_case["id"]:
        st.session_state.step_simulator = RAGPipelineSimulator(selected_case)
        st.session_state.current_case_id = selected_case["id"]

    simulator = st.session_state.step_simulator

    # Display initial test case data
    st.header("📋 Test Case Overview")
    render_initial_test_case_data(simulator)

    # Main content area based on mode
    if analysis_mode == "Step-by-Step Control":
        # Step-by-step mode
        render_step_by_step_pipeline(simulator, updated_params, updated_prompts)

        # Show additional analysis when pipeline is complete
        if simulator.current_step >= len(simulator.pipeline_steps):
            if simulator.intermediate_data["selected_chunks"]:
                render_chunk_comparison(simulator.intermediate_data["selected_chunks"])
                pipeline_result = simulator.get_all_results()
                render_results_analysis(selected_case, pipeline_result)
        else:
            st.info("👆 Use the step controls above to progress through the pipeline")

    else:
        # Full pipeline mode (original)
        if st.button("🔄 Run Full Pipeline Analysis", type="primary"):
            # Simulate full pipeline processing
            pipeline_result = simulator.process_pipeline(
                updated_params,
                updated_prompts["system_prompt"],
                updated_prompts["user_instruction"]
            )

            # Store results in session state
            st.session_state.pipeline_result = pipeline_result
            st.session_state.selected_case = selected_case

        # Display results if available
        if "pipeline_result" in st.session_state:
            render_pipeline_visualization(st.session_state.pipeline_result)
            render_chunk_comparison(st.session_state.pipeline_result["steps"]["selection"]["chunks"])
            render_results_analysis(selected_case, st.session_state.pipeline_result)
        else:
            st.info("👆 Select a test case and click 'Run Full Pipeline Analysis' to begin")

if __name__ == "__main__":
    main()