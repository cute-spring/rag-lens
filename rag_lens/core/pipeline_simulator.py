"""
RAG Pipeline Simulator with enhanced performance monitoring
"""

import random
import time
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
import statistics

from ..config.settings import config
from ..utils.logger import get_logger, log_performance
from ..utils.errors import PipelineError, ValidationError
from ..utils.security import security_manager
from .test_case_manager import TestCase, Chunk

logger = get_logger(__name__)


@dataclass
class PipelineMetrics:
    """Pipeline performance metrics"""
    step_durations: Dict[str, float]
    total_duration: float
    retrieval_rate: float
    filter_rate: float
    selection_rate: float
    average_scores: Dict[str, float]
    memory_usage_mb: float
    success: bool
    error_message: Optional[str] = None


@dataclass
class ProcessingStep:
    """Individual pipeline step result"""
    name: str
    status: str  # "pending", "processing", "completed", "error"
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    duration_ms: Optional[float] = None
    input_data: Optional[Dict[str, Any]] = None
    output_data: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    metrics: Optional[Dict[str, Any]] = None


class RAGPipelineSimulator:
    """Simulate RAG pipeline processing with performance monitoring"""

    def __init__(self, test_case: TestCase):
        self.test_case = test_case
        self.current_step = 0
        self.steps = config.pipeline.steps
        self.processing_steps: List[ProcessingStep] = []
        self.intermediate_data: Dict[str, Any] = {}
        self.metrics: Optional[PipelineMetrics] = None
        self._initialize_steps()

    def _initialize_steps(self):
        """Initialize pipeline steps"""
        for step_name in self.steps:
            step = ProcessingStep(
                name=step_name,
                status="pending"
            )
            self.processing_steps.append(step)

    @log_performance(threshold_seconds=0.5)
    def process_pipeline(self, params: Dict[str, float], system_prompt: str, user_instruction: str) -> Dict[str, Any]:
        """Process entire pipeline"""
        logger.info(f"Starting pipeline processing for test case: {self.test_case.id}")

        start_time = time.time()
        step_times = {}

        try:
            # Step 1: Query Processing
            step_times["Query Processing"] = self._process_query_processing(system_prompt, user_instruction)

            # Step 2: Retrieval
            step_times["Retrieval"] = self._process_retrieval()

            # Step 3: Initial Filtering
            step_times["Initial Filtering"] = self._process_initial_filtering(params)

            # Step 4: Re-ranking
            step_times["Re-ranking"] = self._process_reranking(params)

            # Step 5: Final Selection
            step_times["Final Selection"] = self._process_final_selection(params)

            # Step 6: Context Assembly
            step_times["Context Assembly"] = self._process_context_assembly()

            # Step 7: Response Generation
            step_times["Response Generation"] = self._process_response_generation()

            total_duration = time.time() - start_time

            # Calculate metrics
            self.metrics = self._calculate_metrics(step_times, total_duration)

            logger.info(f"Pipeline completed successfully in {total_duration:.2f}s")

            return self.get_all_results()

        except Exception as e:
            logger.error(f"Pipeline processing failed: {e}")
            self.metrics = PipelineMetrics(
                step_durations=step_times,
                total_duration=time.time() - start_time,
                retrieval_rate=0.0,
                filter_rate=0.0,
                selection_rate=0.0,
                average_scores={},
                memory_usage_mb=0.0,
                success=False,
                error_message=str(e)
            )
            raise PipelineError(f"Pipeline processing failed: {e}")

    def _process_query_processing(self, system_prompt: str, user_instruction: str) -> float:
        """Process query analysis and preparation"""
        logger.debug("Processing query analysis")
        start_time = time.time()

        step = self.processing_steps[0]
        step.status = "processing"
        step.start_time = datetime.utcnow()

        try:
            # Sanitize inputs
            sanitized_query = security_manager.sanitize_input(self.test_case.query)
            sanitized_instruction = security_manager.sanitize_input(user_instruction)

            # Simulate query processing
            query_analysis = {
                "original_query": sanitized_query,
                "query_intent": self._analyze_query_intent(sanitized_query),
                "query_complexity": self._assess_query_complexity(sanitized_query),
                "extracted_keywords": self._extract_keywords(sanitized_query),
                "system_prompt": system_prompt,
                "user_instruction": sanitized_instruction
            }

            # Validate processed data
            self._validate_query_processing_output(query_analysis)

            self.intermediate_data["query_analysis"] = query_analysis
            step.output_data = query_analysis
            step.status = "completed"

        except Exception as e:
            step.status = "error"
            step.error_message = str(e)
            logger.error(f"Query processing failed: {e}")
            raise ValidationError(f"Query processing failed: {e}")

        finally:
            step.end_time = datetime.utcnow()
            step.duration_ms = (step.end_time - step.start_time).total_seconds() * 1000

        duration = time.time() - start_time
        logger.debug(f"Query processing completed in {duration:.3f}s")
        return duration

    def _process_retrieval(self) -> float:
        """Simulate document retrieval"""
        logger.debug("Processing document retrieval")
        start_time = time.time()

        step = self.processing_steps[1]
        step.status = "processing"
        step.start_time = datetime.utcnow()

        try:
            # Simulate retrieval with realistic performance
            available_chunks = self.test_case.chunks
            retrieval_count = min(len(available_chunks), random.randint(10, 20))

            # Score and rank chunks
            retrieved_chunks = []
            for chunk in available_chunks:
                score = self._calculate_retrieval_score(chunk, self.test_case.query)
                retrieved_chunks.append({
                    "chunk": chunk,
                    "score": score,
                    "retrieval_method": "vector_similarity"
                })

            # Sort by score and take top results
            retrieved_chunks.sort(key=lambda x: x["score"], reverse=True)
            retrieved_chunks = retrieved_chunks[:retrieval_count]

            retrieval_metrics = {
                "total_available": len(available_chunks),
                "retrieved_count": len(retrieved_chunks),
                "retrieval_rate": len(retrieved_chunks) / len(available_chunks),
                "average_score": statistics.mean([rc["score"] for rc in retrieved_chunks]),
                "retrieval_latency_ms": random.uniform(50, 200)
            }

            self.intermediate_data["retrieved_chunks"] = retrieved_chunks
            self.intermediate_data["retrieval_metrics"] = retrieval_metrics
            step.output_data = retrieval_metrics
            step.status = "completed"

        except Exception as e:
            step.status = "error"
            step.error_message = str(e)
            logger.error(f"Retrieval failed: {e}")
            raise PipelineError(f"Retrieval failed: {e}")

        finally:
            step.end_time = datetime.utcnow()
            step.duration_ms = (step.end_time - step.start_time).total_seconds() * 1000

        duration = time.time() - start_time
        logger.debug(f"Retrieval completed in {duration:.3f}s")
        return duration

    def _process_initial_filtering(self, params: Dict[str, float]) -> float:
        """Process initial filtering based on relevance threshold"""
        logger.debug("Processing initial filtering")
        start_time = time.time()

        step = self.processing_steps[2]
        step.status = "processing"
        step.start_time = datetime.utcnow()

        try:
            retrieved_chunks = self.intermediate_data.get("retrieved_chunks", [])
            relevance_threshold = params.get("relevance_threshold", 0.6)

            # Apply relevance threshold
            filtered_chunks = [
                rc for rc in retrieved_chunks
                if rc["score"] >= relevance_threshold
            ]

            filtering_metrics = {
                "input_count": len(retrieved_chunks),
                "filtered_count": len(filtered_chunks),
                "filter_rate": len(filtered_chunks) / len(retrieved_chunks) if retrieved_chunks else 0,
                "threshold_used": relevance_threshold,
                "average_score_filtered": statistics.mean([rc["score"] for rc in filtered_chunks]) if filtered_chunks else 0
            }

            self.intermediate_data["filtered_chunks"] = filtered_chunks
            self.intermediate_data["filtering_metrics"] = filtering_metrics
            step.output_data = filtering_metrics
            step.status = "completed"

        except Exception as e:
            step.status = "error"
            step.error_message = str(e)
            logger.error(f"Initial filtering failed: {e}")
            raise PipelineError(f"Initial filtering failed: {e}")

        finally:
            step.end_time = datetime.utcnow()
            step.duration_ms = (step.end_time - step.start_time).total_seconds() * 1000

        duration = time.time() - start_time
        logger.debug(f"Initial filtering completed in {duration:.3f}s")
        return duration

    def _process_reranking(self, params: Dict[str, float]) -> float:
        """Process multi-dimensional re-ranking"""
        logger.debug("Processing re-ranking")
        start_time = time.time()

        step = self.processing_steps[3]
        step.status = "processing"
        step.start_time = datetime.utcnow()

        try:
            filtered_chunks = self.intermediate_data.get("filtered_chunks", [])

            # Extract weights
            semantic_weight = params.get("semantic_weight", 0.5)
            freshness_weight = params.get("freshness_weight", 0.2)
            quality_weight = params.get("quality_weight", 0.3)

            # Normalize weights
            total_weight = semantic_weight + freshness_weight + quality_weight
            semantic_weight /= total_weight
            freshness_weight /= total_weight
            quality_weight /= total_weight

            # Calculate composite scores
            reranked_chunks = []
            for rc in filtered_chunks:
                chunk = rc["chunk"]
                semantic_score = rc["score"]
                freshness_score = self._calculate_freshness_score(chunk)
                quality_score = chunk.user_rating / 5.0  # Normalize to 0-1

                composite_score = (
                    semantic_weight * semantic_score +
                    freshness_weight * freshness_score +
                    quality_weight * quality_score
                )

                reranked_chunks.append({
                    "chunk": chunk,
                    "semantic_score": semantic_score,
                    "freshness_score": freshness_score,
                    "quality_score": quality_score,
                    "composite_score": composite_score,
                    "scores": {
                        "semantic": semantic_score,
                        "freshness": freshness_score,
                        "quality": quality_score,
                        "composite": composite_score
                    }
                })

            # Sort by composite score
            reranked_chunks.sort(key=lambda x: x["composite_score"], reverse=True)

            reranking_metrics = {
                "input_count": len(filtered_chunks),
                "reranked_count": len(reranked_chunks),
                "weights": {
                    "semantic": semantic_weight,
                    "freshness": freshness_weight,
                    "quality": quality_weight
                },
                "average_composite_score": statistics.mean([rc["composite_score"] for rc in reranked_chunks]) if reranked_chunks else 0,
                "score_distribution": self._calculate_score_distribution(reranked_chunks)
            }

            self.intermediate_data["reranked_chunks"] = reranked_chunks
            self.intermediate_data["reranking_metrics"] = reranking_metrics
            step.output_data = reranking_metrics
            step.status = "completed"

        except Exception as e:
            step.status = "error"
            step.error_message = str(e)
            logger.error(f"Re-ranking failed: {e}")
            raise PipelineError(f"Re-ranking failed: {e}")

        finally:
            step.end_time = datetime.utcnow()
            step.duration_ms = (step.end_time - step.start_time).total_seconds() * 1000

        duration = time.time() - start_time
        logger.debug(f"Re-ranking completed in {duration:.3f}s")
        return duration

    def _process_final_selection(self, params: Dict[str, float]) -> float:
        """Process final chunk selection"""
        logger.debug("Processing final selection")
        start_time = time.time()

        step = self.processing_steps[4]
        step.status = "processing"
        step.start_time = datetime.utcnow()

        try:
            reranked_chunks = self.intermediate_data.get("reranked_chunks", [])
            top_n = int(params.get("top_n", 10))

            # Select top N chunks
            selected_chunks = reranked_chunks[:top_n]

            selection_metrics = {
                "input_count": len(reranked_chunks),
                "selected_count": len(selected_chunks),
                "selection_rate": len(selected_chunks) / len(reranked_chunks) if reranked_chunks else 0,
                "top_n": top_n,
                "min_score": selected_chunks[-1]["composite_score"] if selected_chunks else 0,
                "max_score": selected_chunks[0]["composite_score"] if selected_chunks else 0,
                "score_range": selected_chunks[0]["composite_score"] - selected_chunks[-1]["composite_score"] if selected_chunks else 0
            }

            self.intermediate_data["selected_chunks"] = selected_chunks
            self.intermediate_data["selection_metrics"] = selection_metrics
            step.output_data = selection_metrics
            step.status = "completed"

        except Exception as e:
            step.status = "error"
            step.error_message = str(e)
            logger.error(f"Final selection failed: {e}")
            raise PipelineError(f"Final selection failed: {e}")

        finally:
            step.end_time = datetime.utcnow()
            step.duration_ms = (step.end_time - step.start_time).total_seconds() * 1000

        duration = time.time() - start_time
        logger.debug(f"Final selection completed in {duration:.3f}s")
        return duration

    def _process_context_assembly(self) -> float:
        """Process context assembly"""
        logger.debug("Processing context assembly")
        start_time = time.time()

        step = self.processing_steps[5]
        step.status = "processing"
        step.start_time = datetime.utcnow()

        try:
            selected_chunks = self.intermediate_data.get("selected_chunks", [])
            query_analysis = self.intermediate_data.get("query_analysis", {})

            # Assemble context from selected chunks
            context_chunks = []
            for i, sc in enumerate(selected_chunks):
                context_chunk = {
                    "id": sc["chunk"].id,
                    "title": sc["chunk"].title,
                    "content": sc["chunk"].content,
                    "score": sc["composite_score"],
                    "position": i + 1,
                    "metadata": {
                        "user_rating": sc["chunk"].user_rating,
                        "publish_time": sc["chunk"].publish_time,
                        "source_type": "document_chunk"
                    }
                }
                context_chunks.append(context_chunk)

            # Build final context
            context = {
                "query": query_analysis.get("original_query", ""),
                "chunks": context_chunks,
                "total_chunks": len(context_chunks),
                "context_size_chars": sum(len(c["content"]) for c in context_chunks),
                "assembly_method": "ranked_concatenation",
                "context_quality_score": statistics.mean([c["score"] for c in context_chunks]) if context_chunks else 0
            }

            assembly_metrics = {
                "context_length": len(context_chunks),
                "context_size_chars": context["context_size_chars"],
                "average_chunk_score": context["context_quality_score"],
                "assembly_time_ms": random.uniform(10, 50)
            }

            self.intermediate_data["context"] = context
            self.intermediate_data["assembly_metrics"] = assembly_metrics
            step.output_data = assembly_metrics
            step.status = "completed"

        except Exception as e:
            step.status = "error"
            step.error_message = str(e)
            logger.error(f"Context assembly failed: {e}")
            raise PipelineError(f"Context assembly failed: {e}")

        finally:
            step.end_time = datetime.utcnow()
            step.duration_ms = (step.end_time - step.start_time).total_seconds() * 1000

        duration = time.time() - start_time
        logger.debug(f"Context assembly completed in {duration:.3f}s")
        return duration

    def _process_response_generation(self) -> float:
        """Process response generation"""
        logger.debug("Processing response generation")
        start_time = time.time()

        step = self.processing_steps[6]
        step.status = "processing"
        step.start_time = datetime.utcnow()

        try:
            context = self.intermediate_data.get("context", {})
            query_analysis = self.intermediate_data.get("query_analysis", {})

            # Simulate response generation
            generated_response = self._generate_simulated_response(
                query=query_analysis.get("original_query", ""),
                context_chunks=context.get("chunks", []),
                system_prompt=query_analysis.get("system_prompt", ""),
                user_instruction=query_analysis.get("user_instruction", "")
            )

            # Calculate response metrics
            response_metrics = {
                "response_length": len(generated_response),
                "generation_time_ms": random.uniform(100, 500),
                "tokens_used": len(generated_response.split()),
                "context_utilization": len(context.get("chunks", [])),
                "response_quality": self._assess_response_quality(generated_response)
            }

            self.intermediate_data["generated_response"] = generated_response
            self.intermediate_data["response_metrics"] = response_metrics
            step.output_data = response_metrics
            step.status = "completed"

        except Exception as e:
            step.status = "error"
            step.error_message = str(e)
            logger.error(f"Response generation failed: {e}")
            raise PipelineError(f"Response generation failed: {e}")

        finally:
            step.end_time = datetime.utcnow()
            step.duration_ms = (step.end_time - step.start_time).total_seconds() * 1000

        duration = time.time() - start_time
        logger.debug(f"Response generation completed in {duration:.3f}s")
        return duration

    def get_all_results(self) -> Dict[str, Any]:
        """Get complete pipeline results"""
        if not self.metrics:
            raise PipelineError("Pipeline not yet processed")

        return {
            "test_case_id": self.test_case.id,
            "metrics": asdict(self.metrics),
            "steps": {
                step.name: {
                    "status": step.status,
                    "duration_ms": step.duration_ms,
                    "output": step.output_data,
                    "error": step.error_message
                }
                for step in self.processing_steps
            },
            "intermediate_data": {
                key: self._serialize_data(value)
                for key, value in self.intermediate_data.items()
            }
        }

    def _serialize_data(self, data: Any) -> Any:
        """Serialize data for JSON output"""
        if isinstance(data, datetime):
            return data.isoformat()
        elif hasattr(data, '__dict__'):
            return {k: self._serialize_data(v) for k, v in data.__dict__.items()}
        elif isinstance(data, list):
            return [self._serialize_data(item) for item in data]
        elif isinstance(data, dict):
            return {k: self._serialize_data(v) for k, v in data.items()}
        else:
            return data

    def _calculate_metrics(self, step_times: Dict[str, float], total_duration: float) -> PipelineMetrics:
        """Calculate pipeline metrics"""
        retrieval_metrics = self.intermediate_data.get("retrieval_metrics", {})
        filtering_metrics = self.intermediate_data.get("filtering_metrics", {})
        selection_metrics = self.intermediate_data.get("selection_metrics", {})
        reranking_metrics = self.intermediate_data.get("reranking_metrics", {})

        # Calculate average scores
        average_scores = {}
        if reranking_metrics:
            weights = reranking_metrics.get("weights", {})
            for score_type, weight in weights.items():
                average_scores[f"avg_{score_type}_weight"] = weight

        # Simulate memory usage
        memory_usage_mb = sum(
            len(str(data)) for data in self.intermediate_data.values()
        ) / (1024 * 1024) * 0.1  # Rough estimate

        return PipelineMetrics(
            step_durations=step_times,
            total_duration=total_duration,
            retrieval_rate=retrieval_metrics.get("retrieval_rate", 0.0),
            filter_rate=filtering_metrics.get("filter_rate", 0.0),
            selection_rate=selection_metrics.get("selection_rate", 0.0),
            average_scores=average_scores,
            memory_usage_mb=memory_usage_mb,
            success=True
        )

    # Helper methods
    def _analyze_query_intent(self, query: str) -> str:
        """Analyze query intent"""
        query_lower = query.lower()
        if any(word in query_lower for word in ["what", "who", "when", "where", "why", "how"]):
            return "information_seeking"
        elif any(word in query_lower for word in ["compare", "difference", "versus"]):
            return "comparison"
        elif any(word in query_lower for word in ["list", "enumerate", "name"]):
            return "listing"
        else:
            return "general"

    def _assess_query_complexity(self, query: str) -> str:
        """Assess query complexity"""
        word_count = len(query.split())
        if word_count <= 5:
            return "simple"
        elif word_count <= 15:
            return "moderate"
        else:
            return "complex"

    def _extract_keywords(self, query: str) -> List[str]:
        """Extract keywords from query"""
        # Simple keyword extraction
        stop_words = {"the", "is", "at", "which", "on", "a", "an", "and", "or", "but", "in", "with"}
        words = query.lower().split()
        keywords = [word for word in words if word not in stop_words and len(word) > 2]
        return keywords[:10]  # Return top 10 keywords

    def _calculate_retrieval_score(self, chunk: Chunk, query: str) -> float:
        """Calculate retrieval score"""
        # Simple TF-IDF-like scoring
        query_words = set(query.lower().split())
        chunk_words = set(chunk.content.lower().split())

        intersection = query_words.intersection(chunk_words)
        union = query_words.union(chunk_words)

        jaccard_similarity = len(intersection) / len(union) if union else 0

        # Add some randomness for realism
        base_score = jaccard_similarity
        noise = random.uniform(-0.1, 0.1)

        return max(0.0, min(1.0, base_score + noise))

    def _calculate_freshness_score(self, chunk: Chunk) -> float:
        """Calculate freshness score"""
        try:
            publish_time = datetime.fromisoformat(chunk.publish_time.replace('Z', '+00:00'))
            age_days = (datetime.utcnow() - publish_time).days

            # Score based on age (newer = higher score)
            if age_days <= 30:
                return 1.0
            elif age_days <= 365:
                return 0.8
            elif age_days <= 730:
                return 0.6
            else:
                return 0.4
        except:
            return 0.5  # Default score for parsing errors

    def _calculate_score_distribution(self, chunks: List[Dict]) -> Dict[str, Any]:
        """Calculate score distribution statistics"""
        if not chunks:
            return {"mean": 0, "median": 0, "std": 0, "min": 0, "max": 0}

        scores = [chunk["composite_score"] for chunk in chunks]
        return {
            "mean": statistics.mean(scores),
            "median": statistics.median(scores),
            "std": statistics.stdev(scores) if len(scores) > 1 else 0,
            "min": min(scores),
            "max": max(scores)
        }

    def _generate_simulated_response(self, query: str, context_chunks: List[Dict], system_prompt: str, user_instruction: str) -> str:
        """Generate simulated response"""
        if not context_chunks:
            return "I don't have enough information to answer your question."

        # Simple response generation based on context
        response_parts = []
        response_parts.append("Based on the available information, ")

        # Add content from top chunks
        for chunk in context_chunks[:3]:  # Use top 3 chunks
            content = chunk["content"]
            # Extract first few sentences
            sentences = content.split('.')[:2]
            response_parts.extend([s.strip() + '.' for s in sentences if s.strip()])

        # Ensure response is relevant to query
        if "how" in query.lower():
            response_parts.append("The process involves several key steps.")
        elif "what" in query.lower():
            response_parts.append("This refers to several important aspects.")
        elif "why" in query.lower():
            response_parts.append("There are multiple reasons for this.")

        return " ".join(response_parts)

    def _assess_response_quality(self, response: str) -> float:
        """Assess response quality"""
        # Simple quality assessment based on length and content
        if len(response) < 50:
            return 0.3
        elif len(response) < 200:
            return 0.7
        else:
            return 0.9

    def _validate_query_processing_output(self, data: Dict[str, Any]) -> None:
        """Validate query processing output"""
        required_fields = ["original_query", "query_intent", "query_complexity"]
        for field in required_fields:
            if field not in data:
                raise ValidationError(f"Missing required field: {field}")

        if not isinstance(data["original_query"], str) or not data["original_query"].strip():
            raise ValidationError("Original query must be a non-empty string")

    def advance_step(self):
        """Advance to next step (for UI interaction)"""
        if self.current_step < len(self.processing_steps):
            self.current_step += 1

    def reset(self):
        """Reset pipeline to initial state"""
        self.current_step = 0
        self.intermediate_data.clear()
        self.metrics = None
        for step in self.processing_steps:
            step.status = "pending"
            step.start_time = None
            step.end_time = None
            step.duration_ms = None
            step.error_message = None