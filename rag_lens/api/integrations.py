"""
API Integration Components for RAG Lens

This module contains integration components that combine multiple API providers
to implement complete pipeline workflows following the standardized contracts.
"""

from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass
import asyncio
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from .providers import APIManager, APIResponse, BaseAPIProvider
from ..config.settings import config
from ..utils.logger import get_logger
from ..utils.errors import APIError, ErrorHandler

logger = get_logger(__name__)


@dataclass
class PipelineStepConfig:
    """Configuration for a pipeline step"""
    step_number: int
    provider_name: str
    enabled: bool = True
    timeout: int = 30
    retry_count: int = 3
    fallback_provider: str = None


@dataclass
class PipelineExecutionContext:
    """Context for pipeline execution"""
    query: str
    system_prompt: str
    test_case_id: str
    execution_id: str
    start_time: float
    metadata: Dict[str, Any]


class PipelineOrchestrator:
    """Orchestrates the execution of pipeline steps across multiple API providers"""

    def __init__(self, api_manager: APIManager):
        self.api_manager = api_manager
        self.step_configs = self._load_step_configs()
        self.execution_history = []

    def _load_step_configs(self) -> Dict[int, PipelineStepConfig]:
        """Load pipeline step configurations from settings"""
        configs = {}

        # Step 0: Query Generation
        configs[0] = PipelineStepConfig(
            step_number=0,
            provider_name=config.pipeline.query_generation_provider,
            timeout=config.pipeline.query_generation_timeout,
            retry_count=config.pipeline.query_generation_retries
        )

        # Step 1: Query Encoding
        configs[1] = PipelineStepConfig(
            step_number=1,
            provider_name=config.pipeline.query_encoding_provider,
            timeout=config.pipeline.query_encoding_timeout,
            retry_count=config.pipeline.query_encoding_retries
        )

        # Step 2: Candidate Generation
        configs[2] = PipelineStepConfig(
            step_number=2,
            provider_name=config.pipeline.candidate_generation_provider,
            timeout=config.pipeline.candidate_generation_timeout,
            retry_count=config.pipeline.candidate_generation_retries
        )

        # Step 3: Candidate Filtering
        configs[3] = PipelineStepConfig(
            step_number=3,
            provider_name=config.pipeline.candidate_filtering_provider,
            timeout=config.pipeline.candidate_filtering_timeout,
            retry_count=config.pipeline.candidate_filtering_retries
        )

        # Step 4: Pairwise Encoding
        configs[4] = PipelineStepConfig(
            step_number=4,
            provider_name=config.pipeline.pairwise_encoding_provider,
            timeout=config.pipeline.pairwise_encoding_timeout,
            retry_count=config.pipeline.pairwise_encoding_retries
        )

        # Step 5: Re-ranking
        configs[5] = PipelineStepConfig(
            step_number=5,
            provider_name=config.pipeline.reranking_provider,
            timeout=config.pipeline.reranking_timeout,
            retry_count=config.pipeline.reranking_retries
        )

        # Step 6: Final Answer Generation
        configs[6] = PipelineStepConfig(
            step_number=6,
            provider_name=config.pipeline.final_answer_provider,
            timeout=config.pipeline.final_answer_timeout,
            retry_count=config.pipeline.final_answer_retries
        )

        return configs

    def execute_pipeline(
        self,
        query: str,
        system_prompt: str,
        test_case_id: str,
        step_to_execute: int = None
    ) -> Dict[str, Any]:
        """Execute the complete pipeline or a specific step"""
        execution_id = f"exec_{int(time.time())}_{test_case_id}"
        context = PipelineExecutionContext(
            query=query,
            system_prompt=system_prompt,
            test_case_id=test_case_id,
            execution_id=execution_id,
            start_time=time.time(),
            metadata={}
        )

        logger.info(f"Starting pipeline execution {execution_id} for test case {test_case_id}")

        if step_to_execute is not None:
            # Execute only the specified step
            return self._execute_single_step(context, step_to_execute)
        else:
            # Execute complete pipeline
            return self._execute_complete_pipeline(context)

    def _execute_single_step(self, context: PipelineExecutionContext, step_number: int) -> Dict[str, Any]:
        """Execute a single pipeline step"""
        step_config = self.step_configs.get(step_number)
        if not step_config or not step_config.enabled:
            return {
                "success": False,
                "error": f"Step {step_number} is not configured or enabled",
                "execution_time": 0
            }

        step_start_time = time.time()
        step_data = self._prepare_step_data(context, step_number)

        try:
            # Execute step with primary provider
            response = self._execute_with_fallback(context, step_config, step_data)

            # Process results
            step_result = {
                "step_number": step_number,
                "success": response.success,
                "data": response.data,
                "execution_time": time.time() - step_start_time,
                "provider_used": step_config.provider_name,
                "error_message": response.error_message,
                "status_code": response.status_code
            }

            # Update context with results
            self._update_context(context, step_number, step_result)

            logger.info(f"Step {step_number} completed in {step_result['execution_time']:.2f}s")

            return {
                "success": True,
                "step_result": step_result,
                "context_data": context.metadata,
                "total_execution_time": time.time() - context.start_time
            }

        except Exception as e:
            error_info = ErrorHandler.handle_error(e, {
                "step": step_number,
                "execution_id": context.execution_id
            })

            logger.error(f"Step {step_number} failed: {error_info['error_message']}")

            return {
                "success": False,
                "error": error_info['error_message'],
                "step_number": step_number,
                "execution_time": time.time() - step_start_time,
                "total_execution_time": time.time() - context.start_time
            }

    def _execute_complete_pipeline(self, context: PipelineExecutionContext) -> Dict[str, Any]:
        """Execute the complete pipeline"""
        results = {}
        pipeline_success = True

        for step_number in range(7):  # Steps 0-6
            step_result = self._execute_single_step(context, step_number)
            results[step_number] = step_result

            if not step_result["success"]:
                pipeline_success = False
                logger.warning(f"Pipeline execution failed at step {step_number}")
                break

        total_time = time.time() - context.start_time

        return {
            "success": pipeline_success,
            "execution_id": context.execution_id,
            "step_results": results,
            "context_data": context.metadata,
            "total_execution_time": total_time,
            "steps_completed": len([r for r in results.values() if r["success"]])
        }

    def _execute_with_fallback(
        self,
        context: PipelineExecutionContext,
        step_config: PipelineStepConfig,
        step_data: Dict[str, Any]
    ) -> APIResponse:
        """Execute step with fallback provider if available"""
        # Try primary provider first
        response = self.api_manager.execute_pipeline_step(
            step_config.provider_name,
            step_config.step_number,
            step_data
        )

        if response.success or not step_config.fallback_provider:
            return response

        # Try fallback provider
        logger.warning(f"Primary provider {step_config.provider_name} failed, trying fallback {step_config.fallback_provider}")
        fallback_response = self.api_manager.execute_pipeline_step(
            step_config.fallback_provider,
            step_config.step_number,
            step_data
        )

        return fallback_response

    def _prepare_step_data(self, context: PipelineExecutionContext, step_number: int) -> Dict[str, Any]:
        """Prepare data for a specific pipeline step"""
        base_data = {
            "query": context.query,
            "system_prompt": context.system_prompt,
            "execution_id": context.execution_id,
            "metadata": context.metadata
        }

        if step_number == 0:  # Query Generation
            return base_data

        elif step_number == 1:  # Query Encoding
            if "queries" in context.metadata:
                base_data["queries"] = context.metadata["queries"]
            return base_data

        elif step_number == 2:  # Candidate Generation
            if "embeddings" in context.metadata:
                base_data["embeddings"] = context.metadata["embeddings"]
            return base_data

        elif step_number == 3:  # Candidate Filtering
            if "candidates" in context.metadata:
                base_data["candidates"] = context.metadata["candidates"]
            return base_data

        elif step_number == 4:  # Pairwise Encoding
            if "filtered_candidates" in context.metadata:
                base_data["candidates"] = context.metadata["filtered_candidates"]
            return base_data

        elif step_number == 5:  # Re-ranking
            if "pairwise_scores" in context.metadata:
                base_data["candidates"] = context.metadata["pairwise_scores"]
            return base_data

        elif step_number == 6:  # Final Answer Generation
            if "reranked_candidates" in context.metadata:
                base_data["context"] = self._format_context_for_final_answer(context.metadata["reranked_candidates"])
            return base_data

        return base_data

    def _update_context(self, context: PipelineExecutionContext, step_number: int, step_result: Dict[str, Any]):
        """Update execution context with step results"""
        if not step_result["success"] or not step_result["data"]:
            return

        data = step_result["data"]

        if step_number == 0 and "queries" in data:
            context.metadata["queries"] = data["queries"]

        elif step_number == 1 and "embeddings" in data:
            context.metadata["embeddings"] = data["embeddings"]

        elif step_number == 2 and "candidates" in data:
            context.metadata["candidates"] = data["candidates"]

        elif step_number == 3 and "filtered_candidates" in data:
            context.metadata["filtered_candidates"] = data["filtered_candidates"]

        elif step_number == 4 and "pairwise_scores" in data:
            context.metadata["pairwise_scores"] = data["pairwise_scores"]

        elif step_number == 5 and "reranked_candidates" in data:
            context.metadata["reranked_candidates"] = data["reranked_candidates"]

        elif step_number == 6 and "answer" in data:
            context.metadata["final_answer"] = data["answer"]

    def _format_context_for_final_answer(self, candidates: List[Dict[str, Any]]) -> str:
        """Format candidates into context string for final answer generation"""
        if not candidates:
            return ""

        context_parts = []
        for i, candidate in enumerate(candidates[:5], 1):  # Use top 5 candidates
            content = candidate.get("content", candidate.get("text", ""))
            if content:
                context_parts.append(f"Document {i}: {content}")

        return "\n\n".join(context_parts)

    def get_execution_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get recent execution history"""
        return self.execution_history[-limit:]

    def get_provider_performance_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get performance statistics for all providers"""
        stats = {}

        for execution in self.execution_history:
            for step_result in execution.get("step_results", {}).values():
                provider = step_result.get("provider_used")
                if provider:
                    if provider not in stats:
                        stats[provider] = {
                            "total_calls": 0,
                            "successful_calls": 0,
                            "failed_calls": 0,
                            "average_response_time": 0,
                            "total_response_time": 0
                        }

                    stats[provider]["total_calls"] += 1
                    stats[provider]["total_response_time"] += step_result.get("execution_time", 0)

                    if step_result.get("success"):
                        stats[provider]["successful_calls"] += 1
                    else:
                        stats[provider]["failed_calls"] += 1

        # Calculate averages
        for provider_stats in stats.values():
            if provider_stats["total_calls"] > 0:
                provider_stats["average_response_time"] = (
                    provider_stats["total_response_time"] / provider_stats["total_calls"]
                )

        return stats


class TestCaseIntegration:
    """Integration component for test case management with APIs"""

    def __init__(self, api_manager: APIManager):
        self.api_manager = api_manager

    def evaluate_test_case(
        self,
        test_case: Dict[str, Any],
        pipeline_orchestrator: PipelineOrchestrator
    ) -> Dict[str, Any]:
        """Evaluate a single test case using the pipeline"""
        test_case_id = test_case.get("id", "unknown")
        query = test_case.get("query", "")
        system_prompt = test_case.get("system_prompt", "")
        expected_answer = test_case.get("expected_answer", "")

        logger.info(f"Evaluating test case {test_case_id}")

        # Execute pipeline
        pipeline_result = pipeline_orchestrator.execute_pipeline(
            query=query,
            system_prompt=system_prompt,
            test_case_id=test_case_id
        )

        # Evaluate results
        evaluation_result = {
            "test_case_id": test_case_id,
            "pipeline_success": pipeline_result["success"],
            "execution_time": pipeline_result.get("total_execution_time", 0),
            "steps_completed": pipeline_result.get("steps_completed", 0),
            "generated_answer": pipeline_result.get("context_data", {}).get("final_answer", ""),
            "expected_answer": expected_answer,
            "similarity_score": 0.0,
            "evaluation_metrics": {}
        }

        # Calculate similarity if we have both answers
        if evaluation_result["generated_answer"] and expected_answer:
            evaluation_result["similarity_score"] = self._calculate_similarity(
                evaluation_result["generated_answer"],
                expected_answer
            )

        # Add detailed evaluation metrics
        evaluation_result["evaluation_metrics"] = self._calculate_evaluation_metrics(
            evaluation_result["generated_answer"],
            expected_answer,
            test_case
        )

        return evaluation_result

    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity score between two texts"""
        # Simple text similarity (can be enhanced with actual semantic similarity)
        import re
        from collections import Counter

        # Preprocess texts
        def preprocess(text):
            text = re.sub(r'[^\w\s]', '', text.lower())
            words = text.split()
            return Counter(words)

        words1 = preprocess(text1)
        words2 = preprocess(text2)

        # Calculate Jaccard similarity
        intersection = sum((words1 & words2).values())
        union = sum((words1 | words2).values())

        return intersection / union if union > 0 else 0.0

    def _calculate_evaluation_metrics(
        self,
        generated_answer: str,
        expected_answer: str,
        test_case: Dict[str, Any]
    ) -> Dict[str, float]:
        """Calculate comprehensive evaluation metrics"""
        metrics = {}

        # Relevance score (based on similarity)
        metrics["relevance_score"] = self._calculate_similarity(generated_answer, expected_answer)

        # Completeness score (based on length ratio and keyword coverage)
        if expected_answer:
            keywords = set(test_case.get("keywords", expected_answer.lower().split()))
            generated_keywords = set(generated_answer.lower().split())
            keyword_coverage = len(keywords.intersection(generated_keywords)) / len(keywords) if keywords else 0
            length_ratio = len(generated_answer) / len(expected_answer) if expected_answer else 0
            metrics["completeness_score"] = (keyword_coverage + min(length_ratio, 1.0)) / 2

        # Accuracy score (placeholder - would need actual fact-checking)
        metrics["accuracy_score"] = metrics["relevance_score"] * 0.8  # Simplified

        # Overall score
        metrics["overall_score"] = (
            metrics["relevance_score"] * 0.4 +
            metrics["completeness_score"] * 0.3 +
            metrics["accuracy_score"] * 0.3
        )

        return metrics

    def batch_evaluate_test_cases(
        self,
        test_cases: List[Dict[str, Any]],
        pipeline_orchestrator: PipelineOrchestrator,
        max_workers: int = 3
    ) -> List[Dict[str, Any]]:
        """Evaluate multiple test cases in parallel"""
        results = []

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all evaluation tasks
            future_to_test_case = {
                executor.submit(
                    self.evaluate_test_case,
                    test_case,
                    pipeline_orchestrator
                ): test_case for test_case in test_cases
            }

            # Collect results as they complete
            for future in as_completed(future_to_test_case):
                test_case = future_to_test_case[future]
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    logger.error(f"Error evaluating test case {test_case.get('id')}: {e}")
                    results.append({
                        "test_case_id": test_case.get("id", "unknown"),
                        "error": str(e),
                        "success": False
                    })

        return results


class HealthMonitor:
    """Monitor health of all integrated API providers"""

    def __init__(self, api_manager: APIManager):
        self.api_manager = api_manager
        self.health_history = []

    def check_all_providers(self) -> Dict[str, Dict[str, Any]]:
        """Check health of all registered providers"""
        health_results = {}

        for provider_name in self.api_manager.get_available_providers():
            try:
                provider = self.api_manager.get_provider(provider_name)
                response = provider.health_check()

                health_results[provider_name] = {
                    "healthy": response.success,
                    "response_time": response.response_time,
                    "status_code": response.status_code,
                    "error_message": response.error_message,
                    "last_check": time.time()
                }

            except Exception as e:
                health_results[provider_name] = {
                    "healthy": False,
                    "error_message": str(e),
                    "last_check": time.time()
                }

        # Store in history
        self.health_history.append({
            "timestamp": time.time(),
            "results": health_results
        })

        # Keep only last 100 checks
        if len(self.health_history) > 100:
            self.health_history = self.health_history[-100:]

        return health_results

    def get_health_summary(self) -> Dict[str, Any]:
        """Get summary of provider health"""
        if not self.health_history:
            return {"summary": "No health checks performed"}

        latest_results = self.health_history[-1]["results"]
        healthy_count = sum(1 for r in latest_results.values() if r["healthy"])
        total_count = len(latest_results)

        return {
            "healthy_providers": healthy_count,
            "total_providers": total_count,
            "overall_health": healthy_count / total_count if total_count > 0 else 0,
            "last_check": self.health_history[-1]["timestamp"],
            "provider_status": latest_results
        }