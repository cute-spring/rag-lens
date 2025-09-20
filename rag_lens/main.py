"""
RAG Lens - Main Application Entry Point

This is the main entry point for the RAG Lens application.
It provides a modular architecture with clean separation of concerns.
"""

import streamlit as st
import sys
import os
from pathlib import Path
from typing import Dict, List, Any, Optional
import json
import time

# Add the rag_lens directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

from config.settings import config
from utils.logger import get_logger, setup_logging
from utils.errors import handle_streamlit_error
from utils.security import security_manager
from core.test_case_manager import TestCaseManager
from core.pipeline_simulator import PipelineSimulator
from api.providers import api_manager
from api.integrations import PipelineOrchestrator, TestCaseIntegration, HealthMonitor
from ui.components import UI_COMPONENTS

# Initialize logging
setup_logging()
logger = get_logger(__name__)


class RAGLensApp:
    """Main application class"""

    def __init__(self):
        self.test_case_manager = TestCaseManager()
        self.pipeline_simulator = PipelineSimulator()
        self.pipeline_orchestrator = PipelineOrchestrator(api_manager)
        self.test_case_integration = TestCaseIntegration(api_manager)
        self.health_monitor = HealthMonitor(api_manager)
        self.app_state = self._initialize_app_state()

    def _initialize_app_state(self) -> Dict[str, Any]:
        """Initialize application state"""
        return {
            "current_test_case": None,
            "pipeline_step": 0,
            "pipeline_status": "ready",
            "execution_history": [],
            "evaluation_results": [],
            "test_case_source": config.testing.test_source,
            "available_test_cases": [],
            "health_status": {},
            "last_update": time.time()
        }

    def run(self):
        """Run the main application"""
        self._setup_page_config()
        self._initialize_session_state()
        self._render_navigation()
        self._render_main_content()

    def _setup_page_config(self):
        """Setup Streamlit page configuration"""
        st.set_page_config(
            page_title=config.app.name,
            page_icon="🔍",
            layout="wide",
            initial_sidebar_state="expanded"
        )

        # Add custom CSS
        self._inject_custom_css()

    def _inject_custom_css(self):
        """Inject custom CSS for better styling"""
        st.markdown("""
        <style>
        .metric-card {
            background-color: #f8f9fa;
            padding: 1rem;
            border-radius: 0.5rem;
            border: 1px solid #e9ecef;
        }
        .stStatus {
            padding: 0.5rem;
            border-radius: 0.25rem;
        }
        .success { background-color: #d4edda; color: #155724; }
        .warning { background-color: #fff3cd; color: #856404; }
        .error { background-color: #f8d7da; color: #721c24; }
        .info { background-color: #d1ecf1; color: #0c5460; }
        </style>
        """, unsafe_allow_html=True)

    def _initialize_session_state(self):
        """Initialize Streamlit session state"""
        if 'app_initialized' not in st.session_state:
            st.session_state.app_initialized = True
            st.session_state.update(self.app_state)
            self._load_initial_data()

    def _load_initial_data(self):
        """Load initial application data"""
        try:
            # Load test cases
            self._refresh_test_cases()

            # Check API health
            self._check_health_status()

            logger.info("Application initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing application: {e}")
            st.error(f"Error initializing application: {e}")

    def _render_navigation(self):
        """Render navigation sidebar"""
        with st.sidebar:
            st.title("🔍 RAG Lens")
            st.markdown(f"**v{config.app.version}**")
            st.divider()

            # Navigation
            st.subheader("Navigation")
            page_options = [
                "Dashboard",
                "Pipeline Control",
                "Test Cases",
                "Evaluation",
                "Configuration",
                "Health Monitor"
            ]

            selected_page = st.selectbox(
                "Select Page:",
                options=page_options,
                key="selected_page"
            )

            st.session_state.current_page = selected_page

            # Quick stats
            st.divider()
            st.subheader("Quick Stats")

            test_cases_count = len(st.session_state.available_test_cases)
            executions_count = len(st.session_state.execution_history)

            col1, col2 = st.columns(2)
            with col1:
                st.metric("Test Cases", test_cases_count)
            with col2:
                st.metric("Executions", executions_count)

            # Health status
            if st.session_state.health_status:
                healthy_count = sum(1 for status in st.session_state.health_status.values() if status.get("healthy", False))
                total_count = len(st.session_state.health_status)
                if total_count > 0:
                    health_percentage = healthy_count / total_count
                    status_color = "success" if health_percentage > 0.8 else "warning" if health_percentage > 0.5 else "error"
                    st.markdown(f"<div class='stStatus {status_color}'>API Health: {healthy_count}/{total_count}</div>", unsafe_allow_html=True)

            # Actions
            st.divider()
            if st.button("Refresh Data", type="primary", use_container_width=True):
                self._refresh_all_data()

    def _render_main_content(self):
        """Render main content based on selected page"""
        page = st.session_state.get("current_page", "Dashboard")

        try:
            if page == "Dashboard":
                self._render_dashboard()
            elif page == "Pipeline Control":
                self._render_pipeline_control()
            elif page == "Test Cases":
                self._render_test_cases()
            elif page == "Evaluation":
                self._render_evaluation()
            elif page == "Configuration":
                self._render_configuration()
            elif page == "Health Monitor":
                self._render_health_monitor()
            else:
                self._render_not_found()

        except Exception as e:
            logger.error(f"Error rendering page {page}: {e}")
            UI_COMPONENTS["error"].render_error_boundary(
                f"Error loading {page} page",
                str(e) if config.is_development() else None
            )

    def _render_dashboard(self):
        """Render dashboard page"""
        st.title("📊 Dashboard")
        st.markdown("Welcome to RAG Lens - Your comprehensive RAG pipeline testing platform")

        # Key metrics
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            UI_COMPONENTS["error"].safe_component_render(
                UI_COMPONENTS["pipeline"].render_metric_card,
                "Total Test Cases",
                len(st.session_state.available_test_cases)
            )

        with col2:
            successful_executions = len([
                exec for exec in st.session_state.execution_history
                if exec.get("success", False)
            ])
            total_executions = len(st.session_state.execution_history)
            success_rate = successful_executions / total_executions if total_executions > 0 else 0
            UI_COMPONENTS["error"].safe_component_render(
                UI_COMPONENTS["pipeline"].render_metric_card,
                "Success Rate",
                f"{success_rate:.1%}",
                delta=success_rate - 0.7 if success_rate > 0.7 else 0
            )

        with col3:
            avg_execution_time = 0
            if st.session_state.execution_history:
                times = [exec.get("execution_time", 0) for exec in st.session_state.execution_history]
                avg_execution_time = sum(times) / len(times) if times else 0
            UI_COMPONENTS["error"].safe_component_render(
                UI_COMPONENTS["pipeline"].render_metric_card,
                "Avg Execution Time",
                f"{avg_execution_time:.2f}s"
            )

        with col4:
            health_percentage = 0
            if st.session_state.health_status:
                healthy_count = sum(1 for status in st.session_state.health_status.values() if status.get("healthy", False))
                health_percentage = healthy_count / len(st.session_state.health_status)
            UI_COMPONENTS["error"].safe_component_render(
                UI_COMPONENTS["pipeline"].render_metric_card,
                "API Health",
                f"{health_percentage:.1%}"
            )

        # Recent activity
        st.subheader("Recent Activity")
        if st.session_state.execution_history:
            recent_executions = st.session_state.execution_history[-5:]
            for execution in recent_executions:
                with st.expander(f"Execution {execution.get('test_case_id', 'unknown')}"):
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.markdown(f"**Test Case:** {execution.get('test_case_id', 'unknown')}")
                    with col2:
                        status = "✅ Success" if execution.get("success", False) else "❌ Failed"
                        st.markdown(f"**Status:** {status}")
                    with col3:
                        st.markdown(f"**Time:** {execution.get('execution_time', 0):.2f}s")
        else:
            st.info("No executions yet. Start by running a pipeline or evaluating test cases.")

        # Quick actions
        st.subheader("Quick Actions")
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("Run Pipeline", type="primary", use_container_width=True):
                st.session_state.current_page = "Pipeline Control"
                st.rerun()
        with col2:
            if st.button("View Test Cases", use_container_width=True):
                st.session_state.current_page = "Test Cases"
                st.rerun()
        with col3:
            if st.button("Check Health", use_container_width=True):
                st.session_state.current_page = "Health Monitor"
                st.rerun()

    def _render_pipeline_control(self):
        """Render pipeline control page"""
        st.title("⚙️ Pipeline Control")
        st.markdown("Execute and monitor RAG pipeline steps with detailed control")

        # Test case selection
        selected_test_case = UI_COMPONENTS["test_case"].render_test_case_selector(
            st.session_state.available_test_cases,
            st.session_state.current_test_case,
            self._on_test_case_selected
        )

        # Pipeline control
        def on_execute_step():
            self._execute_current_step()

        def on_reset_pipeline():
            self._reset_pipeline()

        selected_step = UI_COMPONENTS["pipeline"].render_pipeline_control_panel(
            st.session_state.pipeline_step,
            7,  # Total steps
            st.session_state.pipeline_status,
            self._on_step_changed,
            on_execute_step,
            on_reset_pipeline
        )

        # Results display
        if st.session_state.get("current_execution_result"):
            UI_COMPONENTS["pipeline"].render_pipeline_results(
                st.session_state.current_execution_result
            )

    def _render_test_cases(self):
        """Render test cases page"""
        st.title("📋 Test Cases")
        st.markdown("Manage and explore your test case collection")

        # Test case source selection
        col1, col2 = st.columns([3, 1])
        with col1:
            source_options = ["static", "bigquery"]
            selected_source = st.selectbox(
                "Test Case Source:",
                options=source_options,
                format_func=lambda x: "Static JSON" if x == "static" else "BigQuery",
                index=source_options.index(st.session_state.test_case_source)
            )
        with col2:
            if st.button("Load Test Cases", type="primary", use_container_width=True):
                self._switch_test_source(selected_source)

        # Test case manager
        def on_add_test_case():
            st.session_state.show_add_test_case = True

        def on_edit_test_case(test_case_id):
            self._edit_test_case(test_case_id)

        def on_delete_test_case(test_case_id):
            self._delete_test_case(test_case_id)

        def on_export_test_cases():
            self._export_test_cases()

        UI_COMPONENTS["test_case"].render_test_case_manager(
            st.session_state.available_test_cases,
            on_add_test_case,
            on_edit_test_case,
            on_delete_test_case,
            on_export_test_cases
        )

        # Add test case form (conditionally shown)
        if st.session_state.get("show_add_test_case"):
            self._render_add_test_case_form()

    def _render_evaluation(self):
        """Render evaluation page"""
        st.title("📈 Evaluation")
        st.markdown("Evaluate pipeline performance and analyze results")

        # Batch evaluation controls
        col1, col2 = st.columns(2)
        with col1:
            selected_test_cases = st.multiselect(
                "Select Test Cases to Evaluate:",
                options=[tc["id"] for tc in st.session_state.available_test_cases],
                format_func=lambda x: next((tc["name"] for tc in st.session_state.available_test_cases if tc["id"] == x), x)
            )
        with col2:
            if st.button("Run Evaluation", type="primary", use_container_width=True) and selected_test_cases:
                self._run_batch_evaluation(selected_test_cases)

        # Results display
        if st.session_state.evaluation_results:
            # Overall metrics
            overall_metrics = self._calculate_overall_metrics()
            UI_COMPONENTS["evaluation"].render_evaluation_dashboard(overall_metrics)

            # Performance charts
            metrics_history = self._get_metrics_history()
            UI_COMPONENTS["evaluation"].render_performance_charts(metrics_history)

        # Individual results
        if st.session_state.evaluation_results:
            st.subheader("Individual Test Case Results")
            results_df = self._create_results_dataframe()
            st.dataframe(results_df, use_container_width=True)

    def _render_configuration(self):
        """Render configuration page"""
        st.title("⚙️ Configuration")
        st.markdown("Manage application settings and API configurations")

        def on_save_config(new_config):
            self._save_configuration(new_config)

        current_config = {
            "app": {
                "name": config.app.name,
                "version": config.app.version,
                "debug": config.app.debug
            },
            "api": {
                "base_url": config.api.base_url,
                "timeout": config.api.timeout,
                "auth_type": config.api.auth_type
            },
            "monitoring": {
                "enabled": config.monitoring.enabled,
                "log_level": config.monitoring.log_level,
                "metrics_interval": config.monitoring.metrics_interval
            },
            "testing": {
                "test_source": config.testing.test_source,
                "test_file_path": config.testing.test_file_path
            }
        }

        UI_COMPONENTS["settings"].render_configuration_editor(
            current_config,
            on_save_config
        )

    def _render_health_monitor(self):
        """Render health monitor page"""
        st.title("🏥 Health Monitor")
        st.markdown("Monitor the health and performance of API providers")

        # Health check button
        if st.button("Check All Providers", type="primary", use_container_width=True):
            self._check_health_status()

        # Health status display
        if st.session_state.health_status:
            st.subheader("Provider Health Status")

            for provider_name, status in st.session_state.health_status.items():
                with st.expander(f"{provider_name}"):
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        health_icon = "🟢" if status.get("healthy") else "🔴"
                        st.markdown(f"**Status:** {health_icon} {'Healthy' if status.get('healthy') else 'Unhealthy'}")
                    with col2:
                        response_time = status.get("response_time", 0)
                        st.markdown(f"**Response Time:** {response_time:.2f}s")
                    with col3:
                        status_code = status.get("status_code", "N/A")
                        st.markdown(f"**Status Code:** {status_code}")

                    if status.get("error_message"):
                        st.error(f"Error: {status['error_message']}")

            # Health summary
            health_summary = self.health_monitor.get_health_summary()
            st.subheader("Health Summary")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Healthy Providers", health_summary.get("healthy_providers", 0))
            with col2:
                overall_health = health_summary.get("overall_health", 0)
                st.metric("Overall Health", f"{overall_health:.1%}")
        else:
            st.info("No health data available. Click 'Check All Providers' to get started.")

    def _render_not_found(self):
        """Render page not found"""
        st.error("Page not found")
        st.markdown("Please select a valid page from the navigation sidebar.")

    # Event handlers
    def _on_test_case_selected(self, test_case_id: str):
        """Handle test case selection"""
        st.session_state.current_test_case = test_case_id
        logger.info(f"Selected test case: {test_case_id}")

    def _on_step_changed(self, step: int):
        """Handle pipeline step change"""
        st.session_state.pipeline_step = step
        logger.info(f"Changed to pipeline step: {step}")

    def _execute_current_step(self):
        """Execute the current pipeline step"""
        if not st.session_state.current_test_case:
            st.error("Please select a test case first")
            return

        test_case = next(
            (tc for tc in st.session_state.available_test_cases if tc["id"] == st.session_state.current_test_case),
            None
        )

        if not test_case:
            st.error("Selected test case not found")
            return

        try:
            st.session_state.pipeline_status = "processing"

            # Execute pipeline step
            result = self.pipeline_orchestrator.execute_pipeline(
                query=test_case["query"],
                system_prompt=test_case.get("system_prompt", ""),
                test_case_id=test_case["id"],
                step_to_execute=st.session_state.pipeline_step
            )

            st.session_state.current_execution_result = result
            st.session_state.execution_history.append(result)

            if result["success"]:
                st.session_state.pipeline_status = "success"
                st.success("Pipeline step executed successfully")
            else:
                st.session_state.pipeline_status = "error"
                st.error(f"Pipeline step failed: {result.get('error', 'Unknown error')}")

        except Exception as e:
            logger.error(f"Error executing pipeline step: {e}")
            st.session_state.pipeline_status = "error"
            st.error(f"Error executing pipeline step: {e}")

    def _reset_pipeline(self):
        """Reset pipeline state"""
        st.session_state.pipeline_step = 0
        st.session_state.pipeline_status = "ready"
        st.session_state.current_execution_result = None
        logger.info("Pipeline reset")

    def _refresh_test_cases(self):
        """Refresh test cases from current source"""
        try:
            test_cases = self.test_case_manager.load_test_cases()
            st.session_state.available_test_cases = test_cases
            logger.info(f"Loaded {len(test_cases)} test cases")
        except Exception as e:
            logger.error(f"Error loading test cases: {e}")
            st.error(f"Error loading test cases: {e}")

    def _refresh_all_data(self):
        """Refresh all application data"""
        self._refresh_test_cases()
        self._check_health_status()
        st.success("Data refreshed successfully")

    def _switch_test_source(self, source: str):
        """Switch test case source"""
        try:
            config.testing.test_source = source
            st.session_state.test_case_source = source
            self._refresh_test_cases()
            st.success(f"Switched to {source} test source")
        except Exception as e:
            logger.error(f"Error switching test source: {e}")
            st.error(f"Error switching test source: {e}")

    def _check_health_status(self):
        """Check health status of all providers"""
        try:
            health_status = self.health_monitor.check_all_providers()
            st.session_state.health_status = health_status
            logger.info("Health status check completed")
        except Exception as e:
            logger.error(f"Error checking health status: {e}")
            st.error(f"Error checking health status: {e}")

    def _run_batch_evaluation(self, test_case_ids: List[str]):
        """Run batch evaluation on selected test cases"""
        test_cases = [
            tc for tc in st.session_state.available_test_cases
            if tc["id"] in test_case_ids
        ]

        if not test_cases:
            st.error("No valid test cases selected")
            return

        try:
            with st.spinner("Running batch evaluation..."):
                results = self.test_case_integration.batch_evaluate_test_cases(
                    test_cases,
                    self.pipeline_orchestrator
                )

            st.session_state.evaluation_results.extend(results)
            st.success(f"Evaluation completed for {len(results)} test cases")

        except Exception as e:
            logger.error(f"Error running batch evaluation: {e}")
            st.error(f"Error running batch evaluation: {e}")

    def _calculate_overall_metrics(self) -> Dict[str, Any]:
        """Calculate overall evaluation metrics"""
        if not st.session_state.evaluation_results:
            return {}

        total_evaluations = len(st.session_state.evaluation_results)
        successful_evaluations = len([
            r for r in st.session_state.evaluation_results
            if r.get("pipeline_success", False)
        ])

        avg_similarity = sum(
            r.get("similarity_score", 0) for r in st.session_state.evaluation_results
        ) / total_evaluations if total_evaluations > 0 else 0

        avg_overall_score = sum(
            r.get("evaluation_metrics", {}).get("overall_score", 0)
            for r in st.session_state.evaluation_results
        ) / total_evaluations if total_evaluations > 0 else 0

        return {
            "overall_metrics": {
                "total_evaluations": total_evaluations,
                "successful_evaluations": successful_evaluations,
                "success_rate": successful_evaluations / total_evaluations if total_evaluations > 0 else 0,
                "average_similarity": avg_similarity,
                "average_overall_score": avg_overall_score
            }
        }

    def _get_metrics_history(self) -> List[Dict[str, Any]]:
        """Get metrics history for charts"""
        return st.session_state.evaluation_results[-50:]  # Last 50 results

    def _create_results_dataframe(self):
        """Create DataFrame from evaluation results"""
        data = []
        for result in st.session_state.evaluation_results:
            data.append({
                "Test Case ID": result.get("test_case_id", "unknown"),
                "Success": result.get("pipeline_success", False),
                "Execution Time": result.get("execution_time", 0),
                "Similarity Score": result.get("similarity_score", 0),
                "Overall Score": result.get("evaluation_metrics", {}).get("overall_score", 0)
            })
        return data

    def _save_configuration(self, new_config: Dict[str, Any]):
        """Save configuration"""
        try:
            # Update configuration
            if "app" in new_config:
                config.app.name = new_config["app"].get("name", config.app.name)
                config.app.version = new_config["app"].get("version", config.app.version)
                config.app.debug = new_config["app"].get("debug", config.app.debug)

            if "monitoring" in new_config:
                config.monitoring.enabled = new_config["monitoring"].get("enabled", config.monitoring.enabled)
                config.monitoring.log_level = new_config["monitoring"].get("log_level", config.monitoring.log_level)
                config.monitoring.metrics_interval = new_config["monitoring"].get("metrics_interval", config.monitoring.metrics_interval)

            if "testing" in new_config:
                config.testing.test_source = new_config["testing"].get("test_source", config.testing.test_source)
                config.testing.test_file_path = new_config["testing"].get("test_file_path", config.testing.test_file_path)

            st.success("Configuration saved successfully")
            logger.info("Configuration updated")

        except Exception as e:
            logger.error(f"Error saving configuration: {e}")
            st.error(f"Error saving configuration: {e}")

    def _edit_test_case(self, test_case_id: str):
        """Edit a test case"""
        st.info(f"Edit functionality for test case {test_case_id} would be implemented here")
        # Implementation would include form for editing test case

    def _delete_test_case(self, test_case_id: str):
        """Delete a test case"""
        st.info(f"Delete functionality for test case {test_case_id} would be implemented here")
        # Implementation would include confirmation dialog and deletion

    def _export_test_cases(self):
        """Export test cases"""
        try:
            export_data = {
                "test_cases": st.session_state.available_test_cases,
                "exported_at": time.time(),
                "total_count": len(st.session_state.available_test_cases)
            }

            st.download_button(
                label="Download Test Cases",
                data=json.dumps(export_data, indent=2),
                file_name="test_cases_export.json",
                mime="application/json"
            )

        except Exception as e:
            logger.error(f"Error exporting test cases: {e}")
            st.error(f"Error exporting test cases: {e}")

    def _render_add_test_case_form(self):
        """Render add test case form"""
        st.subheader("Add New Test Case")
        with st.form("add_test_case"):
            col1, col2 = st.columns(2)
            with col1:
                test_id = st.text_input("Test Case ID")
                test_name = st.text_input("Test Case Name")
            with col2:
                domain = st.text_input("Domain")
                user_rating = st.number_input("User Rating", min_value=1, max_value=5, value=5)

            query = st.text_area("Query", height=100)
            system_prompt = st.text_area("System Prompt", height=100)
            expected_answer = st.text_area("Expected Answer", height=200)
            description = st.text_area("Description", height=100)

            col1, col2 = st.columns(2)
            with col1:
                submit_button = st.form_submit_button("Add Test Case", type="primary")
            with col2:
                cancel_button = st.form_submit_button("Cancel")

            if submit_button:
                try:
                    new_test_case = {
                        "id": test_id,
                        "name": test_name,
                        "domain": domain,
                        "query": query,
                        "system_prompt": system_prompt,
                        "expected_answer": expected_answer,
                        "description": description,
                        "user_rating": user_rating,
                        "publish_time": time.strftime("%Y-%m-%dT%H:%M:%S"),
                        "chunks": []
                    }

                    # Add to available test cases
                    st.session_state.available_test_cases.append(new_test_case)
                    st.session_state.show_add_test_case = False
                    st.success("Test case added successfully")

                except Exception as e:
                    st.error(f"Error adding test case: {e}")

            if cancel_button:
                st.session_state.show_add_test_case = False


def main():
    """Main entry point"""
    try:
        app = RAGLensApp()
        app.run()
    except Exception as e:
        logger.critical(f"Application failed to start: {e}")
        st.error(f"Application failed to start: {e}")
        if config.is_development():
            st.code(str(e), language="python")


if __name__ == "__main__":
    main()