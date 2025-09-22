"""
UI Components Module for RAG Lens

This module contains all Streamlit UI components organized by functionality.
Each component is designed to be modular, reusable, and follow consistent patterns.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime
import json

from ..config.settings import config
from ..utils.logger import get_logger
from ..utils.errors import handle_streamlit_error

logger = get_logger(__name__)


class ComponentRenderer:
    """Base class for UI components with common functionality"""

    @staticmethod
    def render_section_header(title: str, subtitle: str = None, icon: str = "📋"):
        """Render standardized section header"""
        st.markdown(f"### {icon} {title}")
        if subtitle:
            st.caption(subtitle)
        st.divider()

    @staticmethod
    def render_status_indicator(status: str, message: str = None):
        """Render status indicator with appropriate color"""
        status_colors = {
            "success": "🟢",
            "warning": "🟡",
            "error": "🔴",
            "info": "🔵",
            "processing": "⚪"
        }

        icon = status_colors.get(status.lower(), "⚪")
        text = message or status.title()

        st.markdown(f"{icon} {text}")

    @staticmethod
    def render_metric_card(title: str, value: Any, delta: float = None, help_text: str = None):
        """Render a metric card with optional delta"""
        col1, col2 = st.columns([3, 1])

        with col1:
            if help_text:
                st.metric(title, str(value), delta=delta, help=help_text)
            else:
                st.metric(title, str(value), delta=delta)

        with col2:
            if isinstance(value, (int, float)):
                if value > 0.8:
                    st.success("Excellent")
                elif value > 0.6:
                    st.warning("Good")
                else:
                    st.error("Needs Improvement")


class PipelineComponents(ComponentRenderer):
    """Pipeline-related UI components"""

    @staticmethod
    def render_pipeline_control_panel(
        current_step: int,
        total_steps: int,
        pipeline_status: str,
        on_step_change: Callable[[int], None],
        on_execute: Callable[[], None],
        on_reset: Callable[[], None]
    ):
        """Render pipeline control interface"""
        PipelineComponents.render_section_header(
            "Step-by-Step Pipeline Control",
            "Execute each step independently with detailed monitoring"
        )

        # Pipeline progress
        progress_data = []
        step_names = [
            "Query Generation", "Query Encoding", "Candidate Generation",
            "Candidate Filtering", "Pairwise Encoding", "Re-ranking", "Final Answer"
        ]

        for i in range(total_steps + 1):
            status = "completed" if i < current_step else "active" if i == current_step else "pending"
            progress_data.append({
                "step": i,
                "name": step_names[i] if i < len(step_names) else f"Step {i}",
                "status": status
            })

        # Progress visualization
        fig = go.Figure()

        for status, color in [("completed", "lightgreen"), ("active", "lightblue"), ("pending", "lightgray")]:
            status_data = [p for p in progress_data if p["status"] == status]
            if status_data:
                fig.add_trace(go.Bar(
                    x=[p["step"] for p in status_data],
                    y=[1] * len(status_data),
                    name=status.title(),
                    marker_color=color,
                    text=[p["name"] for p in status_data],
                    textposition="auto"
                ))

        fig.update_layout(
            title="Pipeline Progress",
            xaxis_title="Pipeline Steps",
            yaxis_visible=False,
            barmode="stack",
            height=200
        )

        st.plotly_chart(fig, use_container_width=True)

        # Step selector
        selected_step = st.selectbox(
            "Select Pipeline Step to Execute:",
            options=list(range(total_steps + 1)),
            format_func=lambda x: f"Step {x}: {step_names[x]}" if x < len(step_names) else f"Step {x}",
            index=current_step,
            help="Choose which pipeline step to execute"
        )

        # Action buttons
        col1, col2, col3 = st.columns(3)

        with col1:
            if st.button("Execute Selected Step", type="primary", use_container_width=True):
                on_execute()

        with col2:
            if st.button("Reset Pipeline", type="secondary", use_container_width=True):
                on_reset()

        with col3:
            PipelineComponents.render_status_indicator(pipeline_status)

        return selected_step

    @staticmethod
    def render_pipeline_results(results: Dict[str, Any]):
        """Render pipeline execution results"""
        if not results:
            st.info("No pipeline results available. Execute a pipeline step first.")
            return

        PipelineComponents.render_section_header("Pipeline Results", "Detailed execution metrics and outputs")

        # Execution time
        if "execution_time" in results:
            ComponentRenderer.render_metric_card(
                "Execution Time",
                f"{results['execution_time']:.2f}s",
                help_text="Time taken to execute the pipeline step"
            )

        # Metrics
        if "metrics" in results and results["metrics"]:
            st.subheader("Performance Metrics")

            metrics_data = []
            for metric_name, value in results["metrics"].items():
                metrics_data.append({
                    "Metric": metric_name.replace("_", " ").title(),
                    "Value": value
                })

            metrics_df = pd.DataFrame(metrics_data)
            st.dataframe(metrics_df, use_container_width=True)

        # Outputs
        if "outputs" in results and results["outputs"]:
            st.subheader("Generated Outputs")

            for output_type, output_data in results["outputs"].items():
                with st.expander(f"{output_type.replace('_', ' ').title()}"):
                    if isinstance(output_data, (list, dict)):
                        st.json(output_data)
                    else:
                        st.text_area("Content", value=str(output_data), height=200)


class TestCaseComponents(ComponentRenderer):
    """Test case management UI components"""

    @staticmethod
    def render_test_case_selector(
        available_test_cases: List[Dict[str, Any]],
        selected_test_case: Optional[str] = None,
        on_selection_change: Callable[[str], None] = None
    ):
        """Render test case selection interface"""
        TestCaseComponents.render_section_header(
            "Test Case Selection",
            "Choose a test case to evaluate"
        )

        if not available_test_cases:
            st.warning("No test cases available. Please check your test case source.")
            return None

        # Test case display
        test_case_options = {tc["id"]: tc["name"] for tc in available_test_cases}

        selected = st.selectbox(
            "Select Test Case:",
            options=list(test_case_options.keys()),
            format_func=lambda x: test_case_options[x],
            index=list(test_case_options.keys()).index(selected_test_case) if selected_test_case else 0,
            help="Choose a test case to run through the pipeline"
        )

        # Display selected test case details
        if selected:
            test_case = next((tc for tc in available_test_cases if tc["id"] == selected), None)
            if test_case:
                with st.expander("Test Case Details", expanded=True):
                    col1, col2 = st.columns(2)

                    with col1:
                        st.markdown(f"**ID:** `{test_case['id']}`")
                        st.markdown(f"**Name:** {test_case['name']}")
                        st.markdown(f"**Chunks:** {len(test_case.get('chunks', []))}")

                    with col2:
                        if test_case.get('user_rating'):
                            st.markdown(f"**Rating:** {'⭐' * test_case['user_rating']}")
                        if test_case.get('publish_time'):
                            st.markdown(f"**Published:** {test_case['publish_time'][:10]}")

                    st.markdown("**Description:**")
                    st.markdown(test_case.get('description', 'No description available'))

                    if test_case.get('query'):
                        st.markdown("**Query:**")
                        st.info(test_case['query'])

                    if test_case.get('expected_answer'):
                        st.markdown("**Expected Answer:**")
                        st.success(test_case['expected_answer'][:500] + '...' if len(test_case['expected_answer']) > 500 else test_case['expected_answer'])

        if on_selection_change and selected != selected_test_case:
            on_selection_change(selected)

        return selected

    @staticmethod
    def render_test_case_manager(
        test_cases: List[Dict[str, Any]],
        on_add: Callable[[], None] = None,
        on_edit: Callable[[str], None] = None,
        on_delete: Callable[[str], None] = None,
        on_export: Callable[[], None] = None
    ):
        """Render test case management interface"""
        TestCaseComponents.render_section_header(
            "Test Case Manager",
            "Manage your test case collection"
        )

        # Statistics
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            ComponentRenderer.render_metric_card("Total Cases", len(test_cases))

        with col2:
            avg_rating = sum(tc.get('user_rating', 0) for tc in test_cases) / len(test_cases) if test_cases else 0
            ComponentRenderer.render_metric_card("Avg Rating", f"{avg_rating:.1f}/5")

        with col3:
            total_chunks = sum(len(tc.get('chunks', [])) for tc in test_cases)
            ComponentRenderer.render_metric_card("Total Chunks", total_chunks)

        with col4:
            domains = len(set(tc.get('domain', 'unknown') for tc in test_cases))
            ComponentRenderer.render_metric_card("Domains", domains)

        # Action buttons
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            if st.button("Add Test Case", type="primary", use_container_width=True) and on_add:
                on_add()

        with col2:
            if st.button("Export Cases", use_container_width=True) and on_export:
                on_export()

        # Test case list
        if test_cases:
            st.subheader("Test Case Collection")

            # Convert to DataFrame for display
            display_data = []
            for tc in test_cases:
                display_data.append({
                    "ID": tc["id"],
                    "Name": tc["name"],
                    "Domain": tc.get("domain", "unknown"),
                    "Chunks": len(tc.get("chunks", [])),
                    "Rating": tc.get("user_rating", 0),
                    "Created": tc.get("publish_time", "")[:10] if tc.get("publish_time") else "Unknown"
                })

            df = pd.DataFrame(display_data)

            # Search and filter
            search_term = st.text_input("Search test cases...", placeholder="Search by name, ID, or domain...")

            if search_term:
                mask = (
                    df["ID"].str.contains(search_term, case=False) |
                    df["Name"].str.contains(search_term, case=False) |
                    df["Domain"].str.contains(search_term, case=False)
                )
                df = df[mask]

            # Display with actions
            edited_df = st.data_editor(
                df,
                column_config={
                    "Rating": st.column_config.ProgressColumn(
                        "Rating",
                        help="User rating (1-5 stars)",
                        format="⭐ %.0f/5",
                        min_value=0,
                        max_value=5
                    )
                },
                use_container_width=True,
                hide_index=True
            )

            # Row-level actions
            if len(df) > 0:
                selected_rows = st.session_state.get("selected_rows", [])
                if selected_rows:
                    selected_ids = df.iloc[selected_rows]["ID"].tolist()

                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button("Edit Selected", type="primary", use_container_width=True) and on_edit:
                            on_edit(selected_ids[0] if len(selected_ids) == 1 else selected_ids)

                    with col2:
                        if st.button("Delete Selected", type="secondary", use_container_width=True) and on_delete:
                            on_delete(selected_ids if len(selected_ids) == 1 else selected_ids)


class EvaluationComponents(ComponentRenderer):
    """Evaluation and results UI components"""

    @staticmethod
    def render_evaluation_dashboard(results: Dict[str, Any]):
        """Render comprehensive evaluation dashboard"""
        EvaluationComponents.render_section_header(
            "Evaluation Dashboard",
            "Pipeline performance metrics and analysis"
        )

        if not results:
            st.info("No evaluation results available. Run a pipeline evaluation first.")
            return

        # Overall metrics
        if "overall_metrics" in results:
            metrics = results["overall_metrics"]

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                ComponentRenderer.render_metric_card(
                    "Overall Score",
                    f"{metrics.get('overall_score', 0):.3f}",
                    help_text="Combined evaluation score"
                )

            with col2:
                ComponentRenderer.render_metric_card(
                    "Relevance",
                    f"{metrics.get('relevance_score', 0):.3f}",
                    help_text="Answer relevance to query"
                )

            with col3:
                ComponentRenderer.render_metric_card(
                    "Accuracy",
                    f"{metrics.get('accuracy_score', 0):.3f}",
                    help_text="Factual accuracy score"
                )

            with col4:
                ComponentRenderer.render_metric_card(
                    "Completeness",
                    f"{metrics.get('completeness_score', 0):.3f}",
                    help_text="Answer completeness score"
                )

        # Detailed metrics
        if "detailed_metrics" in results:
            st.subheader("Detailed Analysis")

            detailed_data = []
            for metric_name, value in results["detailed_metrics"].items():
                detailed_data.append({
                    "Metric": metric_name.replace("_", " ").title(),
                    "Score": value,
                    "Status": "Good" if value > 0.7 else "Needs Improvement" if value > 0.5 else "Poor"
                })

            detailed_df = pd.DataFrame(detailed_data)

            # Bar chart
            fig = px.bar(
                detailed_df,
                x="Metric",
                y="Score",
                color="Status",
                title="Detailed Metrics Breakdown",
                color_discrete_map={
                    "Good": "green",
                    "Needs Improvement": "orange",
                    "Poor": "red"
                }
            )
            fig.update_layout(yaxis_range=[0, 1])
            st.plotly_chart(fig, use_container_width=True)

            # Data table
            st.dataframe(detailed_df, use_container_width=True)

        # Answer comparison
        if "answer_comparison" in results:
            st.subheader("Answer Comparison")

            comparison = results["answer_comparison"]

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**Generated Answer:**")
                st.info(comparison.get("generated_answer", "No answer generated"))

            with col2:
                st.markdown("**Expected Answer:**")
                st.success(comparison.get("expected_answer", "No expected answer"))

            # Similarity score
            if "similarity_score" in comparison:
                st.markdown(f"**Similarity Score:** `{comparison['similarity_score']:.3f}`")

    @staticmethod
    def render_performance_charts(metrics_history: List[Dict[str, Any]]):
        """Render performance tracking charts"""
        if not metrics_history:
            st.info("No performance history available.")
            return

        EvaluationComponents.render_section_header(
            "Performance Tracking",
            "Historical performance metrics over time"
        )

        # Convert to DataFrame
        df = pd.DataFrame(metrics_history)

        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.sort_values('timestamp')

        # Metric selection
        available_metrics = [col for col in df.columns if col not in ['timestamp', 'test_case_id']]
        selected_metrics = st.multiselect(
            "Select metrics to display:",
            options=available_metrics,
            default=available_metrics[:3] if len(available_metrics) >= 3 else available_metrics
        )

        if selected_metrics:
            # Line chart
            fig = go.Figure()

            for metric in selected_metrics:
                if metric in df.columns:
                    fig.add_trace(go.Scatter(
                        x=df['timestamp'],
                        y=df[metric],
                        mode='lines+markers',
                        name=metric.replace('_', ' ').title(),
                        line=dict(width=2)
                    ))

            fig.update_layout(
                title="Performance Trends",
                xaxis_title="Time",
                yaxis_title="Score",
                yaxis_range=[0, 1],
                height=400
            )

            st.plotly_chart(fig, use_container_width=True)

        # Statistics table
        if available_metrics:
            st.subheader("Metric Statistics")

            stats_data = []
            for metric in available_metrics:
                if metric in df.columns:
                    values = df[metric].dropna()
                    if len(values) > 0:
                        stats_data.append({
                            "Metric": metric.replace('_', ' ').title(),
                            "Mean": values.mean(),
                            "Std Dev": values.std(),
                            "Min": values.min(),
                            "Max": values.max(),
                            "Latest": values.iloc[-1] if len(values) > 0 else None
                        })

            stats_df = pd.DataFrame(stats_data)
            st.dataframe(stats_df, use_container_width=True)


class SettingsComponents(ComponentRenderer):
    """Configuration and settings UI components"""

    @staticmethod
    def render_configuration_editor(config_data: Dict[str, Any], on_save: Callable[[Dict[str, Any]], None]):
        """Render configuration editor interface"""
        SettingsComponents.render_section_header(
            "Configuration Editor",
            "Manage application settings and API configurations"
        )

        # Configuration sections
        sections = {
            "general": "General Settings",
            "api": "API Configuration",
            "monitoring": "Monitoring",
            "testing": "Testing Settings"
        }

        selected_section = st.selectbox(
            "Select Configuration Section:",
            options=list(sections.keys()),
            format_func=lambda x: sections[x]
        )

        # Render configuration form based on section
        if selected_section == "general":
            SettingsComponents._render_general_settings(config_data)
        elif selected_section == "api":
            SettingsComponents._render_api_settings(config_data)
        elif selected_section == "monitoring":
            SettingsComponents._render_monitoring_settings(config_data)
        elif selected_section == "testing":
            SettingsComponents._render_testing_settings(config_data)

        # Save button
        if st.button("Save Configuration", type="primary", use_container_width=True):
            on_save(config_data)
            st.success("Configuration saved successfully!")

    @staticmethod
    def _render_general_settings(config_data: Dict[str, Any]):
        """Render general settings section"""
        st.subheader("General Settings")

        # Application settings
        config_data.setdefault("app", {})
        app_config = config_data["app"]

        app_config["name"] = st.text_input(
            "Application Name",
            value=app_config.get("name", "RAG Lens")
        )

        app_config["version"] = st.text_input(
            "Version",
            value=app_config.get("version", "1.0.0")
        )

        app_config["debug"] = st.checkbox(
            "Debug Mode",
            value=app_config.get("debug", False)
        )

    @staticmethod
    def _render_api_settings(config_data: Dict[str, Any]):
        """Render API configuration section"""
        st.subheader("API Configuration")

        config_data.setdefault("api", {})
        api_config = config_data["api"]

        # API endpoints
        st.markdown("**API Endpoints**")

        api_config["base_url"] = st.text_input(
            "Base URL",
            value=api_config.get("base_url", "")
        )

        api_config["timeout"] = st.number_input(
            "Timeout (seconds)",
            min_value=1,
            max_value=300,
            value=api_config.get("timeout", 30)
        )

        # Reranking Provider Configuration
        st.markdown("**Reranking Provider**")
        
        reranking_provider = st.selectbox(
            "Reranking Provider",
            options=["cross_encoder", "ollama_rerank"],
            format_func=lambda x: "Cross Encoder" if x == "cross_encoder" else "Ollama Rerank",
            index=["cross_encoder", "ollama_rerank"].index(api_config.get("reranking_provider", "cross_encoder"))
        )
        
        api_config["reranking_provider"] = reranking_provider
        
        if reranking_provider == "ollama_rerank":
            st.markdown("**Ollama Configuration**")
            
            api_config["ollama_base_url"] = st.text_input(
                "Ollama Base URL",
                value=api_config.get("ollama_base_url", "http://localhost:11434"),
                help="Base URL for Ollama API server"
            )
            
            api_config["ollama_model"] = st.text_input(
                "Ollama Model",
                value=api_config.get("ollama_model", "llama2"),
                help="Model name to use for embeddings and reranking"
            )
            
            api_config["ollama_api_key"] = st.text_input(
                "Ollama API Key (Optional)",
                value=api_config.get("ollama_api_key", ""),
                type="password",
                help="API key if required by your Ollama setup"
            )

        # Authentication
        st.markdown("**Authentication**")

        auth_type = st.selectbox(
            "Authentication Type",
            options=["none", "api_key", "oauth", "jwt"],
            format_func=lambda x: x.replace("_", " ").title(),
            index=["none", "api_key", "oauth", "jwt"].index(api_config.get("auth_type", "none"))
        )

        api_config["auth_type"] = auth_type

        if auth_type == "api_key":
            api_config["api_key"] = st.text_input(
                "API Key",
                value=api_config.get("api_key", ""),
                type="password"
            )
        elif auth_type == "oauth":
            api_config["client_id"] = st.text_input(
                "Client ID",
                value=api_config.get("client_id", "")
            )
            api_config["client_secret"] = st.text_input(
                "Client Secret",
                value=api_config.get("client_secret", ""),
                type="password"
            )

    @staticmethod
    def _render_monitoring_settings(config_data: Dict[str, Any]):
        """Render monitoring settings section"""
        st.subheader("Monitoring Settings")

        config_data.setdefault("monitoring", {})
        monitoring_config = config_data["monitoring"]

        monitoring_config["enabled"] = st.checkbox(
            "Enable Monitoring",
            value=monitoring_config.get("enabled", True)
        )

        if monitoring_config["enabled"]:
            monitoring_config["log_level"] = st.selectbox(
                "Log Level",
                options=["DEBUG", "INFO", "WARNING", "ERROR"],
                index=["DEBUG", "INFO", "WARNING", "ERROR"].index(monitoring_config.get("log_level", "INFO"))
            )

            monitoring_config["metrics_interval"] = st.number_input(
                "Metrics Collection Interval (seconds)",
                min_value=1,
                max_value=3600,
                value=monitoring_config.get("metrics_interval", 60)
            )

    @staticmethod
    def _render_testing_settings(config_data: Dict[str, Any]):
        """Render testing settings section"""
        st.subheader("Testing Settings")

        config_data.setdefault("testing", {})
        testing_config = config_data["testing"]

        testing_config["test_source"] = st.selectbox(
            "Test Case Source",
            options=["static", "bigquery"],
            format_func=lambda x: "Static JSON" if x == "static" else "BigQuery",
            index=["static", "bigquery"].index(testing_config.get("test_source", "static"))
        )

        if testing_config["test_source"] == "static":
            testing_config["test_file_path"] = st.text_input(
                "Test File Path",
                value=testing_config.get("test_file_path", "real_test_cases_collection.json")
            )
        else:
            testing_config["bigquery_project"] = st.text_input(
                "BigQuery Project ID",
                value=testing_config.get("bigquery_project", "")
            )
            testing_config["bigquery_dataset"] = st.text_input(
                "BigQuery Dataset",
                value=testing_config.get("bigquery_dataset", "")
            )
            testing_config["bigquery_table"] = st.text_input(
                "BigQuery Table",
                value=testing_config.get("bigquery_table", "")
            )


class ErrorComponents:
    """Error handling and notification components"""

    @staticmethod
    @handle_streamlit_error
    def safe_component_render(component_func: Callable, *args, **kwargs):
        """Safely render a component with error handling"""
        try:
            return component_func(*args, **kwargs)
        except Exception as e:
            logger.error(f"Error rendering component {component_func.__name__}: {e}")
            st.error(f"An error occurred while rendering this component: {str(e)}")
            return None

    @staticmethod
    def render_error_boundary(error_message: str, error_details: str = None):
        """Render error boundary UI"""
        st.error("❌ An Error Occurred")
        st.markdown(f"**Error:** {error_message}")

        if error_details and config.is_development():
            with st.expander("Error Details (Development Mode)"):
                st.code(error_details, language="python")

        st.info("Please try refreshing the page or contact support if the issue persists.")

    @staticmethod
    def render_loading_spinner(message: str = "Loading..."):
        """Render standardized loading spinner"""
        with st.spinner(message):
            return st.empty()


# Component registry for easy access
UI_COMPONENTS = {
    "pipeline": PipelineComponents,
    "test_case": TestCaseComponents,
    "evaluation": EvaluationComponents,
    "settings": SettingsComponents,
    "error": ErrorComponents
}