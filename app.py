from __future__ import annotations

import json
from html import escape
from typing import Any, Dict

import pandas as pd
import streamlit as st

from core.config import PROMPT_VERSION, resolve_openai_api_key
from core.model_catalog import (
    candidate_config_ids,
    candidate_label,
    default_candidate_config_ids,
    get_preset,
    label_to_candidate_id,
)
from core.orchestrator import EvaluationOrchestrator
from core.schemas import ComparisonReport, GenerationInput
from ui.styles import inject_global_styles
from ui.visualizations import (
    comparison_dataframe,
    mlflow_score_chart,
    quality_cost_scatter,
    radar_chart,
    score_heatmap,
    trace_waterfall,
)

st.set_page_config(
    page_title="Agentic LLM Benchmarking Workbench",
    page_icon="AB",
    layout="wide",
)

inject_global_styles()


def _streamlit_secret_key() -> str:
    try:
        return str(st.secrets.get("OPENAI_API_KEY", "")).strip()
    except Exception:
        return ""


def _resolved_api_key() -> tuple[str, str]:
    secret_key = _streamlit_secret_key()
    if secret_key:
        return secret_key, "Streamlit secrets"
    env_key = resolve_openai_api_key()
    if env_key:
        return env_key, ".env / environment variable"
    return "", "not configured"


def _render_header() -> None:
    st.title("Agentic LLM Benchmarking Workbench")
    st.caption(
        "Generate answers with selected OpenAI models, assess them with specialist evaluator "
        "agents for faithfulness, toxicity, bias, and reasoning quality, aggregate the evidence "
        "with a separate judge agent, and inspect MLflow runs and OpenTelemetry traces."
    )
    st.markdown(
        """
<div class="author-card">
<strong>Developed by Mayukh Das</strong><br>
TU Braunschweig<br>
<a href="mailto:mayukh@ifis.cs.tu-bs.de">mayukh@ifis.cs.tu-bs.de</a>
</div>
""",
        unsafe_allow_html=True,
    )

    st.subheader("How to use the workbench")
    guide_cols = st.columns(3)
    guide_text = [
        (
            "1. Enter a question or task",
            "Provide a question or instruction. Add optional reference context when you want the faithfulness agent to check support and hallucination.",
        ),
        (
            "2. Select candidate models",
            "Choose up to three curated OpenAI model configurations. The candidate models generate answers, while the evaluation preset in the left sidebar controls the separate evaluator and judge models.",
        ),
        (
            "3. Run and inspect",
            "Compare safety, faithfulness, reasoning, cost, latency, MLflow run metadata, and the OpenTelemetry execution waterfall.",
        ),
    ]
    for column, (heading, body) in zip(guide_cols, guide_text):
        with column:
            st.markdown(
                f'<div class="guide-card"><strong>{heading}</strong><br><br>{body}</div>',
                unsafe_allow_html=True,
            )

    with st.expander("How the agentic evaluation works"):
        st.markdown(
            """
**Question + optional context** → **candidate answer generation** →
**bias, toxicity, faithfulness, and reasoning agents** → **final judge** →
**comparison dashboard + MLflow + OpenTelemetry + JSON evidence bundle**

Candidate identities are hidden from evaluator prompts and execution order is randomized. The report restores model names only for presentation and reproducibility.
"""
        )


def _render_sidebar() -> tuple[str, str, bool, bool]:
    with st.sidebar:
        st.header("Evaluation Configuration")
        api_key, key_source = _resolved_api_key()
        if api_key:
            st.success(f"API key detected via {key_source}.")
        else:
            st.error("No OpenAI API key detected. Configure .env locally or Streamlit secrets in deployment.")

        preset_name = st.selectbox(
            "Evaluation preset",
            options=["Strict", "Balanced"],
            index=0,
            help="Strict uses GPT-5.4 mini for evaluation and judging. Balanced uses GPT-4o mini.",
        )
        preset = get_preset(preset_name)
        st.caption(preset["description"])
        st.markdown(
            f"**Evaluator:** {preset['evaluator']['display_name']}  \n"
            f"**Judge:** {preset['judge']['display_name']}"
        )

        st.divider()
        log_to_mlflow = st.checkbox("Log benchmark to MLflow", value=True)
        save_report = st.checkbox("Save complete JSON locally", value=True)
        st.caption("MLflow UI command")
        st.code("mlflow ui --backend-store-uri sqlite:///mlflow.db", language="bash")
        return api_key, preset_name, log_to_mlflow, save_report


def _model_configuration() -> tuple[list[str], float]:
    all_ids = candidate_config_ids()
    all_labels = [candidate_label(config_id) for config_id in all_ids]
    default_labels = [candidate_label(item) for item in default_candidate_config_ids()]

    st.subheader("Configure Benchmark Models")
    selected_labels = st.multiselect(
        "Candidate model configurations",
        options=all_labels,
        default=default_labels,
        help="Eight curated OpenAI model configurations are available. Select no more than three per run.",
    )
    selected_ids = [label_to_candidate_id(label) for label in selected_labels]
    if len(selected_ids) > 3:
        st.warning(
            "Select at most three candidate configurations to keep the demonstration understandable and cost-controlled."
        )

    with st.expander("Advanced generation setting"):
        temperature = st.slider(
            "Candidate temperature",
            min_value=0.0,
            max_value=1.0,
            value=0.2,
            step=0.1,
            help="Applied to non-reasoning candidate models. Reasoning configurations use their reasoning-effort setting instead.",
        )
    return selected_ids, temperature


def _split_candidate_display_name(display_name: str) -> tuple[str, str]:
    """Separate a model family from an optional reasoning configuration."""
    if " - " not in display_name:
        return display_name, ""
    model_name, configuration = display_name.rsplit(" - ", 1)
    if "reasoning" not in configuration.lower():
        return display_name, ""
    return model_name, configuration


def _row_for_candidate(df: pd.DataFrame, candidate: str) -> pd.Series | None:
    matches = df.loc[df["Candidate"] == candidate]
    if matches.empty:
        return None
    return matches.iloc[0]


def _render_summary_card(
    column: Any,
    *,
    label: str,
    primary: str,
    secondary: str = "",
    badge: str = "",
    badge_tone: str = "neutral",
) -> None:
    """Render a compact result card without st.metric's oversized typography."""
    secondary_html = (
        f'<div class="summary-card-secondary">{escape(secondary)}</div>' if secondary else ""
    )
    badge_html = (
        f'<div class="summary-card-badge {escape(badge_tone)}">{escape(badge)}</div>'
        if badge
        else ""
    )
    with column:
        st.markdown(
            f"""
<div class="summary-card">
  <div class="summary-card-label">{escape(label)}</div>
  <div class="summary-card-primary">{escape(primary)}</div>
  {secondary_html}
  {badge_html}
</div>
""",
            unsafe_allow_html=True,
        )


def _render_overview(report: ComparisonReport, df: pd.DataFrame) -> None:
    best_name, best_config = _split_candidate_display_name(report.best_model)
    fastest_name, fastest_config = _split_candidate_display_name(report.fastest_model)
    value_name, value_config = _split_candidate_display_name(report.best_value_model)

    fastest_row = _row_for_candidate(df, report.fastest_model)
    value_row = _row_for_candidate(df, report.best_value_model)

    fastest_badge = (
        f"{float(fastest_row['Latency (s)']):.2f} s" if fastest_row is not None else ""
    )
    value_badge = (
        f"{float(value_row['Overall']):.1f}/100 · ${float(value_row['Estimated cost (USD)']):.6f}"
        if value_row is not None
        else ""
    )

    metric_cols = st.columns(4)
    _render_summary_card(
        metric_cols[0],
        label="Best overall",
        primary=best_name,
        secondary=best_config,
        badge=f"{report.best_score:.1f}/100",
        badge_tone="positive",
    )
    _render_summary_card(
        metric_cols[1],
        label="Fastest",
        primary=fastest_name,
        secondary=fastest_config,
        badge=fastest_badge,
    )
    _render_summary_card(
        metric_cols[2],
        label="Best value",
        primary=value_name,
        secondary=value_config,
        badge=value_badge,
    )
    _render_summary_card(
        metric_cols[3],
        label="Estimated API cost",
        primary=f"${report.estimated_total_cost_usd:.6f}",
        secondary="Total estimated benchmark cost",
    )

    st.plotly_chart(score_heatmap(df), width="stretch")
    st.plotly_chart(quality_cost_scatter(df), width="stretch")

    st.subheader("Benchmark leaderboard")
    display_columns = [
        "Alias",
        "Candidate",
        "Overall",
        "Bias safety",
        "Toxicity safety",
        "Faithfulness",
        "Reasoning",
        "Latency (s)",
        "Estimated cost (USD)",
        "Verdict",
    ]
    st.dataframe(
        df[display_columns],
        width="stretch",
        hide_index=True,
        column_config={
            "Overall": st.column_config.ProgressColumn(min_value=0, max_value=100),
            "Bias safety": st.column_config.ProgressColumn(min_value=0, max_value=100),
            "Toxicity safety": st.column_config.ProgressColumn(min_value=0, max_value=100),
            "Faithfulness": st.column_config.ProgressColumn(min_value=0, max_value=100),
            "Reasoning": st.column_config.ProgressColumn(min_value=0, max_value=100),
            "Estimated cost (USD)": st.column_config.NumberColumn(format="$%.8f"),
        },
    )


def _render_model_comparison(report: ComparisonReport, df: pd.DataFrame) -> None:
    st.plotly_chart(radar_chart(df), width="stretch")
    for item in report.reports:
        warning = " — self-family evaluator" if item.self_family_evaluation else ""
        with st.expander(
            f"{item.input.candidate_display_name} — {item.overall_score:.1f}/100 — {item.verdict}{warning}",
            expanded=False,
        ):
            answer_col, metadata_col = st.columns([1.4, 1])
            with answer_col:
                st.markdown("**Generated answer**")
                st.write(item.input.answer)
            with metadata_col:
                st.markdown("**Run metadata**")
                st.json(
                    {
                        "blind_alias": item.input.candidate_alias,
                        "candidate_model": item.input.candidate_model,
                        "candidate_reasoning_effort": item.input.candidate_reasoning_effort,
                        "evaluator_model": item.input.evaluator_model,
                        "judge_model": item.input.judge_model,
                        "latency_seconds": item.total_latency_seconds,
                        "estimated_cost_usd": item.estimated_total_cost_usd,
                        "input_tokens": item.total_input_tokens,
                        "output_tokens": item.total_output_tokens,
                        "reasoning_tokens": item.total_reasoning_tokens,
                    }
                )
            if item.self_family_evaluation:
                st.warning(
                    "The candidate and evaluator use the same model family. Correlated preferences may influence the score."
                )


def _render_agent_findings(report: ComparisonReport) -> None:
    for item in report.reports:
        st.subheader(f"{item.input.candidate_display_name} ({item.input.candidate_alias})")
        agents = [item.bias, item.toxicity, item.faithfulness, item.reasoning, item.judge]
        for agent in agents:
            with st.expander(
                f"{agent.agent_name.replace('_', ' ').title()} — {agent.label} — {agent.score:.2f}",
                expanded=False,
            ):
                st.write(agent.explanation)
                if agent.evidence:
                    st.markdown("**Evidence**")
                    for evidence in agent.evidence:
                        st.markdown(f"- {evidence}")
                if agent.uncertainty:
                    st.markdown(f"**Uncertainty:** {agent.uncertainty}")
                if agent.usage:
                    st.caption(
                        f"Model: {agent.usage.model} | Latency: {agent.usage.latency_seconds:.2f}s | "
                        f"Input: {agent.usage.input_tokens} | Output: {agent.usage.output_tokens} | "
                        f"Reasoning: {agent.usage.reasoning_tokens} tokens"
                    )


def _render_mlflow(report: ComparisonReport) -> None:
    manifest = report.mlflow
    if not manifest.enabled:
        st.warning(manifest.error or "MLflow logging was disabled for this run.")
        return
    cols = st.columns(4)
    cols[0].metric("Parent run", manifest.parent_run_id[:12])
    cols[1].metric("Candidate runs", len(manifest.candidate_runs))
    cols[2].metric("Experiment", manifest.experiment_name)
    cols[3].metric("Tracking URI", manifest.tracking_uri)
    st.plotly_chart(mlflow_score_chart(report), width="stretch")
    run_df = pd.DataFrame([record.model_dump() for record in manifest.candidate_runs])
    st.dataframe(run_df, width="stretch", hide_index=True)
    st.info(
        "For MLflow's full experiment UI, run `mlflow ui --backend-store-uri sqlite:///mlflow.db` in a second terminal and open http://localhost:5000."
    )


def _render_trace(report: ComparisonReport) -> None:
    telemetry = report.open_telemetry
    if not telemetry.enabled:
        st.warning(telemetry.error or "No OpenTelemetry trace was captured.")
        return
    spans = telemetry.spans
    slowest = max(spans, key=lambda span: span.duration_ms) if spans else None
    failed = sum(1 for span in spans if span.status == "ERROR")
    cols = st.columns(4)
    cols[0].metric("Trace duration", f"{telemetry.total_duration_ms:.0f} ms")
    cols[1].metric("Finished spans", len(spans))
    cols[2].metric("Failed spans", failed)
    cols[3].metric("Slowest span", slowest.name if slowest else "none")
    st.plotly_chart(trace_waterfall(report), width="stretch")
    trace_df = pd.DataFrame([span.model_dump() for span in spans])
    st.dataframe(
        trace_df[["name", "start_offset_ms", "duration_ms", "status", "attributes"]],
        width="stretch",
        hide_index=True,
    )


def _render_downloads(report: ComparisonReport, df: pd.DataFrame) -> None:
    st.markdown(
        "The complete evidence bundle contains benchmark configuration, generated answers, agent evidence, token/cost data, MLflow run identifiers, and OpenTelemetry spans. It never contains the API key."
    )
    report_json = report.model_dump_json(indent=2)
    st.download_button(
        "Download complete JSON evidence report",
        data=report_json,
        file_name=f"agentic_benchmark_{report.benchmark_id}.json",
        mime="application/json",
        type="primary",
    )
    with st.expander("Technical exports"):
        st.download_button(
            "Download MLflow manifest",
            data=json.dumps(report.mlflow.model_dump(), indent=2),
            file_name=f"mlflow_manifest_{report.benchmark_id}.json",
            mime="application/json",
        )
        st.download_button(
            "Download OpenTelemetry trace JSON",
            data=json.dumps(report.open_telemetry.model_dump(), indent=2),
            file_name=f"otel_trace_{report.benchmark_id}.json",
            mime="application/json",
        )
        st.download_button(
            "Download result table CSV",
            data=df.to_csv(index=False),
            file_name=f"benchmark_results_{report.benchmark_id}.csv",
            mime="text/csv",
        )


def _render_report(report: ComparisonReport) -> None:
    df = comparison_dataframe(report)
    st.success(
        f"Benchmark {report.benchmark_id} completed with {len(report.reports)} successful candidate(s)."
    )
    if report.failures:
        st.warning(f"{len(report.failures)} candidate configuration(s) failed. See details below.")
        st.dataframe(
            pd.DataFrame([failure.model_dump() for failure in report.failures]),
            width="stretch",
            hide_index=True,
        )
    if df.empty:
        st.error("No candidate completed successfully. Review the errors above.")
        return

    tabs = st.tabs(
        [
            "Overview",
            "Model Comparison",
            "Agent Findings",
            "MLflow",
            "OpenTelemetry Traces",
            "Downloads",
        ]
    )
    with tabs[0]:
        _render_overview(report, df)
    with tabs[1]:
        _render_model_comparison(report, df)
    with tabs[2]:
        _render_agent_findings(report)
    with tabs[3]:
        _render_mlflow(report)
    with tabs[4]:
        _render_trace(report)
    with tabs[5]:
        _render_downloads(report, df)


_render_header()
api_key, preset_name, log_to_mlflow, save_report = _render_sidebar()
selected_ids, temperature = _model_configuration()

sample_question = "What is reinforcement learning from human feedback?"
sample_context = (
    "Reinforcement learning from human feedback uses human preference data to train a reward model. "
    "The reward model can then guide optimization of a language model toward outputs preferred by human annotators."
)
question = st.text_area("Question or task", value=sample_question, height=100)
context = st.text_area(
    "Optional reference context",
    value=sample_context,
    height=150,
    help="Faithfulness cannot be fully verified when no reference context is supplied.",
)

invalid_selection = not selected_ids or len(selected_ids) > 3
run_disabled = not api_key or not question.strip() or invalid_selection
if st.button(
    "Run Benchmark",
    type="primary",
    width="stretch",
    disabled=run_disabled,
    help="Configure an API key, enter a question, and select one to three candidates.",
):
    request = GenerationInput(
        question=question.strip(),
        context=context.strip(),
        candidate_config_ids=selected_ids,
        evaluation_preset=preset_name,
        prompt_version=PROMPT_VERSION,
        temperature=temperature,
    )
    try:
        with st.spinner(
            "Generating blind candidate answers, running evaluator agents, judging results, and collecting telemetry..."
        ):
            comparison = EvaluationOrchestrator(api_key).compare_models(
                request,
                log_to_mlflow=log_to_mlflow,
                save_report=save_report,
            )
        st.session_state["latest_benchmark_report"] = comparison.model_dump()
    except Exception as exc:
        st.error(f"Benchmark failed: {type(exc).__name__}: {exc}")
        st.info(
            "Confirm that the key is valid, billing/API access is enabled, and the selected model IDs are available to your OpenAI project."
        )

if run_disabled:
    if not api_key:
        st.caption("The Run Benchmark button is disabled until an OpenAI API key is detected.")
    elif invalid_selection:
        st.caption("Select between one and three candidate configurations.")

stored_report: Dict[str, Any] | None = st.session_state.get("latest_benchmark_report")
if stored_report:
    st.divider()
    _render_report(ComparisonReport.model_validate(stored_report))

st.divider()
st.caption(
    "Agentic LLM Benchmarking Workbench | Developed by Mayukh Das, TU Braunschweig | mayukh@ifis.cs.tu-bs.de"
)
