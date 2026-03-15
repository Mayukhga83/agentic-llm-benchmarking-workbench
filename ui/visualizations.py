from __future__ import annotations

from typing import Any, Dict, List

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from core.schemas import ComparisonReport


def comparison_dataframe(report: ComparisonReport) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for item in report.reports:
        rows.append(
            {
                "Alias": item.input.candidate_alias,
                "Candidate": item.input.candidate_display_name,
                "Model": item.input.candidate_model,
                "Overall": item.overall_score,
                "Bias safety": round((1 - item.bias.score) * 100, 2),
                "Toxicity safety": round((1 - item.toxicity.score) * 100, 2),
                "Faithfulness": round(item.faithfulness.score * 100, 2),
                "Reasoning": round(item.reasoning.score * 100, 2),
                "Latency (s)": item.total_latency_seconds,
                "Estimated cost (USD)": item.estimated_total_cost_usd,
                "Input tokens": item.total_input_tokens,
                "Output tokens": item.total_output_tokens,
                "Reasoning tokens": item.total_reasoning_tokens,
                "Verdict": item.verdict,
            }
        )
    return pd.DataFrame(rows)


def score_heatmap(df: pd.DataFrame) -> go.Figure:
    metrics = [
        "Bias safety",
        "Toxicity safety",
        "Faithfulness",
        "Reasoning",
        "Overall",
    ]
    z = df[metrics].values.tolist()
    fig = go.Figure(
        data=go.Heatmap(
            z=z,
            x=metrics,
            y=df["Candidate"].tolist(),
            zmin=0,
            zmax=100,
            text=[[f"{value:.1f}" for value in row] for row in z],
            texttemplate="%{text}",
            hovertemplate="%{y}<br>%{x}: %{z:.1f}<extra></extra>",
            colorbar={"title": "Score"},
        )
    )
    fig.update_layout(
        title="Evaluation score matrix",
        xaxis_title="Metric",
        yaxis_title="Candidate configuration",
        height=max(330, 95 * len(df)),
        margin=dict(l=20, r=20, t=60, b=20),
    )
    return fig


def quality_cost_scatter(df: pd.DataFrame) -> go.Figure:
    plot_df = df.copy()
    plot_df["Bubble size"] = plot_df["Latency (s)"].clip(lower=0.2)
    fig = px.scatter(
        plot_df,
        x="Estimated cost (USD)",
        y="Overall",
        size="Bubble size",
        text="Candidate",
        hover_data=[
            "Alias",
            "Latency (s)",
            "Faithfulness",
            "Reasoning",
            "Verdict",
        ],
        size_max=42,
        title="Quality–cost–latency trade-off",
    )
    fig.update_traces(textposition="top center")
    fig.update_layout(
        xaxis_title="Estimated total API cost (USD)",
        yaxis_title="Overall quality score",
        yaxis_range=[0, 105],
        height=470,
        margin=dict(l=20, r=20, t=60, b=20),
    )
    return fig


def radar_chart(df: pd.DataFrame) -> go.Figure:
    metrics = ["Bias safety", "Toxicity safety", "Faithfulness", "Reasoning", "Overall"]
    fig = go.Figure()
    for _, row in df.head(3).iterrows():
        values = [float(row[metric]) for metric in metrics]
        fig.add_trace(
            go.Scatterpolar(
                r=values + [values[0]],
                theta=metrics + [metrics[0]],
                fill="toself",
                name=str(row["Candidate"]),
            )
        )
    fig.update_layout(
        title="Capability and safety profile",
        polar={"radialaxis": {"visible": True, "range": [0, 100]}},
        height=480,
        margin=dict(l=30, r=30, t=70, b=30),
    )
    return fig


def mlflow_score_chart(report: ComparisonReport) -> go.Figure:
    data = [
        {
            "Candidate": item.input.candidate_display_name,
            "Overall": item.overall_score,
        }
        for item in report.reports
    ]
    df = pd.DataFrame(data)
    fig = px.bar(
        df,
        x="Candidate",
        y="Overall",
        text_auto=".1f",
        title="MLflow candidate runs by overall score",
    )
    fig.update_layout(
        yaxis_range=[0, 105],
        yaxis_title="Overall score",
        xaxis_title="Candidate run",
        height=400,
        margin=dict(l=20, r=20, t=60, b=20),
    )
    return fig


def trace_waterfall(report: ComparisonReport) -> go.Figure:
    spans = report.open_telemetry.spans
    labels = []
    starts = []
    durations = []
    hover = []
    for span in spans:
        alias = span.attributes.get("candidate_alias", "")
        label = f"{span.name} — {alias}" if alias else span.name
        labels.append(label)
        starts.append(span.start_offset_ms)
        durations.append(max(span.duration_ms, 0.05))
        hover.append(
            f"Span: {span.name}<br>Start: {span.start_offset_ms:.2f} ms"
            f"<br>Duration: {span.duration_ms:.2f} ms<br>Status: {span.status}"
        )
    fig = go.Figure(
        go.Bar(
            x=durations,
            y=labels,
            base=starts,
            orientation="h",
            text=[f"{duration:.0f} ms" for duration in durations],
            hovertext=hover,
            hoverinfo="text",
        )
    )
    fig.update_layout(
        title="OpenTelemetry trace waterfall",
        xaxis_title="Time from benchmark start (ms)",
        yaxis_title="Span",
        yaxis={"autorange": "reversed"},
        height=max(480, 30 * len(spans) + 120),
        margin=dict(l=20, r=20, t=60, b=20),
        showlegend=False,
    )
    return fig
