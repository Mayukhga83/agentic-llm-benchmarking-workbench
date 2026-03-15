from __future__ import annotations

import json
import tempfile
from pathlib import Path

from core.config import MLFLOW_EXPERIMENT_NAME, MLFLOW_TRACKING_URI
from core.schemas import ComparisonReport, MlflowManifest, MlflowRunRecord

try:
    import mlflow
except Exception:  # pragma: no cover
    mlflow = None


def log_benchmark_report(report: ComparisonReport, enabled: bool = True) -> MlflowManifest:
    if not enabled:
        return MlflowManifest(enabled=False, tracking_uri=MLFLOW_TRACKING_URI)
    if mlflow is None:
        return MlflowManifest(
            enabled=False,
            tracking_uri=MLFLOW_TRACKING_URI,
            experiment_name=MLFLOW_EXPERIMENT_NAME,
            error="MLflow is not installed.",
        )

    try:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
        candidate_records = []
        with mlflow.start_run(run_name=f"benchmark_{report.benchmark_id}") as parent:
            mlflow.set_tag("project", "Agentic LLM Benchmarking Workbench")
            mlflow.set_tag("schema_version", report.schema_version)
            mlflow.set_tag("evaluation_preset", report.evaluation_preset)
            mlflow.log_param("benchmark_id", report.benchmark_id)
            mlflow.log_param("prompt_version", report.prompt_version)
            mlflow.log_param("candidate_count", len(report.reports))
            mlflow.log_param("failure_count", len(report.failures))
            mlflow.log_param("has_context", bool(report.context.strip()))
            mlflow.log_param(
                "evaluator_model", report.evaluator_configuration.get("model", "")
            )
            mlflow.log_param(
                "judge_model", report.judge_configuration.get("model", "")
            )

            for candidate in report.reports:
                run_name = (
                    f"{candidate.input.candidate_alias}_"
                    f"{candidate.input.candidate_config_id}"
                )
                with mlflow.start_run(run_name=run_name, nested=True) as child:
                    _log_candidate(candidate)
                    candidate_records.append(
                        MlflowRunRecord(
                            candidate_alias=candidate.input.candidate_alias,
                            candidate_config_id=candidate.input.candidate_config_id,
                            run_id=child.info.run_id,
                            experiment_id=child.info.experiment_id,
                            run_name=run_name,
                            artifact_uri=child.info.artifact_uri or "",
                        )
                    )

            if report.reports:
                mlflow.log_metric("best_overall_score", report.best_score)
                mlflow.log_metric(
                    "total_estimated_cost_usd", report.estimated_total_cost_usd
                )
                mlflow.log_metric(
                    "benchmark_latency_seconds", report.total_latency_seconds
                )
            with tempfile.TemporaryDirectory() as tmpdir:
                manifest_path = Path(tmpdir) / "benchmark_report_pre_telemetry.json"
                manifest_path.write_text(
                    report.model_dump_json(indent=2), encoding="utf-8"
                )
                mlflow.log_artifact(str(manifest_path), artifact_path="benchmark")

            return MlflowManifest(
                enabled=True,
                tracking_uri=MLFLOW_TRACKING_URI,
                experiment_name=MLFLOW_EXPERIMENT_NAME,
                parent_run_id=parent.info.run_id,
                parent_experiment_id=parent.info.experiment_id,
                candidate_runs=candidate_records,
            )
    except Exception as exc:  # MLflow should not destroy an otherwise valid benchmark
        return MlflowManifest(
            enabled=False,
            tracking_uri=MLFLOW_TRACKING_URI,
            experiment_name=MLFLOW_EXPERIMENT_NAME,
            error=f"{type(exc).__name__}: {exc}",
        )


def _log_candidate(candidate) -> None:
    mlflow.set_tag("candidate_alias", candidate.input.candidate_alias)
    mlflow.set_tag("candidate_model", candidate.input.candidate_model)
    mlflow.set_tag("evaluator_model", candidate.input.evaluator_model)
    mlflow.set_tag("judge_model", candidate.input.judge_model)
    mlflow.log_param("candidate_config_id", candidate.input.candidate_config_id)
    mlflow.log_param("candidate_display_name", candidate.input.candidate_display_name)
    mlflow.log_param(
        "candidate_reasoning_effort",
        candidate.input.candidate_reasoning_effort or "none",
    )
    mlflow.log_param("evaluation_preset", candidate.input.evaluation_preset)
    mlflow.log_param("prompt_version", candidate.input.prompt_version)
    mlflow.log_param("has_context", bool(candidate.input.context.strip()))
    mlflow.log_param("self_family_evaluation", candidate.self_family_evaluation)

    mlflow.log_metric("bias_risk", candidate.bias.score)
    mlflow.log_metric("toxicity_risk", candidate.toxicity.score)
    mlflow.log_metric("bias_safety", 1 - candidate.bias.score)
    mlflow.log_metric("toxicity_safety", 1 - candidate.toxicity.score)
    mlflow.log_metric("faithfulness_quality", candidate.faithfulness.score)
    mlflow.log_metric("reasoning_quality", candidate.reasoning.score)
    mlflow.log_metric("overall_score", candidate.overall_score)
    mlflow.log_metric("total_latency_seconds", candidate.total_latency_seconds)
    mlflow.log_metric("estimated_total_cost_usd", candidate.estimated_total_cost_usd)
    mlflow.log_metric("total_input_tokens", candidate.total_input_tokens)
    mlflow.log_metric("total_output_tokens", candidate.total_output_tokens)
    mlflow.log_metric("total_reasoning_tokens", candidate.total_reasoning_tokens)

    with tempfile.TemporaryDirectory() as tmpdir:
        report_path = Path(tmpdir) / "evaluation_report.json"
        answer_path = Path(tmpdir) / "candidate_answer.txt"
        report_path.write_text(candidate.model_dump_json(indent=2), encoding="utf-8")
        answer_path.write_text(candidate.input.answer, encoding="utf-8")
        mlflow.log_artifact(str(report_path), artifact_path="evidence")
        mlflow.log_artifact(str(answer_path), artifact_path="evidence")
