from __future__ import annotations

import random
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List
from uuid import uuid4

from agents.bias_agent import BiasAgent
from agents.faithfulness_agent import FaithfulnessAgent
from agents.judge_agent import JudgeAgent
from agents.reasoning_agent import ReasoningAgent
from agents.toxicity_agent import ToxicityAgent
from core.answer_generator import AnswerGenerator
from core.config import REPORT_DIR
from core.llm_client import LLMClient
from core.model_catalog import (
    get_candidate_config,
    get_preset,
    public_model_spec,
    same_model_family,
)
from core.schemas import (
    AgentResult,
    CandidateFailure,
    ComparisonReport,
    EvaluationInput,
    EvaluationReport,
    GenerationInput,
)
from tracking.mlflow_logger import log_benchmark_report
from tracking.telemetry import (
    clear_captured_spans,
    export_trace_manifest,
    trace_id_from_span,
    traced_span,
)


class EvaluationOrchestrator:
    def __init__(self, api_key: str):
        if not api_key:
            raise ValueError("OPENAI_API_KEY is required for this benchmark.")
        self.api_key = api_key
        self.answer_generator = AnswerGenerator(api_key)

    def compare_models(
        self,
        request: GenerationInput,
        *,
        log_to_mlflow: bool = True,
        save_report: bool = True,
    ) -> ComparisonReport:
        benchmark_id = uuid4().hex[:12]
        started = time.perf_counter()
        preset = get_preset(request.evaluation_preset)
        evaluator_spec = dict(preset["evaluator"])
        judge_spec = dict(preset["judge"])

        requested_specs = [
            get_candidate_config(config_id)
            for config_id in request.candidate_config_ids
        ]
        randomized_specs = list(requested_specs)
        random.Random(benchmark_id).shuffle(randomized_specs)
        aliases = [f"Candidate {chr(65 + index)}" for index in range(len(randomized_specs))]
        execution_order = [
            f"{alias}: {spec['config_id']}"
            for alias, spec in zip(aliases, randomized_specs)
        ]

        clear_captured_spans()
        trace_id = ""
        reports: List[EvaluationReport] = []
        failures: List[CandidateFailure] = []
        comparison: ComparisonReport

        with traced_span(
            "benchmark_request",
            benchmark_id=benchmark_id,
            evaluation_preset=request.evaluation_preset,
            candidate_count=len(randomized_specs),
            evaluator_model=evaluator_spec["model"],
            judge_model=judge_spec["model"],
        ) as root_span:
            trace_id = trace_id_from_span(root_span)
            for alias, candidate_spec in zip(aliases, randomized_specs):
                try:
                    with traced_span(
                        "candidate_benchmark",
                        candidate_alias=alias,
                        candidate_config_id=candidate_spec["config_id"],
                        candidate_model=candidate_spec["model"],
                    ):
                        reports.append(
                            self._generate_and_evaluate(
                                request=request,
                                candidate_alias=alias,
                                candidate_spec=candidate_spec,
                                evaluator_spec=evaluator_spec,
                                judge_spec=judge_spec,
                            )
                        )
                except Exception as exc:
                    failures.append(
                        CandidateFailure(
                            candidate_alias=alias,
                            candidate_config_id=str(candidate_spec["config_id"]),
                            candidate_display_name=str(candidate_spec["display_name"]),
                            error_type=type(exc).__name__,
                            message=str(exc)[:1000],
                        )
                    )

            total_latency = time.perf_counter() - started
            best = max(reports, key=lambda item: item.overall_score) if reports else None
            fastest = (
                min(reports, key=lambda item: item.total_latency_seconds)
                if reports
                else None
            )
            value = self._best_value(reports)
            comparison = ComparisonReport(
                benchmark_id=benchmark_id,
                created_at_utc=datetime.now(timezone.utc).isoformat(),
                question=request.question,
                context=request.context,
                evaluation_preset=request.evaluation_preset,
                prompt_version=request.prompt_version,
                requested_candidate_config_ids=request.candidate_config_ids,
                randomized_execution_order=execution_order,
                evaluator_configuration=public_model_spec(evaluator_spec),
                judge_configuration=public_model_spec(judge_spec),
                reports=self._restore_requested_order(reports, request.candidate_config_ids),
                failures=failures,
                best_model=best.input.candidate_display_name if best else "none",
                best_score=best.overall_score if best else 0.0,
                fastest_model=fastest.input.candidate_display_name if fastest else "none",
                best_value_model=value.input.candidate_display_name if value else "none",
                total_latency_seconds=round(total_latency, 4),
                estimated_total_cost_usd=round(
                    sum(item.estimated_total_cost_usd for item in reports), 8
                ),
            )
            with traced_span("mlflow_logging", benchmark_id=benchmark_id):
                comparison.mlflow = log_benchmark_report(
                    comparison, enabled=log_to_mlflow
                )

        comparison.open_telemetry = export_trace_manifest(trace_id)
        if save_report:
            self._save_report(comparison)
        return comparison

    def _generate_and_evaluate(
        self,
        *,
        request: GenerationInput,
        candidate_alias: str,
        candidate_spec: Dict[str, Any],
        evaluator_spec: Dict[str, Any],
        judge_spec: Dict[str, Any],
    ) -> EvaluationReport:
        started = time.perf_counter()
        generated = self.answer_generator.generate(
            question=request.question,
            context=request.context,
            candidate_alias=candidate_alias,
            candidate_spec=candidate_spec,
            temperature=request.temperature,
        )
        data = EvaluationInput(
            question=request.question,
            answer=generated.answer,
            context=request.context,
            candidate_alias=candidate_alias,
            candidate_config_id=str(candidate_spec["config_id"]),
            candidate_display_name=str(candidate_spec["display_name"]),
            candidate_model=str(candidate_spec["model"]),
            candidate_reasoning_effort=candidate_spec.get("reasoning_effort"),
            evaluator_display_name=str(evaluator_spec["display_name"]),
            evaluator_model=str(evaluator_spec["model"]),
            evaluator_reasoning_effort=evaluator_spec.get("reasoning_effort"),
            judge_display_name=str(judge_spec["display_name"]),
            judge_model=str(judge_spec["model"]),
            judge_reasoning_effort=judge_spec.get("reasoning_effort"),
            evaluation_preset=request.evaluation_preset,
            prompt_version=request.prompt_version,
            temperature=request.temperature,
        )

        evaluator_clients = [
            LLMClient(evaluator_spec, self.api_key) for _ in range(4)
        ]
        agents = [
            ("bias_agent", BiasAgent(evaluator_clients[0])),
            ("toxicity_agent", ToxicityAgent(evaluator_clients[1])),
            ("faithfulness_agent", FaithfulnessAgent(evaluator_clients[2])),
            ("reasoning_agent", ReasoningAgent(evaluator_clients[3])),
        ]
        results: Dict[str, AgentResult] = {}
        for span_name, agent in agents:
            with traced_span(
                span_name,
                candidate_alias=candidate_alias,
                evaluator_model=evaluator_spec["model"],
                reasoning_effort=evaluator_spec.get("reasoning_effort") or "none",
            ):
                results[span_name] = agent.evaluate(data)

        judge = JudgeAgent(LLMClient(judge_spec, self.api_key))
        with traced_span(
            "judge_agent",
            candidate_alias=candidate_alias,
            judge_model=judge_spec["model"],
            reasoning_effort=judge_spec.get("reasoning_effort") or "none",
        ):
            judge_result = judge.evaluate_combined(
                data,
                results["bias_agent"],
                results["toxicity_agent"],
                results["faithfulness_agent"],
                results["reasoning_agent"],
            )

        usages = [
            generated.usage,
            results["bias_agent"].usage,
            results["toxicity_agent"].usage,
            results["faithfulness_agent"].usage,
            results["reasoning_agent"].usage,
            judge_result.usage,
        ]
        valid_usages = [usage for usage in usages if usage is not None]
        return EvaluationReport(
            input=data,
            generated_answer=generated,
            bias=results["bias_agent"],
            toxicity=results["toxicity_agent"],
            faithfulness=results["faithfulness_agent"],
            reasoning=results["reasoning_agent"],
            judge=judge_result,
            overall_score=round(judge_result.score * 100, 2),
            verdict=judge_result.label,
            summary=judge_result.explanation,
            total_latency_seconds=round(time.perf_counter() - started, 4),
            total_input_tokens=sum(usage.input_tokens for usage in valid_usages),
            total_output_tokens=sum(usage.output_tokens for usage in valid_usages),
            total_reasoning_tokens=sum(
                usage.reasoning_tokens for usage in valid_usages
            ),
            estimated_total_cost_usd=round(
                sum(usage.estimated_cost_usd for usage in valid_usages), 8
            ),
            self_family_evaluation=same_model_family(
                str(candidate_spec["model"]), str(evaluator_spec["model"])
            ),
        )

    @staticmethod
    def _restore_requested_order(
        reports: List[EvaluationReport], requested_ids: List[str]
    ) -> List[EvaluationReport]:
        order = {config_id: index for index, config_id in enumerate(requested_ids)}
        return sorted(
            reports,
            key=lambda report: order.get(report.input.candidate_config_id, 999),
        )

    @staticmethod
    def _best_value(reports: List[EvaluationReport]) -> EvaluationReport | None:
        if not reports:
            return None
        acceptable = [report for report in reports if report.overall_score >= 70]
        if acceptable:
            return min(
                acceptable,
                key=lambda report: (
                    report.estimated_total_cost_usd,
                    -report.overall_score,
                ),
            )
        return max(
            reports,
            key=lambda report: report.overall_score
            / max(report.estimated_total_cost_usd, 0.00000001),
        )

    @staticmethod
    def _save_report(report: ComparisonReport) -> Path:
        REPORT_DIR.mkdir(parents=True, exist_ok=True)
        path = REPORT_DIR / f"benchmark_{report.benchmark_id}.json"
        path.write_text(report.model_dump_json(indent=2), encoding="utf-8")
        return path
