from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class GenerationInput(BaseModel):
    question: str = Field(..., min_length=1)
    context: str = ""
    candidate_config_ids: List[str]
    evaluation_preset: str = "Strict"
    prompt_version: str = "v4_blind_rubric"
    temperature: float = 0.2


class EvaluationInput(BaseModel):
    question: str
    answer: str
    context: str = ""
    candidate_alias: str
    candidate_config_id: str
    candidate_display_name: str
    candidate_model: str
    candidate_reasoning_effort: Optional[str] = None
    evaluator_display_name: str
    evaluator_model: str
    evaluator_reasoning_effort: Optional[str] = None
    judge_display_name: str
    judge_model: str
    judge_reasoning_effort: Optional[str] = None
    evaluation_preset: str
    prompt_version: str
    temperature: float = 0.2


class LLMUsage(BaseModel):
    model: str
    reasoning_effort: Optional[str] = None
    response_id: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0
    total_tokens: int = 0
    latency_seconds: float = 0.0
    estimated_cost_usd: float = 0.0


class GeneratedAnswer(BaseModel):
    candidate_alias: str
    candidate_config_id: str
    candidate_display_name: str
    model: str
    reasoning_effort: Optional[str] = None
    answer: str
    usage: LLMUsage


class AgentResult(BaseModel):
    agent_name: str
    score: float = Field(..., ge=0.0, le=1.0)
    label: str
    evidence: List[str] = Field(default_factory=list)
    explanation: str
    uncertainty: str = ""
    usage: Optional[LLMUsage] = None
    duration_seconds: float = 0.0


class EvaluationReport(BaseModel):
    input: EvaluationInput
    generated_answer: GeneratedAnswer
    bias: AgentResult
    toxicity: AgentResult
    faithfulness: AgentResult
    reasoning: AgentResult
    judge: AgentResult
    overall_score: float = Field(..., ge=0.0, le=100.0)
    verdict: str
    summary: str
    total_latency_seconds: float
    total_input_tokens: int
    total_output_tokens: int
    total_reasoning_tokens: int
    estimated_total_cost_usd: float
    self_family_evaluation: bool = False


class CandidateFailure(BaseModel):
    candidate_alias: str
    candidate_config_id: str
    candidate_display_name: str
    error_type: str
    message: str


class MlflowRunRecord(BaseModel):
    candidate_alias: str
    candidate_config_id: str
    run_id: str
    experiment_id: str = ""
    run_name: str = ""
    artifact_uri: str = ""


class MlflowManifest(BaseModel):
    enabled: bool = False
    tracking_uri: str = ""
    experiment_name: str = ""
    parent_run_id: str = ""
    parent_experiment_id: str = ""
    candidate_runs: List[MlflowRunRecord] = Field(default_factory=list)
    error: str = ""


class TraceSpanRecord(BaseModel):
    trace_id: str
    span_id: str
    parent_span_id: str = ""
    name: str
    start_offset_ms: float
    duration_ms: float
    status: str
    attributes: Dict[str, Any] = Field(default_factory=dict)


class TelemetryManifest(BaseModel):
    enabled: bool = False
    service_name: str = ""
    trace_id: str = ""
    total_duration_ms: float = 0.0
    spans: List[TraceSpanRecord] = Field(default_factory=list)
    error: str = ""


class ComparisonReport(BaseModel):
    schema_version: str = "1.0"
    benchmark_id: str
    created_at_utc: str
    question: str
    context: str = ""
    evaluation_preset: str
    prompt_version: str
    requested_candidate_config_ids: List[str]
    randomized_execution_order: List[str]
    evaluator_configuration: Dict[str, Any]
    judge_configuration: Dict[str, Any]
    reports: List[EvaluationReport]
    failures: List[CandidateFailure] = Field(default_factory=list)
    best_model: str = "none"
    best_score: float = 0.0
    fastest_model: str = "none"
    best_value_model: str = "none"
    total_latency_seconds: float = 0.0
    estimated_total_cost_usd: float = 0.0
    mlflow: MlflowManifest = Field(default_factory=MlflowManifest)
    open_telemetry: TelemetryManifest = Field(default_factory=TelemetryManifest)
