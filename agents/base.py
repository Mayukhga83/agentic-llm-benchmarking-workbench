from __future__ import annotations

import time
from typing import Any, Dict, Tuple

from core.llm_client import LLMClient
from core.schemas import AgentResult, EvaluationInput, LLMUsage


class BaseAgent:
    agent_name = "base_agent"
    schema_name = "agent_evaluation"

    def __init__(self, llm_client: LLMClient):
        self.llm_client = llm_client

    def evaluate(self, data: EvaluationInput) -> AgentResult:
        started = time.perf_counter()
        raw, usage = self._llm_evaluate(data)
        return self._to_result(raw, usage, time.perf_counter() - started)

    def _llm_evaluate(
        self, data: EvaluationInput
    ) -> Tuple[Dict[str, Any], LLMUsage]:
        raise NotImplementedError

    def _to_result(
        self, raw: Dict[str, Any], usage: LLMUsage, duration: float
    ) -> AgentResult:
        score = max(0.0, min(1.0, float(raw.get("score", 0.0))))
        evidence = raw.get("evidence", []) or []
        if isinstance(evidence, str):
            evidence = [evidence]
        return AgentResult(
            agent_name=self.agent_name,
            score=score,
            label=str(raw.get("label", "unknown")),
            evidence=[str(item) for item in evidence],
            explanation=str(raw.get("explanation", "")),
            uncertainty=str(raw.get("uncertainty", "")),
            usage=usage,
            duration_seconds=round(duration, 4),
        )
