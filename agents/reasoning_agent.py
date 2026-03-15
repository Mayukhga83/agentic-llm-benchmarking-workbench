from __future__ import annotations

from typing import Any, Dict, Tuple

from agents.base import BaseAgent
from core.schemas import EvaluationInput, LLMUsage


class ReasoningAgent(BaseAgent):
    agent_name = "reasoning_agent"
    schema_name = "reasoning_evaluation"

    def _llm_evaluate(
        self, data: EvaluationInput
    ) -> Tuple[Dict[str, Any], LLMUsage]:
        system = """You are an independent reasoning-quality evaluation agent. The candidate model identity
is hidden. Assess logical consistency, relevance, causal validity, support for conclusions, treatment of
uncertainty, contradictions, and whether the response actually answers the task.
The score is reasoning QUALITY: 0 means fundamentally defective and 1 means excellent.
Use labels poor, fair, good, or excellent. Provide concise observable evidence and uncertainty; never expose
or request hidden chain-of-thought. Return the required structured object."""
        user = (
            f"Question:\n{data.question}\n\nReference context (may be empty):\n"
            f"{data.context or '[No reference context supplied]'}\n\nCandidate answer:\n{data.answer}"
        )
        return self.llm_client.json_eval(
            system, user, schema_name=self.schema_name, max_output_tokens=5000
        )
