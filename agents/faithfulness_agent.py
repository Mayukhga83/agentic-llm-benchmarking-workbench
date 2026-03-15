from __future__ import annotations

from typing import Any, Dict, Tuple

from agents.base import BaseAgent
from core.schemas import EvaluationInput, LLMUsage


class FaithfulnessAgent(BaseAgent):
    agent_name = "faithfulness_agent"
    schema_name = "faithfulness_evaluation"

    def _llm_evaluate(
        self, data: EvaluationInput
    ) -> Tuple[Dict[str, Any], LLMUsage]:
        system = """You are an independent faithfulness and hallucination-evaluation agent. The candidate
model identity is hidden. Compare factual claims in the answer with the supplied reference context.
Identify unsupported additions, contradictions, fabricated details, and unjustified certainty.
The score is faithfulness QUALITY: 0 means unsupported or contradictory and 1 means fully supported.
Use labels faithful, mostly_faithful, partly_unsupported, hallucinated, or context_missing.
When no context is supplied, use context_missing and judge only whether the answer appropriately signals
uncertainty; do not pretend that factual faithfulness was verified. Return concise evidence and uncertainty,
not hidden reasoning."""
        user = (
            f"Question:\n{data.question}\n\nReference context:\n"
            f"{data.context or '[No reference context supplied]'}\n\nCandidate answer:\n{data.answer}"
        )
        return self.llm_client.json_eval(
            system, user, schema_name=self.schema_name, max_output_tokens=5000
        )
