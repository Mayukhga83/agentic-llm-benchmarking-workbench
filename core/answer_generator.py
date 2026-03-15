from __future__ import annotations

from typing import Any, Dict

from core.llm_client import LLMClient
from core.schemas import GeneratedAnswer
from tracking.telemetry import traced_span


class AnswerGenerator:
    def __init__(self, api_key: str):
        self.api_key = api_key

    def generate(
        self,
        question: str,
        context: str,
        candidate_alias: str,
        candidate_spec: Dict[str, Any],
        temperature: float,
    ) -> GeneratedAnswer:
        client = LLMClient(candidate_spec, self.api_key)
        system = (
            "You are a candidate model in a controlled LLM benchmark. Answer the user clearly, "
            "accurately, and directly. When reference context is provided, treat it as the primary "
            "source of truth and do not invent unsupported details. Do not mention the benchmark."
        )
        user = (
            f"Question or task:\n{question}\n\n"
            f"Reference context (may be empty):\n{context or '[No reference context supplied]'}"
        )
        with traced_span(
            "generate_answer",
            candidate_alias=candidate_alias,
            candidate_config_id=candidate_spec["config_id"],
            model=candidate_spec["model"],
            reasoning_effort=candidate_spec.get("reasoning_effort") or "none",
        ):
            answer, usage = client.complete(
                system,
                user,
                temperature=temperature,
                max_output_tokens=1800,
            )
        return GeneratedAnswer(
            candidate_alias=candidate_alias,
            candidate_config_id=str(candidate_spec["config_id"]),
            candidate_display_name=str(candidate_spec["display_name"]),
            model=str(candidate_spec["model"]),
            reasoning_effort=candidate_spec.get("reasoning_effort"),
            answer=answer,
            usage=usage,
        )
