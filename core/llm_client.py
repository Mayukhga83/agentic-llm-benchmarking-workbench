from __future__ import annotations

import json
import time
from typing import Any, Dict, Tuple

from core.model_catalog import estimate_cost_from_usage
from core.schemas import LLMUsage


EVALUATION_JSON_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "score": {"type": "number"},
        "label": {"type": "string"},
        "evidence": {"type": "array", "items": {"type": "string"}},
        "explanation": {"type": "string"},
        "uncertainty": {"type": "string"},
    },
    "required": ["score", "label", "evidence", "explanation", "uncertainty"],
    "additionalProperties": False,
}


class LLMClient:
    """Small OpenAI Responses API wrapper with model configuration and usage capture."""

    def __init__(self, model_spec: Dict[str, Any], api_key: str):
        if not api_key:
            raise ValueError("OPENAI_API_KEY is not configured.")
        self.model_spec = dict(model_spec)
        self.model = str(model_spec["model"])
        self.reasoning_effort = model_spec.get("reasoning_effort")
        from openai import OpenAI

        self._client = OpenAI(api_key=api_key)

    def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        temperature: float = 0.2,
        max_output_tokens: int = 1800,
    ) -> Tuple[str, LLMUsage]:
        kwargs: Dict[str, Any] = {
            "model": self.model,
            "instructions": system_prompt,
            "input": user_prompt,
            "store": False,
            "max_output_tokens": max_output_tokens,
        }
        if self.reasoning_effort:
            kwargs["reasoning"] = {"effort": self.reasoning_effort}
        else:
            kwargs["temperature"] = temperature

        started = time.perf_counter()
        response = self._client.responses.create(**kwargs)
        latency = time.perf_counter() - started
        text = response.output_text or ""
        if not text.strip():
            status = getattr(response, "status", "unknown")
            incomplete = getattr(response, "incomplete_details", None)
            raise RuntimeError(
                f"Model returned no visible text (status={status}, incomplete_details={incomplete})."
            )
        return text, self._usage_from_response(response, latency)

    def json_eval(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        schema_name: str,
        max_output_tokens: int = 5000,
    ) -> Tuple[Dict[str, Any], LLMUsage]:
        kwargs: Dict[str, Any] = {
            "model": self.model,
            "instructions": system_prompt,
            "input": user_prompt,
            "store": False,
            "max_output_tokens": max_output_tokens,
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": schema_name,
                    "strict": True,
                    "schema": EVALUATION_JSON_SCHEMA,
                },
                "verbosity": "low",
            },
        }
        if self.reasoning_effort:
            kwargs["reasoning"] = {"effort": self.reasoning_effort}
        else:
            kwargs["temperature"] = 0

        started = time.perf_counter()
        response = self._client.responses.create(**kwargs)
        latency = time.perf_counter() - started
        content = response.output_text or ""
        if not content.strip():
            status = getattr(response, "status", "unknown")
            incomplete = getattr(response, "incomplete_details", None)
            raise RuntimeError(
                f"Evaluator returned no JSON text (status={status}, incomplete_details={incomplete})."
            )
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Evaluator returned invalid JSON: {exc}") from exc
        return parsed, self._usage_from_response(response, latency)

    def _usage_from_response(self, response: Any, latency: float) -> LLMUsage:
        usage = getattr(response, "usage", None)
        input_tokens = int(getattr(usage, "input_tokens", 0) or 0)
        output_tokens = int(getattr(usage, "output_tokens", 0) or 0)
        total_tokens = int(getattr(usage, "total_tokens", 0) or 0)
        details = getattr(usage, "output_tokens_details", None)
        reasoning_tokens = int(getattr(details, "reasoning_tokens", 0) or 0)
        cost = estimate_cost_from_usage(self.model_spec, input_tokens, output_tokens)
        return LLMUsage(
            model=self.model,
            reasoning_effort=self.reasoning_effort,
            response_id=str(getattr(response, "id", "") or ""),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            reasoning_tokens=reasoning_tokens,
            total_tokens=total_tokens or input_tokens + output_tokens,
            latency_seconds=round(latency, 4),
            estimated_cost_usd=cost,
        )
