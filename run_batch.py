from __future__ import annotations

import json
from pathlib import Path

from core.config import PROMPT_VERSION, resolve_openai_api_key
from core.model_catalog import default_candidate_config_ids
from core.orchestrator import EvaluationOrchestrator
from core.schemas import GenerationInput


def main() -> None:
    api_key = resolve_openai_api_key()
    if not api_key:
        raise SystemExit(
            "OPENAI_API_KEY was not found. Copy .env.example to .env and add your key."
        )

    examples = json.loads(
        Path("data/sample_inputs.json").read_text(encoding="utf-8")
    )
    orchestrator = EvaluationOrchestrator(api_key)

    for index, example in enumerate(examples, start=1):
        print(f"\n=== Example {index}: {example['name']} ===")
        request = GenerationInput(
            question=example["question"],
            context=example.get("context", ""),
            candidate_config_ids=default_candidate_config_ids(),
            evaluation_preset="Strict",
            prompt_version=PROMPT_VERSION,
            temperature=0.2,
        )
        report = orchestrator.compare_models(
            request,
            log_to_mlflow=True,
            save_report=True,
        )
        print(
            f"Best: {report.best_model} ({report.best_score:.1f}/100) | "
            f"Fastest: {report.fastest_model} | "
            f"Estimated cost: ${report.estimated_total_cost_usd:.6f}"
        )
        for candidate in report.reports:
            print(
                f"- {candidate.input.candidate_display_name}: "
                f"overall={candidate.overall_score:.1f}, "
                f"verdict={candidate.verdict}, "
                f"latency={candidate.total_latency_seconds:.2f}s, "
                f"cost=${candidate.estimated_total_cost_usd:.6f}"
            )
        for failure in report.failures:
            print(
                f"- FAILED {failure.candidate_display_name}: "
                f"{failure.error_type}: {failure.message}"
            )


if __name__ == "__main__":
    main()
