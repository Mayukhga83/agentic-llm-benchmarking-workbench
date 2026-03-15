from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional

CATALOG_PATH = Path(__file__).resolve().parents[1] / "data" / "models.json"


@lru_cache(maxsize=1)
def load_catalog() -> Dict[str, Any]:
    return json.loads(CATALOG_PATH.read_text(encoding="utf-8"))


def candidate_configurations() -> List[Dict[str, Any]]:
    return [
        item
        for item in load_catalog().get("candidate_configurations", [])
        if item.get("enabled", True)
    ]


def candidate_config_ids() -> List[str]:
    return [item["config_id"] for item in candidate_configurations()]


def default_candidate_config_ids() -> List[str]:
    valid = set(candidate_config_ids())
    return [
        config_id
        for config_id in load_catalog().get("default_candidate_configurations", [])
        if config_id in valid
    ]


def get_candidate_config(config_id: str) -> Dict[str, Any]:
    for item in candidate_configurations():
        if item["config_id"] == config_id:
            return dict(item)
    raise KeyError(f"Unknown candidate configuration: {config_id}")


def candidate_label(config_id: str) -> str:
    return str(get_candidate_config(config_id)["display_name"])


def label_to_candidate_id(label: str) -> str:
    for item in candidate_configurations():
        if item["display_name"] == label:
            return str(item["config_id"])
    raise KeyError(f"Unknown candidate label: {label}")


def presets() -> Dict[str, Dict[str, Any]]:
    return dict(load_catalog().get("presets", {}))


def get_preset(name: str) -> Dict[str, Any]:
    try:
        return dict(presets()[name])
    except KeyError as exc:
        raise KeyError(f"Unknown evaluation preset: {name}") from exc


def estimate_cost_from_usage(
    model_spec: Dict[str, Any], input_tokens: int, output_tokens: int
) -> float:
    input_price = float(model_spec.get("input_price_per_million", 0.0))
    output_price = float(model_spec.get("output_price_per_million", 0.0))
    return round(
        (input_tokens / 1_000_000) * input_price
        + (output_tokens / 1_000_000) * output_price,
        8,
    )


def same_model_family(candidate_model: str, evaluator_model: str) -> bool:
    return candidate_model == evaluator_model


def public_model_spec(spec: Dict[str, Any]) -> Dict[str, Optional[str] | float]:
    """Return only report-safe model metadata."""
    return {
        "display_name": str(spec.get("display_name", spec.get("model", "unknown"))),
        "model": str(spec.get("model", "unknown")),
        "reasoning_effort": spec.get("reasoning_effort"),
        "input_price_per_million": float(spec.get("input_price_per_million", 0.0)),
        "output_price_per_million": float(spec.get("output_price_per_million", 0.0)),
    }
