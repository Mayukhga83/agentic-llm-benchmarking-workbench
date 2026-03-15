from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(ROOT_DIR / ".env", override=False)

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
MLFLOW_EXPERIMENT_NAME = os.getenv(
    "MLFLOW_EXPERIMENT_NAME", "agentic-llm-benchmarking-workbench"
)
OTEL_SERVICE_NAME = os.getenv(
    "OTEL_SERVICE_NAME", "agentic-llm-benchmarking-workbench"
)
OTEL_CONSOLE_EXPORTER = os.getenv("OTEL_CONSOLE_EXPORTER", "false").lower() == "true"
REPORT_DIR = ROOT_DIR / "outputs" / "reports"
PROMPT_VERSION = os.getenv("PROMPT_VERSION", "v4_blind_rubric")


def resolve_openai_api_key(explicit_key: Optional[str] = None) -> str:
    """Resolve a key supplied by the UI first, then fall back to the .env value."""
    candidate = (explicit_key or "").strip() or os.getenv("OPENAI_API_KEY", "").strip()
    if candidate.lower() in {"", "your_key_here", "replace_me"}:
        return ""
    return candidate


def has_openai_api_key(explicit_key: Optional[str] = None) -> bool:
    return bool(resolve_openai_api_key(explicit_key))
