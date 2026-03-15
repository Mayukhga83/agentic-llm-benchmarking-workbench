from __future__ import annotations

import threading
from contextlib import contextmanager
from typing import Any, Dict, Iterator, List, Optional

from core.config import OTEL_CONSOLE_EXPORTER, OTEL_SERVICE_NAME
from core.schemas import TelemetryManifest, TraceSpanRecord

try:
    from opentelemetry import trace
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
    from opentelemetry.sdk.trace.export import (
        ConsoleSpanExporter,
        SimpleSpanProcessor,
        SpanExportResult,
        SpanExporter,
    )
    from opentelemetry.trace import Status, StatusCode

    OTEL_AVAILABLE = True
except Exception:  # pragma: no cover
    OTEL_AVAILABLE = False
    trace = None
    Resource = None
    TracerProvider = None
    ConsoleSpanExporter = None
    SimpleSpanProcessor = None
    SpanExportResult = None
    SpanExporter = object
    Status = None
    StatusCode = None


class CaptureSpanExporter(SpanExporter):  # type: ignore[misc]
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._spans: List[Any] = []

    def export(self, spans: List[Any]) -> Any:
        with self._lock:
            self._spans.extend(spans)
        return SpanExportResult.SUCCESS if SpanExportResult is not None else None

    def shutdown(self) -> None:
        return None

    def clear(self) -> None:
        with self._lock:
            self._spans.clear()

    def spans_for_trace(self, trace_id_hex: str) -> List[Any]:
        with self._lock:
            return [
                span
                for span in self._spans
                if format(span.context.trace_id, "032x") == trace_id_hex
            ]


_capture_exporter = CaptureSpanExporter() if OTEL_AVAILABLE else None
_initialized = False
_tracer = None


def setup_telemetry() -> None:
    global _initialized, _tracer
    if _initialized or not OTEL_AVAILABLE:
        _initialized = True
        return

    resource = Resource.create({"service.name": OTEL_SERVICE_NAME})
    provider = TracerProvider(resource=resource)
    provider.add_span_processor(SimpleSpanProcessor(_capture_exporter))
    if OTEL_CONSOLE_EXPORTER:
        provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))
    trace.set_tracer_provider(provider)
    _tracer = trace.get_tracer(OTEL_SERVICE_NAME)
    _initialized = True


def clear_captured_spans() -> None:
    setup_telemetry()
    if _capture_exporter is not None:
        _capture_exporter.clear()


def get_tracer() -> Any:
    setup_telemetry()
    return _tracer if OTEL_AVAILABLE else None


@contextmanager
def traced_span(name: str, **attributes: Any) -> Iterator[Optional[Any]]:
    tracer = get_tracer()
    if tracer is None:
        yield None
        return

    with tracer.start_as_current_span(name) as span:
        for key, value in attributes.items():
            if value is not None:
                span.set_attribute(str(key), _safe_attribute(value))
        try:
            yield span
            span.set_status(Status(StatusCode.OK))
        except Exception as exc:
            span.record_exception(exc)
            span.set_attribute("error.type", type(exc).__name__)
            span.set_status(Status(StatusCode.ERROR, str(exc)[:500]))
            raise


def trace_id_from_span(span: Optional[Any]) -> str:
    if span is None:
        return ""
    return format(span.get_span_context().trace_id, "032x")


def export_trace_manifest(trace_id_hex: str) -> TelemetryManifest:
    if not OTEL_AVAILABLE or not trace_id_hex or _capture_exporter is None:
        return TelemetryManifest(
            enabled=False,
            service_name=OTEL_SERVICE_NAME,
            error="OpenTelemetry SDK is unavailable or no trace was captured.",
        )

    provider = trace.get_tracer_provider()
    if hasattr(provider, "force_flush"):
        provider.force_flush()
    spans = _capture_exporter.spans_for_trace(trace_id_hex)
    if not spans:
        return TelemetryManifest(
            enabled=False,
            service_name=OTEL_SERVICE_NAME,
            trace_id=trace_id_hex,
            error="No finished spans were found for the benchmark trace.",
        )

    origin_ns = min(span.start_time for span in spans)
    end_ns = max(span.end_time for span in spans)
    records: List[TraceSpanRecord] = []
    for span in sorted(spans, key=lambda item: (item.start_time, item.end_time)):
        parent_span_id = ""
        if span.parent is not None:
            parent_span_id = format(span.parent.span_id, "016x")
        status = getattr(getattr(span, "status", None), "status_code", None)
        status_name = getattr(status, "name", str(status or "UNSET"))
        records.append(
            TraceSpanRecord(
                trace_id=trace_id_hex,
                span_id=format(span.context.span_id, "016x"),
                parent_span_id=parent_span_id,
                name=span.name,
                start_offset_ms=round((span.start_time - origin_ns) / 1_000_000, 3),
                duration_ms=round((span.end_time - span.start_time) / 1_000_000, 3),
                status=status_name,
                attributes={str(k): _json_safe(v) for k, v in span.attributes.items()},
            )
        )
    return TelemetryManifest(
        enabled=True,
        service_name=OTEL_SERVICE_NAME,
        trace_id=trace_id_hex,
        total_duration_ms=round((end_ns - origin_ns) / 1_000_000, 3),
        spans=records,
    )


def _safe_attribute(value: Any) -> Any:
    if isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value]
    return str(value)


def _json_safe(value: Any) -> Any:
    if isinstance(value, (str, bool, int, float)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return str(value)
