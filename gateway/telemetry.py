"""Prometheus metrics and OpenTelemetry tracing for the gateway."""

import os

from prometheus_client import Counter, Gauge, Histogram, make_asgi_app


HTTP_REQUESTS = Counter(
    "intelliway_gateway_http_requests_total",
    "Completed gateway HTTP requests.",
    ("method", "route", "status"),
)
HTTP_DURATION = Histogram(
    "intelliway_gateway_http_request_duration_seconds",
    "Gateway HTTP request duration in seconds.",
    ("method", "route"),
)
UPSTREAM_CALLS = Counter(
    "intelliway_gateway_upstream_calls_total",
    "Upstream calls made by the gateway.",
    ("upstream", "outcome"),
)
CIRCUIT_STATE = Gauge(
    "intelliway_gateway_circuit_open",
    "Whether an upstream circuit is open.",
    ("upstream",),
)
UPSTREAM_HEALTH = Gauge(
    "intelliway_gateway_upstream_health_score",
    "Recent upstream success ratio from 0 to 100.",
    ("upstream",),
)

metrics_app = make_asgi_app()


def configure_tracing(app) -> None:
    endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT")
    if not endpoint:
        return
    from opentelemetry import trace
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
    from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
    from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor

    provider = TracerProvider(
        resource=Resource.create({"service.name": os.getenv("OTEL_SERVICE_NAME", "intelliway-gateway")})
    )
    exporter = OTLPSpanExporter(endpoint=endpoint, insecure=endpoint.startswith("http://"))
    provider.add_span_processor(BatchSpanProcessor(exporter))
    trace.set_tracer_provider(provider)
    FastAPIInstrumentor.instrument_app(app)
    HTTPXClientInstrumentor().instrument()


def observe_upstream(upstream: str, outcome: str) -> None:
    UPSTREAM_CALLS.labels(upstream=upstream, outcome=outcome).inc()

