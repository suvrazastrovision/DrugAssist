"""Optional metadata-only tracing; telemetry never controls application behavior."""

import os
from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps
from inspect import signature
from threading import RLock

from dotenv import load_dotenv

from src.utils.config import ROOT

_client = None
_initialized = False
_lock = RLock()
_active = ContextVar("drugassist_observation", default=None)
_RESULT_FIELDS = ("method", "no_match", "no_match_reason", "answer_status")


def _get_client():
    global _client, _initialized
    with _lock:
        if not _initialized:
            _initialized = True
            try:
                load_dotenv(ROOT / ".env")
                if os.getenv("DRUGASSIST_OBSERVABILITY_ENABLED", "false").lower() != "true":
                    return None
                if not all(os.getenv(key) for key in ("LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY")):
                    return None
                from langfuse import Langfuse
                from opentelemetry.sdk.trace import TracerProvider

                _client = Langfuse(
                    public_key=os.environ["LANGFUSE_PUBLIC_KEY"],
                    secret_key=os.environ["LANGFUSE_SECRET_KEY"],
                    base_url=os.getenv("LANGFUSE_BASE_URL", "https://cloud.langfuse.com"),
                    tracer_provider=TracerProvider(),
                )
            except Exception:  # noqa: BLE001 -- Missing SDK or telemetry must not break the app.
                _client = None
        return _client


def _update(observation, **fields):
    if observation is not None:
        try:
            observation.update(**fields)
        except Exception:  # noqa: BLE001 -- Best-effort telemetry only.
            pass


@contextmanager
def _observation(name, metadata, model=None):
    manager, observation = None, None
    try:
        client = _get_client()
        if client is not None:
            manager = client.start_as_current_observation(
                name=name, as_type="generation" if name == "generation" else "span",
                metadata=metadata, **({"model": model} if model else {}),
            )
            observation = manager.__enter__()
    except Exception:  # noqa: BLE001 -- Initialization/export errors are isolated from application work.
        manager = None
    token = _active.set(observation)
    try:
        yield observation
    except BaseException:
        # Never forward exceptions to the SDK: messages/stack traces can contain private text.
        _update(observation, level="ERROR", status_message="operation_failed")
        raise
    finally:
        _active.reset(token)
        if manager is not None:
            try:
                manager.__exit__(None, None, None)
            except Exception:  # noqa: BLE001 -- Preserve the result or original application exception.
                pass


def traced(name):
    """Trace a service operation without capturing its arguments or returned content."""
    def decorate(function):
        parameters = signature(function)

        @wraps(function)
        def wrapped(*args, **kwargs):
            bound = parameters.bind(*args, **kwargs)
            bound.apply_defaults()
            metadata = {}
            if type(bound.arguments.get("k")) is int:
                metadata["requested_k"] = bound.arguments["k"]
            if bound.arguments.get("audience") in {"patient", "clinician"}:
                metadata["audience"] = bound.arguments["audience"]
            settings = bound.arguments.get("settings")
            model = settings.model if name == "generation" else None
            with _observation(name, metadata, model) as observation:
                result = function(*args, **kwargs)
                if name != "generation":
                    summary = {key: result[key] for key in _RESULT_FIELDS if key in result}
                    summary["result_count"] = len(result.get("results", []))
                    _update(observation, metadata={**metadata, **summary})
                    if result.get("answer_status") == "unavailable":
                        _update(observation, level="ERROR", status_message="answer_unavailable")
                return result

        return wrapped
    return decorate


def record_generation(body):
    """Record model and numeric token counts from this call's provider response only."""
    if not isinstance(body, dict):
        return
    usage = body.get("usage") or {}
    if not isinstance(usage, dict):
        usage = {}
    counts = {
        target: usage[source] for source, target in (
            ("input_tokens", "input"), ("output_tokens", "output"), ("total_tokens", "total"),
        ) if type(usage.get(source)) is int and usage[source] >= 0
    }
    fields = {"usage_details": counts}
    if isinstance(body.get("model"), str):
        fields["model"] = body["model"]
    _update(_active.get(), **fields)


def shutdown():
    """Drain queued telemetry without initializing a client or preventing app cleanup."""
    global _client, _initialized
    with _lock:
        client, _client = _client, None
        _initialized = False
    if client is not None:
        try:
            client.shutdown()
        except Exception:  # noqa: BLE001 -- Shutdown cannot affect application cleanup.
            pass
