"""Small, dependency-free inference API adapter for latency benchmarks."""

from __future__ import annotations

import inspect
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any


INFERENCE_MODES = ("legacy", "streaming")
COMPILE_SCOPES = (
    "none",
    "legacy_infer_action",
    "streaming_action_kernels",
)


@dataclass(frozen=True)
class InferenceAPI:
    mode: str
    canonical_name: str
    canonical: Callable[..., Any]
    profiled_name: str | None
    profiled: Callable[..., Any] | None

    @property
    def stateful(self) -> bool:
        return self.mode == "streaming"


@dataclass(frozen=True)
class CompileExecution:
    enabled: bool
    scope: str
    canonical_execution_mode: str
    profiled_execution_mode: str
    public_wrapper_execution_mode: str
    action_core_execution_mode: str
    profiled_matches_canonical: bool


def describe_compile_execution(model: Any, mode: str) -> CompileExecution:
    """Describe model-level compile scope without guessing from method names."""
    if mode not in INFERENCE_MODES:
        raise ValueError(f"Unsupported inference mode {mode!r}; expected one of {INFERENCE_MODES}.")

    enabled = bool(getattr(model, "torch_compile_infer_action", False))
    scope = str(getattr(model, "torch_compile_scope", "none"))
    if not enabled:
        return CompileExecution(
            enabled=False,
            scope="none",
            canonical_execution_mode="eager",
            profiled_execution_mode="eager_instrumented",
            public_wrapper_execution_mode="eager",
            action_core_execution_mode="eager",
            profiled_matches_canonical=True,
        )

    if scope not in COMPILE_SCOPES or scope == "none":
        raise RuntimeError(
            "torch.compile is enabled but the model did not expose a supported "
            f"compile scope, got {scope!r}."
        )
    if mode == "legacy" and scope != "legacy_infer_action":
        raise RuntimeError(
            f"Legacy inference requires compile scope 'legacy_infer_action', got {scope!r}."
        )
    if mode == "streaming" and scope != "streaming_action_kernels":
        raise RuntimeError(
            "Streaming inference requires compile scope "
            f"'streaming_action_kernels', got {scope!r}."
        )

    if scope == "legacy_infer_action":
        return CompileExecution(
            enabled=True,
            scope=scope,
            canonical_execution_mode="compiled_public_method",
            profiled_execution_mode="eager_instrumented",
            public_wrapper_execution_mode="compiled",
            action_core_execution_mode="compiled",
            profiled_matches_canonical=False,
        )

    return CompileExecution(
        enabled=True,
        scope=scope,
        canonical_execution_mode="eager_wrapper_with_compiled_action_kernels",
        profiled_execution_mode="eager_wrapper_with_compiled_action_kernels_instrumented",
        public_wrapper_execution_mode="eager",
        action_core_execution_mode="compiled",
        profiled_matches_canonical=True,
    )


def _require_callable(model: Any, name: str) -> Callable[..., Any]:
    method = getattr(model, name, None)
    if not callable(method):
        raise RuntimeError(
            f"{type(model).__name__} does not provide required {name}(). "
            "The benchmark will not substitute legacy latency for a missing streaming API."
        )
    return method


def resolve_inference_api(
    model: Any,
    mode: str,
    *,
    require_profiled: bool,
) -> InferenceAPI:
    """Resolve explicit legacy or streaming methods without silent fallback."""

    if mode not in INFERENCE_MODES:
        raise ValueError(f"Unsupported inference mode {mode!r}; expected one of {INFERENCE_MODES}.")

    if mode == "legacy":
        canonical_name = "infer_action"
        profiled_name = "infer_action_profiled"
    else:
        canonical_name = "infer_action_streaming"
        profiled_name = "infer_action_streaming_profiled"

    canonical = _require_callable(model, canonical_name)
    profiled = getattr(model, profiled_name, None)
    if profiled is not None and not callable(profiled):
        raise RuntimeError(f"{type(model).__name__}.{profiled_name} exists but is not callable.")
    if require_profiled and profiled is None:
        raise RuntimeError(
            f"{type(model).__name__} does not provide required {profiled_name}(). "
            "A two-stage report requires real model instrumentation; no synthetic breakdown is emitted."
        )

    return InferenceAPI(
        mode=mode,
        canonical_name=canonical_name,
        canonical=canonical,
        profiled_name=profiled_name if profiled is not None else None,
        profiled=profiled,
    )


def accepts_keyword(method: Callable[..., Any], name: str) -> bool:
    """Return whether ``method`` explicitly or variadically accepts a keyword."""

    parameters = inspect.signature(method).parameters
    return name in parameters or any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in parameters.values()
    )


def unpack_inference_result(result: Any, mode: str) -> tuple[Any, Any | None]:
    """Return ``(action, state)`` from the benchmark inference protocol.

    Legacy inference must return a mapping with ``action``. Streaming inference
    may return ``(prediction, state)`` or a mapping with both ``action`` and
    ``streaming_state``. ``action=None`` is reserved for cold start.
    """

    if mode == "legacy":
        if not isinstance(result, Mapping) or "action" not in result:
            raise RuntimeError("Legacy infer_action() must return a mapping containing 'action'.")
        return result["action"], None

    payload = result
    state = None
    if isinstance(result, tuple):
        if len(result) != 2:
            raise RuntimeError("Streaming tuple results must be (prediction, streaming_state).")
        payload, state = result
    elif isinstance(result, Mapping):
        if "streaming_state" not in result:
            raise RuntimeError(
                "Streaming mapping results must contain 'streaming_state'; state is caller-owned."
            )
        state = result["streaming_state"]
    else:
        raise RuntimeError(
            "Streaming inference must return (prediction, state) or a mapping with streaming_state."
        )

    if isinstance(payload, Mapping):
        if "action" not in payload:
            raise RuntimeError("Streaming prediction mapping must contain 'action'.")
        action = payload["action"]
    else:
        action = payload
    if state is None:
        raise RuntimeError("Streaming inference returned no state; state must remain caller-owned.")
    return action, state
