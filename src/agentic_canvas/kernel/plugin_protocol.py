from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass(frozen=True)
class PluginInputRequest:
    plugin_name: str
    mode: str
    request_id: str
    message: str
    patch: dict[str, Any]
    params: dict[str, Any]
    raw: dict[str, Any]


PluginInputHandler = Callable[[PluginInputRequest], str]


@dataclass(frozen=True)
class PluginCallRequest:
    source_plugin_name: str
    source_mode: str
    request_id: str
    plugin_name: str
    params: dict[str, Any]
    metadata: dict[str, Any]
    raw: dict[str, Any]


PluginCallHandler = Callable[[PluginCallRequest], dict[str, Any]]


@dataclass
class PluginExecutionResult:
    plugin_name: str
    mode: str
    kind: str
    ok: bool
    result: dict[str, Any] = field(default_factory=dict)
    decision: str | None = None
    patch: dict[str, Any] = field(default_factory=dict)
    response: str | None = None
    message: str | None = None
    reason: str | None = None
    raw: dict[str, Any] = field(default_factory=dict)
    stderr: str = ""
    duration_seconds: float = 0.0
    input_requests: list[dict[str, Any]] = field(default_factory=list)
    plugin_call_requests: list[dict[str, Any]] = field(default_factory=list)

    @property
    def is_run_control(self) -> bool:
        return self.kind == "run_control"

    def to_trace_dict(self) -> dict[str, Any]:
        return {
            "plugin_name": self.plugin_name,
            "mode": self.mode,
            "kind": self.kind,
            "ok": self.ok,
            "result": self.result,
            "decision": self.decision,
            "patch": self.patch,
            "response": self.response,
            "message": self.message,
            "reason": self.reason,
            "raw": self.raw,
            "stderr": self.stderr,
            "duration_seconds": self.duration_seconds,
            "input_requests": self.input_requests,
            "plugin_call_requests": self.plugin_call_requests,
        }

    @classmethod
    def transport_error(
        cls,
        *,
        plugin_name: str,
        mode: str,
        message: str,
        stderr: str = "",
        raw: dict[str, Any] | None = None,
        input_requests: list[dict[str, Any]] | None = None,
        plugin_call_requests: list[dict[str, Any]] | None = None,
        duration_seconds: float = 0.0,
    ) -> "PluginExecutionResult":
        return cls(
            plugin_name=plugin_name,
            mode=mode,
            kind="error",
            ok=False,
            message=message,
            stderr=stderr,
            raw=raw or {},
            input_requests=input_requests or [],
            plugin_call_requests=plugin_call_requests or [],
            duration_seconds=duration_seconds,
        )

    @classmethod
    def from_transport(
        cls,
        *,
        plugin_name: str,
        mode: str,
        payload: dict[str, Any],
        stderr: str,
        duration_seconds: float,
        input_requests: list[dict[str, Any]] | None = None,
        plugin_call_requests: list[dict[str, Any]] | None = None,
    ) -> "PluginExecutionResult":
        if not payload.get("ok", False):
            return cls.transport_error(
                plugin_name=plugin_name,
                mode=mode,
                message=str(payload.get("message") or "Plugin execution failed."),
                stderr=stderr,
                raw=payload,
                input_requests=input_requests,
                plugin_call_requests=plugin_call_requests,
                duration_seconds=duration_seconds,
            )

        kind = str(payload.get("kind") or "result")
        if kind == "run_control":
            return cls(
                plugin_name=plugin_name,
                mode=mode,
                kind=kind,
                ok=True,
                decision=str(payload.get("decision")),
                patch=dict(payload.get("patch") or {}),
                response=payload.get("response"),
                message=payload.get("message"),
                reason=payload.get("reason"),
                raw=payload,
                stderr=stderr,
                duration_seconds=duration_seconds,
                input_requests=input_requests or [],
                plugin_call_requests=plugin_call_requests or [],
            )

        result = payload.get("result") or {}
        if not isinstance(result, dict):
            result = {"value": result}
        return cls(
            plugin_name=plugin_name,
            mode=mode,
            kind="result",
            ok=True,
            result=result,
            patch=dict(payload.get("patch") or {}),
            raw=payload,
            stderr=stderr,
            duration_seconds=duration_seconds,
            input_requests=input_requests or [],
            plugin_call_requests=plugin_call_requests or [],
        )
