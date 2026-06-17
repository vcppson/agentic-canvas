from __future__ import annotations

import json
import sys
import uuid
from typing import Any, NoReturn


_PATCH: dict[str, Any] = {}


def _deep_merge(target: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any]:
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _deep_merge(target[key], value)
        else:
            target[key] = value
    return target


class RunControlSignal(Exception):
    """Structured run-control exception consumed by the plugin bootloader."""

    run_control_signal = True

    def __init__(
        self,
        decision: str,
        *,
        message: str | None = None,
        reason: str | None = None,
        response: str | None = None,
        patch: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message or reason or response or decision)
        self.decision = decision
        self.message = message
        self.reason = reason
        self.response = response
        self.patch = patch or {}


def patch_context(patch: dict[str, Any]) -> None:
    """Accumulate a patch that the kernel will merge into Run Context."""
    if not isinstance(patch, dict):
        raise TypeError("patch_context expects a dict.")
    _deep_merge(_PATCH, patch)


def set_input(value: str) -> None:
    """Request that the kernel replace Run Context input with a new value."""
    patch_context({"input": value})


def consume_patch() -> dict[str, Any]:
    """Return and clear the accumulated Run Context patch."""
    global _PATCH
    patch = _PATCH
    _PATCH = {}
    return patch


def _signal(
    decision: str,
    *,
    message: str | None = None,
    reason: str | None = None,
    response: str | None = None,
    patch: dict[str, Any] | None = None,
) -> NoReturn:
    merged = consume_patch()
    if patch:
        _deep_merge(merged, patch)
    raise RunControlSignal(
        decision,
        message=message,
        reason=reason,
        response=response,
        patch=merged,
    )


def await_user(message: str, *, patch: dict[str, Any] | None = None) -> str:
    """Ask the kernel for user input and return the answer to this plugin call."""
    merged = consume_patch()
    if patch:
        _deep_merge(merged, patch)
    request_id = uuid.uuid4().hex
    _write_protocol(
        {
            "type": "input_request",
            "request_id": request_id,
            "message": message,
            "patch": merged,
        }
    )
    response = _read_protocol()
    if response.get("type") != "input_response" or response.get("request_id") != request_id:
        raise RuntimeError("Invalid input response from plugin runner.")
    return str(response.get("value", ""))


def stop(
    reason: str = "",
    *,
    response: str | None = None,
    patch: dict[str, Any] | None = None,
) -> NoReturn:
    """Stop the current run successfully and skip remaining execution."""
    _signal("stop", reason=reason, response=response, patch=patch)


def abort(reason: str, *, patch: dict[str, Any] | None = None) -> NoReturn:
    """Abort the current run with a reason."""
    _signal("abort", reason=reason, patch=patch)


def _write_protocol(payload: dict[str, Any]) -> None:
    sys.__stdout__.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")
    sys.__stdout__.flush()


def _read_protocol() -> dict[str, Any]:
    line = sys.__stdin__.readline()
    if not line:
        raise RuntimeError("Plugin runner closed before user input was provided.")
    payload = json.loads(line)
    if not isinstance(payload, dict):
        raise RuntimeError("Plugin runner sent a non-object input response.")
    return payload
