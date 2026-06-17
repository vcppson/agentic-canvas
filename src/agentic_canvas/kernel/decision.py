from __future__ import annotations

from enum import StrEnum


class Decision(StrEnum):
    CONTINUE = "continue"
    STOP = "stop"
    AWAIT_USER = "await_user"
    ABORT = "abort"


class RunStatus(StrEnum):
    RUNNING = "running"
    AWAITING_USER_INPUT = "awaiting_user_input"
    COMPLETED = "completed"
    ABORTED = "aborted"


def parse_decision(value: str | Decision) -> Decision:
    try:
        return value if isinstance(value, Decision) else Decision(str(value))
    except ValueError as exc:
        allowed = ", ".join(d.value for d in Decision)
        raise ValueError(f"Unknown decision {value!r}. Expected one of: {allowed}.") from exc
