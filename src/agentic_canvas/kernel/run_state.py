from __future__ import annotations

import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from agentic_canvas.kernel.decision import Decision, RunStatus
from agentic_canvas.kernel.trigger import Trigger


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def deep_merge(target: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any]:
    for key, value in patch.items():
        if (
            isinstance(value, dict)
            and isinstance(target.get(key), dict)
        ):
            deep_merge(target[key], value)
        else:
            target[key] = value
    return target


@dataclass
class RunContext:
    """The persisted living state of a Run."""

    run_id: str
    workspace_root: str
    status: str
    stage: str | None
    decision: str
    user_input: str
    input: str
    orchestrator_system_prompt: str
    additional_prompts: list[dict[str, Any]] = field(default_factory=list)
    orchestrator_response: str | None = None
    final_response: str | None = None
    awaiting: dict[str, Any] | None = None
    user_inputs: list[dict[str, Any]] = field(default_factory=list)
    user_input_requests: list[dict[str, Any]] = field(default_factory=list)
    state: dict[str, Any] = field(default_factory=dict)
    events: list[dict[str, Any]] = field(default_factory=list)
    created_at: str = field(default_factory=utc_now)
    updated_at: str = field(default_factory=utc_now)

    @classmethod
    def from_trigger(
        cls,
        trigger: Trigger,
        *,
        workspace_root: Path,
        orchestrator_system_prompt: str,
    ) -> "RunContext":
        run_id = uuid.uuid4().hex
        context = cls(
            run_id=run_id,
            workspace_root=str(workspace_root),
            status=RunStatus.RUNNING.value,
            stage=None,
            decision=Decision.CONTINUE.value,
            user_input=trigger.user_input,
            input=trigger.user_input,
            orchestrator_system_prompt=orchestrator_system_prompt,
        )
        context.record_event(
            "trigger_received",
            trigger_type=trigger.type,
            metadata=trigger.metadata,
        )
        return context

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RunContext":
        return cls(
            run_id=data["run_id"],
            workspace_root=data["workspace_root"],
            status=data["status"],
            stage=data.get("stage"),
            decision=data["decision"],
            user_input=data["user_input"],
            input=data.get("input", data["user_input"]),
            orchestrator_system_prompt=data.get("orchestrator_system_prompt", ""),
            additional_prompts=list(data.get("additional_prompts", [])),
            orchestrator_response=data.get("orchestrator_response"),
            final_response=data.get("final_response"),
            awaiting=data.get("awaiting"),
            user_inputs=list(data.get("user_inputs", data.get("resume_inputs", []))),
            user_input_requests=list(data.get("user_input_requests", [])),
            state=dict(data.get("state", {})),
            events=list(data.get("events", [])),
            created_at=data.get("created_at", utc_now()),
            updated_at=data.get("updated_at", utc_now()),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_plugin_context(self) -> dict[str, Any]:
        """Return the context view passed to plugins."""

        return self.to_dict()

    def record_event(self, event_type: str, **data: Any) -> None:
        self.events.append({"type": event_type, "timestamp": utc_now(), **data})
        self.updated_at = utc_now()

    def apply_patch(self, patch: dict[str, Any] | None) -> None:
        if not patch:
            return

        direct_fields = {
            "status",
            "stage",
            "decision",
            "input",
            "orchestrator_system_prompt",
            "orchestrator_response",
            "final_response",
            "awaiting",
        }

        for key, value in patch.items():
            if key == "state":
                if not isinstance(value, dict):
                    raise ValueError("Context patch field 'state' must be an object.")
                deep_merge(self.state, value)
            elif key == "additional_prompts":
                if not isinstance(value, list):
                    raise ValueError("Context patch field 'additional_prompts' must be a list.")
                self.additional_prompts.extend(
                    item for item in value if isinstance(item, dict)
                )
            elif key == "events":
                if isinstance(value, list):
                    self.events.extend(value)
                else:
                    self.events.append(value)
            elif key in direct_fields:
                setattr(self, key, value)
            else:
                self.state[key] = value

        self.updated_at = utc_now()
