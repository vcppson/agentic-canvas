from __future__ import annotations

from pathlib import Path
from typing import Any

from agentic_canvas.kernel.run_state import RunContext, utc_now
from agentic_canvas.kernel.storage import FileSystemWorkspaceStorage, WorkspaceStorage


class RunContextStore:
    """File persistence for inspectable and resumable Run Contexts."""

    def __init__(self, storage: WorkspaceStorage | Path, runs_dir: str = "runs") -> None:
        self.storage = (
            storage
            if isinstance(storage, WorkspaceStorage)
            else FileSystemWorkspaceStorage(storage)
        )
        self.runs_dir = runs_dir
        self.storage.ensure_dir(self.runs_dir)

    def path_for(self, run_id: str) -> Path:
        return self.storage.resolve_path(f"{self.runs_dir}/{run_id}.json")  # type: ignore[attr-defined]

    @property
    def current_path(self) -> Path:
        return self.storage.resolve_path(f"{self.runs_dir}/current.json")  # type: ignore[attr-defined]

    def save(self, context: RunContext) -> None:
        context.updated_at = utc_now()
        self.storage.write_json(f"{self.runs_dir}/{context.run_id}.json", context.to_dict())
        self.storage.write_json(f"{self.runs_dir}/current.json", context.to_dict())

    def load(self, run_id: str) -> RunContext:
        path = f"{self.runs_dir}/{run_id}.json"
        if not self.storage.is_file(path):
            raise FileNotFoundError(f"Run {run_id!r} was not found at {self.path_for(run_id)}.")
        return RunContext.from_dict(self.storage.read_json(path))

    def load_current(self) -> RunContext:
        if not self.storage.is_file(f"{self.runs_dir}/current.json"):
            raise FileNotFoundError(f"No current run exists at {self.current_path}.")
        return RunContext.from_dict(self.storage.read_json(f"{self.runs_dir}/current.json"))


class RunTraceStore:
    """Append-only JSONL trace persistence for detailed Run execution records."""

    def __init__(self, storage: WorkspaceStorage | Path, runs_dir: str = "runs") -> None:
        self.storage = (
            storage
            if isinstance(storage, WorkspaceStorage)
            else FileSystemWorkspaceStorage(storage)
        )
        self.runs_dir = runs_dir
        self.storage.ensure_dir(self.runs_dir)
        self._next_sequence_by_run: dict[str, int] = {}

    def path_for(self, run_id: str) -> Path:
        return self.storage.resolve_path(f"{self.runs_dir}/{run_id}.trace.jsonl")  # type: ignore[attr-defined]

    def append(self, context: RunContext, event_type: str, **data: Any) -> dict[str, Any]:
        record = {
            "sequence": self._next_sequence(context.run_id),
            "timestamp": utc_now(),
            "run_id": context.run_id,
            "type": event_type,
            "stage": context.stage,
            "status": context.status,
            "decision": context.decision,
            **data,
        }
        self.storage.append_jsonl(f"{self.runs_dir}/{context.run_id}.trace.jsonl", record)
        return record

    def read(self, run_id: str) -> list[dict[str, Any]]:
        return self.storage.read_jsonl(f"{self.runs_dir}/{run_id}.trace.jsonl")

    def _next_sequence(self, run_id: str) -> int:
        if run_id not in self._next_sequence_by_run:
            count = len(self.read(run_id))
            self._next_sequence_by_run[run_id] = count + 1

        sequence = self._next_sequence_by_run[run_id]
        self._next_sequence_by_run[run_id] = sequence + 1
        return sequence
