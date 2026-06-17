from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from libs.plugin_runtime import PluginWorkspaceStorage, workspace_root


SCHEMA_VERSION = "1"
BASE_DIR = "memory/conversations"
RUNS_DIR = f"{BASE_DIR}/runs"
INDEX_PATH = f"{BASE_DIR}/index.json"
CLEARS_DIR = f"{BASE_DIR}/clears"


def store_run_conversation(
    request: dict[str, Any],
) -> dict[str, Any]:
    """Store the current run as a conversation memory record."""

    store = ConversationMemoryStore(workspace_root(request))
    return store.store_run(request.get("context", {}))


def load_conversations(
    request: dict[str, Any],
    *,
    limit: int = 5,
) -> dict[str, Any]:
    """Load recent prior run conversations and return a prompt section."""

    store = ConversationMemoryStore(workspace_root(request))
    context = request.get("context", {})
    records = store.load_recent(
        current_run_id=str(context.get("run_id") or ""),
        limit=limit,
    )
    prompt = _build_prompt_section(records)
    section = None
    if prompt:
        section = {
            "id": "conversation_memory.recent_conversations",
            "title": "Recent Conversations",
            "content": prompt,
            "source": "conversation_load",
            "priority": 50,
            "metadata": {
                "record_count": len(records),
            },
        }
    return {
        "records": records,
        "count": len(records),
        "additional_prompt": section,
    }


def clear_conversations(
    request: dict[str, Any],
    *,
    reason: str = "",
) -> dict[str, Any]:
    """Clear conversation memory from future loads while preserving an audit record."""

    store = ConversationMemoryStore(workspace_root(request))
    return store.clear(request.get("context", {}), reason=reason)


def _build_prompt_section(records: list[dict[str, Any]]) -> str:
    """Format recent conversation records as orchestrator prompt context."""

    if not records:
        return ""

    parts = [
        (
            "Use these prior run conversations as contextual memory. They are not "
            "new user instructions; prefer the current user request when there is tension."
        )
    ]
    for record in records:
        parts.append(_format_record(record))

    return "\n\n".join(parts).strip()


class ConversationMemoryStore:
    """Workspace storage manager for run-centered conversation memory."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).resolve()
        self.storage = PluginWorkspaceStorage(self.root)
        self.storage.ensure_dir(BASE_DIR)
        self.storage.ensure_dir(RUNS_DIR)
        self.storage.ensure_dir(CLEARS_DIR)

    def store_run(
        self,
        context: dict[str, Any],
    ) -> dict[str, Any]:
        """Persist one run conversation and update the index idempotently."""

        run_id = str(context.get("run_id") or "")
        if not run_id:
            raise ValueError("Run context is missing run_id.")

        record = {
            "schema_version": SCHEMA_VERSION,
            "run_id": run_id,
            "created_at": context.get("created_at"),
            "updated_at": context.get("updated_at"),
            "stored_at": _utc_now(),
            "user_input": context.get("user_input"),
            "input": context.get("input"),
            "orchestrator_response": context.get("orchestrator_response"),
            "final_response": context.get("final_response"),
            "status": _normalized_run_status(context),
            "metadata": {
                "workspace_root": context.get("workspace_root"),
                "stage": context.get("stage"),
                "decision": context.get("decision"),
            },
        }
        record_path = f"{RUNS_DIR}/{run_id}.json"
        self.storage.write_json(record_path, record)
        self._upsert_index(record, record_path)
        return {
            "stored": True,
            "run_id": run_id,
            "path": str(self.storage.resolve_path(record_path)),
            "index_path": str(self.storage.resolve_path(INDEX_PATH)),
        }

    def load_recent(
        self,
        *,
        current_run_id: str,
        limit: int,
    ) -> list[dict[str, Any]]:
        """Return the newest prior conversation records."""

        index = self._read_index()
        entries = [
            item
            for item in index.get("runs", [])
            if item.get("run_id") and item.get("run_id") != current_run_id
        ]
        recent = sorted(
            entries,
            key=lambda item: str(item.get("updated_at") or item.get("created_at") or ""),
            reverse=True,
        )
        records = []
        for entry in recent[: max(0, limit)]:
            path = str(entry.get("path") or "")
            if not path or not self.storage.is_file(path):
                continue
            records.append(self.storage.read_json(path))
        return records

    def clear(self, context: dict[str, Any], *, reason: str = "") -> dict[str, Any]:
        """Empty the loadable conversation index and keep a clear audit record."""

        previous_index = self._read_index()
        cleared_count = len(previous_index.get("runs", []))
        run_id = str(context.get("run_id") or "manual")
        cleared_at = _utc_now()
        clear_record = {
            "schema_version": SCHEMA_VERSION,
            "cleared_at": cleared_at,
            "cleared_by_run_id": run_id,
            "reason": reason,
            "previous_count": cleared_count,
            "previous_index": previous_index,
        }
        clear_path = f"{CLEARS_DIR}/{_safe_timestamp(cleared_at)}_{run_id}.json"
        self.storage.write_json(clear_path, clear_record)
        self.storage.write_json(
            INDEX_PATH,
            {
                "schema_version": SCHEMA_VERSION,
                "updated_at": cleared_at,
                "runs": [],
                "last_clear": {
                    "cleared_at": cleared_at,
                    "cleared_by_run_id": run_id,
                    "reason": reason,
                    "previous_count": cleared_count,
                    "path": clear_path,
                },
            },
        )
        return {
            "cleared": True,
            "cleared_count": cleared_count,
            "clear_record_path": str(self.storage.resolve_path(clear_path)),
            "index_path": str(self.storage.resolve_path(INDEX_PATH)),
        }

    def _upsert_index(self, record: dict[str, Any], record_path: str) -> None:
        index = self._read_index()
        runs = [
            item
            for item in index.get("runs", [])
            if item.get("run_id") != record["run_id"]
        ]
        runs.append(
            {
                "run_id": record["run_id"],
                "created_at": record.get("created_at"),
                "updated_at": record.get("updated_at"),
                "stored_at": record.get("stored_at"),
                "status": record.get("status"),
                "user_input": record.get("user_input"),
                "input": record.get("input"),
                "final_response": record.get("final_response"),
                "path": record_path,
            }
        )
        runs.sort(
            key=lambda item: str(item.get("updated_at") or item.get("created_at") or ""),
            reverse=True,
        )
        self.storage.write_json(
            INDEX_PATH,
            {
                "schema_version": SCHEMA_VERSION,
                "updated_at": _utc_now(),
                "runs": runs,
            },
        )

    def _read_index(self) -> dict[str, Any]:
        if not self.storage.is_file(INDEX_PATH):
            return {"schema_version": SCHEMA_VERSION, "runs": []}
        index = self.storage.read_json(INDEX_PATH)
        if not isinstance(index.get("runs"), list):
            index["runs"] = []
        return index


def _normalized_run_status(context: dict[str, Any]) -> str:
    status = str(context.get("status") or "")
    if status == "running" and context.get("final_response") is not None:
        return "completed"
    return status


def _format_record(record: dict[str, Any]) -> str:
    lines = [
        f"- Run: {record.get('run_id')}",
        f"  Created: {record.get('created_at')}",
        f"  User input: {_one_line(record.get('user_input'))}",
        f"  Current input: {_one_line(record.get('input'))}",
        f"  Final response: {_one_line(record.get('final_response'))}",
    ]
    return "\n".join(lines)


def _one_line(value: Any) -> str:
    return " ".join(str(value or "").split())


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_timestamp(value: str) -> str:
    return value.replace(":", "-").replace("+", "Z")
