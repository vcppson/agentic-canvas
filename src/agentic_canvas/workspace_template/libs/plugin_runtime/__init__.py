from __future__ import annotations

import json
import re
import shutil
from pathlib import Path
from typing import Any, Iterable


_SAFE_NAME = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]*$")


def workspace_root(request: dict[str, Any]) -> Path:
    """Return the workspace root path from a plugin execution request."""
    return Path(request["workspace_root"]).resolve()


def params(request: dict[str, Any]) -> dict[str, Any]:
    """Return request parameters as a dictionary."""
    value = request.get("params") or {}
    return value if isinstance(value, dict) else {"value": value}


def result(
    request: dict[str, Any],
    data: dict[str, Any],
    *,
    decision: str = "continue",
    patch: dict[str, Any] | None = None,
    response: str | None = None,
) -> dict[str, Any]:
    """Shape plugin output for either stage execution or direct call mode."""
    if request.get("mode") == "stage":
        payload: dict[str, Any] = {
            "decision": decision,
            "patch": patch or {},
            "response": response,
            "data": data,
        }
        return payload
    return data


def ensure_safe_name(name: str, label: str = "name") -> str:
    """Validate a plugin or library identifier and return it unchanged."""
    if not _SAFE_NAME.fullmatch(name):
        raise ValueError(
            f"Invalid {label} {name!r}. Use letters, numbers, and underscores; do not start with a number."
        )
    return name


class PluginWorkspaceStorage:
    """Workspace-relative file helper for plugins that create or inspect files."""

    def __init__(self, root: Path) -> None:
        self.root = root.resolve()

    def resolve_path(self, relative_path: str | Path) -> Path:
        relative_path = Path(relative_path)
        if relative_path.is_absolute():
            raise ValueError(f"Workspace paths must be relative: {relative_path!r}")
        target = (self.root / relative_path).resolve()
        if target != self.root and self.root not in target.parents:
            raise ValueError(f"Path escapes workspace root: {relative_path!r}")
        return target

    def exists(self, relative_path: str | Path) -> bool:
        return self.resolve_path(relative_path).exists()

    def is_file(self, relative_path: str | Path) -> bool:
        return self.resolve_path(relative_path).is_file()

    def is_dir(self, relative_path: str | Path) -> bool:
        return self.resolve_path(relative_path).is_dir()

    def read_json(self, relative_path: str | Path) -> dict[str, Any]:
        return json.loads(self.resolve_path(relative_path).read_text(encoding="utf-8"))

    def write_json(self, relative_path: str | Path, data: dict[str, Any]) -> None:
        target = self.resolve_path(relative_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    def write_text(self, relative_path: str | Path, content: str) -> None:
        target = self.resolve_path(relative_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")

    def write_text_under(
        self,
        base_relative_path: str | Path,
        child_relative_path: str | Path,
        content: str,
    ) -> None:
        base = self.resolve_path(base_relative_path)
        child = Path(child_relative_path)
        if child.is_absolute():
            raise ValueError(f"Child path must be relative: {child_relative_path!r}")
        target = (base / child).resolve()
        if target != base and base not in target.parents:
            raise ValueError(f"Child path escapes base directory: {child_relative_path!r}")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")

    def ensure_dir(self, relative_path: str | Path) -> None:
        self.resolve_path(relative_path).mkdir(parents=True, exist_ok=True)

    def replace_tree(self, relative_path: str | Path) -> None:
        target = self.resolve_path(relative_path)
        if target == self.root:
            raise ValueError("Refusing to replace the workspace root.")
        if target.exists():
            if not target.is_dir():
                raise ValueError(f"Workspace path exists but is not a directory: {relative_path}")
            shutil.rmtree(target)
        target.mkdir(parents=True, exist_ok=True)

    def delete_tree(self, relative_path: str | Path) -> None:
        target = self.resolve_path(relative_path)
        if target == self.root:
            raise ValueError("Refusing to delete the workspace root.")
        if target.exists():
            if not target.is_dir():
                raise ValueError(f"Workspace path exists but is not a directory: {relative_path}")
            shutil.rmtree(target)

    def list_dir(self, relative_path: str | Path) -> list[dict[str, Any]]:
        directory = self.resolve_path(relative_path)
        if not directory.is_dir():
            raise FileNotFoundError(f"Workspace directory not found: {relative_path}")
        return [
            {
                "name": item.name,
                "path": str(item.relative_to(self.root)),
                "is_dir": item.is_dir(),
                "is_file": item.is_file(),
            }
            for item in sorted(directory.iterdir())
        ]


def workspace_storage(request: dict[str, Any]) -> PluginWorkspaceStorage:
    """Return a workspace storage helper for the current plugin request."""
    return PluginWorkspaceStorage(workspace_root(request))


def _iter_manifests(root: Path) -> Iterable[dict[str, Any]]:
    storage = PluginWorkspaceStorage(root)
    if not storage.is_dir("."):
        return []
    manifests = []
    for item in storage.list_dir("."):
        manifest_path = f"{item['path']}/manifest.json"
        if item["is_dir"] and storage.is_file(manifest_path):
            manifest = storage.read_json(manifest_path)
            manifest["path"] = str(storage.resolve_path(item["path"]))
            manifests.append(manifest)
    return manifests


def iter_plugin_manifests(root: Path) -> Iterable[dict[str, Any]]:
    """Yield plugin manifest dictionaries from a workspace root."""
    return _iter_manifests(root / "plugins")


def load_plugin_manifest(root: Path, name: str) -> dict[str, Any]:
    """Load a named plugin manifest from a workspace root."""
    ensure_safe_name(name, "plugin name")
    storage = PluginWorkspaceStorage(root)
    path = f"plugins/{name}/manifest.json"
    if not storage.is_file(path):
        raise FileNotFoundError(f"Plugin {name!r} does not exist.")
    manifest = storage.read_json(path)
    manifest["path"] = str(storage.resolve_path(f"plugins/{name}"))
    return manifest


def iter_library_manifests(root: Path) -> Iterable[dict[str, Any]]:
    """Yield library manifest dictionaries from a workspace root."""
    return _iter_manifests(root / "libs")


def load_library_manifest(root: Path, name: str) -> dict[str, Any]:
    """Load a named library manifest from a workspace root."""
    ensure_safe_name(name, "library name")
    storage = PluginWorkspaceStorage(root)
    path = f"libs/{name}/manifest.json"
    if not storage.is_file(path):
        raise FileNotFoundError(f"Library {name!r} does not exist.")
    manifest = storage.read_json(path)
    manifest["path"] = str(storage.resolve_path(f"libs/{name}"))
    return manifest
