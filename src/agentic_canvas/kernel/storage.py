from __future__ import annotations

import json
import shutil
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable


class WorkspaceStorageError(ValueError):
    pass


@dataclass(frozen=True)
class WorkspaceEntry:
    name: str
    path: str
    is_dir: bool
    is_file: bool


class WorkspaceStorage(ABC):
    """Abstract workspace persistence and file operation boundary."""

    @abstractmethod
    def resolve_path(self, path: str | Path) -> Path:
        ...

    @abstractmethod
    def exists(self, path: str | Path) -> bool:
        ...

    @abstractmethod
    def is_file(self, path: str | Path) -> bool:
        ...

    @abstractmethod
    def is_dir(self, path: str | Path) -> bool:
        ...

    @abstractmethod
    def read_text(self, path: str | Path) -> str:
        ...

    @abstractmethod
    def write_text(self, path: str | Path, text: str) -> None:
        ...

    @abstractmethod
    def read_json(self, path: str | Path) -> dict[str, Any]:
        ...

    @abstractmethod
    def write_json(self, path: str | Path, data: dict[str, Any]) -> None:
        ...

    @abstractmethod
    def append_jsonl(self, path: str | Path, data: dict[str, Any]) -> None:
        ...

    @abstractmethod
    def read_jsonl(self, path: str | Path) -> list[dict[str, Any]]:
        ...

    @abstractmethod
    def list_dir(self, path: str | Path) -> list[WorkspaceEntry]:
        ...

    @abstractmethod
    def ensure_dir(self, path: str | Path) -> None:
        ...

    @abstractmethod
    def copy_tree(
        self,
        source: str | Path,
        target: str | Path,
        *,
        ignore: Callable[[str, list[str]], set[str]] | None = None,
    ) -> None:
        ...

    @abstractmethod
    def replace_tree(self, path: str | Path) -> None:
        ...

    @abstractmethod
    def delete_tree(self, path: str | Path) -> None:
        ...


class FileSystemWorkspaceStorage(WorkspaceStorage):
    """Filesystem-backed workspace storage with strict workspace scoping."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).resolve()

    def resolve_path(self, path: str | Path) -> Path:
        relative = Path(path)
        if relative.is_absolute():
            raise WorkspaceStorageError(f"Workspace paths must be relative: {path!r}")
        target = (self.root / relative).resolve()
        if target != self.root and self.root not in target.parents:
            raise WorkspaceStorageError(f"Workspace path escapes root: {path!r}")
        return target

    def exists(self, path: str | Path) -> bool:
        return self.resolve_path(path).exists()

    def is_file(self, path: str | Path) -> bool:
        return self.resolve_path(path).is_file()

    def is_dir(self, path: str | Path) -> bool:
        return self.resolve_path(path).is_dir()

    def read_text(self, path: str | Path) -> str:
        return self.resolve_path(path).read_text(encoding="utf-8")

    def write_text(self, path: str | Path, text: str) -> None:
        target = self.resolve_path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")

    def read_json(self, path: str | Path) -> dict[str, Any]:
        return json.loads(self.read_text(path))

    def write_json(self, path: str | Path, data: dict[str, Any]) -> None:
        self.write_text(path, json.dumps(data, indent=2, ensure_ascii=False) + "\n")

    def append_jsonl(self, path: str | Path, data: dict[str, Any]) -> None:
        target = self.resolve_path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(data, ensure_ascii=False, default=str) + "\n")

    def read_jsonl(self, path: str | Path) -> list[dict[str, Any]]:
        if not self.is_file(path):
            return []
        return [
            json.loads(line)
            for line in self.read_text(path).splitlines()
            if line.strip()
        ]

    def list_dir(self, path: str | Path) -> list[WorkspaceEntry]:
        directory = self.resolve_path(path)
        if not directory.is_dir():
            raise FileNotFoundError(f"Workspace directory not found: {path}")
        entries = []
        for item in sorted(directory.iterdir()):
            entries.append(
                WorkspaceEntry(
                    name=item.name,
                    path=str(item.relative_to(self.root)),
                    is_dir=item.is_dir(),
                    is_file=item.is_file(),
                )
            )
        return entries

    def ensure_dir(self, path: str | Path) -> None:
        self.resolve_path(path).mkdir(parents=True, exist_ok=True)

    def copy_tree(
        self,
        source: str | Path,
        target: str | Path,
        *,
        ignore: Callable[[str, list[str]], set[str]] | None = None,
    ) -> None:
        source_path = Path(source).resolve()
        target_path = self.resolve_path(target)
        shutil.copytree(source_path, target_path, dirs_exist_ok=True, ignore=ignore)

    def replace_tree(self, path: str | Path) -> None:
        target = self.resolve_path(path)
        if target == self.root:
            raise WorkspaceStorageError("Refusing to replace the workspace root tree.")
        if target.exists():
            if not target.is_dir():
                raise WorkspaceStorageError(f"Workspace path is not a directory: {path}")
            shutil.rmtree(target)
        target.mkdir(parents=True, exist_ok=True)

    def delete_tree(self, path: str | Path) -> None:
        target = self.resolve_path(path)
        if target == self.root:
            raise WorkspaceStorageError("Refusing to delete the workspace root tree.")
        if target.exists():
            if not target.is_dir():
                raise WorkspaceStorageError(f"Workspace path is not a directory: {path}")
            shutil.rmtree(target)
