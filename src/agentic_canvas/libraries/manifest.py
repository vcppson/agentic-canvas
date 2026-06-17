from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


REQUIRED_LIBRARY_FIELDS = {
    "name",
    "version",
    "summary",
    "documentation",
    "exports",
    "requirements",
    "libraries",
    "compatibility",
}


class LibraryManifestError(ValueError):
    pass


@dataclass(frozen=True)
class LibraryManifest:
    name: str
    version: str
    summary: str
    documentation: str
    exports: list[str]
    requirements: list[str] = field(default_factory=list)
    libraries: list[str] = field(default_factory=list)
    compatibility: dict[str, Any] = field(default_factory=dict)
    path: str = ""

    @classmethod
    def from_dict(cls, data: dict[str, Any], *, path: Path | None = None) -> "LibraryManifest":
        missing = sorted(REQUIRED_LIBRARY_FIELDS - set(data))
        if missing:
            raise LibraryManifestError(
                f"Library manifest is missing required field(s): {missing}"
            )
        if not isinstance(data["exports"], list):
            raise LibraryManifestError("Library manifest exports must be a list.")
        if not isinstance(data["requirements"], list):
            raise LibraryManifestError("Library manifest requirements must be a list.")
        if not isinstance(data["libraries"], list):
            raise LibraryManifestError("Library manifest libraries must be a list.")

        return cls(
            name=str(data["name"]),
            version=str(data["version"]),
            summary=str(data["summary"]),
            documentation=str(data["documentation"]),
            exports=[str(item) for item in data.get("exports", [])],
            requirements=[str(item) for item in data.get("requirements", [])],
            libraries=[str(item) for item in data.get("libraries", [])],
            compatibility=dict(data.get("compatibility", {})),
            path=str(path.parent if path else data.get("path", "")),
        )

    @classmethod
    def load(cls, path: Path) -> "LibraryManifest":
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise LibraryManifestError(f"Invalid library manifest JSON at {path}: {exc}") from exc
        return cls.from_dict(data, path=path)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "summary": self.summary,
            "documentation": self.documentation,
            "exports": self.exports,
            "requirements": self.requirements,
            "libraries": self.libraries,
            "compatibility": self.compatibility,
            "path": self.path,
        }
