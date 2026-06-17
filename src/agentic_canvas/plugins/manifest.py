from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


REQUIRED_PLUGIN_FIELDS = {
    "name",
    "version",
    "summary",
    "documentation",
    "entry_point",
    "input_schema",
    "output_schema",
    "requirements",
    "libraries",
    "compatibility",
}


class ManifestError(ValueError):
    pass


@dataclass(frozen=True)
class PluginManifest:
    name: str
    version: str
    summary: str
    documentation: str
    entry_point: str
    input_schema: dict[str, Any]
    output_schema: dict[str, Any]
    requirements: list[str] = field(default_factory=list)
    libraries: list[str] = field(default_factory=list)
    compatibility: dict[str, Any] = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)
    path: str = ""

    @classmethod
    def from_dict(cls, data: dict[str, Any], *, path: Path | None = None) -> "PluginManifest":
        missing = sorted(REQUIRED_PLUGIN_FIELDS - set(data))
        if missing:
            raise ManifestError(f"Plugin manifest is missing required field(s): {missing}")

        if not isinstance(data["input_schema"], dict):
            raise ManifestError("Plugin manifest input_schema must be an object.")
        if not isinstance(data["output_schema"], dict):
            raise ManifestError("Plugin manifest output_schema must be an object.")
        if not isinstance(data["requirements"], list):
            raise ManifestError("Plugin manifest requirements must be a list.")
        if not isinstance(data["libraries"], list):
            raise ManifestError("Plugin manifest libraries must be a list.")

        return cls(
            name=str(data["name"]),
            version=str(data["version"]),
            summary=str(data["summary"]),
            documentation=str(data["documentation"]),
            entry_point=str(data["entry_point"]),
            input_schema=dict(data["input_schema"]),
            output_schema=dict(data["output_schema"]),
            requirements=[str(item) for item in data.get("requirements", [])],
            libraries=[str(item) for item in data.get("libraries", [])],
            compatibility=dict(data.get("compatibility", {})),
            tags=[str(item) for item in data.get("tags", [])],
            path=str(path.parent if path else data.get("path", "")),
        )

    @classmethod
    def load(cls, path: Path) -> "PluginManifest":
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ManifestError(f"Invalid plugin manifest JSON at {path}: {exc}") from exc
        return cls.from_dict(data, path=path)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "summary": self.summary,
            "documentation": self.documentation,
            "entry_point": self.entry_point,
            "input_schema": self.input_schema,
            "output_schema": self.output_schema,
            "requirements": self.requirements,
            "libraries": self.libraries,
            "compatibility": self.compatibility,
            "tags": self.tags,
            "path": self.path,
        }
