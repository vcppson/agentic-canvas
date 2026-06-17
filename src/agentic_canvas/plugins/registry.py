from __future__ import annotations

from pathlib import Path

from agentic_canvas.kernel.manifest_registry import load_manifest_directory, registry_storage
from agentic_canvas.kernel.storage import WorkspaceStorage
from agentic_canvas.plugins.manifest import ManifestError, PluginManifest


class PluginRegistry:
    """Loads plugin manifests from a workspace."""

    def __init__(self, storage: WorkspaceStorage | str | Path, plugins_dir: str = "plugins") -> None:
        self.storage, self.plugins_dir = registry_storage(storage, plugins_dir)

    def load_all(self) -> dict[str, PluginManifest]:
        return load_manifest_directory(
            storage=self.storage,
            directory=self.plugins_dir,
            kind="plugin",
            manifest_factory=lambda data, path: PluginManifest.from_dict(data, path=path),
            manifest_name=lambda manifest: manifest.name,
            error_type=ManifestError,
        )

    def get(self, name: str) -> PluginManifest:
        manifests = self.load_all()
        if name not in manifests:
            available = ", ".join(sorted(manifests)) or "none"
            raise KeyError(f"Plugin {name!r} is not installed. Available: {available}.")
        return manifests[name]
