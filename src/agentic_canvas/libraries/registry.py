from __future__ import annotations

from pathlib import Path

from agentic_canvas.kernel.manifest_registry import load_manifest_directory, registry_storage
from agentic_canvas.kernel.storage import WorkspaceStorage
from agentic_canvas.libraries.manifest import LibraryManifest, LibraryManifestError


class LibraryRegistry:
    """Loads library manifests from a workspace."""

    def __init__(self, storage: WorkspaceStorage | str | Path, libraries_dir: str = "libs") -> None:
        self.storage, self.libraries_dir = registry_storage(storage, libraries_dir)

    def load_all(self) -> dict[str, LibraryManifest]:
        return load_manifest_directory(
            storage=self.storage,
            directory=self.libraries_dir,
            kind="library",
            manifest_factory=lambda data, path: LibraryManifest.from_dict(data, path=path),
            manifest_name=lambda manifest: manifest.name,
            error_type=LibraryManifestError,
        )

    def get(self, name: str) -> LibraryManifest:
        manifests = self.load_all()
        if name not in manifests:
            available = ", ".join(sorted(manifests)) or "none"
            raise KeyError(f"Library {name!r} is not installed. Available: {available}.")
        return manifests[name]

    def resolve(self, names: list[str]) -> list[LibraryManifest]:
        """Resolve a library dependency graph in dependency-first order."""

        manifests = self.load_all()
        resolved: list[LibraryManifest] = []
        visited: set[str] = set()
        visiting: list[str] = []

        def visit(name: str) -> None:
            if name in visited:
                return
            if name in visiting:
                start = visiting.index(name)
                cycle = [*visiting[start:], name]
                raise LibraryManifestError(
                    f"Library dependency cycle detected: {' -> '.join(cycle)}"
                )
            manifest = manifests.get(name)
            if manifest is None:
                parent = visiting[-1] if visiting else None
                if parent:
                    raise LibraryManifestError(
                        f"Library {parent!r} requires missing library {name!r}."
                    )
                raise LibraryManifestError(f"Required library {name!r} is not installed.")

            visiting.append(name)
            for dependency in manifest.libraries:
                visit(dependency)
            visiting.pop()
            visited.add(name)
            resolved.append(manifest)

        for requested in names:
            visit(requested)
        return resolved

    def requirements_for(self, names: list[str]) -> list[str]:
        """Return deterministic external requirements for a library graph."""

        requirements = {
            requirement.strip()
            for manifest in self.resolve(names)
            for requirement in manifest.requirements
            if requirement.strip()
        }
        return sorted(requirements, key=str.casefold)
