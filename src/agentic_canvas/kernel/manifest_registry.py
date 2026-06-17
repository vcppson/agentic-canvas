from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, TypeVar

from agentic_canvas.kernel.storage import FileSystemWorkspaceStorage, WorkspaceStorage


ManifestT = TypeVar("ManifestT")
ManifestFactory = Callable[[dict[str, Any], Path], ManifestT]
NameGetter = Callable[[ManifestT], str]


def registry_storage(
    storage: WorkspaceStorage | str | Path,
    default_directory: str,
) -> tuple[WorkspaceStorage, str]:
    if isinstance(storage, WorkspaceStorage):
        return storage, default_directory

    path = Path(storage)
    return FileSystemWorkspaceStorage(path.parent), path.name


def load_manifest_directory(
    *,
    storage: WorkspaceStorage,
    directory: str,
    kind: str,
    manifest_factory: ManifestFactory[ManifestT],
    manifest_name: NameGetter[ManifestT],
    error_type: type[Exception],
) -> dict[str, ManifestT]:
    if not storage.is_dir(directory):
        raise error_type(f"{kind.capitalize()}s directory not found: {directory}")

    manifests: dict[str, ManifestT] = {}
    for item in storage.list_dir(directory):
        if not item.is_dir or item.name.startswith(".") or item.name == "__pycache__":
            continue
        manifest_path = f"{item.path}/manifest.json"
        if not storage.is_file(manifest_path):
            raise error_type(f"{kind.capitalize()} directory {item.path} has no manifest.json.")
        manifest = manifest_factory(
            storage.read_json(manifest_path),
            storage.resolve_path(manifest_path),
        )
        name = manifest_name(manifest)
        if name != item.name:
            raise error_type(
                f"{kind.capitalize()} manifest name {name!r} does not match directory {item.name!r}."
            )
        if name in manifests:
            raise error_type(f"Duplicate {kind} manifest name: {name}")
        manifests[name] = manifest
    return manifests
