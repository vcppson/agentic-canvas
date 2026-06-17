from __future__ import annotations

from typing import Any

from libs.manifest_docs import generate_library_documentation
from libs.plugin_runtime import (
    ensure_safe_name,
    load_library_manifest,
    params,
    result,
    workspace_storage,
    workspace_root,
)


DEFAULT_LIBRARY = '''from __future__ import annotations


def describe() -> str:
    """Describe the reusable library."""
    return "A reusable Agentic Canvas library."
'''


def plugin_main(request: dict) -> dict:
    root = workspace_root(request)
    storage = workspace_storage(request)
    values = params(request)
    name = ensure_safe_name(str(values.get("name") or ""), "library name")
    summary = str(values.get("summary") or "")
    files = values.get("files") or {"__init__.py": DEFAULT_LIBRARY}
    if not isinstance(files, dict):
        raise TypeError("files must be an object mapping relative paths to text.")
    if "__init__.py" not in files:
        files["__init__.py"] = DEFAULT_LIBRARY

    requirements = [str(item) for item in values.get("requirements", [])]
    libraries = [str(item) for item in values.get("libraries", [])]
    exports = [str(item) for item in values.get("exports", [])]
    overwrite = bool(values.get("overwrite", False))

    for library_name in libraries:
        load_library_manifest(root, library_name)

    library_dir = f"libs/{name}"
    if storage.exists(library_dir):
        if not overwrite:
            raise FileExistsError(f"Library {name!r} already exists.")
        if not storage.is_dir(library_dir):
            raise RuntimeError(f"Library path exists but is not a directory: {library_dir}")
        storage.replace_tree(library_dir)
    else:
        storage.ensure_dir(library_dir)

    for relative_path, content in files.items():
        storage.write_text_under(library_dir, str(relative_path), str(content))

    documentation = generate_library_documentation(
        storage.resolve_path(library_dir),
        name,
        summary,
        exports,
    )
    manifest: dict[str, Any] = {
        "name": name,
        "version": str(values.get("version") or "0.1.0"),
        "summary": summary,
        "documentation": documentation,
        "exports": exports,
        "requirements": requirements,
        "libraries": libraries,
        "compatibility": values.get("compatibility") or {"agentic_canvas": ">=0.1.0"},
    }
    storage.write_json(f"{library_dir}/manifest.json", manifest)

    return result(
        request,
        {
            "created": True,
            "library": name,
            "path": str(storage.resolve_path(library_dir)),
            "manifest": manifest,
        },
    )
