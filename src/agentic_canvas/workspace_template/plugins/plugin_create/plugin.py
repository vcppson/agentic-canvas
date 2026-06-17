from __future__ import annotations

from typing import Any

from libs.manifest_docs import generate_plugin_documentation
from libs.plugin_runtime import (
    ensure_safe_name,
    load_library_manifest,
    params,
    result,
    workspace_storage,
    workspace_root,
)


DEFAULT_PLUGIN = '''from __future__ import annotations


def plugin_main(request: dict) -> dict:
    return {
        "echo": request.get("params", {}),
        "context_run_id": request.get("context", {}).get("run_id"),
    }
'''


def plugin_main(request: dict) -> dict:
    root = workspace_root(request)
    storage = workspace_storage(request)
    values = params(request)
    name = ensure_safe_name(str(values.get("name") or ""), "plugin name")
    summary = str(values.get("summary") or "")
    entry_point = str(values.get("entry_point") or "plugin.py:plugin_main")
    files = values.get("files") or {"plugin.py": DEFAULT_PLUGIN}
    if not isinstance(files, dict):
        raise TypeError("files must be an object mapping relative paths to text.")

    requirements = [str(item) for item in values.get("requirements", [])]
    libraries = [str(item) for item in values.get("libraries", [])]
    tags = [str(item) for item in values.get("tags", [])]
    input_schema = values.get("input_schema") or {"type": "object"}
    output_schema = values.get("output_schema") or {"type": "object"}
    documentation = str(
        values.get("documentation")
        or generate_plugin_documentation(
            name,
            summary,
            entry_point,
            input_schema,
            output_schema,
            libraries,
            requirements,
            tags,
        )
    )
    overwrite = bool(values.get("overwrite", False))

    for library_name in libraries:
        load_library_manifest(root, library_name)

    plugin_dir = f"plugins/{name}"
    if storage.exists(plugin_dir):
        if not overwrite:
            raise FileExistsError(f"Plugin {name!r} already exists.")
        if not storage.is_dir(plugin_dir):
            raise RuntimeError(f"Plugin path exists but is not a directory: {plugin_dir}")
        storage.replace_tree(plugin_dir)
    else:
        storage.ensure_dir(plugin_dir)

    for relative_path, content in files.items():
        storage.write_text_under(plugin_dir, str(relative_path), str(content))

    entry_file = entry_point.split(":", 1)[0]
    if not storage.is_file(f"{plugin_dir}/{entry_file}"):
        raise FileNotFoundError(f"Entry point file {entry_file!r} was not created.")

    manifest: dict[str, Any] = {
        "name": name,
        "version": str(values.get("version") or "0.1.0"),
        "summary": summary,
        "documentation": documentation,
        "entry_point": entry_point,
        "input_schema": input_schema,
        "output_schema": output_schema,
        "requirements": requirements,
        "libraries": libraries,
        "compatibility": values.get("compatibility") or {"agentic_canvas": ">=0.1.0"},
        "tags": tags,
    }
    storage.write_json(f"{plugin_dir}/manifest.json", manifest)

    return result(
        request,
        {
            "created": True,
            "plugin": name,
            "path": str(storage.resolve_path(plugin_dir)),
            "manifest": manifest,
        },
    )
