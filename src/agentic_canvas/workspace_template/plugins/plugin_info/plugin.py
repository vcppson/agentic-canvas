from __future__ import annotations

from libs.plugin_runtime import (
    load_library_manifest,
    load_plugin_manifest,
    params,
    result,
    workspace_root,
)


def plugin_main(request: dict) -> dict:
    root = workspace_root(request)
    values = params(request)
    name = str(values.get("name") or "")
    kind = str(values.get("kind") or "plugin")

    if kind == "library":
        manifest = load_library_manifest(root, name)
        return result(request, {"kind": "library", "manifest": manifest})

    if kind != "plugin":
        raise ValueError("kind must be 'plugin' or 'library'.")

    manifest = load_plugin_manifest(root, name)
    return result(request, {"kind": "plugin", "manifest": manifest})
