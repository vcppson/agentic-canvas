from __future__ import annotations

from libs.plugin_runtime import iter_plugin_manifests, params, result, workspace_root


def plugin_main(request: dict) -> dict:
    root = workspace_root(request)
    values = params(request)
    query = str(values.get("query") or "").lower()
    include_schemas = bool(values.get("include_schemas", False))

    plugins = []
    for manifest in iter_plugin_manifests(root):
        searchable = " ".join(
            [
                str(manifest.get("name", "")),
                str(manifest.get("summary", "")),
                str(manifest.get("documentation", "")),
                " ".join(manifest.get("tags", [])),
            ]
        ).lower()
        if query and query not in searchable:
            continue

        item = {
            "name": manifest.get("name"),
            "version": manifest.get("version"),
            "summary": manifest.get("summary"),
            "tags": manifest.get("tags", []),
        }
        if include_schemas:
            item["input_schema"] = manifest.get("input_schema", {})
            item["output_schema"] = manifest.get("output_schema", {})
        plugins.append(item)

    return result(request, {"plugins": plugins, "count": len(plugins)})
