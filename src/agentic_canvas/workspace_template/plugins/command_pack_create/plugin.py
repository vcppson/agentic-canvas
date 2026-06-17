from __future__ import annotations

from libs.command_packs import create_command_pack
from libs.plugin_runtime import params, result


def plugin_main(request: dict) -> dict:
    values = params(request)
    manifest = values.get("manifest")
    if not isinstance(manifest, dict):
        raise TypeError("manifest must be an object.")
    return result(
        request,
        create_command_pack(
            request,
            manifest,
            overwrite=bool(values.get("overwrite", False)),
        ),
    )
