from __future__ import annotations

from libs.command_packs import remove_command_pack
from libs.plugin_runtime import params, result


def plugin_main(request: dict) -> dict:
    values = params(request)
    name = str(values.get("name") or "")
    return result(request, remove_command_pack(request, name))
