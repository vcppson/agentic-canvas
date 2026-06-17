from __future__ import annotations

from libs.command_packs import list_command_packs
from libs.plugin_runtime import params, result


def plugin_main(request: dict) -> dict:
    values = params(request)
    return result(
        request,
        list_command_packs(request, query=str(values.get("query") or "")),
    )
