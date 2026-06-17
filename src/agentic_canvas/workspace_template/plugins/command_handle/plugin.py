from __future__ import annotations

from libs.plugin_runtime import params, result
from libs.command_packs import call_plugin, match_command, params_for_match


def plugin_main(request: dict) -> dict:
    values = params(request)
    matched = match_command(
        request,
        prefix=str(values.get("prefix") or "/"),
    )
    if not matched.get("is_command"):
        return result(request, {"handled": False})

    match = matched["match"]
    command = match["command"]
    plugin_params = params_for_match(match)
    plugin_result = call_plugin(
        command["plugin"],
        plugin_params,
        metadata={
            "command_pack": match["pack_name"],
            "command": match["command_name"],
            "alias": match["alias"],
            "raw_input": match["raw_input"],
        },
    )
    if not plugin_result.get("ok", False):
        raise RuntimeError(plugin_result.get("message") or "Command target plugin failed.")
    if plugin_result.get("kind") != "result":
        raise RuntimeError(f"Command target plugin returned unsupported kind: {plugin_result.get('kind')}")

    command_output = plugin_result.get("result") or {}
    response = (
        command_output.get("response")
        or command_output.get("message")
        or command_output.get("data", {}).get("response")
    )
    if response is None:
        response = command_output

    return result(
        request,
        {
            "handled": True,
            "command": match,
            "target_plugin": command["plugin"],
            "target_params": plugin_params,
            "target_result": plugin_result,
        },
        decision=str(command.get("decision") or "stop"),
        response=response if isinstance(response, str) else str(response),
    )
