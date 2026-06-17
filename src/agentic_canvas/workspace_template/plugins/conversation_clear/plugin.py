from __future__ import annotations

from libs.conversation_memory import clear_conversations
from libs.plugin_runtime import params, result


def plugin_main(request: dict) -> dict:
    values = params(request)
    cleared = clear_conversations(
        request,
        reason=str(values.get("reason") or "conversation clear command"),
    )
    response = (
        "Conversation memory cleared.\n"
        f"- Cleared indexed conversations: {cleared['cleared_count']}\n"
        "- Stored run files were preserved for audit."
    )
    cleared["response"] = response
    return result(request, cleared)
