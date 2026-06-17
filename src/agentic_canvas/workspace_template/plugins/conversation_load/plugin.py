from __future__ import annotations

from libs.conversation_memory import load_conversations
from libs.plugin_runtime import params, result


def plugin_main(request: dict) -> dict:
    values = params(request)
    loaded = load_conversations(
        request,
        limit=int(values.get("limit") or 5),
    )
    patch = {}
    if loaded.get("additional_prompt"):
        patch = {"additional_prompts": [loaded["additional_prompt"]]}

    return result(
        request,
        {
            "loaded": loaded["count"],
            "additional_prompt": loaded.get("additional_prompt"),
        },
        patch=patch,
    )
