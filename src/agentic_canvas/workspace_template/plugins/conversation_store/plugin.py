from __future__ import annotations

from libs.conversation_memory import store_run_conversation
from libs.plugin_runtime import result


def plugin_main(request: dict) -> dict:
    stored = store_run_conversation(request)
    return result(request, stored)
