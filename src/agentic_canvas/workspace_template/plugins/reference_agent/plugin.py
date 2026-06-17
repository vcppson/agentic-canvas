from __future__ import annotations

from libs.agent_runtime import Agent, tool
from libs.llm_google import GoogleProvider
from libs.plugin_calls import plugin_tool
from libs.plugin_runtime import params, result


@tool(description="Count words and characters in supplied text.")
def text_metrics(text: str) -> dict[str, int]:
    """Return simple deterministic text metrics."""

    return {"words": len(text.split()), "characters": len(text)}


def plugin_main(request: dict) -> dict:
    values = params(request)
    prompt = str(values.get("prompt") or "")
    if not prompt:
        raise ValueError("prompt is required.")

    provider = GoogleProvider(model=str(values.get("model") or "") or None)
    inspect_workspace_item = plugin_tool(
        request,
        "plugin_info",
        name="inspect_workspace_item",
        description="Inspect the manifest of an installed workspace plugin or library.",
        metadata={"source": "reference_agent"},
    )
    agent = Agent(
        provider=provider,
        tools=[text_metrics, inspect_workspace_item],
        system_prompt=(
            "You are a reference Agentic Canvas agent. Use tools only when useful, "
            "and clearly distinguish inspected workspace facts from your reasoning."
        ),
        max_turns=int(values.get("max_turns") or 8),
    )
    agent_result = agent.run(prompt)
    return result(
        request,
        {
            "response": agent_result.response,
            "agent": agent_result.to_dict(),
        },
    )
