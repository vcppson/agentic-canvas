from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from agentic_canvas.kernel.context import RunContext
from agentic_canvas.kernel.context import RunTraceStore
from agentic_canvas.kernel.plugin_runner import (
    PluginExecutionResult,
    PluginCallHandler,
    PluginInputHandler,
    PluginRunner,
)
from agentic_canvas.kernel.workspace import Workspace
from agentic_canvas.plugins.registry import PluginRegistry
from agentic_canvas.providers.base import LLMProvider, ToolDefinition


class RunControlInterrupt(Exception):
    def __init__(self, result: PluginExecutionResult) -> None:
        super().__init__(result.message or result.reason or result.decision or "run control")
        self.result = result


@dataclass
class OrchestratorResult:
    response: str | None = None
    control: PluginExecutionResult | None = None


class Orchestrator:
    """Reasoning role with exactly one tool: call_plugin."""

    def __init__(
        self,
        *,
        workspace: Workspace,
        plugin_runner: PluginRunner,
        provider: LLMProvider,
        trace_store: RunTraceStore | None = None,
        input_handler: PluginInputHandler | None = None,
        plugin_call_handler: PluginCallHandler | None = None,
        max_turns: int = 20,
    ) -> None:
        self.workspace = workspace
        self.plugin_runner = plugin_runner
        self.provider = provider
        self.trace_store = trace_store
        self.input_handler = input_handler
        self.plugin_call_handler = plugin_call_handler
        self.max_turns = max_turns

    def tool_definitions(self, context: RunContext) -> list[ToolDefinition]:
        def call_plugin(name: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
            call_params = params or {}
            manifest = PluginRegistry(
                self.workspace.storage,
                self.workspace.config.get("plugins_dir", "plugins"),
            ).get(name)
            result = self.plugin_runner.run(
                workspace=self.workspace,
                manifest=manifest,
                context=context,
                mode="call",
                params=call_params,
                input_handler=self.input_handler,
                plugin_call_handler=self.plugin_call_handler,
            )
            context.record_event(
                "plugin_called",
                plugin=name,
                mode="call",
                kind=result.kind,
                ok=result.ok,
            )
            if self.trace_store:
                self.trace_store.append(
                    context,
                    "plugin_execution",
                    plugin=name,
                    mode="call",
                    params=call_params,
                    result=result.to_trace_dict(),
                )

            if result.kind == "error":
                raise RunControlInterrupt(
                    PluginExecutionResult(
                        plugin_name=name,
                        mode="call",
                        kind="run_control",
                        ok=True,
                        decision="abort",
                        reason=result.message,
                        patch={},
                        stderr=result.stderr,
                        raw=result.raw,
                        input_requests=result.input_requests,
                    )
                )

            if result.is_run_control:
                raise RunControlInterrupt(result)

            if result.patch:
                context.apply_patch(result.patch)
            return result.result

        return [
            ToolDefinition(
                name="call_plugin",
                description="Call one installed workspace plugin by name with JSON parameters.",
                input_schema={
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                            "description": "Installed plugin name.",
                        },
                        "params": {
                            "type": "object",
                            "description": "JSON parameters matching the plugin manifest input_schema.",
                            "additionalProperties": True,
                        },
                    },
                    "required": ["name"],
                    "additionalProperties": False,
                },
                handler=call_plugin,
            )
        ]

    def invoke(self, context: RunContext) -> OrchestratorResult:
        messages = [
            {
                "role": "user",
                "content": context.input,
            }
        ]
        try:
            response = self.provider.chat(
                system_prompt=compose_system_prompt(context),
                messages=messages,
                tools=self.tool_definitions(context),
                max_turns=self.max_turns,
            )
        except RunControlInterrupt as exc:
            return OrchestratorResult(control=exc.result)

        return OrchestratorResult(response=response)


def compose_system_prompt(context: RunContext) -> str:
    """Compose the base orchestrator prompt with plugin-contributed sections."""

    sections = [context.orchestrator_system_prompt.rstrip()]
    additional_prompts = sorted(
        enumerate(context.additional_prompts),
        key=lambda item: (int(item[1].get("priority", 100)), item[0]),
    )
    for _, prompt in additional_prompts:
        title = str(prompt.get("title") or prompt.get("id") or "Additional Context")
        content = str(prompt.get("content") or "").strip()
        if not content:
            continue
        sections.append(f"## {title}\n{content}")
    return "\n\n".join(section for section in sections if section)
