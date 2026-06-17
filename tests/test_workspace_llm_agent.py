from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from pydantic import BaseModel


ROOT = Path(__file__).resolve().parents[1]
TEMPLATE_ROOT = ROOT / "src" / "agentic_canvas" / "workspace_template"
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(TEMPLATE_ROOT))

from agentic_canvas.kernel.plugin_runner import SubprocessPluginRunner
from agentic_canvas.kernel.run import Kernel
from agentic_canvas.kernel.trigger import Trigger
from agentic_canvas.kernel.workspace import Workspace, init_workspace
from agentic_canvas.libraries.manifest import LibraryManifestError
from agentic_canvas.libraries.registry import LibraryRegistry
from agentic_canvas.plugins.registry import PluginRegistry
from libs.agent_runtime import (
    Agent,
    AgentMaxTurnsError,
    FunctionTool,
    ToolDefinitionError,
    tool,
)
from libs.llm_anthropic import AnthropicProvider
from libs.llm_core import (
    LLMProvider,
    LLMProviderError,
    LLMRequest,
    LLMResponse,
    Message,
    ToolCall,
    ToolDefinition,
    Usage,
)
from libs.llm_google import GoogleProvider
from libs.llm_openai import OpenAIProvider
from tests.support import CallPluginProvider


class ScriptedProvider(LLMProvider):
    def __init__(self, responses: list[LLMResponse]) -> None:
        self.responses = list(responses)
        self.requests: list[LLMRequest] = []

    def complete(self, request: LLMRequest) -> LLMResponse:
        self.requests.append(request)
        return self.responses.pop(0)


class Payload(BaseModel):
    name: str
    count: int


class ProviderAdapterTest(unittest.TestCase):
    def setUp(self) -> None:
        self.request = LLMRequest(
            system_prompt="system",
            messages=[Message(role="user", content="hello")],
            tools=[
                ToolDefinition(
                    name="lookup",
                    description="Look up a value.",
                    input_schema={
                        "type": "object",
                        "properties": {"key": {"type": "string"}},
                        "required": ["key"],
                    },
                )
            ],
            max_output_tokens=128,
        )

    def test_openai_adapter_normalizes_tool_calls_text_and_usage(self) -> None:
        calls: list[dict[str, Any]] = []

        class Responses:
            def create(self, **kwargs: Any) -> Any:
                calls.append(kwargs)
                return SimpleNamespace(
                    output_text="working",
                    output=[
                        SimpleNamespace(
                            type="function_call",
                            call_id="call_1",
                            name="lookup",
                            arguments='{"key": "alpha"}',
                        )
                    ],
                    usage=SimpleNamespace(input_tokens=10, output_tokens=4, total_tokens=14),
                    status="completed",
                )

        provider = OpenAIProvider(
            model="gpt-test",
            client=SimpleNamespace(responses=Responses()),
        )
        response = provider.complete(self.request)

        self.assertEqual(response.text, "working")
        self.assertEqual(response.tool_calls[0].arguments, {"key": "alpha"})
        self.assertEqual(response.usage.total_tokens, 14)
        self.assertEqual(calls[0]["tools"][0]["name"], "lookup")

    def test_openai_adapter_correlates_tool_results_and_wraps_provider_errors(self) -> None:
        captured: list[dict[str, Any]] = []

        class Responses:
            def create(self, **kwargs: Any) -> Any:
                captured.append(kwargs)
                return SimpleNamespace(
                    output_text="done",
                    output=[],
                    usage=SimpleNamespace(input_tokens=1, output_tokens=1, total_tokens=2),
                    status="completed",
                )

        provider = OpenAIProvider(
            model="gpt-test",
            client=SimpleNamespace(responses=Responses()),
        )
        provider.complete(
            LLMRequest(
                messages=[
                    Message(
                        role="assistant",
                        tool_calls=[
                            ToolCall(id="call_9", name="lookup", arguments={"key": "x"})
                        ],
                    ),
                    Message(
                        role="tool",
                        tool_call_id="call_9",
                        name="lookup",
                        content='{"ok": true}',
                    ),
                ]
            )
        )
        self.assertEqual(captured[0]["input"][0]["call_id"], "call_9")
        self.assertEqual(captured[0]["input"][1]["call_id"], "call_9")

        class BrokenResponses:
            def create(self, **kwargs: Any) -> Any:
                raise OSError("offline")

        broken = OpenAIProvider(
            model="gpt-test",
            client=SimpleNamespace(responses=BrokenResponses()),
        )
        with self.assertRaisesRegex(LLMProviderError, "offline"):
            broken.complete(self.request)

    def test_google_adapter_normalizes_multiple_parts_and_usage(self) -> None:
        class Models:
            def generate_content(self, **kwargs: Any) -> Any:
                return SimpleNamespace(
                    candidates=[
                        SimpleNamespace(
                            finish_reason="STOP",
                            content=SimpleNamespace(
                                parts=[
                                    SimpleNamespace(text="checking", function_call=None),
                                    SimpleNamespace(
                                        text=None,
                                        function_call=SimpleNamespace(
                                            id="call_2",
                                            name="lookup",
                                            args={"key": "beta"},
                                        ),
                                    ),
                                ]
                            ),
                        )
                    ],
                    usage_metadata=SimpleNamespace(
                        prompt_token_count=8,
                        candidates_token_count=3,
                        total_token_count=11,
                    ),
                )

        provider = GoogleProvider(
            model="gemini-test",
            client=SimpleNamespace(models=Models()),
        )
        response = provider.complete(self.request)

        self.assertEqual(response.text, "checking")
        self.assertEqual(response.tool_calls[0].name, "lookup")
        self.assertEqual(response.usage, Usage(8, 3, 11))

    def test_anthropic_adapter_normalizes_content_blocks_and_usage(self) -> None:
        class Messages:
            def create(self, **kwargs: Any) -> Any:
                return SimpleNamespace(
                    content=[
                        SimpleNamespace(type="text", text="checking"),
                        SimpleNamespace(
                            type="tool_use",
                            id="call_3",
                            name="lookup",
                            input={"key": "gamma"},
                        ),
                    ],
                    usage=SimpleNamespace(input_tokens=9, output_tokens=5),
                    stop_reason="tool_use",
                )

        provider = AnthropicProvider(
            model="claude-test",
            client=SimpleNamespace(messages=Messages()),
        )
        response = provider.complete(self.request)

        self.assertEqual(response.text, "checking")
        self.assertEqual(response.tool_calls[0].id, "call_3")
        self.assertEqual(response.usage, Usage(9, 5, 14))


class FunctionToolTest(unittest.TestCase):
    def test_functions_methods_callable_objects_and_structured_models(self) -> None:
        def process(payload: Payload, tags: list[str] | None = None) -> dict[str, Any]:
            """Process a structured payload."""

            return {"name": payload.name, "count": payload.count, "tags": tags or []}

        class Prefixer:
            def add(self, value: str, prefix: str = "x") -> str:
                return prefix + value

        class Multiplier:
            def __call__(self, value: int, factor: int = 2) -> int:
                return value * factor

        structured = FunctionTool.from_callable(process)
        method = FunctionTool.from_callable(Prefixer().add)
        callable_object = FunctionTool.from_callable(Multiplier())

        self.assertIn("$defs", structured.input_schema)
        self.assertEqual(
            structured.invoke(
                {"payload": {"name": "item", "count": 2}, "tags": ["a"]}
            ),
            {"name": "item", "count": 2, "tags": ["a"]},
        )
        self.assertEqual(method.invoke({"value": "1"}), "x1")
        self.assertEqual(callable_object.invoke({"value": 3}), 6)

    def test_decorator_overrides_tool_metadata(self) -> None:
        @tool(name="sum_values", description="Add two values.")
        def add(left: int, right: int = 1) -> int:
            return left + right

        wrapped = FunctionTool.from_callable(add)
        self.assertEqual(wrapped.name, "sum_values")
        self.assertEqual(wrapped.description, "Add two values.")
        self.assertEqual(wrapped.invoke({"left": 4}), 5)

    def test_rejects_duplicate_variadic_and_async_tools(self) -> None:
        def duplicate(value: str) -> str:
            return value

        def variadic(*values: str) -> int:
            return len(values)

        async def asynchronous(value: str) -> str:
            return value

        with self.assertRaises(ToolDefinitionError):
            Agent(provider=ScriptedProvider([]), tools=[duplicate, duplicate])
        with self.assertRaises(ToolDefinitionError):
            FunctionTool.from_callable(variadic)
        with self.assertRaises(ToolDefinitionError):
            FunctionTool.from_callable(asynchronous)


class AgentLoopTest(unittest.TestCase):
    def test_executes_calls_sequentially_and_returns_errors_to_model(self) -> None:
        order: list[str] = []

        def first(value: int) -> int:
            order.append("first")
            return value + 1

        def broken(value: int) -> int:
            order.append("broken")
            raise ValueError("bad value")

        provider = ScriptedProvider(
            [
                LLMResponse(
                    tool_calls=[
                        ToolCall(id="1", name="first", arguments={"value": 2}),
                        ToolCall(id="2", name="broken", arguments={"value": 3}),
                    ],
                    usage=Usage(5, 2, 7),
                ),
                LLMResponse(text="recovered", usage=Usage(4, 3, 7)),
            ]
        )
        result = Agent(provider=provider, tools=[first, broken]).run("go")

        self.assertEqual(order, ["first", "broken"])
        self.assertEqual(result.response, "recovered")
        self.assertEqual(result.usage.total_tokens, 14)
        self.assertFalse(result.tool_executions[1].ok)
        tool_messages = [
            message for message in provider.requests[1].messages if message.role == "tool"
        ]
        self.assertTrue(tool_messages[1].is_error)
        self.assertIn("bad value", tool_messages[1].content)

    def test_max_turns_is_a_hard_limit(self) -> None:
        def again() -> str:
            return "again"

        provider = ScriptedProvider(
            [
                LLMResponse(tool_calls=[ToolCall(id="1", name="again", arguments={})]),
                LLMResponse(tool_calls=[ToolCall(id="2", name="again", arguments={})]),
            ]
        )
        with self.assertRaises(AgentMaxTurnsError):
            Agent(provider=provider, tools=[again], max_turns=2).run("loop")


class LibraryDependencyTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.workspace = init_workspace(Path(self.tmp.name) / "workspace")

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def _write_library(
        self,
        name: str,
        *,
        libraries: list[str] | None = None,
        requirements: list[str] | None = None,
    ) -> None:
        directory = self.workspace / "libs" / name
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "__init__.py").write_text("", encoding="utf-8")
        (directory / "manifest.json").write_text(
            json.dumps(
                {
                    "name": name,
                    "version": "0.1.0",
                    "summary": name,
                    "documentation": name,
                    "exports": [],
                    "requirements": requirements or [],
                    "libraries": libraries or [],
                    "compatibility": {"agentic_canvas": ">=0.1.0"},
                }
            ),
            encoding="utf-8",
        )

    def test_resolves_dependencies_requirements_missing_and_cycles(self) -> None:
        self._write_library("leaf", requirements=["leaf-package>=1"])
        self._write_library(
            "branch",
            libraries=["leaf"],
            requirements=["branch-package>=2", "leaf-package>=1"],
        )
        registry = LibraryRegistry(self.workspace / "libs")

        self.assertEqual(
            [manifest.name for manifest in registry.resolve(["branch"])],
            ["leaf", "branch"],
        )
        self.assertEqual(
            registry.requirements_for(["branch"]),
            ["branch-package>=2", "leaf-package>=1"],
        )

        self._write_library("missing_parent", libraries=["not_installed"])
        with self.assertRaisesRegex(LibraryManifestError, "missing library"):
            registry.resolve(["missing_parent"])

        self._write_library("cycle_a", libraries=["cycle_b"])
        self._write_library("cycle_b", libraries=["cycle_a"])
        with self.assertRaisesRegex(LibraryManifestError, "cycle detected"):
            registry.resolve(["cycle_a"])

    def test_runner_collects_plugin_and_transitive_library_requirements(self) -> None:
        self._write_library("sdk", requirements=["sdk-package>=2"])
        plugin_dir = self.workspace / "plugins" / "uses_sdk"
        plugin_dir.mkdir(parents=True)
        (plugin_dir / "plugin.py").write_text(
            "def plugin_main(request):\n    return {}\n",
            encoding="utf-8",
        )
        (plugin_dir / "manifest.json").write_text(
            json.dumps(
                {
                    "name": "uses_sdk",
                    "version": "0.1.0",
                    "summary": "uses sdk",
                    "documentation": "uses sdk",
                    "entry_point": "plugin.py:plugin_main",
                    "input_schema": {"type": "object"},
                    "output_schema": {"type": "object"},
                    "requirements": ["plugin-package>=1"],
                    "libraries": ["sdk"],
                    "compatibility": {"agentic_canvas": ">=0.1.0"},
                }
            ),
            encoding="utf-8",
        )
        workspace = Workspace(self.workspace)
        manifest = PluginRegistry(workspace.storage).get("uses_sdk")
        requirements = SubprocessPluginRunner._requirements_for(workspace, manifest)
        self.assertEqual(requirements, ["plugin-package>=1", "sdk-package>=2"])


class AgentPluginIntegrationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.workspace = init_workspace(Path(self.tmp.name) / "workspace")

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_agent_uses_local_function_and_wrapped_workspace_plugin(self) -> None:
        agent_runtime_manifest = self.workspace / "libs" / "agent_runtime" / "manifest.json"
        agent_runtime_data = json.loads(agent_runtime_manifest.read_text(encoding="utf-8"))
        agent_runtime_data["requirements"] = []
        agent_runtime_manifest.write_text(
            json.dumps(agent_runtime_data),
            encoding="utf-8",
        )

        target_dir = self.workspace / "plugins" / "echo_target"
        target_dir.mkdir(parents=True)
        (target_dir / "plugin.py").write_text(
            "def plugin_main(request):\n"
            "    return {'echo': request.get('params', {}).get('value')}\n",
            encoding="utf-8",
        )
        (target_dir / "manifest.json").write_text(
            json.dumps(
                {
                    "name": "echo_target",
                    "version": "0.1.0",
                    "summary": "Echo a value.",
                    "documentation": "Echo a value.",
                    "entry_point": "plugin.py:plugin_main",
                    "input_schema": {
                        "type": "object",
                        "properties": {"value": {"type": "string"}},
                        "required": ["value"],
                    },
                    "output_schema": {"type": "object"},
                    "requirements": [],
                    "libraries": [],
                    "compatibility": {"agentic_canvas": ">=0.1.0"},
                }
            ),
            encoding="utf-8",
        )

        agent_dir = self.workspace / "plugins" / "scripted_agent"
        agent_dir.mkdir(parents=True)
        (agent_dir / "plugin.py").write_text(
            '''from libs.agent_runtime import Agent
from libs.llm_core import LLMProvider, LLMResponse, ToolCall
from libs.plugin_calls import plugin_tool

class Scripted(LLMProvider):
    def __init__(self):
        self.turn = 0

    def complete(self, request):
        self.turn += 1
        if self.turn == 1:
            return LLMResponse(tool_calls=[
                ToolCall(id="local", name="uppercase", arguments={"value": "hello"}),
                ToolCall(id="plugin", name="echo_target", arguments={"value": "workspace"}),
            ])
        return LLMResponse(text="done")

def uppercase(value: str) -> str:
    return value.upper()

def plugin_main(request):
    agent = Agent(
        provider=Scripted(),
        tools=[uppercase, plugin_tool(request, "echo_target")],
        max_turns=3,
    )
    return agent.run("run tools").to_dict()
''',
            encoding="utf-8",
        )
        (agent_dir / "manifest.json").write_text(
            json.dumps(
                {
                    "name": "scripted_agent",
                    "version": "0.1.0",
                    "summary": "scripted agent",
                    "documentation": "scripted agent",
                    "entry_point": "plugin.py:plugin_main",
                    "input_schema": {"type": "object"},
                    "output_schema": {"type": "object"},
                    "requirements": [],
                    "libraries": ["agent_runtime", "plugin_calls"],
                    "compatibility": {"agentic_canvas": ">=0.1.0"},
                }
            ),
            encoding="utf-8",
        )

        context = Kernel(
            self.workspace,
            provider=CallPluginProvider("scripted_agent"),
        ).start(Trigger("run scripted agent"))

        trace = [
            json.loads(line)
            for line in (self.workspace / "runs" / f"{context.run_id}.trace.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        ]
        agent_execution = next(
            item
            for item in trace
            if item.get("type") == "plugin_execution" and item.get("plugin") == "scripted_agent"
        )
        self.assertEqual(
            context.status,
            "completed",
            agent_execution["result"].get("stderr") or context.final_response,
        )
        executions = agent_execution["result"]["result"]["tool_executions"]
        self.assertEqual([item["call"]["name"] for item in executions], ["uppercase", "echo_target"])
        self.assertEqual(executions[0]["result"], "HELLO")
        self.assertEqual(executions[1]["result"], {"echo": "workspace"})


if __name__ == "__main__":
    unittest.main()
