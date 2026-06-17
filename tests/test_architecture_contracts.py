from __future__ import annotations

import ast
import contextlib
import io
import json
import sys
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agentic_canvas.kernel.context import RunContext
from agentic_canvas.kernel.plugin_runner import SubprocessPluginRunner
from agentic_canvas.kernel.run import Kernel
from agentic_canvas.kernel.storage import FileSystemWorkspaceStorage, WorkspaceStorageError
from agentic_canvas.kernel.trigger import Trigger
from agentic_canvas.kernel.workspace import Workspace, init_workspace
from agentic_canvas.cli.app import main as cli_main
from agentic_canvas.libraries.registry import LibraryRegistry
from agentic_canvas.orchestrator.orchestrator import Orchestrator
from agentic_canvas.plugins.registry import PluginRegistry
from agentic_canvas.providers.gemini import _gemini_schema
from tests.support import (
    CallPluginProvider,
    RaisingProvider,
    StaticProvider,
    read_trace,
    set_stages,
    write_plugin,
)


class ArchitectureContractsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.workspace = init_workspace(Path(self.tmp.name) / "workspace")

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_template_initializes_with_fundamental_plugins_and_libraries(self) -> None:
        plugins = PluginRegistry(self.workspace / "plugins").load_all()
        libraries = LibraryRegistry(self.workspace / "libs").load_all()

        self.assertLessEqual(
            {
                "plugin_catalog",
                "plugin_info",
                "plugin_create",
                "library_create",
                "command_handle",
                "command_pack_catalog",
                "command_pack_create",
                "command_pack_remove",
                "conversation_load",
                "conversation_store",
                "conversation_clear",
                "reference_agent",
            },
            set(plugins),
        )
        self.assertLessEqual(
            {
                "run_control",
                "plugin_runtime",
                "manifest_docs",
                "plugin_calls",
                "conversation_memory",
                "command_packs",
                "llm_core",
                "llm_google",
                "llm_openai",
                "llm_anthropic",
                "agent_runtime",
            },
            set(libraries),
        )
        self.assertTrue((self.workspace / "commands" / "conversation" / "manifest.json").is_file())

    def test_template_wires_conversation_plugins_into_stages(self) -> None:
        workspace = Workspace(self.workspace)

        pre = workspace.stage_plugins("pre_orchestrator")
        post = workspace.stage_plugins("post_orchestrator")

        self.assertEqual([entry.name for entry in pre], ["command_handle", "conversation_load"])
        self.assertEqual([entry.name for entry in post], ["conversation_store"])
        self.assertEqual(pre[0].params["prefix"], "/")
        self.assertEqual(pre[1].params["limit"], 5)
        self.assertEqual(post[0].params, {})

    def test_workspace_storage_rejects_path_escapes(self) -> None:
        storage = FileSystemWorkspaceStorage(self.workspace)

        for operation in [
            lambda: storage.read_text("../outside.txt"),
            lambda: storage.write_text("../outside.txt", "nope"),
            lambda: storage.ensure_dir("../outside"),
            lambda: storage.delete_tree("../outside"),
            lambda: storage.replace_tree("../outside"),
        ]:
            with self.assertRaises(WorkspaceStorageError):
                operation()

    def test_registries_load_through_workspace_storage(self) -> None:
        workspace = Workspace(self.workspace)

        plugins = PluginRegistry(workspace.storage).load_all()
        libraries = LibraryRegistry(workspace.storage).load_all()

        self.assertIn("plugin_catalog", plugins)
        self.assertIn("run_control", libraries)

    def test_trigger_creates_persisted_run_context_and_input_defaults(self) -> None:
        context = Kernel(self.workspace, provider=StaticProvider("done")).start(Trigger("hello"))

        self.assertEqual(context.status, "completed")
        self.assertEqual(context.user_input, "hello")
        self.assertEqual(context.input, "hello")
        self.assertEqual(context.final_response, "done")
        self.assertTrue((self.workspace / "runs" / "current.json").is_file())
        self.assertTrue((self.workspace / "runs" / f"{context.run_id}.json").is_file())
        self.assertTrue((self.workspace / "runs" / f"{context.run_id}.trace.jsonl").is_file())

    def test_conversation_memory_stores_and_loads_recent_prompt_context(self) -> None:
        first_provider = StaticProvider("planner memory saved")
        first = Kernel(self.workspace, provider=first_provider).start(
            Trigger("teach the planner about canvas growth")
        )
        self.assertEqual(first.status, "completed")

        record_path = self.workspace / "memory" / "conversations" / "runs" / f"{first.run_id}.json"
        index_path = self.workspace / "memory" / "conversations" / "index.json"
        self.assertTrue(record_path.is_file())
        self.assertTrue(index_path.is_file())
        record = json.loads(record_path.read_text(encoding="utf-8"))
        self.assertEqual(record["run_id"], first.run_id)
        self.assertEqual(record["user_input"], "teach the planner about canvas growth")
        self.assertEqual(record["input"], "teach the planner about canvas growth")
        self.assertEqual(record["final_response"], "planner memory saved")
        self.assertEqual(record["status"], "completed")
        self.assertNotIn("plugin_executions", record)

        second_provider = StaticProvider("loaded memory")
        second = Kernel(self.workspace, provider=second_provider).start(
            Trigger("ask a totally unrelated question")
        )

        self.assertEqual(second.input, "ask a totally unrelated question")
        self.assertTrue(second.additional_prompts)
        self.assertIn("Recent Conversations", second_provider.seen_system_prompt)
        self.assertIn("teach the planner about canvas growth", second_provider.seen_system_prompt)
        trace = read_trace(self.workspace, second.run_id)
        started = next(item for item in trace if item["type"] == "orchestrator_started")
        self.assertIn("Recent Conversations", started["orchestrator_system_prompt"])
        self.assertEqual(started["base_orchestrator_system_prompt"], second.orchestrator_system_prompt)
        self.assertEqual(started["additional_prompts"][0]["source"], "conversation_load")

        workspace_obj = Workspace(self.workspace)
        rerun = SubprocessPluginRunner(timeout_seconds=20).run(
            workspace=workspace_obj,
            manifest=PluginRegistry(workspace_obj.storage).get("conversation_store"),
            context=second,
            mode="call",
            params={},
        )
        self.assertTrue(rerun.ok)
        index = json.loads(index_path.read_text(encoding="utf-8"))
        self.assertEqual(
            1,
            sum(1 for item in index["runs"] if item["run_id"] == second.run_id),
        )

    def test_command_pack_catalog_lists_builtin_conversation_pack(self) -> None:
        context = Kernel(
            self.workspace,
            provider=CallPluginProvider("command_pack_catalog"),
        ).start(Trigger("catalog commands"))

        self.assertEqual(context.status, "completed")
        trace = read_trace(self.workspace, context.run_id)
        catalog_trace = next(
            item
            for item in trace
            if item["type"] == "plugin_execution" and item["plugin"] == "command_pack_catalog"
        )
        packs = catalog_trace["result"]["result"]["command_packs"]
        self.assertIn("conversation", [item["name"] for item in packs])
        aliases = [
            alias
            for pack in packs
            for command in pack["commands"]
            for alias in command["aliases"]
        ]
        self.assertIn("/conversation clear", aliases)

    def test_command_pack_create_validates_target_plugin_and_remove_deletes_pack(self) -> None:
        write_plugin(
            self.workspace,
            "echo_command",
            """
def plugin_main(request):
    return {"response": request.get("params", {}).get("value", "")}
""".strip()
            + "\n",
        )

        created = Kernel(
            self.workspace,
            provider=CallPluginProvider(
                "command_pack_create",
                {
                    "manifest": {
                        "name": "echo",
                        "version": "0.1.0",
                        "summary": "Echo commands.",
                        "documentation": "Routes echo commands.",
                        "commands": [
                            {
                                "name": "echo.say",
                                "aliases": ["/echo"],
                                "plugin": "echo_command",
                                "arguments": [
                                    {"name": "value", "type": "string", "required": True}
                                ],
                            }
                        ],
                    }
                },
            ),
        ).start(Trigger("create command pack"))

        self.assertEqual(created.status, "completed")
        self.assertTrue((self.workspace / "commands" / "echo" / "manifest.json").is_file())

        missing = Kernel(
            self.workspace,
            provider=CallPluginProvider(
                "command_pack_create",
                {
                    "manifest": {
                        "name": "bad_command_pack",
                        "version": "0.1.0",
                        "summary": "Bad.",
                        "documentation": "Bad target.",
                        "commands": [
                            {
                                "name": "bad.missing",
                                "aliases": ["/bad"],
                                "plugin": "does_not_exist",
                            }
                        ],
                    }
                },
            ),
        ).start(Trigger("create bad command pack"))

        self.assertEqual(missing.status, "aborted")
        self.assertIn("does not exist", missing.final_response)

        removed = Kernel(
            self.workspace,
            provider=CallPluginProvider("command_pack_remove", {"name": "echo"}),
        ).start(Trigger("remove command pack"))

        self.assertEqual(removed.status, "completed")
        self.assertFalse((self.workspace / "commands" / "echo").exists())

    def test_unknown_slash_command_aborts_before_orchestrator(self) -> None:
        context = Kernel(self.workspace, provider=RaisingProvider()).start(Trigger("/nope"))

        self.assertEqual(context.status, "aborted")
        self.assertIn("Unknown command", context.final_response)
        trace = read_trace(self.workspace, context.run_id)
        self.assertFalse(any(item["type"] == "orchestrator_started" for item in trace))
        self.assertFalse(
            any(
                item["type"] == "plugin_execution" and item["plugin"] == "conversation_load"
                for item in trace
            )
        )

    def test_conversation_command_clears_conversation_memory_index_with_audit(self) -> None:
        first = Kernel(self.workspace, provider=StaticProvider("remembered")).start(
            Trigger("remember command-test memory")
        )
        index_path = self.workspace / "memory" / "conversations" / "index.json"
        index = json.loads(index_path.read_text(encoding="utf-8"))
        self.assertTrue(any(item["run_id"] == first.run_id for item in index["runs"]))

        cleared = Kernel(self.workspace, provider=RaisingProvider()).start(
            Trigger("/conversation clear")
        )

        self.assertEqual(cleared.status, "completed")
        self.assertEqual(cleared.decision, "stop")
        self.assertIn("Conversation memory cleared.", cleared.final_response)
        index = json.loads(index_path.read_text(encoding="utf-8"))
        self.assertEqual(index["runs"], [])
        self.assertEqual(index["last_clear"]["cleared_by_run_id"], cleared.run_id)
        clear_path = self.workspace / index["last_clear"]["path"]
        self.assertTrue(clear_path.is_file())
        self.assertTrue((self.workspace / "memory" / "conversations" / "runs" / f"{first.run_id}.json").is_file())
        trace = read_trace(self.workspace, cleared.run_id)
        command_trace = next(
            item
            for item in trace
            if item["type"] == "plugin_execution" and item["plugin"] == "conversation_clear"
        )
        self.assertEqual(command_trace["mode"], "command")
        self.assertEqual(command_trace["command"]["command"], "conversation.clear")
        self.assertFalse(any(item["type"] == "orchestrator_started" for item in trace))
        self.assertFalse(
            any(
                item["type"] == "plugin_execution" and item["plugin"] == "conversation_store"
                for item in trace
            )
        )

    def test_command_missing_required_argument_aborts_without_target_call(self) -> None:
        write_plugin(
            self.workspace,
            "echo_command",
            """
def plugin_main(request):
    return {"response": request.get("params", {}).get("value", "")}
""".strip()
            + "\n",
        )
        Kernel(
            self.workspace,
            provider=CallPluginProvider(
                "command_pack_create",
                {
                    "manifest": {
                        "name": "echo",
                        "version": "0.1.0",
                        "summary": "Echo commands.",
                        "documentation": "Routes echo commands.",
                        "commands": [
                            {
                                "name": "echo.say",
                                "aliases": ["/echo"],
                                "plugin": "echo_command",
                                "arguments": [
                                    {"name": "value", "type": "string", "required": True}
                                ],
                            }
                        ],
                    }
                },
            ),
        ).start(Trigger("create command pack"))

        context = Kernel(self.workspace, provider=RaisingProvider()).start(Trigger("/echo"))

        self.assertEqual(context.status, "aborted")
        self.assertIn("missing required argument", context.final_response)
        trace = read_trace(self.workspace, context.run_id)
        self.assertFalse(
            any(
                item["type"] == "plugin_execution" and item["plugin"] == "echo_command"
                for item in trace
            )
        )

    def test_pre_stage_can_patch_input_and_post_stage_can_patch_final_response(self) -> None:
        write_plugin(
            self.workspace,
            "normalize",
            """
def plugin_main(request):
    return {"decision": "continue", "patch": {"input": "normalized"}}
""".strip()
            + "\n",
        )
        write_plugin(
            self.workspace,
            "decorate",
            """
def plugin_main(request):
    response = request["context"].get("orchestrator_response") or ""
    return {"decision": "continue", "patch": {"final_response": response + "!"}}
""".strip()
            + "\n",
        )
        set_stages(self.workspace, pre_orchestrator=["normalize"], post_orchestrator=["decorate"])

        provider = StaticProvider("answer")
        context = Kernel(self.workspace, provider=provider).start(Trigger("raw"))

        self.assertEqual(provider.seen_messages[-1]["content"], "normalized")
        self.assertEqual(context.final_response, "answer!")
        trace = read_trace(self.workspace, context.run_id)
        normalize_trace = next(
            item
            for item in trace
            if item["type"] == "plugin_execution" and item["plugin"] == "normalize"
        )
        self.assertEqual(normalize_trace["mode"], "stage")
        self.assertEqual(normalize_trace["params"], {})
        self.assertEqual(normalize_trace["result"]["result"]["decision"], "continue")
        self.assertEqual(normalize_trace["result"]["result"]["patch"]["input"], "normalized")
        self.assertIsInstance(normalize_trace["result"]["duration_seconds"], float)

    def test_pre_stage_stop_skips_orchestrator(self) -> None:
        write_plugin(
            self.workspace,
            "stopper",
            """
def plugin_main(request):
    return {"decision": "stop", "response": "handled before orchestrator"}
""".strip()
            + "\n",
        )
        set_stages(self.workspace, pre_orchestrator=["stopper"])

        context = Kernel(self.workspace, provider=RaisingProvider()).start(Trigger("stop now"))

        self.assertEqual(context.status, "completed")
        self.assertEqual(context.decision, "stop")
        self.assertEqual(context.final_response, "handled before orchestrator")

    def test_stage_await_user_returns_input_to_same_plugin_invocation(self) -> None:
        write_plugin(
            self.workspace,
            "asker",
            """
from libs.run_control import await_user


def plugin_main(request):
    answer = await_user("Choose a target")
    return {
        "decision": "continue",
        "patch": {"state": {"chosen_target": answer}},
    }
""".strip()
            + "\n",
            libraries=["run_control"],
        )
        set_stages(self.workspace, pre_orchestrator=["asker"])

        context = Kernel(
            self.workspace,
            provider=StaticProvider("answered"),
            input_provider=lambda request: "target A",
        ).start(
            Trigger("begin")
        )

        self.assertEqual(context.status, "completed")
        self.assertEqual(context.user_input, "begin")
        self.assertEqual(context.input, "begin")
        self.assertEqual(context.awaiting, None)
        self.assertEqual(context.user_inputs[-1]["input"], "target A")
        self.assertEqual(context.user_input_requests[-1]["response"], "target A")
        self.assertEqual(context.final_response, "answered")
        self.assertEqual(context.state["chosen_target"], "target A")
        trace = read_trace(self.workspace, context.run_id)
        self.assertEqual([item["sequence"] for item in trace], list(range(1, len(trace) + 1)))
        request_trace = next(item for item in trace if item["type"] == "user_input_requested")
        self.assertEqual(request_trace["message"], "Choose a target")
        self.assertEqual(request_trace["status"], "awaiting_user_input")
        received_trace = next(item for item in trace if item["type"] == "user_input_received")
        self.assertEqual(received_trace["user_input"], "target A")
        asker_traces = [
            item
            for item in trace
            if item["type"] == "plugin_execution" and item["plugin"] == "asker"
        ]
        self.assertEqual(len(asker_traces), 1)
        asker_trace = asker_traces[0]
        self.assertEqual(asker_trace["result"]["kind"], "result")
        self.assertEqual(asker_trace["result"]["result"]["decision"], "continue")
        self.assertEqual(len(asker_trace["result"]["input_requests"]), 1)
        self.assertEqual(asker_trace["result"]["input_requests"][0]["message"], "Choose a target")
        self.assertEqual(asker_trace["result"]["input_requests"][0]["response"], "target A")

    def test_await_user_without_input_provider_aborts_with_trace(self) -> None:
        write_plugin(
            self.workspace,
            "asker",
            """
from libs.run_control import await_user


def plugin_main(request):
    answer = await_user("Choose a target")
    return {"decision": "continue", "patch": {"state": {"chosen_target": answer}}}
""".strip()
            + "\n",
            libraries=["run_control"],
        )
        set_stages(self.workspace, pre_orchestrator=["asker"])

        context = Kernel(self.workspace, provider=StaticProvider("unused")).start(Trigger("begin"))

        self.assertEqual(context.status, "aborted")
        self.assertIn("No input provider", context.final_response)
        self.assertEqual(context.user_input_requests[-1]["message"], "Choose a target")
        trace = read_trace(self.workspace, context.run_id)
        requested = next(item for item in trace if item["type"] == "user_input_requested")
        self.assertEqual(requested["status"], "awaiting_user_input")
        plugin_trace = next(
            item
            for item in trace
            if item["type"] == "plugin_execution" and item["plugin"] == "asker"
        )
        self.assertEqual(plugin_trace["result"]["kind"], "error")
        self.assertEqual(plugin_trace["result"]["input_requests"][0]["message"], "Choose a target")

    def test_cli_prompts_immediately_for_plugin_await_user(self) -> None:
        write_plugin(
            self.workspace,
            "asker",
            """
from libs.run_control import await_user


def plugin_main(request):
    answer = await_user("CLI target?")
    return {"decision": "stop", "response": f"answer={answer}"}
""".strip()
            + "\n",
            libraries=["run_control"],
        )

        output = io.StringIO()
        with patch("builtins.input", return_value="from cli"):
            with contextlib.redirect_stdout(output):
                cli_main(["run", str(self.workspace), "call_plugin", "asker", "{}"])

        text = output.getvalue()
        self.assertIn("CLI target?", text)
        self.assertIn("answer=from cli", text)

    def test_run_control_stop_is_traced(self) -> None:
        write_plugin(
            self.workspace,
            "run_stop",
            """
from libs.run_control import stop


def plugin_main(request):
    stop("finished early", response="stopped by run control")
""".strip()
            + "\n",
            libraries=["run_control"],
        )
        set_stages(self.workspace, pre_orchestrator=["run_stop"])

        context = Kernel(self.workspace, provider=RaisingProvider()).start(Trigger("begin"))

        self.assertEqual(context.status, "completed")
        self.assertEqual(context.final_response, "stopped by run control")
        trace = read_trace(self.workspace, context.run_id)
        plugin_trace = next(
            item
            for item in trace
            if item["type"] == "plugin_execution" and item["plugin"] == "run_stop"
        )
        self.assertEqual(plugin_trace["result"]["kind"], "run_control")
        self.assertEqual(plugin_trace["result"]["decision"], "stop")
        stopped_trace = next(item for item in trace if item["type"] == "run_stopped")
        self.assertEqual(stopped_trace["response"], "stopped by run control")

    def test_run_control_abort_is_traced(self) -> None:
        write_plugin(
            self.workspace,
            "run_abort",
            """
from libs.run_control import abort


def plugin_main(request):
    abort("blocked")
""".strip()
            + "\n",
            libraries=["run_control"],
        )
        set_stages(self.workspace, pre_orchestrator=["run_abort"])

        context = Kernel(self.workspace, provider=RaisingProvider()).start(Trigger("begin"))

        self.assertEqual(context.status, "aborted")
        trace = read_trace(self.workspace, context.run_id)
        plugin_trace = next(
            item
            for item in trace
            if item["type"] == "plugin_execution" and item["plugin"] == "run_abort"
        )
        self.assertEqual(plugin_trace["result"]["kind"], "run_control")
        self.assertEqual(plugin_trace["result"]["decision"], "abort")
        aborted_trace = next(item for item in trace if item["type"] == "run_aborted")
        self.assertEqual(aborted_trace["reason"], "blocked")

    def test_orchestrator_exposes_only_call_plugin(self) -> None:
        workspace_obj = Workspace(self.workspace)
        context = RunContext.from_trigger(
            Trigger("hello"),
            workspace_root=self.workspace,
            orchestrator_system_prompt=workspace_obj.load_orchestrator_prompt(),
        )
        orchestrator = Orchestrator(
            workspace=workspace_obj,
            plugin_runner=SubprocessPluginRunner(timeout_seconds=20),
            provider=StaticProvider(),
        )

        tools = orchestrator.tool_definitions(context)

        self.assertEqual(len(tools), 1)
        self.assertEqual(tools[0].name, "call_plugin")

    def test_orchestrator_called_plugin_await_user_returns_before_orchestrator_continues(self) -> None:
        write_plugin(
            self.workspace,
            "needs_user",
            """
from libs.run_control import await_user


def plugin_main(request):
    answer = await_user("Need missing value")
    return {"received": answer}
""".strip()
            + "\n",
            libraries=["run_control"],
        )

        context = Kernel(
            self.workspace,
            provider=CallPluginProvider("needs_user"),
            input_provider=lambda request: "the missing value",
        ).start(Trigger("do something"))

        self.assertEqual(context.status, "completed")
        self.assertEqual(context.stage, "post_orchestrator")
        self.assertEqual(context.awaiting, None)
        self.assertEqual(context.orchestrator_response, "plugin returned")
        trace = read_trace(self.workspace, context.run_id)
        started = next(item for item in trace if item["type"] == "orchestrator_started")
        self.assertEqual(started["input"], "do something")
        self.assertIn("orchestrator_system_prompt", started)
        self.assertEqual([tool["name"] for tool in started["tools"]], ["call_plugin"])
        requested = next(item for item in trace if item["type"] == "user_input_requested")
        self.assertEqual(requested["plugin"], "needs_user")
        self.assertEqual(requested["message"], "Need missing value")
        received = next(item for item in trace if item["type"] == "user_input_received")
        self.assertEqual(received["user_input"], "the missing value")
        plugin_trace = next(
            item
            for item in trace
            if item["type"] == "plugin_execution" and item["plugin"] == "needs_user"
        )
        self.assertEqual(plugin_trace["mode"], "call")
        self.assertEqual(plugin_trace["params"], {})
        self.assertEqual(plugin_trace["result"]["kind"], "result")
        self.assertEqual(plugin_trace["result"]["result"], {"received": "the missing value"})
        self.assertEqual(len(plugin_trace["result"]["input_requests"]), 1)
        self.assertEqual(plugin_trace["result"]["input_requests"][0]["response"], "the missing value")

    def test_stage_plugin_without_decision_aborts(self) -> None:
        write_plugin(
            self.workspace,
            "bad_stage",
            """
def plugin_main(request):
    return {"data": "no decision"}
""".strip()
            + "\n",
        )
        set_stages(self.workspace, pre_orchestrator=["bad_stage"])

        context = Kernel(self.workspace, provider=StaticProvider()).start(Trigger("hello"))

        self.assertEqual(context.status, "aborted")
        self.assertIn("returned no decision", context.final_response)
        trace = read_trace(self.workspace, context.run_id)
        plugin_trace = next(
            item
            for item in trace
            if item["type"] == "plugin_execution" and item["plugin"] == "bad_stage"
        )
        self.assertEqual(plugin_trace["result"]["kind"], "result")
        self.assertEqual(plugin_trace["result"]["result"], {"data": "no decision"})
        abort_trace = next(item for item in trace if item["type"] == "run_aborted")
        self.assertIn("returned no decision", abort_trace["reason"])

    def test_missing_plugin_manifest_is_rejected(self) -> None:
        (self.workspace / "plugins" / "broken").mkdir()

        with self.assertRaisesRegex(Exception, "manifest"):
            PluginRegistry(self.workspace / "plugins").load_all()

    def test_creation_is_available_through_call_plugin_not_kernel_api(self) -> None:
        context = Kernel(
            self.workspace,
            provider=CallPluginProvider(
                "plugin_create",
                {"name": "made_plugin", "summary": "Created by a plugin."},
            ),
        ).start(Trigger("create plugin"))

        self.assertEqual(context.status, "completed")
        self.assertTrue((self.workspace / "plugins" / "made_plugin" / "manifest.json").is_file())
        self.assertIn(
            "plugin_create",
            [event.get("plugin") for event in context.events if event["type"] == "plugin_called"],
        )
        trace = read_trace(self.workspace, context.run_id)
        plugin_trace = next(
            item
            for item in trace
            if item["type"] == "plugin_execution" and item["plugin"] == "plugin_create"
        )
        self.assertEqual(plugin_trace["mode"], "call")
        self.assertEqual(plugin_trace["params"]["name"], "made_plugin")
        self.assertTrue(plugin_trace["result"]["result"]["created"])

    def test_plugin_create_generates_structured_manifest_documentation(self) -> None:
        context = Kernel(
            self.workspace,
            provider=CallPluginProvider(
                "plugin_create",
                {
                    "name": "documented_plugin",
                    "summary": "Created plugin documentation.",
                    "input_schema": {
                        "type": "object",
                        "properties": {"value": {"type": "string"}},
                        "required": ["value"],
                        "additionalProperties": False,
                    },
                    "output_schema": {
                        "type": "object",
                        "properties": {"response": {"type": "string"}},
                    },
                    "libraries": ["plugin_runtime"],
                    "tags": ["example"],
                },
            ),
        ).start(Trigger("create documented plugin"))

        self.assertEqual(context.status, "completed")
        manifest_path = self.workspace / "plugins" / "documented_plugin" / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        documentation = manifest["documentation"]
        self.assertIn("Created plugin documentation.", documentation)
        self.assertIn("## Behavior", documentation)
        self.assertIn("## Input Schema", documentation)
        self.assertIn("Required: value", documentation)
        self.assertIn("- value: string", documentation)
        self.assertIn("## Output Schema", documentation)
        self.assertIn("- response: string", documentation)
        self.assertIn("## Dependencies", documentation)
        self.assertIn("Libraries: plugin_runtime", documentation)
        self.assertIn("Tags: example", documentation)

    def test_plugin_create_rejects_files_that_escape_plugin_directory(self) -> None:
        context = Kernel(
            self.workspace,
            provider=CallPluginProvider(
                "plugin_create",
                {
                    "name": "bad_paths",
                    "summary": "Bad paths.",
                    "files": {
                        "../escape.py": "x = 1",
                        "plugin.py": "def plugin_main(request): return {}",
                    },
                },
            ),
        ).start(Trigger("create bad plugin"))

        self.assertEqual(context.status, "aborted")
        self.assertFalse((self.workspace / "plugins" / "escape.py").exists())

    def test_failed_plugin_execution_is_traced_before_abort(self) -> None:
        write_plugin(
            self.workspace,
            "raiser",
            """
def plugin_main(request):
    raise RuntimeError("boom")
""".strip()
            + "\n",
        )
        set_stages(self.workspace, pre_orchestrator=["raiser"])

        context = Kernel(self.workspace, provider=StaticProvider()).start(Trigger("hello"))

        self.assertEqual(context.status, "aborted")
        trace = read_trace(self.workspace, context.run_id)
        plugin_trace = next(
            item
            for item in trace
            if item["type"] == "plugin_execution" and item["plugin"] == "raiser"
        )
        self.assertEqual(plugin_trace["result"]["kind"], "error")
        self.assertFalse(plugin_trace["result"]["ok"])
        self.assertIn("boom", plugin_trace["result"]["message"])
        self.assertEqual(trace[-1]["type"], "run_aborted")

    def test_library_creation_is_available_through_call_plugin(self) -> None:
        context = Kernel(
            self.workspace,
            provider=CallPluginProvider(
                "library_create",
                {"name": "made_library", "summary": "Created by a plugin."},
            ),
        ).start(Trigger("create library"))

        self.assertEqual(context.status, "completed")
        self.assertTrue((self.workspace / "libs" / "made_library" / "manifest.json").is_file())
        libraries = LibraryRegistry(self.workspace / "libs").load_all()
        self.assertIn("made_library", libraries)

    def test_library_create_generates_manifest_documentation_from_exports(self) -> None:
        context = Kernel(
            self.workspace,
            provider=CallPluginProvider(
                "library_create",
                {
                    "name": "documented_library",
                    "summary": "Documentation generated from code.",
                    "documentation": "Manual documentation should not be stored.",
                    "exports": ["make_title", "Worker", "NO_DOC", "MISSING_EXPORT"],
                    "files": {
                        "__init__.py": '''
NO_DOC = 1


def make_title(value: str) -> str:
    """Build a display title."""
    return value.title()


def helper() -> str:
    """This helper is not exported."""
    return "hidden"


class Worker:
    """Run a unit of library work."""
'''.strip()
                        + "\n"
                    },
                },
            ),
        ).start(Trigger("create documented library"))

        self.assertEqual(context.status, "completed")
        manifest_path = self.workspace / "libs" / "documented_library" / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        documentation = manifest["documentation"]
        self.assertIn("Documentation generated from code.", documentation)
        self.assertIn("### make_title", documentation)
        self.assertIn("Type: function", documentation)
        self.assertIn("Signature: make_title(value: str) -> str", documentation)
        self.assertIn("Build a display title.", documentation)
        self.assertIn("### Worker", documentation)
        self.assertIn("Type: class", documentation)
        self.assertIn("Run a unit of library work.", documentation)
        self.assertIn("### NO_DOC", documentation)
        self.assertIn("Documentation: Missing docstring.", documentation)
        self.assertIn("### MISSING_EXPORT", documentation)
        self.assertIn("Documentation: Missing exported API item in library code.", documentation)
        self.assertNotIn("helper", documentation)
        self.assertNotIn("Manual documentation should not be stored.", documentation)

    def test_library_create_manifest_schema_no_longer_accepts_documentation_param(self) -> None:
        manifest_path = (
            Path(__file__).resolve().parents[1]
            / "src"
            / "agentic_canvas"
            / "workspace_template"
            / "plugins"
            / "library_create"
            / "manifest.json"
        )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

        self.assertNotIn("documentation", manifest["input_schema"]["properties"])
        self.assertFalse(manifest["input_schema"].get("additionalProperties", True))

    def test_documentation_generation_uses_template_library(self) -> None:
        template_root = (
            Path(__file__).resolve().parents[1]
            / "src"
            / "agentic_canvas"
            / "workspace_template"
        )
        template_dir = template_root / "libs" / "manifest_docs" / "templates"
        plugin_create = json.loads(
            (template_root / "plugins" / "plugin_create" / "manifest.json").read_text(
                encoding="utf-8"
            )
        )
        library_create = json.loads(
            (template_root / "plugins" / "library_create" / "manifest.json").read_text(
                encoding="utf-8"
            )
        )

        self.assertTrue((template_dir / "plugin_documentation.md").is_file())
        self.assertTrue((template_dir / "library_documentation.md").is_file())
        self.assertIn("manifest_docs", plugin_create["libraries"])
        self.assertIn("manifest_docs", library_create["libraries"])

    def test_workspace_template_manifest_documentation_uses_structured_formats(self) -> None:
        template_root = (
            Path(__file__).resolve().parents[1]
            / "src"
            / "agentic_canvas"
            / "workspace_template"
        )

        for manifest_path in sorted((template_root / "libs").glob("*/manifest.json")):
            with self.subTest(library=manifest_path.parent.name):
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                documentation = manifest["documentation"]
                self.assertIn(manifest["summary"], documentation)
                self.assertIn("## API Reference", documentation)
                for export in manifest["exports"]:
                    self.assertIn(f"### {export}", documentation)
                self.assertIn("Type:", documentation)
                self.assertIn("Source:", documentation)

        for manifest_path in sorted((template_root / "plugins").glob("*/manifest.json")):
            with self.subTest(plugin=manifest_path.parent.name):
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                documentation = manifest["documentation"]
                self.assertIn(manifest["summary"], documentation)
                self.assertIn("## Behavior", documentation)
                self.assertIn("## Input Schema", documentation)
                self.assertIn("## Output Schema", documentation)
                self.assertIn("## Dependencies", documentation)
                self.assertIn("Libraries:", documentation)
                self.assertIn("Requirements:", documentation)

    def test_template_plugin_runtime_exports_match_actual_api_items(self) -> None:
        template_root = Path(__file__).resolve().parents[1] / "src" / "agentic_canvas" / "workspace_template"
        manifest_path = template_root / "libs" / "plugin_runtime" / "manifest.json"
        code_path = template_root / "libs" / "plugin_runtime" / "__init__.py"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        tree = ast.parse(code_path.read_text(encoding="utf-8"))
        actual_api_items = {
            node.name
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
        }
        actual_api_items.update(
            target.id
            for node in tree.body
            if isinstance(node, ast.Assign)
            for target in node.targets
            if isinstance(target, ast.Name)
        )

        self.assertLessEqual(set(manifest["exports"]), actual_api_items)
        self.assertNotIn("ensure_child_path", manifest["exports"])
        self.assertNotIn("write_json", manifest["exports"])

    def test_gemini_schema_drops_unsupported_json_schema_keys(self) -> None:
        schema = _gemini_schema(
            {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Plugin name",
                        "additionalProperties": False,
                    }
                },
                "required": ["name"],
                "additionalProperties": False,
            }
        )

        self.assertEqual(schema["type"], "object")
        self.assertEqual(schema["properties"]["name"]["type"], "string")
        self.assertNotIn("additionalProperties", schema)
        self.assertNotIn("additionalProperties", schema["properties"]["name"])


if __name__ == "__main__":
    unittest.main()
