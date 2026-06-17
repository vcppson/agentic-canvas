from __future__ import annotations

from pathlib import Path
from typing import Callable

from agentic_canvas.kernel.context import RunContext, RunContextStore, RunTraceStore
from agentic_canvas.kernel.plugin_runner import (
    PluginInputRequest,
    PluginRunner,
    SubprocessPluginRunner,
)
from agentic_canvas.kernel.run_lifecycle import RunLifecycleMixin
from agentic_canvas.kernel.trigger import Trigger
from agentic_canvas.kernel.workspace import Workspace
from agentic_canvas.libraries.registry import LibraryRegistry
from agentic_canvas.plugins.registry import PluginRegistry
from agentic_canvas.providers.base import LLMProvider
from agentic_canvas.providers.factory import provider_from_workspace


InputProvider = Callable[[PluginInputRequest], str]


class Kernel(RunLifecycleMixin):
    """Privileged runtime facade for Runs."""

    def __init__(
        self,
        workspace_root: str | Path,
        *,
        provider: LLMProvider | None = None,
        plugin_runner: PluginRunner | None = None,
        input_provider: InputProvider | None = None,
    ) -> None:
        self.workspace = Workspace(workspace_root)
        self.workspace.stage_order
        self.store = RunContextStore(self.workspace.storage, self.workspace.runs_dir_name)
        self.trace_store = RunTraceStore(self.workspace.storage, self.workspace.runs_dir_name)
        self.provider = provider or provider_from_workspace(self.workspace)
        self.plugin_runner = plugin_runner or self._plugin_runner_from_workspace()
        self.input_provider = input_provider
        self.plugin_registry = PluginRegistry(
            self.workspace.storage,
            self.workspace.config.get("plugins_dir", "plugins"),
        )
        self.library_registry = LibraryRegistry(
            self.workspace.storage,
            self.workspace.config.get("libraries_dir", "libs"),
        )

    def start(self, trigger: Trigger) -> RunContext:
        """Start a new Run from a Trigger."""

        self._validate_workspace_manifests()
        context = RunContext.from_trigger(
            trigger,
            workspace_root=self.workspace.root,
            orchestrator_system_prompt=self.workspace.load_orchestrator_prompt(),
        )
        self.trace_store.append(
            context,
            "trigger_received",
            trigger_type=trigger.type,
            user_input=trigger.user_input,
            metadata=trigger.metadata,
            orchestrator_system_prompt=context.orchestrator_system_prompt,
        )
        self.store.save(context)
        return self._run_from_start(context)

    def _plugin_runner_from_workspace(self) -> PluginRunner:
        config = self.workspace.plugin_runner_config
        strategy = str(config.get("strategy", "subprocess"))
        timeout_seconds = int(config.get("timeout_seconds", 120))
        if strategy != "subprocess":
            raise RuntimeError(f"Unsupported plugin runner strategy: {strategy!r}")
        return SubprocessPluginRunner(timeout_seconds=timeout_seconds)

    def _validate_workspace_manifests(self) -> None:
        plugins = self.plugin_registry.load_all()
        self.library_registry.load_all()
        for manifest in plugins.values():
            try:
                self.library_registry.resolve(manifest.libraries)
            except Exception as exc:
                raise RuntimeError(
                    f"Plugin {manifest.name!r} has invalid library dependencies: {exc}"
                ) from exc
