from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from agentic_canvas.kernel.context import RunContext
from agentic_canvas.kernel.plugin_protocol import (
    PluginCallHandler,
    PluginExecutionResult,
    PluginInputHandler,
)
from agentic_canvas.kernel.workspace import Workspace
from agentic_canvas.plugins.manifest import PluginManifest


class PluginRunner(ABC):
    @abstractmethod
    def run(
        self,
        *,
        workspace: Workspace,
        manifest: PluginManifest,
        context: RunContext,
        mode: str,
        params: dict[str, Any] | None = None,
        input_handler: PluginInputHandler | None = None,
        plugin_call_handler: PluginCallHandler | None = None,
    ) -> PluginExecutionResult:
        """Execute one plugin through the plugin protocol."""
