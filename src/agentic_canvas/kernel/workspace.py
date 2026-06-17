from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from agentic_canvas.kernel.storage import FileSystemWorkspaceStorage, WorkspaceStorage


CORE_STAGE_ORDER = ["pre_orchestrator", "orchestrator", "post_orchestrator"]


@dataclass(frozen=True)
class StagePluginConfig:
    name: str
    params: dict[str, Any] = field(default_factory=dict)


class WorkspaceConfigError(ValueError):
    pass


class Workspace:
    """The composition boundary for a project instance."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).resolve()
        self.storage: WorkspaceStorage = FileSystemWorkspaceStorage(self.root)
        if not self.storage.is_file("workspace.json"):
            raise WorkspaceConfigError(f"Workspace config not found: {self.root / 'workspace.json'}")
        self.config = self.storage.read_json("workspace.json")

    @property
    def plugins_dir(self) -> Path:
        return self.root / self.config.get("plugins_dir", "plugins")

    @property
    def libraries_dir(self) -> Path:
        return self.root / self.config.get("libraries_dir", "libs")

    @property
    def runs_dir_name(self) -> str:
        return self.config.get("runs_dir", "runs")

    @property
    def stage_order(self) -> list[str]:
        configured = self.config.get("stage_order", CORE_STAGE_ORDER)
        if configured != CORE_STAGE_ORDER:
            raise WorkspaceConfigError(
                "The core stage order must be "
                f"{CORE_STAGE_ORDER!r}; got {configured!r}."
            )
        return list(CORE_STAGE_ORDER)

    @property
    def provider_config(self) -> dict[str, Any]:
        return dict(self.config.get("provider", {}))

    @property
    def plugin_runner_config(self) -> dict[str, Any]:
        return dict(self.config.get("plugin_runner", {}))

    def stage_plugins(self, stage_name: str) -> list[StagePluginConfig]:
        raw_stages = self.config.get("stages", {})
        entries = raw_stages.get(stage_name, [])
        parsed: list[StagePluginConfig] = []
        for entry in entries:
            if isinstance(entry, str):
                parsed.append(StagePluginConfig(name=entry))
            elif isinstance(entry, dict):
                name = entry.get("plugin") or entry.get("name")
                if not name:
                    raise WorkspaceConfigError(
                        f"Stage entry in {stage_name!r} is missing 'plugin'."
                    )
                parsed.append(
                    StagePluginConfig(name=str(name), params=dict(entry.get("params", {})))
                )
            else:
                raise WorkspaceConfigError(
                    f"Stage entry in {stage_name!r} must be a plugin name or object."
                )
        return parsed

    def load_orchestrator_prompt(self) -> str:
        prompt_path = self.root / self.config.get(
            "orchestrator_prompt", "orchestrator.system_prompt"
        )
        relative_prompt = self.config.get("orchestrator_prompt", "orchestrator.system_prompt")
        if not self.storage.is_file(relative_prompt):
            raise WorkspaceConfigError(f"Orchestrator prompt not found: {prompt_path}")
        return self.storage.read_text(relative_prompt)

    def load_env(self) -> None:
        env_path = self.root / ".env"
        if env_path.is_file():
            load_dotenv(env_path, override=False)


def workspace_template_path() -> Path:
    return Path(str(resources.files("agentic_canvas").joinpath("workspace_template")))


def init_workspace(
    target: str | Path,
    *,
    force: bool = False,
    template: str | Path | None = None,
) -> Path:
    """Copy the workspace template into ``target``."""

    target_path = Path(target).resolve()
    source = Path(template).resolve() if template else workspace_template_path()

    if target_path.exists() and any(target_path.iterdir()):
        if not force:
            raise FileExistsError(
                f"{target_path} is not empty. Pass force=True to replace template files."
            )
        shutil.rmtree(target_path)

    storage = FileSystemWorkspaceStorage(target_path)
    storage.copy_tree(
        source,
        ".",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo"),
    )
    storage.ensure_dir("runs")
    storage.ensure_dir("artifacts")
    storage.ensure_dir("logs")
    return target_path
