from __future__ import annotations

import contextlib
import json
import os
import queue
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

from agentic_canvas.kernel.context import RunContext
from agentic_canvas.kernel.plugin_protocol import (
    PluginCallHandler,
    PluginCallRequest,
    PluginExecutionResult,
    PluginInputHandler,
    PluginInputRequest,
)
from agentic_canvas.kernel.plugin_runner_base import PluginRunner
from agentic_canvas.kernel.workspace import Workspace
from agentic_canvas.libraries.manifest import LibraryManifestError
from agentic_canvas.libraries.registry import LibraryRegistry
from agentic_canvas.plugins.manifest import PluginManifest


class SubprocessPluginRunner(PluginRunner):
    """Default plugin runner: execute plugins in an isolated Python subprocess."""

    def __init__(self, *, timeout_seconds: int = 120) -> None:
        self.timeout_seconds = timeout_seconds

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
        plugin_dir = workspace.plugins_dir / manifest.name
        plugin_params = params or {}
        try:
            requirements = self._requirements_for(workspace, manifest)
        except LibraryManifestError as exc:
            return PluginExecutionResult.transport_error(
                plugin_name=manifest.name,
                mode=mode,
                message=f"Plugin library resolution failed: {exc}",
            )

        request = {
            "protocol_version": 1,
            "mode": mode,
            "plugin": manifest.to_dict(),
            "params": plugin_params,
            "context": context.to_plugin_context(),
            "workspace_root": str(workspace.root),
        }

        source_root = Path(__file__).resolve().parents[2]
        env = dict(os.environ)
        existing_pythonpath = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = os.pathsep.join(
            part for part in [str(source_root), str(workspace.root), existing_pythonpath] if part
        )
        env["WORKSPACE_ROOT"] = str(workspace.root)
        env["AGENTIC_CANVAS_PLUGIN"] = manifest.name

        cmd = self._command_for(manifest, plugin_dir, requirements)
        if cmd is None:
            return PluginExecutionResult.transport_error(
                plugin_name=manifest.name,
                mode=mode,
                message=(
                    f"Plugin {manifest.name!r} resolves external requirements, "
                    "but 'uv' is not available to create the subprocess environment."
                ),
            )

        started = time.monotonic()
        stdout_lines: queue.Queue[str | None] = queue.Queue()
        stderr_lines: list[str] = []
        try:
            proc = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                cwd=str(plugin_dir),
                env=env,
                bufsize=1,
            )
        except subprocess.TimeoutExpired:
            return PluginExecutionResult.transport_error(
                plugin_name=manifest.name,
                mode=mode,
                message=f"Plugin {manifest.name!r} timed out after {self.timeout_seconds} seconds.",
            )
        except OSError as exc:
            return PluginExecutionResult.transport_error(
                plugin_name=manifest.name,
                mode=mode,
                message=f"Plugin process could not start: {exc}",
            )

        assert proc.stdin is not None
        assert proc.stdout is not None
        assert proc.stderr is not None
        stdout_thread = threading.Thread(
            target=self._read_stdout,
            args=(proc.stdout, stdout_lines),
            daemon=True,
        )
        stderr_thread = threading.Thread(
            target=self._read_stderr,
            args=(proc.stderr, stderr_lines),
            daemon=True,
        )
        stdout_thread.start()
        stderr_thread.start()

        proc.stdin.write(json.dumps(request, ensure_ascii=False, default=str) + "\n")
        proc.stdin.flush()

        input_requests: list[dict[str, Any]] = []
        plugin_call_requests: list[dict[str, Any]] = []
        final_payload: dict[str, Any] | None = None
        while final_payload is None:
            duration = time.monotonic() - started
            if duration > self.timeout_seconds:
                proc.kill()
                return PluginExecutionResult.transport_error(
                    plugin_name=manifest.name,
                    mode=mode,
                    message=f"Plugin {manifest.name!r} timed out after {self.timeout_seconds} seconds.",
                    stderr="".join(stderr_lines),
                    input_requests=input_requests,
                    plugin_call_requests=plugin_call_requests,
                    duration_seconds=duration,
                )

            try:
                line = stdout_lines.get(timeout=0.1)
            except queue.Empty:
                if proc.poll() is not None and stdout_lines.empty():
                    break
                continue

            if line is None:
                break

            stdout = line.strip()
            if not stdout:
                continue

            try:
                payload = json.loads(stdout)
            except json.JSONDecodeError:
                proc.kill()
                return PluginExecutionResult.transport_error(
                    plugin_name=manifest.name,
                    mode=mode,
                    message="Plugin returned invalid JSON protocol output.",
                    stderr="".join(stderr_lines),
                    raw={"stdout": stdout},
                    input_requests=input_requests,
                    plugin_call_requests=plugin_call_requests,
                    duration_seconds=time.monotonic() - started,
                )

            if payload.get("type") == "input_request":
                request_record = self._handle_input_request(
                    proc,
                    manifest=manifest,
                    mode=mode,
                    params=plugin_params,
                    payload=payload,
                    input_handler=input_handler,
                    input_requests=input_requests,
                    stderr="".join(stderr_lines),
                    started=started,
                )
                if isinstance(request_record, PluginExecutionResult):
                    return request_record
                continue

            if payload.get("type") == "plugin_call_request":
                request_record = self._handle_plugin_call_request(
                    proc,
                    manifest=manifest,
                    mode=mode,
                    payload=payload,
                    plugin_call_handler=plugin_call_handler,
                    plugin_call_requests=plugin_call_requests,
                    stderr="".join(stderr_lines),
                    started=started,
                )
                if isinstance(request_record, PluginExecutionResult):
                    return request_record
                continue

            final_payload = payload

        duration = time.monotonic() - started
        try:
            return_code = proc.wait(timeout=1)
        except subprocess.TimeoutExpired:
            proc.kill()
            return_code = proc.wait()
        stdout_thread.join(timeout=1)
        stderr_thread.join(timeout=1)
        stderr = "".join(stderr_lines)

        if return_code != 0:
            return PluginExecutionResult.transport_error(
                plugin_name=manifest.name,
                mode=mode,
                message=f"Plugin process exited with code {return_code}.",
                stderr=stderr,
                input_requests=input_requests,
                plugin_call_requests=plugin_call_requests,
                duration_seconds=duration,
            )

        if final_payload is None:
            return PluginExecutionResult.transport_error(
                plugin_name=manifest.name,
                mode=mode,
                message="Plugin produced no protocol output.",
                stderr=stderr,
                input_requests=input_requests,
                plugin_call_requests=plugin_call_requests,
                duration_seconds=duration,
            )

        return PluginExecutionResult.from_transport(
            plugin_name=manifest.name,
            mode=mode,
            payload=final_payload,
            stderr=stderr,
            duration_seconds=duration,
            input_requests=input_requests,
            plugin_call_requests=plugin_call_requests,
        )

    def _handle_input_request(
        self,
        proc: subprocess.Popen[str],
        *,
        manifest: PluginManifest,
        mode: str,
        params: dict[str, Any],
        payload: dict[str, Any],
        input_handler: PluginInputHandler | None,
        input_requests: list[dict[str, Any]],
        stderr: str,
        started: float,
    ) -> PluginExecutionResult | None:
        request = PluginInputRequest(
            plugin_name=manifest.name,
            mode=mode,
            request_id=str(payload.get("request_id") or ""),
            message=str(payload.get("message") or ""),
            patch=dict(payload.get("patch") or {}),
            params=params,
            raw=payload,
        )
        record = {
            "request_id": request.request_id,
            "message": request.message,
            "patch": request.patch,
            "status": "waiting_for_user_input",
        }
        input_requests.append(record)
        if input_handler is None:
            proc.kill()
            with contextlib.suppress(Exception):
                proc.wait(timeout=1)
            return PluginExecutionResult.transport_error(
                plugin_name=manifest.name,
                mode=mode,
                message="Plugin requested user input, but no input provider is configured.",
                stderr=stderr,
                input_requests=input_requests,
                duration_seconds=time.monotonic() - started,
            )

        try:
            answer = input_handler(request)
        except Exception as exc:
            proc.kill()
            with contextlib.suppress(Exception):
                proc.wait(timeout=1)
            record["error"] = str(exc)
            return PluginExecutionResult.transport_error(
                plugin_name=manifest.name,
                mode=mode,
                message=f"Plugin input request failed: {exc}",
                stderr=stderr,
                input_requests=input_requests,
                duration_seconds=time.monotonic() - started,
            )

        record["response"] = answer
        record["status"] = "answered"
        assert proc.stdin is not None
        proc.stdin.write(
            json.dumps(
                {
                    "type": "input_response",
                    "request_id": request.request_id,
                    "value": answer,
                },
                ensure_ascii=False,
                default=str,
            )
            + "\n"
        )
        proc.stdin.flush()
        return None

    def _handle_plugin_call_request(
        self,
        proc: subprocess.Popen[str],
        *,
        manifest: PluginManifest,
        mode: str,
        payload: dict[str, Any],
        plugin_call_handler: PluginCallHandler | None,
        plugin_call_requests: list[dict[str, Any]],
        stderr: str,
        started: float,
    ) -> PluginExecutionResult | None:
        params = payload.get("params") or {}
        if not isinstance(params, dict):
            params = {"value": params}
        metadata = payload.get("metadata") or {}
        if not isinstance(metadata, dict):
            metadata = {}
        request = PluginCallRequest(
            source_plugin_name=manifest.name,
            source_mode=mode,
            request_id=str(payload.get("request_id") or ""),
            plugin_name=str(payload.get("plugin") or ""),
            params=params,
            metadata=metadata,
            raw=payload,
        )
        record = {
            "request_id": request.request_id,
            "plugin": request.plugin_name,
            "params": request.params,
            "metadata": request.metadata,
            "status": "calling_plugin",
        }
        plugin_call_requests.append(record)
        if plugin_call_handler is None:
            proc.kill()
            with contextlib.suppress(Exception):
                proc.wait(timeout=1)
            return PluginExecutionResult.transport_error(
                plugin_name=manifest.name,
                mode=mode,
                message="Plugin requested a plugin call, but no plugin call handler is configured.",
                stderr=stderr,
                plugin_call_requests=plugin_call_requests,
                duration_seconds=time.monotonic() - started,
            )

        try:
            response = plugin_call_handler(request)
        except Exception as exc:
            proc.kill()
            with contextlib.suppress(Exception):
                proc.wait(timeout=1)
            record["error"] = str(exc)
            return PluginExecutionResult.transport_error(
                plugin_name=manifest.name,
                mode=mode,
                message=f"Plugin call request failed: {exc}",
                stderr=stderr,
                plugin_call_requests=plugin_call_requests,
                duration_seconds=time.monotonic() - started,
            )

        record["response"] = response
        record["status"] = "answered"
        assert proc.stdin is not None
        proc.stdin.write(
            json.dumps(
                {
                    "type": "plugin_call_response",
                    "request_id": request.request_id,
                    "value": response,
                },
                ensure_ascii=False,
                default=str,
            )
            + "\n"
        )
        proc.stdin.flush()
        return None

    def _command_for(
        self,
        manifest: PluginManifest,
        plugin_dir: Path,
        requirements: list[str],
    ) -> list[str] | None:
        bootloader_args = [
            "-m",
            "agentic_canvas.kernel.plugin_bootloader",
            str(plugin_dir),
            manifest.entry_point,
        ]
        if not requirements:
            return [sys.executable, *bootloader_args]

        uv = shutil.which("uv")
        if not uv:
            return None

        requirement_args: list[str] = []
        for requirement in ["python-dotenv", *requirements]:
            requirement_args.extend(["--with", requirement])
        return [uv, "run", *requirement_args, "python", *bootloader_args]

    @staticmethod
    def _requirements_for(
        workspace: Workspace,
        manifest: PluginManifest,
    ) -> list[str]:
        library_registry = LibraryRegistry(
            workspace.storage,
            workspace.config.get("libraries_dir", "libs"),
        )
        library_requirements = library_registry.requirements_for(manifest.libraries)
        return sorted(
            {
                requirement.strip()
                for requirement in [*manifest.requirements, *library_requirements]
                if requirement.strip()
            },
            key=str.casefold,
        )

    @staticmethod
    def _read_stdout(stream: Any, output: queue.Queue[str | None]) -> None:
        try:
            for line in stream:
                output.put(line)
        finally:
            output.put(None)

    @staticmethod
    def _read_stderr(stream: Any, output: list[str]) -> None:
        for line in stream:
            output.append(line)
