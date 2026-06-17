from __future__ import annotations

import contextlib
import importlib
import importlib.util
import inspect
import json
import sys
import traceback
from pathlib import Path
from typing import Any

from dotenv import load_dotenv


def main() -> None:
    plugin_dir = Path(sys.argv[1]).resolve()
    entry_point = sys.argv[2]
    try:
        raw = sys.stdin.readline()
        request = json.loads(raw) if raw.strip() else {}
        workspace_root = Path(request["workspace_root"]).resolve()

        _load_dotenv(workspace_root, plugin_dir)
        sys.path.insert(0, str(plugin_dir))
        sys.path.insert(0, str(workspace_root))

        with contextlib.redirect_stdout(sys.stderr):
            func = _load_entry(plugin_dir, entry_point)
            result = _call_entry(func, request)

        patch = _consume_run_control_patch()
        _emit({"ok": True, "kind": "result", "result": result, "patch": patch})
    except Exception as exc:
        if getattr(exc, "run_control_signal", False):
            _emit(
                {
                    "ok": True,
                    "kind": "run_control",
                    "decision": getattr(exc, "decision", "abort"),
                    "message": getattr(exc, "message", None),
                    "reason": getattr(exc, "reason", None),
                    "response": getattr(exc, "response", None),
                    "patch": getattr(exc, "patch", {}),
                }
            )
            return

        _emit(
            {
                "ok": False,
                "kind": "error",
                "message": str(exc),
                "traceback": traceback.format_exc(),
            }
        )


def _emit(payload: dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")
    sys.stdout.flush()


def _load_dotenv(workspace_root: Path, plugin_dir: Path) -> None:
    candidates = [
        workspace_root / ".env",
        workspace_root / "plugins" / ".env",
        plugin_dir / ".env",
    ]
    for path in candidates:
        if path.is_file():
            load_dotenv(path, override=True)


def _load_entry(plugin_dir: Path, entry_point: str):
    module_name, _, func_name = entry_point.partition(":")
    if not module_name or not func_name:
        raise RuntimeError("Plugin entry_point must have the form 'module.py:function'.")

    if module_name.endswith(".py"):
        module_path = (plugin_dir / module_name).resolve()
        if not module_path.is_file():
            raise RuntimeError(f"Plugin entry module not found: {module_path}")
        spec = importlib.util.spec_from_file_location(
            f"_agentic_canvas_plugin_{module_path.stem}",
            module_path,
        )
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Could not load plugin module: {module_path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
    else:
        module = importlib.import_module(module_name)

    func = getattr(module, func_name, None)
    if not callable(func):
        raise RuntimeError(f"Plugin entry function {func_name!r} is not callable.")
    return func


def _call_entry(func, request: dict[str, Any]):
    signature = inspect.signature(func)
    parameter_count = len(signature.parameters)
    if parameter_count == 0:
        return func()
    if parameter_count == 1:
        return func(request)
    if parameter_count == 2:
        return func(request.get("params", {}), request.get("context", {}))
    raise RuntimeError("Plugin entry functions may accept 0, 1, or 2 parameters.")


def _consume_run_control_patch() -> dict[str, Any]:
    try:
        run_control = importlib.import_module("libs.run_control")
    except Exception:
        return {}
    consume = getattr(run_control, "consume_patch", None)
    if not callable(consume):
        return {}
    patch = consume()
    return patch if isinstance(patch, dict) else {}


if __name__ == "__main__":
    main()
