from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from agentic_canvas.kernel.plugin_runner import PluginInputRequest
from agentic_canvas.kernel.run import Kernel
from agentic_canvas.kernel.trigger import Trigger
from agentic_canvas.kernel.workspace import init_workspace


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="agentic-canvas")
    subparsers = parser.add_subparsers(dest="command", required=True)

    init_parser = subparsers.add_parser("init", help="Initialize a workspace from the template.")
    init_parser.add_argument("workspace", type=Path)
    init_parser.add_argument("--force", action="store_true")

    run_parser = subparsers.add_parser("run", help="Start a run from user input.")
    run_parser.add_argument("workspace", type=Path)
    run_parser.add_argument("input", nargs="+")
    run_parser.add_argument("--json", action="store_true", dest="as_json")

    args = parser.parse_args(argv)

    if args.command == "init":
        path = init_workspace(args.workspace, force=args.force)
        print(f"Initialized workspace: {path}")
        return

    if args.command == "run":
        kernel = Kernel(args.workspace, input_provider=_prompt_for_plugin_input)
        context = kernel.start(Trigger(user_input=" ".join(args.input)))
        _print_context(context.to_dict(), as_json=args.as_json)
        return


def _print_context(context: dict[str, Any], *, as_json: bool) -> None:
    if as_json:
        print(json.dumps(context, indent=2, ensure_ascii=False))
        return

    print(f"run_id: {context['run_id']}")
    print(f"status: {context['status']}")
    print(f"decision: {context['decision']}")
    if context.get("awaiting"):
        print(f"awaiting: {context['awaiting'].get('message', '')}")
    if context.get("final_response") is not None:
        print()
        print(context["final_response"])


def _prompt_for_plugin_input(request: PluginInputRequest) -> str:
    print()
    print(request.message)
    return input("> ")
