from __future__ import annotations

import argparse
from pathlib import Path

import uvicorn

from agentic_canvas_api.app import create_app


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="agentic-canvas-api")
    parser.add_argument("workspace", type=Path)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args(argv)

    app = create_app(args.workspace)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
