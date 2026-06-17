"""Agentic Canvas.

The package exposes a small privileged kernel and a workspace-composed plugin
runtime. Product behavior belongs in plugins and libraries.
"""

from __future__ import annotations

from agentic_canvas.cli.app import main

__all__ = ["__version__", "main"]

__version__ = "0.1.0"
