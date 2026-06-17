from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class Trigger:
    """A kernel-level event that starts a Run."""

    user_input: str
    type: str = "user_input"
    metadata: dict[str, Any] = field(default_factory=dict)
