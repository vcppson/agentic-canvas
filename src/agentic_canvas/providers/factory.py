from __future__ import annotations

import os

from agentic_canvas.kernel.workspace import Workspace
from agentic_canvas.providers.base import LLMProvider
from agentic_canvas.providers.gemini import GeminiProvider
from agentic_canvas.providers.local import LocalProvider
from agentic_canvas.providers.openai_compatible import OpenAICompatibleProvider


def provider_from_workspace(workspace: Workspace) -> LLMProvider:
    workspace.load_env()
    config = workspace.provider_config
    provider_type = (
        os.getenv("AGENTIC_CANVAS_PROVIDER")
        or config.get("type")
        or "local"
    ).lower()

    if provider_type in {"gemini", "google", "google_gemini"}:
        api_key = (
            os.getenv("GEMINI_API_KEY")
            or os.getenv("GOOGLE_API_KEY")
            or os.getenv("AGENTIC_CANVAS_API_KEY")
            or config.get("api_key")
        )
        model = (
            os.getenv("GEMINI_MODEL")
            or os.getenv("AGENTIC_CANVAS_MODEL")
            or config.get("model")
            or "gemini-2.5-flash"
        )
        if not api_key:
            return LocalProvider("Gemini provider selected, but GEMINI_API_KEY is not set.")
        return GeminiProvider(api_key=str(api_key), model=str(model))

    if provider_type in {"openai", "openai_compatible"}:
        api_key = (
            os.getenv("OPENAI_API_KEY")
            or os.getenv("AGENTIC_CANVAS_API_KEY")
            or config.get("api_key")
        )
        model = (
            os.getenv("OPENAI_MODEL")
            or os.getenv("AGENTIC_CANVAS_MODEL")
            or config.get("model")
            or "gpt-4.1-mini"
        )
        base_url = (
            os.getenv("OPENAI_BASE_URL")
            or os.getenv("AGENTIC_CANVAS_BASE_URL")
            or config.get("base_url")
            or "https://api.openai.com/v1"
        )
        if not api_key:
            return LocalProvider("OpenAI-compatible provider selected, but no API key is set.")
        return OpenAICompatibleProvider(api_key=str(api_key), model=str(model), base_url=str(base_url))

    return LocalProvider()
