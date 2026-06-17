# Agentic Canvas

## Core Shape

- Kernel is a small privileged facade that wires workspace config, providers,
  registries, stores, and plugin runners.
- Run lifecycle behavior is split by concern: persisted state, context/trace
  stores, stage/orchestrator execution, decisions, and live plugin requests.
- Plugin execution is split into protocol/result models, runner contracts, and
  subprocess transport.
- Registries share manifest-directory loading while retaining explicit plugin
  and library manifest types.
- Plugins are the only capability units.
- Libraries are reusable code for plugins and are not directly orchestrated.
- Workspace template supplies the fundamental plugins and libraries.
- Orchestrator has exactly one tool: `call_plugin`.

## Quick Start

```powershell
uv sync
uv run agentic-canvas init .\workspace
uv run agentic-canvas run .\workspace "call_plugin plugin_catalog {}"
```

When a plugin calls `await_user("...")`, the CLI prompts immediately and the
answer is returned to that same live plugin invocation.

When running directly from the source tree without installing:

```powershell
uv run python -m agentic_canvas init .\workspace
```

The default provider is local and deterministic. Configure Gemini in
`workspace.json` or `.env`:

```text
AGENTIC_CANVAS_PROVIDER=gemini
GEMINI_API_KEY=...
GEMINI_MODEL=gemini-2.5-flash
```

Or configure an OpenAI-compatible provider:

```text
AGENTIC_CANVAS_PROVIDER=openai_compatible
OPENAI_API_KEY=...
OPENAI_MODEL=gpt-4.1-mini
OPENAI_BASE_URL=https://api.openai.com/v1
```

## Tests

Run the contract tests with pytest:

```powershell
$env:UV_CACHE_DIR = (Resolve-Path ".uv-cache-run").Path
uv run pytest -q
```

The repo-local uv cache avoids machine-level cache issues and keeps subprocess
plugin tests deterministic across workspaces.

## Workspace LLMs And Agents

Workspace agents are ordinary Python objects used inside plugins. Providers are
independent libraries that implement `libs.llm_core.LLMProvider`; plugins import
and construct the provider they want directly.

```python
from libs.agent_runtime import Agent
from libs.llm_anthropic import AnthropicProvider


def lookup(topic: str) -> dict:
    """Look up one topic in this plugin's data source."""
    return {"topic": topic}


provider = AnthropicProvider(model="your-model-id")
agent = Agent(provider=provider, tools=[lookup], max_turns=8)
answer = agent.run("Investigate the workspace architecture.")
```

Available adapters are `llm_google.GoogleProvider`,
`llm_openai.OpenAIProvider`, and `llm_anthropic.AnthropicProvider`. Model IDs
come from constructor arguments or `GEMINI_MODEL`, `OPENAI_MODEL`, and
`ANTHROPIC_MODEL`. The corresponding API keys are read from the standard
provider environment variables.

Only callables explicitly supplied to `Agent` are model-callable. Use
`libs.plugin_calls.plugin_tool(...)` when an agent should call a selected
workspace plugin through the existing plugin runner protocol.

Plugins are trusted code in this release. They can access the filesystem,
network, environment credentials, and provider SDKs directly. Kernel-mediated
privileged services, streaming, async tools, retries, caching, budgets, and
parallel tool execution remain future work.
