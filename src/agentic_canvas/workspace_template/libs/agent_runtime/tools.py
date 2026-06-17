from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from typing import Any, Callable, get_type_hints

from pydantic import BaseModel, ConfigDict, create_model

from libs.agent_runtime.models import AgentTool, ToolDefinitionError
from libs.llm_core import ToolDefinition


@dataclass
class FunctionTool:
    """Validated model tool backed by an explicitly supplied Python callable."""

    name: str
    description: str
    input_schema: dict[str, Any]
    function: Callable[..., Any]
    _argument_model: type[BaseModel] | None = field(default=None, repr=False)

    @classmethod
    def from_callable(
        cls,
        function: Callable[..., Any],
        *,
        name: str | None = None,
        description: str | None = None,
    ) -> "FunctionTool":
        """Create a validated tool definition from a function, method, or callable object."""

        if not callable(function):
            raise ToolDefinitionError(f"Tool target is not callable: {function!r}")
        if _is_async_callable(function):
            raise ToolDefinitionError("Asynchronous callables are not supported yet.")

        metadata = _tool_metadata(function)
        tool_name = name or metadata.get("name") or _callable_name(function)
        tool_description = (
            description
            or metadata.get("description")
            or inspect.getdoc(function)
            or f"Call {tool_name}."
        )
        explicit_schema = metadata.get("input_schema")
        if explicit_schema is not None:
            if not isinstance(explicit_schema, dict):
                raise ToolDefinitionError("Explicit tool input_schema must be an object.")
            return cls(
                name=str(tool_name),
                description=str(tool_description),
                input_schema=dict(explicit_schema),
                function=function,
            )

        signature = inspect.signature(function)
        hint_target = function if inspect.isfunction(function) or inspect.ismethod(function) else function.__call__
        try:
            hints = get_type_hints(hint_target)
        except Exception as exc:
            raise ToolDefinitionError(
                f"Could not resolve type hints for tool {tool_name!r}: {exc}"
            ) from exc

        fields: dict[str, tuple[Any, Any]] = {}
        for parameter in signature.parameters.values():
            if parameter.kind in {
                inspect.Parameter.VAR_POSITIONAL,
                inspect.Parameter.VAR_KEYWORD,
                inspect.Parameter.POSITIONAL_ONLY,
            }:
                raise ToolDefinitionError(
                    f"Tool {tool_name!r} has unsupported parameter {parameter.name!r} "
                    f"of kind {parameter.kind.description}."
                )
            annotation = hints.get(parameter.name, Any)
            default = ... if parameter.default is inspect.Parameter.empty else parameter.default
            fields[parameter.name] = (annotation, default)

        model_name = "".join(part.capitalize() for part in str(tool_name).split("_")) + "Arguments"
        try:
            argument_model = create_model(
                model_name or "ToolArguments",
                __config__=ConfigDict(extra="forbid"),
                **fields,
            )
            schema = argument_model.model_json_schema()
        except Exception as exc:
            raise ToolDefinitionError(
                f"Could not build argument schema for tool {tool_name!r}: {exc}"
            ) from exc
        schema.pop("title", None)
        schema.setdefault("type", "object")
        schema.setdefault("properties", {})
        schema.setdefault("additionalProperties", False)
        return cls(
            name=str(tool_name),
            description=str(tool_description),
            input_schema=schema,
            function=function,
            _argument_model=argument_model,
        )

    def invoke(self, arguments: dict[str, Any]) -> Any:
        """Validate keyword arguments and invoke the wrapped callable."""

        if not isinstance(arguments, dict):
            raise TypeError(f"Tool {self.name!r} arguments must be an object.")
        if self._argument_model is None:
            return self.function(**arguments)
        validated = self._argument_model.model_validate(arguments)
        kwargs = {
            field_name: getattr(validated, field_name)
            for field_name in self._argument_model.model_fields
        }
        return self.function(**kwargs)

    def definition(self) -> ToolDefinition:
        """Return the provider-neutral definition sent to an LLM."""

        return ToolDefinition(
            name=self.name,
            description=self.description,
            input_schema=self.input_schema,
        )


def tool(
    function: Callable[..., Any] | None = None,
    *,
    name: str | None = None,
    description: str | None = None,
) -> Callable[..., Any]:
    """Attach optional model-tool metadata to a callable."""

    def decorate(target: Callable[..., Any]) -> Callable[..., Any]:
        setattr(
            target,
            "__agent_tool__",
            {
                "name": name,
                "description": description,
            },
        )
        return target

    return decorate(function) if function is not None else decorate


def normalize_tool(item: AgentTool | Callable[..., Any]) -> AgentTool:
    if isinstance(item, AgentTool):
        return item
    return FunctionTool.from_callable(item)


def _tool_metadata(function: Callable[..., Any]) -> dict[str, Any]:
    metadata = getattr(function, "__agent_tool__", None)
    if metadata is None and not inspect.isfunction(function) and not inspect.ismethod(function):
        metadata = getattr(function.__call__, "__agent_tool__", None)
    return dict(metadata or {})


def _callable_name(function: Callable[..., Any]) -> str:
    return str(getattr(function, "__name__", function.__class__.__name__))


def _is_async_callable(function: Callable[..., Any]) -> bool:
    if inspect.iscoroutinefunction(function):
        return True
    return not inspect.isfunction(function) and inspect.iscoroutinefunction(function.__call__)
