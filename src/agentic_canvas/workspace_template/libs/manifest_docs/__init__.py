from __future__ import annotations

import ast
from pathlib import Path
from typing import Any


TEMPLATE_DIR = Path(__file__).resolve().parent / "templates"


def generate_plugin_documentation(
    plugin_name: str,
    summary: str,
    entry_point: str,
    input_schema: dict[str, Any],
    output_schema: dict[str, Any],
    libraries: list[str],
    requirements: list[str],
    tags: list[str],
) -> str:
    """Render structured plugin manifest documentation from schema metadata."""

    template = _read_template("plugin_documentation.md")
    return _render_template(
        template,
        {
            "summary": summary.strip() or f"{plugin_name} plugin.",
            "entry_point": entry_point,
            "input_schema": "\n".join(describe_schema(input_schema)),
            "output_schema": "\n".join(describe_schema(output_schema)),
            "libraries": _join_or_none(libraries),
            "requirements": _join_or_none(requirements),
            "tags": _join_or_none(tags),
        },
    )


def generate_library_documentation(
    library_dir: str | Path,
    library_name: str,
    summary: str,
    exports: list[str],
) -> str:
    """Render structured library manifest documentation from exported Python API items."""

    items = collect_library_api_items(Path(library_dir))
    sections = []
    if not exports:
        sections.append("No exported API items declared.")
    for export in exports:
        sections.append(_api_reference_section(export, items.get(export)))
    template = _read_template("library_documentation.md")
    return _render_template(
        template,
        {
            "summary": summary.strip() or f"{library_name} library.",
            "api_reference": "\n\n".join(sections),
        },
    )


def describe_schema(schema: dict[str, Any]) -> list[str]:
    """Describe a JSON-object schema in manifest documentation format."""

    if not schema:
        return ["Type: object", "Properties: none declared."]

    lines = [f"Type: {schema.get('type', 'object')}"]
    required = [str(item) for item in schema.get("required", [])]
    lines.append(f"Required: {', '.join(required) if required else 'none'}")

    properties = schema.get("properties") or {}
    if not isinstance(properties, dict) or not properties:
        lines.append("Properties: none declared.")
        return lines

    lines.append("Properties:")
    for property_name, definition in properties.items():
        if not isinstance(definition, dict):
            lines.append(f"- {property_name}: unspecified")
            continue
        kind = definition.get("type", "unspecified")
        details = [str(kind)]
        if "enum" in definition:
            details.append("one of " + ", ".join(str(item) for item in definition["enum"]))
        if definition.get("description"):
            details.append(str(definition["description"]))
        lines.append(f"- {property_name}: {'; '.join(details)}")
    return lines


def collect_library_api_items(library_dir: Path) -> dict[str, dict[str, str]]:
    """Collect top-level Python functions, classes, and values for library docs."""

    items: dict[str, dict[str, str]] = {}
    for path in _library_python_files(library_dir):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        source = str(path.relative_to(library_dir)).replace("\\", "/")
        for node in tree.body:
            api_item = _api_item_from_node(node, source)
            if api_item is not None and api_item["name"] not in items:
                items[api_item["name"]] = api_item
    return items


def _api_reference_section(export: str, item: dict[str, str] | None) -> str:
    lines = [f"### {export}"]
    if item is None:
        lines.append("Type: missing")
        lines.append("Documentation: Missing exported API item in library code.")
        return "\n".join(lines)
    lines.append(f"Type: {item['type']}")
    lines.append(f"Source: {item['source']}")
    if item.get("signature"):
        lines.append(f"Signature: {item['signature']}")
    lines.append("")
    lines.append(str(item.get("docstring") or "").strip() or "Documentation: Missing docstring.")
    return "\n".join(lines)


def _read_template(name: str) -> str:
    return (TEMPLATE_DIR / name).read_text(encoding="utf-8")


def _render_template(template: str, values: dict[str, str]) -> str:
    rendered = template.format(**values)
    return rendered.rstrip() + "\n"


def _join_or_none(values: list[str]) -> str:
    return ", ".join(values) if values else "none"


def _library_python_files(library_dir: Path) -> list[Path]:
    files = sorted(library_dir.rglob("*.py"), key=lambda path: path.as_posix())
    return sorted(files, key=lambda path: 0 if path.name == "__init__.py" else 1)


def _api_item_from_node(node: ast.AST, source: str) -> dict[str, str] | None:
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return {
            "name": node.name,
            "type": "function",
            "source": source,
            "signature": _function_signature(node),
            "docstring": ast.get_docstring(node) or "",
        }
    if isinstance(node, ast.ClassDef):
        return {
            "name": node.name,
            "type": "class",
            "source": source,
            "signature": _class_signature(node),
            "docstring": ast.get_docstring(node) or "",
        }
    value_name = _value_name_from_node(node)
    if value_name is not None:
        return {
            "name": value_name,
            "type": "value",
            "source": source,
            "signature": "",
            "docstring": "",
        }
    return None


def _function_signature(node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    prefix = "async " if isinstance(node, ast.AsyncFunctionDef) else ""
    return_annotation = ""
    if node.returns is not None:
        return_annotation = f" -> {ast.unparse(node.returns)}"
    return f"{prefix}{node.name}({_format_arguments(node.args)}){return_annotation}"


def _class_signature(node: ast.ClassDef) -> str:
    bases = [ast.unparse(base) for base in node.bases]
    keywords = [
        f"{keyword.arg}={ast.unparse(keyword.value)}"
        for keyword in node.keywords
        if keyword.arg is not None
    ]
    args = ", ".join([*bases, *keywords])
    return f"class {node.name}({args})" if args else f"class {node.name}"


def _format_arguments(arguments: ast.arguments) -> str:
    parts: list[str] = []
    positional = [*arguments.posonlyargs, *arguments.args]
    defaults = [None] * (len(positional) - len(arguments.defaults)) + list(arguments.defaults)
    for index, (arg, default) in enumerate(zip(positional, defaults)):
        if index == len(arguments.posonlyargs) and arguments.posonlyargs:
            parts.append("/")
        parts.append(_format_arg(arg, default))
    if arguments.vararg is not None:
        parts.append("*" + _format_arg(arguments.vararg, None))
    elif arguments.kwonlyargs:
        parts.append("*")
    for arg, default in zip(arguments.kwonlyargs, arguments.kw_defaults):
        parts.append(_format_arg(arg, default))
    if arguments.kwarg is not None:
        parts.append("**" + _format_arg(arguments.kwarg, None))
    return ", ".join(parts)


def _format_arg(arg: ast.arg, default: ast.AST | None) -> str:
    value = arg.arg
    if arg.annotation is not None:
        value += f": {ast.unparse(arg.annotation)}"
    if default is not None:
        value += f" = {ast.unparse(default)}"
    return value


def _value_name_from_node(node: ast.AST) -> str | None:
    if isinstance(node, ast.Assign):
        for target in node.targets:
            if isinstance(target, ast.Name):
                return target.id
    if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
        return node.target.id
    return None
