from __future__ import annotations

import json
import shlex
from dataclasses import dataclass
from typing import Any

from libs.plugin_calls import call_plugin
from libs.plugin_runtime import (
    ensure_safe_name,
    load_plugin_manifest,
    params,
    workspace_root,
    workspace_storage,
)


COMMANDS_DIR = "commands"
COMMAND_PREFIX = "/"


@dataclass(frozen=True)
class CommandMatch:
    """Resolved command pack entry for one user input."""

    raw_input: str
    pack_name: str
    command_name: str
    alias: str
    args: list[str]
    command: dict[str, Any]


def list_command_packs(request: dict[str, Any], *, query: str = "") -> dict[str, Any]:
    """List command packs and configured commands."""

    storage = workspace_storage(request)
    packs = []
    query_text = query.lower()
    if not storage.is_dir(COMMANDS_DIR):
        return {"command_packs": [], "count": 0}
    for entry in storage.list_dir(COMMANDS_DIR):
        manifest_path = f"{entry['path']}/manifest.json"
        if not entry["is_dir"] or not storage.is_file(manifest_path):
            continue
        manifest = validate_command_pack_manifest(storage.read_json(manifest_path), request=request)
        searchable = " ".join(
            [
                manifest["name"],
                manifest["summary"],
                manifest["documentation"],
                json.dumps(manifest["commands"], ensure_ascii=False),
            ]
        ).lower()
        if query_text and query_text not in searchable:
            continue
        packs.append(
            {
                "name": manifest["name"],
                "version": manifest["version"],
                "summary": manifest["summary"],
                "commands": [
                    {
                        "name": command["name"],
                        "aliases": command["aliases"],
                        "plugin": command["plugin"],
                    }
                    for command in manifest["commands"]
                ],
                "path": str(storage.resolve_path(entry["path"])),
            }
        )
    return {"command_packs": packs, "count": len(packs)}


def create_command_pack(
    request: dict[str, Any],
    manifest: dict[str, Any],
    *,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Create or replace a command pack manifest."""

    storage = workspace_storage(request)
    validated = validate_command_pack_manifest(manifest, request=request)
    pack_dir = f"{COMMANDS_DIR}/{validated['name']}"
    if storage.exists(pack_dir):
        if not overwrite:
            raise FileExistsError(f"Command pack {validated['name']!r} already exists.")
        storage.replace_tree(pack_dir)
    else:
        storage.ensure_dir(pack_dir)
    storage.write_json(f"{pack_dir}/manifest.json", validated)
    return {
        "created": True,
        "command_pack": validated["name"],
        "path": str(storage.resolve_path(pack_dir)),
        "manifest": validated,
    }


def remove_command_pack(request: dict[str, Any], name: str) -> dict[str, Any]:
    """Remove a command pack directory."""

    storage = workspace_storage(request)
    safe_name = ensure_safe_name(name, "command pack name")
    pack_dir = f"{COMMANDS_DIR}/{safe_name}"
    if not storage.exists(pack_dir):
        raise FileNotFoundError(f"Command pack {safe_name!r} does not exist.")
    storage.delete_tree(pack_dir)
    return {"removed": True, "command_pack": safe_name}


def match_command(request: dict[str, Any], *, prefix: str = COMMAND_PREFIX) -> dict[str, Any]:
    """Match the current run input to a configured command."""

    context = request.get("context", {})
    text = str(context.get("input") or "").strip()
    if not text.startswith(prefix):
        return {"is_command": False, "match": None}

    matches = _configured_matches(request, text)
    if not matches:
        raise ValueError(f"Unknown command: {text}")
    match = sorted(matches, key=lambda item: len(item.alias), reverse=True)[0]
    return {
        "is_command": True,
        "match": {
            "raw_input": match.raw_input,
            "pack_name": match.pack_name,
            "command_name": match.command_name,
            "alias": match.alias,
            "args": match.args,
            "command": match.command,
        },
    }


def params_for_match(match: dict[str, Any]) -> dict[str, Any]:
    """Map matched command args into target plugin params."""

    command = match["command"]
    mapped = dict(command.get("params") or {})
    args = list(match.get("args") or [])
    arguments = list(command.get("arguments") or [])
    if len(args) > len(arguments):
        raise ValueError(f"Command {command['name']!r} received too many arguments.")
    for index, argument in enumerate(arguments):
        name = str(argument.get("name") or "")
        if not name:
            raise ValueError(f"Command {command['name']!r} has an unnamed argument.")
        if index < len(args):
            mapped[name] = _coerce_arg(args[index], str(argument.get("type") or "string"), name)
        elif argument.get("required", False):
            raise ValueError(f"Command {command['name']!r} is missing required argument {name!r}.")
        elif "default" in argument:
            mapped[name] = argument["default"]
    return mapped


def validate_command_pack_manifest(
    manifest: dict[str, Any],
    *,
    request: dict[str, Any],
) -> dict[str, Any]:
    """Validate and normalize one command pack manifest."""

    required = {"name", "version", "summary", "documentation", "commands"}
    missing = sorted(required - set(manifest))
    if missing:
        raise ValueError(f"Command pack manifest missing required fields: {missing}")
    name = ensure_safe_name(str(manifest["name"]), "command pack name")
    commands = manifest["commands"]
    if not isinstance(commands, list):
        raise ValueError("Command pack manifest field 'commands' must be a list.")
    normalized_commands = [
        _normalize_command(command, request=request)
        for command in commands
    ]
    return {
        "name": name,
        "version": str(manifest["version"]),
        "summary": str(manifest["summary"]),
        "documentation": str(manifest["documentation"]),
        "commands": normalized_commands,
    }


def _configured_matches(request: dict[str, Any], text: str) -> list[CommandMatch]:
    matches = []
    storage = workspace_storage(request)
    if not storage.is_dir(COMMANDS_DIR):
        return []
    for entry in storage.list_dir(COMMANDS_DIR):
        manifest_path = f"{entry['path']}/manifest.json"
        if not entry["is_dir"] or not storage.is_file(manifest_path):
            continue
        manifest = validate_command_pack_manifest(storage.read_json(manifest_path), request=request)
        for command in manifest["commands"]:
            for alias in command["aliases"]:
                args = _remaining_args(text, alias)
                if args is None:
                    continue
                matches.append(
                    CommandMatch(
                        raw_input=text,
                        pack_name=manifest["name"],
                        command_name=command["name"],
                        alias=alias,
                        args=args,
                        command=command,
                    )
                )
    return matches


def _normalize_command(command: dict[str, Any], *, request: dict[str, Any]) -> dict[str, Any]:
    required = {"name", "aliases", "plugin"}
    missing = sorted(required - set(command))
    if missing:
        raise ValueError(f"Command entry missing required fields: {missing}")
    aliases = command["aliases"]
    if not isinstance(aliases, list) or not aliases:
        raise ValueError(f"Command {command.get('name')!r} must declare at least one alias.")
    plugin = ensure_safe_name(str(command["plugin"]), "plugin name")
    load_plugin_manifest(workspace_root(request), plugin)
    params_value = command.get("params") or {}
    if not isinstance(params_value, dict):
        raise ValueError(f"Command {command.get('name')!r} params must be an object.")
    arguments = command.get("arguments") or []
    if not isinstance(arguments, list):
        raise ValueError(f"Command {command.get('name')!r} arguments must be a list.")
    return {
        "name": str(command["name"]),
        "aliases": [str(alias).strip() for alias in aliases if str(alias).strip()],
        "plugin": plugin,
        "params": params_value,
        "arguments": [_normalize_argument(argument, command_name=str(command["name"])) for argument in arguments],
        "decision": str(command.get("decision") or "stop"),
    }


def _normalize_argument(argument: dict[str, Any], *, command_name: str) -> dict[str, Any]:
    if not isinstance(argument, dict):
        raise ValueError(f"Command {command_name!r} argument definitions must be objects.")
    name = ensure_safe_name(str(argument.get("name") or ""), "command argument name")
    return {
        "name": name,
        "type": str(argument.get("type") or "string"),
        "required": bool(argument.get("required", False)),
        **({"default": argument["default"]} if "default" in argument else {}),
    }


def _remaining_args(text: str, alias: str) -> list[str] | None:
    normalized_text = text.strip()
    normalized_alias = alias.strip()
    if normalized_text == normalized_alias:
        return []
    if not normalized_text.startswith(normalized_alias + " "):
        return None
    raw_args = normalized_text[len(normalized_alias):].strip()
    try:
        return shlex.split(raw_args)
    except ValueError:
        return raw_args.split()


def _coerce_arg(value: str, kind: str, name: str) -> Any:
    if kind == "string":
        return value
    if kind == "integer":
        return int(value)
    if kind == "number":
        return float(value)
    if kind == "boolean":
        lowered = value.lower()
        if lowered in {"true", "yes", "1", "on"}:
            return True
        if lowered in {"false", "no", "0", "off"}:
            return False
        raise ValueError(f"Argument {name!r} expects a boolean.")
    raise ValueError(f"Unsupported argument type {kind!r} for {name!r}.")

