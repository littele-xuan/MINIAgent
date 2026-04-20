"""
mcp/tools/internal/registry_ops.py
──────────────────────────────────
Governance tools for registry-backed MCP servers.

Production-oriented changes:
  - accepts declarative JSON Schema for tool arguments
  - supports safer module-based handlers in addition to inline Python
  - preserves backward compatibility with the original code-based API
"""

from __future__ import annotations

import importlib
import json
import types as py_types
from typing import Any, Callable

try:
    from jsonschema import Draft202012Validator
except Exception:  # pragma: no cover - optional dependency guard
    Draft202012Validator = None  # type: ignore[assignment]

from mcp_lib.registry.models import ToolCategory, ToolEntry, ToolStatus
from mcp_lib.registry.registry import (
    RegistryError,
    ToolAlreadyExistsError,
    ToolNotFoundError,
    ToolProtectedError,
)


def _json_argument_to_dict(value: str | dict[str, Any] | None, field_name: str, *, default: dict[str, Any] | None = None) -> dict[str, Any]:
    if value is None:
        return dict(default or {})
    if isinstance(value, dict):
        return dict(value)
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise RegistryError(f'{field_name} must be valid JSON: {exc}') from exc
    if not isinstance(parsed, dict):
        raise RegistryError(f'{field_name} must decode to a JSON object')
    return parsed


def _normalize_csv(value: str | list[str] | None) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    return [item.strip() for item in value.split(',') if item.strip()]


def _validate_json_schema(schema: dict[str, Any]) -> None:
    if not schema:
        return
    if Draft202012Validator is not None:
        Draft202012Validator.check_schema(schema)


def _load_inline_handler(code: str, callable_name: str = 'handler') -> Callable[..., Any]:
    module = py_types.ModuleType('dynamic_tool_module')
    exec(code, module.__dict__)
    fn = getattr(module, callable_name, None)
    if not callable(fn):
        raise RegistryError(f'Inline code must define a callable named {callable_name!r}')
    return fn


def _load_module_handler(module_path: str, callable_name: str = 'handler') -> Callable[..., Any]:
    module = importlib.import_module(module_path)
    fn = getattr(module, callable_name, None)
    if not callable(fn):
        raise RegistryError(f'Module {module_path!r} has no callable named {callable_name!r}')
    return fn


def _resolve_handler(
    *,
    code: str | None,
    handler_mode: str | None,
    module_path: str | None,
    callable_name: str | None,
) -> tuple[Callable[..., Any], dict[str, Any]]:
    callable_name = callable_name or 'handler'
    handler_mode = (handler_mode or ('python_module' if module_path else 'python_inline')).strip()

    if handler_mode == 'python_module':
        if not module_path:
            raise RegistryError('module_path is required when handler_mode=python_module')
        return _load_module_handler(module_path, callable_name), {
            'handler_mode': handler_mode,
            'module_path': module_path,
            'callable_name': callable_name,
        }

    if handler_mode == 'python_inline':
        if not code or not code.strip():
            raise RegistryError('code is required when handler_mode=python_inline')
        return _load_inline_handler(code, callable_name), {
            'handler_mode': handler_mode,
            'callable_name': callable_name,
        }

    raise RegistryError('handler_mode must be one of: python_inline, python_module')


# ══════════════════════════════════════════════════════════════════════════════
# Tool handlers
# Each function returns a JSON-serializable dict/string consumed by MCP
# ══════════════════════════════════════════════════════════════════════════════

def tool_list(registry, query: str = '') -> dict[str, Any]:
    tools = registry.search(query) if query else registry.list_tools(enabled_only=False)
    return {
        'count': len(tools),
        'tools': [t.to_dict(include_history=False) for t in tools],
    }


def tool_get(registry, name: str, include_history: bool = True) -> dict[str, Any]:
    entry = registry.get(name)
    return entry.to_dict(include_history=include_history)


def tool_stats(registry) -> dict[str, Any]:
    return registry.stats()


def tool_add(
    registry,
    *,
    name: str,
    description: str,
    code: str = '',
    version: str = '1.0.0',
    input_schema_json: str | dict[str, Any] | None = None,
    aliases: str | list[str] = '',
    tags: str | list[str] = '',
    created_by: str = 'llm',
    handler_mode: str | None = None,
    module_path: str | None = None,
    callable_name: str = 'handler',
    metadata_json: str | dict[str, Any] | None = None,
) -> dict[str, Any]:
    if registry.has(name):
        raise ToolAlreadyExistsError(f"Tool '{name}' already exists")

    input_schema = _json_argument_to_dict(
        input_schema_json,
        'input_schema_json',
        default={'type': 'object', 'properties': {}, 'required': []},
    )
    _validate_json_schema(input_schema)
    metadata = _json_argument_to_dict(metadata_json, 'metadata_json')
    handler, handler_meta = _resolve_handler(
        code=code,
        handler_mode=handler_mode,
        module_path=module_path,
        callable_name=callable_name,
    )
    metadata = {**metadata, **handler_meta}

    entry = ToolEntry(
        name=name,
        version=version,
        description=description,
        input_schema=input_schema,
        handler=handler,
        category=ToolCategory.EXTERNAL,
        aliases=_normalize_csv(aliases),
        tags=_normalize_csv(tags),
        created_by=created_by,
        metadata=metadata,
    )
    registry.register(entry)
    return {'ok': True, 'message': f"Tool '{name}' added", 'tool': entry.to_dict(include_history=False)}


def tool_update(
    registry,
    *,
    name: str,
    description: str | None = None,
    code: str | None = None,
    version: str | None = None,
    input_schema_json: str | dict[str, Any] | None = None,
    enabled: bool | None = None,
    tags: str | list[str] | None = None,
    aliases: str | list[str] | None = None,
    status: str | None = None,
    deprecated: bool | None = None,
    replaced_by: str | None = None,
    changelog: str = '',
    handler_mode: str | None = None,
    module_path: str | None = None,
    callable_name: str | None = None,
    metadata_json: str | dict[str, Any] | None = None,
) -> dict[str, Any]:
    entry = registry.get(name)
    updates: dict[str, Any] = {}

    if description is not None:
        updates['description'] = description
    if version is not None:
        updates['version'] = version
    if enabled is not None:
        updates['enabled'] = enabled
    if tags is not None:
        updates['tags'] = _normalize_csv(tags)
    if aliases is not None:
        updates['aliases'] = _normalize_csv(aliases)
    if deprecated is not None:
        updates['deprecated'] = deprecated
    if replaced_by is not None:
        updates['replaced_by'] = replaced_by
    if status is not None:
        updates['status'] = ToolStatus(status)
    if input_schema_json is not None:
        schema = _json_argument_to_dict(input_schema_json, 'input_schema_json')
        _validate_json_schema(schema)
        updates['input_schema'] = schema
    if code is not None or module_path is not None or handler_mode is not None:
        handler, handler_meta = _resolve_handler(
            code=code,
            handler_mode=handler_mode,
            module_path=module_path,
            callable_name=callable_name,
        )
        updates['handler'] = handler
        merged_meta = dict(entry.metadata or {})
        merged_meta.update(handler_meta)
        updates['metadata'] = merged_meta
    if metadata_json is not None:
        merged_meta = dict(entry.metadata or {})
        merged_meta.update(_json_argument_to_dict(metadata_json, 'metadata_json'))
        updates['metadata'] = merged_meta

    updated = registry.update(name, changelog=changelog, **updates)
    return {'ok': True, 'message': f"Tool '{name}' updated", 'tool': updated.to_dict(include_history=True)}


def tool_remove(registry, name: str) -> dict[str, Any]:
    removed = registry.remove(name)
    return {'ok': True, 'message': f"Tool '{removed.name}' removed"}


def tool_enable(registry, name: str) -> dict[str, Any]:
    entry = registry.enable(name)
    return {'ok': True, 'message': f"Tool '{entry.name}' enabled"}


def tool_disable(registry, name: str) -> dict[str, Any]:
    entry = registry.disable(name)
    return {'ok': True, 'message': f"Tool '{entry.name}' disabled"}


def tool_deprecate(registry, name: str, replaced_by: str = '') -> dict[str, Any]:
    entry = registry.deprecate(name, replaced_by=replaced_by or None)
    msg = f"Tool '{entry.name}' deprecated"
    if replaced_by:
        msg += f"; use '{replaced_by}' instead"
    return {'ok': True, 'message': msg}


def tool_alias(registry, name: str, alias: str) -> dict[str, Any]:
    registry.add_alias(name, alias)
    return {'ok': True, 'message': f"Alias '{alias}' -> '{name}' created"}


def tool_merge(registry, source: str, target: str, keep_source: bool = False) -> dict[str, Any]:
    registry.merge(source, target, keep_source=keep_source)
    msg = f"Merged '{source}' into '{target}'"
    if keep_source:
        msg += ' (source preserved but disabled)'
    return {'ok': True, 'message': msg}


def tool_versions(registry, name: str) -> dict[str, Any]:
    return {'name': name, 'versions': registry.list_versions(name)}


def build_internal_registry_tools(registry) -> list[ToolEntry]:
    """Return governance ToolEntry objects bound to the provided registry."""

    return [
        ToolEntry(
            name='tool_list',
            version='2.0.0',
            description='List tools in the registry. Supports optional fuzzy query.',
            input_schema={
                'type': 'object',
                'properties': {'query': {'type': 'string'}},
                'required': [],
                'additionalProperties': False,
            },
            handler=lambda query='': tool_list(registry, query=query),
            category=ToolCategory.INTERNAL_SYSTEM,
            tags=['registry', 'governance'],
            metadata={'surface': 'registry'},
        ),
        ToolEntry(
            name='tool_get',
            version='2.0.0',
            description='Get one tool definition, metadata and version history.',
            input_schema={
                'type': 'object',
                'properties': {
                    'name': {'type': 'string'},
                    'include_history': {'type': 'boolean'},
                },
                'required': ['name'],
                'additionalProperties': False,
            },
            handler=lambda name, include_history=True: tool_get(registry, name=name, include_history=include_history),
            category=ToolCategory.INTERNAL_SYSTEM,
            tags=['registry', 'governance'],
            metadata={'surface': 'registry'},
        ),
        ToolEntry(
            name='tool_stats',
            version='2.0.0',
            description='Return high-level registry statistics.',
            input_schema={
                'type': 'object',
                'properties': {},
                'required': [],
                'additionalProperties': False,
            },
            handler=lambda: tool_stats(registry),
            category=ToolCategory.INTERNAL_SYSTEM,
            tags=['registry', 'governance'],
            metadata={'surface': 'registry'},
        ),
        ToolEntry(
            name='tool_add',
            version='2.0.0',
            description='Add a new external MCP tool using inline Python or a module handler and a JSON Schema input contract.',
            input_schema={
                'type': 'object',
                'properties': {
                    'name': {'type': 'string'},
                    'description': {'type': 'string'},
                    'code': {'type': 'string'},
                    'version': {'type': 'string'},
                    'input_schema_json': {
                        'oneOf': [{'type': 'string'}, {'type': 'object'}],
                    },
                    'aliases': {'oneOf': [{'type': 'string'}, {'type': 'array', 'items': {'type': 'string'}}]},
                    'tags': {'oneOf': [{'type': 'string'}, {'type': 'array', 'items': {'type': 'string'}}]},
                    'created_by': {'type': 'string'},
                    'handler_mode': {'type': 'string', 'enum': ['python_inline', 'python_module']},
                    'module_path': {'type': 'string'},
                    'callable_name': {'type': 'string'},
                    'metadata_json': {'oneOf': [{'type': 'string'}, {'type': 'object'}]},
                },
                'required': ['name', 'description'],
                'additionalProperties': False,
            },
            handler=lambda **kwargs: tool_add(registry, **kwargs),
            category=ToolCategory.INTERNAL_SYSTEM,
            tags=['registry', 'governance', 'mutation'],
            metadata={'surface': 'registry'},
        ),
        ToolEntry(
            name='tool_update',
            version='2.0.0',
            description='Update an external MCP tool, including schema, handler, aliases, lifecycle state, and metadata.',
            input_schema={
                'type': 'object',
                'properties': {
                    'name': {'type': 'string'},
                    'description': {'type': 'string'},
                    'code': {'type': 'string'},
                    'version': {'type': 'string'},
                    'input_schema_json': {'oneOf': [{'type': 'string'}, {'type': 'object'}]},
                    'enabled': {'type': 'boolean'},
                    'tags': {'oneOf': [{'type': 'string'}, {'type': 'array', 'items': {'type': 'string'}}]},
                    'aliases': {'oneOf': [{'type': 'string'}, {'type': 'array', 'items': {'type': 'string'}}]},
                    'status': {'type': 'string', 'enum': [s.value for s in ToolStatus]},
                    'deprecated': {'type': 'boolean'},
                    'replaced_by': {'type': 'string'},
                    'changelog': {'type': 'string'},
                    'handler_mode': {'type': 'string', 'enum': ['python_inline', 'python_module']},
                    'module_path': {'type': 'string'},
                    'callable_name': {'type': 'string'},
                    'metadata_json': {'oneOf': [{'type': 'string'}, {'type': 'object'}]},
                },
                'required': ['name'],
                'additionalProperties': False,
            },
            handler=lambda **kwargs: tool_update(registry, **kwargs),
            category=ToolCategory.INTERNAL_SYSTEM,
            tags=['registry', 'governance', 'mutation'],
            metadata={'surface': 'registry'},
        ),
        ToolEntry(
            name='tool_remove',
            version='2.0.0',
            description='Remove an external tool from the registry.',
            input_schema={
                'type': 'object',
                'properties': {'name': {'type': 'string'}},
                'required': ['name'],
                'additionalProperties': False,
            },
            handler=lambda name: tool_remove(registry, name),
            category=ToolCategory.INTERNAL_SYSTEM,
            tags=['registry', 'governance', 'mutation'],
            metadata={'surface': 'registry'},
        ),
        ToolEntry(
            name='tool_enable',
            version='2.0.0',
            description='Enable a disabled external tool.',
            input_schema={
                'type': 'object',
                'properties': {'name': {'type': 'string'}},
                'required': ['name'],
                'additionalProperties': False,
            },
            handler=lambda name: tool_enable(registry, name),
            category=ToolCategory.INTERNAL_SYSTEM,
            tags=['registry', 'governance', 'mutation'],
            metadata={'surface': 'registry'},
        ),
        ToolEntry(
            name='tool_disable',
            version='2.0.0',
            description='Disable an external tool without deleting it.',
            input_schema={
                'type': 'object',
                'properties': {'name': {'type': 'string'}},
                'required': ['name'],
                'additionalProperties': False,
            },
            handler=lambda name: tool_disable(registry, name),
            category=ToolCategory.INTERNAL_SYSTEM,
            tags=['registry', 'governance', 'mutation'],
            metadata={'surface': 'registry'},
        ),
        ToolEntry(
            name='tool_deprecate',
            version='2.0.0',
            description='Mark an external tool as deprecated and optionally point to a replacement.',
            input_schema={
                'type': 'object',
                'properties': {
                    'name': {'type': 'string'},
                    'replaced_by': {'type': 'string'},
                },
                'required': ['name'],
                'additionalProperties': False,
            },
            handler=lambda name, replaced_by='': tool_deprecate(registry, name, replaced_by=replaced_by),
            category=ToolCategory.INTERNAL_SYSTEM,
            tags=['registry', 'governance', 'mutation'],
            metadata={'surface': 'registry'},
        ),
        ToolEntry(
            name='tool_alias',
            version='2.0.0',
            description='Add an alias for an external tool.',
            input_schema={
                'type': 'object',
                'properties': {
                    'name': {'type': 'string'},
                    'alias': {'type': 'string'},
                },
                'required': ['name', 'alias'],
                'additionalProperties': False,
            },
            handler=lambda name, alias: tool_alias(registry, name, alias),
            category=ToolCategory.INTERNAL_SYSTEM,
            tags=['registry', 'governance', 'mutation'],
            metadata={'surface': 'registry'},
        ),
        ToolEntry(
            name='tool_merge',
            version='2.0.0',
            description='Merge one external tool into another, forwarding aliases and optionally keeping the source disabled.',
            input_schema={
                'type': 'object',
                'properties': {
                    'source': {'type': 'string'},
                    'target': {'type': 'string'},
                    'keep_source': {'type': 'boolean'},
                },
                'required': ['source', 'target'],
                'additionalProperties': False,
            },
            handler=lambda source, target, keep_source=False: tool_merge(registry, source, target, keep_source=keep_source),
            category=ToolCategory.INTERNAL_SYSTEM,
            tags=['registry', 'governance', 'mutation'],
            metadata={'surface': 'registry'},
        ),
        ToolEntry(
            name='tool_versions',
            version='2.0.0',
            description='List archived versions for one tool.',
            input_schema={
                'type': 'object',
                'properties': {'name': {'type': 'string'}},
                'required': ['name'],
                'additionalProperties': False,
            },
            handler=lambda name: tool_versions(registry, name),
            category=ToolCategory.INTERNAL_SYSTEM,
            tags=['registry', 'governance'],
            metadata={'surface': 'registry'},
        ),
    ]


def make_entries(registry):
    return build_internal_registry_tools(registry)
