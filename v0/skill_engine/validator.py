from __future__ import annotations

import re
from pathlib import Path
from typing import Any


NAME_PATTERN = re.compile(r'^[a-z0-9](?:[a-z0-9-]{0,62}[a-z0-9])?$|^[a-z0-9]$')
XML_PATTERN = re.compile(r'<[^>]+>')
ALLOWED_FIELDS = {
    'name',
    'description',
    'license',
    'compatibility',
    'metadata',
    'when_to_use',
    'argument-hint',
    'disable-model-invocation',
    'user-invocable',
    'allowed-tools',
    'model',
    'effort',
    'context',
    'agent',
    'hooks',
    'paths',
    'shell',
    'output-modes',
    'accepted-output-modes',
    'mcp',
    'a2a',
    'examples',
}


class SkillValidationError(ValueError):
    pass


def validate_frontmatter_keys(raw: dict[str, Any]) -> None:
    unknown = sorted(set(raw.keys()) - ALLOWED_FIELDS)
    if unknown:
        raise SkillValidationError('Unsupported SKILL.md frontmatter fields: ' + ', '.join(unknown))


def validate_name(name: str, folder_name: str) -> None:
    if not (1 <= len(name) <= 64):
        raise SkillValidationError('name must be 1-64 characters')
    if not NAME_PATTERN.match(name):
        raise SkillValidationError('name must contain only lowercase letters, numbers, and hyphens')
    if '--' in name or name.startswith('-') or name.endswith('-'):
        raise SkillValidationError('name must not start/end with a hyphen or contain consecutive hyphens')
    if XML_PATTERN.search(name):
        raise SkillValidationError('name cannot contain XML tags')
    if folder_name and name != folder_name:
        raise SkillValidationError('name should match the parent folder name for filesystem portability')


def validate_description(description: str) -> None:
    if not (1 <= len(description) <= 1024):
        raise SkillValidationError('description must be 1-1024 characters')
    if XML_PATTERN.search(description):
        raise SkillValidationError('description cannot contain XML tags')


def normalize_allowed_tools(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [item.strip() for item in value.split() if item.strip()]
    if isinstance(value, list):
        normalized: list[str] = []
        for item in value:
            if not isinstance(item, str) or not item.strip():
                raise SkillValidationError('allowed-tools entries must be non-empty strings')
            normalized.append(item.strip())
        return normalized
    raise SkillValidationError('allowed-tools must be a YAML list or space-separated string')


def normalize_paths(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [item.strip() for item in value.split(',') if item.strip()]
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    raise SkillValidationError('paths must be a YAML list or comma-separated string')


def validate_effort(value: str | None) -> None:
    if value is None:
        return
    if value not in {'low', 'medium', 'high', 'max'}:
        raise SkillValidationError('effort must be one of low/medium/high/max')


def validate_context_agent(context: str | None, agent: str | None) -> None:
    if context is None:
        return
    if context != 'fork':
        raise SkillValidationError('context currently supports only: fork')
    if agent is not None and not str(agent).strip():
        raise SkillValidationError('agent cannot be blank when provided')


def validate_shell(shell: str | None) -> None:
    if shell is None:
        return
    if shell not in {'bash', 'powershell'}:
        raise SkillValidationError('shell must be bash or powershell')


def validate_local_tool_spec(name: str, description: str, handler_file: Path) -> None:
    if not name.strip():
        raise SkillValidationError('local MCP tool name cannot be empty')
    if not description.strip():
        raise SkillValidationError('local MCP tool description cannot be empty')
    if not handler_file.exists():
        raise SkillValidationError(f'local MCP tool handler not found: {handler_file}')


def normalize_string_list(value: Any, field_name: str) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [item.strip() for item in value.split(',') if item.strip()]
    if isinstance(value, list):
        normalized: list[str] = []
        for item in value:
            if not isinstance(item, str) or not item.strip():
                raise SkillValidationError(f'{field_name} entries must be non-empty strings')
            normalized.append(item.strip())
        return normalized
    raise SkillValidationError(f'{field_name} must be a YAML list or comma-separated string')


def normalize_mapping(value: Any, field_name: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise SkillValidationError(f'{field_name} must be a YAML mapping')
    return dict(value)
