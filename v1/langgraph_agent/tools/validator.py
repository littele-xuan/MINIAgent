from __future__ import annotations

from typing import Any


class ToolArgumentValidator:
    """JSON-schema validator for MCP tool arguments.

    Uses jsonschema when installed; otherwise falls back to a safe subset so the
    baseline still runs in minimal research environments.
    """

    def validate(self, *, tool_name: str, input_schema: dict[str, Any] | None, arguments: dict[str, Any]) -> None:
        schema = input_schema or {'type': 'object'}
        if not isinstance(arguments, dict):
            raise ValueError(f'Tool {tool_name} arguments must be a JSON object')
        try:
            from jsonschema import Draft202012Validator
            Draft202012Validator(schema).validate(arguments)
            return
        except ImportError:
            pass
        except Exception as exc:
            raise ValueError(f'Tool {tool_name} arguments do not match input_schema: {exc}') from exc

        required = schema.get('required') or []
        missing = [name for name in required if name not in arguments]
        if missing:
            raise ValueError(f'Tool {tool_name} missing required arguments: {missing}')
        if schema.get('type') == 'object' and not isinstance(arguments, dict):
            raise ValueError(f'Tool {tool_name} arguments must be a JSON object')
        properties = schema.get('properties') or {}
        for name, prop in properties.items():
            if name not in arguments or not isinstance(prop, dict):
                continue
            expected = prop.get('type')
            if expected == 'string' and not isinstance(arguments[name], str):
                raise ValueError(f'Tool {tool_name}.{name} must be string')
            if expected == 'boolean' and not isinstance(arguments[name], bool):
                raise ValueError(f'Tool {tool_name}.{name} must be boolean')
            if expected == 'integer' and not isinstance(arguments[name], int):
                raise ValueError(f'Tool {tool_name}.{name} must be integer')
            if expected == 'number' and not isinstance(arguments[name], (int, float)):
                raise ValueError(f'Tool {tool_name}.{name} must be number')
