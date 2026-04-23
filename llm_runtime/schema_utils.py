from __future__ import annotations

import copy
from typing import Any


_ALLOWED_META_KEYS = {'description'}


def build_openai_responses_json_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Convert a Pydantic JSON schema into the strict subset expected by OpenAI Structured Outputs.

    This sanitizer does four things:
    1. inline local $ref references so the model sees a single resolved schema
    2. remove noisy keys such as title/default/examples
    3. force object schemas to use additionalProperties=false and mark every property as required
    4. rewrite open-ended dict/Any shapes into JSON-string fields, which the runtime can parse back

    The runtime models remain expressive; only the transport schema is narrowed.
    """

    root = copy.deepcopy(schema)
    defs = root.pop('$defs', {}) if isinstance(root, dict) else {}
    normalized = _transform_node(_inline_refs(root, defs))
    if isinstance(normalized, dict):
        normalized.setdefault('type', 'object')
        if normalized.get('type') == 'object':
            normalized.setdefault('properties', {})
            normalized['required'] = list(normalized['properties'].keys())
            normalized['additionalProperties'] = False
    return normalized


def _inline_refs(node: Any, defs: dict[str, Any]) -> Any:
    if isinstance(node, list):
        return [_inline_refs(item, defs) for item in node]
    if not isinstance(node, dict):
        return node

    if '$ref' in node:
        ref = node['$ref']
        if not isinstance(ref, str) or not ref.startswith('#/$defs/'):
            raise ValueError(f'unsupported schema ref: {ref}')
        name = ref.split('/')[-1]
        target = defs.get(name)
        if target is None:
            raise ValueError(f'missing schema ref target: {ref}')
        merged = copy.deepcopy(target)
        sibling_items = {k: v for k, v in node.items() if k != '$ref'}
        if sibling_items:
            if isinstance(merged, dict):
                merged.update(sibling_items)
        return _inline_refs(merged, defs)

    return {key: _inline_refs(value, defs) for key, value in node.items() if key != '$defs'}



def _transform_node(node: Any) -> Any:
    if isinstance(node, list):
        return [_transform_node(item) for item in node]
    if not isinstance(node, dict):
        return node

    # Open-ended Any schema -> transport it as a JSON string.
    if not node:
        return {
            'type': 'string',
            'description': 'JSON-encoded value',
        }

    # Open-ended object maps are not stable under strict structured outputs.
    if node.get('type') == 'object' and not node.get('properties'):
        return {
            'type': 'string',
            'description': node.get('description', 'JSON-encoded object'),
        }

    out: dict[str, Any] = {}
    for key, value in node.items():
        if key in {'title', 'default', 'examples', '$schema', 'discriminator'}:
            continue
        if key == 'properties':
            out[key] = _transform_properties(value)
            continue
        if key in {'items', 'prefixItems', 'anyOf', 'oneOf', 'allOf'}:
            out[key] = _transform_node(value)
            continue
        if key == 'additionalProperties':
            # Strict transport schema forbids open dictionaries.
            if value is not False and value is not None:
                return {
                    'type': 'string',
                    'description': node.get('description', 'JSON-encoded object'),
                }
            out[key] = False
            continue
        if key in {'required', 'enum', 'const', 'type', 'description', 'minLength', 'maxLength', 'minimum', 'maximum', 'minItems', 'maxItems'}:
            out[key] = _transform_node(value)
            continue
        if key in _ALLOWED_META_KEYS:
            out[key] = _transform_node(value)
            continue
        # Drop unsupported / noisy JSON schema keys by default.

    if out.get('type') == 'object':
        properties = out.get('properties') or {}
        out['properties'] = properties
        out['required'] = list(properties.keys())
        out['additionalProperties'] = False

    return out


def _transform_properties(node: Any) -> dict[str, Any]:
    if not isinstance(node, dict):
        return {}
    return {
        str(key): _transform_node(value)
        for key, value in node.items()
    }
