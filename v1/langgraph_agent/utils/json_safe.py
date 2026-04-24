from __future__ import annotations

import dataclasses
import json
from collections.abc import Mapping, Sequence
from datetime import date, datetime
from pathlib import Path
from typing import Any

try:
    from pydantic import BaseModel
except Exception:  # pragma: no cover
    BaseModel = object  # type: ignore


_PRIMITIVE = (str, int, float, bool, type(None))


def compact_text(text: Any, *, limit: int = 4000) -> str:
    value = str(text or '')
    if len(value) <= limit:
        return value
    return value[: limit - 80] + f"\n...[truncated {len(value) - limit + 80} chars]"


def json_safe(value: Any, *, max_depth: int = 8, max_items: int = 64, max_string: int = 8000) -> Any:
    """Return a checkpoint-safe JSON value.

    LangGraph checkpoints are serialized between graph super-steps. Runtime objects
    such as LangChain callbacks, MCP sessions, file handles, or model clients must
    never be written into graph state. This helper normalizes tool payloads, traces,
    memory values, and errors before they enter state.
    """
    if max_depth <= 0:
        return repr(value)
    if isinstance(value, _PRIMITIVE):
        if isinstance(value, str):
            return compact_text(value, limit=max_string)
        return value
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, bytes):
        return compact_text(value.decode('utf-8', errors='replace'), limit=max_string)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, BaseModel):
        try:
            return json_safe(value.model_dump(mode='json'), max_depth=max_depth - 1, max_items=max_items, max_string=max_string)
        except Exception:
            return repr(value)
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        try:
            return json_safe(dataclasses.asdict(value), max_depth=max_depth - 1, max_items=max_items, max_string=max_string)
        except Exception:
            return repr(value)
    if isinstance(value, Mapping):
        out: dict[str, Any] = {}
        for idx, (key, item) in enumerate(value.items()):
            if idx >= max_items:
                out['__truncated__'] = f'{len(value) - max_items} more items'
                break
            out[str(key)] = json_safe(item, max_depth=max_depth - 1, max_items=max_items, max_string=max_string)
        return out
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        items = list(value)
        out = [json_safe(item, max_depth=max_depth - 1, max_items=max_items, max_string=max_string) for item in items[:max_items]]
        if len(items) > max_items:
            out.append({'__truncated__': f'{len(items) - max_items} more items'})
        return out
    try:
        json.dumps(value)
        return value
    except Exception:
        return repr(value)
