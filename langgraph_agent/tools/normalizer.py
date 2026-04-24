from __future__ import annotations

import json
from typing import Any

from ..schemas import ToolObservation
from ..utils import compact_text, json_safe


class ToolResultNormalizer:
    def __init__(self, *, max_text_chars: int = 6000) -> None:
        self.max_text_chars = max_text_chars

    def normalize(self, *, tool_name: str, arguments: dict[str, Any], result: Any) -> dict[str, Any]:
        try:
            if isinstance(result, ToolObservation):
                obs = result
            elif isinstance(result, str):
                obs = ToolObservation(tool_name=tool_name, arguments=json_safe(arguments), text=result, payload=None)
            elif isinstance(result, dict):
                obs = ToolObservation(
                    tool_name=str(result.get('tool_name') or tool_name),
                    arguments=json_safe(result.get('arguments') or arguments),
                    text=str(result.get('text') if result.get('text') is not None else json.dumps(result, ensure_ascii=False, default=str)),
                    payload=json_safe(result.get('payload')),
                    status=result.get('status', 'ok') if result.get('status') in ('ok', 'error') else 'ok',
                    error=result.get('error'),
                    metadata=json_safe(result.get('metadata') or {}),
                )
            else:
                obs = ToolObservation(tool_name=tool_name, arguments=json_safe(arguments), text=str(result), payload=json_safe(getattr(result, 'artifact', None)))
            data = obs.model_dump(mode='json')
            data['text'] = compact_text(data.get('text', ''), limit=self.max_text_chars)
            return data
        except Exception as exc:
            return ToolObservation(
                tool_name=tool_name,
                arguments=json_safe(arguments),
                text='',
                status='error',
                error=str(exc),
            ).model_dump(mode='json')
