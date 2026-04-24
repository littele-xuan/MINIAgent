from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Literal

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, model_validator

from llm_runtime import BaseLLM
from prompt_runtime import render_prompt

from .base import BaseFailureClassifier, BaseFactExtractor
from .models import FactRecord, MemoryEvent, MessageRecord
from .store import new_id


MemoryOp = Literal['upsert', 'revoke']
MemoryScope = Literal['session', 'cross_session']
FailureLabel = Literal['NONE', 'F1_DATA_STATE', 'F2_TOOL_ERROR', 'F3_WORKFLOW_ERROR']


@dataclass(slots=True)
class ExtractedFact:
    op: MemoryOp
    category: str
    key: str
    value: str
    importance: float = 0.6
    scope: MemoryScope = 'cross_session'
    replace_existing: bool = False
    metadata: dict | None = None


# ---- compatibility normalization -------------------------------------------------

def _normalize_memory_op(value: Any) -> MemoryOp:
    normalized = str(value or 'upsert').strip().lower()
    if normalized in {'add', 'insert', 'create', 'store', 'remember', 'upsert'}:
        return 'upsert'
    if normalized in {'remove', 'delete', 'revoke', 'forget'}:
        return 'revoke'
    return 'upsert'


def _normalize_memory_scope(value: Any) -> MemoryScope:
    normalized = str(value or 'cross_session').strip().lower()
    if normalized in {'session', 'task', 'task_local', 'current_task', 'ephemeral'}:
        return 'session'
    return 'cross_session'


def _clean_text(text: str) -> str:
    return re.sub(r'\s+', ' ', text).strip(' 。；;')


def _clean_value(text: str) -> str:
    return _clean_text(text).strip('[](){}"\'')


def _looks_like_instructional_noise(text: str) -> bool:
    lowered = _clean_text(text)
    request_markers = [
        '不要调用工具',
        '不要再调用工具',
        '请直接',
        '请仅用一句话',
        '仅用一句话',
        '用一句话',
        '请重申',
        '请总结',
        '直接总结',
        '请列出',
        '列出我',
        '不要补充',
        '保留项目名',
        '保留项目名、文档编号',
        '用逗号分隔',
        '新的会话里',
        '当前会话上下文',
        '当前会话',
        '输出模式',
        'accepted_output_modes',
    ]
    if any(marker in lowered for marker in request_markers):
        return True
    if '长期偏好' in lowered and any(marker in lowered for marker in ['技术名称', '总结', '重申', '列出', '直接']):
        return True
    if lowered.startswith(('请', '不要', '直接', '仅用', '用一句话', '列出', '总结', '重申')):
        return True
    return False


def _split_preferences(text: str) -> list[str]:
    cleaned = text.replace('，', '、').replace(',', '、').replace(' 和 ', '、').replace('与', '、').replace('/', '、')
    parts = [part.strip(' ：:;；。') for part in re.split(r'[、]+', cleaned) if part.strip(' ：:;；。')]
    deduped: list[str] = []
    seen: set[str] = set()
    for part in parts:
        if part in seen:
            continue
        seen.add(part)
        deduped.append(part)
    return deduped


def _normalize_preference_value(text: str) -> str:
    value = _clean_text(text)
    for marker in ['长期偏好', '偏好']:
        if marker in value:
            value = value.split(marker, 1)[1]
            break
    value = value.lstrip('是为:： ')
    prefs = _split_preferences(value)
    return '、'.join(prefs) if prefs else _clean_text(text)


def _normalize_enum_value(text: str) -> str:
    tokens = re.findall(r'[A-Z][A-Z0-9_]+', text.upper())
    deduped: list[str] = []
    seen: set[str] = set()
    for token in tokens:
        if token in seen:
            continue
        seen.add(token)
        deduped.append(token)
    return '、'.join(deduped)


def _normalize_stack_value(text: str) -> str:
    parts = _split_preferences(text)
    cleaned = [_clean_value(part) for part in parts if _clean_value(part)]
    return '、'.join(cleaned)


def _normalize_constraint_value(text: str) -> str:
    if _looks_like_instructional_noise(text):
        return ''
    parts = re.split(r'[；;]+', text.replace('\n', '；'))
    cleaned = [_clean_value(part) for part in parts if _clean_value(part)]
    return '；'.join(cleaned)


def _build_fact(
    *,
    op: MemoryOp,
    category: str,
    key: str,
    value: str,
    scope: MemoryScope,
    importance: float,
    replace_existing: bool,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    normalized_value = _clean_value(value)
    if not normalized_value or _looks_like_instructional_noise(normalized_value):
        return None
    return {
        'op': op,
        'category': category,
        'key': key,
        'value': normalized_value,
        'importance': importance,
        'scope': scope,
        'replace_existing': replace_existing,
        'metadata': dict(metadata or {}),
    }


def _normalize_known_fact_from_text(raw_fact: str, *, scope: MemoryScope, op: MemoryOp, original: dict[str, Any] | None = None) -> dict[str, Any] | None:
    text = _clean_text(raw_fact)
    base_metadata = dict((original or {}).get('metadata') or {})
    importance = float((original or {}).get('importance', 0.6))

    if _looks_like_instructional_noise(text):
        return None

    if '长期偏好' in text or (scope == 'cross_session' and any(token in text for token in ['偏好', '喜欢'])):
        value = _normalize_preference_value(text)
        if value:
            return _build_fact(
                op=op,
                category='profile',
                key='long_term_preferences',
                value=value,
                importance=max(0.8, importance if importance > 0 else 0.95),
                scope=scope,
                replace_existing=True,
                metadata={**base_metadata, 'kind': 'durable_preference', 'source_format': 'legacy_memory_fact'},
            )

    m = re.search(r'(?:记住)?(?:用户|我)(?:是|为)?(?:一名|一位|个)?(.+)$', text)
    if m and not _looks_like_instructional_noise(m.group(1)):
        return _build_fact(
            op=op,
            category='profile',
            key='profession',
            value=_clean_text(m.group(1)),
            importance=max(0.7, importance if importance > 0 else 0.82),
            scope=scope,
            replace_existing=True,
            metadata={**base_metadata, 'kind': 'profession', 'source_format': 'legacy_memory_fact'},
        )

    if '当前任务' in text:
        value = _clean_value(text.split('当前任务', 1)[1].lstrip('：: ').strip() or text)
        return _build_fact(
            op=op,
            category='task',
            key='current_task',
            value=value,
            importance=max(0.7, importance if importance > 0 else 0.88),
            scope='session',
            replace_existing=True,
            metadata={**base_metadata, 'kind': 'current_task', 'source_format': 'legacy_memory_fact'},
        )

    project = re.search(r'(?:项目名(?:叫|是)?|项目叫|project)\s*[：:\s"]*([A-Za-z0-9_\-]+)', text, re.IGNORECASE)
    if project:
        return _build_fact(
            op=op,
            category='task',
            key='project_name',
            value=project.group(1),
            importance=max(0.7, importance if importance > 0 else 0.85),
            scope='session',
            replace_existing=True,
            metadata={**base_metadata, 'kind': 'project', 'source_format': 'legacy_memory_fact'},
        )

    design_doc = re.search(r'(?:设计文档编号|design_doc_id|doc_id)\s*[：:\s"]*([A-Za-z0-9_.\-]+)', text, re.IGNORECASE)
    if design_doc:
        return _build_fact(
            op=op,
            category='task',
            key='design_doc_id',
            value=design_doc.group(1),
            importance=max(0.78, importance if importance > 0 else 0.9),
            scope='session',
            replace_existing=True,
            metadata={**base_metadata, 'kind': 'design_doc', 'source_format': 'legacy_memory_fact'},
        )

    enum_match = re.search(r'(?:订单状态枚举|order_status_enum)\s*[：:\s"]*(.+)$', text, re.IGNORECASE)
    if enum_match:
        value = _normalize_enum_value(enum_match.group(1))
        if value:
            return _build_fact(
                op=op,
                category='task',
                key='order_status_enum',
                value=value,
                importance=max(0.8, importance if importance > 0 else 0.92),
                scope='session',
                replace_existing=True,
                metadata={**base_metadata, 'kind': 'enum', 'source_format': 'legacy_memory_fact'},
            )

    stack = re.search(r'(?:技术栈|tech_stack)\s*(?:使用)?\s*[：:\s"]*(.+)$', text, re.IGNORECASE)
    if stack:
        value = _normalize_stack_value(stack.group(1))
        if value:
            return _build_fact(
                op=op,
                category='task',
                key='tech_stack',
                value=value,
                importance=max(0.7, importance if importance > 0 else 0.84),
                scope='session',
                replace_existing=True,
                metadata={**base_metadata, 'kind': 'stack', 'source_format': 'legacy_memory_fact'},
            )

    constraints_match = re.search(r'(?:禁止项|constraints?)\s*[：:\s"]*(.+)$', text, re.IGNORECASE)
    if constraints_match:
        value = _normalize_constraint_value(constraints_match.group(1))
        if value:
            return _build_fact(
                op=op,
                category='task',
                key='constraints',
                value=value,
                importance=max(0.75, importance if importance > 0 else 0.88),
                scope='session',
                replace_existing=True,
                metadata={**base_metadata, 'kind': 'constraint', 'source_format': 'legacy_memory_fact'},
            )

    constraints = re.findall(r'(?:do not use|不要|必须保持|keep)\s*([^。；;]+)', text, re.IGNORECASE)
    if constraints:
        value = '；'.join(_clean_text(item) for item in constraints if _clean_text(item))
        if value:
            return _build_fact(
                op=op,
                category='task',
                key='constraints',
                value=value,
                importance=max(0.7, importance if importance > 0 else 0.84),
                scope='session',
                replace_existing=True,
                metadata={**base_metadata, 'kind': 'constraint', 'source_format': 'legacy_memory_fact'},
            )

    return None


def _normalize_legacy_fact(raw_fact: str, *, scope: MemoryScope, op: MemoryOp, original: dict[str, Any] | None = None) -> dict[str, Any]:
    normalized = _normalize_known_fact_from_text(raw_fact, scope=scope, op=op, original=original)
    if normalized is not None:
        return normalized
    return {
        'op': op,
        'category': 'ignored',
        'key': 'ignored',
        'value': '[ignored]',
        'importance': 0.0,
        'scope': 'session',
        'replace_existing': False,
        'metadata': {'ignored': True, **dict((original or {}).get('metadata') or {})},
    }


class ExtractedFactPayload(BaseModel):
    model_config = ConfigDict(extra='forbid')

    op: MemoryOp = 'upsert'
    category: str = Field(min_length=1)
    key: str = Field(min_length=1)
    value: str = Field(min_length=1)
    importance: float = 0.6
    scope: MemoryScope = 'cross_session'
    replace_existing: bool = False
    metadata: dict = Field(default_factory=dict)

    @model_validator(mode='before')
    @classmethod
    def _compat_normalize(cls, raw: Any) -> Any:
        if isinstance(raw, str):
            raw = {'fact': raw}
        if not isinstance(raw, dict):
            return raw
        item = dict(raw)
        item['op'] = _normalize_memory_op(item.get('op'))
        item['scope'] = _normalize_memory_scope(item.get('scope'))

        if isinstance(item.get('metadata'), str):
            try:
                parsed_metadata = json.loads(item['metadata'])
                item['metadata'] = parsed_metadata if isinstance(parsed_metadata, dict) else {}
            except Exception:
                item['metadata'] = {}

        if item.get('category') and item.get('key') and item.get('value'):
            item['category'] = str(item['category']).strip()
            item['key'] = str(item['key']).strip()
            item['value'] = str(item['value']).strip()
            return item

        legacy_fact = item.get('fact') or item.get('memory') or item.get('text') or item.get('content')
        if legacy_fact:
            normalized = _normalize_legacy_fact(str(legacy_fact), scope=item['scope'], op=item['op'], original=item)
            return normalized
        return item


class FactExtractionResult(BaseModel):
    model_config = ConfigDict(extra='forbid')

    facts: list[ExtractedFactPayload] = Field(default_factory=list, validation_alias=AliasChoices('facts', 'memories'))

    @model_validator(mode='before')
    @classmethod
    def _compat_normalize(cls, raw: Any) -> Any:
        if not isinstance(raw, dict):
            return raw
        payload = dict(raw)
        if 'facts' not in payload and 'memories' in payload:
            payload['facts'] = payload.pop('memories')
        payload.pop('name', None)
        return payload


class FailureClassificationResult(BaseModel):
    model_config = ConfigDict(extra='forbid')

    label: FailureLabel = 'NONE'
    reason: str = ''
    normalized_content: str = ''


class LLMTurnFactExtractor(BaseFactExtractor):
    """Memory extraction is runtime-owned, but uses the model for normalization only."""

    def __init__(self, llm: BaseLLM) -> None:
        self.llm = llm

    async def materialize(self, *, namespace: str, session_id: str, message: MessageRecord, created_at: str) -> list[ExtractedFact]:
        if message.role not in {'user', 'assistant', 'tool'}:
            return []
        output = self._heuristic_extract(message.content, role=message.role)
        if message.role != 'user':
            return self._dedupe(output)
        try:
            result = await self.llm.chat_json_model(
                messages=[
                    {'role': 'system', 'content': render_prompt('memory/fact_extraction_system')},
                    {
                        'role': 'user',
                        'content': json.dumps(
                            {
                                'namespace': namespace,
                                'session_id': session_id,
                                'message': {
                                    'role': message.role,
                                    'content': message.content,
                                    'metadata': message.metadata,
                                },
                            },
                            ensure_ascii=False,
                            indent=2,
                        ),
                    },
                ],
                model_type=FactExtractionResult,
                schema_name='memory_fact_extraction',
                temperature=0.0,
                max_output_tokens=900,
            )
        except Exception:
            return self._dedupe(output)

        for item in result.facts:
            if item.category == 'ignored' or item.key == 'ignored':
                continue
            output.append(
                ExtractedFact(
                    op=item.op,
                    category=item.category.strip(),
                    key=item.key.strip(),
                    value=item.value.strip(),
                    importance=max(0.0, min(1.0, float(item.importance))),
                    scope=item.scope,
                    replace_existing=bool(item.replace_existing),
                    metadata=dict(item.metadata or {}),
                )
            )
        return self._dedupe(output)

    def to_record(self, *, namespace: str, session_id: str, extracted: ExtractedFact, source_message_id: str, created_at: str) -> FactRecord:
        return FactRecord(
            id=new_id('fact'),
            namespace=namespace,
            session_id=session_id if extracted.scope == 'session' else None,
            category=extracted.category,
            key=extracted.key,
            value=extracted.value,
            scope='session' if extracted.scope == 'session' else 'cross_session',
            created_at=created_at,
            valid_at=created_at,
            importance=extracted.importance,
            source_message_id=source_message_id,
            metadata={**(extracted.metadata or {}), 'replace_existing': extracted.replace_existing},
        )

    def _dedupe(self, facts: list[ExtractedFact]) -> list[ExtractedFact]:
        seen: set[tuple[str, str, str, str, str]] = set()
        result: list[ExtractedFact] = []
        for fact in facts:
            if self._should_drop_fact(fact):
                continue
            key = (fact.op, fact.scope, fact.category, fact.key, fact.value)
            if key in seen:
                continue
            seen.add(key)
            result.append(fact)
        return result

    def _heuristic_extract(self, content: str, *, role: str) -> list[ExtractedFact]:
        json_candidates = self._extract_from_json_payload(content, role=role)
        candidates = [content]
        candidates.extend(part.strip() for part in re.split(r'[\n，,。；;]+', content) if part.strip())
        output: list[ExtractedFact] = list(json_candidates)
        seen_inputs: set[str] = set()
        for candidate in candidates:
            normalized_candidate = _clean_text(candidate)
            if not normalized_candidate or normalized_candidate in seen_inputs:
                continue
            seen_inputs.add(normalized_candidate)
            if _looks_like_instructional_noise(normalized_candidate):
                continue
            if not any(token in normalized_candidate for token in ['我', '用户', '长期偏好', '偏好', '喜欢', '当前任务', '项目名', '项目叫', '技术栈', '设计文档编号', '订单状态枚举', '禁止项', 'project', 'doc_id', 'constraints', 'tech_stack', 'order_status_enum']):
                continue
            try:
                item = ExtractedFactPayload.model_validate(
                    {
                        'fact': normalized_candidate,
                        'metadata': {'source_format': 'heuristic_memory_fact'},
                    }
                )
            except Exception:
                continue
            if item.category == 'ignored' or item.key == 'ignored':
                continue
            output.append(
                ExtractedFact(
                    op=item.op,
                    category=item.category.strip(),
                    key=item.key.strip(),
                    value=item.value.strip(),
                    importance=max(0.0, min(1.0, float(item.importance))),
                    scope=item.scope,
                    replace_existing=bool(item.replace_existing),
                    metadata=dict(item.metadata or {}),
                )
            )
        return self._dedupe(output)

    def _extract_from_json_payload(self, content: str, *, role: str) -> list[ExtractedFact]:
        try:
            payload = json.loads(content)
        except Exception:
            return []
        if not isinstance(payload, dict):
            return []

        output: list[ExtractedFact] = []
        role_scope: MemoryScope = 'session'

        def append_fact(item: dict[str, Any] | None) -> None:
            if not item or item.get('category') == 'ignored':
                return
            try:
                normalized = ExtractedFactPayload.model_validate(item)
            except Exception:
                return
            output.append(
                ExtractedFact(
                    op=normalized.op,
                    category=normalized.category.strip(),
                    key=normalized.key.strip(),
                    value=normalized.value.strip(),
                    importance=max(0.0, min(1.0, float(normalized.importance))),
                    scope=normalized.scope,
                    replace_existing=bool(normalized.replace_existing),
                    metadata=dict(normalized.metadata or {}),
                )
            )

        if isinstance(payload.get('project'), str):
            append_fact(
                _build_fact(
                    op='upsert',
                    category='task',
                    key='project_name',
                    value=str(payload['project']),
                    importance=0.85,
                    scope=role_scope,
                    replace_existing=True,
                    metadata={'kind': 'project', 'source_format': 'json_payload'},
                )
            )
        doc_id = payload.get('design_doc_id') or payload.get('doc_id')
        if isinstance(doc_id, str):
            append_fact(
                _build_fact(
                    op='upsert',
                    category='task',
                    key='design_doc_id',
                    value=doc_id,
                    importance=0.9,
                    scope=role_scope,
                    replace_existing=True,
                    metadata={'kind': 'design_doc', 'source_format': 'json_payload'},
                )
            )
        if isinstance(payload.get('tech_stack'), list):
            append_fact(
                _build_fact(
                    op='upsert',
                    category='task',
                    key='tech_stack',
                    value='、'.join(str(item) for item in payload['tech_stack'] if str(item).strip()),
                    importance=0.84,
                    scope=role_scope,
                    replace_existing=True,
                    metadata={'kind': 'stack', 'source_format': 'json_payload'},
                )
            )
        if isinstance(payload.get('order_status_enum'), list):
            append_fact(
                _build_fact(
                    op='upsert',
                    category='task',
                    key='order_status_enum',
                    value='、'.join(str(item) for item in payload['order_status_enum'] if str(item).strip()),
                    importance=0.92,
                    scope=role_scope,
                    replace_existing=True,
                    metadata={'kind': 'enum', 'source_format': 'json_payload'},
                )
            )
        if isinstance(payload.get('constraints'), list):
            append_fact(
                _build_fact(
                    op='upsert',
                    category='task',
                    key='constraints',
                    value='；'.join(str(item) for item in payload['constraints'] if str(item).strip()),
                    importance=0.88,
                    scope=role_scope,
                    replace_existing=True,
                    metadata={'kind': 'constraint', 'source_format': 'json_payload'},
                )
            )
        return output

    def _should_drop_fact(self, fact: ExtractedFact) -> bool:
        if fact.category == 'ignored' or fact.key == 'ignored':
            return True
        if not fact.value or fact.value == '[ignored]':
            return True
        if _looks_like_instructional_noise(fact.value):
            return True
        if fact.key == 'statement':
            return True
        return False


class FailureClassifier(BaseFailureClassifier):
    """Observation classification stays inside the memory loop, not the MCP tool layer."""

    def __init__(self, llm: BaseLLM) -> None:
        self.llm = llm

    async def classify(self, *, content: str, metadata: dict | None = None) -> FailureClassificationResult:
        return await self.llm.chat_json_model(
            messages=[
                {'role': 'system', 'content': render_prompt('memory/failure_classification_system')},
                {'role': 'user', 'content': json.dumps({'content': content, 'metadata': metadata or {}}, ensure_ascii=False, indent=2)},
            ],
            model_type=FailureClassificationResult,
            schema_name='memory_failure_classification',
            temperature=0.0,
            max_output_tokens=400,
        )

    async def from_observation(self, *, session_id: str, content: str, created_at: str, metadata: dict | None = None) -> MemoryEvent | None:
        result = await self.classify(content=content, metadata=metadata)
        if result.label == 'NONE':
            return None
        return MemoryEvent(
            id=new_id('evt'),
            session_id=session_id,
            event_type='failure',
            classifier=result.label,
            content=(result.normalized_content or content).strip(),
            created_at=created_at,
            metadata={**(metadata or {}), 'reason': result.reason},
        )
