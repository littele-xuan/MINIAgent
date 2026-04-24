from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable


@dataclass(slots=True)
class ClassifiedMemory:
    bucket: str
    kind: str
    text: str
    confidence: float = 1.0


class HeuristicMemoryClassifier:
    """A deterministic hot-path classifier.

    It is intentionally simple and replaceable. For research experiments, swap this
    with an LLM extractor or a Mem0/Graphiti adapter without touching the graph.
    """

    preference_tokens = ('长期偏好', '偏好', '喜欢', '更喜欢', 'prefer', 'preference', 'i like')
    task_tokens = ('当前任务', '任务', '必须', '不要', '约束', '要求', 'need to', 'must', 'do not', 'constraint')
    procedural_tokens = ('以后都', '默认', 'always', 'whenever', 'by default')

    def classify(self, text: str, *, long_term_bucket: str = 'memories') -> list[ClassifiedMemory]:
        cleaned = text.strip()
        if not cleaned:
            return []
        lower = cleaned.lower()
        items: list[ClassifiedMemory] = []
        if any(token in lower or token in cleaned for token in self.preference_tokens):
            items.append(ClassifiedMemory('profile', 'preference', cleaned, 0.95))
        if any(token in lower or token in cleaned for token in self.task_tokens):
            items.append(ClassifiedMemory('task', 'task_constraint', cleaned, 0.9))
        if any(token in lower or token in cleaned for token in self.procedural_tokens):
            items.append(ClassifiedMemory('procedural', 'procedural', cleaned, 0.75))
        profession = re.search(r'(?:我是一名|我是一位|我是)(.+)', cleaned)
        if profession:
            items.append(ClassifiedMemory('profile', 'profile', profession.group(1).strip(), 0.8))
        if cleaned.startswith('记住') or lower.startswith('remember'):
            items.append(ClassifiedMemory(long_term_bucket, 'explicit_memory', cleaned, 1.0))
        return self._dedupe(items)

    def _dedupe(self, items: Iterable[ClassifiedMemory]) -> list[ClassifiedMemory]:
        out: list[ClassifiedMemory] = []
        seen: set[tuple[str, str, str]] = set()
        for item in items:
            key = (item.bucket, item.kind, item.text)
            if key not in seen:
                seen.add(key)
                out.append(item)
        return out
