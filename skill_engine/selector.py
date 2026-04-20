from __future__ import annotations

import re
from typing import Iterable

from .models import SkillActivation, SkillBundle


STOPWORDS = {
    'the', 'a', 'an', 'and', 'or', 'to', 'for', 'of', 'in', 'on', 'with', 'use', 'this', 'when',
    '我', '你', '他', '她', '它', '的', '了', '和', '或', '与', '在', '对', '把', '将', '一个',
    '需要', '使用', '用于', '用户', '请求', '进行', '处理', '分析', '查看', '读取', '工具', '技能',
}


class SkillSelector:
    def choose(self, query: str, bundles: Iterable[SkillBundle]) -> SkillBundle | None:
        terms = self._terms(query)
        best: tuple[float, SkillBundle] | None = None
        for bundle in bundles:
            text = ' '.join(filter(None, [bundle.name, bundle.description, bundle.when_to_use or '', ' '.join(bundle.allowed_tools), ' '.join(bundle.frontmatter.examples or [])]))
            lower_text = text.lower()
            score_terms = self._terms(text)
            overlap = terms & score_terms
            score = float(len(overlap))
            for term in terms:
                if term in lower_text:
                    score += 0.5
            if bundle.name in query.lower():
                score += 3.0
            if score <= 0:
                continue
            if best is None or score > best[0]:
                best = (score, bundle)
        return best[1] if best is not None else None

    def _terms(self, text: str) -> set[str]:
        raw_tokens = re.findall(r'[a-zA-Z0-9_-]+|[一-鿿]{2,}', text.lower())
        output: set[str] = set()
        for token in raw_tokens:
            if token in STOPWORDS or len(token) < 2:
                continue
            output.add(token)
            if re.fullmatch(r'[一-鿿]+', token):
                for size in (2, 3, 4):
                    for idx in range(max(0, len(token) - size + 1)):
                        gram = token[idx: idx + size]
                        if gram not in STOPWORDS:
                            output.add(gram)
        return output
