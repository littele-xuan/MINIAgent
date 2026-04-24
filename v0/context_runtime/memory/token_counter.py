from __future__ import annotations

import re

_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+|[一-鿿]|[^\s]")


def estimate_tokens(text: str | None) -> int:
    if not text:
        return 0
    return len(_TOKEN_RE.findall(text))
