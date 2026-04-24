from __future__ import annotations

import hashlib
from typing import Any


def stable_memory_key(*, user_id: str, bucket: str, text: str, kind: str = '') -> str:
    digest = hashlib.sha256(f'{user_id}\0{bucket}\0{kind}\0{text.strip().lower()}'.encode('utf-8')).hexdigest()
    return digest[:32]


class MemoryResolver:
    """Dedup/update policy for hot-path memory writes.

    This is intentionally conservative: identical facts update the same key;
    non-identical facts are preserved with provenance so a later consolidation
    module can perform contradiction resolution.
    """

    def make_record(self, *, raw: dict[str, Any], user_id: str, thread_id: str, bucket: str) -> tuple[str, dict[str, Any]]:
        text = str(raw.get('text') or raw.get('content') or '').strip()
        kind = str(raw.get('kind') or 'explicit_memory')
        key = stable_memory_key(user_id=user_id, bucket=bucket, kind=kind, text=text)
        record = {
            **raw,
            'text': text,
            'kind': kind,
            'bucket': bucket,
            'user_id': user_id,
            'thread_id': thread_id,
        }
        return key, record
