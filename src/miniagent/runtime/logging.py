from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class JsonlRunLogger:
    """Simple JSONL logger for real runs and debugging."""

    log_dir: Path
    session_id: str

    @classmethod
    def create(cls, log_dir: str | Path, session_id: str) -> "JsonlRunLogger":
        p = Path(log_dir)
        p.mkdir(parents=True, exist_ok=True)
        return cls(log_dir=p, session_id=session_id)

    @property
    def path(self) -> Path:
        return self.log_dir / f"{self.session_id}.jsonl"

    def write(self, event: str, payload: dict[str, Any]) -> None:
        record = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "session_id": self.session_id,
            "event": event,
            "payload": payload,
        }
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
