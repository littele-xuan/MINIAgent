from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .models import MemoryItem, MemoryRecallResult
from .policy import MemoryPolicy


@dataclass(slots=True)
class FileMemoryStore:
    root: Path
    policy: MemoryPolicy

    @classmethod
    def create(cls, root: str | Path) -> "FileMemoryStore":
        store = cls(root=Path(root), policy=MemoryPolicy())
        store.initialize()
        return store

    def initialize(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "skills").mkdir(exist_ok=True)
        (self.root / "sessions").mkdir(exist_ok=True)
        (self.root / "pending").mkdir(exist_ok=True)
        defaults = {
            "l0_policy.md": "# L0 Memory Policy\n\n- Write durable memory only when it is stable and useful across future runs.\n- Require explicit user or tool evidence.\n- Keep temporary task state in working checkpoint instead of long-term memory.\n",
            "l1_index.md": "# L1 Memory Index\n\n",
            "l2_facts.md": "# L2 Durable Facts\n\n",
            "skills/code_refactor.md": "# Skill: Code Refactor\n\nReusable implementation notes learned from verified runs.\n",
            "skills/file_editing.md": "# Skill: File Editing\n\nPrefer exact `file_patch` replacements over full-file rewrites.\n",
            "skills/tool_debugging.md": "# Skill: Tool Debugging\n\nWhen a tool fails, read the latest state and retry with narrower inputs.\n",
        }
        for rel, text in defaults.items():
            p = self.root / rel
            if not p.exists():
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text(text, encoding="utf-8")

    def recall(self, query: str, *, max_items: int = 6) -> MemoryRecallResult:
        terms = _terms(query)
        candidates: list[dict[str, Any]] = []
        for path in [self.root / "l1_index.md", self.root / "l2_facts.md", *sorted((self.root / "skills").glob("*.md"))]:
            if not path.exists():
                continue
            text = path.read_text(encoding="utf-8", errors="replace")
            chunks = _split_chunks(text)
            for chunk in chunks:
                score = _score(chunk, terms)
                if score > 0:
                    candidates.append({"source": str(path.relative_to(self.root)), "score": score, "text": chunk[:1200]})
        candidates.sort(key=lambda x: x["score"], reverse=True)
        return MemoryRecallResult(query=query, items=candidates[:max_items])

    def propose_update(self, item: MemoryItem) -> tuple[bool, str, str | None]:
        ok, reason = self.policy.validate(item.content, item.evidence)
        if not ok:
            return False, reason, None
        pending_path = self.root / "pending" / "memory_proposals.jsonl"
        with pending_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(item), ensure_ascii=False) + "\n")
        return True, "proposal recorded", item.id

    def commit_item(self, item: MemoryItem) -> tuple[bool, str]:
        ok, reason = self.policy.validate(item.content, item.evidence)
        if not ok:
            return False, reason
        target = self._target_file(item.layer)
        with target.open("a", encoding="utf-8") as f:
            f.write("\n" + item.to_markdown() + "\n")
        self._append_index(item)
        return True, f"committed to {target.relative_to(self.root)}"

    def commit_pending(self, proposal_id: str) -> tuple[bool, str]:
        pending_path = self.root / "pending" / "memory_proposals.jsonl"
        if not pending_path.exists():
            return False, "no pending proposals"
        lines = pending_path.read_text(encoding="utf-8").splitlines()
        remaining: list[str] = []
        found: MemoryItem | None = None
        for line in lines:
            if not line.strip():
                continue
            data = json.loads(line)
            if data.get("id") == proposal_id and found is None:
                found = MemoryItem(**data)
            else:
                remaining.append(line)
        if found is None:
            return False, f"proposal not found: {proposal_id}"
        ok, msg = self.commit_item(found)
        if ok:
            pending_path.write_text("\n".join(remaining) + ("\n" if remaining else ""), encoding="utf-8")
        return ok, msg

    def log_session_event(self, session_id: str, event: dict[str, Any]) -> None:
        path = self.root / "sessions" / f"{session_id}.jsonl"
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False, default=str) + "\n")

    def _target_file(self, layer: str) -> Path:
        layer = (layer or "facts").lower()
        if layer in {"l2", "facts", "fact"}:
            return self.root / "l2_facts.md"
        if layer in {"skill", "skills", "l3"}:
            return self.root / "skills" / "tool_debugging.md"
        return self.root / "l2_facts.md"

    def _append_index(self, item: MemoryItem) -> None:
        idx = self.root / "l1_index.md"
        with idx.open("a", encoding="utf-8") as f:
            f.write(f"- {item.id}: {item.content[:140].replace(chr(10), ' ')}\n")


def _terms(query: str) -> set[str]:
    return {t.lower() for t in re.findall(r"[\w\u4e00-\u9fff]{2,}", query or "")}


def _split_chunks(text: str) -> list[str]:
    chunks = [c.strip() for c in re.split(r"\n(?=#+\s)|\n\s*- id:\s", text) if c.strip()]
    return chunks or [text]


def _score(text: str, terms: set[str]) -> int:
    lowered = text.lower()
    return sum(1 for term in terms if term in lowered)
