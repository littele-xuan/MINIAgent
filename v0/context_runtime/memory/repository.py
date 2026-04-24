from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .base import BaseMemoryRepository
from .models import FactRecord, FileArtifact, MemoryEvent, MessageRecord, SummaryNode


@dataclass(slots=True)
class FileSnapshot:
    path: str
    exists: bool
    mtime: float | None
    size_bytes: int | None
    checksum: str | None


class MemoryRepository(BaseMemoryRepository):
    """Filesystem-first mirror of memory state.

    SQLite is the execution store. This repository is a durable, human-inspectable,
    greppable mirror inspired by context-repository style systems.
    """

    def __init__(self, root: str | Path, *, namespace: str, auto_git_commit: bool = False) -> None:
        self.root = Path(root)
        self.namespace = namespace
        self.auto_git_commit = auto_git_commit
        self.logs_dir = self.root / 'logs'
        self.system_dir = self.root / 'system'
        self.profile_dir = self.root / 'profiles' / namespace
        self.session_dir = self.root / 'sessions'
        self.summary_dir = self.root / 'summaries'
        self.failure_dir = self.root / 'failures'
        self.artifact_dir = self.root / 'artifacts'
        self.state_dir = self.root / 'state'
        for directory in [self.logs_dir, self.system_dir, self.profile_dir, self.session_dir, self.summary_dir, self.failure_dir, self.artifact_dir, self.state_dir]:
            directory.mkdir(parents=True, exist_ok=True)
        self._ensure_system_files()
        if self.auto_git_commit:
            self._ensure_git_repo()

    def append_message(self, message: MessageRecord) -> None:
        day = message.created_at[:10]
        path = self.logs_dir / f'{day}.jsonl'
        with path.open('a', encoding='utf-8') as handle:
            handle.write(json.dumps({
                'type': 'message',
                'id': message.id,
                'session_id': message.session_id,
                'turn_id': message.turn_id,
                'role': message.role,
                'created_at': message.created_at,
                'content': message.content,
                'metadata': message.metadata,
            }, ensure_ascii=False) + '\n')
        self._append_session_markdown(message)

    def write_summary(self, summary: SummaryNode) -> None:
        path = self.summary_dir / f'{summary.id}.md'
        path.write_text(
            '\n'.join([
                '---',
                f'id: {summary.id}',
                f'session_id: {summary.session_id}',
                f'level: {summary.level}',
                f'created_at: {summary.created_at}',
                f'source_items: {json.dumps(summary.source_item_ids, ensure_ascii=False)}',
                f'leaf_messages: {json.dumps(summary.leaf_message_ids, ensure_ascii=False)}',
                f'metadata: {json.dumps(summary.metadata, ensure_ascii=False)}',
                '---',
                '',
                summary.content,
                '',
            ]),
            encoding='utf-8',
        )

    def write_fact(self, fact: FactRecord) -> None:
        category_dir = self.profile_dir / fact.category
        category_dir.mkdir(parents=True, exist_ok=True)
        path = category_dir / f'{fact.key.replace(":", "__")}.md'
        with path.open('a', encoding='utf-8') as handle:
            handle.write(
                f'- [{fact.status}] value={fact.value} | scope={fact.scope} | valid={fact.valid_at or fact.created_at} -> {fact.invalid_at or "present"} | metadata={json.dumps(fact.metadata, ensure_ascii=False)}\n'
            )

    def write_event(self, event: MemoryEvent) -> None:
        path = self.failure_dir / f'{event.created_at[:10]}.md'
        with path.open('a', encoding='utf-8') as handle:
            handle.write(f'- [{event.classifier}] {event.created_at} :: {event.content}\n')

    def write_artifact(self, artifact: FileArtifact, *, body: str | None = None) -> None:
        path = Path(artifact.path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if body is not None:
            path.write_text(body, encoding='utf-8')
        index_path = self.artifact_dir / 'INDEX.md'
        with index_path.open('a', encoding='utf-8') as handle:
            handle.write(f'- `{artifact.id}` [{artifact.kind}] {artifact.path} :: {artifact.preview}\n')

    def write_session_state(self, *, session_id: str, payload: dict[str, Any]) -> None:
        path = self.state_dir / f'{session_id}.json'
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')

    def system_notes(self) -> list[str]:
        notes: list[str] = []
        for path in sorted(self.system_dir.glob('*.md')):
            try:
                notes.append(path.read_text(encoding='utf-8').strip())
            except Exception:
                continue
        return [note for note in notes if note]

    def snapshot_file(self, path: str | Path) -> FileSnapshot:
        file_path = Path(path)
        if not file_path.exists():
            return FileSnapshot(path=str(file_path), exists=False, mtime=None, size_bytes=None, checksum=None)
        data = file_path.read_bytes()
        return FileSnapshot(
            path=str(file_path),
            exists=True,
            mtime=file_path.stat().st_mtime,
            size_bytes=len(data),
            checksum=hashlib.sha256(data).hexdigest(),
        )

    def maybe_commit(self, message: str) -> None:
        if not self.auto_git_commit:
            return
        self._git_commit(message)

    def _append_session_markdown(self, message: MessageRecord) -> None:
        path = self.session_dir / f'{message.session_id}.md'
        with path.open('a', encoding='utf-8') as handle:
            handle.write(f'## {message.created_at} [{message.role}]\n\n{message.content}\n\n')

    def _ensure_system_files(self) -> None:
        instructions = self.system_dir / 'CONTEXT_RULES.md'
        if not instructions.exists():
            instructions.write_text(
                '\n'.join([
                    '# Context runtime rules',
                    '',
                    '- Source of truth: immutable log + summary DAG + structured facts.',
                    '- Task-local context stays task-local unless explicitly promoted to cross-session memory.',
                    '- Prefer re-reading mutable files when mtimes/checksums differ from captured memory.',
                    '- Summaries are derived views; original messages remain recoverable.',
                    '- Large tool outputs should be persisted as artifacts and referenced from active context.',
                ]),
                encoding='utf-8',
            )

    def _ensure_git_repo(self) -> None:
        if (self.root / '.git').exists():
            return
        try:
            subprocess.run(['git', 'init'], cwd=self.root, check=True, capture_output=True)
            subprocess.run(['git', 'config', 'user.email', 'agent@example.local'], cwd=self.root, check=True, capture_output=True)
            subprocess.run(['git', 'config', 'user.name', 'Agent Context Runtime'], cwd=self.root, check=True, capture_output=True)
        except Exception:
            self.auto_git_commit = False

    def _git_commit(self, message: str) -> None:
        if not (self.root / '.git').exists():
            return
        try:
            subprocess.run(['git', 'add', '.'], cwd=self.root, check=True, capture_output=True)
            status = subprocess.run(['git', 'status', '--porcelain'], cwd=self.root, check=True, capture_output=True, text=True)
            if not status.stdout.strip():
                return
            subprocess.run(['git', 'commit', '-m', message], cwd=self.root, check=True, capture_output=True)
        except Exception:
            return


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec='seconds')
