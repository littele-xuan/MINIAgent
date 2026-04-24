from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from pathlib import Path
from typing import Any, Iterable

from .base import BaseMemoryStore
from .models import ActiveContextItem, FactRecord, FileArtifact, MemoryEvent, MessageRecord, SummaryNode


class SQLiteMemoryStore(BaseMemoryStore):
    def __init__(self, db_path: str | Path) -> None:
        self.db_path = str(db_path)
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._fts_enabled = True
        self._ensure_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute('PRAGMA journal_mode=WAL;')
        conn.execute('PRAGMA foreign_keys=ON;')
        return conn

    def _ensure_schema(self) -> None:
        with self._lock:
            conn = self._connect()
            try:
                conn.executescript(
                    '''
                    CREATE TABLE IF NOT EXISTS sessions (
                        session_id TEXT PRIMARY KEY,
                        namespace TEXT NOT NULL,
                        created_at TEXT NOT NULL,
                        last_turn_id TEXT,
                        metadata_json TEXT NOT NULL DEFAULT '{}'
                    );

                    CREATE TABLE IF NOT EXISTS messages (
                        id TEXT PRIMARY KEY,
                        session_id TEXT NOT NULL,
                        turn_id TEXT NOT NULL,
                        role TEXT NOT NULL,
                        content TEXT NOT NULL,
                        created_at TEXT NOT NULL,
                        token_count INTEGER NOT NULL,
                        kind TEXT NOT NULL DEFAULT 'message',
                        metadata_json TEXT NOT NULL DEFAULT '{}',
                        FOREIGN KEY(session_id) REFERENCES sessions(session_id)
                    );

                    CREATE TABLE IF NOT EXISTS summaries (
                        id TEXT PRIMARY KEY,
                        session_id TEXT NOT NULL,
                        level INTEGER NOT NULL,
                        content TEXT NOT NULL,
                        created_at TEXT NOT NULL,
                        token_count INTEGER NOT NULL,
                        source_item_ids_json TEXT NOT NULL DEFAULT '[]',
                        leaf_message_ids_json TEXT NOT NULL DEFAULT '[]',
                        metadata_json TEXT NOT NULL DEFAULT '{}',
                        FOREIGN KEY(session_id) REFERENCES sessions(session_id)
                    );

                    CREATE TABLE IF NOT EXISTS active_context_items (
                        session_id TEXT NOT NULL,
                        position INTEGER NOT NULL,
                        kind TEXT NOT NULL,
                        item_id TEXT NOT NULL,
                        token_count INTEGER NOT NULL,
                        PRIMARY KEY(session_id, position),
                        FOREIGN KEY(session_id) REFERENCES sessions(session_id)
                    );

                    CREATE TABLE IF NOT EXISTS facts (
                        id TEXT PRIMARY KEY,
                        namespace TEXT NOT NULL,
                        session_id TEXT,
                        category TEXT NOT NULL,
                        key TEXT NOT NULL,
                        value TEXT NOT NULL,
                        scope TEXT NOT NULL,
                        created_at TEXT NOT NULL,
                        valid_at TEXT,
                        invalid_at TEXT,
                        expired_at TEXT,
                        importance REAL NOT NULL DEFAULT 0.5,
                        status TEXT NOT NULL DEFAULT 'active',
                        source_message_id TEXT,
                        metadata_json TEXT NOT NULL DEFAULT '{}'
                    );

                    CREATE TABLE IF NOT EXISTS memory_events (
                        id TEXT PRIMARY KEY,
                        session_id TEXT NOT NULL,
                        event_type TEXT NOT NULL,
                        classifier TEXT NOT NULL,
                        content TEXT NOT NULL,
                        created_at TEXT NOT NULL,
                        metadata_json TEXT NOT NULL DEFAULT '{}',
                        FOREIGN KEY(session_id) REFERENCES sessions(session_id)
                    );

                    CREATE TABLE IF NOT EXISTS artifacts (
                        id TEXT PRIMARY KEY,
                        session_id TEXT NOT NULL,
                        kind TEXT NOT NULL,
                        path TEXT NOT NULL,
                        preview TEXT NOT NULL,
                        checksum TEXT NOT NULL,
                        created_at TEXT NOT NULL,
                        size_bytes INTEGER NOT NULL DEFAULT 0,
                        mtime REAL,
                        metadata_json TEXT NOT NULL DEFAULT '{}',
                        FOREIGN KEY(session_id) REFERENCES sessions(session_id)
                    );

                    CREATE INDEX IF NOT EXISTS idx_messages_session_time ON messages(session_id, created_at);
                    CREATE INDEX IF NOT EXISTS idx_summaries_session_time ON summaries(session_id, created_at);
                    CREATE INDEX IF NOT EXISTS idx_facts_lookup ON facts(namespace, category, key, scope, status, created_at);
                    CREATE INDEX IF NOT EXISTS idx_events_session_time ON memory_events(session_id, created_at);
                    CREATE INDEX IF NOT EXISTS idx_artifacts_session_time ON artifacts(session_id, created_at);
                    '''
                )
                try:
                    conn.executescript(
                        '''
                        CREATE VIRTUAL TABLE IF NOT EXISTS message_fts USING fts5(
                            message_id UNINDEXED,
                            session_id UNINDEXED,
                            content,
                            metadata,
                            tokenize='unicode61'
                        );
                        CREATE VIRTUAL TABLE IF NOT EXISTS summary_fts USING fts5(
                            summary_id UNINDEXED,
                            session_id UNINDEXED,
                            content,
                            metadata,
                            tokenize='unicode61'
                        );
                        CREATE VIRTUAL TABLE IF NOT EXISTS fact_fts USING fts5(
                            fact_id UNINDEXED,
                            namespace UNINDEXED,
                            session_id UNINDEXED,
                            category,
                            fact_key,
                            value,
                            metadata,
                            tokenize='unicode61'
                        );
                        CREATE VIRTUAL TABLE IF NOT EXISTS artifact_fts USING fts5(
                            artifact_id UNINDEXED,
                            session_id UNINDEXED,
                            content,
                            tokenize='unicode61'
                        );
                        '''
                    )
                except sqlite3.OperationalError:
                    self._fts_enabled = False
                conn.commit()
            finally:
                conn.close()

    def ensure_session(self, session_id: str, *, namespace: str, created_at: str, metadata: dict[str, Any] | None = None) -> None:
        with self._lock:
            conn = self._connect()
            try:
                conn.execute(
                    'INSERT OR IGNORE INTO sessions(session_id, namespace, created_at, metadata_json) VALUES (?, ?, ?, ?)',
                    (session_id, namespace, created_at, json.dumps(metadata or {}, ensure_ascii=False)),
                )
                conn.commit()
            finally:
                conn.close()

    def append_message(self, record: MessageRecord) -> None:
        with self._lock:
            conn = self._connect()
            try:
                conn.execute(
                    '''
                    INSERT INTO messages(id, session_id, turn_id, role, content, created_at, token_count, kind, metadata_json)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''',
                    (
                        record.id,
                        record.session_id,
                        record.turn_id,
                        record.role,
                        record.content,
                        record.created_at,
                        record.token_count,
                        record.kind,
                        json.dumps(record.metadata, ensure_ascii=False),
                    ),
                )
                pos = conn.execute(
                    'SELECT COALESCE(MAX(position), -1) + 1 FROM active_context_items WHERE session_id = ?',
                    (record.session_id,),
                ).fetchone()[0]
                conn.execute(
                    'INSERT INTO active_context_items(session_id, position, kind, item_id, token_count) VALUES (?, ?, ?, ?, ?)',
                    (record.session_id, pos, 'message', record.id, record.token_count),
                )
                conn.execute('UPDATE sessions SET last_turn_id = ? WHERE session_id = ?', (record.turn_id, record.session_id))
                self._index_message(conn, record)
                conn.commit()
            finally:
                conn.close()

    def get_message(self, message_id: str) -> MessageRecord | None:
        with self._lock:
            conn = self._connect()
            try:
                row = conn.execute('SELECT * FROM messages WHERE id = ?', (message_id,)).fetchone()
                return self._row_to_message(row) if row else None
            finally:
                conn.close()

    def list_recent_messages(self, session_id: str, *, limit: int = 12) -> list[MessageRecord]:
        with self._lock:
            conn = self._connect()
            try:
                rows = conn.execute(
                    'SELECT * FROM messages WHERE session_id = ? ORDER BY created_at DESC, rowid DESC LIMIT ?',
                    (session_id, limit),
                ).fetchall()
                items = [self._row_to_message(row) for row in rows]
                items.reverse()
                return items
            finally:
                conn.close()

    def insert_summary(self, node: SummaryNode) -> None:
        with self._lock:
            conn = self._connect()
            try:
                conn.execute(
                    '''
                    INSERT INTO summaries(id, session_id, level, content, created_at, token_count, source_item_ids_json, leaf_message_ids_json, metadata_json)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''',
                    (
                        node.id,
                        node.session_id,
                        node.level,
                        node.content,
                        node.created_at,
                        node.token_count,
                        json.dumps(node.source_item_ids, ensure_ascii=False),
                        json.dumps(node.leaf_message_ids, ensure_ascii=False),
                        json.dumps(node.metadata, ensure_ascii=False),
                    ),
                )
                self._index_summary(conn, node)
                conn.commit()
            finally:
                conn.close()

    def get_summary(self, summary_id: str) -> SummaryNode | None:
        with self._lock:
            conn = self._connect()
            try:
                row = conn.execute('SELECT * FROM summaries WHERE id = ?', (summary_id,)).fetchone()
                return self._row_to_summary(row) if row else None
            finally:
                conn.close()

    def list_recent_summaries(self, session_id: str, *, limit: int = 12) -> list[SummaryNode]:
        with self._lock:
            conn = self._connect()
            try:
                rows = conn.execute(
                    'SELECT * FROM summaries WHERE session_id = ? ORDER BY created_at DESC, rowid DESC LIMIT ?',
                    (session_id, limit),
                ).fetchall()
                items = [self._row_to_summary(row) for row in rows]
                items.reverse()
                return items
            finally:
                conn.close()

    def list_active_items_with_content(self, session_id: str) -> list[tuple[ActiveContextItem, MessageRecord | SummaryNode | None]]:
        with self._lock:
            conn = self._connect()
            try:
                rows = conn.execute(
                    'SELECT * FROM active_context_items WHERE session_id = ? ORDER BY position ASC',
                    (session_id,),
                ).fetchall()
                result: list[tuple[ActiveContextItem, MessageRecord | SummaryNode | None]] = []
                for row in rows:
                    item = ActiveContextItem(
                        position=int(row['position']),
                        kind=row['kind'],
                        item_id=row['item_id'],
                        token_count=int(row['token_count']),
                    )
                    payload: MessageRecord | SummaryNode | None
                    if item.kind == 'message':
                        payload = self.get_message(item.item_id)
                    else:
                        payload = self.get_summary(item.item_id)
                    result.append((item, payload))
                return result
            finally:
                conn.close()

    def replace_active_items_with_summary(
        self,
        session_id: str,
        *,
        start_position: int,
        end_position: int,
        summary_id: str,
        token_count: int,
    ) -> None:
        with self._lock:
            conn = self._connect()
            try:
                rows = conn.execute(
                    'SELECT position FROM active_context_items WHERE session_id = ? AND position BETWEEN ? AND ? ORDER BY position ASC',
                    (session_id, start_position, end_position),
                ).fetchall()
                if not rows:
                    return
                removed_count = len(rows)
                conn.execute(
                    'DELETE FROM active_context_items WHERE session_id = ? AND position BETWEEN ? AND ?',
                    (session_id, start_position, end_position),
                )
                conn.execute(
                    'UPDATE active_context_items SET position = position - ? WHERE session_id = ? AND position > ?',
                    (removed_count - 1, session_id, end_position),
                )
                conn.execute(
                    'INSERT INTO active_context_items(session_id, position, kind, item_id, token_count) VALUES (?, ?, ?, ?, ?)',
                    (session_id, start_position, 'summary', summary_id, token_count),
                )
                conn.commit()
            finally:
                conn.close()

    def add_fact(self, fact: FactRecord) -> None:
        with self._lock:
            conn = self._connect()
            try:
                conn.execute(
                    '''
                    INSERT INTO facts(id, namespace, session_id, category, key, value, scope, created_at, valid_at, invalid_at, expired_at, importance, status, source_message_id, metadata_json)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''',
                    (
                        fact.id,
                        fact.namespace,
                        fact.session_id,
                        fact.category,
                        fact.key,
                        fact.value,
                        fact.scope,
                        fact.created_at,
                        fact.valid_at,
                        fact.invalid_at,
                        fact.expired_at,
                        fact.importance,
                        fact.status,
                        fact.source_message_id,
                        json.dumps(fact.metadata, ensure_ascii=False),
                    ),
                )
                self._index_fact(conn, fact)
                conn.commit()
            finally:
                conn.close()

    def expire_active_facts(
        self,
        *,
        namespace: str,
        category: str,
        key: str,
        scope: str,
        session_id: str | None,
        expired_at: str,
        invalid_at: str | None = None,
    ) -> int:
        where = 'namespace = ? AND category = ? AND key = ? AND scope = ? AND status = "active"'
        params: list[Any] = [namespace, category, key, scope]
        if scope == 'session':
            where += ' AND session_id = ?'
            params.append(session_id)
        elif session_id is not None:
            where += ' AND (session_id IS NULL OR session_id = ?)'
            params.append(session_id)
        with self._lock:
            conn = self._connect()
            try:
                cur = conn.execute(
                    f'''UPDATE facts SET status = 'superseded', expired_at = ?, invalid_at = COALESCE(invalid_at, ?) WHERE {where}''',
                    [expired_at, invalid_at] + params,
                )
                conn.commit()
                return int(cur.rowcount)
            finally:
                conn.close()

    def list_active_facts(
        self,
        *,
        namespace: str,
        session_id: str,
        include_session: bool = True,
        include_cross_session: bool = True,
        limit: int = 200,
    ) -> list[FactRecord]:
        clauses = ['namespace = ?', 'status = "active"']
        params: list[Any] = [namespace]
        scope_clauses: list[str] = []
        if include_cross_session:
            scope_clauses.append("scope = 'cross_session'")
        if include_session:
            scope_clauses.append("(scope = 'session' AND session_id = ?)")
            params.append(session_id)
        if scope_clauses:
            clauses.append('(' + ' OR '.join(scope_clauses) + ')')
        query = 'SELECT * FROM facts WHERE ' + ' AND '.join(clauses) + ' ORDER BY importance DESC, created_at DESC LIMIT ?'
        params.append(limit)
        with self._lock:
            conn = self._connect()
            try:
                rows = conn.execute(query, params).fetchall()
                return [self._row_to_fact(row) for row in rows]
            finally:
                conn.close()

    def add_event(self, event: MemoryEvent) -> None:
        with self._lock:
            conn = self._connect()
            try:
                conn.execute(
                    'INSERT INTO memory_events(id, session_id, event_type, classifier, content, created_at, metadata_json) VALUES (?, ?, ?, ?, ?, ?, ?)',
                    (event.id, event.session_id, event.event_type, event.classifier, event.content, event.created_at, json.dumps(event.metadata, ensure_ascii=False)),
                )
                conn.commit()
            finally:
                conn.close()

    def list_recent_events(self, session_id: str, *, limit: int = 8) -> list[MemoryEvent]:
        with self._lock:
            conn = self._connect()
            try:
                rows = conn.execute(
                    'SELECT * FROM memory_events WHERE session_id = ? ORDER BY created_at DESC, rowid DESC LIMIT ?',
                    (session_id, limit),
                ).fetchall()
                items = [self._row_to_event(row) for row in rows]
                items.reverse()
                return items
            finally:
                conn.close()

    def add_artifact(self, artifact: FileArtifact) -> None:
        with self._lock:
            conn = self._connect()
            try:
                conn.execute(
                    'INSERT INTO artifacts(id, session_id, kind, path, preview, checksum, created_at, size_bytes, mtime, metadata_json) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)',
                    (
                        artifact.id,
                        artifact.session_id,
                        artifact.kind,
                        artifact.path,
                        artifact.preview,
                        artifact.checksum,
                        artifact.created_at,
                        artifact.size_bytes,
                        artifact.mtime,
                        json.dumps(artifact.metadata, ensure_ascii=False),
                    ),
                )
                self._index_artifact(conn, artifact)
                conn.commit()
            finally:
                conn.close()

    def list_recent_artifacts(self, session_id: str, *, limit: int = 8) -> list[FileArtifact]:
        with self._lock:
            conn = self._connect()
            try:
                rows = conn.execute(
                    'SELECT * FROM artifacts WHERE session_id = ? ORDER BY created_at DESC, rowid DESC LIMIT ?',
                    (session_id, limit),
                ).fetchall()
                items = [self._row_to_artifact(row) for row in rows]
                items.reverse()
                return items
            finally:
                conn.close()

    def expand_summary_to_messages(self, summary_id: str) -> list[MessageRecord]:
        summary = self.get_summary(summary_id)
        if summary is None:
            return []
        messages: list[MessageRecord] = []
        for message_id in summary.leaf_message_ids:
            record = self.get_message(message_id)
            if record is not None:
                messages.append(record)
        return messages

    def session_stats(self, session_id: str) -> dict[str, Any]:
        with self._lock:
            conn = self._connect()
            try:
                messages = conn.execute('SELECT COUNT(*) AS count, COALESCE(SUM(token_count), 0) AS tokens FROM messages WHERE session_id = ?', (session_id,)).fetchone()
                summaries = conn.execute('SELECT COUNT(*) AS count, COALESCE(SUM(token_count), 0) AS tokens FROM summaries WHERE session_id = ?', (session_id,)).fetchone()
                active = conn.execute('SELECT COUNT(*) AS count, COALESCE(SUM(token_count), 0) AS tokens FROM active_context_items WHERE session_id = ?', (session_id,)).fetchone()
                facts = conn.execute('SELECT COUNT(*) AS count FROM facts WHERE ((scope = "session" AND session_id = ?) OR scope = "cross_session") AND status = "active"', (session_id,)).fetchone()
                return {
                    'message_count': int(messages['count']),
                    'message_tokens': int(messages['tokens']),
                    'summary_count': int(summaries['count']),
                    'summary_tokens': int(summaries['tokens']),
                    'active_item_count': int(active['count']),
                    'active_tokens': int(active['tokens']),
                    'active_fact_count': int(facts['count']),
                }
            finally:
                conn.close()

    def search_message_candidates(self, session_id: str, query_text: str, *, limit: int = 24) -> list[MessageRecord]:
        with self._lock:
            conn = self._connect()
            try:
                rows: list[sqlite3.Row]
                if self._fts_enabled and query_text.strip():
                    fts_query = self._to_fts_query(query_text)
                    try:
                        rows = conn.execute(
                            '''
                            SELECT m.* FROM message_fts f
                            JOIN messages m ON m.id = f.message_id
                            WHERE f.session_id = ? AND message_fts MATCH ?
                            ORDER BY bm25(message_fts)
                            LIMIT ?
                            ''',
                            (session_id, fts_query, limit),
                        ).fetchall()
                    except sqlite3.OperationalError:
                        rows = []
                else:
                    rows = []
                if not rows:
                    rows = conn.execute(
                        'SELECT * FROM messages WHERE session_id = ? ORDER BY created_at DESC LIMIT ?',
                        (session_id, limit),
                    ).fetchall()
                items = [self._row_to_message(row) for row in rows]
                items.reverse()
                return items
            finally:
                conn.close()

    def search_summary_candidates(self, session_id: str, query_text: str, *, limit: int = 12) -> list[SummaryNode]:
        with self._lock:
            conn = self._connect()
            try:
                rows: list[sqlite3.Row]
                if self._fts_enabled and query_text.strip():
                    fts_query = self._to_fts_query(query_text)
                    try:
                        rows = conn.execute(
                            '''
                            SELECT s.* FROM summary_fts f
                            JOIN summaries s ON s.id = f.summary_id
                            WHERE f.session_id = ? AND summary_fts MATCH ?
                            ORDER BY bm25(summary_fts)
                            LIMIT ?
                            ''',
                            (session_id, fts_query, limit),
                        ).fetchall()
                    except sqlite3.OperationalError:
                        rows = []
                else:
                    rows = []
                if not rows:
                    rows = conn.execute(
                        'SELECT * FROM summaries WHERE session_id = ? ORDER BY created_at DESC LIMIT ?',
                        (session_id, limit),
                    ).fetchall()
                items = [self._row_to_summary(row) for row in rows]
                items.reverse()
                return items
            finally:
                conn.close()

    def search_artifact_candidates(self, session_id: str, query_text: str, *, limit: int = 10) -> list[FileArtifact]:
        with self._lock:
            conn = self._connect()
            try:
                rows: list[sqlite3.Row]
                if self._fts_enabled and query_text.strip():
                    fts_query = self._to_fts_query(query_text)
                    try:
                        rows = conn.execute(
                            '''
                            SELECT a.* FROM artifact_fts f
                            JOIN artifacts a ON a.id = f.artifact_id
                            WHERE f.session_id = ? AND artifact_fts MATCH ?
                            ORDER BY bm25(artifact_fts)
                            LIMIT ?
                            ''',
                            (session_id, fts_query, limit),
                        ).fetchall()
                    except sqlite3.OperationalError:
                        rows = []
                else:
                    rows = []
                if not rows:
                    rows = conn.execute(
                        'SELECT * FROM artifacts WHERE session_id = ? ORDER BY created_at DESC LIMIT ?',
                        (session_id, limit),
                    ).fetchall()
                items = [self._row_to_artifact(row) for row in rows]
                items.reverse()
                return items
            finally:
                conn.close()

    def _to_fts_query(self, text: str) -> str:
        tokens = [token.strip() for token in text.replace(':', ' ').replace('/', ' ').split() if token.strip()]
        if not tokens:
            return '""'
        safe = [token.replace('"', '') for token in tokens if token]
        return ' OR '.join(safe[:12])

    def _index_message(self, conn: sqlite3.Connection, record: MessageRecord) -> None:
        if not self._fts_enabled:
            return
        conn.execute('DELETE FROM message_fts WHERE message_id = ?', (record.id,))
        conn.execute(
            'INSERT INTO message_fts(message_id, session_id, content, metadata) VALUES (?, ?, ?, ?)',
            (record.id, record.session_id, record.content, json.dumps(record.metadata, ensure_ascii=False)),
        )

    def _index_summary(self, conn: sqlite3.Connection, node: SummaryNode) -> None:
        if not self._fts_enabled:
            return
        conn.execute('DELETE FROM summary_fts WHERE summary_id = ?', (node.id,))
        conn.execute(
            'INSERT INTO summary_fts(summary_id, session_id, content, metadata) VALUES (?, ?, ?, ?)',
            (node.id, node.session_id, node.content, json.dumps(node.metadata, ensure_ascii=False)),
        )

    def _index_fact(self, conn: sqlite3.Connection, fact: FactRecord) -> None:
        if not self._fts_enabled:
            return
        conn.execute('DELETE FROM fact_fts WHERE fact_id = ?', (fact.id,))
        conn.execute(
            'INSERT INTO fact_fts(fact_id, namespace, session_id, category, fact_key, value, metadata) VALUES (?, ?, ?, ?, ?, ?, ?)',
            (
                fact.id,
                fact.namespace,
                fact.session_id or '',
                fact.category,
                fact.key,
                fact.value,
                json.dumps(fact.metadata, ensure_ascii=False),
            ),
        )

    def _index_artifact(self, conn: sqlite3.Connection, artifact: FileArtifact) -> None:
        if not self._fts_enabled:
            return
        conn.execute('DELETE FROM artifact_fts WHERE artifact_id = ?', (artifact.id,))
        conn.execute(
            'INSERT INTO artifact_fts(artifact_id, session_id, content) VALUES (?, ?, ?)',
            (artifact.id, artifact.session_id, f'{artifact.kind} {artifact.path} {artifact.preview} {json.dumps(artifact.metadata, ensure_ascii=False)}'),
        )

    def _row_to_message(self, row: sqlite3.Row | None) -> MessageRecord:
        if row is None:
            raise KeyError('message row missing')
        return MessageRecord(
            id=row['id'],
            session_id=row['session_id'],
            turn_id=row['turn_id'],
            role=row['role'],
            content=row['content'],
            created_at=row['created_at'],
            token_count=int(row['token_count']),
            kind=row['kind'],
            metadata=json.loads(row['metadata_json'] or '{}'),
        )

    def _row_to_summary(self, row: sqlite3.Row | None) -> SummaryNode:
        if row is None:
            raise KeyError('summary row missing')
        return SummaryNode(
            id=row['id'],
            session_id=row['session_id'],
            level=int(row['level']),
            content=row['content'],
            created_at=row['created_at'],
            token_count=int(row['token_count']),
            source_item_ids=json.loads(row['source_item_ids_json'] or '[]'),
            leaf_message_ids=json.loads(row['leaf_message_ids_json'] or '[]'),
            metadata=json.loads(row['metadata_json'] or '{}'),
        )

    def _row_to_fact(self, row: sqlite3.Row | None) -> FactRecord:
        if row is None:
            raise KeyError('fact row missing')
        return FactRecord(
            id=row['id'],
            namespace=row['namespace'],
            session_id=row['session_id'],
            category=row['category'],
            key=row['key'],
            value=row['value'],
            scope=row['scope'],
            created_at=row['created_at'],
            valid_at=row['valid_at'],
            invalid_at=row['invalid_at'],
            expired_at=row['expired_at'],
            importance=float(row['importance']),
            status=row['status'],
            source_message_id=row['source_message_id'],
            metadata=json.loads(row['metadata_json'] or '{}'),
        )

    def _row_to_event(self, row: sqlite3.Row | None) -> MemoryEvent:
        if row is None:
            raise KeyError('event row missing')
        return MemoryEvent(
            id=row['id'],
            session_id=row['session_id'],
            event_type=row['event_type'],
            classifier=row['classifier'],
            content=row['content'],
            created_at=row['created_at'],
            metadata=json.loads(row['metadata_json'] or '{}'),
        )

    def _row_to_artifact(self, row: sqlite3.Row | None) -> FileArtifact:
        if row is None:
            raise KeyError('artifact row missing')
        return FileArtifact(
            id=row['id'],
            session_id=row['session_id'],
            kind=row['kind'],
            path=row['path'],
            preview=row['preview'],
            checksum=row['checksum'],
            created_at=row['created_at'],
            size_bytes=int(row['size_bytes']),
            mtime=row['mtime'],
            metadata=json.loads(row['metadata_json'] or '{}'),
        )


def new_id(prefix: str) -> str:
    return f'{prefix}_{uuid.uuid4().hex[:16]}'
