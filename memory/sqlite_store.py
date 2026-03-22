"""
memory/sqlite_store.py  —  Long-term memory: request history + human rejection feedback.

Stores every OCR request result and any human rejection reasons so the
supervisor agent can learn from them and adjust future decisions.
"""
from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime
from pathlib import Path
from typing import Any

from config.settings import MEMORY_DB


class AgentMemory:
    """Thread-safe SQLite-backed long-term memory for the multi-agent pipeline."""

    def __init__(self, db_path: Path = MEMORY_DB):
        self._db_path = db_path
        self._lock = threading.Lock()
        self._init_db()

    # ── Schema ────────────────────────────────────────────────────────
    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS ocr_requests (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at  TEXT NOT NULL,
                    session_id  TEXT,
                    raw_text    TEXT,
                    extracted   TEXT,
                    status      TEXT DEFAULT 'pending'
                );

                CREATE TABLE IF NOT EXISTS human_feedback (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    request_id  INTEGER REFERENCES ocr_requests(id),
                    created_at  TEXT NOT NULL,
                    decision    TEXT NOT NULL,
                    reason      TEXT,
                    field       TEXT,
                    correction  TEXT
                );

                CREATE TABLE IF NOT EXISTS rejection_patterns (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    pattern     TEXT NOT NULL,
                    count       INTEGER DEFAULT 1,
                    last_seen   TEXT NOT NULL
                );
            """)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    # ── Request tracking ──────────────────────────────────────────────
    def create_request(self, session_id: str, raw_text: str = "") -> int:
        ts = datetime.utcnow().isoformat()
        with self._lock, self._connect() as conn:
            cur = conn.execute(
                "INSERT INTO ocr_requests (created_at, session_id, raw_text) VALUES (?,?,?)",
                (ts, session_id, raw_text),
            )
            return cur.lastrowid  # type: ignore[return-value]

    def update_request(self, req_id: int, extracted: dict[str, Any], status: str) -> None:
        with self._lock, self._connect() as conn:
            conn.execute(
                "UPDATE ocr_requests SET extracted=?, status=? WHERE id=?",
                (json.dumps(extracted, ensure_ascii=False), status, req_id),
            )

    # ── Human feedback ────────────────────────────────────────────────
    def store_feedback(
        self,
        request_id: int,
        decision: str,
        reason: str = "",
        field: str = "",
        correction: str = "",
    ) -> None:
        """Persist a human approval/rejection decision with optional correction."""
        ts = datetime.utcnow().isoformat()
        with self._lock, self._connect() as conn:
            conn.execute(
                "INSERT INTO human_feedback (request_id, created_at, decision, reason, field, correction)"
                " VALUES (?,?,?,?,?,?)",
                (request_id, ts, decision, reason, field, correction),
            )

        # Track rejection patterns for learning
        if decision == "rejected" and reason:
            self._record_rejection_pattern(reason)

    def _record_rejection_pattern(self, reason: str) -> None:
        ts = datetime.utcnow().isoformat()
        # Normalize: lower, first 120 chars
        pattern = reason.lower().strip()[:120]
        with self._lock, self._connect() as conn:
            existing = conn.execute(
                "SELECT id, count FROM rejection_patterns WHERE pattern=?", (pattern,)
            ).fetchone()
            if existing:
                conn.execute(
                    "UPDATE rejection_patterns SET count=count+1, last_seen=? WHERE id=?",
                    (ts, existing["id"]),
                )
            else:
                conn.execute(
                    "INSERT INTO rejection_patterns (pattern, last_seen) VALUES (?,?)",
                    (pattern, ts),
                )

    # ── Query helpers ─────────────────────────────────────────────────
    def get_top_rejection_patterns(self, limit: int = 5) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT pattern, count FROM rejection_patterns ORDER BY count DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [dict(r) for r in rows]

    def get_recent_approvals(self, limit: int = 10) -> list[dict]:
        """Return recently approved extractions as few-shot examples for the LLM."""
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT r.extracted FROM ocr_requests r
                JOIN human_feedback f ON f.request_id = r.id
                WHERE f.decision = 'approved' AND r.extracted IS NOT NULL
                ORDER BY f.created_at DESC LIMIT ?
                """,
                (limit,),
            ).fetchall()
        results = []
        for row in rows:
            try:
                results.append(json.loads(row["extracted"]))
            except (json.JSONDecodeError, TypeError):
                pass
        return results
