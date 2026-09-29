"""SQLite logging of attempts and failures."""

from __future__ import annotations

import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from .schemas import CallRecord

_SCHEMA = """
CREATE TABLE IF NOT EXISTS attempts (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id       TEXT NOT NULL,
    input_hash   TEXT NOT NULL,
    attempt_no   INTEGER NOT NULL,
    raw_output   TEXT,
    error_type   TEXT,
    error_detail TEXT,
    success      INTEGER NOT NULL,
    latency_ms   INTEGER,
    tokens_in    INTEGER,
    tokens_out   INTEGER,
    created_at   TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_attempts_run ON attempts(run_id);
"""


class AttemptLogger:
    def __init__(self, path: str | Path | None = None):
        self.path = str(path or os.getenv("AGENT_DB", "agent.db"))
        self.conn = sqlite3.connect(self.path)
        self.conn.executescript(_SCHEMA)

    def log(self, run_id: str, input_hash: str, rec: CallRecord) -> None:
        self.conn.execute(
            "INSERT INTO attempts (run_id, input_hash, attempt_no, raw_output, error_type,"
            " error_detail, success, latency_ms, tokens_in, tokens_out, created_at)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                run_id, input_hash, rec.attempt_no, rec.raw_output, rec.error_type,
                rec.error_detail, int(rec.success), rec.latency_ms, rec.tokens_in,
                rec.tokens_out, datetime.now(timezone.utc).isoformat(),
            ),
        )
        self.conn.commit()

    def rows(self, run_id: str | None = None) -> list[sqlite3.Row]:
        self.conn.row_factory = sqlite3.Row
        q = "SELECT * FROM attempts" + (" WHERE run_id = ?" if run_id else "") + " ORDER BY id"
        return self.conn.execute(q, (run_id,) if run_id else ()).fetchall()

    def close(self) -> None:
        self.conn.close()
