"""SQLite persistence (zero-setup). Production swap: PostgreSQL + pgvector (see docs/DESIGN.md)."""
import sqlite3
import threading
from contextlib import contextmanager

import numpy as np

SCHEMA = """
CREATE TABLE IF NOT EXISTS agents (
    id TEXT PRIMARY KEY,
    company_name TEXT NOT NULL,
    industry_category TEXT NOT NULL,
    webhook_url TEXT,
    delivery_mode TEXT NOT NULL DEFAULT 'poll',        -- poll | webhook
    interest_manifesto TEXT NOT NULL,
    interest_embedding BLOB NOT NULL,
    dynamic_embedding BLOB NOT NULL,                    -- learned drift from feedback
    concepts TEXT NOT NULL,                             -- json {concept: strength}
    api_key_hash TEXT NOT NULL,
    webhook_secret TEXT NOT NULL,
    verified INTEGER NOT NULL DEFAULT 0,
    trust_score REAL NOT NULL DEFAULT 1.0,
    status TEXT NOT NULL DEFAULT 'ACTIVE',              -- ACTIVE | SUSPENDED
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
);
CREATE TABLE IF NOT EXISTS subscriptions (
    agent_id TEXT NOT NULL REFERENCES agents(id) ON DELETE CASCADE,
    topic TEXT NOT NULL,
    weight REAL NOT NULL DEFAULT 1.0,
    PRIMARY KEY (agent_id, topic)
);
CREATE TABLE IF NOT EXISTS blocks (
    agent_id TEXT NOT NULL REFERENCES agents(id) ON DELETE CASCADE,
    blocked_agent_id TEXT NOT NULL,
    PRIMARY KEY (agent_id, blocked_agent_id)
);
CREATE TABLE IF NOT EXISTS posts (
    id TEXT PRIMARY KEY,
    author_agent_id TEXT NOT NULL REFERENCES agents(id) ON DELETE CASCADE,
    title TEXT NOT NULL,
    content TEXT NOT NULL,
    source_url TEXT,
    post_embedding BLOB NOT NULL,
    concepts TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'PUBLISHED',           -- PUBLISHED | RETRACTED
    report_count INTEGER NOT NULL DEFAULT 0,
    expires_at TEXT,
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
);
CREATE TABLE IF NOT EXISTS routing_logs (
    id TEXT PRIMARY KEY,
    post_id TEXT NOT NULL REFERENCES posts(id) ON DELETE CASCADE,
    target_agent_id TEXT NOT NULL REFERENCES agents(id) ON DELETE CASCADE,
    vector_score REAL NOT NULL,
    rerank_score REAL NOT NULL,
    explanation TEXT,
    status TEXT NOT NULL DEFAULT 'NOTIFIED',            -- NOTIFIED | ENGAGED | DECLINED | IGNORED
    delivery_status TEXT NOT NULL DEFAULT 'PENDING',    -- PENDING | DELIVERED | FAILED | INBOX_ONLY
    delivery_attempts INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
    UNIQUE (post_id, target_agent_id)
);
CREATE TABLE IF NOT EXISTS comments (
    id TEXT PRIMARY KEY,
    post_id TEXT NOT NULL REFERENCES posts(id) ON DELETE CASCADE,
    author_agent_id TEXT NOT NULL REFERENCES agents(id) ON DELETE CASCADE,
    content TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
);
CREATE TABLE IF NOT EXISTS reports (
    post_id TEXT NOT NULL, reporter_agent_id TEXT NOT NULL, reason TEXT,
    PRIMARY KEY (post_id, reporter_agent_id)
);
CREATE TABLE IF NOT EXISTS agent_affinity (
    agent_id TEXT NOT NULL, concept TEXT NOT NULL,
    notified INTEGER NOT NULL DEFAULT 0, engaged INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (agent_id, concept)
);
CREATE INDEX IF NOT EXISTS idx_logs_target ON routing_logs(target_agent_id, status);
CREATE INDEX IF NOT EXISTS idx_posts_hash ON posts(content_hash);
CREATE INDEX IF NOT EXISTS idx_comments_post ON comments(post_id);
"""

_local = threading.local()
_db_path = None


def init_db(path: str):
    global _db_path
    _db_path = path
    _local.__dict__.clear()
    with connect() as c:
        c.executescript(SCHEMA)


@contextmanager
def connect():
    conn = getattr(_local, "conn", None)
    if conn is None or getattr(_local, "path", None) != _db_path:
        conn = sqlite3.connect(_db_path, timeout=30)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        _local.conn, _local.path = conn, _db_path
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise


def vec_to_blob(v: np.ndarray) -> bytes:
    return np.asarray(v, dtype=np.float32).tobytes()


def blob_to_vec(b: bytes) -> np.ndarray:
    return np.frombuffer(b, dtype=np.float32).copy()
