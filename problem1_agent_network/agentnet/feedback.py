"""Closed-loop learning.

Two signals per (agent, concept): affinity = smoothed engagement rate  (engaged+1)/(notified+2)
and a dynamic embedding that drifts toward posts the agent engaged with and away from ignored ones
(a Rocchio-style update). Effective interest vector = normalize(manifesto_vec + dynamic_vec).
"""
import json
from datetime import datetime, timedelta, timezone

from .config import settings
from .db import blob_to_vec, connect, vec_to_blob

LR_ENGAGE, LR_IGNORE, DRIFT_CAP = 0.15, 0.05, 0.6


def _bump(conn, agent_id, concepts, engaged: bool, first_notice: bool):
    for c in concepts:
        conn.execute("INSERT OR IGNORE INTO agent_affinity(agent_id, concept) VALUES (?,?)", (agent_id, c))
        if first_notice:
            conn.execute("UPDATE agent_affinity SET notified=notified+1 WHERE agent_id=? AND concept=?",
                         (agent_id, c))
        if engaged:
            conn.execute("UPDATE agent_affinity SET engaged=engaged+1 WHERE agent_id=? AND concept=?",
                         (agent_id, c))


def _drift(conn, agent_id, post_vec, lr):
    row = conn.execute("SELECT dynamic_embedding FROM agents WHERE id=?", (agent_id,)).fetchone()
    dyn = blob_to_vec(row["dynamic_embedding"]) + lr * post_vec
    n = (dyn ** 2).sum() ** 0.5
    if n > DRIFT_CAP:
        dyn = dyn * (DRIFT_CAP / n)
    conn.execute("UPDATE agents SET dynamic_embedding=? WHERE id=?", (vec_to_blob(dyn), agent_id))


def record_notified(conn, agent_id, post_concepts):
    _bump(conn, agent_id, post_concepts, engaged=False, first_notice=True)


def record_engaged(conn, log_id: str):
    row = conn.execute("""SELECT l.target_agent_id a, l.status s, p.concepts c, p.post_embedding e
                          FROM routing_logs l JOIN posts p ON p.id=l.post_id WHERE l.id=?""", (log_id,)).fetchone()
    if not row or row["s"] == "ENGAGED":
        return
    conn.execute("UPDATE routing_logs SET status='ENGAGED' WHERE id=?", (log_id,))
    _bump(conn, row["a"], json.loads(row["c"]), engaged=True, first_notice=False)
    _drift(conn, row["a"], blob_to_vec(row["e"]), LR_ENGAGE)


def record_negative(conn, log_id: str, status: str):
    """status: DECLINED (explicit) or IGNORED (timeout). Pushes the agent's vector away from the post."""
    row = conn.execute("""SELECT l.target_agent_id a, l.status s, p.post_embedding e
                          FROM routing_logs l JOIN posts p ON p.id=l.post_id WHERE l.id=?""", (log_id,)).fetchone()
    if not row or row["s"] != "NOTIFIED":
        return
    conn.execute("UPDATE routing_logs SET status=? WHERE id=?", (status, log_id))
    _drift(conn, row["a"], blob_to_vec(row["e"]), -LR_IGNORE)


def sweep_ignored(hours: float = None) -> int:
    hours = settings.ignore_after_hours if hours is None else hours
    cutoff = (datetime.now(timezone.utc) - timedelta(hours=hours)).strftime("%Y-%m-%dT%H:%M:%fZ")
    with connect() as conn:
        ids = [r["id"] for r in conn.execute(
            "SELECT id FROM routing_logs WHERE status='NOTIFIED' AND created_at<=?", (cutoff,))]
        for i in ids:
            record_negative(conn, i, "IGNORED")
    return len(ids)


def affinity_for(conn, agent_id: str, post_concepts) -> float:
    """Weighted mean smoothed engagement rate over the post's concepts; 0.5 when no history."""
    if not post_concepts:
        return 0.5
    num = den = 0.0
    for c, s in post_concepts.items():
        r = conn.execute("SELECT notified n, engaged e FROM agent_affinity WHERE agent_id=? AND concept=?",
                         (agent_id, c)).fetchone()
        rate = (r["e"] + 1) / (r["n"] + 2) if r else 0.5
        num += s * rate
        den += s
    return num / den
