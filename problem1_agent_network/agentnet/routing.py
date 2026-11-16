"""Multi-stage relevance routing.

Stage A  SQL guardrails   : active, not author, not blocked, not already notified.
Stage B  Dense retrieval  : cosine similarity vs. effective interest vectors, top-K.
         (brute-force numpy here; pgvector HNSW in production = O(log N))
Stage C  Rerank           : hybrid dense + lexical + subscriptions + learned affinity + trust,
                            then threshold and cap on notifications per post.
"""
import json
import time
import uuid
from typing import List

import numpy as np

from . import delivery, feedback
from .config import settings
from .db import blob_to_vec, connect
from .embeddings import normalize
from .rerank import heuristic_score, llm_judge


def route_post(post_id: str) -> List[dict]:
    t0 = time.perf_counter()
    with connect() as conn:
        post = conn.execute("SELECT * FROM posts WHERE id=?", (post_id,)).fetchone()
        if not post or post["status"] != "PUBLISHED":
            return []
        author = conn.execute("SELECT * FROM agents WHERE id=?", (post["author_agent_id"],)).fetchone()
        post_vec = blob_to_vec(post["post_embedding"])
        post_concepts = json.loads(post["concepts"])

        # ---- Stage A: SQL guardrails -------------------------------------------------------
        rows = conn.execute("""
            SELECT a.* FROM agents a
            WHERE a.status='ACTIVE' AND a.id != :author
              AND NOT EXISTS (SELECT 1 FROM blocks b WHERE b.agent_id=a.id AND b.blocked_agent_id=:author)
              AND NOT EXISTS (SELECT 1 FROM blocks b WHERE b.agent_id=:author AND b.blocked_agent_id=a.id)
              AND NOT EXISTS (SELECT 1 FROM routing_logs l WHERE l.post_id=:post AND l.target_agent_id=a.id)
        """, {"author": author["id"], "post": post_id}).fetchall()
        if not rows:
            return []
        stage_a = len(rows)

        # ---- Stage B: dense ANN (matrix cosine) ---------------------------------------------
        rows = [r for r in rows if len(blob_to_vec(r["interest_embedding"])) == len(post_vec)]  # stale dims: run /admin/reembed
        if not rows:
            return []
        mat = np.stack([normalize(blob_to_vec(r["interest_embedding"]) + blob_to_vec(r["dynamic_embedding"]))
                        for r in rows])
        sims = np.clip(mat @ post_vec, 0.0, 1.0)
        order = np.argsort(-sims)[: settings.top_k]
        shortlist = [(rows[i], float(sims[i])) for i in order]

        subs = {}
        for s in conn.execute("SELECT agent_id, topic, weight FROM subscriptions"):
            subs.setdefault(s["agent_id"], {})[s["topic"]] = s["weight"]

        # ---- Stage C: rerank -----------------------------------------------------------------
        scored = []
        author_trust = author["trust_score"] * (1.0 if author["verified"] else settings.unverified_trust)
        for agent, dense in shortlist:
            if dense < settings.dense_floor:
                continue
            my_subs = subs.get(agent["id"], {})
            topic_bonus = max([w for t, w in my_subs.items() if t in post_concepts], default=0.0)
            affinity = feedback.affinity_for(conn, agent["id"], post_concepts)
            score, why = heuristic_score(dense, post_concepts, dict(agent), json.loads(agent["concepts"]),
                                         topic_bonus, affinity, author_trust)
            if settings.rerank_mode == "llm":
                j = llm_judge(dict(post), dict(agent))
                if j >= 0:
                    score, why = 0.5 * score + 0.5 * j, why + f" llm={j:.2f}"
            scored.append((agent, dense, score, why))

        scored.sort(key=lambda x: -x[2])
        chosen = [s for s in scored if s[2] >= settings.threshold][: settings.max_notify]

        decisions = []
        for agent, dense, score, why in chosen:
            log_id = str(uuid.uuid4())
            conn.execute("""INSERT INTO routing_logs(id, post_id, target_agent_id, vector_score, rerank_score, explanation)
                            VALUES (?,?,?,?,?,?)""", (log_id, post_id, agent["id"], dense, score, why))
            feedback.record_notified(conn, agent["id"], post_concepts)
            decisions.append({"log_id": log_id, "agent_id": agent["id"], "company": agent["company_name"],
                              "vector_score": round(dense, 3), "rerank_score": round(score, 3)})
    latency_ms = (time.perf_counter() - t0) * 1000
    for d in decisions:
        delivery.deliver(d["log_id"])
    for d in decisions:
        d["routing_latency_ms"] = round(latency_ms, 2)
        d["stage_a_candidates"] = stage_a
    return decisions
