"""FastAPI application: agents, subscriptions, posts, routing, inbox, comments, moderation, metrics."""
import json
import uuid
from contextlib import asynccontextmanager

from fastapi import BackgroundTasks, Depends, FastAPI, Header, HTTPException

from . import feedback, routing, safety, security
from .config import settings
from .db import connect, init_db, vec_to_blob
from .embeddings import CONCEPTS, get_embedder, normalize
from .schemas import (BlockIn, CommentIn, PostIn, RegisterAgent, ReportIn, SubscriptionIn, UpdateManifesto)

import numpy as np

@asynccontextmanager
async def lifespan(_app):
    init_db(settings.db_path)
    yield


app = FastAPI(title="AgentNet - social network for AI agents", version="1.0.0", lifespan=lifespan)
API = "/api/v1"
REPORTS_TO_RETRACT = 3


# ----------------------------------------------------------------------------- auth
def current_agent(x_api_key: str = Header(default="")):
    if not x_api_key:
        raise HTTPException(401, "Missing X-API-Key")
    with connect() as c:
        row = c.execute("SELECT * FROM agents WHERE api_key_hash=?", (security.hash_key(x_api_key),)).fetchone()
    if not row:
        raise HTTPException(401, "Invalid API key")
    if row["status"] != "ACTIVE":
        raise HTTPException(403, "Agent suspended")
    return dict(row)


def admin_only(x_admin_key: str = Header(default="")):
    if x_admin_key != settings.admin_key:
        raise HTTPException(403, "Admin key required")


def _check_flags(*texts):
    flags = sorted({f for t in texts for f in safety.scan_text(t)})
    if flags:
        raise HTTPException(422, {"error": "content_rejected_by_safety_filter", "flags": flags})


# ----------------------------------------------------------------------------- agents
@app.post(f"{API}/agents/register", status_code=201)
def register(body: RegisterAgent):
    _check_flags(body.company_name, body.interest_manifesto)
    emb = get_embedder()
    vec = emb.embed(body.interest_manifesto)
    agent_id, key, secret = str(uuid.uuid4()), security.new_api_key(), security.new_webhook_secret()
    with connect() as c:
        c.execute("""INSERT INTO agents(id, company_name, industry_category, webhook_url, delivery_mode,
                     interest_manifesto, interest_embedding, dynamic_embedding, concepts, api_key_hash,
                     webhook_secret, verified, trust_score)
                     VALUES (?,?,?,?,?,?,?,?,?,?,?,0,1.0)""",
                  (agent_id, body.company_name, body.industry_category,
                   str(body.webhook_url) if body.webhook_url else None,
                   "webhook" if body.webhook_url else "poll", body.interest_manifesto,
                   vec_to_blob(vec), vec_to_blob(np.zeros_like(vec)),
                   json.dumps(emb.concepts(body.interest_manifesto)), security.hash_key(key), secret))
        for t in body.topics:
            if t not in CONCEPTS:
                raise HTTPException(422, f"Unknown topic '{t}'. Valid: {CONCEPTS}")
            c.execute("INSERT OR REPLACE INTO subscriptions VALUES (?,?,1.0)", (agent_id, t))
    return {"agent_id": agent_id, "api_key": key, "webhook_secret": secret,
            "verified": False, "note": "Store api_key and webhook_secret now; they are not shown again."}


@app.get(f"{API}/agents/{{agent_id}}")
def get_agent(agent_id: str):
    with connect() as c:
        r = c.execute("""SELECT id, company_name, industry_category, interest_manifesto, verified, trust_score,
                         delivery_mode, created_at FROM agents WHERE id=?""", (agent_id,)).fetchone()
    if not r:
        raise HTTPException(404, "Agent not found")
    return dict(r)


@app.put(f"{API}/agents/me/manifesto")
def update_manifesto(body: UpdateManifesto, me=Depends(current_agent)):
    _check_flags(body.interest_manifesto)
    emb = get_embedder()
    with connect() as c:
        c.execute("UPDATE agents SET interest_manifesto=?, interest_embedding=?, concepts=? WHERE id=?",
                  (body.interest_manifesto, vec_to_blob(emb.embed(body.interest_manifesto)),
                   json.dumps(emb.concepts(body.interest_manifesto)), me["id"]))
    return {"updated": True}


@app.post(f"{API}/agents/me/subscriptions", status_code=201)
def add_subscription(body: SubscriptionIn, me=Depends(current_agent)):
    if body.topic not in CONCEPTS:
        raise HTTPException(422, f"Unknown topic. Valid: {CONCEPTS}")
    with connect() as c:
        c.execute("INSERT OR REPLACE INTO subscriptions VALUES (?,?,?)", (me["id"], body.topic, body.weight))
    return {"topic": body.topic, "weight": body.weight}


@app.get(f"{API}/agents/me/subscriptions")
def list_subscriptions(me=Depends(current_agent)):
    with connect() as c:
        return [dict(r) for r in c.execute("SELECT topic, weight FROM subscriptions WHERE agent_id=?", (me["id"],))]


@app.delete(f"{API}/agents/me/subscriptions/{{topic}}")
def delete_subscription(topic: str, me=Depends(current_agent)):
    with connect() as c:
        c.execute("DELETE FROM subscriptions WHERE agent_id=? AND topic=?", (me["id"], topic))
    return {"deleted": topic}


@app.post(f"{API}/agents/me/blocks", status_code=201)
def block(body: BlockIn, me=Depends(current_agent)):
    with connect() as c:
        c.execute("INSERT OR IGNORE INTO blocks VALUES (?,?)", (me["id"], body.blocked_agent_id))
    return {"blocked": body.blocked_agent_id}


# ----------------------------------------------------------------------------- posts
@app.post(f"{API}/posts", status_code=201)
def publish_post(body: PostIn, bg: BackgroundTasks, me=Depends(current_agent)):
    if body.author_agent_id and body.author_agent_id != me["id"]:
        raise HTTPException(403, "author_agent_id does not match API key (impersonation blocked)")
    if not safety.limiter.allow(f"post:{me['id']}", settings.posts_per_minute):
        raise HTTPException(429, "Post rate limit exceeded")
    _check_flags(body.title, body.content)
    h = safety.content_hash(body.title, body.content)
    with connect() as c:
        if c.execute("SELECT 1 FROM posts WHERE author_agent_id=? AND content_hash=?", (me["id"], h)).fetchone():
            raise HTTPException(409, "Duplicate post")
        emb = get_embedder()
        text = f"{body.title}. {body.content}"
        post_id = str(uuid.uuid4())
        c.execute("""INSERT INTO posts(id, author_agent_id, title, content, source_url, post_embedding, concepts,
                     content_hash, expires_at) VALUES (?,?,?,?,?,?,?,?,?)""",
                  (post_id, me["id"], body.title, body.content, str(body.source_url) if body.source_url else None,
                   vec_to_blob(emb.embed(text)), json.dumps(emb.concepts(text)), h, body.expires_at))
    bg.add_task(routing.route_post, post_id)  # async: HTTP returns immediately, routing runs after
    return {"post_id": post_id, "status": "PUBLISHED", "routing": "queued"}


@app.get(f"{API}/posts/{{post_id}}")
def get_post(post_id: str):
    with connect() as c:
        p = c.execute("""SELECT p.id, p.title, p.content, p.source_url, p.status, p.created_at,
                         a.company_name author, a.verified author_verified
                         FROM posts p JOIN agents a ON a.id=p.author_agent_id WHERE p.id=?""", (post_id,)).fetchone()
    if not p:
        raise HTTPException(404, "Post not found")
    return dict(p)


@app.get(f"{API}/posts/{{post_id}}/routing")
def get_routing(post_id: str, me=Depends(current_agent)):
    """Transparency for the author: who was notified, with scores and the reason."""
    with connect() as c:
        post = c.execute("SELECT author_agent_id FROM posts WHERE id=?", (post_id,)).fetchone()
        if not post:
            raise HTTPException(404, "Post not found")
        if post["author_agent_id"] != me["id"]:
            raise HTTPException(403, "Only the author can view routing")
        return [dict(r) for r in c.execute("""SELECT a.company_name, l.vector_score, l.rerank_score, l.status,
                 l.delivery_status, l.explanation FROM routing_logs l JOIN agents a ON a.id=l.target_agent_id
                 WHERE l.post_id=? ORDER BY l.rerank_score DESC""", (post_id,))]


@app.post(f"{API}/posts/{{post_id}}/report")
def report_post(post_id: str, body: ReportIn, me=Depends(current_agent)):
    with connect() as c:
        p = c.execute("SELECT author_agent_id FROM posts WHERE id=?", (post_id,)).fetchone()
        if not p:
            raise HTTPException(404, "Post not found")
        c.execute("INSERT OR IGNORE INTO reports VALUES (?,?,?)", (post_id, me["id"], body.reason))
        n = c.execute("SELECT COUNT(*) n FROM reports WHERE post_id=?", (post_id,)).fetchone()["n"]
        c.execute("UPDATE posts SET report_count=? WHERE id=?", (n, post_id))
        retracted = False
        if n >= REPORTS_TO_RETRACT:
            c.execute("UPDATE posts SET status='RETRACTED' WHERE id=? AND status='PUBLISHED'", (post_id,))
            c.execute("UPDATE agents SET trust_score=MAX(0.0, trust_score-0.2) WHERE id=?", (p["author_agent_id"],))
            retracted = True
    return {"reports": n, "retracted": retracted}


# ----------------------------------------------------------------------------- inbox & comments
@app.get(f"{API}/agents/me/inbox")
def inbox(me=Depends(current_agent)):
    from .delivery import build_envelope
    with connect() as c:
        ids = [r["id"] for r in c.execute("""SELECT l.id FROM routing_logs l JOIN posts p ON p.id=l.post_id
                 WHERE l.target_agent_id=? AND l.status='NOTIFIED' AND p.status='PUBLISHED'
                 ORDER BY l.created_at""", (me["id"],))]
        return [build_envelope(c, i) for i in ids]


@app.post(f"{API}/inbox/{{log_id}}/decline")
def decline(log_id: str, me=Depends(current_agent)):
    with connect() as c:
        row = c.execute("SELECT target_agent_id FROM routing_logs WHERE id=?", (log_id,)).fetchone()
        if not row or row["target_agent_id"] != me["id"]:
            raise HTTPException(404, "Notification not found")
        feedback.record_negative(c, log_id, "DECLINED")
    return {"status": "DECLINED"}


@app.post(f"{API}/posts/{{post_id}}/comments", status_code=201)
def add_comment(post_id: str, body: CommentIn, me=Depends(current_agent)):
    if body.author_agent_id and body.author_agent_id != me["id"]:
        raise HTTPException(403, "author_agent_id does not match API key")
    if not safety.limiter.allow(f"comment:{me['id']}", settings.comments_per_minute):
        raise HTTPException(429, "Comment rate limit exceeded")
    _check_flags(body.content)
    with connect() as c:
        post = c.execute("SELECT * FROM posts WHERE id=?", (post_id,)).fetchone()
        if not post or post["status"] != "PUBLISHED":
            raise HTTPException(404, "Post not found")
        log = c.execute("SELECT id FROM routing_logs WHERE post_id=? AND target_agent_id=?",
                        (post_id, me["id"])).fetchone()
        if not log and post["author_agent_id"] != me["id"]:
            raise HTTPException(403, "Only notified agents (or the author) may comment")
        cid = str(uuid.uuid4())
        c.execute("INSERT INTO comments(id, post_id, author_agent_id, content) VALUES (?,?,?,?)",
                  (cid, post_id, me["id"], body.content))
        if log:
            feedback.record_engaged(c, log["id"])  # closes the feedback loop
    return {"comment_id": cid}


@app.get(f"{API}/posts/{{post_id}}/comments")
def list_comments(post_id: str):
    with connect() as c:
        return [dict(r) for r in c.execute("""SELECT cm.id, a.company_name author, a.verified, cm.content, cm.created_at
                 FROM comments cm JOIN agents a ON a.id=cm.author_agent_id WHERE cm.post_id=?
                 ORDER BY cm.created_at""", (post_id,))]


# ----------------------------------------------------------------------------- admin & ops
@app.post(f"{API}/admin/agents/{{agent_id}}/verify", dependencies=[Depends(admin_only)])
def verify_agent(agent_id: str):
    with connect() as c:
        c.execute("UPDATE agents SET verified=1 WHERE id=?", (agent_id,))
    return {"verified": True}


@app.post(f"{API}/admin/agents/{{agent_id}}/suspend", dependencies=[Depends(admin_only)])
def suspend_agent(agent_id: str):
    with connect() as c:
        c.execute("UPDATE agents SET status='SUSPENDED' WHERE id=?", (agent_id,))
    return {"suspended": True}


@app.post(f"{API}/admin/sweep-ignored", dependencies=[Depends(admin_only)])
def sweep(hours: float = None):
    return {"marked_ignored": feedback.sweep_ignored(hours)}


@app.post(f"{API}/admin/reembed", dependencies=[Depends(admin_only)])
def reembed():
    """Re-embed every agent manifesto with the CURRENT embedder (resets learned drift, which is dimension-specific)."""
    emb, n = get_embedder(), 0
    with connect() as c:
        for r in c.execute("SELECT id, interest_manifesto FROM agents").fetchall():
            v = emb.embed(r["interest_manifesto"])
            c.execute("UPDATE agents SET interest_embedding=?, dynamic_embedding=?, concepts=? WHERE id=?",
                      (vec_to_blob(v), vec_to_blob(np.zeros_like(v)), json.dumps(emb.concepts(r["interest_manifesto"])), r["id"]))
            n += 1
    return {"reembedded_agents": n}


@app.get(f"{API}/metrics")
def metrics():
    with connect() as c:
        by = {r["status"]: r["n"] for r in c.execute("SELECT status, COUNT(*) n FROM routing_logs GROUP BY status")}
        posts = c.execute("SELECT COUNT(*) n FROM posts").fetchone()["n"]
        agents = c.execute("SELECT COUNT(*) n FROM agents").fetchone()["n"]
    notified = sum(by.values())
    return {"agents": agents, "posts": posts, "notifications": by,
            "engagement_rate": round(by.get("ENGAGED", 0) / notified, 3) if notified else None,
            "avg_notifications_per_post": round(notified / posts, 2) if posts else None}


@app.get("/health")
def health():
    return {"ok": True}
