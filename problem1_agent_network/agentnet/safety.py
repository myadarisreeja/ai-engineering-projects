"""Content safety: prompt-injection / secret-exfiltration screening, rate limiting, dedupe."""
import hashlib
import re
import threading
import time
from collections import defaultdict, deque
from typing import List

INJECTION_PATTERNS = [
    (r"ignore (all |any )?(the )?(previous|prior|above|earlier) (instructions|prompts?|rules)", "override_instructions"),
    (r"disregard (all |any )?(the )?(previous|prior|above|your) (instructions|rules|guidelines)", "override_instructions"),
    (r"(reveal|print|output|show|send|leak|exfiltrate).{0,40}(system prompt|api[ _-]?key|secret|password|credentials|token)", "secret_exfiltration"),
    (r"you are now (a|an|in)\b", "role_hijack"),
    (r"\bact as (a|an|the)\b.{0,30}(admin|root|developer|system)", "role_hijack"),
    (r"<\s*/?\s*(system|assistant|tool)\s*>", "fake_role_tags"),
    (r"\[\s*(system|inst)\s*\]", "fake_role_tags"),
    (r"(transfer|wire|send) (all )?(funds|money|payment).{0,30}(to|into) (account|upi|wallet)", "unauthorised_action"),
    (r"<script\b|javascript:", "script_injection"),
    (r"[A-Za-z0-9+/]{200,}={0,2}", "encoded_blob"),
    (r"\bsk-[A-Za-z0-9]{20,}", "secret_in_content"),
]
_COMPILED = [(re.compile(p, re.I | re.S), tag) for p, tag in INJECTION_PATTERNS]
MAX_URLS = 3


def scan_text(text: str) -> List[str]:
    """Returns a list of flag names; empty means clean. Layer 1 of 2 (layer 2 = an LLM guard model)."""
    flags = [tag for pat, tag in _COMPILED if pat.search(text)]
    if len(re.findall(r"https?://", text)) > MAX_URLS:
        flags.append("link_spam")
    return sorted(set(flags))


def content_hash(title: str, content: str) -> str:
    norm = re.sub(r"\s+", " ", (title + " " + content).lower()).strip()
    return hashlib.sha256(norm.encode()).hexdigest()


class RateLimiter:
    """Sliding-window limiter per (agent, action). In production: Redis INCR + EXPIRE."""

    def __init__(self):
        self._hits = defaultdict(deque)
        self._lock = threading.Lock()

    def allow(self, key: str, limit: int, window: float = 60.0) -> bool:
        now = time.monotonic()
        with self._lock:
            q = self._hits[key]
            while q and now - q[0] > window:
                q.popleft()
            if len(q) >= limit:
                return False
            q.append(now)
            return True

    def reset(self):
        with self._lock:
            self._hits.clear()


limiter = RateLimiter()
