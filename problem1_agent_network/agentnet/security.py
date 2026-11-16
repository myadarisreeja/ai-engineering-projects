"""API keys and HMAC webhook signing."""
import hashlib
import hmac
import secrets
import time

REPLAY_WINDOW_SECONDS = 300


def new_api_key() -> str:
    return "ank_" + secrets.token_urlsafe(32)


def hash_key(key: str) -> str:
    return hashlib.sha256(key.encode()).hexdigest()


def new_webhook_secret() -> str:
    return "whs_" + secrets.token_hex(24)


def sign(secret: str, timestamp: str, body: bytes) -> str:
    msg = timestamp.encode() + b"." + body
    return hmac.new(secret.encode(), msg, hashlib.sha256).hexdigest()


def verify(secret: str, timestamp: str, body: bytes, signature: str, now: float = None) -> bool:
    """Used by receiving agents to confirm a webhook really came from the platform."""
    now = time.time() if now is None else now
    try:
        if abs(now - float(timestamp)) > REPLAY_WINDOW_SECONDS:
            return False
    except ValueError:
        return False
    return hmac.compare_digest(sign(secret, timestamp, body), signature)
