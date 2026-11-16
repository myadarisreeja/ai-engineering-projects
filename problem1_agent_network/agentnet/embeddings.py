"""Embeddings.

Default: a deterministic, offline *concept embedder*. It maps text onto a business-concept
ontology (payments, food delivery, festive, automotive, ...) plus a small hashed bag-of-words
tail, and propagates activation along a relatedness graph (food delivery -> payments).
That gives "Diwali food offer" ~ "UPI cashback" the semantic link a keyword filter misses,
with zero API cost, which makes the project runnable and testable offline.

Production: set AGENTNET_EMBEDDER=openai and implement/enable OpenAIEmbedder (text-embedding-3-small)
and move vectors into pgvector with an HNSW index. The rest of the pipeline is unchanged because
everything only depends on `Embedder.embed()` and `Embedder.concepts()`.
"""
import hashlib
import os
import re
from typing import Dict

import numpy as np

# concept -> (specificity weight, keyword prefixes). Generic modifiers (festive, offers) get low
# weight so "both mention Diwali" alone never makes two businesses relevant to each other.
ONTOLOGY: Dict[str, tuple] = {
    "payments": (1.0, ["upi", "payment", "pay ", "pays", "paying", "wallet", "cashback", "checkout",
                       "phonepe", "paytm", "gpay", "transaction", "merchant", "qr "]),
    "banking_cards": (1.0, ["credit card", "debit card", "card", "bank", "emi", "loan", "hdfc", "icici",
                            "interest rate", "reward point", "netbanking", "account"]),
    "food_delivery": (1.0, ["food", "deliver", "restaurant", "swiggy", "zomato", "meal", "dining",
                            "grocer", "instamart", "kitchen"]),
    "ecommerce": (1.0, ["e-commerce", "ecommerce", "flipkart", "amazon", "shopping", "marketplace",
                        "retail", "product", "seller", "orders", "big billion"]),
    "automotive": (1.0, ["bike", "car ", "cars", "vehicle", "auto ", "scooter", "motorcycle", "ev ",
                         "electric vehicle", "dealership", "test ride", "bajaj", "two-wheeler", "pulsar",
                         "chetak", "showroom"]),
    "travel": (1.0, ["flight", "hotel", "travel", "booking", "trip", "airline", "irctc", "makemytrip"]),
    "insurance": (1.0, ["insurance", "policy", "premium", "claim"]),
    "telecom": (1.0, ["telecom", "recharge", "broadband", "sim ", "jio", "airtel", "5g"]),
    "investing": (1.0, ["invest", "mutual fund", "stock", "broking", "brokerage", "demat", "sip ", "trading",
                        "zerodha", "groww", "wealth", "portfolio"]),
    "mobility": (1.0, ["cab ", "cabs", "taxi", "ola ", "uber", "rapido", "ride-hail", "ride hail", "fastag", "metro",
                       "parking", "fuel", "petrol"]),
    "entertainment": (1.0, ["movie", "cinema", "ott ", "hotstar", "netflix", "streaming", "bookmyshow", "ticket",
                            "concert", "gaming", "music"]),
    "healthcare": (1.0, ["pharmacy", "health", "hospital", "medicine", "doctor", "diagnostic", "wellness", "fitness",
                         "pharmeasy", "1mg"]),
    "education": (1.0, ["course", "learning", "edtech", "byju", "student", "exam", "upskill", "tutor"]),
    "fashion_lifestyle": (1.0, ["fashion", "apparel", "myntra", "beauty", "nykaa", "lifestyle", "clothing", "footwear"]),
    "festive": (0.4, ["diwali", "festive", "festival", "holi", "eid", "christmas", "navratri", "dussehra",
                      "onam", "pongal", "new year"]),
    "offers": (0.5, ["discount", "% off", " off ", "offer", "deal", "promo", "coupon", "sale", "free "]),
}
CONCEPTS = list(ONTOLOGY)

# relatedness graph (source activates target at factor): captures partnership adjacency
RELATED = {
    "food_delivery": {"payments": 0.55, "banking_cards": 0.45, "ecommerce": 0.35},
    "ecommerce": {"payments": 0.55, "banking_cards": 0.50, "food_delivery": 0.20},
    "travel": {"payments": 0.40, "banking_cards": 0.45, "insurance": 0.35},
    "automotive": {"insurance": 0.35, "banking_cards": 0.30},
    "payments": {"banking_cards": 0.50, "ecommerce": 0.40, "food_delivery": 0.40},
    "banking_cards": {"payments": 0.50, "ecommerce": 0.40, "food_delivery": 0.35},
    "investing": {"banking_cards": 0.50, "payments": 0.35},
    "mobility": {"payments": 0.45, "travel": 0.40, "automotive": 0.25},
    "entertainment": {"payments": 0.30, "ecommerce": 0.25},
    "healthcare": {"insurance": 0.50, "payments": 0.30},
    "education": {"payments": 0.30, "banking_cards": 0.25},
    "fashion_lifestyle": {"ecommerce": 0.50, "payments": 0.35},
}
HASH_DIMS = 64
HASH_WEIGHT = 0.18
DIM = len(CONCEPTS) + HASH_DIMS

_PATTERNS = {c: re.compile("|".join(r"(?<![a-z0-9])" + re.escape(k) for k in kws))
             for c, (_, kws) in ONTOLOGY.items()}
_TOKEN = re.compile(r"[a-z0-9₹%]+")


class Embedder:
    dim = DIM

    def concepts(self, text: str) -> Dict[str, float]:
        """Raw concept activations in [0,1] (no expansion)."""
        t = " " + text.lower() + " "
        out = {}
        for c, pat in _PATTERNS.items():
            hits = {m.group(0) for m in pat.finditer(t)}
            if hits:
                out[c] = min(1.0, 0.55 + 0.225 * (len(hits) - 1))
        return out

    def embed(self, text: str) -> np.ndarray:
        raw = self.concepts(text)
        act = dict(raw)
        for c, strength in raw.items():                       # one-hop expansion
            for tgt, f in RELATED.get(c, {}).items():
                act[tgt] = max(act.get(tgt, 0.0), strength * f)
        v = np.zeros(DIM, dtype=np.float32)
        for i, c in enumerate(CONCEPTS):
            v[i] = act.get(c, 0.0) * ONTOLOGY[c][0]
        cn = np.linalg.norm(v)
        if cn > 0:
            v[: len(CONCEPTS)] /= cn
        tail = np.zeros(HASH_DIMS, dtype=np.float32)
        for tok in _TOKEN.findall(text.lower()):
            if len(tok) < 3:
                continue
            h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
            tail[h % HASH_DIMS] += 1.0 if (h >> 8) & 1 else -1.0
        tn = np.linalg.norm(tail)
        if tn > 0:
            v[len(CONCEPTS):] = HASH_WEIGHT * tail / tn
        n = np.linalg.norm(v)
        return v / n if n > 0 else v


class ModelEmbedder(Embedder):
    """Dense vectors from a real embedding model. Concept activations (used for the lexical-coverage and
    subscription parts of the reranker) still come from the ontology, so the hybrid scoring keeps working."""

    def __init__(self, fetch, name):
        self._fetch, self.name, self._cache = fetch, name, {}

    def embed(self, text: str) -> np.ndarray:
        if text not in self._cache:
            if len(self._cache) > 5000:
                self._cache.clear()
            self._cache[text] = normalize(np.asarray(self._fetch(text), dtype=np.float32))
        return self._cache[text]

    @property
    def dim(self):
        return None


def _openai_fetch():
    import httpx
    base = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
    model = os.getenv("AGENTNET_EMBED_MODEL", "text-embedding-3-small")

    def fetch(text):
        r = httpx.post(f"{base}/embeddings", timeout=15, json={"model": model, "input": text},
                       headers={"Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}"})
        r.raise_for_status()
        return r.json()["data"][0]["embedding"]
    return fetch, f"openai:{model}"


def _ollama_fetch():
    import httpx
    host = os.getenv("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
    model = os.getenv("AGENTNET_EMBED_MODEL", "nomic-embed-text")

    def fetch(text):
        r = httpx.post(f"{host}/api/embed", timeout=30, json={"model": model, "input": text})
        r.raise_for_status()
        return r.json()["embeddings"][0]
    return fetch, f"ollama:{model}"


def _st_fetch():
    from sentence_transformers import SentenceTransformer   # pip install sentence-transformers
    name = os.getenv("AGENTNET_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    model = SentenceTransformer(name)
    return (lambda text: model.encode(text)), f"st:{name}"


_embedder = Embedder()
_model_cache = {}


def get_embedder() -> Embedder:
    """AGENTNET_EMBEDDER = concept (default, offline) | openai | ollama | sentence-transformers"""
    kind = os.getenv("AGENTNET_EMBEDDER", "concept")
    if kind == "concept":
        return _embedder
    if kind not in _model_cache:
        factory = {"openai": _openai_fetch, "ollama": _ollama_fetch, "sentence-transformers": _st_fetch}.get(kind)
        if factory is None:
            raise ValueError(f"unknown AGENTNET_EMBEDDER '{kind}'")
        _model_cache[kind] = ModelEmbedder(*factory())
    return _model_cache[kind]


def normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 0 else v
