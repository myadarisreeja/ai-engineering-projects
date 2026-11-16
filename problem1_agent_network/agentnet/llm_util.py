"""Small helpers shared by every place the platform talks to an LLM / embedding API."""
import json
import os
import re

import httpx


def anthropic_base() -> str:
    return os.getenv("ANTHROPIC_BASE_URL", "https://api.anthropic.com").rstrip("/")


def extract_json(text: str):
    """Real models wrap JSON in ```json fences or add a sentence around it. Pull out the first JSON object."""
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    if fence:
        text = fence.group(1).strip()
    try:
        return json.loads(text)
    except ValueError:
        pass
    start = text.find("{")
    while start != -1:                      # scan for the first balanced, parseable object
        depth = 0
        for i in range(start, len(text)):
            depth += text[i] == "{"
            depth -= text[i] == "}"
            if depth == 0:
                try:
                    return json.loads(text[start:i + 1])
                except ValueError:
                    break
        start = text.find("{", start + 1)
    raise ValueError("no JSON object found in model output")


def call_anthropic(api_key: str, model: str, system: str, user: str, max_tokens: int = 300, timeout: float = 20):
    r = httpx.post(f"{anthropic_base()}/v1/messages", timeout=timeout,
                   headers={"x-api-key": api_key, "anthropic-version": "2023-06-01"},
                   json={"model": model, "max_tokens": max_tokens, "temperature": 0, "system": system,
                         "messages": [{"role": "user", "content": user}]})
    r.raise_for_status()
    return "".join(b.get("text", "") for b in r.json()["content"] if b.get("type") == "text")
