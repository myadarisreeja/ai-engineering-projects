"""LLM backends. All speak a neutral message format:
  {"role":"user","content"} | {"role":"assistant","content","tool_calls":[{id,name,arguments}]}
  | {"role":"tool","tool_call_id","name","content"}
and return {"text": str, "tool_calls": [...]}. Wire-format conversion happens inside each backend."""
import json
import os
import uuid


class OllamaBackend:
    """Local, private: no data leaves the machine. Requires `ollama serve` and a tool-capable model."""
    name = "ollama"

    def __init__(self, model=None, host=None):
        self.model = model or os.getenv("HW_MODEL", "qwen2.5:7b")
        self.host = host or os.getenv("OLLAMA_HOST", "http://localhost:11434")

    def chat(self, messages, tools, system):
        import httpx
        wire = [{"role": "system", "content": system}]
        for m in messages:
            if m["role"] == "assistant" and m.get("tool_calls"):
                wire.append({"role": "assistant", "content": m["content"], "tool_calls": [
                    {"function": {"name": c["name"], "arguments": c["arguments"]}} for c in m["tool_calls"]]})
            elif m["role"] == "tool":
                wire.append({"role": "tool", "content": m["content"], "tool_name": m["name"]})
            else:
                wire.append({"role": m["role"], "content": m["content"]})
        r = httpx.post(f"{self.host}/api/chat", timeout=120, json={
            "model": self.model, "messages": wire, "stream": False, "options": {"temperature": 0},
            "tools": [{"type": "function", "function": t} for t in tools]})
        r.raise_for_status()
        msg = r.json()["message"]
        calls = []
        for c in msg.get("tool_calls") or []:
            args = c["function"].get("arguments") or {}
            if isinstance(args, str):                      # some versions return a JSON string
                try:
                    args = json.loads(args)
                except ValueError:
                    args = {"_unparseable": args}          # fails validation -> the model gets an error and can retry
            calls.append({"id": uuid.uuid4().hex[:8], "name": c["function"]["name"], "arguments": args})
        return {"text": msg.get("content") or "", "tool_calls": calls}


class AnthropicBackend:
    """Cloud fallback (sends tool JSON + your questions to the API). Needs ANTHROPIC_API_KEY."""
    name = "anthropic"

    def __init__(self, model=None):
        self.model = model or os.getenv("HW_MODEL", "claude-haiku-5-5")

    def chat(self, messages, tools, system):
        import httpx
        wire, pending = [], []
        def flush():
            if pending:
                wire.append({"role": "user", "content": list(pending)})
                pending.clear()
        for m in messages:
            if m["role"] == "tool":
                pending.append({"type": "tool_result", "tool_use_id": m["tool_call_id"], "content": m["content"]})
                continue
            flush()
            if m["role"] == "assistant" and m.get("tool_calls"):
                blocks = ([{"type": "text", "text": m["content"]}] if m["content"] else []) + [
                    {"type": "tool_use", "id": c["id"], "name": c["name"], "input": c["arguments"]} for c in m["tool_calls"]]
                wire.append({"role": "assistant", "content": blocks})
            else:
                wire.append({"role": m["role"], "content": m["content"]})
        flush()
        base = os.getenv("ANTHROPIC_BASE_URL", "https://api.anthropic.com").rstrip("/")
        r = httpx.post(f"{base}/v1/messages", timeout=60, headers={
            "x-api-key": os.environ["ANTHROPIC_API_KEY"], "anthropic-version": "2023-06-01"},
            json={"model": self.model, "max_tokens": 800, "system": system, "messages": wire, "temperature": 0,
                  "tools": [{"name": t["name"], "description": t["description"], "input_schema": t["parameters"]} for t in tools]})
        r.raise_for_status()
        blocks = r.json()["content"]
        return {"text": "".join(b["text"] for b in blocks if b["type"] == "text"),
                "tool_calls": [{"id": b["id"], "name": b["name"], "arguments": b["input"]} for b in blocks if b["type"] == "tool_use"]}


class FallbackBackend:
    """Use the LLM when reachable; if it errors (server down, timeout, bad key) answer with the offline router
    instead of crashing the chat. The offline backend reads the same neutral message format."""

    def __init__(self, primary, fallback, on_fallback=None):
        self.primary, self.fallback, self.on_fallback = primary, fallback, on_fallback
        self.name = f"{primary.name}(+offline fallback)"
        self.degraded = False

    def chat(self, messages, tools, system):
        if not self.degraded:
            try:
                return self.primary.chat(messages, tools, system)
            except Exception as e:                      # network, HTTP status, malformed response
                self.degraded = True                    # stay degraded for the rest of the session
                if self.on_fallback:
                    self.on_fallback(f"{self.primary.name} unavailable ({type(e).__name__}); using offline mode")
        return self.fallback.chat(messages, tools, system)


def make_backend(name: str, model: str = None, fallback: bool = True, on_fallback=None):
    from .offline import OfflineBackend
    if name == "offline":
        return OfflineBackend()
    primary = OllamaBackend(model) if name == "ollama" else AnthropicBackend(model) if name == "anthropic" else None
    if primary is None:
        raise ValueError(f"unknown backend '{name}'")
    return FallbackBackend(primary, OfflineBackend(), on_fallback) if fallback else primary
