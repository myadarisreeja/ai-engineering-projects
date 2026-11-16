"""Runs the full tool-calling loop over REAL HTTP against servers that enforce the documented Ollama and Anthropic
wire-format rules (role ordering, tool_use/tool_result pairing, tool schema shape). It is still not a real model,
but it proves the request/response plumbing a real model would use."""
import json
import os

import pytest

from hwassistant.agent import Assistant
from hwassistant.backends import AnthropicBackend, FallbackBackend, OllamaBackend, make_backend
from hwassistant.executor import FixtureExecutor
from hwassistant.offline import OfflineBackend
from netutil import Recorder
from conftest import FIX


def laptop():
    return FixtureExecutor(os.path.join(FIX, "laptop_16gb"))


# ------------------------------------------------------------------ strict fake Ollama
def fake_ollama(args_as_string=False, bad_tool=False):
    def responder(n, path, headers, body):
        assert path == "/api/chat"
        req = json.loads(body)
        assert req["stream"] is False and req["model"]
        assert req["messages"][0]["role"] == "system"
        for t in req["tools"]:
            assert t["type"] == "function" and {"name", "description", "parameters"} <= set(t["function"])
        msgs = req["messages"]
        prev = None
        for m in msgs:
            if m["role"] == "tool":                                  # must follow an assistant turn with tool_calls
                assert prev and prev["role"] in ("assistant", "tool") and "tool_name" in m
            if m["role"] == "assistant" and m.get("tool_calls"):
                for c in m["tool_calls"]:
                    assert isinstance(c["function"]["arguments"], dict)  # requests must carry an object
            prev = m
        tool_msgs = [m for m in msgs if m["role"] == "tool"]
        if not tool_msgs:
            name = "run_shell" if bad_tool else "analyze_ram_upgrade"
            args = '{"include_empty_slots": true}' if args_as_string else {}
            return 200, {"message": {"role": "assistant", "content": "", "tool_calls": [
                {"function": {"name": name, "arguments": {"x": 1} if bad_tool else args}}]}}
        data = json.loads(tool_msgs[-1]["content"])["data"]
        if data.get("reason") == "not_allowed":
            return 200, {"message": {"role": "assistant", "content": "That tool is not allowed."}}
        if data.get("reason") == "invalid_arguments":
            return 200, {"message": {"role": "assistant", "content": "My arguments were invalid."}}
        return 200, {"message": {"role": "assistant", "content":
                     f"Max supported is {data['max_supported']} and headroom is {data['headroom']}."}}
    return Recorder(responder)


def test_ollama_full_loop_over_http():
    srv = fake_ollama()
    bot = Assistant(OllamaBackend("qwen2.5:7b", host=srv.url), laptop())
    a = bot.ask("Can I add more RAM?")
    srv.close()
    assert a == "Max supported is 64 GB and headroom is 48 GB." and len(srv.requests) == 2


def test_ollama_arguments_as_json_string_are_handled():
    srv = fake_ollama(args_as_string=True)
    bot = Assistant(OllamaBackend("m", host=srv.url), laptop())
    # analyze_ram_upgrade takes no args, so a string-encoded {"include_empty_slots": true} must be rejected by validation
    a = bot.ask("RAM upgrade?")
    srv.close()
    assert a == "My arguments were invalid."
    assert bot.messages[1]["tool_calls"][0]["arguments"] == {"include_empty_slots": True}   # string was parsed to an object
    assert json.loads(bot.messages[2]["content"])["data"]["reason"] == "invalid_arguments"


def test_ollama_hallucinated_tool_is_rejected_and_model_informed():
    srv = fake_ollama(bad_tool=True)
    bot = Assistant(OllamaBackend("m", host=srv.url), laptop())
    assert bot.ask("run something") == "That tool is not allowed."
    srv.close()


# ------------------------------------------------------------------ strict fake Anthropic Messages API
def fake_anthropic():
    def responder(n, path, headers, body):
        assert path == "/v1/messages" and headers["x-api-key"] == "k" and headers["anthropic-version"]
        req = json.loads(body)
        assert isinstance(req["system"], str) and req["max_tokens"] > 0
        assert all({"name", "description", "input_schema"} <= set(t) for t in req["tools"])
        msgs = req["messages"]
        assert msgs[0]["role"] == "user"
        for i, m in enumerate(msgs):
            assert m["role"] in ("user", "assistant")
            if i and msgs[i - 1]["role"] == m["role"]:
                raise AssertionError("consecutive same-role messages")          # roles must alternate
            blocks = m["content"] if isinstance(m["content"], list) else []
            uses = [b["id"] for b in blocks if b.get("type") == "tool_use"]
            if uses:                                                              # next message must answer ALL of them
                nxt = msgs[i + 1]["content"]
                assert sorted(b["tool_use_id"] for b in nxt if b["type"] == "tool_result") == sorted(uses)
        last = msgs[-1]
        if isinstance(last["content"], str):
            return 200, {"role": "assistant", "stop_reason": "tool_use", "content": [
                {"type": "text", "text": "Let me check."},
                {"type": "tool_use", "id": "toolu_01", "name": "get_cpu_info", "input": {}},
                {"type": "tool_use", "id": "toolu_02", "name": "get_gpu_info", "input": {}}]}
        results = [json.loads(b["content"])["data"] for b in last["content"]]
        return 200, {"role": "assistant", "stop_reason": "end_turn", "content": [
            {"type": "text", "text": f"CPU {results[0]['logical_cpus']} threads; GPU vendor {results[1]['gpus'][0]['vendor']}."}]}
    return Recorder(responder)


def test_anthropic_parallel_tool_use_loop_over_http(monkeypatch):
    srv = fake_anthropic()
    monkeypatch.setenv("ANTHROPIC_BASE_URL", srv.url)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    bot = Assistant(AnthropicBackend("claude-haiku-5-5"), laptop())
    a = bot.ask("tell me about cpu and gpu")
    follow = bot.ask("and again?")                      # multi-turn: history re-sent in valid alternating format
    srv.close()
    assert a == "CPU 12 threads; GPU vendor Intel Corporation."
    assert follow == a


# ------------------------------------------------------------------ resilience
def test_fallback_when_llm_server_is_down_still_answers():
    msgs = []
    b = make_backend("ollama", "m", on_fallback=msgs.append)
    b.primary.host = "http://127.0.0.1:9"                # nothing listens
    bot = Assistant(b, laptop())
    a = bot.ask("Can I add more RAM, and how much?")
    assert "64 GB" in a and msgs and "offline" in msgs[0]
    assert b.degraded                                       # stays in offline mode for the rest of the session
    assert "Iris Xe" in bot.ask("and what about the GPU?")


def test_fallback_on_http_500_and_no_fallback_flag_raises():
    srv = Recorder(lambda *a: (500, {"error": "boom"}))
    b = FallbackBackend(OllamaBackend("m", host=srv.url), OfflineBackend())
    assert "64 GB" in Assistant(b, laptop()).ask("Can I add more RAM?")
    strict = make_backend("ollama", "m", fallback=False)
    strict.host = srv.url
    with pytest.raises(Exception):
        Assistant(strict, laptop()).ask("Can I add more RAM?")
    srv.close()
