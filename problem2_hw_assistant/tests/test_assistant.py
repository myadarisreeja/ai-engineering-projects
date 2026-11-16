import json

from hwassistant.agent import Assistant
from hwassistant.executor import FixtureExecutor
from conftest import FIX
import os


def test_ram_upgrade_example_from_problem_statement(bot):
    a = bot.ask("Can I add more RAM, and how much?")
    assert "16 GB in 1 of 2 slots" in a and "64 GB" in a and "32 GB" in a


def test_upgrade_math_is_deterministic(bot):
    r = bot._call_tool("analyze_ram_upgrade", {})
    assert r["can_add_memory"] and r["headroom"] == "48 GB" and r["largest_single_module_that_fits"] == "32 GB"
    assert r["slots"] == {"total": 2, "used": 1, "free": 1}


def test_no_root_says_cannot_determine_instead_of_guessing(bot_no_root):
    a = bot_no_root.ask("Can I add more RAM?")
    assert "can't determine" in a.lower() and "15.4 GB" in a and "64" not in a
    assert "sudoers" in a or "sudo" in a


def test_follow_up_gpu_and_cache(bot):
    bot.ask("How much RAM do I have?")
    a = bot.ask("and what about the GPU?")
    assert "Iris Xe" in a
    bot.ask("what about the GPU again")
    calls = [m for m in bot.messages if m["role"] == "assistant" and m["tool_calls"]]
    assert len(calls) >= 3 and len(bot.cache) >= 2


def test_pronoun_follow_up_reuses_last_topic(bot):
    bot.ask("Is my SSD NVMe?")
    a = bot.ask("tell me more about it")
    assert "nvme0n1" in a


def test_nvme_and_hdd(bot):
    a = bot.ask("Is my SSD NVMe?")
    assert "NVMe SSD" in a and "hard disk" in a


def test_missing_gpu_tool_is_hidden_and_reported(bot_no_root):
    names = {t.name for t in bot_no_root.tools}
    assert "get_battery_info" not in names                    # no battery on the desktop fixture
    a = bot_no_root.ask("what gpu do I have?")
    assert "RTX 3060" in a and "12 GB" in a


def test_unknown_question_says_cannot_determine(bot):
    a = bot.ask("What's the weather in Delhi?")
    assert "can't determine" in a.lower()


def test_system_and_battery_and_diskspace(bot):
    assert "Latitude 5530" in bot.ask("which laptop model is this?") and "Ubuntu 24.04" in bot.ask("what distro")
    assert "77.8%" in bot.ask("how healthy is my battery")
    assert "free of" in bot.ask("how much disk space is left")


def test_llm_tool_loop_with_scripted_backend():
    """Proves the orchestration works with a (mock) real LLM: tool_call -> tool result -> final answer."""
    class Scripted:
        def __init__(self): self.turn = 0; self.seen_system = None
        def chat(self, messages, tools, system):
            self.seen_system = system
            self.turn += 1
            if self.turn == 1:
                return {"text": "", "tool_calls": [
                    {"id": "1", "name": "analyze_ram_upgrade", "arguments": {}},
                    {"id": "2", "name": "rm_rf", "arguments": {"path": "/"}}]}
            tool_msgs = [m for m in messages if m["role"] == "tool"]
            assert json.loads(tool_msgs[0]["content"])["data"]["max_supported"] == "64 GB"
            assert json.loads(tool_msgs[1]["content"])["data"]["reason"] == "not_allowed"
            return {"text": "done", "tool_calls": []}
    b = Scripted()
    bot = Assistant(b, FixtureExecutor(os.path.join(FIX, "laptop_16gb")))
    assert bot.ask("upgrade?") == "done"
    assert "ONLY from tool results" in b.seen_system and "root_access=yes" in b.seen_system


def test_infinite_tool_loop_is_capped():
    class Looper:
        def chat(self, messages, tools, system):
            return {"text": "", "tool_calls": [{"id": "x", "name": "get_cpu_info", "arguments": {}}]}
    bot = Assistant(Looper(), FixtureExecutor(os.path.join(FIX, "laptop_16gb")))
    assert "too many tool rounds" in bot.ask("hi")
