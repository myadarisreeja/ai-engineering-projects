import argparse
import os
import sys

from .agent import Assistant
from .backends import make_backend
from .executor import FixtureExecutor, SystemExecutor
from .tools import available_tools

BANNER = "Hardware Q&A assistant ({backend}). Ask about CPU, RAM, storage, GPU, battery... /help for commands."


def doctor(ex):
    print("Capability report")
    from .executor import ALLOWED_COMMANDS
    for k in ALLOWED_COMMANDS:
        print(f"  {'✓' if ex.has_command(k) else '✗'} {k}")
    print(f"  {'✓' if ex.root_available() else '✗'} DMI memory access (dmidecode + root or sudoers rule)")
    print("Tools exposed to the model:", ", ".join(t.name for t in available_tools(ex)))


def main(argv=None):
    ap = argparse.ArgumentParser(prog="hwassistant", description="Local hardware Q&A assistant (read-only, allow-listed).")
    ap.add_argument("--backend", choices=["offline", "ollama", "anthropic"], default=os.getenv("HW_BACKEND", "offline"))
    ap.add_argument("--model", help="Model name for ollama/anthropic (or env HW_MODEL)")
    ap.add_argument("--fixtures", help="Replay captured outputs from a fixture dir instead of the real machine")
    ap.add_argument("--once", metavar="QUESTION", help="Ask one question and exit")
    ap.add_argument("--debug", action="store_true", help="Show which tools were called")
    ap.add_argument("--no-fallback", action="store_true", help="Fail instead of falling back to offline mode")
    ap.add_argument("--doctor", action="store_true", help="Show which tools/commands are available and exit")
    a = ap.parse_args(argv)

    ex = FixtureExecutor(a.fixtures) if a.fixtures else SystemExecutor()
    if a.doctor:
        doctor(ex)
        return 0
    backend = make_backend(a.backend, a.model, fallback=not a.no_fallback, on_fallback=lambda m: print(f"[warning] {m}", file=sys.stderr))
    bot = Assistant(backend, ex, debug=a.debug)
    if a.once:
        print(bot.ask(a.once))
        return 0
    print(BANNER.format(backend=a.backend))
    while True:
        try:
            q = input("\nyou › ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not q:
            continue
        if q in ("/quit", "/exit", "exit", "quit"):
            return 0
        if q == "/help":
            print("/doctor  capability report   /tools  list tools   /reset  forget context   /quit")
            continue
        if q == "/doctor":
            doctor(ex)
            continue
        if q == "/tools":
            for t in bot.tools:
                print(f"  {t.name}: {t.description}")
            continue
        if q == "/reset":
            bot.messages.clear(); bot.cache.clear()
            print("context cleared")
            continue
        try:
            print("\nassistant ›", bot.ask(q))
        except Exception as e:
            print(f"\nassistant › Backend error: {e}. Try --backend offline (or drop --no-fallback).")


if __name__ == "__main__":
    sys.exit(main())
