# Problem 2 - Local Linux Hardware Q&A Assistant

A chatbot that answers questions about *this* machine's hardware by calling safe, read-only, allow-listed tools -
never by guessing.

```bash
pip install -r requirements.txt            # httpx is only needed for LLM backends
python -m hwassistant                      # interactive chat (offline backend, no LLM needed)
python -m hwassistant --once "Can I add more RAM, and how much?" --debug
python -m hwassistant --doctor             # which commands/tools exist, is root data reachable?
python -m hwassistant --fixtures fixtures/laptop_16gb   # replay a captured 16 GB / 2-slot Dell laptop
python -m pytest -q tests                  # 44 tests on Linux (Linux-only ones skip on Windows)
```

Example (fixture): *"You have 16 GB in 1 of 2 slots. The board supports up to 64 GB, so you can add RAM: up to 48 GB more,
and the largest single module that fits is 32 GB."* Without root access it says *"I can't determine whether you can add RAM"*
and explains how to fix it.

## Backends
| flag | model | data leaves machine? |
|---|---|---|
| `--backend offline` (default) | deterministic router + templates | no |
| `--backend ollama --model qwen2.5:7b` | local LLM via Ollama tool-calling | **no** |
| `--backend anthropic` (needs `ANTHROPIC_API_KEY`, optional `HW_MODEL`) | cloud fallback | tool JSON + questions |

Add `--no-fallback` to fail instead of silently dropping to offline mode when the LLM server is unreachable (default: warn on stderr and keep answering offline).

## Fixtures
`laptop_16gb`, `desktop_no_root` are hand-written. **`real_vm_captured` is real captured output** (see its `PROVENANCE.md`),
including messy firmware tables the assistant must not over-trust.

## Enabling root-only data (RAM slots / max capacity)
`sudo ./deploy/install_sudoers.sh` installs `/etc/sudoers.d/hw-assistant` allowing exactly
`dmidecode -t memory` without a password (validated with `visudo -c`). Remove: `sudo rm /etc/sudoers.d/hw-assistant`.
Without it the assistant degrades gracefully to non-root data.

## Tools
`get_cpu_info`, `get_memory_info`, `get_memory_topology`, `analyze_ram_upgrade`, `get_storage_info`, `get_disk_usage`,
`get_gpu_info`, `get_system_info`, `get_battery_info` (hidden automatically if the machine has no battery / no lspci...).

## Layout
```
hwassistant/ executor.py (ONLY OS access) parsers.py (pure) tools.py agent.py offline.py backends.py cli.py
fixtures/    laptop_16gb  desktop_no_root   (captured outputs for tests/demos)
deploy/      install_sudoers.sh + template  tests/  docs/DESIGN.md
```
