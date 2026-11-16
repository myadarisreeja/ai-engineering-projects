# AI Engineering Interview Projects

Two complete, tested, runnable solutions. Both work **offline with zero API keys** (deterministic
fallbacks) and upgrade to real LLMs by flipping a flag.

| | Project | Run it | Tests |
|---|---|---|---|
| 1 | `problem1_agent_network/` - social network for AI agents with semantic relevance routing | `python demo.py` | 29 |
| 2 | `problem2_hw_assistant/` - local, read-only Linux hardware Q&A assistant | `python -m hwassistant` | 44 (a few Linux-only tests skip on Windows) |

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r problem1_agent_network/requirements.txt
cd problem1_agent_network && python demo.py && python -m pytest -q tests     # Problem 1
cd ../problem2_hw_assistant && python -m hwassistant --once "Can I add more RAM?"   # Problem 2
python -m pytest -q tests
```

Each project has a `README.md` (usage) and `docs/DESIGN.md` (architecture, trade-offs, interview talking points).

## What is verified, and how
| Area | Verified how | NOT verified |
|---|---|---|
| Hardware parsing | Real `lscpu/lsblk/free/df/lspci/uname` captured from a real Linux VM; real `dmidecode` memory dumps from a CentOS/VMware guest (open-source `jc` fixtures). Found and fixed 3 real-data bugs (inconsistent SMBIOS tables, virtio disks reported as HDD, `dmidecode` failing in containers blamed on permissions) | `dmidecode` on a **physical laptop with root** (nothing in this sandbox exposes real DMI) |
| Local/cloud LLM tool-calling | Full loop over real HTTP against servers that enforce Ollama and Anthropic request rules (role order, tool_use/tool_result pairing, schemas, string-encoded args, hallucinated tools, outages -> offline fallback) | A **real model** (no Ollama / API key here): tool-choice quality is untested |
| Real embeddings | OpenAI-format embeddings API over HTTP, re-embed after switching, dimension mismatch handled | Actual embedding quality (OpenAI / Ollama `nomic-embed-text` / sentence-transformers) |
| LLM replies & judge | Fenced/chatty JSON parsing, prompt-injection separation, blend/veto/fallback, agent survives LLM outage | Real model output |
| Webhooks | Real loopback HTTP receivers: HMAC signature verified, retry 500,500,200, dead endpoint -> breaker + inbox | Internet-facing endpoints, TLS |
