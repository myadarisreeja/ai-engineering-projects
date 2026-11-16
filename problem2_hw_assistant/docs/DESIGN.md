# Design notes - Hardware Q&A Assistant

## Pipeline
`question -> LLM picks tool(s) -> strict arg validation -> allow-listed execution -> parser -> normalised JSON (with status)
-> LLM answers from JSON only`. Tool results are cached per session (static hardware forever, volatile data 5 s) so
follow-ups ("and the GPU?") are cheap and context is preserved in the message history.

## Safety (the main signal)
1. **LLM never writes commands.** It chooses a tool *name* + typed args. Unknown tool -> rejected.
2. **Fixed argv, `shell=False`**, so `;`, `&&`, `|`, `$()` have no meaning. No tool takes free text into a command; the one
   string parameter (`device`) is regex-validated *and* only used to filter parsed output, never placed in argv.
3. **Binaries resolved from fixed system dirs**, not `$PATH`; scrubbed env, `LC_ALL=C`, 5 s timeout, stdin closed, output cap.
4. **File reads via regex allow-list** (`/proc/meminfo`, `/etc/os-release`, selected DMI & battery files). Serial numbers
   (`product_serial`), `/etc/shadow`, `..` traversal are refused (tested).
5. **Read-only by construction:** every command is an inspection command.

## Permissions (root)
Only `dmidecode -t memory` needs root. Options considered: scoped sudoers (**chosen**: simple, auditable, argument-exact),
privileged helper daemon on a Unix socket (stronger isolation, more moving parts - right for desktop products),
non-root fallbacks (`/proc`, `/sys`, `lscpu`, `lsblk`) used automatically when root data is unavailable. The agent is never
run as root. `sudo -n` guarantees no password prompt can hang the UI.

## Accuracy / anti-hallucination
* Every tool returns `status: ok|partial|unavailable` plus `reason` and `how_to_fix`; missing fields are explicit.
* System prompt: answer only from tool JSON; say "I can't determine..." otherwise; no outside knowledge about this machine.
* **Arithmetic is done in code**, not by the LLM (`analyze_ram_upgrade`: headroom, free slots, per-slot limit, largest module).
* Caveats are data: SMBIOS max capacity can be wrong; soldered memory detected; "OS-visible RAM" vs "installed RAM" differ.
* Loop guard: max 4 tool rounds.

## Portability
Capability discovery at start (`--doctor`): tools whose data source is absent are hidden from the model (no battery -> no
battery tool). Fallbacks: `lscpu` -> `/proc/cpuinfo`, `free` -> `/proc/meminfo`, `lspci` missing -> `nvidia-smi` only, `lsblk` JSON
tolerant of old/new util-linux (string vs bool/int `ROTA`, size types). `LC_ALL=C` makes output locale-independent.

## Privacy
Default backends are offline/local. With Ollama no hardware data leaves the machine. The cloud backend sends only the tool
JSON (no serials/MACs are ever collected) and the user's questions.

## Testing
Parsers are pure and tested on captured fixtures (`fixtures/`), including a no-root, no-lspci, NVIDIA desktop. Safety tests try
injection strings, traversal, unknown tools, bad args. The orchestration test uses a scripted fake LLM that also attempts a
hostile tool call (`rm_rf`) to prove it is rejected. A smoke test runs against the real machine on Linux.

## Extending
Add a tool = write a collector using `ex.run/read_file`, a parser, a `ToolSpec`, and (optionally) an offline renderer.
Add a command = add one entry to `ALLOWED_COMMANDS` (fixed argv). Nothing else can reach the OS.

## Lessons from testing on real captured data
1. **Firmware lies.** A real VMware guest dump declares a 64-slot array but lists 128 memory devices not linked to that array.
   The tool cross-checks slot counts and array linkage; on disagreement it returns `partial` + `data_quality_warnings` and the
   assistant refuses to claim an upgrade path. VMs are detected via DMI vendor/product strings.
2. **Virtual disks fake `ROTA=1`.** virtio/VMware disks look like HDDs; they are reported as `virtual_disk` (media unknowable).
3. **Failure classification.** `dmidecode` inside a container fails with "/dev/mem"; that is `no_dmi_data`, not "permission".
   Errors are classified from stderr so the user gets the right advice (sudoers rule vs. "this system has no DMI tables").
4. **Bounded output.** 128-device tables are truncated before reaching the LLM context.
5. **LLM failure is a normal case.** `FallbackBackend` degrades to the offline router for the rest of the session.
