"""Offline backend: a deterministic intent router + template renderer that speaks the same interface as
the LLM backends. It needs no model, so the assistant (and the tests) work with zero setup, and it
doubles as a safe fallback when no LLM is reachable. Every sentence is generated from tool JSON only."""
import json
import re
import uuid
from typing import List

TOPICS = {
    "diskspace": ["free space", "disk space", "disk usage", "storage left", "space left", "how full", "space do i have"],
    "ram": ["ram", "memory", "dimm", "dimms"],
    "cpu": ["cpu", "processor", "cores", "core count", "threads", "ryzen", "clock speed"],
    "storage": ["ssd", "nvme", "hdd", "disk", "disks", "storage", "drive", "drives", "hard drive"],
    "gpu": ["gpu", "graphics", "nvidia", "radeon", "video card", "vram"],
    "battery": ["battery", "charge", "cycles"],
    "system": ["os", "operating system", "distro", "distribution", "kernel", "bios", "laptop model", "which laptop",
               "manufacturer", "what machine", "model is my", "computer model", "linux version"],
    "overview": ["specs", "specifications", "overview", "everything", "summary", "full report"],
}
UPGRADE = ["upgrade", "add more", "add ram", "more ram", "free slot", "slots", "slot", "maximum", "max ram", "expand",
           "can i add", "how much can", "supports up to"]
FOLLOWUP = ["it", "that", "those", "this", "them", "more", "and", "also", "what about", "how about", "details"]
TOOLS_FOR = {"ram": ["get_memory_info"], "cpu": ["get_cpu_info"], "storage": ["get_storage_info"],
             "gpu": ["get_gpu_info"], "battery": ["get_battery_info"], "system": ["get_system_info"],
             "diskspace": ["get_disk_usage"],
             "overview": ["get_system_info", "get_cpu_info", "get_memory_topology", "get_storage_info", "get_gpu_info"]}


def _has(q: str, words) -> bool:
    return any(re.search(rf"(?<![a-z]){re.escape(w)}(?![a-z])", q) for w in words)


def route(question: str, last_topics: List[str]) -> List[str]:
    q = question.lower()
    topics = [t for t, w in TOPICS.items() if _has(q, w)]
    if "diskspace" in topics and "storage" in topics:
        topics.remove("storage")
    names: List[str] = []
    upgrade = _has(q, UPGRADE)
    if upgrade and ("ram" in topics or not topics) and (topics or "ram" in last_topics or not last_topics):
        return ["analyze_ram_upgrade"]
    if not topics and last_topics and _has(q, FOLLOWUP):
        topics = list(last_topics)
    for t in topics:
        if t == "ram" and "ram" in topics and _has(q, ["slots", "slot", "dimm", "installed", "type", "ddr", "speed", "modules"]):
            names.append("get_memory_topology")
        else:
            names += TOOLS_FOR[t]
    return list(dict.fromkeys(names))


def topics_of(tools: List[str]) -> List[str]:
    rev = {"get_memory_info": "ram", "get_memory_topology": "ram", "analyze_ram_upgrade": "ram", "get_cpu_info": "cpu",
           "get_storage_info": "storage", "get_gpu_info": "gpu", "get_battery_info": "battery",
           "get_system_info": "system", "get_disk_usage": "diskspace"}
    return list(dict.fromkeys(rev[t] for t in tools if t in rev))


# ------------------------------------------------------------------ renderers (JSON -> sentences)
def _unavailable(name: str, d: dict) -> str:
    msg = f"I can't determine that ({name}): {d.get('reason') or d.get('detail') or 'data unavailable'}."
    if d.get("how_to_fix"):
        msg += f" {d['how_to_fix']}"
    return msg


def r_cpu(d):
    cores, threads = d.get("physical_cores"), d.get("logical_cpus")
    s = f"Your CPU is {d.get('model_name') or 'an unknown model'}"
    if cores and threads:
        s += f" with {cores} core{'s' if cores != 1 else ''} / {threads} thread{'s' if threads != 1 else ''}"
    elif threads:
        s += f" with {threads} logical CPUs"
    if d.get("max_mhz"):
        s += f", boosting up to {d['max_mhz'] / 1000:.1f} GHz"
    return s + "."


def r_memory(d):
    return (f"The OS sees {d['os_visible_total']} of RAM, with {d['available_now']} currently available"
            + (f" (swap: {d['swap_total']})." if d.get("swap_total") else "."))


def r_topology(d):
    if d["status"] != "ok":
        s = f"I can see the OS reports {d.get('os_visible_total')} of RAM, but I can't determine the slot layout."
        return s + (" " + d["how_to_fix"] if d.get("how_to_fix") else "")
    mods = ", ".join(f"{m['size']} {m['type'] or ''} {m['speed'] or ''}".replace("  ", " ").strip() for m in d["modules"])
    return (f"You have {d['installed_total']} installed ({mods}) in {d['used_slots']} of {d['total_slots']} slots; "
            f"maximum supported is {d['max_supported'] or 'unknown'}.")


def r_upgrade(d):
    if d.get("reason") == "unreliable_firmware_data":
        k = d["known"]
        return ("I can't reliably determine whether RAM can be added. The firmware reports " + str(k["installed"]) +
                " installed, " + str(k["reported_slots"]) + " slots and a " + str(k["reported_max"]) + " maximum, but: "
                + " ".join(d["warnings"]))
    if d["status"] != "ok":
        s = (f"I can't determine whether you can add RAM. The OS reports {d['known']['os_visible_total']}, but slot count "
             f"and the board's maximum aren't available ({d.get('reason')}).")
        return s + (" " + d["how_to_fix"] if d.get("how_to_fix") else "")
    s = (f"You have {d['installed']} in {d['slots']['used']} of {d['slots']['total']} slots. "
         f"The board supports up to {d['max_supported']}")
    if d["can_add_memory"]:
        s += (f", so you can add RAM: up to {d['headroom']} more, and the largest single module that fits is "
              f"{d['largest_single_module_that_fits']}.")
    elif d["slots"]["free"] == 0:
        s += ". There are no free slots, so you'd have to replace existing modules to add capacity."
    else:
        s += ", and you're already at (or near) the maximum."
    return s + " Caveat: " + d["caveats"][0]


def r_storage(d):
    kinds = {"nvme_ssd": "NVMe SSD", "sata_or_other_ssd": "SATA/other SSD", "hdd": "hard disk (HDD)",
             "virtual_disk": "virtual disk (underlying media type can't be determined from inside the VM)"}
    if not d["disks"]:
        return "I didn't find any matching physical disks."
    return " ".join(f"{x['name']}: {x['size_human']} {kinds[x['kind']]}" + (f" ({x['model']})." if x["model"] else ".")
                    for x in d["disks"])


def r_usage(d):
    return " ".join(f"{f['mountpoint']}: {f['free_human']} free of {f['size_human']} ({f['used_percent']}% used)."
                    for f in d["filesystems"]) or "No filesystems found."


def r_gpu(d):
    parts = [f"{g['vendor']} {g['device']}" for g in d["gpus"]]
    for n in d["nvidia_details"]:
        parts.append(f"{n['name']} with {n['vram_human']} VRAM (driver {n['driver_version']})")
    return ("Graphics: " + "; ".join(parts) + ".") if parts else "I couldn't find a graphics adapter in the data available."


def r_battery(d):
    return " ".join(f"{b['name']}: {b['charge_percent']}% ({b['status']})"
                    + (f", battery health {b['health_percent']}% of design capacity" if b["health_percent"] else "")
                    + (f", {b['cycle_count']} cycles." if b["cycle_count"] else ".") for b in d["batteries"])


def r_system(d):
    m, o = d.get("machine") or {}, (d.get("os") or {})
    s = f"{(m.get('sys_vendor') or 'Unknown vendor')} {m.get('product_name') or ''}".strip()
    s += f" running {o.get('pretty_name') or 'an unknown OS'}"
    if d.get("kernel"):
        s += f", kernel {d['kernel']}"
    return s + "."


RENDER = {"get_cpu_info": r_cpu, "get_memory_info": r_memory, "get_memory_topology": r_topology,
          "analyze_ram_upgrade": r_upgrade, "get_storage_info": r_storage, "get_disk_usage": r_usage,
          "get_gpu_info": r_gpu, "get_battery_info": r_battery, "get_system_info": r_system}


class OfflineBackend:
    name = "offline"

    def __init__(self):
        self.last_topics: List[str] = []

    def chat(self, messages: list, tools: list, system: str) -> dict:
        last = messages[-1]
        if last["role"] == "user":
            names = [n for n in route(last["content"], self.last_topics) if n in {t["name"] for t in tools}]
            if not names:
                return {"text": ("I can't determine that from the hardware tools I have. I can answer questions about CPU, "
                                 "RAM (including upgrades), storage, GPU, battery, disk space and system/OS details."),
                        "tool_calls": []}
            self.last_topics = topics_of(names) or self.last_topics
            return {"text": "", "tool_calls": [{"id": uuid.uuid4().hex[:8], "name": n, "arguments": {}} for n in names]}
        results = []
        for m in reversed(messages):
            if m["role"] != "tool":
                break
            results.append(m)
        lines = []
        for m in reversed(results):
            d = json.loads(m["content"])["data"]
            lines.append(_unavailable(m["name"], d) if d.get("status") == "unavailable" else RENDER[m["name"]](d))
        return {"text": "\n".join(lines), "tool_calls": []}
