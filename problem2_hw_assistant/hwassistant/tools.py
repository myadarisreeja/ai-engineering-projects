"""Tool registry: what the LLM can call. Each tool returns normalised JSON with an explicit `status`
(ok | partial | unavailable) so the model can say "I can't determine that" instead of guessing."""
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from . import parsers as P
from .executor import BaseExecutor, ToolError

FIX_ROOT = ("Root access is needed for DMI memory data. Run deploy/install_sudoers.sh once (adds a narrow "
            "passwordless rule for `dmidecode -t memory` only), or run the assistant with sudo.")


@dataclass
class ToolSpec:
    name: str
    description: str
    func: Callable
    parameters: Dict[str, Any] = field(default_factory=lambda: {"type": "object", "properties": {}, "required": []})
    volatile: bool = False                      # volatile results are re-collected, static ones are cached
    probe: Optional[Callable[[BaseExecutor], bool]] = None   # hide the tool if no data source exists

    def schema(self):
        return {"name": self.name, "description": self.description, "parameters": self.parameters}


def validate_args(schema: dict, args: dict) -> dict:
    """Strict validation: unknown keys, wrong types, enums and patterns are rejected."""
    props, clean = schema.get("properties", {}), {}
    if not isinstance(args, dict):
        raise ValueError("arguments must be an object")
    for k, v in args.items():
        if k not in props:
            raise ValueError(f"unknown argument '{k}'")
        spec = props[k]
        t = spec.get("type")
        if t == "boolean" and not isinstance(v, bool):
            raise ValueError(f"'{k}' must be boolean")
        if t == "string":
            if not isinstance(v, str) or len(v) > 64:
                raise ValueError(f"'{k}' must be a short string")
            if "enum" in spec and v not in spec["enum"]:
                raise ValueError(f"'{k}' must be one of {spec['enum']}")
            if "pattern" in spec and not re.fullmatch(spec["pattern"], v):
                raise ValueError(f"'{k}' has invalid format")
        clean[k] = v
    for r in schema.get("required", []):
        if r not in clean:
            raise ValueError(f"missing required argument '{r}'")
    return clean


FIX_BINARY = "Install the missing tool (e.g. `sudo apt install dmidecode pciutils`) and ask again."


FIX_NO_DMI = ("This system does not expose firmware memory tables (typical in containers, some virtual machines and "
              "non-x86 boards), so slot layout and maximum RAM cannot be read here.")
VM_MARKERS = ("qemu", "kvm", "vmware", "virtualbox", "innotek", "xen", "bochs", "parallels", "bhyve",
              "virtual machine", "amazon ec2", "google compute", "openstack")


def _fix(reason: str):
    return (FIX_ROOT if reason == "permission" else FIX_BINARY if reason == "missing_binary"
            else FIX_NO_DMI if reason == "no_dmi_data" else None)


def detect_virtualization(ex: BaseExecutor) -> Optional[str]:
    """Returns a hint like 'VMware, Inc.' when DMI vendor/product strings indicate a VM, else None."""
    vals = []
    for f in ("sys_vendor", "product_name"):
        try:
            vals.append(ex.read_file(f"/sys/class/dmi/id/{f}").strip())
        except ToolError:
            pass
    blob = " ".join(vals).lower()
    return " / ".join(vals) if any(m in blob for m in VM_MARKERS) else None


def _err(e: ToolError, extra=None) -> dict:
    fix = _fix(e.reason)
    return {"status": "unavailable", "reason": e.reason, "detail": e.detail, "how_to_fix": fix, **(extra or {})}


# --------------------------------------------------------------------------- tools
def get_cpu_info(ex: BaseExecutor, args: dict) -> dict:
    try:
        d = P.parse_lscpu(ex.run("lscpu"))
    except ToolError as e:
        try:
            raw = ex.read_file("/proc/cpuinfo")
            kv = P.parse_kv_lines(raw)
            return {"status": "partial", "model_name": kv.get("model name"), "logical_cpus": raw.count("processor\t:"),
                    "notes": ["lscpu unavailable; used /proc/cpuinfo, core/cache details missing"]}
        except ToolError:
            return _err(e)
    return {"status": "ok", **d}


def get_memory_info(ex: BaseExecutor, args: dict) -> dict:
    try:
        d = P.parse_free(ex.run("free"))
    except ToolError:
        try:
            m = P.parse_meminfo(ex.read_file("/proc/meminfo"))
            d = {"total": m["total"], "available": m["available"], "free": m["free"], "swap_total": m["swap_total"]}
        except ToolError as e:
            return _err(e)
    return {"status": "ok", "os_visible_total": P.human(d.get("total")), "os_visible_total_bytes": d.get("total"),
            "available_now": P.human(d.get("available")), "swap_total": P.human(d.get("swap_total")),
            "notes": ["os_visible_total is slightly below installed RAM (firmware/GPU reservations). "
                      "Use get_memory_topology for installed module sizes."]}


def get_memory_topology(ex: BaseExecutor, args: dict) -> dict:
    include_empty = args.get("include_empty_slots", True)
    try:
        parsed = P.parse_dmidecode_memory(ex.run("dmidecode_memory"))
    except ToolError as e:
        fallback = get_memory_info(ex, {})
        return {"status": "partial" if fallback.get("status") == "ok" else "unavailable", "reason": e.reason,
                "how_to_fix": _fix(e.reason),
                "os_visible_total": fallback.get("os_visible_total"),
                "unknown": ["installed module sizes", "total slots", "free slots", "maximum supported capacity"]}
    devs, arrays = parsed["devices"], parsed["arrays"]
    if not devs and not arrays:
        return {"status": "unavailable", "reason": "unparseable", "detail": "dmidecode returned no memory records"}
    installed = [d for d in devs if d["installed"]]
    max_caps = [a["max_capacity_bytes"] for a in arrays if a["max_capacity_bytes"]]
    max_cap = sum(max_caps) if max_caps else None
    declared_slots = sum(a["slots"] or 0 for a in arrays) or None
    total_slots = len(devs) if devs else (declared_slots or 0)
    warnings = []
    # real firmware is messy: cross-check the two independent slot counts and the array linkage
    array_handles = {a["handle"] for a in arrays if a["handle"]}
    if declared_slots and devs and declared_slots != len(devs):
        warnings.append(f"Firmware tables disagree: the memory array declares {declared_slots} slots but "
                        f"{len(devs)} memory-device records exist. Slot counts may be unreliable.")
    if array_handles and devs and not any(d["array_handle"] in array_handles for d in devs):
        warnings.append("Memory-device records are not linked to the declared memory array (common on virtual machines).")
    vm = detect_virtualization(ex)
    if vm:
        warnings.append(f"This looks like a virtual machine ({vm}); memory is virtual and normally resized in the "
                        "hypervisor, not with physical modules.")
    out = {
        "status": "ok", "total_slots": total_slots, "used_slots": len(installed),
        "free_slots": total_slots - len(installed),
        "installed_total_bytes": sum(d["size_bytes"] for d in installed),
        "installed_total": P.human(sum(d["size_bytes"] for d in installed)),
        "max_supported_bytes": max_cap, "max_supported": P.human(max_cap),
        "memory_type": next((d["type"] for d in installed if d["type"]), None),
        "any_soldered": any(d["soldered"] for d in installed),
        "virtual_machine": vm, "data_quality_warnings": warnings,
        "modules": [{"slot": d["locator"], "size": P.human(d["size_bytes"]), "type": d["type"], "speed": d["speed"],
                     "form_factor": d["form_factor"], "manufacturer": d["manufacturer"], "soldered": d["soldered"]}
                    for d in installed],
    }
    if include_empty:
        empty = [d["locator"] for d in devs if not d["installed"]]
        out["empty_slots"] = empty if len(empty) <= 16 else empty[:16] + [f"... and {len(empty) - 16} more"]
    if max_cap is None:
        out["status"] = "partial"
        out["unknown"] = ["maximum supported capacity"]
    elif warnings:
        out["status"] = "partial"          # present the numbers, but flag that they are not trustworthy
    return out


def analyze_ram_upgrade(ex: BaseExecutor, args: dict) -> dict:
    """Deterministic arithmetic lives here, NOT in the LLM."""
    t = get_memory_topology(ex, {"include_empty_slots": False})
    if t["status"] == "partial" and t.get("data_quality_warnings") and t.get("max_supported_bytes"):
        return {"status": "partial", "can_determine_upgradeability": False, "reason": "unreliable_firmware_data",
                "how_to_fix": None, "warnings": t["data_quality_warnings"], "virtual_machine": t.get("virtual_machine"),
                "known": {"installed": t["installed_total"], "os_visible_total": None,
                          "reported_max": t["max_supported"], "reported_slots": t["total_slots"]},
                "unknown": ["whether memory can really be added"]}
    if t["status"] != "ok":
        return {"status": "unavailable" if t["status"] == "unavailable" else "partial",
                "can_determine_upgradeability": False, "reason": t.get("reason"), "how_to_fix": t.get("how_to_fix"),
                "known": {"os_visible_total": t.get("os_visible_total")}, "unknown": t.get("unknown")}
    installed, max_cap, slots, free = t["installed_total_bytes"], t["max_supported_bytes"], t["total_slots"], t["free_slots"]
    headroom = max(0, max_cap - installed)
    per_slot_limit = max_cap // slots if slots else None
    candidates = [s << 30 for s in (4, 8, 16, 32, 64, 128)]
    fits = [c for c in candidates if c <= headroom and (per_slot_limit is None or c <= per_slot_limit)]
    can_add = free > 0 and bool(fits)
    caveats = ["Maximum capacity comes from firmware tables (SMBIOS); some vendors report it inaccurately - "
               "check the manufacturer's spec sheet before buying.",
               "Match the memory type/speed of installed modules (" + (t["memory_type"] or "unknown type") + ")."]
    if t["any_soldered"]:
        caveats.append("Some installed memory appears to be soldered onto the board and cannot be replaced.")
    return {"status": "ok", "can_add_memory": can_add, "installed": t["installed_total"],
            "slots": {"total": slots, "used": t["used_slots"], "free": free},
            "max_supported": t["max_supported"], "headroom": P.human(headroom),
            "largest_single_module_that_fits": P.human(max(fits)) if fits else None,
            "per_slot_limit_estimate": P.human(per_slot_limit), "memory_type": t["memory_type"],
            "caveats": caveats}


def get_storage_info(ex: BaseExecutor, args: dict) -> dict:
    try:
        disks = P.parse_lsblk(ex.run("lsblk"))
    except ToolError as e:
        return _err(e)
    except (ValueError, KeyError) as e:
        return {"status": "unavailable", "reason": "unparseable", "detail": str(e)}
    if args.get("device"):
        disks = [d for d in disks if d["name"] == args["device"]]
    return {"status": "ok", "disks": disks, "notes": ["kind: nvme_ssd | sata_or_other_ssd | hdd | virtual_disk (from name, transport and rotational flag; virtual disks fake the flag)"]}


def get_disk_usage(ex: BaseExecutor, args: dict) -> dict:
    try:
        return {"status": "ok", "filesystems": P.parse_df(ex.run("df"))}
    except ToolError as e:
        return _err(e)


def get_gpu_info(ex: BaseExecutor, args: dict) -> dict:
    gpus, notes, got_any = [], [], False
    try:
        gpus = P.parse_lspci_gpus(ex.run("lspci"))
        got_any = True
    except ToolError as e:
        notes.append(f"lspci unavailable ({e.reason}); install pciutils for full GPU detection")
    nv = []
    if ex.has_command("nvidia_smi"):
        try:
            nv = P.parse_nvidia_smi(ex.run("nvidia_smi"))
            got_any = True
        except ToolError:
            notes.append("nvidia-smi present but failed (driver not loaded?)")
    if not got_any:
        return {"status": "unavailable", "reason": "missing_binary", "notes": notes}
    return {"status": "ok", "gpus": gpus, "nvidia_details": nv, "notes": notes}


def get_system_info(ex: BaseExecutor, args: dict) -> dict:
    out: Dict[str, Any] = {"status": "ok", "missing": []}
    try:
        out["os"] = P.parse_os_release(ex.read_file("/etc/os-release"))
    except ToolError:
        out["missing"].append("os-release")
    try:
        out["kernel"] = ex.run("uname").strip()
    except ToolError:
        out["missing"].append("kernel")
    dmi = {}
    for f in ("sys_vendor", "product_name", "product_version", "bios_version", "bios_date"):
        try:
            dmi[f] = ex.read_file(f"/sys/class/dmi/id/{f}").strip()
        except ToolError:
            pass
    out["machine"] = dmi or None
    return out


def get_battery_info(ex: BaseExecutor, args: dict) -> dict:
    bats = [n for n in ex.list_dir("/sys/class/power_supply") if re.fullmatch(r"BAT[0-9]+", n)]
    if not bats:
        return {"status": "unavailable", "reason": "no_battery", "detail": "No battery found (desktop or VM?)"}
    res = []
    for b in bats:
        def rd(k):
            try:
                return ex.read_file(f"/sys/class/power_supply/{b}/{k}").strip()
            except ToolError:
                return None
        full = rd("energy_full") or rd("charge_full")
        design = rd("energy_full_design") or rd("charge_full_design")
        health = round(100 * int(full) / int(design), 1) if full and design and design.isdigit() and full.isdigit() and int(design) else None
        res.append({"name": b, "charge_percent": rd("capacity"), "status": rd("status"), "health_percent": health,
                    "cycle_count": rd("cycle_count"), "technology": rd("technology"),
                    "manufacturer": rd("manufacturer"), "model": rd("model_name")})
    return {"status": "ok", "batteries": res}


# --------------------------------------------------------------------------- registry
def _has_battery(ex):
    return any(n.startswith("BAT") for n in ex.list_dir("/sys/class/power_supply"))


TOOLS: Dict[str, ToolSpec] = {t.name: t for t in [
    ToolSpec("get_cpu_info", "CPU model, architecture, physical cores, logical CPUs/threads, max frequency, cache.", get_cpu_info),
    ToolSpec("get_memory_info", "RAM as seen by the OS: total, currently available, swap. Does NOT include slot information.",
             get_memory_info, volatile=True),
    ToolSpec("get_memory_topology", "Physical RAM layout from firmware: installed modules, total/free slots, memory type and speed, "
             "maximum supported capacity. May need root.", get_memory_topology,
             parameters={"type": "object", "properties": {"include_empty_slots": {"type": "boolean",
                         "description": "Also list empty slot names (default true)."}}, "required": []}),
    ToolSpec("analyze_ram_upgrade", "Answers 'can I add RAM and how much?': free slots, max supported, headroom, largest module that fits. "
             "Use for any RAM upgrade question.", analyze_ram_upgrade),
    ToolSpec("get_storage_info", "Physical disks: model, size, type (NVMe SSD / SATA SSD / HDD), partitions.", get_storage_info,
             parameters={"type": "object", "properties": {"device": {"type": "string", "pattern": "[a-z0-9]{2,16}",
                         "description": "Optional device name such as nvme0n1 or sda."}}, "required": []}),
    ToolSpec("get_disk_usage", "Filesystem free/used space per mount point.", get_disk_usage, volatile=True),
    ToolSpec("get_gpu_info", "Graphics adapters (integrated and discrete) and NVIDIA VRAM/driver when available.", get_gpu_info,
             probe=lambda ex: ex.has_command("lspci") or ex.has_command("nvidia_smi")),
    ToolSpec("get_system_info", "Linux distribution, kernel, laptop vendor/model, BIOS version.", get_system_info),
    ToolSpec("get_battery_info", "Battery charge, status, health (wear) and cycle count.", get_battery_info, volatile=True,
             probe=_has_battery),
]}


def available_tools(ex: BaseExecutor) -> List[ToolSpec]:
    """Dynamic tool registry: only expose tools that can possibly work on this machine."""
    out = []
    for t in TOOLS.values():
        try:
            if t.probe is None or t.probe(ex):
                out.append(t)
        except Exception:
            continue
    return out
