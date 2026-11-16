"""Pure functions: raw command output -> normalised dicts. No I/O, fully unit-testable."""
import json
import re
import shlex
from typing import Dict, List, Optional

UNIT = {"B": 1, "KB": 1 << 10, "MB": 1 << 20, "GB": 1 << 30, "TB": 1 << 40}


def parse_size(text: str) -> Optional[int]:
    """'16 GB' / '64 GB' / '16384 MB' -> bytes (binary units, as dmidecode uses). None if unknown."""
    m = re.match(r"^\s*(\d+(?:\.\d+)?)\s*([KMGT]?B)\s*$", text or "", re.I)
    return int(float(m.group(1)) * UNIT[m.group(2).upper()]) if m else None


def human(n: Optional[int]) -> Optional[str]:
    if n is None:
        return None
    for unit, div in (("TB", 1 << 40), ("GB", 1 << 30), ("MB", 1 << 20)):
        if n >= div:
            v = n / div
            return f"{v:.1f}".rstrip("0").rstrip(".") + f" {unit}"
    return f"{n} B"


def parse_kv_lines(text: str, sep: str = ":") -> Dict[str, str]:
    out = {}
    for line in text.splitlines():
        if sep in line:
            k, v = line.split(sep, 1)
            out[k.strip()] = v.strip().strip('"')
    return out


def parse_lscpu(text: str) -> dict:
    kv = parse_kv_lines(text)
    def i(k):
        try:
            return int(kv[k])
        except (KeyError, ValueError):
            return None
    def f(k):
        try:
            return float(kv[k])
        except (KeyError, ValueError):
            return None
    sockets, cps, tpc = i("Socket(s)"), i("Core(s) per socket"), i("Thread(s) per core")
    return {
        "model_name": kv.get("Model name"), "vendor": kv.get("Vendor ID"), "architecture": kv.get("Architecture"),
        "logical_cpus": i("CPU(s)"),
        "physical_cores": sockets * cps if sockets and cps else None,
        "threads_per_core": tpc, "sockets": sockets,
        "max_mhz": f("CPU max MHz"), "min_mhz": f("CPU min MHz"),
        "l3_cache": kv.get("L3 cache"),
    }


def parse_free(text: str) -> dict:
    out = {}
    for line in text.splitlines():
        parts = line.split()
        if parts and parts[0] == "Mem:" and len(parts) >= 7:
            out.update(total=int(parts[1]), used=int(parts[2]), free=int(parts[3]), available=int(parts[6]))
        elif parts and parts[0] == "Swap:" and len(parts) >= 3:
            out.update(swap_total=int(parts[1]), swap_used=int(parts[2]))
    return out


def parse_meminfo(text: str) -> dict:
    out = {}
    for line in text.splitlines():
        m = re.match(r"(\w+):\s+(\d+)\s*kB", line)
        if m:
            out[m.group(1)] = int(m.group(2)) * 1024
    return {"total": out.get("MemTotal"), "available": out.get("MemAvailable"), "free": out.get("MemFree"),
            "swap_total": out.get("SwapTotal")}


def _truthy(v) -> bool:
    return v in (True, 1, "1", "true", "True")


def parse_lsblk(text: str) -> List[dict]:
    data = json.loads(text)
    disks = []
    for dev in data.get("blockdevices", []):
        if dev.get("type") != "disk" or re.match(r"^(loop|ram|zram|sr|fd)", dev.get("name", "")):
            continue
        name, tran = dev["name"], (dev.get("tran") or "").lower()
        size = int(dev["size"]) if dev.get("size") not in (None, "") else None
        model_l = (dev.get("model") or "").lower()
        if re.match(r"^(vd|xvd)[a-z]", name) or any(m in model_l for m in ("qemu", "vmware", "virtual", "vbox")):
            kind = "virtual_disk"        # hypervisors fake the rotational flag; real media type is unknowable
        elif name.startswith("nvme") or tran == "nvme":
            kind = "nvme_ssd"
        elif _truthy(dev.get("rota")):
            kind = "hdd"
        else:
            kind = "sata_or_other_ssd"
        parts = [{"name": c["name"], "size_bytes": int(c["size"]), "size_human": human(int(c["size"])),
                  "mountpoint": c.get("mountpoint")} for c in dev.get("children", []) if c.get("type") == "part"]
        disks.append({"name": name, "model": (dev.get("model") or "").strip() or None, "size_bytes": size,
                      "size_human": human(size), "kind": kind, "transport": tran or None,
                      "rotational": _truthy(dev.get("rota")), "partitions": parts})
    return disks


def parse_dmidecode_memory(text: str) -> dict:
    """Returns {'arrays': [{max_capacity_bytes, slots}], 'devices': [{locator,size_bytes,...}]}."""
    arrays, devices = [], []
    for block in re.split(r"\n\s*\n", text):
        hm = re.search(r"Handle (0x[0-9A-Fa-f]+), DMI type", block)
        handle = hm.group(1).lower() if hm else None
        if "Physical Memory Array" in block:
            kv = parse_kv_lines(block)
            try:
                slots = int(kv.get("Number Of Devices", ""))
            except ValueError:
                slots = None
            arrays.append({"handle": handle, "max_capacity_bytes": parse_size(kv.get("Maximum Capacity", "")), "slots": slots,
                           "max_capacity_raw": kv.get("Maximum Capacity")})
        elif re.search(r"^Memory Device\s*$", block, re.M):
            kv = parse_kv_lines(block)
            size_raw = kv.get("Size", "")
            installed = "no module" not in size_raw.lower() and parse_size(size_raw) is not None
            speed = kv.get("Configured Memory Speed") or kv.get("Configured Clock Speed") or kv.get("Speed")
            form = kv.get("Form Factor", "")
            devices.append({
                "array_handle": (kv.get("Array Handle") or "").lower() or None,
                "locator": kv.get("Locator"), "installed": installed,
                "size_bytes": parse_size(size_raw) if installed else 0,
                "type": kv.get("Type") if installed else None,
                "speed": speed if installed and speed and "unknown" not in speed.lower() else None,
                "form_factor": form or None, "manufacturer": kv.get("Manufacturer") if installed else None,
                "soldered": form.lower() in ("row of chips", "chip", "tssop", "soic", "srimm") or
                            "soldered" in kv.get("Locator", "").lower() or kv.get("Bank Locator", "").lower().startswith("onboard"),
            })
    return {"arrays": arrays, "devices": devices}


def parse_lspci_gpus(text: str) -> List[dict]:
    gpus = []
    for line in text.splitlines():
        try:
            parts = shlex.split(line)
        except ValueError:
            continue
        if len(parts) >= 4 and re.search(r"VGA|3D controller|Display controller", parts[1], re.I):
            gpus.append({"pci_slot": parts[0], "class": parts[1], "vendor": parts[2], "device": parts[3]})
    return gpus


def parse_nvidia_smi(text: str) -> List[dict]:
    out = []
    for line in text.strip().splitlines():
        cols = [c.strip() for c in line.split(",")]
        if len(cols) >= 3:
            try:
                vram = int(float(cols[1])) << 20
            except ValueError:
                vram = None
            out.append({"name": cols[0], "vram_bytes": vram, "vram_human": human(vram), "driver_version": cols[2]})
    return out


def parse_os_release(text: str) -> dict:
    kv = parse_kv_lines(text, "=")
    return {"pretty_name": kv.get("PRETTY_NAME"), "id": kv.get("ID"), "version": kv.get("VERSION_ID")}


def parse_df(text: str) -> List[dict]:
    rows = []
    for line in text.splitlines()[1:]:
        p = line.split(None, 3)
        if len(p) == 4 and p[1].isdigit():
            size, used = int(p[1]), int(p[2])
            rows.append({"filesystem": p[0], "mountpoint": p[3], "size_bytes": size, "size_human": human(size),
                         "used_human": human(used), "free_human": human(size - used),
                         "used_percent": round(100 * used / size, 1) if size else None})
    return rows
