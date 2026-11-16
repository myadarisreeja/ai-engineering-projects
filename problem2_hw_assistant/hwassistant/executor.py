"""Safe system access layer. This is the ONLY module that touches the OS.

Security model
  * The LLM never supplies a command string. It picks a tool NAME + validated scalar args.
  * Commands come from ALLOWED_COMMANDS (fixed argv, no user-controlled arguments at all).
  * Binaries are resolved from fixed system directories (not $PATH -> no PATH hijacking).
  * subprocess is invoked with a list and shell=False; stdin closed; timeout; scrubbed env; LC_ALL=C.
  * Files readable only if they match a strict regex allow-list (/proc, /sys, /etc/os-release...).
  * Root-only data (dmidecode) goes through `sudo -n` against ONE fixed binary + fixed args, matched by
    an equally narrow sudoers rule (see deploy/). No password prompts, no general root.
"""
import os
import re
import shutil
import subprocess
from typing import Dict, List, Optional, Tuple

BIN_DIRS = ("/usr/bin", "/usr/sbin", "/bin", "/sbin", "/usr/local/bin")
TIMEOUT_S = 5
MAX_BYTES = 256 * 1024

# key -> (argv after the binary name, needs_root)
ALLOWED_COMMANDS: Dict[str, Tuple[List[str], bool]] = {
    "lscpu": (["lscpu"], False),
    "lsblk": (["lsblk", "-J", "-b", "-o", "NAME,SIZE,TYPE,MOUNTPOINT,MODEL,ROTA,TRAN"], False),
    "free": (["free", "-b"], False),
    "df": (["df", "-B1", "-x", "tmpfs", "-x", "devtmpfs", "-x", "squashfs", "-x", "overlay",
            "--output=source,size,used,target"], False),
    "lspci": (["lspci", "-mm"], False),
    "nvidia_smi": (["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
                    "--format=csv,noheader,nounits"], False),
    "uname": (["uname", "-srm"], False),
    "dmidecode_memory": (["dmidecode", "-t", "memory"], True),
}

ALLOWED_FILE_PATTERNS = [re.compile(p) for p in (
    r"^/proc/meminfo$",
    r"^/proc/cpuinfo$",
    r"^/etc/os-release$",
    r"^/sys/class/dmi/id/(sys_vendor|product_name|product_version|bios_vendor|bios_version|bios_date|board_vendor|board_name)$",
    r"^/sys/class/power_supply/BAT[0-9]+/(capacity|status|energy_full|energy_full_design|energy_now|"
    r"charge_full|charge_full_design|charge_now|cycle_count|technology|manufacturer|model_name)$",
)]
ALLOWED_DIRS = {"/sys/class/power_supply"}


class ToolError(Exception):
    """reason: not_allowed | missing_binary | permission | timeout | failed | missing_file"""

    def __init__(self, reason: str, detail: str = ""):
        super().__init__(f"{reason}: {detail}")
        self.reason, self.detail = reason, detail


class BaseExecutor:
    def _check_command(self, key: str):
        if key not in ALLOWED_COMMANDS:
            raise ToolError("not_allowed", f"command '{key}' is not allow-listed")
        return ALLOWED_COMMANDS[key]

    def _check_file(self, path: str):
        if not any(p.match(path) for p in ALLOWED_FILE_PATTERNS):
            raise ToolError("not_allowed", f"file '{path}' is not allow-listed")

    def _check_dir(self, path: str):
        if path not in ALLOWED_DIRS:
            raise ToolError("not_allowed", f"directory '{path}' is not allow-listed")

    # interface
    def run(self, key: str) -> str: raise NotImplementedError
    def read_file(self, path: str) -> str: raise NotImplementedError
    def list_dir(self, path: str) -> List[str]: raise NotImplementedError
    def has_command(self, key: str) -> bool: raise NotImplementedError
    def root_available(self) -> bool: raise NotImplementedError


def resolve_binary(name: str) -> Optional[str]:
    for d in BIN_DIRS:
        p = os.path.join(d, name)
        if os.path.isfile(p) and os.access(p, os.X_OK):
            return p
    return None


class SystemExecutor(BaseExecutor):
    ENV = {"LC_ALL": "C", "PATH": "/usr/sbin:/usr/bin:/sbin:/bin"}

    def has_command(self, key: str) -> bool:
        argv, _ = self._check_command(key)
        return resolve_binary(argv[0]) is not None

    def root_available(self) -> bool:
        """True if dmidecode exists AND we are root or a sudoers rule allows `dmidecode -t memory` without a password."""
        dmi = resolve_binary("dmidecode")
        if not dmi:
            return False
        if os.geteuid() == 0:
            return True
        sudo = resolve_binary("sudo")
        if not sudo:
            return False
        try:
            r = subprocess.run([sudo, "-n", "-l", dmi, "-t", "memory"], capture_output=True, text=True,
                               timeout=TIMEOUT_S, stdin=subprocess.DEVNULL, env=self.ENV)
            return r.returncode == 0
        except Exception:
            return False

    def run(self, key: str) -> str:
        argv, needs_root = self._check_command(key)
        binary = resolve_binary(argv[0])
        if not binary:
            raise ToolError("missing_binary", f"'{argv[0]}' is not installed")
        full = [binary, *argv[1:]]
        if needs_root and os.geteuid() != 0:
            sudo = resolve_binary("sudo")
            if not sudo:
                raise ToolError("permission", "root required and sudo is not available")
            full = [sudo, "-n", *full]          # -n: never prompt; fail fast if no sudoers rule
        try:
            r = subprocess.run(full, capture_output=True, text=True, timeout=TIMEOUT_S, shell=False,
                               stdin=subprocess.DEVNULL, env=self.ENV)
        except subprocess.TimeoutExpired:
            raise ToolError("timeout", f"'{key}' exceeded {TIMEOUT_S}s")
        except OSError as e:
            raise ToolError("failed", str(e))
        if r.returncode != 0:
            err = ((r.stderr or "") + (r.stdout or "")).strip()
            low = err.lower()
            if "no smbios nor dmi entry point" in low or "/dev/mem" in low or "can't read memory" in low:
                raise ToolError("no_dmi_data", "firmware (SMBIOS/DMI) tables are not exposed on this system "
                                "(container, some VMs, or non-x86 boards)")
            if needs_root and ("password is required" in low or "not allowed" in low or "permission denied" in low
                               or "must be root" in low or low.startswith("sudo:")):
                raise ToolError("permission", "root privileges are required (no passwordless sudo rule found)")
            raise ToolError("failed", err[:200] or f"exit code {r.returncode}")
        return r.stdout[:MAX_BYTES]

    def read_file(self, path: str) -> str:
        self._check_file(path)
        try:
            with open(path, "r", errors="replace") as f:
                return f.read(MAX_BYTES)
        except FileNotFoundError:
            raise ToolError("missing_file", path)
        except PermissionError:
            raise ToolError("permission", path)
        except OSError as e:
            raise ToolError("failed", str(e))

    def list_dir(self, path: str) -> List[str]:
        self._check_dir(path)
        try:
            return sorted(os.listdir(path))
        except OSError:
            return []


class FixtureExecutor(BaseExecutor):
    """Replays captured outputs from a fixture directory. Used for tests, demos and non-Linux hosts.
    Layout: <dir>/cmd/<key>.txt and <dir>/rootfs/<absolute path>. A missing root-only command file
    simulates 'permission denied'."""

    def __init__(self, directory: str):
        self.dir = directory
        self.root = os.path.exists(os.path.join(directory, "cmd", "dmidecode_memory.txt"))

    def has_command(self, key: str) -> bool:
        self._check_command(key)
        return os.path.exists(os.path.join(self.dir, "cmd", f"{key}.txt"))

    def root_available(self) -> bool:
        return self.root

    def run(self, key: str) -> str:
        _, needs_root = self._check_command(key)
        p = os.path.join(self.dir, "cmd", f"{key}.txt")
        if not os.path.exists(p):
            raise ToolError("permission" if needs_root else "missing_binary", f"fixture has no '{key}'")
        return open(p).read()

    def read_file(self, path: str) -> str:
        self._check_file(path)
        p = os.path.join(self.dir, "rootfs", path.lstrip("/"))
        if not os.path.exists(p):
            raise ToolError("missing_file", path)
        return open(p).read()

    def list_dir(self, path: str) -> List[str]:
        self._check_dir(path)
        p = os.path.join(self.dir, "rootfs", path.lstrip("/"))
        return sorted(os.listdir(p)) if os.path.isdir(p) else []
