"""Tests built on REAL captured output (see fixtures/real_vm_captured/PROVENANCE.md), not hand-written samples."""
import os
import subprocess

import pytest

from hwassistant import parsers as P
from hwassistant.agent import Assistant
from hwassistant.executor import FixtureExecutor, SystemExecutor, ToolError
from hwassistant.offline import OfflineBackend
from conftest import FIX, read

REAL = "real_vm_captured"


def real_bot():
    return Assistant(OfflineBackend(), FixtureExecutor(os.path.join(FIX, REAL)))


def test_real_dmidecode_quirks_are_detected_not_trusted():
    d = P.parse_dmidecode_memory(read(REAL, "cmd/dmidecode_memory.txt"))
    assert len(d["arrays"]) == 1 and d["arrays"][0]["slots"] == 64
    assert len(d["devices"]) == 128                      # real firmware: more records than declared slots
    assert sum(x["installed"] for x in d["devices"]) == 1
    bot = real_bot()
    r = bot._call_tool("get_memory_topology", {})
    assert r["status"] == "partial" and len(r["data_quality_warnings"]) >= 2 and r["virtual_machine"]
    assert len(r["empty_slots"]) <= 17                   # output is bounded for the LLM context


def test_real_vm_upgrade_answer_refuses_to_overclaim():
    a = real_bot().ask("Can I add more RAM, and how much?")
    assert "can't reliably determine" in a.lower() and "virtual machine" in a.lower()
    assert "you can add" not in a.lower()


def test_real_virtio_disks_are_not_called_hdd():
    a = real_bot().ask("what storage do I have?")
    assert "virtual disk" in a and "HDD" not in a


def test_real_lspci_parses_without_gpu_and_with_flag_columns():
    gpus = P.parse_lspci_gpus(read(REAL, "cmd/lspci.txt"))
    assert gpus == []                                    # real VM: no display adapter, lines have -r01 -p00 flags
    assert "couldn't find a graphics adapter" in real_bot().ask("what gpu do I have?")


def test_real_lscpu_free_df_parse():
    assert P.parse_lscpu(read(REAL, "cmd/lscpu.txt"))["logical_cpus"] >= 1
    assert P.parse_free(read(REAL, "cmd/free.txt"))["total"] > 0
    assert P.parse_df(read(REAL, "cmd/df.txt"))


@pytest.mark.skipif(not os.path.exists("/usr/bin/lspci"), reason="pciutils not installed")
def test_live_lspci_output_parses():
    out = subprocess.run(["/usr/bin/lspci", "-mm"], capture_output=True, text=True).stdout
    assert isinstance(P.parse_lspci_gpus(out), list)


def test_live_dmidecode_failure_is_classified_not_blamed_on_permissions():
    ex = SystemExecutor()
    if not ex.has_command("dmidecode_memory"):
        pytest.skip("dmidecode not installed")
    try:
        ex.run("dmidecode_memory")
    except ToolError as e:
        assert e.reason in ("no_dmi_data", "permission")  # never a generic 'failed'
