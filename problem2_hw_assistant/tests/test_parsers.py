from conftest import read
from hwassistant import parsers as P


def test_parse_size_and_human():
    assert P.parse_size("64 GB") == 64 << 30
    assert P.parse_size("16384 MB") == 16 << 30
    assert P.parse_size("No Module Installed") is None
    assert P.parse_size("Unknown") is None
    assert P.human(16 << 30) == "16 GB" and P.human(int(476.9 * (1 << 30))) == "476.9 GB"


def test_lscpu():
    d = P.parse_lscpu(read("laptop_16gb", "cmd/lscpu.txt"))
    assert d["model_name"].startswith("12th Gen Intel") and d["logical_cpus"] == 12 and d["physical_cores"] == 10


def test_free_and_meminfo():
    assert P.parse_free(read("laptop_16gb", "cmd/free.txt"))["total"] == 16556871680
    assert P.parse_meminfo("MemTotal:  1024 kB\nMemAvailable:  512 kB\n")["total"] == 1024 * 1024


def test_lsblk_kinds_and_filters_loop():
    disks = {d["name"]: d for d in P.parse_lsblk(read("laptop_16gb", "cmd/lsblk.txt"))}
    assert set(disks) == {"nvme0n1", "sda"}
    assert disks["nvme0n1"]["kind"] == "nvme_ssd" and disks["sda"]["kind"] == "hdd"
    assert len(disks["nvme0n1"]["partitions"]) == 2


def test_lsblk_string_rota_and_sata_ssd():
    txt = '{"blockdevices":[{"name":"sdb","size":"1000","type":"disk","rota":"0","tran":"sata","model":"X"}]}'
    assert P.parse_lsblk(txt)[0]["kind"] == "sata_or_other_ssd"


def test_dmidecode_memory():
    d = P.parse_dmidecode_memory(read("laptop_16gb", "cmd/dmidecode_memory.txt"))
    assert d["arrays"][0]["max_capacity_bytes"] == 64 << 30 and d["arrays"][0]["slots"] == 2
    assert [x["installed"] for x in d["devices"]] == [True, False]
    assert d["devices"][0]["type"] == "DDR4" and d["devices"][0]["speed"] == "3200 MT/s"
    assert not d["devices"][0]["soldered"]


def test_dmidecode_soldered_detection():
    txt = "Physical Memory Array\n\tMaximum Capacity: 16 GB\n\tNumber Of Devices: 1\n\nMemory Device\n\tSize: 16 GB\n\tForm Factor: Row Of Chips\n\tLocator: Onboard\n\tType: LPDDR4\n"
    assert P.parse_dmidecode_memory(txt)["devices"][0]["soldered"] is True


def test_lspci_gpu_and_nvidia():
    g = P.parse_lspci_gpus(read("laptop_16gb", "cmd/lspci.txt"))
    assert len(g) == 1 and "Iris Xe" in g[0]["device"]
    n = P.parse_nvidia_smi("NVIDIA GeForce RTX 3060, 12288, 550.54.14")
    assert n[0]["vram_human"] == "12 GB"


def test_df():
    rows = P.parse_df(read("laptop_16gb", "cmd/df.txt"))
    assert rows[0]["mountpoint"] == "/" and 41 < rows[0]["used_percent"] < 43


def test_garbage_input_does_not_crash():
    assert P.parse_lscpu("garbage")["model_name"] is None
    assert P.parse_dmidecode_memory("nonsense") == {"arrays": [], "devices": []}
    assert P.parse_lspci_gpus('unterminated "quote') == []
