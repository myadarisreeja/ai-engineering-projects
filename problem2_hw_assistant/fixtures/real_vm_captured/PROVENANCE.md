Captured from REAL systems (not hand-written):
- lscpu, lsblk, free, df, lspci -mm, uname, os-release: captured from a real Linux VM (Ubuntu 24.04, virtio disks).
- dmidecode_memory.txt: memory sections (DMI types 16/17) of a real CentOS 7.7 VMware guest dump from the
  open-source `jc` project's test fixtures (github.com/kellyjonbrazil/jc, MIT). Real firmware quirks included:
  128 memory-device records for a 64-slot array, devices not linked to the array handle.
- sys_vendor / product_name: set to the VMware strings matching that dump.
