"""Verify actual no-swap/core and bounded host cgroup before lab execution."""
import os
import pathlib
import resource
import sys

def memory_limit(environment, arguments):
    value = environment.get("KUNO_LAB_LARGE_WEIGHT_CACHE", "0")
    if value == "0":
        return 8 * 1024**3
    if value != "1" or environment.get("KUNO_LAB_MAPPED_WEIGHTS") != "1" or len(arguments) != 4 or arguments[1] not in ("intel", "nvidia") or arguments[2] != "resident-count1-hybrid-chat" or arguments[3] not in tuple(str(cap) for cap in range(11)):
        raise ValueError("Larger public cache requires the fixed mapped hybrid fixture")
    # Public model page cache, not a private-buffer allowance. The native MAIN
    # and CPU owning ledgers are unchanged. Bound the whole process to48GiB.
    return 48 * 1024**3

def main():
    try:
        expected = memory_limit(os.environ, sys.argv[1:])
    except ValueError as error:
        raise SystemExit(str(error))
    cgroup = next(x.split(":", 2)[2].strip() for x in pathlib.Path("/proc/self/cgroup").read_text().splitlines() if x.startswith("0::"))
    path = pathlib.Path("/sys/fs/cgroup") / cgroup.lstrip("/")
    if (path / "memory.swap.max").read_text().strip() != "0" or resource.getrlimit(resource.RLIMIT_CORE) != (0, 0):
        raise SystemExit("Protected laboratory cgroup limits missing")
    if (path / "memory.max").read_text().strip() != str(expected):
        raise SystemExit("Protected laboratory memory limit mismatch")
    print(f"verified_lab_limits swap_max=0 core_soft=0 core_hard=0 memory_max={expected}", flush=True)
    os.execv(sys.argv[1], sys.argv[1:])

if __name__ == "__main__":
    main()
