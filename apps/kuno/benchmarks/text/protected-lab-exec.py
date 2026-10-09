"""Verify actual no-swap/core and bounded host cgroup before lab execution."""
import os
import pathlib
import resource
import sys

def selected_target_policy(environment, mtp=False):
    selected = {key: value for key, value in environment.items() if key.startswith(("KUNO_LAB_", "KUNO_TEST_", "KUNO_PUBLIC_"))}
    if "KUNO_LAB_TARGET_DENSE" in selected:
        if selected.pop("KUNO_LAB_TARGET_DENSE") != "verify3":
            raise ValueError("Target dense policy requires canonical verify3")
    if "KUNO_LAB_TARGET_IO" in selected:
        if selected.pop("KUNO_LAB_TARGET_IO") != "chunk":
            raise ValueError("Target I/O policy requires canonical chunk")
    if "KUNO_LAB_TARGET_EXPERTS" in selected:
        if selected.pop("KUNO_LAB_TARGET_EXPERTS") != "prepared-verify3":
            raise ValueError("Target expert policy requires canonical prepared-verify3")
    if "KUNO_LAB_TARGET_FILL" in selected:
        if selected.pop("KUNO_LAB_TARGET_FILL") != "direct-public":
            raise ValueError("Target fill policy requires canonical direct-public")
    if mtp and "KUNO_LAB_MTP_SWEEPS" in selected:
        if selected.pop("KUNO_LAB_MTP_SWEEPS") != "fused":
            raise ValueError("MTP sweep policy requires canonical fused")
    return selected

def memory_limit(environment, arguments):
    if "KUNO_LAB_VERIFY3_DENSE_PUBLIC_ONLY" in environment:
        selected = {key: value for key, value in environment.items() if key.startswith(("KUNO_LAB_", "KUNO_TEST_", "KUNO_PUBLIC_"))}
        if (set(selected) != {"KUNO_LAB_VERIFY3_DENSE_PUBLIC_ONLY", "KUNO_PUBLIC_RECEIPT"}
                or any(not value for value in selected.values())
                or selected["KUNO_LAB_VERIFY3_DENSE_PUBLIC_ONLY"] != "1"
                or len(arguments) != 3 or arguments[1] not in ("intel", "nvidia") or arguments[2] != "verify3-dense-probe"):
            raise ValueError("Verify3 operator requires the fixed public fixture")
        return 8 * 1024**3
    if "KUNO_LAB_SPECULATIVE_PUBLIC_ONLY" in environment:
        permitted = {"KUNO_LAB_SPECULATIVE_PUBLIC_ONLY", "KUNO_LAB_STATIC_CACHE_BYTES", "KUNO_TEST_Q3_DIR", "KUNO_TEST_MTP", "KUNO_PUBLIC_TARGET_CARRY13_REFERENCE", "KUNO_PUBLIC_RECEIPT"}
        selected = selected_target_policy(environment, mtp=True)
        if (set(selected) != permitted or any(not value for value in selected.values())
                or selected["KUNO_LAB_SPECULATIVE_PUBLIC_ONLY"] != "1"
                or selected["KUNO_LAB_STATIC_CACHE_BYTES"] != "8589934592"
                or len(arguments) != 3 or arguments[1:] != ["intel", "daslab-speculative-probe"]):
            raise ValueError("Speculative cache requires the fixed public dual-GPU fixture")
        return 48 * 1024**3
    if "KUNO_LAB_RESIDENT_MTP_PUBLIC_ONLY" in environment or (len(arguments) > 2 and arguments[2] == "resident-mtp-probe"):
        selected = {key: value for key, value in environment.items() if key.startswith(("KUNO_LAB_", "KUNO_TEST_", "KUNO_PUBLIC_"))}
        if "KUNO_LAB_MTP_SWEEPS" in selected and selected.pop("KUNO_LAB_MTP_SWEEPS") != "fused":
            raise ValueError("MTP sweep policy requires canonical fused")
        permitted={"KUNO_LAB_RESIDENT_MTP_PUBLIC_ONLY", "KUNO_TEST_MTP", "KUNO_PUBLIC_DASLAB_CARRY", "KUNO_PUBLIC_MTP_REFERENCE", "KUNO_PUBLIC_RECEIPT"}
        if (set(selected) != permitted or any(not value for value in selected.values())
                or selected["KUNO_LAB_RESIDENT_MTP_PUBLIC_ONLY"] != "1"
                or len(arguments) != 3 or arguments[1] not in ("intel", "nvidia") or arguments[2] != "resident-mtp-probe"):
            raise ValueError("Resident MTP requires fixed public artifacts")
        return 8 * 1024**3
    target_modes = {"KUNO_LAB_TARGET_ROWS_PUBLIC_ONLY": "daslab-target-rows-probe", "KUNO_LAB_TARGET_REPAIR_PUBLIC_ONLY": "daslab-target-repair-probe"}
    target_gate = next((key for key in target_modes if key in environment), None)
    if target_gate is not None:
        permitted = {target_gate, "KUNO_LAB_STATIC_CACHE_BYTES", "KUNO_TEST_Q3_DIR", "KUNO_PUBLIC_TARGET_ROWS_REFERENCE", "KUNO_PUBLIC_RECEIPT"}
        selected = selected_target_policy(environment)
        if (set(selected) != permitted or any(not value for value in selected.values())
                or selected[target_gate] != "1"
                or selected["KUNO_LAB_STATIC_CACHE_BYTES"] != "8589934592"
                or len(arguments) != 3 or arguments[1:] != ["intel", target_modes[target_gate]]):
            raise ValueError("Larger target cache requires the fixed public DASLab target fixture")
        # Original mapped model pages and the separately bounded 8GiB cache;
        # this does not enlarge any request-derived owning allocation.
        return 48 * 1024**3
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
