"""Protected resident/upload/CPU expert cost comparison; no dual-device inference."""
import argparse
import importlib.util
import json
import math
import os
import pathlib
import re
import subprocess
import sys

HERE = pathlib.Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("resident_calibration_base", HERE / "run-native-max8-lab.py")
BASE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BASE)
MODE = "resident-cpu-gpu-expert-calibration"
MARKER = "public_resident_expert_calibration modes=resident,upload,cpu workers=16 backend=pooled-avx2-fma owned_cleanup=true dual_device_accepted=false"


def validate_receipt(receipt, card, output):
    expected = {
        "scope": "fixed-public-original-resident-expert-calibration",
        "resident_expert_calibration": True, "device_dispatch": "inline",
        "vendor": 0x8086 if card == "intel" else 0x10DE,
        "weights": "borrowed-unmodified-ollama-blob", "fixed_public_points": 6,
        "numeric_gates": 12, "timing_samples": 24, "ranks_per_case": 10,
        "group": 0, "vector": 0, "backend": "pooled-avx2-fma", "workers": 16,
        "owned_cleanup_accepted": True, "context_retired": False,
        "full_model_or_automatic_placement_accepted": False,
    }
    if any(type(receipt.get(k)) is not type(v) or receipt[k] != v for k, v in expected.items()):
        raise ValueError("Resident calibration identity, CPU path or complete cleanup rejected")
    if MARKER not in output.splitlines() or "verified_lab_limits swap_max=0 core_soft=0 core_hard=0 memory_max=8589934592" not in output.splitlines():
        raise ValueError("Protected resident calibration completion markers missing")
    points = receipt.get("points")
    if type(points) is not list or len(points) != 6:
        raise ValueError("All six original Q5/Q8 resident/upload/CPU points required")
    for index, point in enumerate(points):
        down = 6 if index < 3 else 8
        misses, cpu = [(0, 0), (10, 0), (10, 10)][index % 3]
        if point.get("down_dtype") != down or point.get("cpu_misses") != cpu or point.get("numeric_modes") != ["concurrent", "cpu-first"]:
            raise ValueError("Original resident calibration point ordering changed")
        numeric, samples = point.get("numeric"), point.get("samples")
        if type(numeric) is not list or len(numeric) != 2 or type(samples) is not list or len(samples) != 4:
            raise ValueError("Both numeric gates and all four ABBA samples required")
        for n, sample in enumerate(samples):
            if sample.get("mode") != ["concurrent", "cpu-first", "cpu-first", "concurrent"][n]:
                raise ValueError("Original ABBA order changed")
        for timed, report in [(False, r) for r in numeric] + [(True, s.get("report", {})) for s in samples]:
            counters = {
                "down_dtype": down, "workers": 16, "misses": misses,
                "cpu_misses": cpu, "gpu_misses": misses - cpu,
                "private_pages": 18, "private_bytes": 884736, "bank_bytes": 134217728,
                "source_precharge": (misses - cpu) * 4 * 1024**2,
                "zero_buffers": 18, "zero_bytes": 884736,
            }
            if any(type(report.get(k)) is not int or report[k] != v for k, v in counters.items()):
                raise ValueError("Actual resident/miss/source or full erasure evidence changed")
            for k, low, high in [("layer", 0, 47), ("io_inspected_bytes", 1, 4*1024**2), ("metadata_bytes", 1, 1024**2), ("snapshot_bytes", 1, 2*1024**2)]:
                if type(report.get(k)) is not int or not low <= report[k] <= high:
                    raise ValueError("Original bounded ownership evidence missing")
            for key in ("max_tree_error", "max_legacy_error", "max_core_error"):
                value = report.get(key)
                if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                    raise ValueError("Finite original numerical comparison missing")
            times = report.get("timing_ns")
            if not timed and times is not None:
                raise ValueError("Numeric gate must not report timing")
            if timed:
                if type(times) is not dict or any(type(times.get(k)) is not int or times[k] < 0 for k in ("cpu", "gpu", "joined", "bridge_and_merge")):
                    raise ValueError("Complete bounded timing fields required")
                if cpu == 0 and times["cpu"] != 0:
                    raise ValueError("Resident GPU point unexpectedly executes CPU")
        for sample in samples:
            times = sample["report"]["timing_ns"]
            keys = ("candidate_completed_wall_ns", "acceptance_cleanup_wall_ns", "full_run_including_references_wall_ns")
            if any(type(sample.get(k)) is not int or sample[k] < 0 for k in keys) or sample[keys[0]] < times["joined"] + times["bridge_and_merge"] or sample[keys[2]] < sample[keys[0]] + sample[keys[1]]:
                raise ValueError("Completed wall intervals cannot be enqueue-only timings")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cargo-artifacts", type=pathlib.Path, required=True)
    parser.add_argument("--model", type=pathlib.Path, required=True)
    parser.add_argument("--card", choices=("intel", "nvidia"), required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--receipt", type=pathlib.Path, required=True)
    args = parser.parse_args()
    if not re.fullmatch(r"[a-z0-9-]{1,48}", args.run_id):
        parser.error("run-id must be1..48 lowercase letters, digits or hyphens")
    target = args.receipt.resolve()
    record, log = target.with_suffix(".run.json"), target.with_suffix(".log")
    if any(p.exists() for p in (target, record, log)):
        parser.error("Refusing to overwrite previous evidence")
    frozen, executable_hash, build_hash = BASE.cargo_binary(args.cargo_artifacts)
    env = {
        "XDG_RUNTIME_DIR": "/run/user/" + str(os.getuid()), "CUBECL_DEBUG_LOG": "0",
        "KUNO_PUBLIC_RECEIPT": str(target), "KUNO_TEST_GGUF": str(args.model.resolve()),
        "KUNO_LAB_TRANSIENT_IO": "1", "KUNO_LAB_OWNED_ZERO_AUDIT": "1",
        "KUNO_LAB_RESIDENT_EXPERT_CALIBRATION": "1", "KUNO_LAB_DEVICE_DISPATCH": "inline",
    }
    # This is not a count1 model run. Explicitly exclude inherited model policies.
    excluded = set(k for k in os.environ if k.startswith("KUNO_LAB_") and k not in env)
    command = ["systemd-run", "--user", "--unit=kuno-user-" + args.run_id,
               "--collect", "--wait", "--pipe", "--property=MemorySwapMax=0",
               "--property=LimitCORE=0", "--property=MemoryMax=8G", "--property=RuntimeMaxSec=300"]
    if excluded:
        command.append("--property=UnsetEnvironment=" + " ".join(sorted(excluded)))
    command.extend("--setenv=" + k + "=" + v for k, v in env.items())
    command.extend([sys.executable, str(HERE / "protected-lab-exec.py"), str(frozen), args.card, MODE])
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    lines = []
    with log.open("x") as file:
        for line in process.stdout:
            file.write(line)
            file.flush()
            lines.append(line)
    if process.wait() != 0 or not target.is_file():
        raise SystemExit("Resident calibration failed; no acceptance recorded")
    validate_receipt(json.loads(target.read_text()), args.card, "".join(lines))
    record.write_text(json.dumps({
        "scope": "fixed-public resident expert operator calibration only",
        "card": args.card, "cargo_artifacts_sha256": build_hash,
        "frozen_executable_sha256": executable_hash,
        "receipt_sha256": BASE.digest(target.read_bytes()), "protected_output_sha256": BASE.digest(log.read_bytes()),
        "protected_lifecycle_completion_checked": True, "dual_device_inference_accepted": False,
    }, indent=2) + "\n")
    print("verified_resident_calibration_receipt=" + str(target))


if __name__ == "__main__":
    main()
