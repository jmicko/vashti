"""Protected fixed-public actual-model lifecycle fixture, never private requests."""

import argparse
import importlib.util
import json
import math
import os
import pathlib
import re
import subprocess
import sys


_spec = importlib.util.spec_from_file_location(
    "native_max8_lab", pathlib.Path(__file__).with_name("run-native-max8-lab.py")
)
_native = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_native)


def validate_receipt(receipt, card, output):
    expected = {
        "fixture": "actual-native-model-lifecycle",
        "scope": "fixed-public whole-model lifecycle; numeric mismatch needs parent review",
        "vendor": 0x8086 if card == "intel" else 0x10DE,
        "weights": "borrowed-unmodified-ollama-blob",
        "fixed_public_input": "original tokenizer ! repeated4/8",
        "seed": 424242,
        "performance_claim": False,
        "lifecycle_accepted": True,
        "owned_io_cleanup_accepted": True,
        "context_retired": False,
        "extra_main_bytes": 0,
        "native_main_bytes": 818249728,
        "native_main_pages": 1376,
        "scalar_main_bytes": 785186816,
        "scalar_main_pages": 1286,
        "probe_host_cap": 4194304,
        "probe_metadata_cap": 65536,
        "model_host_cap": 12582912,
        "independent_runtime_cap": 67108864,
        "live_source_cap": 67108864,
        "source_audits_cumulative": True,
    }
    if any(type(receipt.get(key)) is not type(value) or receipt[key] != value
           for key, value in expected.items()):
        raise ValueError("Actual-model fixture identity, ownership or cleanup did not match")
    cases = receipt.get("cases")
    ids = [0, 2, 5, 1, 3, 4, 6, 7, 8, 11, 9, 12, 10, 13, 14, 18, 15, 19, 16, 20, 17, 21]
    if (not isinstance(cases, list) or len(cases) != len(ids)
        or any(not isinstance(case, dict) or type(case.get("case")) is not int
               or case["case"] != identity for case, identity in zip(cases, ids))):
        raise ValueError("All22 actual owner cases in the original order are required")
    terminal = {0, 1, 2, 3, 4, 8, 9, 10, 14, 15, 16, 17}
    numeric = {2, 3, 4, 5, 6, 11, 12, 13, 18, 19, 20, 21}
    bit_equal = greedy_equal = True
    for case in cases:
        identity = case["case"]
        context = 8 if identity in {1, 3, 4, 6} else 4
        for key, value in {"context": context, "terminal_cleanup": identity in terminal,
                           "hook_observed": identity in {14, 15, 16, 17}}.items():
            if type(case.get(key)) is not type(value) or case[key] != value:
                raise ValueError("Actual context, terminal cleanup or hook evidence missing")
        audit = case.get("audit")
        scalar = identity in {0, 1, 7}
        pages, size = (1286, 785186816) if scalar else (1376, 818249728)
        if (not isinstance(audit, dict)
            or any(type(audit.get(key)) is not int for key in
                   ("main_buffers", "main_bytes", "source_buffers", "source_bytes"))
            or audit["main_buffers"] != pages or audit["main_bytes"] != size):
            raise ValueError("Each actual owner needs its complete MAIN audit")
        count, source = audit["source_buffers"], audit["source_bytes"]
        if scalar:
            if count <= 0 or source < count * 32768 or source > context * 67108864:
                raise ValueError("Original scalar cumulative source audit rejected")
        elif count != 0 or source != 0:
            raise ValueError("Native sources must retain their separate checked runtime ownership")
        stats = case.get("numeric")
        if identity not in numeric:
            if stats is not None:
                raise ValueError("Unexpected numerical comparison on negative/reference case")
            continue
        if not isinstance(stats, dict):
            raise ValueError("Complete full-vocabulary numerical statistics are required")
        for key in ("bit_mismatches", "diagnostic_outside"):
            if type(stats.get(key)) is not int or not 0 <= stats[key] <= 248320:
                raise ValueError("Full-vocabulary numerical count rejected")
        for key in ("max_absolute", "max_relative"):
            if type(stats.get(key)) not in (int, float) or not math.isfinite(stats[key]) or stats[key] < 0:
                raise ValueError("Finite full-vocabulary numerical statistics are required")
        if (type(stats.get("greedy_equal")) is not bool
            or type(stats.get("bit_equal")) is not bool
            or stats["bit_equal"] != (stats["bit_mismatches"] == 0)
            or stats.get("diagnostic_bound_only") != "1e-4+3e-4*abs(reference)"):
            raise ValueError("Numerical acceptance cannot silently relax bitwise equality")
        if stats["bit_equal"] and any(stats[key] != 0 for key in
                                      ("max_absolute", "max_relative", "diagnostic_outside")):
            raise ValueError("Bitwise equality requires zero numerical differences")
        bit_equal &= stats["bit_equal"]
        greedy_equal &= stats["greedy_equal"]
    for key, value in {
        "all_logits_bit_equal": bit_equal, "all_greedy_equal": greedy_equal,
        "whole_model_numeric_accepted": bit_equal and greedy_equal,
        "numeric_not_accepted": not (bit_equal and greedy_equal),
    }.items():
        if type(receipt.get(key)) is not bool or receipt[key] != value:
            raise ValueError("Aggregate numerical acceptance did not match all actual cases")
    for message in (
        "verified_lab_limits swap_max=0 core_soft=0 core_hard=0 memory_max=8589934592",
        "public_actual_model_lifecycle_complete cases=22 owned_cleanup=true context_retired=false performance_claim=false",
    ):
        if message not in output:
            raise ValueError("Protected completion evidence is missing")
    return bit_equal and greedy_equal


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
    metadata = target.with_suffix(target.suffix + ".run.json")
    log = target.with_suffix(target.suffix + ".log")
    if any(path.exists() for path in (target, metadata, log)):
        parser.error("Refusing to replace a prior receipt, run record or log")
    frozen, binary_hash, build_hash = _native.cargo_binary(args.cargo_artifacts)
    env = {
        "XDG_RUNTIME_DIR": "/run/user/" + str(os.getuid()),
        "CUBECL_DEBUG_LOG": "0", "KUNO_PUBLIC_RECEIPT": str(target),
        "KUNO_TEST_GGUF": str(args.model.resolve()),
        "KUNO_LAB_TRANSIENT_IO": "1", "KUNO_LAB_OWNED_ZERO_AUDIT": "1",
    }
    for suffix in ("BLOCK_MOE", "BATCH_EXPERTS", "WORD_DOWN", "ORDERED_PUBLIC_UPLOAD",
                   "FUSED_MATRIX_FAULT", "SMALL_CPU", "CPU_GDN", "PREFILL_LAST_HEAD",
                   "IO_TOKEN_CHECKPOINT", "INFO_SLAB_BYTES", "NATIVE_MAX8", "NATIVE_DECODE"):
        env["KUNO_LAB_" + suffix] = "0"
    command = [
        "systemd-run", "--user", "--unit=kuno-user-" + args.run_id,
        "--collect", "--wait", "--pipe", "--property=MemorySwapMax=0",
        "--property=LimitCORE=0", "--property=MemoryMax=8G",
        "--property=RuntimeMaxSec=1800",
        "--property=UnsetEnvironment=KUNO_LAB_BLOCK_KERNEL KUNO_LAB_EXPERT_SLOTS "
        "KUNO_LAB_STATIC_CACHE_BYTES KUNO_LAB_EXPERT_BATCH_BYTES KUNO_LAB_EXPERT_SCATTER "
        "KUNO_LAB_NATIVE_BATCH_COUNT",
    ]
    command.extend("--setenv=" + key + "=" + value for key, value in env.items())
    command.extend([sys.executable, str(pathlib.Path(__file__).with_name("protected-lab-exec.py")),
                    str(frozen), args.card, "resident-lifecycle-probe"])
    print("verified_cargo_executable_sha256=" + binary_hash, flush=True)
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    lines = []
    with log.open("x") as output:
        for line in process.stdout:
            print(line, end="", flush=True)
            output.write(line)
            output.flush()
            lines.append(line)
    if process.wait() != 0 or not target.is_file():
        raise SystemExit("Actual-model fixture failed; no acceptance recorded")
    snapshot = target.read_bytes()
    numerical = validate_receipt(json.loads(snapshot), args.card, "".join(lines))
    with metadata.open("x") as output:
        json.dump({
            "scope": "fixed-public actual-model lifecycle and full-vocabulary numeric fixture only",
            "card": args.card, "cargo_artifacts_sha256": build_hash,
            "frozen_executable_sha256": binary_hash, "receipt_sha256": _native.digest(snapshot),
            "protected_output_sha256": _native.digest(log.read_bytes()),
            "protected_lifecycle_completion_checked": True,
            "whole_model_numeric_accepted": numerical, "matched_performance_measurement": False,
        }, output, indent=2, sort_keys=True)
        output.write("\n")
    if not numerical:
        raise SystemExit("Lifecycle completed; full-logit numerical acceptance failed; original report retained")
    print("verified_actual_model_lifecycle_and_numeric_receipt=" + str(target), flush=True)


if __name__ == "__main__":
    main()
