"""Protected fixed-public whole-model CPU-miss acceptance, not production inference.

Requires externally coordinated exclusive GPU/admission access. Does not obtain
sudo, stop services, reset hardware or accept arbitrary prompts. Device output
and timings establish nothing until every output/lifecycle/owning audit passes.
"""
import argparse
import importlib.util
import json
import pathlib
import re
import subprocess
import sys

HERE = pathlib.Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("native_max8_lab", HERE / "run-native-max8-lab.py")
BASE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BASE)
MODE = "resident-count1-hybrid-chat"
GUARD = "verified_lab_limits swap_max=0 core_soft=0 core_hard=0 memory_max=8589934592"


def validate_cap(cap):
    if type(cap) is not int or cap not in (0, 10):
        raise ValueError("CPU miss cap must be independently selected as integer 0 or 10")


def completion(cap):
    validate_cap(cap)
    return (
        f"public_count1_hybrid_fresh_continuation cap={cap} matches=true",
        f"public_count1_hybrid_mid_token_cancel cap={cap} layer=0 erased=true",
        f"public_count1_hybrid_mid_token_cancel cap={cap} layer=3 erased=true",
    )


def validate_receipt(receipt, card, cap, output, workers="scoped", overlap=False, mapped=False, profile=False, fast=False, long=False, cpu_q8=False, large_cache=False, compact_info=False, batch_tasks=False):
    validate_cap(cap)
    for field, selected in (("mapped_weights", mapped), ("observed_mapped_weights", mapped), ("kernel_profile", profile), ("fast_gpu", fast), ("long_fixture", long), ("cpu_q8_activations", cpu_q8), ("observed_cpu_q8_activations", cpu_q8), ("large_weight_cache", large_cache), ("compact_info", compact_info), ("batch_tasks", batch_tasks)):
        if type(selected) is not bool or (selected or field in receipt) and receipt.get(field) is not selected:
            raise ValueError("Explicit optimization selection and completed receipt must match")
    for field, expected, selected in (("info_slab_bytes",4096 if compact_info else 0,compact_info),("runtime_tasks_max",256 if batch_tasks else 32,batch_tasks)):
        if (selected or field in receipt) and (type(receipt.get(field)) is not int or receipt[field]!=expected):
            raise ValueError("Completed bounded runtime configuration must match explicit selection")
    calls = receipt.get("observed_fast_dense_calls", 0)
    if type(calls) is not int or not 0 <= calls <= 100000 or (calls > 0) is not fast:
        raise ValueError("Actual count1 fast dense dispatch evidence must match explicit selection")
    if workers not in ("scoped", "pooled"):
        raise ValueError("Unknown CPU worker mode")
    if cpu_q8 and workers != "pooled":
        raise ValueError("Q8 CPU activation arithmetic requires pooled workers")
    for field in ("count1_hybrid_cpu_workers", "observed_count1_hybrid_cpu_workers"):
        if receipt.get(field) != workers:
            raise ValueError("Requested and completed CPU workers must match explicit selection")
    if large_cache and not mapped:
        raise ValueError("Larger public cache requires mapped weights")
    guard = GUARD.replace("8589934592", "51539607552") if large_cache else GUARD
    if guard not in output.splitlines():
        raise ValueError("Protected execution limits missing")
    for field in ("count1_hybrid_cpu_miss_cap", "observed_count1_hybrid_cpu_miss_cap"):
        if type(receipt.get(field)) is not int or receipt[field] != cap:
            raise ValueError("Requested and completed hybrid cap mismatch")
    if type(overlap) is not bool or (overlap and workers != "pooled"):
        raise ValueError("Overlap requires explicit pooled workers")
    for field in ("count1_hybrid_cpu_gpu_overlap", "observed_count1_hybrid_cpu_gpu_overlap"):
        if receipt.get(field) is not overlap:
            raise ValueError("Requested and completed overlap policy mismatch")
    for field, limit in (("completed_cpu_misses", (127 if long else 7) * 48 * 10),
                         ("fresh_completed_cpu_misses", 48 * 10)):
        value = receipt.get(field)
        if type(value) is not int or not 0 <= value <= limit:
            raise ValueError("Completed CPU work outside accepted fixture bounds")
        if cap == 0 and value != 0 or cap == 10 and value == 0:
            raise ValueError("Actual completed CPU-miss evidence missing or incompatible")
    if any(type(receipt.get(field)) is not int for field in (
        "resident_private_owned_bytes", "prompt_tokens", "seed", "temperature", "native_public_chunk_count"
    )) or not isinstance(receipt.get("public_generated_ids"), list) or any(
        type(token) is not int for token in receipt["public_generated_ids"]
    ):
        raise ValueError("Hybrid fixture and inventory must use integer counts")
    audits = receipt.get("owned_zero_audits")
    if not isinstance(audits, list) or len(audits) != 4 or any(
        not isinstance(audit, dict) or any(type(audit.get(field)) is not int
            for field in ("main_buffers", "main_bytes", "source_buffers", "source_bytes"))
        for audit in audits
    ):
        raise ValueError("Four typed complete hybrid owner audits required")
    BASE._validate_model_receipt(
        receipt, card, "resident-rainwater" if long else "resident-chat", 8, output,
        owner_inventory=(1378, 818413568), reported_fixture="resident-count1-hybrid-rainwater" if long else MODE,
        native_decode=True, native_decode_tile=1, native_decode_kernels="single-q4-all",
    )
    if any(marker not in output.splitlines() for marker in completion(cap)):
        raise ValueError("Hybrid continuation and both cancellation proofs required")


def worker_environment(model, receipt, workers="scoped", overlap=False, mapped=False, profile=False, fast=False, long=False, cpu_q8=False, large_cache=False, compact_info=False, batch_tasks=False):
    if workers not in ("scoped", "pooled"):
        raise ValueError("Unknown CPU worker mode")
    if type(overlap) is not bool or (overlap and workers != "pooled"):
        raise ValueError("Overlap requires explicit pooled workers")
    if any(type(value) is not bool for value in (mapped,profile,fast,long,cpu_q8,large_cache,compact_info,batch_tasks)):
        raise ValueError("Explicit boolean optimization selections required")
    if cpu_q8 and workers != "pooled":
        raise ValueError("Q8 CPU activation arithmetic requires pooled workers")
    if large_cache and not mapped:
        raise ValueError("Larger public cache requires mapped weights")
    env = BASE.worker_environment(model, receipt, 8, True, 1, "single-q4-all")
    env["KUNO_LAB_CPU_WORKERS"] = workers
    env["KUNO_LAB_CPU_GPU_OVERLAP"] = "1" if overlap else "0"
    env["KUNO_LAB_MAPPED_WEIGHTS"] = "1" if mapped else "0"
    env["KUNO_LAB_KERNEL_PROFILE"] = "1" if profile else "0"
    env["KUNO_LAB_FAST_GPU"] = "1" if fast else "0"
    env["KUNO_LAB_LONG_FIXTURE"] = "1" if long else "0"
    env["KUNO_LAB_CPU_Q8"] = "1" if cpu_q8 else "0"
    env["KUNO_LAB_LARGE_WEIGHT_CACHE"] = "1" if large_cache else "0"
    env["KUNO_LAB_COMPACT_INFO"] = "1" if compact_info else "0"
    env["KUNO_LAB_INFO_SLAB_BYTES"] = "4096" if compact_info else "0"
    env["KUNO_LAB_BATCH_TASKS"] = "1" if batch_tasks else "0"
    return env


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cargo-artifacts", type=pathlib.Path, required=True)
    parser.add_argument("--model", type=pathlib.Path, required=True)
    parser.add_argument("--card", choices=("intel", "nvidia"), required=True)
    parser.add_argument("--cpu-miss-cap", type=int, choices=(0, 10), required=True)
    parser.add_argument("--cpu-workers", choices=("scoped", "pooled"), default="scoped")
    parser.add_argument("--cpu-gpu-overlap", action="store_true")
    parser.add_argument("--mapped-weights", action="store_true")
    parser.add_argument("--kernel-profile", action="store_true")
    parser.add_argument("--fast-gpu", action="store_true")
    parser.add_argument("--long-fixture", action="store_true")
    parser.add_argument("--cpu-q8", action="store_true")
    parser.add_argument("--large-weight-cache", action="store_true")
    parser.add_argument("--compact-info", action="store_true")
    parser.add_argument("--batch-tasks", action="store_true")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--receipt", type=pathlib.Path, required=True)
    args = parser.parse_args()
    if args.cpu_gpu_overlap and args.cpu_workers != "pooled":
        parser.error("CPU/GPU overlap requires --cpu-workers pooled")
    if args.cpu_q8 and args.cpu_workers != "pooled":
        parser.error("Q8 CPU activation arithmetic requires --cpu-workers pooled")
    if args.large_weight_cache and not args.mapped_weights:
        parser.error("Larger public cache requires --mapped-weights")
    if not re.fullmatch(r"[a-z0-9-]{1,48}", args.run_id):
        parser.error("run-id must be 1..48 lowercase letters, digits or hyphens")
    target = args.receipt.resolve()
    metadata = target.with_suffix(target.suffix + ".run.json")
    if target.exists() or metadata.exists():
        parser.error("Refusing to replace an existing receipt or run record")
    frozen, identity, artifacts_identity = BASE.cargo_binary(args.cargo_artifacts)
    command = [
        "systemd-run", "--user", "--unit=kuno-user-" + args.run_id,
        "--collect", "--wait", "--pipe", "--property=MemorySwapMax=0",
        "--property=LimitCORE=0", "--property=MemoryMax=" + ("48G" if args.large_weight_cache else "8G"),
        "--property=RuntimeMaxSec=900",
        "--property=UnsetEnvironment=KUNO_LAB_BLOCK_KERNEL KUNO_LAB_EXPERT_SLOTS "
        "KUNO_LAB_STATIC_CACHE_BYTES KUNO_LAB_EXPERT_BATCH_BYTES KUNO_LAB_EXPERT_SCATTER",
    ]
    command.extend("--setenv=" + key + "=" + value
                   for key, value in worker_environment(args.model.resolve(), target, args.cpu_workers, args.cpu_gpu_overlap, args.mapped_weights, args.kernel_profile, args.fast_gpu, args.long_fixture, args.cpu_q8, args.large_weight_cache, args.compact_info, args.batch_tasks).items())
    command.extend([sys.executable, str(HERE / "protected-lab-exec.py"), str(frozen),
                    args.card, MODE, str(args.cpu_miss_cap)])
    print("verified_cargo_executable_sha256=" + identity, flush=True)
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    lines = []
    for line in process.stdout:
        print(line, end="", flush=True)
        lines.append(line)
    if process.wait() != 0:
        raise SystemExit("Protected hybrid fixture failed; no acceptance recorded")
    if not target.is_file():
        raise SystemExit("Missing whole-model hybrid receipt")
    snapshot = target.read_bytes()
    receipt = json.loads(snapshot)
    validate_receipt(receipt, args.card, args.cpu_miss_cap, "".join(lines), args.cpu_workers, args.cpu_gpu_overlap, args.mapped_weights, args.kernel_profile, args.fast_gpu, args.long_fixture, args.cpu_q8, args.large_weight_cache, args.compact_info, args.batch_tasks)
    record = {
        "scope": "fixed-public-count1-hybrid-output-and-lifecycle-only",
        "cargo_artifacts_sha256": artifacts_identity, "frozen_executable_sha256": identity,
        "card": args.card, "fixture": MODE, "cpu_miss_cap": args.cpu_miss_cap,
        "receipt_sha256": BASE.digest(snapshot), "protected_lifecycle_completion_checked": True,
        "matched_performance_measurement": False, "cpu_gpu_overlap": args.cpu_gpu_overlap, "cpu_workers": args.cpu_workers,
        "mapped_weights": args.mapped_weights, "kernel_profile": args.kernel_profile,
        "fast_gpu": args.fast_gpu,
        "long_fixture": args.long_fixture,
        "cpu_q8_activations": args.cpu_q8,
        "large_weight_cache": args.large_weight_cache,
        "compact_info": args.compact_info,
        "batch_tasks": args.batch_tasks,
    }
    with metadata.open("x") as output:
        json.dump(record, output, indent=2, sort_keys=True)
        output.write("\n")
    print("verified_native_count1_hybrid_receipt=true", flush=True)


if __name__ == "__main__":
    main()
