"""Run one compiled public fixture; verify the actual Cargo bin and model receipt.

Laboratory only: no arbitrary prompt input or HTTP request capture. Numeric
receipts from private inference must never be supplied to this runner.
"""

import argparse
import hashlib
import json
import os
import pathlib
import re
import subprocess
import sys
import tomllib


TILE_ONE_COMPLETION = (
    "public_native_tile1_fresh_continuation_matches=true",
    "public_native_tile1_mid_token_cancel layer=0 erased=true",
    "public_native_tile1_mid_token_cancel layer=3 erased=true",
)
KERNEL_PROFILES = ("chunk", "single-moe", "single-gdn", "single-all", "single-q4-all")


def kernel_completion(profile):
    return (
        f"public_native_kernel_fresh_continuation profile={profile} matches=true",
        f"public_native_kernel_mid_token_cancel profile={profile} layer=0 erased=true",
        f"public_native_kernel_mid_token_cancel profile={profile} layer=3 erased=true",
    ) if profile != "chunk" else ()


def validate_kernel_selection(profile, native_decode, native_decode_tile):
    if (type(profile) is not str or profile not in KERNEL_PROFILES
        or type(native_decode) is not bool or type(native_decode_tile) is not int
        or native_decode_tile not in (1, 8)
        or native_decode_tile == 1 and not native_decode
        or profile != "chunk" and (not native_decode or native_decode_tile != 1)):
        raise ValueError("Decode kernel profile requires an explicit protected native tile1 selection")


def validate_kernel_profile(receipt, native_decode, native_decode_tile, profile, allow_legacy_decode):
    validate_kernel_selection(profile, native_decode, native_decode_tile)
    fields = ("native_decode_kernels", "observed_native_decode_kernels")
    present = [field in receipt for field in fields]
    if not any(present) and allow_legacy_decode and profile == "chunk":
        # Explicit offline historical inspection only. Live new-binary runners
        # never accept missing profile observations or any non-chunk fallback.
        return
    if not all(present) or any(type(receipt[field]) is not str or receipt[field] != profile
                               for field in fields):
        raise ValueError("Selected and observed decode kernel profiles did not match the caller")


def digest(data):
    return hashlib.sha256(data).hexdigest()


def cargo_binary(path):
    transcript = path.read_bytes()
    records = [json.loads(line) for line in transcript.splitlines()]
    if not records or records[-1] != {"reason": "build-finished", "success": True} or sum(
        record.get("reason") == "build-finished" for record in records
    ) != 1:
        raise ValueError("One completed successful Cargo build is required")
    matches = [
        record
        for record in records
        if record.get("reason") == "compiler-artifact"
        and record.get("target", {}).get("name") == "kuno-text-lab"
        and record.get("executable")
    ]
    if len(matches) != 1:
        raise ValueError("Exactly one Cargo kuno-text-lab executable is required")
    artifact = matches[0]
    manifest = pathlib.Path(__file__).resolve().parents[1] / "native-text/Cargo.toml"
    package = tomllib.loads(manifest.read_text())["package"]
    expected_package = "path+" + manifest.parent.as_uri() + "#" + package["name"] + "@" + package["version"]
    if (artifact.get("manifest_path") != str(manifest)
        or artifact.get("package_id") != expected_package
        or artifact.get("target", {}).get("kind") != ["bin"]
        or artifact.get("target", {}).get("crate_types") != ["bin"]
        or artifact.get("target", {}).get("src_path") != str(manifest.parent / "src/main.rs")
        or artifact.get("profile", {}).get("test") is not False
        or artifact.get("profile", {}).get("opt_level") != "3"
        or artifact.get("profile", {}).get("debug_assertions") is not False):
        raise ValueError("Cargo artifact must be this manifest's release laboratory binary")
    source = pathlib.Path(artifact["executable"])
    payload = source.read_bytes()
    identity = digest(payload)
    frozen = pathlib.Path("/tmp") / ("kuno-native-fixed-lab-" + identity)
    try:
        with frozen.open("xb") as output:
            output.write(payload)
        frozen.chmod(0o700)
    except FileExistsError:
        if digest(frozen.read_bytes()) != identity:
            raise ValueError("Existing frozen executable identity mismatch") from None
    return frozen, identity, digest(transcript)


def decode_counts(receipt, steps, native_decode, allow_legacy_decode):
    if type(native_decode) is not bool or type(allow_legacy_decode) is not bool:
        raise ValueError("Decode validation settings must be explicit booleans")
    fields = ("native_decode", "native_decode_steps", "scalar_decode_steps")
    present = [field in receipt for field in fields]
    if not any(present):
        # Offline inspection of archived pre-flag receipts only. Live runners
        # never select this compatibility path, and it cannot accept native decode.
        if allow_legacy_decode and not native_decode:
            return 0, steps
        raise ValueError("Successful generation decode counts are required")
    if not all(present):
        raise ValueError("Decode mode and both observed counts are required together")
    native = receipt["native_decode_steps"]
    scalar = receipt["scalar_decode_steps"]
    if (type(receipt["native_decode"]) is not bool or receipt["native_decode"] is not native_decode
        or type(native) is not int or type(scalar) is not int
        or native != (steps if native_decode else 0)
        or scalar != (0 if native_decode else steps)):
        raise ValueError("Observed decode mode/counts did not match the selected public experiment")
    return native, scalar


def validate_decode_tile(receipt, native_decode, native_decode_tile, allow_legacy_decode):
    if (type(native_decode_tile) is not int or native_decode_tile not in (1, 8)
        or native_decode_tile == 1 and not native_decode):
        raise ValueError("Expected decode tile requires an explicit protected native selection")
    if "native_decode_projection_tile" not in receipt:
        # Offline pre-tile receipts only. Earlier native/scalar modes both used
        # tile8; this never accepts a missing tile1 declaration.
        if allow_legacy_decode and native_decode_tile == 8:
            return
        raise ValueError("Observed native decode projection tile is required")
    if (type(receipt["native_decode_projection_tile"]) is not int
        or receipt["native_decode_projection_tile"] != native_decode_tile):
        raise ValueError("Observed decode tile did not match the selected public experiment")


def _validate_model_receipt(receipt, card, fixture, count, output, *, owner_inventory,
                            reported_fixture, native_decode=False, native_decode_tile=8,
                            native_decode_kernels="chunk", allow_legacy_decode=False, audit_count=4, deferred_io=False):
    # Only the ordinary public entry below and the separately fixed hybrid
    # validator supply these constants. A receipt cannot select its inventory.
    owner_buffers, owner_bytes = owner_inventory
    if type(audit_count) is not int or audit_count not in (4,5):
        raise ValueError("Only fixed ordinary or deferred hybrid audit inventories are accepted")
    if type(deferred_io) is not bool or audit_count!=(5 if deferred_io else 4):
        raise ValueError("Audit inventory must match the independently selected IO cadence")
    expected = {
        "resident-bang": [49276, 2349, 1074, 5928],
        "resident-chat": [9419, 0, 2500, 628, 353, 1438, 488, 3242],
        "resident-rainwater": json.loads(pathlib.Path(__file__).with_name("receipts")
            .joinpath("rust-full-resident-rainwater-intel-20261005.json").read_bytes())
            ["public_generated_ids"],
    }[fixture]
    prompt = {"resident-bang": 1, "resident-chat": 15, "resident-rainwater": 75}[fixture]
    partition = [min(count, prompt - at) for at in range(0, prompt, count)]
    slots, cache, reserved = {
        "intel": (64, 8 * 1024**3, 26 * 1024**3),
        "nvidia": (16, 6 * 1024**3, 10 * 1024**3),
    }[card]
    expected_settings = {
        "vendor": 0x8086 if card == "intel" else 0x10DE,
        "scope": "fixed-public-synthetic-operator-test-only",
        "weights": "borrowed-unmodified-ollama-blob",
        "resident_graph": True,
        "resident_transient_io": True,
        "resident_owned_zero_audit_enabled": True,
        "resident_runtime_io_limit_bytes": 67108864,
        "native_public_max_batch": 8,
        "native_projection_tile": 8,
        "block_kernel": True,
        "block_large_only": True,
        "expert_bank_slots": slots,
        "expert_bank_segments": [slots],
        "expert_bank_accounted_limit_bytes": slots * 201326592,
        "static_cache_budget_bytes": cache,
        "device_reserved_limit_bytes": reserved,
        "explicit_static_cache": False,
        "expert_batch_window_bytes": 8388608,
        "expert_scatter_mode": "dynamic-1d",
        "small_cpu_cache_bytes": 0,
        "small_cpu_cache_matrices": 0,
        "small_cpu_calls": 0,
        "ordered_public_source_limit_bytes": None,
        "ordered_public_source_peak_bytes": 0,
    }
    for setting in ("block_moe", "batch_experts", "word_down", "ordered_public_upload",
                    "fused_matrix_fault", "small_cpu_enabled", "gpu_gdn_cpu_state_owner",
                    "prefill_last_head", "io_token_checkpoint"):
        expected_settings[setting] = False
    expected_settings["io_token_checkpoint"]=deferred_io
    if any(key not in receipt or type(receipt[key]) is not type(value)
           or receipt[key] != value for key, value in expected_settings.items()):
        raise ValueError("Effective native/device/default configuration did not match")
    if (
        receipt.get("fixture") != reported_fixture
        or receipt.get("native_max8") is not True
        or receipt.get("native_public_chunk_count") != count
        or receipt.get("native_public_chunk_partition") != partition
        or receipt.get("prompt_tokens") != prompt
        or receipt.get("seed") != 424242
        or receipt.get("temperature") != 0
        or receipt.get("public_generated_ids") != expected
        or receipt.get("resident_private_owned_bytes") != owner_bytes
        or receipt.get("owned_io_cleanup_accepted") is not True
    ):
        raise ValueError("Native compiled fixture/model receipt did not match")
    _, scalar_steps = decode_counts(receipt, len(expected) - 1, native_decode, allow_legacy_decode)
    validate_decode_tile(receipt, native_decode, native_decode_tile, allow_legacy_decode)
    validate_kernel_profile(receipt, native_decode, native_decode_tile, native_decode_kernels,
                            allow_legacy_decode)
    audits = receipt.get("owned_zero_audits", [])
    if len(audits) != audit_count or any(not isinstance(item, dict) or
        item.get("main_buffers") != owner_buffers or item.get("main_bytes") != owner_bytes or
        any(type(item.get(key)) is not int for key in ("source_buffers", "source_bytes")) or
        item["source_buffers"] < 0 or item["source_bytes"] < 0 or
        (item["source_buffers"] == 0) != (item["source_bytes"] == 0) or
        item["source_bytes"] < item["source_buffers"] * 32768
        for item in audits):
        raise ValueError("Four complete private-owner audits are required")
    # Source counts accumulate checked resets, not simultaneous live residency.
    # Each original scalar continuation audits15 buffers/17 full32-KiB pages;
    # Native prompt/decode chunks use their separate owned runtime sources.
    if (audits[0]["source_buffers"] != scalar_steps * 15 or
        audits[0]["source_bytes"] != scalar_steps * 17 * 32768) or any(
        item["source_buffers"] != 0 for item in audits[1:]
    ):
        raise ValueError("Exact observed scalar and empty cancelled/fresh source-owner audits required")
    required = (
        "public_mid_token_cancel layer=0 erased=true",
        "public_mid_token_cancel layer=3 erased=true",
        "public_resident_cancel_erased=true poisoned_retry_rejected=true fresh_first_token_matches=true",
        "public_resident_owned_cleanup=true context_retired=false",
    )
    if any(message not in output for message in required):
        raise ValueError("Protected lifecycle completion evidence is incomplete")
    if native_decode_tile == 1 and any(message not in output.splitlines() for message in TILE_ONE_COMPLETION):
        raise ValueError("Native tile1 continuation/recovery completion evidence is incomplete")
    if any(message not in output.splitlines() for message in kernel_completion(native_decode_kernels)):
        raise ValueError("Native kernel profile continuation/recovery completion evidence is incomplete")


def validate_model_receipt(receipt, card, fixture, count, output, *, native_decode=False,
                           native_decode_tile=8, native_decode_kernels="chunk", allow_legacy_decode=False):
    # Ordinary acceptance remains exactly 1376 owners; no hybrid opt-in or
    # caller/receipt-selected inventory is exposed by this entry.
    _validate_model_receipt(receipt, card, fixture, count, output,
                           owner_inventory=(1376, 818249728), reported_fixture=fixture,
                           native_decode=native_decode, native_decode_tile=native_decode_tile,
                           native_decode_kernels=native_decode_kernels,
                           allow_legacy_decode=allow_legacy_decode)


def validate_receipt(receipt, card, fixture, count, output, *, native_decode=False,
                     native_decode_tile=8, native_decode_kernels="chunk", allow_legacy_decode=False):
    if "verified_lab_limits swap_max=0 core_soft=0 core_hard=0 memory_max=8589934592" not in output:
        raise ValueError("Protected lifecycle completion evidence is incomplete")
    validate_model_receipt(receipt, card, fixture, count, output, native_decode=native_decode,
                           native_decode_tile=native_decode_tile, native_decode_kernels=native_decode_kernels,
                           allow_legacy_decode=allow_legacy_decode)


def worker_environment(model, receipt, count, native_decode=False, native_decode_tile=8,
                       native_decode_kernels="chunk"):
    if type(native_decode) is not bool:
        raise ValueError("Native decode selection must be an explicit boolean")
    if (type(native_decode_tile) is not int or native_decode_tile not in (1, 8)
        or native_decode_tile == 1 and not native_decode):
        raise ValueError("Native decode tile1 requires explicit native decode")
    validate_kernel_selection(native_decode_kernels, native_decode, native_decode_tile)
    env = {
        "XDG_RUNTIME_DIR": "/run/user/" + str(os.getuid()),
        "CUBECL_DEBUG_LOG": "0",
        "KUNO_PUBLIC_RECEIPT": str(receipt),
        "KUNO_TEST_GGUF": str(model),
        "KUNO_LAB_TRANSIENT_IO": "1",
        "KUNO_LAB_NATIVE_MAX8": "1",
        "KUNO_LAB_NATIVE_BATCH_COUNT": str(count),
        "KUNO_LAB_NATIVE_DECODE": "1" if native_decode else "0",
        "KUNO_LAB_NATIVE_DECODE_TILE": str(native_decode_tile),
        "KUNO_LAB_NATIVE_DECODE_KERNELS": native_decode_kernels,
        "KUNO_LAB_CANCEL_CHECK": "1",
        "KUNO_LAB_OWNED_ZERO_AUDIT": "1",
    }
    for suffix in ("BLOCK_MOE", "BATCH_EXPERTS", "WORD_DOWN", "ORDERED_PUBLIC_UPLOAD",
                   "FUSED_MATRIX_FAULT", "SMALL_CPU", "CPU_GDN", "PREFILL_LAST_HEAD",
                   "IO_TOKEN_CHECKPOINT"):
        env["KUNO_LAB_" + suffix] = "0"
    return env


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cargo-artifacts", type=pathlib.Path, required=True)
    parser.add_argument("--model", type=pathlib.Path, required=True)
    parser.add_argument("--card", choices=("intel", "nvidia"), required=True)
    parser.add_argument("--fixture", choices=("resident-bang", "resident-chat", "resident-rainwater"), required=True)
    parser.add_argument("--count", type=int, choices=(1, 3, 5, 8), required=True)
    parser.add_argument("--native-decode", action="store_true",
                        help="Continue generation using native count1 chunks on the same owner")
    parser.add_argument("--native-decode-tile", type=int, choices=(1, 8), default=8,
                        help="Public decode projection tile; tile1 requires --native-decode")
    parser.add_argument("--native-decode-kernels", choices=KERNEL_PROFILES, default="chunk",
                        help="Public kernel profile; non-chunk requires protected native tile1 decode")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--receipt", type=pathlib.Path, required=True)
    args = parser.parse_args()
    if args.native_decode_tile == 1 and not args.native_decode:
        parser.error("--native-decode-tile 1 requires --native-decode")
    if args.native_decode_kernels != "chunk" and (not args.native_decode or args.native_decode_tile != 1):
        parser.error("--native-decode-kernels requires --native-decode and --native-decode-tile 1")
    if not re.fullmatch(r"[a-z0-9-]{1,48}", args.run_id):
        parser.error("run-id must be 1..48 lowercase letters, digits or hyphens")
    target = args.receipt.resolve()
    metadata = target.with_suffix(target.suffix + ".run.json")
    if target.exists() or metadata.exists():
        parser.error("Refusing to replace an existing receipt or run record")
    frozen, identity, artifacts_identity = cargo_binary(args.cargo_artifacts)
    env = worker_environment(args.model.resolve(), target, args.count, args.native_decode,
                             args.native_decode_tile, args.native_decode_kernels)
    command = [
        "systemd-run", "--user", "--unit=kuno-user-" + args.run_id,
        "--collect", "--wait", "--pipe", "--property=MemorySwapMax=0",
        "--property=LimitCORE=0", "--property=MemoryMax=8G",
        "--property=RuntimeMaxSec=900",
        "--property=UnsetEnvironment=KUNO_LAB_BLOCK_KERNEL KUNO_LAB_EXPERT_SLOTS "
        "KUNO_LAB_STATIC_CACHE_BYTES KUNO_LAB_EXPERT_BATCH_BYTES KUNO_LAB_EXPERT_SCATTER",
    ]
    command.extend("--setenv=" + key + "=" + value for key, value in env.items())
    command.extend([
        sys.executable, str(pathlib.Path(__file__).with_name("protected-lab-exec.py")),
        str(frozen), args.card, args.fixture,
    ])
    print("verified_cargo_executable_sha256=" + identity, flush=True)
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    lines = []
    for line in process.stdout:
        print(line, end="", flush=True)
        lines.append(line)
    if process.wait() != 0:
        raise SystemExit("Protected compiled-public fixture failed; no acceptance recorded")
    if not target.is_file():
        raise SystemExit("Missing native model receipt; successful probe exit is insufficient")
    receipt_snapshot = target.read_bytes()
    receipt = json.loads(receipt_snapshot)
    validate_receipt(receipt, args.card, args.fixture, args.count, "".join(lines),
                     native_decode=args.native_decode, native_decode_tile=args.native_decode_tile,
                     native_decode_kernels=args.native_decode_kernels)
    record = {
        "scope": "fixed-public-model-partition-output-and-lifecycle-fixture-only",
        "cargo_artifacts_sha256": artifacts_identity,
        "frozen_executable_sha256": identity,
        "card": args.card, "fixture": args.fixture, "count": args.count,
        "native_decode": args.native_decode,
        "native_decode_projection_tile": args.native_decode_tile,
        "native_decode_kernels": args.native_decode_kernels,
        "receipt_sha256": digest(receipt_snapshot),
        "protected_lifecycle_completion_checked": True,
        "matched_performance_measurement": False,
    }
    with metadata.open("x") as output:
        json.dump(record, output, indent=2, sort_keys=True)
        output.write("\n")
    print("verified_native_model_lifecycle_receipt=" + str(target), flush=True)


if __name__ == "__main__":
    main()
