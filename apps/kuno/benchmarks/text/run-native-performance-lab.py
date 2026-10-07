"""Fixed-public, same-binary memory or native-decode ABBA experiment; never a request logger."""

import argparse
import errno
import hashlib
import importlib.util
import json
import os
import pathlib
import re
import resource
import subprocess
import sys
import threading
import time

HERE = pathlib.Path(__file__).resolve().parent
ABBA = (8, 52, 52, 8)
MEMORY_SCOPE = "fixed-public-native-cgroup-abba-only"
DECODE_SCOPE = "fixed-public-native-decode-abba-only"
TILE_SCOPE = "fixed-public-native-decode-tile-abba-only"
DECODE_CAPS = (52, 52, 52, 52)
DECODE_FLAGS = (False, True, True, False)
TILE_FLAGS = (True,) * 4
TILE_TILES = (8, 1, 1, 8)
FIXED_PLAN = {"fixture": "resident-rainwater", "prompt_tokens": 75, "output_tokens": 128,
              "seed": 424242, "temperature": 0, "native_count": 8,
              "native_context": 203, "model_unloaded_before_each_child": True,
              "private_main_pages": 1376, "private_main_bytes": 818249728,
              "runtime_io_cap_bytes": 67108864}
PLAN_IDENTITIES = {"card", "unit", "model", "executable", "executable_sha256",
                   "cargo_artifacts_sha256", "cargo_executable_at_build", "fixture_sha256",
                   "reference_sha256", "supervisor_sha256", "source_head_at_freeze"}
MODEL_NAME = "sha256-89059c880c2f000b24520919d4ba2edd2a8101327c7f8ab33b2280e01addda75"
MODEL_SIZE = 119150722112
UNIT_PATTERN = r"kuno-public-performance-[a-z0-9-]{1,40}\.service"
SPEC = importlib.util.spec_from_file_location("native_acceptance", HERE / "run-native-max8-lab.py")
ACCEPTANCE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ACCEPTANCE)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def write_new(path, value):
    with pathlib.Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")


def numeric_map(text):
    result = {}
    for line in text.splitlines():
        key, value = line.split()
        if not re.fullmatch(r"[a-z_]+", key) or not value.isdecimal() or key in result:
            raise ValueError("Invalid numeric counter")
        result[key] = int(value)
    return result


def pressure_map(text):
    result = {}
    for line in text.splitlines():
        fields = line.split()
        if fields[0] not in ("some", "full") or fields[0] in result:
            raise ValueError("Invalid pressure counter")
        values = dict(field.split("=", 1) for field in fields[1:])
        result[fields[0]] = {key: int(values[key]) if key == "total" else float(values[key])
                             for key in ("avg10", "avg60", "avg300", "total")}
    return result


def cgroup_path():
    entries = pathlib.Path("/proc/self/cgroup").read_text().splitlines()
    path = next(line.split(":", 2)[2] for line in entries if line.startswith("0::"))
    return pathlib.Path("/sys/fs/cgroup") / path.lstrip("/")


def verify_limits(cgroup, unit, cap, core=None):
    if (not re.fullmatch(UNIT_PATTERN, unit) or cgroup.name != unit
        or type(cap) is not int or cap not in ABBA):
        raise ValueError("Unexpected laboratory unit or memory cap")
    core = resource.getrlimit(resource.RLIMIT_CORE) if core is None else core
    if (cgroup / "memory.max").read_text().strip() != str(cap * 1024**3):
        raise ValueError("Actual memory cap did not match")
    if (cgroup / "memory.swap.max").read_text().strip() != "0" or core != (0, 0):
        raise ValueError("Actual swap/core protection did not match")
    if (cgroup / "memory.swap.current").read_text().strip() != "0":
        raise ValueError("Laboratory cgroup has swapped pages")


def snapshot(cgroup, pid=None):
    value = {"monotonic_ns": time.monotonic_ns()}
    for name in ("memory.current", "memory.peak", "memory.swap.current", "memory.max"):
        value[name] = int((cgroup / name).read_text().strip())
    for name in ("memory.stat", "memory.events"):
        value[name] = numeric_map((cgroup / name).read_text())
    value["memory.pressure"] = pressure_map((cgroup / "memory.pressure").read_text())
    if pid is not None:
        try:
            text = pathlib.Path(f"/proc/{pid}/io").read_text()
            value["process_io"] = numeric_map(text.replace(":", ""))
        except OSError as error:
            if error.errno not in (errno.ENOENT, errno.ESRCH, errno.EACCES, errno.EPERM):
                raise
            value["process_io"] = None
            value["process_io_errno"] = error.errno
    return value


def model_identity(path):
    path = pathlib.Path(path).resolve(strict=True)
    stat = path.stat()
    if path.name != MODEL_NAME or not path.is_file() or stat.st_size != MODEL_SIZE:
        raise ValueError("Only the existing declared Qwen artifact is accepted")
    return {"path": str(path), "device": stat.st_dev, "inode": stat.st_ino,
            "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def experiment_schedule(plan):
    scope = plan.get("scope")
    if scope == MEMORY_SCOPE:
        caps, flags = ABBA, (False,) * 4
        # Historical memory plans predate this explicit field. They retain only
        # their original all-scalar schedule; completed plans still cannot replay.
        declared_flags = plan.get("native_decode_flags", list(flags))
    elif scope == DECODE_SCOPE:
        caps, flags = DECODE_CAPS, DECODE_FLAGS
        declared_flags = plan.get("native_decode_flags")
    elif scope == TILE_SCOPE:
        caps, flags = DECODE_CAPS, TILE_FLAGS
        declared_flags = plan.get("native_decode_flags")
    else:
        raise ValueError("Unknown fixed-public experiment scope")
    declared_caps = plan.get("caps_gib")
    if (type(declared_caps) is not list or any(type(cap) is not int for cap in declared_caps)
        or declared_caps != list(caps) or type(declared_flags) is not list
        or any(type(flag) is not bool for flag in declared_flags) or declared_flags != list(flags)):
        raise ValueError("Fixed cap/decode schedule did not match the experiment scope")
    projection_tiles(plan)
    return tuple(zip(caps, flags))


def projection_tiles(plan):
    scope = plan.get("scope")
    if scope == TILE_SCOPE:
        expected = TILE_TILES
        declared = plan.get("native_decode_tiles")
    elif scope in (MEMORY_SCOPE, DECODE_SCOPE):
        expected = (8,) * 4
        # Earlier memory/native comparison plans had no tile field and used8.
        # This explicit historical interpretation never permits tile1.
        declared = plan.get("native_decode_tiles", list(expected))
    else:
        raise ValueError("Unknown fixed-public tile experiment scope")
    if (type(declared) is not list or any(type(tile) is not int for tile in declared)
        or declared != list(expected)):
        raise ValueError("Fixed projection tiles did not match the experiment scope")
    return expected


def selected_sample(plan, index, cap=None):
    schedule = experiment_schedule(plan)
    if type(index) is not int or not 1 <= index <= len(schedule):
        raise ValueError("Invalid public sample index")
    selected_cap, native_decode = schedule[index - 1]
    if cap is not None and (type(cap) is not int or cap != selected_cap):
        raise ValueError("Sample cap did not match its declared index")
    return selected_cap, native_decode


def selected_tile(plan, index):
    selected_sample(plan, index)
    return projection_tiles(plan)[index - 1]


def worker_environment(plan, receipt, index=None):
    if plan.get("scope") in (DECODE_SCOPE, TILE_SCOPE) and index is None:
        raise ValueError("Native decode comparison requires an indexed child")
    native_decode = selected_sample(plan, index)[1] if index is not None else False
    native_decode_tile = selected_tile(plan, index) if index is not None else 8
    env = {"PATH": "/usr/local/bin:/usr/bin:/bin", "HOME": str(pathlib.Path.home()),
           "XDG_RUNTIME_DIR": f"/run/user/{os.getuid()}", "CUBECL_DEBUG_LOG": "0",
           "KUNO_TEST_GGUF": plan["model"]["path"], "KUNO_PUBLIC_RECEIPT": str(receipt),
           "KUNO_LAB_TRANSIENT_IO": "1", "KUNO_LAB_NATIVE_MAX8": "1",
           "KUNO_LAB_NATIVE_DECODE": "1" if native_decode else "0",
           "KUNO_LAB_NATIVE_DECODE_TILE": str(native_decode_tile),
           "KUNO_LAB_NATIVE_BATCH_COUNT": "8", "KUNO_LAB_CANCEL_CHECK": "1",
           "KUNO_LAB_OWNED_ZERO_AUDIT": "1"}
    for suffix in ("BLOCK_MOE", "BATCH_EXPERTS", "WORD_DOWN", "ORDERED_PUBLIC_UPLOAD",
                   "FUSED_MATRIX_FAULT", "SMALL_CPU", "CPU_GDN", "PREFILL_LAST_HEAD",
                   "IO_TOKEN_CHECKPOINT"):
        env["KUNO_LAB_" + suffix] = "0"
    return env


def load_plan(path):
    plan = json.loads(pathlib.Path(path).read_text())
    if type(plan) is not dict:
        raise ValueError("Public laboratory plan must be an object")
    experiment_schedule(plan)
    if (plan.get("card") not in ("intel", "nvidia") or type(plan.get("unit")) is not str or
        not re.fullmatch(UNIT_PATTERN, plan.get("unit", ""))):
        raise ValueError("Invalid public laboratory plan")
    if plan["scope"] in (DECODE_SCOPE, TILE_SCOPE):
        expected_keys = set(FIXED_PLAN) | PLAN_IDENTITIES | {"scope", "caps_gib", "native_decode_flags"}
        if plan["scope"] == TILE_SCOPE or "native_decode_tiles" in plan:
            expected_keys.add("native_decode_tiles")
        if set(plan) != expected_keys or any(
            key not in plan or type(plan[key]) is not type(value) or plan[key] != value
            for key, value in FIXED_PLAN.items()
        ):
            raise ValueError("Native decode requires the exact fixed public plan")
        model = plan["model"]
        if (type(model) is not dict or set(model) != {"path", "device", "inode", "size", "mtime_ns"}
            or type(model["path"]) is not str or not model["path"]
            or any(type(model[key]) is not int or model[key] < 0
                   for key in ("device", "inode", "size", "mtime_ns"))):
            raise ValueError("Invalid original model identity")
        for key in ("executable_sha256", "cargo_artifacts_sha256", "fixture_sha256",
                    "reference_sha256", "supervisor_sha256"):
            if type(plan[key]) is not str or not re.fullmatch(r"[0-9a-f]{64}", plan[key]):
                raise ValueError("Invalid public plan hash")
        if (type(plan["source_head_at_freeze"]) is not str
            or not re.fullmatch(r"[0-9a-f]{40}", plan["source_head_at_freeze"])
            or any(type(plan[key]) is not str or not plan[key]
                   for key in ("executable", "cargo_executable_at_build"))):
            raise ValueError("Invalid public source/artifact identity")
        if (plan["fixture_sha256"] != sha((HERE / "qwen38-baseline.json").read_bytes())
            or plan["reference_sha256"] != sha(
                (HERE / "receipts/rust-full-resident-rainwater-intel-20261005.json").read_bytes())
            or plan["supervisor_sha256"] != sha(pathlib.Path(__file__).read_bytes())):
            raise ValueError("Public fixture/reference/supervisor identity changed")
    if model_identity(plan["model"]["path"]) != plan["model"]:
        raise ValueError("Original model identity changed")
    executable = pathlib.Path(plan["executable"])
    if sha(executable.read_bytes()) != plan["executable_sha256"]:
        raise ValueError("Frozen executable identity changed")
    if not executable.is_file() or not os.access(executable, os.X_OK):
        raise ValueError("Frozen executable is not executable")
    return plan


def guard(plan_path, cap, index):
    plan = load_plan(plan_path)
    _, native_decode = selected_sample(plan, index, cap)
    native_decode_tile = selected_tile(plan, index)
    verify_limits(cgroup_path(), plan["unit"], cap)
    if os.environ.get("KUNO_LAB_NATIVE_DECODE") != ("1" if native_decode else "0"):
        raise ValueError("Worker decode setting did not match the indexed plan")
    if os.environ.get("KUNO_LAB_NATIVE_DECODE_TILE") != str(native_decode_tile):
        raise ValueError("Worker decode tile did not match the indexed plan")
    print(f"verified_performance_limits swap_max=0 core_soft=0 core_hard=0 memory_max={cap * 1024**3}", flush=True)
    print(f"verified_performance_sample sample_index={index} native_decode={int(native_decode)} native_decode_tile={native_decode_tile}", flush=True)
    os.execv(plan["executable"], [plan["executable"], plan["card"], "resident-rainwater"])


def accepted_result(receipt, output, card, cap, status, before, after, *, native_decode=False,
                    native_decode_tile=8, index=None):
    if type(native_decode) is not bool or type(cap) is not int or cap not in ABBA or native_decode and cap != 52:
        raise ValueError("Invalid expected public decode mode/cap")
    if native_decode and index is None:
        raise ValueError("Native decode requires a protected indexed sample")
    marker = f"verified_performance_limits swap_max=0 core_soft=0 core_hard=0 memory_max={cap * 1024**3}"
    if status != 0 or marker not in output:
        raise ValueError("Protected public worker did not finish")
    if index is not None:
        sample_marker = f"verified_performance_sample sample_index={index} native_decode={int(native_decode)} native_decode_tile={native_decode_tile}"
        if type(index) is not int or not 1 <= index <= 4 or sample_marker not in output:
            raise ValueError("Protected sample index/decode proof did not match")
    for key in ("oom", "oom_kill", "oom_group_kill"):
        if after["memory.events"].get(key, 0) != before["memory.events"].get(key, 0):
            raise ValueError("Cgroup OOM occurred during the worker")
    ACCEPTANCE.validate_model_receipt(receipt, card, "resident-rainwater", 8, output,
                                      native_decode=native_decode, native_decode_tile=native_decode_tile)
    if (type(receipt.get("generation_ms")) is not int or receipt["generation_ms"] <= 0
        or type(receipt.get("prefill_ms")) is not int or receipt["prefill_ms"] < 0):
        raise ValueError("Completed public timing must be nonnegative with positive generation time")


def run_child(plan, plan_path, index, cap, cgroup):
    _, native_decode = selected_sample(plan, index, cap)
    native_decode_tile = selected_tile(plan, index)
    verify_limits(cgroup, plan["unit"], cap)
    target = plan_path.parent / f"sample-{index}-cap-{cap}.json"
    if any(path.exists() for path in (target, target.with_suffix(".cgroup.json"),
                                     target.with_suffix(".telemetry.jsonl"))):
        raise ValueError("Refusing to overwrite a sample")
    before = snapshot(cgroup)
    command = [sys.executable, str(pathlib.Path(__file__).resolve()), "--guard", str(plan_path), str(cap), str(index)]
    process = subprocess.Popen(command, env=worker_environment(plan, target, index),
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    # Never retain raw backend errors, generated text or arbitrary output.
    markers = []
    def drain():
        for raw in iter(lambda: process.stdout.readline(4096), b""):
            text = raw.decode("utf-8", errors="replace").strip()
            if text.startswith(("verified_performance_limits ", "verified_performance_sample ", "public_mid_token_cancel ",
                                "public_resident_cancel_erased=", "public_resident_owned_cleanup=")):
                if len(markers) < 8:
                    markers.append(text)
    drainer = threading.Thread(target=drain, daemon=True)
    drainer.start()
    samples = []
    telemetry = target.with_suffix(".telemetry.jsonl").open("x")
    try:
        while process.poll() is None:
            sample = snapshot(cgroup, process.pid)
            samples.append(sample)
            telemetry.write(json.dumps(sample, sort_keys=True) + "\n")
            telemetry.flush()
            time.sleep(0.5)
        drainer.join()
        after = snapshot(cgroup)
        verify_limits(cgroup, plan["unit"], cap)
        result = {"sample_index": index, "cap_gib": cap, "native_decode": native_decode,
                  "native_decode_projection_tile": native_decode_tile,
                  "exit_status": process.returncode,
                  "before": before, "after": after, "samples": samples,
                  "sampled_memory_peak_bytes": max([before["memory.current"], after["memory.current"]] +
                                                    [s["memory.current"] for s in samples]),
                  "unit_memory_peak_is_cumulative": True, "accepted": False}
        if target.exists():
            receipt = json.loads(target.read_text())
            result["model_receipt_sha256"] = sha(target.read_bytes())
        else:
            receipt = None
        try:
            if receipt is None:
                raise ValueError("Missing checked model receipt")
            accepted_result(receipt, "\n".join(markers), plan["card"], cap, process.returncode, before, after,
                            native_decode=native_decode, native_decode_tile=native_decode_tile, index=index)
            result["accepted"] = True
            result["prefill_ms"] = receipt["prefill_ms"]
            result["generation_ms"] = receipt["generation_ms"]
            result["following_decode_steps"] = 127
            result["decode_tokens_per_second"] = 127000 / receipt["generation_ms"]
            result["native_decode_steps"] = receipt["native_decode_steps"]
            result["scalar_decode_steps"] = receipt["scalar_decode_steps"]
            result["native_prefill_projection_tile"] = receipt["native_projection_tile"]
        except (ValueError, KeyError, TypeError):
            result["failure_class"] = "worker-or-model-or-cleanup-validation"
        write_new(target.with_suffix(".cgroup.json"), result)
        if not result["accepted"]:
            raise ValueError("Public experiment rejected; inspect numeric sample record")
        return {key: value for key, value in result.items() if key != "samples"}
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        process.stdout.close()
        drainer.join(timeout=2)
        telemetry.close()


def supervise(plan_path):
    plan_path = pathlib.Path(plan_path).resolve(strict=True)
    plan = load_plan(plan_path)
    schedule = experiment_schedule(plan)
    if (plan_path.parent / "abba-receipt.json").exists():
        raise ValueError("Refusing to replay a completed or failed experiment")
    cgroup = cgroup_path()
    verify_limits(cgroup, plan["unit"], schedule[0][0])
    results = []
    record = {"scope": plan["scope"], "plan_sha256": sha(plan_path.read_bytes()),
              "cgroup": str(cgroup), "accepted": False, "results": results,
              "global_caches_flushed": False, "true_hot_full_request_implemented": False,
              "production_limits_changed": False}
    record["caps_gib"] = [cap for cap, _ in schedule]
    record["native_decode_flags"] = [native for _, native in schedule]
    record["native_decode_tiles"] = list(projection_tiles(plan))
    stage = "sample-transition"
    try:
        for index, (cap, native_decode) in enumerate(schedule, 1):
            stage = "sample-transition"
            transition_before = snapshot(cgroup)
            # Only this declared, currently enclosing, public unit can be changed.
            if cgroup.name != plan["unit"] or not re.fullmatch(UNIT_PATTERN, plan["unit"]):
                raise ValueError("Refusing to change a different service")
            subprocess.run(["systemctl", "--user", "set-property", "--runtime", plan["unit"],
                            f"MemoryMax={cap}G"], check=True, stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL)
            verify_limits(cgroup, plan["unit"], cap)
            transition_after = snapshot(cgroup)
            print(f"public_abba_sample={index} cap_gib={cap} native_decode={int(native_decode)} native_decode_tile={selected_tile(plan, index)}", flush=True)
            stage = "sample-worker"
            result = run_child(plan, plan_path, index, cap, cgroup)
            result["cap_transition_before"] = transition_before
            result["cap_transition_after"] = transition_after
            results.append(result)
        stage = "model-identity"
        if model_identity(plan["model"]["path"]) != plan["model"]:
            raise ValueError("Original model changed during experiment")
        record["accepted"] = True
    except (ValueError, OSError, subprocess.SubprocessError):
        record["failure_class"] = "limits-or-worker-or-identity"
        record["failure_stage"] = stage
        if stage != "model-identity":
            record["failed_sample_index"] = index
        raise
    finally:
        write_new(plan_path.parent / "abba-receipt.json", record)


def launch(args):
    if not re.fullmatch(r"[a-z0-9-]{1,40}", args.run_id):
        raise ValueError("run-id must be1..40 lowercase letters, digits or hyphens")
    if type(args.native_decode_comparison) is not bool:
        raise ValueError("Native decode comparison must be an explicit boolean")
    if type(args.native_decode_tile_comparison) is not bool or args.native_decode_comparison and args.native_decode_tile_comparison:
        raise ValueError("Select only one explicit native comparison mode")
    output = args.output.resolve()
    output.mkdir(mode=0o700, parents=True, exist_ok=False)
    # Actual Cargo release metadata selection is unchanged from correctness.
    temporary, identity, artifacts_sha = ACCEPTANCE.cargo_binary(args.cargo_artifacts)
    artifact = next(record for record in map(json.loads, args.cargo_artifacts.read_bytes().splitlines())
                    if record.get("reason") == "compiler-artifact" and
                    record.get("target", {}).get("name") == "kuno-text-lab" and record.get("executable"))
    original_executable = pathlib.Path(artifact["executable"]).resolve()
    if output.is_relative_to(original_executable.parent.parent):
        raise ValueError("The durable record must be outside the Cargo target directory")
    frozen = output / ("kuno-text-lab-" + identity)
    with frozen.open("xb") as stream:
        stream.write(temporary.read_bytes())
    frozen.chmod(0o500)
    if sha(frozen.read_bytes()) != identity:
        raise ValueError("Durable artifact copy failed")
    artifact_copy = output / "cargo-artifacts.jsonl"
    with artifact_copy.open("xb") as stream:
        stream.write(args.cargo_artifacts.read_bytes())
    native_comparison = args.native_decode_comparison
    tile_comparison = args.native_decode_tile_comparison
    caps = DECODE_CAPS if native_comparison or tile_comparison else ABBA
    flags = TILE_FLAGS if tile_comparison else DECODE_FLAGS if native_comparison else (False,) * 4
    tiles = TILE_TILES if tile_comparison else (8,) * 4
    scope = TILE_SCOPE if tile_comparison else DECODE_SCOPE if native_comparison else MEMORY_SCOPE
    plan = {"scope": scope, "caps_gib": list(caps),
            "native_decode_flags": list(flags),
            "native_decode_tiles": list(tiles),
            "card": args.card, "unit": "kuno-public-performance-" + args.run_id + ".service",
            "model": model_identity(args.model), "executable": str(frozen),
            "executable_sha256": identity, "cargo_artifacts_sha256": artifacts_sha,
            "cargo_executable_at_build": str(original_executable),
            "fixture_sha256": sha((HERE / "qwen38-baseline.json").read_bytes()),
            "reference_sha256": sha((HERE / "receipts/rust-full-resident-rainwater-intel-20261005.json").read_bytes()),
            "supervisor_sha256": sha(pathlib.Path(__file__).read_bytes()),
            "source_head_at_freeze": subprocess.check_output(
                ["git", "-C", str(HERE), "rev-parse", "HEAD"], text=True).strip(),
            **FIXED_PLAN}
    plan_path = output / "plan.json"
    write_new(plan_path, plan)
    command = ["systemd-run", "--user", "--unit=" + plan["unit"], "--collect", "--wait", "--pipe",
               "--property=MemorySwapMax=0", "--property=LimitCORE=0", f"--property=MemoryMax={caps[0]}G",
               "--property=RuntimeMaxSec=3600", "--setenv=XDG_RUNTIME_DIR=/run/user/" + str(os.getuid()),
               "--setenv=DBUS_SESSION_BUS_ADDRESS=unix:path=/run/user/" + str(os.getuid()) + "/bus",
               sys.executable, str(pathlib.Path(__file__).resolve()), "--supervise", str(plan_path)]
    print("durable_public_plan=" + str(plan_path), flush=True)
    subprocess.run(command, check=True)


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--supervise":
        if len(sys.argv) != 3:
            raise ValueError("Invalid supervisor arguments")
        supervise(sys.argv[2])
    elif len(sys.argv) > 1 and sys.argv[1] == "--guard":
        if len(sys.argv) != 5:
            raise ValueError("Invalid guard arguments")
        guard(sys.argv[2], int(sys.argv[3]), int(sys.argv[4]))
    else:
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("--cargo-artifacts", type=pathlib.Path, required=True)
        parser.add_argument("--model", type=pathlib.Path, required=True)
        parser.add_argument("--card", choices=("intel", "nvidia"), required=True)
        parser.add_argument("--run-id", required=True)
        parser.add_argument("--output", type=pathlib.Path, required=True)
        comparison = parser.add_mutually_exclusive_group()
        comparison.add_argument("--native-decode-comparison", action="store_true",
                            help="Compare scalar/native/native/scalar decode at the fixed 52-GiB host cap")
        comparison.add_argument("--native-decode-tile-comparison", action="store_true",
                                help="Compare native tile8/tile1/tile1/tile8 at the fixed 52-GiB host cap")
        launch(parser.parse_args())


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError, subprocess.SubprocessError):
        raise SystemExit("Public performance experiment failed; no acceptance claim") from None
