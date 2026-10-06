"""Fixed-public, same-binary ABBA memory-containment experiment; never a request logger."""

import argparse
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
    if not re.fullmatch(UNIT_PATTERN, unit) or cgroup.name != unit or cap not in ABBA:
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
        except FileNotFoundError:
            value["process_io"] = None
    return value


def model_identity(path):
    path = pathlib.Path(path).resolve(strict=True)
    stat = path.stat()
    if path.name != MODEL_NAME or not path.is_file() or stat.st_size != MODEL_SIZE:
        raise ValueError("Only the existing declared Qwen artifact is accepted")
    return {"path": str(path), "device": stat.st_dev, "inode": stat.st_ino,
            "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def worker_environment(plan, receipt):
    env = {"PATH": "/usr/local/bin:/usr/bin:/bin", "HOME": str(pathlib.Path.home()),
           "XDG_RUNTIME_DIR": f"/run/user/{os.getuid()}", "CUBECL_DEBUG_LOG": "0",
           "KUNO_TEST_GGUF": plan["model"]["path"], "KUNO_PUBLIC_RECEIPT": str(receipt),
           "KUNO_LAB_TRANSIENT_IO": "1", "KUNO_LAB_NATIVE_MAX8": "1",
           "KUNO_LAB_NATIVE_BATCH_COUNT": "8", "KUNO_LAB_CANCEL_CHECK": "1",
           "KUNO_LAB_OWNED_ZERO_AUDIT": "1"}
    for suffix in ("BLOCK_MOE", "BATCH_EXPERTS", "WORD_DOWN", "ORDERED_PUBLIC_UPLOAD",
                   "FUSED_MATRIX_FAULT", "SMALL_CPU", "CPU_GDN", "PREFILL_LAST_HEAD",
                   "IO_TOKEN_CHECKPOINT"):
        env["KUNO_LAB_" + suffix] = "0"
    return env


def load_plan(path):
    plan = json.loads(pathlib.Path(path).read_text())
    if (plan.get("scope") != "fixed-public-native-cgroup-abba-only" or
        plan.get("caps_gib") != list(ABBA) or plan.get("card") not in ("intel", "nvidia") or
        not re.fullmatch(UNIT_PATTERN, plan.get("unit", ""))):
        raise ValueError("Invalid public laboratory plan")
    if model_identity(plan["model"]["path"]) != plan["model"]:
        raise ValueError("Original model identity changed")
    executable = pathlib.Path(plan["executable"])
    if sha(executable.read_bytes()) != plan["executable_sha256"]:
        raise ValueError("Frozen executable identity changed")
    if not executable.is_file() or not os.access(executable, os.X_OK):
        raise ValueError("Frozen executable is not executable")
    return plan


def guard(plan_path, cap):
    plan = load_plan(plan_path)
    verify_limits(cgroup_path(), plan["unit"], cap)
    print(f"verified_performance_limits swap_max=0 core_soft=0 core_hard=0 memory_max={cap * 1024**3}", flush=True)
    os.execv(plan["executable"], [plan["executable"], plan["card"], "resident-rainwater"])


def accepted_result(receipt, output, card, cap, status, before, after):
    marker = f"verified_performance_limits swap_max=0 core_soft=0 core_hard=0 memory_max={cap * 1024**3}"
    if status != 0 or marker not in output:
        raise ValueError("Protected public worker did not finish")
    for key in ("oom", "oom_kill", "oom_group_kill"):
        if after["memory.events"].get(key, 0) != before["memory.events"].get(key, 0):
            raise ValueError("Cgroup OOM occurred during the worker")
    ACCEPTANCE.validate_model_receipt(receipt, card, "resident-rainwater", 8, output)


def run_child(plan, plan_path, index, cap, cgroup):
    target = plan_path.parent / f"sample-{index}-cap-{cap}.json"
    if any(path.exists() for path in (target, target.with_suffix(".cgroup.json"),
                                     target.with_suffix(".telemetry.jsonl"))):
        raise ValueError("Refusing to overwrite a sample")
    before = snapshot(cgroup)
    command = [sys.executable, str(pathlib.Path(__file__).resolve()), "--guard", str(plan_path), str(cap)]
    process = subprocess.Popen(command, env=worker_environment(plan, target),
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    # Never retain raw backend errors, generated text or arbitrary output.
    markers = []
    def drain():
        for raw in iter(lambda: process.stdout.readline(4096), b""):
            text = raw.decode("utf-8", errors="replace").strip()
            if text.startswith(("verified_performance_limits ", "public_mid_token_cancel ",
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
        result = {"sample_index": index, "cap_gib": cap, "exit_status": process.returncode,
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
            accepted_result(receipt, "\n".join(markers), plan["card"], cap, process.returncode, before, after)
            result["accepted"] = True
            result["prefill_ms"] = receipt["prefill_ms"]
            result["generation_ms"] = receipt["generation_ms"]
            result["following_decode_steps"] = 127
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
    if (plan_path.parent / "abba-receipt.json").exists():
        raise ValueError("Refusing to replay a completed or failed experiment")
    cgroup = cgroup_path()
    verify_limits(cgroup, plan["unit"], 8)
    results = []
    record = {"scope": plan["scope"], "plan_sha256": sha(plan_path.read_bytes()),
              "cgroup": str(cgroup), "accepted": False, "results": results,
              "global_caches_flushed": False, "true_hot_full_request_implemented": False,
              "production_limits_changed": False}
    try:
        for index, cap in enumerate(ABBA, 1):
            transition_before = snapshot(cgroup)
            # Only this declared, currently enclosing, public unit can be changed.
            if cgroup.name != plan["unit"] or not re.fullmatch(UNIT_PATTERN, plan["unit"]):
                raise ValueError("Refusing to change a different service")
            subprocess.run(["systemctl", "--user", "set-property", "--runtime", plan["unit"],
                            f"MemoryMax={cap}G"], check=True, stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL)
            verify_limits(cgroup, plan["unit"], cap)
            transition_after = snapshot(cgroup)
            print(f"public_abba_sample={index} cap_gib={cap}", flush=True)
            result = run_child(plan, plan_path, index, cap, cgroup)
            result["cap_transition_before"] = transition_before
            result["cap_transition_after"] = transition_after
            results.append(result)
        if model_identity(plan["model"]["path"]) != plan["model"]:
            raise ValueError("Original model changed during experiment")
        record["accepted"] = True
    except (ValueError, OSError, subprocess.SubprocessError):
        record["failure_class"] = "limits-or-worker-or-identity"
        raise
    finally:
        write_new(plan_path.parent / "abba-receipt.json", record)


def launch(args):
    if not re.fullmatch(r"[a-z0-9-]{1,40}", args.run_id):
        raise ValueError("run-id must be1..40 lowercase letters, digits or hyphens")
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
    plan = {"scope": "fixed-public-native-cgroup-abba-only", "caps_gib": list(ABBA),
            "card": args.card, "unit": "kuno-public-performance-" + args.run_id + ".service",
            "model": model_identity(args.model), "executable": str(frozen),
            "executable_sha256": identity, "cargo_artifacts_sha256": artifacts_sha,
            "cargo_executable_at_build": str(original_executable),
            "fixture_sha256": sha((HERE / "qwen38-baseline.json").read_bytes()),
            "reference_sha256": sha((HERE / "receipts/rust-full-resident-rainwater-intel-20261005.json").read_bytes()),
            "supervisor_sha256": sha(pathlib.Path(__file__).read_bytes()),
            "source_head_at_freeze": subprocess.check_output(
                ["git", "-C", str(HERE), "rev-parse", "HEAD"], text=True).strip(),
            "fixture": "resident-rainwater", "prompt_tokens": 75, "output_tokens": 128,
            "seed": 424242, "temperature": 0, "native_count": 8,
            "native_context": 203, "model_unloaded_before_each_child": True,
            "private_main_pages": 1376, "private_main_bytes": 818249728,
            "runtime_io_cap_bytes": 67108864}
    plan_path = output / "plan.json"
    write_new(plan_path, plan)
    command = ["systemd-run", "--user", "--unit=" + plan["unit"], "--collect", "--wait", "--pipe",
               "--property=MemorySwapMax=0", "--property=LimitCORE=0", "--property=MemoryMax=8G",
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
        if len(sys.argv) != 4:
            raise ValueError("Invalid guard arguments")
        guard(sys.argv[2], int(sys.argv[3]))
    else:
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("--cargo-artifacts", type=pathlib.Path, required=True)
        parser.add_argument("--model", type=pathlib.Path, required=True)
        parser.add_argument("--card", choices=("intel", "nvidia"), required=True)
        parser.add_argument("--run-id", required=True)
        parser.add_argument("--output", type=pathlib.Path, required=True)
        launch(parser.parse_args())


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError, subprocess.SubprocessError):
        raise SystemExit("Public performance experiment failed; no acceptance claim") from None
