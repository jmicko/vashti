#!/usr/bin/env python3
"""Launch one frozen fixed-public resident trial with swap/core dumps disabled."""
import hashlib
import json
import pathlib
import re
import subprocess
import sys

base = pathlib.Path("/media/jmicko/faststore/kuno-development")
if len(sys.argv) != 5:
    raise SystemExit("usage: run-ornith-resident.py FROZEN_BUILD TAG scalar|block|vector[+trusted][+router][+alternate][+profile] probe|8|128|512|profile|teacher")
build = pathlib.Path(sys.argv[1]).resolve()
tag, selection, mode = sys.argv[2:]
if build.parent != base or not re.fullmatch(r"[a-z0-9-]{1,60}", tag):
    raise SystemExit("invalid public build/tag")
parts = selection.split("+")
if parts[0] not in ("scalar", "block", "vector") or any(x not in ("trusted", "profile", "router", "alternate") for x in parts[1:]) or mode not in ("probe", "8", "128", "512", "profile", "teacher"):
    raise SystemExit("invalid public selection")
if ("profile" in parts) != (mode == "profile"):
    raise SystemExit("instrumented profiles must be explicit")
executable = pathlib.Path((build / "executable-path.txt").read_text().strip()).resolve()
expected = (build / "sha256.txt").read_text().strip()
if executable.parent != build or hashlib.file_digest(executable.open("rb"), "sha256").hexdigest() != expected:
    raise SystemExit("frozen artifact digest mismatch")
output = base / "ornith-speed-20261008"
output.mkdir(exist_ok=True)
receipt, log = output / (tag + ".json"), output / (tag + ".log")
if receipt.exists():
    raise SystemExit("public receipt already exists")
log.touch(exist_ok=False)
options = {"MemoryMax": "48G", "MemorySwapMax": "0", "LimitCORE": "0", "RuntimeMaxSec": "300"}
command = ["systemd-run", "--user", "--unit=kuno-ornith-" + tag, "--service-type=exec"]
command += ["--property=" + k + "=" + v for k, v in options.items()]
command += ["--property=StandardOutput=append:" + str(log), "--property=StandardError=append:" + str(log), "--setenv=KUNO_LAB_ORNITH_PUBLIC=1"]
if parts[0] == "block":
    command += ["--setenv=KUNO_LAB_ORNITH_BLOCK_DOT=1"]
if parts[0] == "vector":
    command += ["--setenv=KUNO_LAB_ORNITH_VECTOR_DOT=1"]
if "trusted" in parts:
    command += ["--setenv=KUNO_LAB_ORNITH_TRUSTED_BOUNDS=1"]
if "router" in parts:
    command += ["--setenv=KUNO_LAB_ORNITH_PARALLEL_ROUTER=1"]
if "profile" in parts:
    command += ["--setenv=KUNO_LAB_ORNITH_PROFILE=1"]
if "alternate" in parts:
    command += ["--setenv=KUNO_LAB_ORNITH_ALTERNATE_PUBLIC=1"]
blob = pathlib.Path("/media/jmicko/faststore/ollama/blobs/sha256-6540411d9e4ca638983e905959188e1be74db7ff62151c69d71a6f2748584f68")
command += [str(executable), str(blob), str(receipt), mode]
with (output / (tag + ".launch.json")).open("x") as f:
    json.dump({"source_commit": (build / "source-commit.txt").read_text().strip(), "executable_sha256": expected, "selection": selection, "mode": mode, "cgroup_properties": options, "command": command}, f, indent=2)
subprocess.run(command, check=True)
