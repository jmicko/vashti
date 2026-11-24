#!/usr/bin/env python3
"""Assemble a managed image runtime directory from a built Kuno executable (its default
features carry the Rust image engine): the worker, Kuno's license and the manifest. It
never runs, installs or packages the runtime; `package-image-runtime.py` archives it and
`kuno install-image-runtime` verifies and installs it.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[3]
spec = importlib.util.spec_from_file_location(
    "package_image_runtime", Path(__file__).with_name("package-image-runtime.py"))
packaging = importlib.util.module_from_spec(spec)
spec.loader.exec_module(packaging)


def assemble(worker, output, worker_abi):
    if worker_abi not in packaging.WORKER_ABIS:
        raise ValueError("unsupported worker ABI")
    if output.exists():
        raise ValueError("choose a fresh output directory")
    output.mkdir(parents=True)
    shutil.copyfile(worker, output / "worker")
    shutil.copyfile(ROOT / "LICENSE", output / "LICENSE-kuno.txt")
    (output / "worker").chmod(0o755)
    files = [dict(name=name, bytes=(output / name).stat().st_size,
                  sha256=hashlib.sha256((output / name).read_bytes()).hexdigest())
             for name in ["LICENSE-kuno.txt", "worker"]]
    manifest = dict(packaging.PINS, worker_abi=worker_abi, files=files)
    raw = (json.dumps(manifest, indent=2) + "\n").encode()
    (output / "manifest.json").write_bytes(raw)
    return dict(directory=str(output), manifest_sha256=hashlib.sha256(raw).hexdigest(),
                worker_sha256=files[1]["sha256"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", required=True, type=Path, help="target/release/kuno")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--worker-abi", type=int, default=6)
    args = parser.parse_args()
    print(json.dumps(assemble(args.worker, args.output, args.worker_abi)))
