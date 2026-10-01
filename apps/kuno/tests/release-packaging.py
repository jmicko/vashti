#!/usr/bin/env python3
"""CPU-only release archive regressions; no services, models or GPU code."""

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import tempfile
import unittest


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.dont_write_bytecode = True


def load(name):
    spec = importlib.util.spec_from_file_location(name.replace("-", "_"), SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


runtime = load("package-image-runtime")
notices = load("collect-native-notices")


class RuntimePackageTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.bundle = self.root / "bundle"
        self.bundle.mkdir()
        worker = bytearray(20)
        worker[:6] = b"\x7fELF\x02\x01"
        worker[18:20] = b"\x3e\x00"
        (self.bundle / "worker").write_bytes(worker)
        (self.bundle / "libfixture.so.1").write_bytes(b"public test library")
        self.manifest = dict(runtime.PINS, files=[
            {"name": p.name, "bytes": p.stat().st_size,
             "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
            for p in sorted(self.bundle.iterdir())])
        self.approve()

    def approve(self):
        raw = json.dumps(self.manifest).encode()
        (self.bundle / "manifest.json").write_bytes(raw)
        self.checksum = hashlib.sha256(raw).hexdigest()

    def test_deterministic_regular_file_round_trip(self):
        first, second = self.root / "a.tar.gz", self.root / "b.tar.gz"
        runtime.package(self.bundle, self.checksum, first)
        runtime.package(self.bundle, self.checksum, second)
        self.assertEqual(first.read_bytes(), second.read_bytes())
        with tarfile.open(first) as archive:
            self.assertEqual(set(archive.getnames()), {p.name for p in self.bundle.iterdir()})
            for member in archive:
                self.assertTrue(member.isreg())
                self.assertEqual(member.uid, 0)
                self.assertEqual(member.mtime, 0)
                self.assertEqual(member.mode, 0o755 if member.name == "worker" else 0o644)
                self.assertEqual(archive.extractfile(member).read(), (self.bundle / member.name).read_bytes())

    def test_changed_content_leaves_no_archive_or_temporary(self):
        (self.bundle / "libfixture.so.1").write_bytes(b"PUBLIC test library")
        with self.assertRaises(ValueError):
            runtime.package(self.bundle, self.checksum, self.root / "bad.tar.gz")
        self.assertEqual({p.name for p in self.root.iterdir()}, {"bundle"})

    def test_symlink_and_fifo_rejected_without_blocking(self):
        member = self.bundle / "libfixture.so.1"
        member.unlink()
        member.symlink_to("worker")
        with self.assertRaises(ValueError):
            runtime.package(self.bundle, self.checksum, self.root / "bad.tar.gz")
        member.unlink()
        os.mkfifo(member)
        with self.assertRaises(ValueError):
            runtime.package(self.bundle, self.checksum, self.root / "bad.tar.gz")

    def test_existing_output_preserved(self):
        output = self.root / "existing.tar.gz"
        output.write_bytes(b"existing archive")
        with self.assertRaises(ValueError):
            runtime.package(self.bundle, self.checksum, output)
        self.assertEqual(output.read_bytes(), b"existing archive")

    def test_unapproved_or_incompatible_manifest_rejected(self):
        with self.assertRaises(ValueError):
            runtime.package(self.bundle, "0" * 64, self.root / "bad.tar.gz")
        self.manifest["ggml_max_name"] = 64
        self.approve()
        with self.assertRaises(ValueError):
            runtime.package(self.bundle, self.checksum, self.root / "bad.tar.gz")


class StandardPackageTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="kuno packaging ")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        scripts = self.root / "apps/kuno/scripts"
        scripts.mkdir(parents=True)
        for name in ("package-release.sh", "collect-native-notices.py"):
            shutil.copyfile(SCRIPTS / name, scripts / name)
        readme = self.root / "apps/kuno/packaging/README-install.txt"
        readme.parent.mkdir()
        readme.write_text("test setup instructions\n")
        (self.root / "LICENSE").write_text("test repository license\n")
        self.source = self.root / "native-source"
        for relative in notices.FILES.values():
            path = self.source / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("test notice\n")
        (self.source / "LICENSE").write_text("test copyright\nPermission is hereby granted test MIT text\n")
        (self.source / "thirdparty/json.hpp").write_text("// test JSON copyright\n// SPDX-License-Identifier: MIT\n")
        build = self.root / "native-build"
        build.mkdir()
        (build / "CMakeCache.txt").write_text("SD_CUDA:BOOL=OFF\n")
        stale = self.root / "target/release/kuno"
        stale.parent.mkdir(parents=True)
        self.elf = Path("/bin/true").read_bytes()
        stale.write_bytes(self.elf + b"STALE_PRIVATE_BINARY")
        mockbin = self.root / "mockbin"
        mockbin.mkdir()
        (mockbin / "rustc").write_text("#!/bin/sh\nprintf 'host: x86_64-unknown-linux-gnu\\n'\n")
        (mockbin / "cargo").write_text(f"#!{sys.executable}\n" + '''
import os, pathlib, sys
pathlib.Path("cargo-called").touch()
if os.environ.get("FAIL_BUILD"): sys.exit(1)
args = sys.argv
target_dir = args[args.index("--target-dir") + 1] if "--target-dir" in args else os.environ.get("CARGO_TARGET_DIR", "target")
target = args[args.index("--target") + 1] if "--target" in args else os.environ.get("CARGO_BUILD_TARGET", "")
out = pathlib.Path(target_dir) / target / "release/kuno"
out.parent.mkdir(parents=True, exist_ok=True)
out.write_bytes(pathlib.Path("/bin/true").read_bytes() + b"FRESH_STANDARD_BINARY")
''')
        for path in mockbin.iterdir():
            path.chmod(0o755)
        self.env = dict(os.environ, PATH=f"{mockbin}:{os.environ['PATH']}",
                        KUNO_DEV_SYCL_BUILD="0", KUNO_SD_SOURCE_DIR=str(self.source),
                        KUNO_SD_BUILD_DIR=str(build), VERSION="0.0.0-packaging-test",
                        CARGO_TARGET_DIR=str(self.root / "private target"),
                        CARGO_BUILD_TARGET="unrelated-private-target")
        self.archive = self.root / "apps/vashti/dist/release/v0.0.0-packaging-test/kuno-linux-x86_64.tar.gz"

    def run_package(self):
        return subprocess.run(["bash", "apps/kuno/scripts/package-release.sh"],
                              cwd=self.root, env=self.env, capture_output=True, timeout=30)

    def test_archive_uses_actual_build_with_inherited_private_targets(self):
        result = self.run_package()
        self.assertEqual(result.returncode, 0, result.stderr.decode())
        with tarfile.open(self.archive) as archive:
            self.assertEqual(archive.extractfile("kuno").read(), self.elf + b"FRESH_STANDARD_BINARY")
            self.assertIn("licenses/NOTICE-miniz-source.txt", archive.getnames())
            self.assertIn("licenses/LICENSE-json.txt", archive.getnames())
        expected = hashlib.sha256(self.archive.read_bytes()).hexdigest()
        self.assertIn(expected, (self.archive.parent / "SHA256SUMS").read_text())

    def test_failed_build_never_packages_stale_binary(self):
        self.env["FAIL_BUILD"] = "1"
        self.assertNotEqual(self.run_package().returncode, 0)
        self.assertFalse(self.archive.exists())

    def test_private_worker_gate_rejects_before_build(self):
        self.env["KUNO_DEV_SYCL_BUILD"] = "1"
        self.assertNotEqual(self.run_package().returncode, 0)
        self.assertFalse((self.root / "cargo-called").exists())
        self.assertFalse(self.archive.exists())


if __name__ == "__main__":
    unittest.main()
