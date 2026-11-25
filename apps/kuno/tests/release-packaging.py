#!/usr/bin/env python3
"""CPU-only release archive regressions; no services, models or GPU code."""

import hashlib
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


class StandardPackageTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="kuno packaging ")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        scripts = self.root / "apps/kuno/scripts"
        scripts.mkdir(parents=True)
        shutil.copyfile(SCRIPTS / "package-release.sh", scripts / "package-release.sh")
        readme = self.root / "apps/kuno/packaging/README-install.txt"
        readme.parent.mkdir()
        readme.write_text("test setup instructions\n")
        (self.root / "LICENSE").write_text("test repository license\n")
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
                        VERSION="0.0.0-packaging-test",
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
            self.assertEqual(sorted(archive.getnames()),
                             ["LICENSE", "README-install.txt", "VERSION", "kuno"])
        expected = hashlib.sha256(self.archive.read_bytes()).hexdigest()
        self.assertIn(expected, (self.archive.parent / "SHA256SUMS").read_text())

    def test_failed_build_never_packages_stale_binary(self):
        self.env["FAIL_BUILD"] = "1"
        self.assertNotEqual(self.run_package().returncode, 0)
        self.assertFalse(self.archive.exists())


if __name__ == "__main__":
    unittest.main()
