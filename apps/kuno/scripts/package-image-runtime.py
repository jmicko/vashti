#!/usr/bin/env python3
"""Package an independently approved flat runtime without loading its code.

Developer-only Python stdlib helper. Installation still uses Kuno's root-owned
file validation. This does not download, enable or publish a runtime.
"""

import argparse
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import re
import stat
import tarfile
import tempfile


PINS = {
    "version": 1, "id": "qwen21-sycl-2026.1-v1", "worker_abi": 2,
    "ggml_max_name": 160,
    "engine_revision": "88411ef1e0688ff2df1010aeeb5d92b2d8cea2be",
    "ggml_revision": "4bf5f6000653b7881d00963cd6ddb665ccd62a8d",
    "engine_patch_sha256": "ac51db708fef026fe7abab850774309e4f1f439a05c56d06ff33e49fcc290584",
    "ggml_patch_sha256": "7298590faed44cad4733c78f6f6c2bf261b647712b5ec5f75840da450f0067dc",
    "sycl_patch_sha256": "7f6a75c75544c0e3776d780b0710cab2fabb94ddfe57eeb436dff02891bd4771",
}
HEX = re.compile(r"[0-9a-f]{64}\Z")
NAME = re.compile(r"[A-Za-z0-9_.-]{1,128}\Z")


def require(condition):
    if not condition:
        raise ValueError("runtime package verification failed")


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result)
        result[key] = value
    return result


def open_regular(path):
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    stream = os.fdopen(fd, "rb")
    if not stat.S_ISREG(os.fstat(fd).st_mode):
        stream.close()
        require(False)
    return stream


class CheckedReader:
    def __init__(self, stream):
        self.stream = stream
        self.hash = hashlib.sha256()
        self.count = 0

    def read(self, size):
        data = self.stream.read(size)
        self.hash.update(data)
        self.count += len(data)
        return data


def add_member(archive, name, size, stream, expected=None):
    info = tarfile.TarInfo(name)
    info.size = size
    info.mode = 0o755 if name == "worker" else 0o644
    reader = CheckedReader(stream)
    archive.addfile(info, reader)
    require(reader.count == size and not stream.read(1))
    if expected:
        require(reader.hash.hexdigest() == expected)


def package(directory, manifest_sha256, output):
    require(HEX.fullmatch(manifest_sha256) and directory.is_dir())
    with open_regular(directory / "manifest.json") as stream:
        raw = stream.read(65537)
    require(len(raw) <= 65536 and hashlib.sha256(raw).hexdigest() == manifest_sha256)
    manifest = json.loads(raw, object_pairs_hook=unique_object)
    require(set(manifest) == set(PINS) | {"files"})
    require(all(type(manifest[key]) is type(value) and manifest[key] == value
                for key, value in PINS.items()))
    entries = manifest["files"]
    require(isinstance(entries, list) and 2 <= len(entries) <= 96)
    names = set()
    total = 0
    for entry in entries:
        require(isinstance(entry, dict) and set(entry) == {"name", "bytes", "sha256"})
        name, size, digest = entry["name"], entry["bytes"], entry["sha256"]
        require(isinstance(name, str) and NAME.fullmatch(name) and ".." not in name)
        require(name == "worker" or (name.startswith("lib") and ".so" in name)
                or (name.startswith(("LICENSE-", "NOTICE-")) and name.endswith(".txt")))
        require(name not in names and type(size) is int and 0 < size <= 1024**3)
        require(isinstance(digest, str) and HEX.fullmatch(digest))
        names.add(name)
        total += size
    require("worker" in names and total <= 3 * 1024**3)
    require({path.name for path in directory.iterdir()} == names | {"manifest.json"})
    require(all(stat.S_ISREG((directory / name).lstat().st_mode) for name in names))
    with open_regular(directory / "worker") as stream:
        header = stream.read(20)
    require(len(header) == 20 and header[:6] == b"\x7fELF\x02\x01"
            and header[18:20] == b"\x3e\x00")
    require(not output.exists())
    output.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".kuno-runtime-", dir=output.parent)
    try:
        with os.fdopen(fd, "wb") as raw_output:
            with gzip.GzipFile(fileobj=raw_output, mode="wb", filename="", mtime=0) as zipped:
                with tarfile.open(fileobj=zipped, mode="w|", format=tarfile.PAX_FORMAT) as archive:
                    add_member(archive, "manifest.json", len(raw), io.BytesIO(raw))
                    for entry in sorted(entries, key=lambda entry: entry["name"]):
                        with open_regular(directory / entry["name"]) as stream:
                            require(os.fstat(stream.fileno()).st_size == entry["bytes"])
                            add_member(archive, entry["name"], entry["bytes"], stream, entry["sha256"])
            raw_output.flush()
            os.fsync(raw_output.fileno())
        # Atomic no-clobber creation of a local archive, not an external upload.
        os.link(temporary, output)
        output.chmod(0o644)
    finally:
        Path(temporary).unlink(missing_ok=True)
    digest = hashlib.sha256()
    with output.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return {"archive_sha256": digest.hexdigest(), "archive_bytes": output.stat().st_size,
            "manifest_sha256": manifest_sha256, "members": len(entries)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", required=True, type=Path)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    try:
        print(json.dumps(package(args.directory, args.manifest_sha256, args.output)))
    except (OSError, ValueError, TypeError, KeyError, tarfile.TarError):
        parser.exit(1, "Runtime package verification failed; no archive approved.\n")
