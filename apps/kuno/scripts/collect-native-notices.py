#!/usr/bin/env python3
"""Collect pinned native notices; not a Rust/model license audit."""

import argparse
from pathlib import Path


FILES = {
    "LICENSE-stable-diffusion-cpp.txt": "LICENSE",
    "LICENSE-ggml.txt": "ggml/LICENSE",
    "LICENSE-oniguruma.txt": "thirdparty/oniguruma/COPYING",
    "LICENSE-utf8proc.txt": "thirdparty/utf8proc/LICENSE.md",
    "LICENSE-darts-clone.txt": "thirdparty/LICENSE.darts_clone.txt",
    "LICENSE-sageattention.txt": "ggml/src/ggml-cuda/sageattention/LICENSE",
    "NOTICE-miniz-source.txt": "thirdparty/miniz.h",
    "NOTICE-stb-image-source.txt": "thirdparty/stb_image.h",
    "NOTICE-stb-image-write-source.txt": "thirdparty/stb_image_write.h",
    "NOTICE-stb-image-resize-source.txt": "thirdparty/stb_image_resize.h",
    "NOTICE-zip-c-source.txt": "thirdparty/zip.c",
    "NOTICE-zip-h-source.txt": "thirdparty/zip.h",
    "NOTICE-httplib-source.txt": "thirdparty/httplib.h",
    "NOTICE-json-source.txt": "thirdparty/json.hpp",
}


def collect(source, output):
    # Retain complete embedded copyright/license blocks instead of selecting
    # individual lines. Read everything before writing an incomplete collection.
    contents = {name: (source / relative).read_bytes()
                for name, relative in FILES.items()}
    json_header = contents["NOTICE-json-source.txt"].split(b"\n", 8)[:8]
    if b"// SPDX-License-Identifier: MIT" not in json_header:
        raise ValueError("unexpected JSON license")
    mit = contents["LICENSE-stable-diffusion-cpp.txt"]
    permission = mit[mit.index(b"Permission is hereby granted"):]
    contents["LICENSE-json.txt"] = b"\n".join(json_header) + b"\n\n" + permission
    output.mkdir(parents=True, exist_ok=True)
    for name, content in contents.items():
        (output / name).write_bytes(content)
        (output / name).chmod(0o644)
    return len(contents)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    print(f"Collected {collect(args.source, args.output)} native notices")
