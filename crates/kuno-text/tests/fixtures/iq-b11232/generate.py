#!/usr/bin/env python3
"""Regenerate public packed-IQ oracle fixtures using the pinned upstream C decoder.

Development-only: Python and a C compiler are not Kuno runtime dependencies.
The three dequantizers and their structs/tables are extracted verbatim from
SHA256-verified llama.cpp b11232 sources, then compiled in a temporary directory.
No Kuno decoding code is imported. Every fixture is synthetic public data.
"""
from pathlib import Path
from urllib.request import urlopen
import argparse
import hashlib
import json
import os
import re
import struct
import subprocess
import tempfile

PIN = "b11232"
FILES = {
    "ggml-common.h": "0061131b615c5721fc88a78feeb22c1f8c450f1c2646a317d80796a653bf595c",
    "ggml-quants.c": "7878680cc60493f98469a116e9b14af8b84789292ccf891c249230b2fa3d157f",
}
KINDS = [(18, "iq3_xxs", 98, 256), (20, "iq4_nl", 18, 32), (23, "iq4_xs", 136, 256)]
HALVES = [0x0000, 0x8000, 0x0001, 0x03FF, 0x0400, 0x3C00, 0x3555, 0xBC00, 0x7BFF]


def get_source(name, directory):
    if directory:
        data = (directory / name).read_bytes()
    else:
        url = f"https://raw.githubusercontent.com/ggml-org/llama.cpp/{PIN}/ggml/src/{name}"
        data = urlopen(url, timeout=30).read()
    if hashlib.sha256(data).hexdigest() != FILES[name]:
        raise ValueError("pinned source digest mismatch")
    return data.decode()


def extract_function(source, name):
    start = source.index(f"void dequantize_row_{name}(")
    brace = source.index("{", start)
    depth = 1
    end = brace + 1
    while depth:
        depth += (source[end] == "{") - (source[end] == "}")
        end += 1
    return source[start:end]


def harness(common, quants):
    definitions = []
    for _, name, _, _ in KINDS:
        match = re.search(r"typedef struct \{[^{}]+\} block_" + name + r";", common)
        if match is None:
            raise ValueError("pinned struct absent")
        definitions.append(match.group())
    for name in ["kmask_iq2xs", "ksigns_iq2xs", "iq3xxs_grid", "kvalues_iq4nl"]:
        match = re.search(r"GGML_TABLE_BEGIN\([^\n]+, " + name + r", [^\n]+\).*?GGML_TABLE_END\(\)", common, re.S)
        if match is None:
            raise ValueError("pinned table absent")
        definitions.append(match.group())
    definitions.extend(extract_function(quants, name) for _, name, _, _ in KINDS)
    return r'''
#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#define QK_K 256
#define QK4_NL 32
#define GGML_RESTRICT restrict
#define GGML_TABLE_BEGIN(kind, name, size) static const kind name[size] = {
#define GGML_TABLE_END() };
typedef uint16_t ggml_half;
// Independent exact IEEE-half equations: every finite half is exactly F32.
static float half_value(uint16_t bits) {
    int e = (bits >> 10) & 31;
    float f = bits & 1023;
    float value = e == 0 ? ldexpf(f, -24) : e == 31 ? (f == 0 ? INFINITY : NAN) : ldexpf(1.f + f / 1024.f, e - 15);
    return bits & 0x8000 ? -value : value;
}
#define GGML_FP16_TO_FP32(bits) half_value(bits)
''' + "\n".join(definitions) + r'''
int main(int argc, char **argv) {
    uint16_t endian = 1;
    if (*(uint8_t *)&endian != 1 || argc != 3) return 2;
    int kind = atoi(argv[1]), blocks = atoi(argv[2]);
    int bytes = kind == 18 ? sizeof(block_iq3_xxs) : kind == 20 ? sizeof(block_iq4_nl) : sizeof(block_iq4_xs);
    int width = kind == 20 ? 32 : 256;
    if (blocks < 1 || blocks > 100 || bytes != (kind == 18 ? 98 : kind == 20 ? 18 : 136)) return 3;
    void *packed = malloc((size_t)blocks * bytes);
    float *out = malloc((size_t)blocks * width * sizeof(float));
    if (packed == NULL || out == NULL) return 4;
    if (fread(packed, bytes, blocks, stdin) != (size_t)blocks) return 5;
    if (kind == 18) dequantize_row_iq3_xxs(packed, out, (int64_t)blocks * width);
    else if (kind == 20) dequantize_row_iq4_nl(packed, out, (int64_t)blocks * width);
    else dequantize_row_iq4_xs(packed, out, (int64_t)blocks * width);
    if (fwrite(out, sizeof(float), (size_t)blocks * width, stdout) != (size_t)blocks * width) return 6;
    free(out); free(packed); return 0;
}
'''


def blocks_for(name):
    # First four IQ3 blocks cover every grid index and every parity sign code;
    # first eight IQ4_XS blocks cover every signed six-bit scale.
    first = {"iq3_xxs": 4, "iq4_nl": 2, "iq4_xs": 8}[name]
    result = []
    for b, half in enumerate([0x3555] * first + HALVES):
        if name == "iq3_xxs":
            grids = bytes((b * 64 + j) % 256 for j in range(64))
            words = []
            for group in range(8):
                word = ((b * 8 + group) % 16) << 28
                for octet in range(4):
                    word |= ((b * 32 + group * 4 + octet) % 128) << (7 * octet)
                words.append(word)
            payload = grids + struct.pack("<8I", *words)
        elif name == "iq4_nl":
            payload = bytes(((j + b * 3) % 16) | (((15 - j + b * 5) % 16) << 4) for j in range(16))
        else:
            scales = [(b * 8 + group) % 64 for group in range(8)]
            high = sum((scale >> 4) << (2 * group) for group, scale in enumerate(scales))
            low = bytes((scales[j * 2] & 15) | ((scales[j * 2 + 1] & 15) << 4) for j in range(4))
            codes = bytes(((j + b * 3) % 16) | (((15 - j + b * 5) % 16) << 4) for j in range(128))
            payload = struct.pack("<H", high) + low + codes
        result.append(struct.pack("<H", half) + payload)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path)
    args = parser.parse_args()
    common = get_source("ggml-common.h", args.source_dir)
    quants = get_source("ggml-quants.c", args.source_dir)
    destination = Path(__file__).resolve().parent
    receipt = {"reference": "https://github.com/ggml-org/llama.cpp/tree/" + PIN, "source_sha256": FILES, "fixtures": {}}
    with tempfile.TemporaryDirectory(prefix="kuno-public-iq-reference-") as directory:
        root = Path(directory)
        (root / "reference.c").write_text(harness(common, quants))
        subprocess.run([os.environ.get("CC", "cc"), "-std=c11", "-O2", "-ffp-contract=off", str(root / "reference.c"), "-lm", "-o", str(root / "reference")], check=True)
        for kind, name, packed_bytes, width in KINDS:
            blocks = blocks_for(name)
            assert all(len(block) == packed_bytes for block in blocks)
            outputs = subprocess.run([str(root / "reference"), str(kind), str(len(blocks))], input=b"".join(blocks), capture_output=True, check=True).stdout
            assert len(outputs) == len(blocks) * width * 4
            fixture = b"".join(block + outputs[b * width * 4:(b + 1) * width * 4] for b, block in enumerate(blocks))
            path = destination / (name + ".bin")
            path.write_bytes(fixture)
            receipt["fixtures"][path.name] = {"blocks": len(blocks), "packed_bytes": packed_bytes, "elements": width, "sha256": hashlib.sha256(fixture).hexdigest()}
    (destination / "receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
