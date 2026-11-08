#!/usr/bin/env python3
"""Regenerate public packed-IQ oracle fixtures using the pinned upstream C decoder.

Development-only: Python and a C compiler are not Kuno runtime dependencies.
The four dequantizers and their structs/tables are extracted verbatim from
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
KINDS = [(22, "iq2_s", 82, 256), (21, "iq3_s", 110, 256), (13, "q5_K", 176, 256), (42, "q2_0", 18, 64)]
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


def extract_function(source, signature):
    start = source.index(signature)
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
        # Q5_K has a nested anonymous union, so use its exact typedef boundary.
        end = common.index("} block_" + name + ";") + len("} block_" + name + ";")
        start = common.rfind("typedef struct {", 0, end)
        if start < 0:
            raise ValueError("pinned struct absent")
        definitions.append(common[start:end])
    for name in ["kmask_iq2xs", "iq2s_grid", "iq3s_grid"]:
        match = re.search(r"GGML_TABLE_BEGIN\([^\n]+, " + name + r", [^\n]+\).*?GGML_TABLE_END\(\)", common, re.S)
        if match is None:
            raise ValueError("pinned table absent")
        definitions.append(match.group())
    definitions.append(extract_function(quants, "static inline void get_scale_min_k4("))
    definitions.extend(extract_function(quants, "void dequantize_row_" + name + "(") for _, name, _, _ in KINDS)
    return r'''
#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#define QK_K 256
#define QK2_0 64
#define IQ3S_N_SCALE 4
#define K_SCALE_SIZE 12
#define GGML_EXTENSION
#define GGML_COMMON_AGGR_S
#define GGML_COMMON_AGGR_U
#define GGML_RESTRICT restrict
#define GGML_TABLE_BEGIN(kind, name, size) static const kind name[size] = {
#define GGML_TABLE_END() };
typedef uint16_t ggml_half;
typedef uint32_t ggml_half2;
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
    int bytes = kind == 22 ? sizeof(block_iq2_s) : kind == 21 ? sizeof(block_iq3_s) : kind == 13 ? sizeof(block_q5_K) : sizeof(block_q2_0);
    int width = kind == 42 ? 64 : 256;
    if (blocks < 1 || blocks > 100 || bytes != (kind == 22 ? 82 : kind == 21 ? 110 : kind == 13 ? 176 : 18)) return 3;
    void *packed = malloc((size_t)blocks * bytes);
    float *out = malloc((size_t)blocks * width * sizeof(float));
    if (packed == NULL || out == NULL) return 4;
    if (fread(packed, bytes, blocks, stdin) != (size_t)blocks) return 5;
    if (kind == 22) dequantize_row_iq2_s(packed, out, (int64_t)blocks * width);
    else if (kind == 21) dequantize_row_iq3_s(packed, out, (int64_t)blocks * width);
    else if (kind == 13) dequantize_row_q5_K(packed, out, (int64_t)blocks * width);
    else dequantize_row_q2_0(packed, out, (int64_t)blocks * width);
    if (fwrite(out, sizeof(float), (size_t)blocks * width, stdout) != (size_t)blocks * width) return 6;
    free(out); free(packed); return 0;
}
'''


def packed_scales(scales, mins):
    # Pack all eight independent 6-bit scale/min fields, including split bits.
    payload = bytearray(12)
    for j in range(4):
        payload[j] = scales[j] | ((scales[j + 4] >> 4) << 6)
        payload[j + 4] = mins[j] | ((mins[j + 4] >> 4) << 6)
        payload[j + 8] = (scales[j + 4] & 15) | ((mins[j + 4] & 15) << 4)
    return bytes(payload)


def blocks_for(name):
    # First 32 IQ2_S and eight IQ3_S blocks cover all1024/512 grid indices.
    # Stored sign bytes, scale nibbles, split Q5 scales/mins and codes vary.
    first = {"iq2_s": 32, "iq3_s": 8, "q5_K": 8, "q2_0": 4}[name]
    result = []
    for b, half in enumerate([0x3555] * first + HALVES):
        if name == "iq2_s":
            indices = [((b * 32 + j) * 37) % 1024 for j in range(32)]
            grids = bytes(index & 255 for index in indices)
            high = bytes(sum((indices[g * 4 + o] >> 8) << (2 * o) for o in range(4)) for g in range(8))
            signs = bytes((b * 32 + j) % 256 for j in range(32))
            scales = bytes(((b * 8 + g) % 16) | (((b * 8 + g + 7) % 16) << 4) for g in range(8))
            payload = grids + signs + high + scales
        elif name == "iq3_s":
            indices = [((b * 64 + j) * 29) % 512 for j in range(64)]
            grids = bytes(index & 255 for index in indices)
            high = bytes(sum((indices[g * 8 + j] >> 8) << j for j in range(8)) for g in range(8))
            signs = bytes((b * 32 + j) % 256 for j in range(32))
            scales = bytes(((b * 8 + g * 2) % 16) | (((b * 8 + g * 2 + 1) % 16) << 4) for g in range(4))
            payload = grids + high + signs + scales
        elif name == "q5_K":
            scales = [(b * 8 + j) % 64 for j in range(8)]
            mins = [(63 - b * 8 - j) % 64 for j in range(8)]
            quants = [(b * 3 + j * 7) % 32 for j in range(256)]
            high = bytes(sum((quants[g * 64 + side * 32 + lane] >> 4) << (2 * g + side) for g in range(4) for side in range(2)) for lane in range(32))
            low = bytes((quants[g * 64 + lane] & 15) | ((quants[g * 64 + 32 + lane] & 15) << 4) for g in range(4) for lane in range(32))
            # Ordinary records have finite varied dmin; edge records cover its
            # signed-zero and binary16 range independently of the main scale.
            dmin = 0x2C01 if b < first else HALVES[(b - first + 3) % len(HALVES)]
            payload = struct.pack("<H", dmin) + packed_scales(scales, mins) + high + low
        else:
            payload = bytes(sum(((b + byte * 3 + lane) % 4) << (2 * lane) for lane in range(4)) for byte in range(16))
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
    with tempfile.TemporaryDirectory(prefix="kuno-public-daslab-reference-") as directory:
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
            receipt["fixtures"][path.name] = {"gguf_id": kind, "blocks": len(blocks), "packed_bytes": packed_bytes, "elements": width, "sha256": hashlib.sha256(fixture).hexdigest()}
    (destination / "receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
