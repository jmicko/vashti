#!/usr/bin/env python3
"""Development-only synthetic Q6_K records; compile pinned C, never a C engine."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import struct
import subprocess
import tempfile

PIN = "b11232"
FILES = {
    "ggml-common.h": "0061131b615c5721fc88a78feeb22c1f8c450f1c2646a317d80796a653bf595c",
    "ggml-quants.c": "7878680cc60493f98469a116e9b14af8b84789292ccf891c249230b2fa3d157f",
}
HALVES = [0, 0x8000, 1, 0x03FF, 0x0400, 0x3C00, 0x3555, 0xBC00, 0x7BFF, 0x7C00, 0xFC00, 0x7E01]


def function(source, signature):
    start = source.index(signature)
    brace = source.index("{", start)
    end = brace + 1
    depth = 1
    while depth:
        depth += (source[end] == "{") - (source[end] == "}")
        end += 1
    return source[start:end]


def block(index, half):
    quants = [(index * 17 + column * 5) % 64 for column in range(256)]
    low = bytearray(128)
    high = bytearray(64)
    for h in range(2):
        for lane in range(32):
            q = [quants[h * 128 + quarter * 32 + lane] for quarter in range(4)]
            low[h * 64 + lane] = (q[0] & 15) | ((q[2] & 15) << 4)
            low[h * 64 + 32 + lane] = (q[1] & 15) | ((q[3] & 15) << 4)
            high[h * 32 + lane] = sum((q[quarter] >> 4) << (2 * quarter) for quarter in range(4))
    scales = bytes((index * 16 + group) % 256 for group in range(16))
    return bytes(low + high) + scales + struct.pack("<H", half)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, required=True)
    args = parser.parse_args()
    sources = {}
    for name, digest in FILES.items():
        data = (args.source_dir / name).read_bytes()
        if hashlib.sha256(data).hexdigest() != digest:
            raise ValueError("pinned GGML source digest")
        sources[name] = data.decode()
    common, quants = sources["ggml-common.h"], sources["ggml-quants.c"]
    definition = re.search(r"typedef struct \{[^{}]+\} block_q6_K;", common).group()
    decoder = function(quants, "void dequantize_row_q6_K(")
    source = r'''
#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <math.h>
#define QK_K 256
#define GGML_RESTRICT restrict
typedef uint16_t ggml_half;
static float half_value(uint16_t b) {
    int e=(b>>10)&31; float f=b&1023;
    float x=e==0?ldexpf(f,-24):e==31?(f==0?INFINITY:NAN):ldexpf(1.f+f/1024.f,e-15);
    return b&0x8000?-x:x;
}
#define GGML_FP16_TO_FP32(bits) half_value(bits)
''' + definition + "\n" + decoder + r'''
int main(void) {
    block_q6_K b; float out[256];
    uint16_t endian=1;
    if (*(uint8_t*)&endian!=1 || sizeof(b)!=210) return 2;
    while(fread(&b,sizeof(b),1,stdin)==1) {
        dequantize_row_q6_K(&b,out,256);
        if(fwrite(out,sizeof(float),256,stdout)!=256) return 3;
    }
    return ferror(stdin)?4:0;
}
'''
    blocks = [block(i, 0x1800) for i in range(16)] + [block(16+i, half) for i, half in enumerate(HALVES)]
    assert all(len(b) == 210 for b in blocks)
    with tempfile.TemporaryDirectory(prefix="kuno-public-q6-c-") as temporary:
        root = Path(temporary)
        (root / "reference.c").write_text(source)
        subprocess.run([os.environ.get("CC", "cc"), "-std=c11", "-O2", "-ffp-contract=off", str(root / "reference.c"), "-lm", "-o", str(root / "reference")], check=True)
        decoded = subprocess.run([str(root / "reference")], input=b"".join(blocks), capture_output=True, check=True).stdout
    assert len(decoded) == len(blocks) * 256 * 4
    fixture = b"".join(b + decoded[i * 1024:(i + 1) * 1024] for i, b in enumerate(blocks))
    directory = Path(__file__).resolve().parent
    (directory / "q6_k.bin").write_bytes(fixture)
    receipt = {"reference": "https://github.com/ggml-org/llama.cpp/tree/" + PIN, "source_sha256": FILES,
               "scope": "synthetic-public-original-Q6_K-development-reference-only", "ordinary_blocks": 16,
               "fixture": {"gguf_id": 14, "block_elements": 256, "block_bytes": 210, "blocks": len(blocks),
                           "bytes": len(fixture), "sha256": hashlib.sha256(fixture).hexdigest()},
               "nonfinite_half_blocks": 3, "all_signed_i8_subscales_covered": True, "c_runtime_linked": False}
    (directory / "receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
