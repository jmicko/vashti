"""Strict acceptance for the fixed 91-case public packed dense numeric receipt.

The caller independently declares the original GGUF's seven bounded shapes and
Intel/NVIDIA vendor, then joins model, artifact, adapter and runtime run proofs.
The producer does not include vendor in this receipt. Its CPU-tree/independent
f64 and GPU/CPU/paired checks use 1e-4 + 3e-4 * abs(reference) per element;
reported maxima omit reference magnitudes and cannot reconstruct that gate.
Reported guards cover exact inactive/gap bits, full owning padding/erasure and
unchanged public cache. No timing, whole-model or performance authority follows.
"""

import argparse
import json
import math
import pathlib
import sys


RECEIPT_MAX_BYTES = 262144
F32_MAX = 3.4028234663852886e38
U64_MAX = 2**64 - 1
MAIN_LIMIT = 8 * 1024**2
IO_LIMIT = 32 * 1024**2
REPLAY_LIMIT = 32768
VENDORS = {"intel": 0x8086, "nvidia": 0x10DE}
FORMATS = ((0, 1, 4), (1, 1, 2), (30, 1, 2), (12, 256, 144),
           (6, 32, 22), (14, 256, 210), (8, 32, 34))
AUDIT_KEYS = ("epoch", "upload_bytes", "uniform_bytes", "readback_bytes",
              "inspected_bytes", "inspected_buffers")


def reject():
    raise ValueError("Fixed-public single dense numeric receipt rejected")


def exact_object(value, keys):
    if type(value) is not dict or set(value) != set(keys):
        reject()


def equal(value, expected):
    if type(value) is not type(expected) or value != expected:
        reject()


def integer(value, minimum, maximum):
    if type(value) is not int or not minimum <= value <= maximum:
        reject()


def error_maximum(value):
    if type(value) not in (int, float) or value < 0:
        reject()
    try:
        finite = math.isfinite(value)
    except OverflowError:
        reject()
    # Representation bound only; do not replace the producer's relative gate.
    if not finite or value > F32_MAX:
        reject()


def owned_bytes(words):
    return 1 << (max(32768, words * 4) - 1).bit_length()


def declared_shapes(expected_shapes):
    if type(expected_shapes) is not tuple or len(expected_shapes) != len(FORMATS):
        reject()
    shapes = []
    for shape, (dtype, block, block_bytes) in zip(expected_shapes, FORMATS):
        if type(shape) is not tuple or len(shape) != 3:
            reject()
        actual_dtype, rows, columns = shape
        equal(actual_dtype, dtype)
        integer(rows, 1, 7)
        integer(columns, 1, 65536)
        packed = rows * (columns // block) * block_bytes
        input_owned = owned_bytes(8 * (columns + 11) + 9)
        output_owned = owned_bytes(8 * (rows + 5) + 7)
        if (columns % block or packed > 512 * 1024
                or input_owned + output_owned > MAIN_LIMIT
                or output_owned > REPLAY_LIMIT):
            reject()
        # Original public packed prefix is twelve bytes for f32, three otherwise.
        wire = ((packed + (12 if dtype == 0 else 3) + 3) // 4) * 4
        cache_owned = 1 << (max(256, wire) - 1).bit_length()
        shapes.append((dtype, rows, columns, input_owned, output_owned, cache_owned))
    return shapes


def case_inventory():
    for count in (1, 3, 5, 8):
        for tile in (1, 4, 8):
            yield count, tile, False
        if count == 1:
            yield 1, 1, True


def audit(value, previous_epoch):
    exact_object(value, AUDIT_KEYS)
    # A successful ledger receipt first checks that the next epoch exists.
    integer(value["epoch"], 1, U64_MAX - 1)
    if value["epoch"] <= previous_epoch:
        reject()
    for key in AUDIT_KEYS[1:]:
        integer(value[key], 0, IO_LIMIT)
    total = sum(value[key] for key in ("upload_bytes", "uniform_bytes", "readback_bytes"))
    if (not 0 < total <= IO_LIMIT or value["inspected_bytes"] != total
            or not 0 < value["inspected_buffers"] <= total // 256
            or any(value[key] % 256 for key in AUDIT_KEYS[1:-1])):
        reject()
    return value["epoch"]


def validate_receipt(receipt, *, expected_shapes, expected_vendor):
    # Vendor is independent caller authority, not an absent receipt assertion.
    integer(expected_vendor, 0, 65535)
    if expected_vendor not in VENDORS.values():
        reject()
    shapes = declared_shapes(expected_shapes)
    pinned = {
        "scope": "fixed-public-synthetic-operator-test-only",
        "fixture": "packed-matmul-probe",
        "borrowed_original_gguf": True,
        "format_count": 7,
        "case_count": 91,
        "single_member_cases": 7,
        "private_uploads_owned": True,
        "context_retired": False,
        "operator_timings": None,
        "proof_scope": "owned API-visible main and I/O spans; not opaque physical driver padding",
    }
    exact_object(receipt, {*pinned, "cases"})
    for key, wanted in pinned.items():
        equal(receipt[key], wanted)
    cases = receipt["cases"]
    if type(cases) is not list or len(cases) != 91:
        reject()
    index = 0
    previous_epoch = 0
    for dtype, rows, columns, input_owned, output_owned, cache_owned in shapes:
        for count, tile, single in case_inventory():
            result = cases[index]
            index += 1
            fixed = {"dtype": dtype, "rows": rows, "columns": columns,
                     "count": count, "tile_tokens": tile,
                     "main_full_span_bytes": input_owned + output_owned,
                     "gaps_and_inactive_preserved": True, "main_erased": True,
                     "public_cache_preserved": True}
            if single:
                fixed.update(member_capacity=1, max8_prefill_reused=True)
            keys = {*fixed, "max_abs_error", "io", "main_audit", "cache_audit"}
            if single:
                keys.add("generic_max_abs_error")
            exact_object(result, keys)
            for key, wanted in fixed.items():
                equal(result[key], wanted)
            error_maximum(result["max_abs_error"])
            if single:
                error_maximum(result["generic_max_abs_error"])
            previous_epoch = audit(result["io"], previous_epoch)
            io = result["io"]
            if (any(io[key] == 0 for key in ("upload_bytes", "uniform_bytes", "readback_bytes"))
                    or io["readback_bytes"] != output_owned + io["upload_bytes"] + io["uniform_bytes"]
                    or io["inspected_buffers"] < 5 or io["inspected_buffers"] % 2 != 1):
                reject()
            # Two full erased MAIN reads, followed by one immutable public read.
            for name, buffers, readback in (("main_audit", 2, input_owned + output_owned),
                                            ("cache_audit", 1, cache_owned)):
                previous_epoch = audit(result[name], previous_epoch)
                for key, wanted in {"upload_bytes": 0, "uniform_bytes": 0,
                                    "readback_bytes": readback,
                                    "inspected_buffers": buffers}.items():
                    equal(result[name][key], wanted)
    return True


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            reject()
        result[key] = value
    return result


def load_receipt(path):
    with pathlib.Path(path).open("rb") as source:
        payload = source.read(RECEIPT_MAX_BYTES + 1)
    if len(payload) > RECEIPT_MAX_BYTES:
        reject()
    return json.loads(payload, object_pairs_hook=unique_object,
                      parse_constant=lambda _: reject())


def shape_argument(raw):
    try:
        fields = raw.split(":")
        if len(fields) != 3:
            raise ValueError
        return tuple(int(field) for field in fields)
    except ValueError as error:
        raise argparse.ArgumentTypeError("shape must be DTYPE:ROWS:COLUMNS") from error


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", type=pathlib.Path, required=True)
    parser.add_argument("--vendor", choices=VENDORS, required=True)
    parser.add_argument("--shape", type=shape_argument, action="append", required=True,
                        help="independent original shape, seven entries in dtype order 0,1,30,12,6,14,8")
    args = parser.parse_args()
    try:
        validate_receipt(load_receipt(args.receipt), expected_shapes=tuple(args.shape),
                         expected_vendor=VENDORS[args.vendor])
    except (OSError, ValueError, TypeError, OverflowError, RecursionError):
        print("Fixed-public single dense numeric receipt rejected", file=sys.stderr)
        return 1
    print(f"validated_public_single_dense vendor={args.vendor} cases=91 legacy_grid_cases=84 "
          "single_member_cases=7 whole_model_accepted=false performance_accepted=false")
    return 0


if __name__ == "__main__":
    sys.exit(main())
