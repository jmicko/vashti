"""Validate aggregate-only fixed-public numeric boundaries, never inference content.

Successful validation establishes this diagnostic's reported ownership/budgets
and statistics. It does not establish whole-model numerical or speed acceptance,
and cannot independently establish the producing binary or cgroup identity.
"""

import argparse
import json
import math
import pathlib
import sys


BOUNDARIES = (
    ("layer0_combined_ple_input", 10240, 65536, 524288),
    ("ple_projected_key", 10240, 65536, 524288),
    ("ple_projected_value", 2560, 32768, 131072),
    ("ple_injected_residual", 10240, 65536, 524288),
)
SOURCE_BYTES = 17 * 32768
RUNTIME_EXTRA = 1114112
ORIGINAL_RUNTIME = (64090112, 64000000)
RECEIPT_MAX_BYTES = 65536


def reject():
    # Errors contain no unknown keys, raw values, file data or backend payloads.
    raise ValueError("Fixed-public numeric boundary receipt rejected")


def exact_object(value, keys):
    if type(value) is not dict or set(value) != set(keys):
        reject()


def equal(value, expected):
    if type(value) is not type(expected) or value != expected:
        reject()


def audit(value, native):
    expected = {
        "main_bytes": 818249728 if native else 785186816,
        "main_buffers": 1376 if native else 1286,
        "source_bytes": 0 if native else SOURCE_BYTES,
        "source_buffers": 0 if native else 15,
    }
    exact_object(value, expected)
    for key, wanted in expected.items():
        equal(value[key], wanted)


def validate_receipt(receipt, card):
    if card not in ("intel", "nvidia"):
        reject()
    expected = {
        "fixture": "public_native_numeric_boundaries_v1",
        "position": 0,
        "count": 1,
        "pairs": 4,
        "owners_serialized": 8,
        "extra_main_bytes": 0,
        "guarded_host_bound": 610304,
        "guarded_host_cap": 1048576,
        "extra_audited_runtime_bytes": RUNTIME_EXTRA,
        "runtime_hard_cap": 67108864,
        "normal_family_acceptance": True,
        "explicit_full_cleanup": True,
        "whole_model_numeric_accepted": False,
        "performance_accepted": False,
        "vendor": 0x8086 if card == "intel" else 0x10DE,
        "weights": "borrowed-unmodified-ollama-blob",
        "fixed_public_input": "original tokenizer ! position0 count1",
        "seed": 424242,
    }
    exact_object(receipt, {*expected, "original_runtime_bytes", "combined_runtime_bytes",
                           "boundaries"})
    for key, wanted in expected.items():
        equal(receipt[key], wanted)
    original = receipt["original_runtime_bytes"]
    combined = receipt["combined_runtime_bytes"]
    if (type(original) is not int or original not in ORIGINAL_RUNTIME
            or type(combined) is not int or combined != original + RUNTIME_EXTRA
            or combined > receipt["runtime_hard_cap"]):
        reject()
    boundaries = receipt["boundaries"]
    if type(boundaries) is not list or len(boundaries) != len(BOUNDARIES):
        reject()
    for result, (name, width, scalar_map, native_map) in zip(boundaries, BOUNDARIES):
        exact_object(result, ("boundary", "logical_values", "scalar_owned_map_bytes",
                              "native_owned_map_bytes", "bit_mismatches", "max_absolute",
                              "bit_equal", "scalar_cleanup", "native_cleanup"))
        for key, wanted in {"boundary": name, "logical_values": width,
                            "scalar_owned_map_bytes": scalar_map,
                            "native_owned_map_bytes": native_map}.items():
            equal(result[key], wanted)
        mismatches, absolute = result["bit_mismatches"], result["max_absolute"]
        if (type(mismatches) is not int or not 0 <= mismatches <= width
                or type(absolute) not in (int, float)
                or absolute < 0):
            reject()
        try:
            finite = math.isfinite(absolute)
        except OverflowError:
            reject()
        if not finite:
            reject()
        equal(result["bit_equal"], mismatches == 0)
        if mismatches == 0 and absolute != 0:
            reject()
        # Signed zero is a legitimate bit mismatch with zero absolute distance.
        audit(result["scalar_cleanup"], False)
        audit(result["native_cleanup"], True)
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", type=pathlib.Path, required=True)
    parser.add_argument("--card", choices=("intel", "nvidia"), required=True)
    args = parser.parse_args()
    try:
        validate_receipt(load_receipt(args.receipt), args.card)
    except (OSError, ValueError, TypeError, OverflowError, RecursionError):
        print("Fixed-public numeric boundary receipt rejected", file=sys.stderr)
        return 1
    print("validated_public_numeric_boundaries pairs=4 owners=8 "
          "whole_model_numeric_accepted=false performance_accepted=false")
    return 0


if __name__ == "__main__":
    sys.exit(main())
