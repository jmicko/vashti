"""Strict acceptance for the fixed eight-case staged CPU/GPU numeric receipt.

This checks the reported public fixture, budgets and complete cleanup contract.
The producer checks each element at the unchanged absolute/relative tolerances;
aggregate maxima omit reference magnitudes and cannot reconstruct those checks.
The caller independently pins the original model's two layer IDs and card, and
joins model/artifact/runtime/cgroup proofs. No timing, cache publication,
whole-model or production authority follows from this operator receipt.
"""

import argparse
import json
import math
import pathlib
import sys


RECEIPT_MAX_BYTES = 65536
MIB = 1024**2
F32_MAX = 3.4028234663852886e38
VENDORS = {"intel": 0x8086, "nvidia": 0x10DE}


def reject():
    raise ValueError("Fixed-public staged hybrid numeric receipt rejected")


def exact_object(value, keys):
    if type(value) is not dict or set(value) != set(keys):
        reject()


def equal(value, expected):
    if type(value) is not type(expected) or value != expected:
        reject()


def bounded_integer(value, limit):
    if type(value) is not int or not 0 < value <= limit:
        reject()


def error_maximum(value):
    if type(value) not in (int, float) or value < 0:
        reject()
    try:
        finite = math.isfinite(value)
    except OverflowError:
        reject()
    # A representation bound on a reported f32, not a replacement tolerance.
    if not finite or value > F32_MAX:
        reject()


def validate_receipt(receipt, *, expected_layers, expected_vendor):
    if (type(expected_layers) is not tuple or len(expected_layers) != 2
            or any(type(layer) is not int or not 0 <= layer < 48
                   for layer in expected_layers)
            or expected_layers[0] == expected_layers[1]
            or type(expected_vendor) is not int
            or expected_vendor not in VENDORS.values()):
        reject()
    expected = {
        "scope": "fixed-public-original-staged-cpu-gpu-expert-numeric",
        "vendor": expected_vendor,
        "weights": "borrowed-unmodified-ollama-blob",
        "source_kind": "original-packed-gguf",
        "fixed_public_cases": 8,
        "purpose": "numeric",
        "ranks_per_case": 10,
        "group": 0,
        "vector": 1,
        "backend": "avx2",
        "workers": 16,
        "bank_slots_per_owner": 16,
        "private_pages_per_case": 18,
        "private_bytes_per_case": 884736,
        "bank_bytes_per_case": 128 * MIB,
        "physical_limit_bytes": 256 * MIB,
        "source_window_limit_bytes": 40 * MIB,
        "runtime_io_limit_bytes": 64 * MIB,
        "runtime_io_component_limit_bytes": 8 * MIB,
        "private_host_limit_bytes": 4 * MIB,
        "cpu_private_host_limit_bytes": MIB,
        "public_cpu_packed_limit_bytes": 40 * MIB,
        "bridge_host_limit_bytes": MIB,
        "oracle_host_limit_bytes": 2 * MIB,
        "retained_plan_limit_bytes": 256 * 1024,
        "operator_numeric_accepted": True,
        "owned_cleanup_accepted": True,
        "context_retired": False,
        "cache_publication_accepted": False,
        "whole_model_numeric_accepted": False,
        "timing_accepted": False,
        "performance_accepted": False,
        "production_accepted": False,
        "automatic_split_policy_enabled": False,
    }
    exact_object(receipt, {*expected, "cases"})
    for key, wanted in expected.items():
        equal(receipt[key], wanted)
    cases = receipt["cases"]
    if type(cases) is not list or len(cases) != 8:
        reject()
    # Independent declared order. Never select a lane/config from the receipt.
    for index, result in enumerate(cases):
        position = index % 4
        cpu_misses = (0, 1, 5, 10)[position]
        misses = (5, 5, 5, 10)[position]
        gpu_misses = misses - cpu_misses
        pinned = {
            "layer": expected_layers[index // 4],
            "down_dtype": (6, 8)[index // 4],
            "workers": 16,
            "backend": "avx2",
            "mode": "concurrent",
            "ranks": 10,
            "misses": misses,
            "cpu_misses": cpu_misses,
            "gpu_misses": gpu_misses,
            "group": 0,
            "vector": 1,
            "private_pages": 18,
            "private_bytes": 884736,
            "bank_bytes": 128 * MIB,
            "source_precharge": gpu_misses * 4 * MIB,
            "zero_buffers": 18,
            "zero_bytes": 884736,
            "operator_numeric_accepted": True,
            "owned_cleanup_accepted": True,
        }
        errors = ("max_tree_error", "max_legacy_error", "max_core_error")
        bounds = {"io_inspected_bytes": 4 * MIB,
                  "metadata_bytes": 256 * 1024, "snapshot_bytes": 2 * MIB}
        exact_object(result, {*pinned, *errors, *bounds})
        for key, wanted in pinned.items():
            equal(result[key], wanted)
        for key, limit in bounds.items():
            bounded_integer(result[key], limit)
        for key in errors:
            error_maximum(result[key])
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
    parser.add_argument("--vendor", choices=VENDORS, required=True)
    parser.add_argument("--q5-layer", type=int, choices=range(48), required=True)
    parser.add_argument("--q8-layer", type=int, choices=range(48), required=True)
    args = parser.parse_args()
    try:
        validate_receipt(load_receipt(args.receipt),
                         expected_layers=(args.q5_layer, args.q8_layer),
                         expected_vendor=VENDORS[args.vendor])
    except (OSError, ValueError, TypeError, OverflowError, RecursionError):
        print("Fixed-public staged hybrid numeric receipt rejected", file=sys.stderr)
        return 1
    print("validated_public_staged_hybrid_numeric cases=8 owned_cleanup=true "
          "cache_publication_accepted=false timing_accepted=false "
          "whole_model_accepted=false performance_accepted=false production_accepted=false")
    return 0


if __name__ == "__main__":
    sys.exit(main())
