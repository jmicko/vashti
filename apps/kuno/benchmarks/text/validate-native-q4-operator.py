"""Validate the current fixed-public Q4 operator receipt, without legacy fallback.

This checks the complete reported fixture, ownership and cleanup contract. The
producer checks per-element absolute/relative tolerances before reporting; its
maxima omit reference magnitudes and cannot reconstruct those relative checks.
Card, model/manifest, binary and cgroup identity belong to the caller's run proof.
This receipt establishes neither full-model nor performance acceptance.
"""

import argparse
import json
import math
import pathlib
import sys


RECEIPT_MAX_BYTES = 131072
MIB = 1024**2
F32_MAX = 3.4028234663852886e38
FORMAT_IDS = (0, 1, 30, 12, 6, 14, 8)
GROUPED_PAGES = 1671168
COMBINED_PAGES = 1998848
COMPACT_PAGES = 425984
MAIN_ERASED_BYTES = 64 * COMBINED_PAGES + 12 * GROUPED_PAGES + 14 * COMPACT_PAGES
MAIN_ERASED_BUFFERS = 64 * 14 + 12 * 7 + 14 * 7
HOST_BOUND = (4 * GROUPED_PAGES + 80 * 640 * 4 + 80 * 2560 * 4
              + 2 * 8 * 2560 * 4 + 2560 * 640 * 4 + 1740800 + 327680 + MIB)
BANK_PAGES_MIB = {6: {16: 112, 64: 304, 96: 432},
                  8: {16: 128, 64: 320, 96: 448}}


def reject():
    raise ValueError("Fixed-public Q4 operator receipt rejected")


def exact_object(value, keys):
    if type(value) is not dict or set(value) != set(keys):
        reject()


def equal(value, expected):
    if type(value) is not type(expected) or value != expected:
        reject()


def error_maximum(value):
    if type(value) not in (int, float) or value < 0:
        reject()
    try:
        finite = math.isfinite(value)
    except OverflowError:
        reject()
    # This is the producer's f32 representation bound, not a new tolerance.
    if not finite or value > F32_MAX:
        reject()


def case_inventory():
    """Source-defined order and policy, independent of the receipt's choices."""
    for count in (1, 3, 5, 8):
        for overlap in (True, False):
            for tile in (4, 8):
                yield count, overlap, tile, 16, False, "generic"
    for capacity in (16, 64, 96):
        for policy in ("generic", "q4-block"):
            yield 1, False, 1, capacity, capacity == 96, policy
    for capacity in (64, 96):
        for overlap in (True, False):
            yield 8, overlap, 8, capacity, capacity == 96 and overlap, "generic"


def validate_receipt(receipt, *, expected_layers):
    # The caller pins Q5/Q8 layers from the declared original model/manifest.
    if (type(expected_layers) is not tuple or len(expected_layers) != 2
            or any(type(layer) is not int or not 0 <= layer < 48
                   for layer in expected_layers)
            or expected_layers[0] == expected_layers[1]):
        reject()
    expected = {
        "scope": "fixed-public-grouped-routed-MoE-operator-only",
        "failure_cases": 12,
        "q4_block_cases": 6,
        "generic_tile1_cases": 6,
        "main_limit_bytes": 1024 * MIB,
        "private_limit_bytes": 2 * MIB,
        "combined_private_pages": COMBINED_PAGES,
        "io_cap_bytes": 64 * MIB,
        "io_checkpoint_bound": 8 * MIB,
        "host_scratch_upper_bound": HOST_BOUND,
        "main_erased_bytes": MAIN_ERASED_BYTES,
        "main_erased_buffers": MAIN_ERASED_BUFFERS,
        "full_model_96_nvidia_rejected": True,
        "all_ten_original_ranks_and_probabilities": True,
        "no_original_ledger_publication": True,
        "shared_expert_or_FFN_or_full_graph_accepted": False,
        "performance_or_production_private_worker_accepted": False,
        "private_register_spill_driver_erasure_accepted": False,
        "context_retired": False,
    }
    exact_object(receipt, {*expected, "cases", "all_format_single_member_cases"})
    for key, wanted in expected.items():
        equal(receipt[key], wanted)
    cases = receipt["cases"]
    if type(cases) is not list or len(cases) != 52:
        reject()
    index = 0
    for dtype, layer in zip((6, 8), expected_layers):
        for count, overlap, tile, capacity, boundary, policy in case_inventory():
            result = cases[index]
            index += 1
            pinned = {
                "count": count, "overlap": overlap, "tile": tile,
                "capacity": capacity, "cross_segment_boundary": boundary,
                "gate_policy": policy, "down_dtype": dtype, "layer": layer,
                "bank_full_pages": BANK_PAGES_MIB[dtype][capacity] * MIB,
                "all_ten_ranks": True, "all_full_guards_and_inactive_tails": True,
                "original_bank_ledger_unchanged": True,
                "operator_only_no_publication": True,
            }
            exact_object(result, {*pinned, "tree_activation_partial_output_max_abs_errors",
                                  "core_output_max_abs_error", "scalar_max_abs_error",
                                  "generic_max_abs_error"})
            for key, wanted in pinned.items():
                equal(result[key], wanted)
            errors = result["tree_activation_partial_output_max_abs_errors"]
            if type(errors) is not list or len(errors) != 3:
                reject()
            for error in (*errors, result["core_output_max_abs_error"],
                          result["scalar_max_abs_error"]):
                error_maximum(error)
            # Policy comes from this fixed schedule, never from the receipt.
            if policy == "q4-block":
                error_maximum(result["generic_max_abs_error"])
            else:
                equal(result["generic_max_abs_error"], None)
    compact = receipt["all_format_single_member_cases"]
    if type(compact) is not list or len(compact) != 14:
        reject()
    for index, (dtype, tile) in enumerate((d, t) for d in FORMAT_IDS for t in (8, 1)):
        result = compact[index]
        pinned = {"dtype": dtype, "tile": tile, "count": 1, "groups": 10,
                  "private_pages_bytes": COMPACT_PAGES, "all_ten_ranks": True,
                  "full_offset_tail_status_and_erasure_checks": True}
        exact_object(result, {*pinned, "max_abs_activation_error", "max_abs_partial_error",
                              "max_abs_rank_merge_error"})
        for key, wanted in pinned.items():
            equal(result[key], wanted)
        for key in ("max_abs_activation_error", "max_abs_partial_error",
                    "max_abs_rank_merge_error"):
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
    parser.add_argument("--q5-layer", type=int, choices=range(48), required=True)
    parser.add_argument("--q8-layer", type=int, choices=range(48), required=True)
    args = parser.parse_args()
    try:
        validate_receipt(load_receipt(args.receipt),
                         expected_layers=(args.q5_layer, args.q8_layer))
    except (OSError, ValueError, TypeError, OverflowError, RecursionError):
        print("Fixed-public Q4 operator receipt rejected", file=sys.stderr)
        return 1
    print("validated_public_q4_operator cases=52 q4_block_cases=6 formats=14 "
          "whole_model_accepted=false performance_accepted=false")
    return 0


if __name__ == "__main__":
    sys.exit(main())
