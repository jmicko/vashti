#!/usr/bin/env python3
"""Conserve actual routed work for the independently accepted XMX math variant.

The historical FMA count-stability proof remains unchanged and is not claimed
for a different floating-point reduction. These checks use the authenticated
fixed-public model geometry, route-membership histogram and real launch grammar.
"""
import json
import pathlib
import sys

BANDS = [("1_7", 1, 7), ("8_15", 8, 15), ("16_31", 16, 31),
         ("32_63", 32, 63), ("64_127", 64, 127),
         ("128_255", 128, 255), ("256_plus", 256, 4096)]


def count(profile, name):
    value = profile[name][0]
    assert isinstance(value, int) and not isinstance(value, bool) and value >= 0, name
    return value


def row(profile, metric):
    tokens = metric["prompt_tokens"] - 1
    assert tokens > 0
    groups = members = 0
    for suffix, lower, upper in BANDS:
        g = count(profile, "expert_groups_rows_" + suffix)
        m = count(profile, "expert_members_rows_" + suffix)
        assert g * lower <= m <= g * min(upper, tokens), suffix
        groups += g
        members += m
    assert groups > 0 and groups == count(profile, "expert_group_count")
    # Original authenticated model:48 layers,10 selected experts/token. This
    # expected inventory is independent of the number of nonempty groups.
    assert members == 48 * 10 * tokens, "lost_or_duplicated_original_routes"
    assert groups == count(profile, "expert_resident_groups") + count(profile, "expert_streamed_groups")
    batches = 48 * ((tokens + 4095) // 4096)
    assert count(profile, "expert_native_bulk_batches") == batches
    assert count(profile, "expert_native_bulk_logical_phases") == 6 * groups
    tails = count(profile, "expert_native_bulk_tail_pieces")
    prefixes = count(profile, "expert_native_bulk_wide_prefixes")
    assert 0 <= tails <= prefixes <= 2 * groups
    assert count(profile, "expert_native_bulk_matrix_pieces") == 2 * groups + tails
    assert count(profile, "expert_native_bulk_launches") == 6 * groups + tails
    assert count(profile, "expert_native_scratch_groups") == groups
    assert count(profile, "expert_native_scratch_slots") == 2 * batches
    return dict(groups=groups, original_route_members=members, batches=batches)


def validate(receipt, operators):
    for value in (receipt, operators):
        assert value["compiled_public_fixture_only"] and value["content_saved"] is False
        assert value["passed"] and value["physical_recovery_passed"]
    assert receipt["source_commit"] == operators["source_commit"]
    assert operators["operator_control_count"] == 129
    report = operators["reports"][0]
    assert report["closed"] and len(report["expert_scratch_image_checks"]) == 42
    assert len(report["qsa_math_checks"]) == 12
    xmx = report["qsa_xmx_checks"]
    assert len(xmx) == 7 and all(c["actual_large_xmx_kernel"] and c["numerical_acceptance_passed"] for c in xmx)
    positive = [c for c in xmx if c["control"] == "causal"]
    assert len(positive) == 5 and all(c["independent_f64"]["passed"]
        and c["independent_f64"]["both_gpu_paths_checked"]
        and c["independent_f64"]["bounds_from_operations_not_measurements"] for c in positive)
    assert [m["prompt_tokens"] for m in receipt["metrics"]] == [73, 4245, 4245]
    assert len(receipt["profiles"]) == 3
    rows = [row(p, m) for p, m in zip(receipt["profiles"], receipt["metrics"])]
    # Same seeded repeated long request, after original private erasure.
    assert rows[1] == rows[2], "repeated_routing_inventory_changed"
    return dict(passed=True, compiled_public_fixture_only=True, content_saved=False,
                source_commit=receipt["source_commit"], rows=rows,
                original_route_members_and_launch_grammar_preserved=True,
                original_fma_count_stability_not_claimed=True,
                original_scalar_and_f64_references_unchanged=True)


if __name__ == "__main__":
    assert len(sys.argv) == 3
    print(json.dumps(validate(*[json.loads(pathlib.Path(p).read_text()) for p in sys.argv[1:]])))
