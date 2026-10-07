"""Focused strict-schema, fixed-inventory and corruption acceptance tests."""

import copy
import importlib.util
import json
import pathlib
import subprocess
import sys
import tempfile
import unittest


SCRIPT = pathlib.Path(__file__).with_name("validate-native-q4-operator.py")
SPEC = importlib.util.spec_from_file_location("q4_operator_validator", SCRIPT)
VALIDATOR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(VALIDATOR)


def receipt():
    # Independently pin the public producer schedule and original layers.
    schedule = [(n, overlap, tile, 16, False, "generic")
                for n in (1, 3, 5, 8) for overlap in (True, False) for tile in (4, 8)]
    schedule += [(1, False, 1, cap, cap == 96, policy)
                 for cap in (16, 64, 96) for policy in ("generic", "q4-block")]
    schedule += [(8, overlap, 8, cap, cap == 96 and overlap, "generic")
                 for cap in (64, 96) for overlap in (True, False)]
    bank_pages = {6: {16: 112, 64: 304, 96: 432}, 8: {16: 128, 64: 320, 96: 448}}
    cases = []
    for dtype, layer in ((6, 6), (8, 0)):
        for n, overlap, tile, cap, boundary, policy in schedule:
            cases.append({
                "count": n, "overlap": overlap, "tile": tile, "capacity": cap,
                "cross_segment_boundary": boundary, "gate_policy": policy,
                "down_dtype": dtype, "layer": layer,
                "bank_full_pages": bank_pages[dtype][cap] * 1024**2,
                "tree_activation_partial_output_max_abs_errors": [0., 0., 0.],
                "core_output_max_abs_error": 0., "scalar_max_abs_error": 0.,
                "generic_max_abs_error": 0. if policy == "q4-block" else None,
                "all_ten_ranks": True, "all_full_guards_and_inactive_tails": True,
                "original_bank_ledger_unchanged": True, "operator_only_no_publication": True,
            })
    compact = [{
        "dtype": dtype, "tile": tile, "count": 1, "groups": 10,
        "max_abs_activation_error": 0., "max_abs_partial_error": 0.,
        "max_abs_rank_merge_error": 0., "private_pages_bytes": 425984,
        "all_ten_ranks": True, "full_offset_tail_status_and_erasure_checks": True,
    } for dtype in (0, 1, 30, 12, 6, 14, 8) for tile in (8, 1)]
    return {
        "scope": "fixed-public-grouped-routed-MoE-operator-only", "cases": cases,
        "failure_cases": 12, "q4_block_cases": 6, "generic_tile1_cases": 6,
        "all_format_single_member_cases": compact, "main_limit_bytes": 1073741824,
        "private_limit_bytes": 2097152, "combined_private_pages": 1998848,
        "io_cap_bytes": 67108864, "io_checkpoint_bound": 8388608,
        "host_scratch_upper_bound": 17543168, "main_erased_bytes": 153944064,
        "main_erased_buffers": 1078, "full_model_96_nvidia_rejected": True,
        "all_ten_original_ranks_and_probabilities": True,
        "no_original_ledger_publication": True,
        "shared_expert_or_FFN_or_full_graph_accepted": False,
        "performance_or_production_private_worker_accepted": False,
        "private_register_spill_driver_erasure_accepted": False, "context_retired": False,
    }


class Q4Operator(unittest.TestCase):
    def rejected(self, value, layers=(6, 0)):
        with self.assertRaisesRegex(ValueError, "Fixed-public Q4 operator receipt rejected"):
            VALIDATOR.validate_receipt(value, expected_layers=layers)

    def test_exact_current_inventory_and_nonzero_reported_error_are_accepted(self):
        value = receipt()
        self.assertTrue(VALIDATOR.validate_receipt(value, expected_layers=(6, 0)))
        value["cases"][17]["generic_max_abs_error"] = 1e-6
        value["cases"][17]["core_output_max_abs_error"] = 1e-5
        value["cases"][17]["tree_activation_partial_output_max_abs_errors"][2] = 1e-6
        self.assertTrue(VALIDATOR.validate_receipt(value, expected_layers=(6, 0)))
        self.assertEqual((VALIDATOR.MAIN_ERASED_BYTES, VALIDATOR.MAIN_ERASED_BUFFERS),
                         (153944064, 1078))
        self.assertEqual(VALIDATOR.HOST_BOUND, 17543168)

    def test_every_top_level_identity_limit_audit_flag_and_type_is_strict(self):
        for key, current in receipt().items():
            if type(current) is list:
                continue
            value = receipt()
            value[key] = (not current if type(current) is bool else
                          current + 1 if type(current) is int else "wrong")
            self.rejected(value)
            value = receipt()
            del value[key]
            self.rejected(value)
            if type(current) in (int, bool):
                value = receipt()
                value[key] = int(current) if type(current) is bool else float(current)
                self.rejected(value)
        value = receipt()
        value["unexpected"] = True
        self.rejected(value)

    def test_every_routed_case_public_field_and_guard_is_checked(self):
        fields = ("count", "overlap", "tile", "capacity", "cross_segment_boundary",
                  "gate_policy", "down_dtype", "layer", "bank_full_pages", "all_ten_ranks",
                  "all_full_guards_and_inactive_tails", "original_bank_ledger_unchanged",
                  "operator_only_no_publication")
        for index in range(52):
            for key in fields:
                value = receipt()
                current = value["cases"][index][key]
                value["cases"][index][key] = (int(current) if type(current) is bool else
                                             True if type(current) is int else "wrong")
                self.rejected(value)
        for key in receipt()["cases"][17]:
            value = receipt()
            del value["cases"][17][key]
            self.rejected(value)

    def test_short_extra_duplicate_reordered_and_swapped_pair_inventories_reject(self):
        for field in ("cases", "all_format_single_member_cases"):
            for how in ("short", "extra", "duplicate", "swap", "tuple"):
                value = receipt()
                rows = value[field]
                if how == "short":
                    rows.pop()
                elif how == "extra":
                    rows.append(copy.deepcopy(rows[-1]))
                elif how == "duplicate":
                    rows[-1] = copy.deepcopy(rows[0])
                elif how == "swap":
                    rows[0], rows[1] = rows[1], rows[0]
                else:
                    value[field] = tuple(rows)
                self.rejected(value)
        value = receipt()
        value["cases"][17], value["cases"][19] = value["cases"][19], value["cases"][17]
        self.rejected(value)
        value = receipt()
        value["cases"] = value["cases"][:16] + value["cases"][22:42] + value["cases"][48:]
        self.rejected(value)  # archived 40-case and generic-only receipts cannot pass

    def test_q4_comparison_is_required_only_for_independently_selected_candidate(self):
        for index, bad in ((16, 0.), (17, None), (17, True), (18, False)):
            value = receipt()
            value["cases"][index]["generic_max_abs_error"] = bad
            self.rejected(value)
        value = receipt()
        value["cases"][16]["gate_policy"] = "q4-block"
        self.rejected(value)

    def test_numeric_maxima_reject_nonfinite_negative_boolean_and_wrong_shapes(self):
        for bad in (float("nan"), float("inf"), -float("inf"), -1., True, None, "0", 10**400):
            for key in ("core_output_max_abs_error", "scalar_max_abs_error", "generic_max_abs_error"):
                value = receipt()
                value["cases"][17][key] = bad
                self.rejected(value)
            value = receipt()
            value["cases"][0]["tree_activation_partial_output_max_abs_errors"][1] = bad
            self.rejected(value)
            value = receipt()
            value["all_format_single_member_cases"][0]["max_abs_activation_error"] = bad
            self.rejected(value)
        for bad in ([], [0., 0.], [0.] * 4, (0., 0., 0.)):
            value = receipt()
            value["cases"][0]["tree_activation_partial_output_max_abs_errors"] = bad
            self.rejected(value)

    def test_every_format_tile_and_owning_check_is_strict(self):
        for index in range(14):
            for key in ("dtype", "tile", "count", "groups", "private_pages_bytes",
                        "all_ten_ranks", "full_offset_tail_status_and_erasure_checks"):
                value = receipt()
                current = value["all_format_single_member_cases"][index][key]
                value["all_format_single_member_cases"][index][key] = (
                    int(current) if type(current) is bool else True)
                self.rejected(value)
        value = receipt()
        value["all_format_single_member_cases"][0]["extra"] = True
        self.rejected(value)

    def test_expected_layers_are_independent_strict_and_never_receipt_inferred(self):
        for layers in ((0, 6), (6, 6), (True, 0), (6., 0), (6, 48), [-1, 0], [6, 0], (6,), None):
            self.rejected(receipt(), layers)
        value = receipt()
        for row in value["cases"]:
            row["layer"] = 2 if row["down_dtype"] == 6 else 3
        self.rejected(value)
        self.assertTrue(VALIDATOR.validate_receipt(value, expected_layers=(2, 3)))

    def test_loader_rejects_duplicate_keys_nonfinite_literals_and_size_limit(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = pathlib.Path(temporary) / "receipt.json"
            path.write_text(json.dumps(receipt()))
            self.assertTrue(VALIDATOR.validate_receipt(VALIDATOR.load_receipt(path), expected_layers=(6, 0)))
            for data in ('{"cases":[],"cases":[]}', '{"nested":{"x":1,"x":2}}',
                         '{"max":NaN}', '{"max":Infinity}', '{"max":-Infinity}',
                         " " * (VALIDATOR.RECEIPT_MAX_BYTES + 1)):
                path.write_text(data)
                with self.assertRaisesRegex(ValueError, "Fixed-public Q4 operator receipt rejected"):
                    VALIDATOR.load_receipt(path)

    def test_cli_requires_declared_layers_and_reports_only_fixed_acceptance(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = pathlib.Path(temporary) / "receipt.json"
            path.write_text(json.dumps(receipt()))
            args = [sys.executable, str(SCRIPT), "--receipt", str(path)]
            missing = subprocess.run(args, capture_output=True, text=True)
            self.assertNotEqual(missing.returncode, 0)
            result = subprocess.run(args + ["--q5-layer", "6", "--q8-layer", "0"],
                                    capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("cases=52 q4_block_cases=6 formats=14", result.stdout)
            self.assertIn("whole_model_accepted=false performance_accepted=false", result.stdout)
            value = receipt()
            value["cases"][17]["generic_max_abs_error"] = None
            path.write_text(json.dumps(value))
            failed = subprocess.run(args + ["--q5-layer", "6", "--q8-layer", "0"],
                                    capture_output=True, text=True)
            self.assertEqual(failed.returncode, 1)
            self.assertEqual(failed.stderr.strip(), "Fixed-public Q4 operator receipt rejected")


if __name__ == "__main__":
    unittest.main()
