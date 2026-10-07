"""Focused corruption tests for fixed-public staged numeric acceptance."""

import copy
import importlib.util
import json
import pathlib
import subprocess
import sys
import tempfile
import unittest


SCRIPT = pathlib.Path(__file__).with_name("validate-native-staged-hybrid.py")
SPEC = importlib.util.spec_from_file_location("staged_hybrid_validator", SCRIPT)
VALIDATOR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(VALIDATOR)


def receipt(vendor=0x8086):
    cases = []
    for dtype, layer in ((6, 6), (8, 0)):
        for misses, cpu, gpu, source in ((5, 0, 5, 20971520), (5, 1, 4, 16777216),
                                         (5, 5, 0, 0), (10, 10, 0, 0)):
            cases.append({
                "layer": layer, "down_dtype": dtype, "workers": 16,
                "backend": "avx2", "mode": "concurrent", "ranks": 10,
                "misses": misses, "cpu_misses": cpu, "gpu_misses": gpu,
                "group": 0, "vector": 1, "max_tree_error": 0.,
                "max_legacy_error": 0., "max_core_error": 0.,
                "private_pages": 18, "private_bytes": 884736,
                "bank_bytes": 134217728, "source_precharge": source,
                "io_inspected_bytes": 32768, "zero_buffers": 18,
                "zero_bytes": 884736, "metadata_bytes": 32768, "snapshot_bytes": 32768,
                "operator_numeric_accepted": True, "owned_cleanup_accepted": True,
            })
    return {
        "scope": "fixed-public-original-staged-cpu-gpu-expert-numeric",
        "vendor": vendor, "weights": "borrowed-unmodified-ollama-blob",
        "source_kind": "original-packed-gguf", "fixed_public_cases": 8,
        "purpose": "numeric", "ranks_per_case": 10, "group": 0, "vector": 1,
        "backend": "avx2", "workers": 16, "bank_slots_per_owner": 16,
        "private_pages_per_case": 18, "private_bytes_per_case": 884736,
        "bank_bytes_per_case": 134217728, "physical_limit_bytes": 268435456,
        "source_window_limit_bytes": 41943040, "runtime_io_limit_bytes": 67108864,
        "runtime_io_component_limit_bytes": 8388608, "private_host_limit_bytes": 4194304,
        "cpu_private_host_limit_bytes": 1048576, "public_cpu_packed_limit_bytes": 41943040,
        "bridge_host_limit_bytes": 1048576, "oracle_host_limit_bytes": 2097152,
        "retained_plan_limit_bytes": 262144, "cases": cases,
        "operator_numeric_accepted": True, "owned_cleanup_accepted": True,
        "context_retired": False, "cache_publication_accepted": False,
        "whole_model_numeric_accepted": False, "timing_accepted": False,
        "performance_accepted": False, "production_accepted": False,
        "automatic_split_policy_enabled": False,
    }


class StagedHybrid(unittest.TestCase):
    def accepted(self, value, layers=(6, 0), vendor=0x8086):
        return VALIDATOR.validate_receipt(value, expected_layers=layers, expected_vendor=vendor)

    def rejected(self, value, layers=(6, 0), vendor=0x8086):
        with self.assertRaisesRegex(ValueError, "Fixed-public staged hybrid numeric receipt rejected"):
            self.accepted(value, layers, vendor)

    def test_current_eight_case_contract_accepts_both_pinned_cards(self):
        for vendor in (0x8086, 0x10DE):
            self.assertTrue(self.accepted(receipt(vendor), vendor=vendor))
        value = receipt()
        for result in value["cases"]:
            result["max_tree_error"] = 1e-6
            result["max_legacy_error"] = 1e-5
            result["max_core_error"] = 1e-4
        self.assertTrue(self.accepted(value))  # no zero-error or substitute relative gate

    def test_every_top_level_field_value_and_type_is_strict(self):
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
        for extra in ("timing_ns", "ids", "miss_mask", "slots", "input", "legacy"):
            value = receipt()
            value[extra] = None
            self.rejected(value)

    def test_each_case_has_exact_public_matrix_source_charges_and_clean_owners(self):
        dynamic = {"max_tree_error", "max_legacy_error", "max_core_error",
                   "io_inspected_bytes", "metadata_bytes", "snapshot_bytes"}
        for index in range(8):
            for key, current in receipt()["cases"][index].items():
                if key in dynamic:
                    continue
                value = receipt()
                value["cases"][index][key] = (not current if type(current) is bool else
                                             current + 1 if type(current) is int else "wrong")
                self.rejected(value)
                value = receipt()
                value["cases"][index][key] = (int(current) if type(current) is bool else
                                             float(current) if type(current) is int else None)
                self.rejected(value)
        for key in receipt()["cases"][0]:
            value = receipt()
            del value["cases"][0][key]
            self.rejected(value)
        for extra in ("timing", "timing_ns", "bridge_and_merge", "map", "partials"):
            value = receipt()
            value["cases"][0][extra] = None
            self.rejected(value)

    def test_short_extra_duplicate_reordered_or_nonlist_inventory_rejects(self):
        for how in ("short", "extra", "duplicate", "split-swap", "dtype-swap", "tuple", "null"):
            value = receipt()
            rows = value["cases"]
            if how == "short":
                rows.pop()
            elif how == "extra":
                rows.append(copy.deepcopy(rows[-1]))
            elif how == "duplicate":
                rows[-1] = copy.deepcopy(rows[0])
            elif how == "split-swap":
                rows[0], rows[1] = rows[1], rows[0]
            elif how == "dtype-swap":
                rows[0], rows[4] = rows[4], rows[0]
            else:
                value["cases"] = tuple(rows) if how == "tuple" else None
            self.rejected(value)
        value = receipt()
        value["cases"] = value["cases"][::2]  # old four-case probe is not this authority
        self.rejected(value)

    def test_actual_dynamic_amounts_require_typed_positive_existing_bounds(self):
        for key, limit in (("io_inspected_bytes", 4194304), ("metadata_bytes", 262144),
                           ("snapshot_bytes", 2097152)):
            for amount in (1, limit):
                value = receipt()
                value["cases"][0][key] = amount
                self.assertTrue(self.accepted(value))
            for amount in (0, -1, limit + 1, True, float(limit), "1", None):
                value = receipt()
                value["cases"][0][key] = amount
                self.rejected(value)

    def test_errors_require_finite_nonnegative_numeric_f32_without_new_tolerance(self):
        for key in ("max_tree_error", "max_legacy_error", "max_core_error"):
            for bad in (float("nan"), float("inf"), -float("inf"), -1., True, None, "0", 10**400):
                value = receipt()
                value["cases"][0][key] = bad
                self.rejected(value)
        value = receipt()
        value["cases"][0]["max_core_error"] = VALIDATOR.F32_MAX
        self.assertTrue(self.accepted(value))  # representation, not a fabricated tolerance

    def test_authority_is_required_independently_and_cannot_be_receipt_inferred(self):
        for layers in ((0, 6), (6, 6), (True, 0), (6., 0), (6, 48), (6,), [6, 0], None):
            self.rejected(receipt(), layers=layers)
        for vendor in (0x10DE, 0, True, float(0x8086), "intel", None):
            self.rejected(receipt(), vendor=vendor)
        value = receipt()
        for result in value["cases"]:
            result["layer"] = 2 if result["down_dtype"] == 6 else 3
        self.rejected(value)
        self.assertTrue(self.accepted(value, layers=(2, 3)))
        for kwargs in ({}, {"expected_layers": (6, 0)}, {"expected_vendor": 0x8086}):
            with self.assertRaises(TypeError):
                VALIDATOR.validate_receipt(receipt(), **kwargs)

    def test_loader_rejects_duplicate_keys_nonfinite_literals_and_size_limit(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = pathlib.Path(temporary) / "receipt.json"
            path.write_text(json.dumps(receipt()))
            self.assertTrue(self.accepted(VALIDATOR.load_receipt(path)))
            for data in ('{"vendor":32902,"vendor":4318}', '{"cases":[{"x":1,"x":2}]}',
                         '{"error":NaN}', '{"error":Infinity}', '{"error":-Infinity}',
                         " " * (VALIDATOR.RECEIPT_MAX_BYTES + 1)):
                path.write_text(data)
                with self.assertRaisesRegex(ValueError, "Fixed-public staged hybrid numeric receipt rejected"):
                    VALIDATOR.load_receipt(path)

    def test_cli_requires_independent_card_layers_and_fixed_claims(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = pathlib.Path(temporary) / "receipt.json"
            path.write_text(json.dumps(receipt()))
            args = [sys.executable, str(SCRIPT), "--receipt", str(path)]
            for authority in ([], ["--vendor", "intel"], ["--q5-layer", "6", "--q8-layer", "0"]):
                result = subprocess.run(args + authority, capture_output=True, text=True)
                self.assertNotEqual(result.returncode, 0)
            authority = ["--vendor", "intel", "--q5-layer", "6", "--q8-layer", "0"]
            result = subprocess.run(args + authority, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("cases=8 owned_cleanup=true", result.stdout)
            for claim in ("cache_publication", "timing", "whole_model", "performance", "production"):
                self.assertIn(claim + "_accepted=false", result.stdout)
            value = receipt()
            value["cases"][0]["zero_buffers"] -= 1
            path.write_text(json.dumps(value))
            rejected = subprocess.run(args + authority, capture_output=True, text=True)
            self.assertEqual(rejected.returncode, 1)
            self.assertEqual(rejected.stderr.strip(), "Fixed-public staged hybrid numeric receipt rejected")


if __name__ == "__main__":
    unittest.main()
