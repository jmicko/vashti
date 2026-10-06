"""Focused acceptance and corruption cases for the fixed-public validator."""

import copy
import importlib.util
import json
import pathlib
import subprocess
import sys
import tempfile
import unittest


SCRIPT = pathlib.Path(__file__).with_name("validate-native-numeric-boundaries.py")
SPEC = importlib.util.spec_from_file_location("numeric_validator", SCRIPT)
VALIDATOR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(VALIDATOR)


def receipt(card="intel", original=64090112):
    return {
        "fixture": "public_native_numeric_boundaries_v1", "position": 0, "count": 1,
        "pairs": 4, "owners_serialized": 8, "extra_main_bytes": 0,
        "guarded_host_bound": 610304, "guarded_host_cap": 1048576,
        "extra_audited_runtime_bytes": 1114112, "original_runtime_bytes": original,
        "combined_runtime_bytes": original + 1114112, "runtime_hard_cap": 67108864,
        "normal_family_acceptance": True, "explicit_full_cleanup": True,
        "whole_model_numeric_accepted": False, "performance_accepted": False,
        "vendor": 0x8086 if card == "intel" else 0x10DE,
        "weights": "borrowed-unmodified-ollama-blob",
        "fixed_public_input": "original tokenizer ! position0 count1", "seed": 424242,
        "boundaries": [
            {"boundary": name, "logical_values": width,
             "scalar_owned_map_bytes": scalar_map, "native_owned_map_bytes": native_map,
             "bit_mismatches": 0, "max_absolute": 0.0, "bit_equal": True,
             "scalar_cleanup": {"main_bytes": 785186816, "main_buffers": 1286,
                                "source_bytes": 557056, "source_buffers": 15},
             "native_cleanup": {"main_bytes": 818249728, "main_buffers": 1376,
                                "source_bytes": 0, "source_buffers": 0}}
            for name, width, scalar_map, native_map in VALIDATOR.BOUNDARIES
        ],
    }


class NumericBoundaries(unittest.TestCase):
    def rejected(self, value, card="intel"):
        with self.assertRaisesRegex(ValueError, "Fixed-public numeric boundary receipt rejected"):
            VALIDATOR.validate_receipt(value, card)

    def test_both_cards_both_supported_runtime_plans(self):
        for card in ("intel", "nvidia"):
            for runtime in (64090112, 64000000):
                with self.subTest(card=card, runtime=runtime):
                    self.assertTrue(VALIDATOR.validate_receipt(receipt(card, runtime), card))

    def test_diagnostic_mismatch_and_signed_zero_do_not_claim_acceptance(self):
        value = receipt()
        value["boundaries"][3].update(bit_mismatches=9, max_absolute=1e-6, bit_equal=False)
        self.assertTrue(VALIDATOR.validate_receipt(value, "intel"))
        value["boundaries"][3]["max_absolute"] = 0.0
        self.assertTrue(VALIDATOR.validate_receipt(value, "intel"))

    def test_every_exact_identity_and_budget_field_is_enforced(self):
        value = receipt()
        for key, current in value.items():
            if key == "boundaries":
                continue
            mutated = copy.deepcopy(value)
            mutated[key] = (not current if type(current) is bool else current + 1
                            if type(current) is int else "wrong")
            with self.subTest(key=key):
                self.rejected(mutated)
        self.rejected(receipt("nvidia"), "intel")

    def test_missing_extra_duplicate_or_reordered_boundaries_rejected(self):
        value = receipt()
        for results in (value["boundaries"][:3], value["boundaries"] * 2,
                        value["boundaries"][::-1], [value["boundaries"][0]] * 4, {}):
            mutated = copy.deepcopy(value)
            mutated["boundaries"] = results
            self.rejected(mutated)

    def test_boundary_maps_and_widths_are_exact(self):
        for index in range(4):
            for key in ("boundary", "logical_values", "scalar_owned_map_bytes",
                        "native_owned_map_bytes"):
                mutated = receipt()
                original = mutated["boundaries"][index][key]
                mutated["boundaries"][index][key] = original + 1 if type(original) is int else "x"
                with self.subTest(index=index, key=key):
                    self.rejected(mutated)

    def test_all_eight_owners_need_complete_exact_main_and_source_audits(self):
        for index in range(4):
            for owner in ("scalar_cleanup", "native_cleanup"):
                for field in (None, "main_bytes", "main_buffers", "source_bytes", "source_buffers"):
                    mutated = receipt()
                    if field is None:
                        del mutated["boundaries"][index][owner]
                    else:
                        mutated["boundaries"][index][owner][field] += 1
                    with self.subTest(index=index, owner=owner, field=field):
                        self.rejected(mutated)

    def test_nonfinite_negative_and_inconsistent_numeric_reports_rejected(self):
        for key, invalid in (("max_absolute", float("nan")), ("max_absolute", float("inf")),
                             ("max_absolute", -1), ("max_absolute", 1e-5),
                             ("max_absolute", 10 ** 400),
                             ("max_absolute", True), ("bit_mismatches", -1),
                             ("bit_mismatches", 10241), ("bit_mismatches", True),
                             ("bit_mismatches", 1), ("bit_equal", False), ("bit_equal", 1)):
            mutated = receipt()
            mutated["boundaries"][0][key] = invalid
            with self.subTest(key=key, invalid=invalid):
                self.rejected(mutated)

    def test_bool_cannot_impersonate_integer_and_zero_cannot_impersonate_false(self):
        for key, invalid in (("position", False), ("count", True),
                             ("whole_model_numeric_accepted", 0), ("performance_accepted", 0),
                             ("original_runtime_bytes", 64090112.0),
                             ("combined_runtime_bytes", 65204224.0)):
            mutated = receipt()
            mutated[key] = invalid
            self.rejected(mutated)

    def test_unknown_content_is_rejected_at_every_object_level(self):
        for level in ("receipt", "boundary", "scalar", "native"):
            mutated = receipt()
            target = {"receipt": mutated, "boundary": mutated["boundaries"][0],
                      "scalar": mutated["boundaries"][0]["scalar_cleanup"],
                      "native": mutated["boundaries"][0]["native_cleanup"]}[level]
            target["private_prompt"] = "secret content must never be printed"
            self.rejected(mutated)

    def test_duplicate_json_keys_constants_and_oversized_receipts_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = pathlib.Path(temporary) / "receipt.json"
            for payload in ('{"fixture":"a","fixture":"b"}', '{"value":NaN}',
                            '{"value":Infinity}', ' ' * 65537):
                path.write_text(payload)
                with self.subTest(payload_kind=payload[:24]), self.assertRaises(ValueError):
                    VALIDATOR.load_receipt(path)

    def test_cli_success_and_content_free_failure(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = pathlib.Path(temporary) / "receipt.json"
            value = receipt()
            path.write_text(json.dumps(value))
            command = [sys.executable, str(SCRIPT), "--receipt", str(path), "--card", "intel"]
            success = subprocess.run(command, text=True, capture_output=True, check=False)
            self.assertEqual(success.returncode, 0)
            self.assertIn("whole_model_numeric_accepted=false", success.stdout)
            value["secret-key"] = "private-data-that-must-not-escape"
            path.write_text(json.dumps(value))
            failure = subprocess.run(command, text=True, capture_output=True, check=False)
            self.assertEqual(failure.returncode, 1)
            self.assertEqual(failure.stdout, "")
            self.assertEqual(failure.stderr, "Fixed-public numeric boundary receipt rejected\n")
            self.assertNotIn("private-data", failure.stderr)


if __name__ == "__main__":
    unittest.main()
