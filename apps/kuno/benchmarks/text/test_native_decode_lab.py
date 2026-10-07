import contextlib
import copy
import importlib.util
import io
import json
import pathlib
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock


HERE = pathlib.Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("native_decode_lab", HERE / "run-native-max8-lab.py")
LAB = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(LAB)
GUARD = "verified_lab_limits swap_max=0 core_soft=0 core_hard=0 memory_max=8589934592"
OUTPUT = "\n".join((
    GUARD,
    "public_mid_token_cancel layer=0 erased=true",
    "public_mid_token_cancel layer=3 erased=true",
    "public_resident_cancel_erased=true poisoned_retry_rejected=true fresh_first_token_matches=true",
    "public_resident_owned_cleanup=true context_retired=false",
))


def archived_receipt(card="intel"):
    return json.loads((HERE / f"receipts/rust-native-model-full-{card}-20261005.json").read_text())["report"]


def current_receipt(native_decode, card="intel"):
    receipt = archived_receipt(card)
    steps = len(receipt["public_generated_ids"]) - 1
    receipt.update(native_decode=native_decode, native_decode_steps=steps if native_decode else 0,
                   scalar_decode_steps=0 if native_decode else steps)
    if native_decode:
        receipt["owned_zero_audits"][0].update(source_buffers=0, source_bytes=0)
    return receipt


class NativeDecodeLabTests(unittest.TestCase):
    def validate(self, receipt, native_decode=False, card="intel", **options):
        LAB.validate_receipt(receipt, card, "resident-rainwater", 8, OUTPUT,
                             native_decode=native_decode, **options)

    def test_both_selected_modes_accept_exact_counts_and_unchanged_main_audits_on_both_cards(self):
        for card in ("intel", "nvidia"):
            for native in (False, True):
                self.validate(current_receipt(native, card), native, card)

    def test_archived_missing_fields_require_explicit_offline_legacy_selection(self):
        for card in ("intel", "nvidia"):
            receipt = archived_receipt(card)
            with self.assertRaises(ValueError):
                self.validate(receipt, card=card)
            self.validate(receipt, card=card, allow_legacy_decode=True)
            with self.assertRaises(ValueError):
                self.validate(receipt, True, card, allow_legacy_decode=True)

    def test_partial_fields_never_use_legacy_fallback(self):
        fields = ("native_decode", "native_decode_steps", "scalar_decode_steps")
        for missing in fields:
            receipt = current_receipt(False)
            del receipt[missing]
            for legacy in (False, True):
                with self.assertRaises(ValueError):
                    self.validate(receipt, allow_legacy_decode=legacy)
        for present in fields:
            receipt = archived_receipt()
            receipt[present] = current_receipt(False)[present]
            with self.assertRaises(ValueError):
                self.validate(receipt, allow_legacy_decode=True)

    def test_receipt_cannot_select_its_own_decode_mode(self):
        for selected in (False, True):
            with self.assertRaises(ValueError):
                self.validate(current_receipt(not selected), selected)

    def test_counts_reject_booleans_wrong_types_negative_and_extra_successes(self):
        for native in (False, True):
            original = current_receipt(native)
            for field in ("native_decode_steps", "scalar_decode_steps"):
                for bad in (True, False, -1, 1.0, "127", None, 128, original[field] + 1):
                    receipt = copy.deepcopy(original)
                    receipt[field] = bad
                    with self.assertRaises(ValueError):
                        self.validate(receipt, native)
            for bad in (0, 1, "0", "1", None):
                receipt = copy.deepcopy(original)
                receipt["native_decode"] = bad
                with self.assertRaises(ValueError):
                    self.validate(receipt, native)

    def test_validation_settings_require_actual_booleans(self):
        for bad in (0, 1, "0", "1", None):
            with self.assertRaises(ValueError):
                self.validate(current_receipt(False), bad)
            with self.assertRaises(ValueError):
                self.validate(archived_receipt(), allow_legacy_decode=bad)

    def test_every_main_owner_and_exact_source_counts_remain_required(self):
        for native in (False, True):
            original = current_receipt(native)
            for owner in range(4):
                for field in ("main_buffers", "main_bytes", "source_buffers", "source_bytes"):
                    receipt = copy.deepcopy(original)
                    receipt["owned_zero_audits"][owner][field] += 1
                    with self.assertRaises(ValueError):
                        self.validate(receipt, native)
            for missing in range(4):
                receipt = copy.deepcopy(original)
                del receipt["owned_zero_audits"][missing]
                with self.assertRaises(ValueError):
                    self.validate(receipt, native)

    def test_native_decode_does_not_relax_fixture_limits_or_lifecycle_markers(self):
        original = current_receipt(True)
        for field, bad in (("public_generated_ids", [1]),
                           ("resident_runtime_io_limit_bytes", 67108865),
                           ("native_projection_tile", 1),
                           ("resident_owned_zero_audit_enabled", False),
                           ("resident_transient_io", False), ("native_max8", False)):
            receipt = copy.deepcopy(original)
            receipt[field] = bad
            with self.assertRaises(ValueError):
                self.validate(receipt, True)
        for missing in OUTPUT.splitlines():
            with self.assertRaises(ValueError):
                LAB.validate_receipt(original, "intel", "resident-rainwater", 8,
                                     OUTPUT.replace(missing, ""), native_decode=True)
        with self.assertRaises(ValueError):
            LAB.validate_receipt(original, "intel", "resident-rainwater", 8,
                                 OUTPUT.replace("8589934592", "55834574848"), native_decode=True)

    def test_runner_uses_explicit_selection_and_does_not_copy_ambient_settings(self):
        with mock.patch.dict("os.environ", {"KUNO_LAB_NATIVE_DECODE": "1", "PRIVATE_TOKEN": "never-forward"}):
            for native in (False, True):
                env = LAB.worker_environment("/original/blob", "/public/receipt", 8, native)
                self.assertEqual(env["KUNO_LAB_NATIVE_DECODE"], "1" if native else "0")
                self.assertEqual(env["KUNO_LAB_NATIVE_MAX8"], "1")
                self.assertEqual(env["KUNO_LAB_TRANSIENT_IO"], "1")
                self.assertEqual(env["KUNO_LAB_OWNED_ZERO_AUDIT"], "1")
                self.assertEqual(env["KUNO_LAB_CANCEL_CHECK"], "1")
                self.assertNotIn("PRIVATE_TOKEN", env)

    def test_live_cli_pins_selected_mode_and_validates_current_receipt_before_recording(self):
        for native in (False, True):
            with tempfile.TemporaryDirectory() as directory:
                target = pathlib.Path(directory) / "receipt.json"
                command_seen = []
                def launch(command, **options):
                    command_seen.extend(command)
                    target.write_text(json.dumps(current_receipt(native)))
                    return SimpleNamespace(stdout=io.StringIO(OUTPUT + "\n"), wait=lambda: 0)
                args = [str(HERE / "run-native-max8-lab.py"), "--cargo-artifacts", "/public/artifacts",
                        "--model", "/original/blob", "--card", "intel", "--fixture", "resident-rainwater",
                        "--count", "8", "--run-id", "fixture", "--receipt", str(target)]
                if native:
                    args.append("--native-decode")
                with mock.patch("sys.argv", args), \
                     mock.patch.object(LAB, "cargo_binary", return_value=("/public/frozen", "a" * 64, "b" * 64)), \
                     mock.patch.object(LAB.subprocess, "Popen", side_effect=launch), \
                     contextlib.redirect_stdout(io.StringIO()):
                    LAB.main()
                self.assertIn("--setenv=KUNO_LAB_NATIVE_DECODE=" + ("1" if native else "0"), command_seen)
                record = json.loads(target.with_suffix(".json.run.json").read_text())
                self.assertIs(record["native_decode"], native)


if __name__ == "__main__":
    unittest.main()
