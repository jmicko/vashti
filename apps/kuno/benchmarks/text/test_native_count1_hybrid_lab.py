"""Host-only rejection controls. These synthetic documents are not device receipts."""
import copy
import importlib.util
import io
import json
import os
import pathlib
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

HERE = pathlib.Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("native_hybrid_lab", HERE / "run-native-count1-hybrid-lab.py")
LAB = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(LAB)


def receipt(card, cap):
    # Historical real native inventory, altered solely for validator tests.
    # No test output is saved as a hybrid device acceptance receipt.
    value = json.loads((HERE / f"receipts/rust-native-model-full-{card}-20261005.json").read_text())["report"]
    value.update(fixture=LAB.MODE, prompt_tokens=15, native_public_chunk_partition=[8, 7],
                 public_generated_ids=[9419, 0, 2500, 628, 353, 1438, 488, 3242],
                 native_decode=True, native_decode_steps=7, scalar_decode_steps=0,
                 native_decode_projection_tile=1, native_decode_kernels="single-q4-all",
                 observed_native_decode_kernels="single-q4-all", resident_private_owned_bytes=818413568,
                 count1_hybrid_cpu_miss_cap=cap, observed_count1_hybrid_cpu_miss_cap=cap,
                 completed_cpu_misses=17 if cap else 0, fresh_completed_cpu_misses=3 if cap else 0,
                 count1_hybrid_cpu_gpu_overlap=False, count1_hybrid_cpu_workers="scoped", observed_count1_hybrid_cpu_workers="scoped")
    for audit in value["owned_zero_audits"]:
        audit.update(main_buffers=1378, main_bytes=818413568, source_buffers=0, source_bytes=0)
    return value


def completion(cap):
    return "\n".join((LAB.GUARD,
        "public_mid_token_cancel layer=0 erased=true",
        "public_mid_token_cancel layer=3 erased=true",
        "public_resident_cancel_erased=true poisoned_retry_rejected=true fresh_first_token_matches=true",
        "public_resident_owned_cleanup=true context_retired=false",
        *LAB.BASE.TILE_ONE_COMPLETION, *LAB.BASE.kernel_completion("single-q4-all"),
        *LAB.completion(cap)))


class HybridLabTests(unittest.TestCase):
    def test_both_caps_and_cards_require_fixed_inventory_and_actual_cpu_work(self):
        for card in ("intel", "nvidia"):
            for cap in (0, 10):
                LAB.validate_receipt(receipt(card, cap), card, cap, completion(cap))

    def test_cpu_worker_policy_cannot_be_forged_or_inherited(self):
        for selected in ("scoped", "pooled"):
            value = receipt("intel", 10)
            value.update(count1_hybrid_cpu_workers=selected, observed_count1_hybrid_cpu_workers=selected)
            LAB.validate_receipt(value, "intel", 10, completion(10), selected)
            for field in ("count1_hybrid_cpu_workers", "observed_count1_hybrid_cpu_workers"):
                for bad in (None, True, "auto", "pooled" if selected == "scoped" else "scoped"):
                    wrong = copy.deepcopy(value)
                    wrong[field] = bad
                    with self.assertRaises(ValueError):
                        LAB.validate_receipt(wrong, "intel", 10, completion(10), selected)
            with mock.patch.dict(os.environ, {"KUNO_LAB_CPU_WORKERS": "auto"}):
                env = LAB.worker_environment("/original/model", "/public/receipt", selected)
            self.assertEqual(env["KUNO_LAB_CPU_WORKERS"], selected)
        with self.assertRaises(ValueError):
            LAB.worker_environment("/original/model", "/public/receipt", "auto")

    def test_receipt_cannot_select_cap_or_claim_overlap(self):
        for cap in (0, 10):
            for field in ("count1_hybrid_cpu_miss_cap", "observed_count1_hybrid_cpu_miss_cap"):
                for wrong in (None, True, "10", 10.0, 1, 10 - cap):
                    value = receipt("intel", cap)
                    value[field] = wrong
                    with self.assertRaises(ValueError):
                        LAB.validate_receipt(value, "intel", cap, completion(cap))
            for bad in (None, 0, True, "false"):
                value = receipt("intel", cap)
                value["count1_hybrid_cpu_gpu_overlap"] = bad
                with self.assertRaises(ValueError):
                    LAB.validate_receipt(value, "intel", cap, completion(cap))
        for cap in (False, "0", 0.0, 1, -1, 11):
            with self.assertRaises(ValueError):
                LAB.validate_receipt(receipt("intel", 0), "intel", cap, completion(0))

    def test_completed_counts_reject_attempt_only_missing_and_impossible_successes(self):
        for cap in (0, 10):
            for field, maximum in (("completed_cpu_misses", 3360), ("fresh_completed_cpu_misses", 480)):
                for bad in (None, True, "1", 1.0, -1, maximum + 1, 0 if cap else 1):
                    value = receipt("intel", cap)
                    value[field] = bad
                    with self.assertRaises(ValueError):
                        LAB.validate_receipt(value, "intel", cap, completion(cap))

    def test_every_actual_owner_and_output_and_lifecycle_proof_is_mandatory(self):
        original = receipt("intel", 10)
        for at in range(4):
            for field in ("main_buffers", "main_bytes", "source_buffers", "source_bytes"):
                for bad in (None, True, "0", -1, original["owned_zero_audits"][at][field] + 1):
                    value = copy.deepcopy(original)
                    value["owned_zero_audits"][at][field] = bad
                    with self.assertRaises(ValueError):
                        LAB.validate_receipt(value, "intel", 10, completion(10))
        for field, bad in (("fixture", "resident-chat"), ("public_generated_ids", [9419]),
                           ("native_decode_steps", 8), ("scalar_decode_steps", 1), ("temperature", False),
                           ("native_projection_tile", 1), ("native_decode_projection_tile", 8),
                           ("observed_native_decode_kernels", "chunk"), ("batch_experts", True),
                           ("resident_private_owned_bytes", 818249728), ("owned_zero_audits", [])):
            value = copy.deepcopy(original)
            value[field] = bad
            with self.assertRaises(ValueError):
                LAB.validate_receipt(value, "intel", 10, completion(10))
        for line in completion(10).splitlines():
            with self.assertRaises(ValueError):
                LAB.validate_receipt(original, "intel", 10, completion(10).replace(line, ""))

    def test_ordinary_validator_still_rejects_hybrid_inventory(self):
        value = receipt("intel", 0)
        value["fixture"] = "resident-chat"
        with self.assertRaises(ValueError):
            LAB.BASE.validate_receipt(value, "intel", "resident-chat", 8, completion(0),
                native_decode=True, native_decode_tile=1, native_decode_kernels="single-q4-all")

    def test_environment_does_not_inherit_ambient_policy(self):
        with mock.patch.dict(os.environ, {"KUNO_LAB_SMALL_CPU": "1", "KUNO_LAB_EXPERT_SLOTS": "96"}):
            env = LAB.worker_environment("/original/model", "/public/receipt")
        self.assertEqual(env["KUNO_LAB_SMALL_CPU"], "0")
        self.assertNotIn("KUNO_LAB_EXPERT_SLOTS", env)
        self.assertEqual(env["KUNO_LAB_NATIVE_BATCH_COUNT"], "8")
        self.assertEqual(env["KUNO_LAB_NATIVE_DECODE_KERNELS"], "single-q4-all")
        self.assertEqual(env["KUNO_LAB_CANCEL_CHECK"], "1")

    def test_live_runner_pins_cap_validates_before_recording_and_rejects_incomplete_proofs(self):
        for valid in (False, True):
            with tempfile.TemporaryDirectory() as directory:
                target = pathlib.Path(directory) / "receipt.json"
                command_seen = []
                def launch(command, **options):
                    command_seen.extend(command)
                    value = receipt("intel", 10)
                    if not valid:
                        value["completed_cpu_misses"] = 0
                    target.write_text(json.dumps(value))
                    return SimpleNamespace(stdout=io.StringIO(completion(10)), wait=lambda: 0)
                args = ["lab", "--cargo-artifacts", "/build.jsonl", "--model", "/original/model",
                        "--card", "intel", "--cpu-miss-cap", "10", "--run-id", "test",
                        "--receipt", str(target)]
                with mock.patch.object(LAB.BASE, "cargo_binary", return_value=(pathlib.Path("/frozen"), "binary", "build")), \
                     mock.patch.object(LAB.subprocess, "Popen", side_effect=launch), \
                     mock.patch.object(LAB.sys, "argv", args), mock.patch("sys.stdout", new_callable=io.StringIO):
                    if valid:
                        LAB.main()
                    else:
                        with self.assertRaises(ValueError):
                            LAB.main()
                self.assertEqual(command_seen[-3:], ["intel", LAB.MODE, "10"])
                self.assertIn("--property=MemorySwapMax=0", command_seen)
                self.assertIn("--property=LimitCORE=0", command_seen)
                record = target.with_suffix(".json.run.json")
                self.assertEqual(record.exists(), valid)
                if valid:
                    data = json.loads(record.read_text())
                    self.assertFalse(data["matched_performance_measurement"])
                    self.assertFalse(data["cpu_gpu_overlap"])
                    self.assertEqual(data["cpu_miss_cap"], 10)


if __name__ == "__main__":
    unittest.main()
