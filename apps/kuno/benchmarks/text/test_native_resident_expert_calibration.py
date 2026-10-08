"""Reject misleading residency and incomplete completion in public lab receipts."""
import copy
import importlib.util
import pathlib
import unittest

SPEC = importlib.util.spec_from_file_location("resident_expert", pathlib.Path(__file__).with_name("run-native-resident-expert-calibration.py"))
LAB = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(LAB)


def fixture():
    receipt = dict(scope="fixed-public-original-resident-expert-calibration",
        resident_expert_calibration=True, device_dispatch="inline", vendor=0x10DE,
        weights="borrowed-unmodified-ollama-blob", fixed_public_points=6,
        numeric_gates=12, timing_samples=24, ranks_per_case=10, group=0, vector=0,
        backend="pooled-avx2-fma", workers=16, owned_cleanup_accepted=True,
        context_retired=False, full_model_or_automatic_placement_accepted=False, points=[])
    for index in range(6):
        down = 6 if index < 3 else 8
        misses, cpu = [(0, 0), (10, 0), (10, 10)][index % 3]
        report = dict(layer=6, down_dtype=down, workers=16, misses=misses, cpu_misses=cpu,
            gpu_misses=misses-cpu, private_pages=18, private_bytes=884736,
            bank_bytes=134217728, source_precharge=(misses-cpu)*4*1024**2,
            zero_buffers=18, zero_bytes=884736, io_inspected_bytes=32768,
            metadata_bytes=1024, snapshot_bytes=435200,
            max_tree_error=0., max_legacy_error=0., max_core_error=0., timing_ns=None)
        samples = []
        for mode in ["concurrent", "cpu-first", "cpu-first", "concurrent"]:
            timed = copy.deepcopy(report)
            timed["timing_ns"] = dict(cpu=10 if cpu else 0, gpu=10, joined=20, bridge_and_merge=5)
            samples.append(dict(mode=mode, report=timed, candidate_completed_wall_ns=30,
                acceptance_cleanup_wall_ns=10, full_run_including_references_wall_ns=50))
        receipt["points"].append(dict(down_dtype=down, cpu_misses=cpu,
            numeric_modes=["concurrent", "cpu-first"], numeric=[copy.deepcopy(report), copy.deepcopy(report)], samples=samples))
    output = LAB.MARKER + "\nverified_lab_limits swap_max=0 core_soft=0 core_hard=0 memory_max=8589934592"
    return receipt, output


class Acceptance(unittest.TestCase):
    def test_requires_separate_resident_and_upload_costs(self):
        receipt, output = fixture()
        LAB.validate_receipt(receipt, "nvidia", output)
        receipt["points"][0]["samples"][0]["report"]["source_precharge"] = 41943040
        with self.assertRaises(ValueError):
            LAB.validate_receipt(receipt, "nvidia", output)

    def test_rejects_wrong_cpu_backend_or_missing_gate(self):
        for key, value in [("backend", "avx2"), ("numeric_gates", 10), ("owned_cleanup_accepted", False)]:
            receipt, output = fixture()
            receipt[key] = value
            with self.assertRaises(ValueError):
                LAB.validate_receipt(receipt, "nvidia", output)

    def test_rejects_enqueue_only_or_nonfinite_measurement(self):
        receipt, output = fixture()
        receipt["points"][2]["samples"][0]["candidate_completed_wall_ns"] = 1
        with self.assertRaises(ValueError):
            LAB.validate_receipt(receipt, "nvidia", output)
        receipt, output = fixture()
        receipt["points"][1]["numeric"][1]["max_core_error"] = float("nan")
        with self.assertRaises(ValueError):
            LAB.validate_receipt(receipt, "nvidia", output)


if __name__ == "__main__":
    unittest.main()
