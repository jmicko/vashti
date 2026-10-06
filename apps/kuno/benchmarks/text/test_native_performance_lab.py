import copy
import importlib.util
import json
import pathlib
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

HERE = pathlib.Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("performance_lane", HERE / "run-native-performance-lab.py")
LANE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(LANE)


class PerformanceLaneTests(unittest.TestCase):
    def setUp(self):
        self.receipt = json.loads((HERE / "receipts/rust-native-model-full-intel-20261005.json").read_text())["report"]
        self.output = "\n".join((
            "public_mid_token_cancel layer=0 erased=true",
            "public_mid_token_cancel layer=3 erased=true",
            "public_resident_cancel_erased=true poisoned_retry_rejected=true fresh_first_token_matches=true",
            "public_resident_owned_cleanup=true context_retired=false",
        ))
        self.before = {"memory.events": {"oom": 0, "oom_kill": 0, "oom_group_kill": 0}}

    def marker(self, cap):
        return f"verified_performance_limits swap_max=0 core_soft=0 core_hard=0 memory_max={cap * 1024**3}\n"

    def test_abba_caps_are_fixed(self):
        self.assertEqual(LANE.ABBA, (8, 52, 52, 8))

    def test_environment_has_only_fixed_public_options(self):
        with mock.patch.dict("os.environ", {"KUNO_LAB_EXPERT_SLOTS": "96", "PRIVATE_TOKEN": "never-forward"}):
            env = LANE.worker_environment({"model": {"path": "/original/blob"}}, "/public/numeric.json")
        self.assertNotIn("PRIVATE_TOKEN", env)
        self.assertNotIn("KUNO_LAB_EXPERT_SLOTS", env)
        self.assertEqual(env["KUNO_LAB_NATIVE_BATCH_COUNT"], "8")
        self.assertEqual(env["KUNO_LAB_TRANSIENT_IO"], "1")
        self.assertEqual(env["KUNO_LAB_OWNED_ZERO_AUDIT"], "1")
        self.assertEqual(env["KUNO_LAB_CANCEL_CHECK"], "1")
        self.assertEqual(env["KUNO_LAB_SMALL_CPU"], "0")
        self.assertEqual(env["KUNO_LAB_ORDERED_PUBLIC_UPLOAD"], "0")

    def test_numeric_stat_and_proc_io(self):
        self.assertEqual(LANE.numeric_map("file 100\nworkingset_refault_file 3\nread_bytes 9\n"),
                         {"file": 100, "workingset_refault_file": 3, "read_bytes": 9})
        for text in ("file x", "file 1\nfile 2", "prompt-text 4"):
            with self.assertRaises(ValueError):
                LANE.numeric_map(text)

    def test_pressure_values_and_totals(self):
        value = LANE.pressure_map("some avg10=1.25 avg60=0.50 avg300=0.10 total=123\n"
                                  "full avg10=0.00 avg60=0.00 avg300=0.00 total=2\n")
        self.assertEqual(value["some"]["total"], 123)
        self.assertEqual(value["some"]["avg10"], 1.25)
        self.assertEqual(value["full"]["total"], 2)

    def test_actual_unit_limits_are_checked_for_both_caps(self):
        with tempfile.TemporaryDirectory() as directory:
            unit = "kuno-public-performance-fixture.service"
            cgroup = pathlib.Path(directory) / unit
            cgroup.mkdir()
            (cgroup / "memory.swap.max").write_text("0")
            (cgroup / "memory.swap.current").write_text("0")
            for cap in (8, 52):
                (cgroup / "memory.max").write_text(str(cap * 1024**3))
                LANE.verify_limits(cgroup, unit, cap, (0, 0))
                with self.assertRaises(ValueError):
                    LANE.verify_limits(cgroup, unit, 52 if cap == 8 else 8, (0, 0))
            for core in ((0, 1), (1, 1)):
                with self.assertRaises(ValueError):
                    LANE.verify_limits(cgroup, unit, 52, core)
            with self.assertRaises(ValueError):
                LANE.verify_limits(cgroup, "kuno.service", 52, (0, 0))
            (cgroup / "memory.swap.current").write_text("1")
            with self.assertRaises(ValueError):
                LANE.verify_limits(cgroup, unit, 52, (0, 0))

    def test_valid_full_model_receipt_accepts_actual_8_and_52_guards(self):
        for cap in (8, 52):
            LANE.accepted_result(self.receipt, self.marker(cap) + self.output, "intel", cap,
                                 0, self.before, self.before)

    def test_failed_child_missing_cleanup_or_wrong_guard_reject(self):
        for output, status in ((self.marker(52) + self.output, 1),
                               (self.marker(8) + self.output, 0),
                               (self.marker(52) + "public_mid_token_cancel layer=0 erased=true", 0)):
            with self.assertRaises(ValueError):
                LANE.accepted_result(self.receipt, output, "intel", 52, status, self.before, self.before)

    def test_oom_event_or_modified_model_options_reject(self):
        after = copy.deepcopy(self.before)
        after["memory.events"]["oom_kill"] = 1
        with self.assertRaises(ValueError):
            LANE.accepted_result(self.receipt, self.marker(52) + self.output, "intel", 52,
                                 0, self.before, after)
        for key, value in (("resident_runtime_io_limit_bytes", 100000000),
                           ("seed", 1), ("native_public_chunk_count", 1),
                           ("resident_owned_zero_audit_enabled", False),
                           ("public_generated_ids", [1]), ("owned_zero_audits", [])):
            receipt = copy.deepcopy(self.receipt)
            receipt[key] = value
            with self.assertRaises(ValueError):
                LANE.accepted_result(receipt, self.marker(52) + self.output, "intel", 52,
                                     0, self.before, self.before)

    def test_original_correctness_validator_still_requires_real_8g_guard(self):
        LANE.ACCEPTANCE.validate_receipt(self.receipt, "intel", "resident-rainwater", 8,
                                        "verified_lab_limits swap_max=0 core_soft=0 core_hard=0 memory_max=8589934592\n" + self.output)
        for output in (self.output, self.marker(8) + self.output,
                       "verified_lab_limits swap_max=0 core_soft=0 core_hard=0 memory_max=55834574848\n" + self.output):
            with self.assertRaises(ValueError):
                LANE.ACCEPTANCE.validate_receipt(self.receipt, "intel", "resident-rainwater", 8, output)

    def supervisor_fixture(self, directory):
        unit = "kuno-public-performance-fixture.service"
        cgroup = pathlib.Path(directory) / unit
        cgroup.mkdir()
        (cgroup / "memory.max").write_text(str(8 * 1024**3))
        (cgroup / "memory.swap.max").write_text("0")
        (cgroup / "memory.swap.current").write_text("0")
        plan_path = pathlib.Path(directory) / "plan.json"
        plan_path.write_text("{}")
        plan = {"unit": unit, "scope": "fixed-public-native-cgroup-abba-only",
                "model": {"path": "/declared/original"}}
        def set_property(command, **kwargs):
            self.assertEqual(command[:5], ["systemctl", "--user", "set-property", "--runtime", unit])
            cap = int(command[5].removeprefix("MemoryMax=").removesuffix("G"))
            (cgroup / "memory.max").write_text(str(cap * 1024**3))
            return SimpleNamespace(returncode=0)
        return plan_path, plan, cgroup, set_property

    def test_supervisor_runs_all_caps_in_one_cgroup_and_only_changes_declared_unit(self):
        with tempfile.TemporaryDirectory() as directory:
            plan_path, plan, cgroup, setter = self.supervisor_fixture(directory)
            observed = []
            def child(current_plan, current_path, index, cap, current_cgroup):
                self.assertIs(current_plan, plan)
                self.assertEqual(current_path, plan_path)
                self.assertEqual(current_cgroup, cgroup)
                self.assertEqual((cgroup / "memory.max").read_text(), str(cap * 1024**3))
                observed.append((index, cap))
                return {"accepted": True, "sample_index": index, "cap_gib": cap}
            with mock.patch.object(LANE, "load_plan", return_value=plan), \
                 mock.patch.object(LANE, "cgroup_path", return_value=cgroup), \
                 mock.patch.object(LANE.resource, "getrlimit", return_value=(0, 0)), \
                 mock.patch.object(LANE, "snapshot", return_value={"numeric": 1}), \
                 mock.patch.object(LANE.subprocess, "run", side_effect=setter), \
                 mock.patch.object(LANE, "run_child", side_effect=child), \
                 mock.patch.object(LANE, "model_identity", return_value=plan["model"]):
                LANE.supervise(plan_path)
            self.assertEqual(observed, [(1, 8), (2, 52), (3, 52), (4, 8)])
            receipt = json.loads((pathlib.Path(directory) / "abba-receipt.json").read_text())
            self.assertTrue(receipt["accepted"])
            self.assertEqual(len(receipt["results"]), 4)
            self.assertFalse(receipt["true_hot_full_request_implemented"])

    def test_supervisor_aborts_at_failed_child_and_preserves_completed_numeric_results(self):
        with tempfile.TemporaryDirectory() as directory:
            plan_path, plan, cgroup, setter = self.supervisor_fixture(directory)
            with mock.patch.object(LANE, "load_plan", return_value=plan), \
                 mock.patch.object(LANE, "cgroup_path", return_value=cgroup), \
                 mock.patch.object(LANE.resource, "getrlimit", return_value=(0, 0)), \
                 mock.patch.object(LANE, "snapshot", return_value={"numeric": 1}), \
                 mock.patch.object(LANE.subprocess, "run", side_effect=setter), \
                 mock.patch.object(LANE, "run_child", side_effect=[{"accepted": True}, ValueError("missing cleanup")]) as worker:
                with self.assertRaises(ValueError):
                    LANE.supervise(plan_path)
            self.assertEqual(worker.call_count, 2)
            receipt = json.loads((pathlib.Path(directory) / "abba-receipt.json").read_text())
            self.assertFalse(receipt["accepted"])
            self.assertEqual(len(receipt["results"]), 1)
            self.assertNotIn("missing cleanup", json.dumps(receipt))


if __name__ == "__main__":
    unittest.main()
