import copy
import contextlib
import errno
import importlib.util
import io
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
        # Represent the current binary's explicit scalar mode; the archived
        # receipt itself is unchanged and its offline compatibility is separate.
        self.receipt.update(native_decode=False, native_decode_steps=0,
                            scalar_decode_steps=len(self.receipt["public_generated_ids"]) - 1,
                            native_decode_projection_tile=8)
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
        self.assertEqual(env["KUNO_LAB_NATIVE_DECODE"], "0")
        self.assertEqual(env["KUNO_LAB_NATIVE_DECODE_TILE"], "8")
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

    def test_live_performance_worker_requires_new_explicit_decode_fields(self):
        archived = json.loads((HERE / "receipts/rust-native-model-full-intel-20261005.json").read_text())["report"]
        with self.assertRaises(ValueError):
            LANE.accepted_result(archived, self.marker(52) + self.output, "intel", 52,
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
                           ("native_decode", True), ("native_decode_steps", 127),
                           ("scalar_decode_steps", True),
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

    def supervisor_fixture(self, directory, native_comparison=False, tile_comparison=False):
        unit = "kuno-public-performance-fixture.service"
        cgroup = pathlib.Path(directory) / unit
        cgroup.mkdir()
        (cgroup / "memory.max").write_text(str((52 if native_comparison or tile_comparison else 8) * 1024**3))
        (cgroup / "memory.swap.max").write_text("0")
        (cgroup / "memory.swap.current").write_text("0")
        plan_path = pathlib.Path(directory) / "plan.json"
        plan_path.write_text("{}")
        plan = {"unit": unit, "scope": LANE.TILE_SCOPE if tile_comparison else LANE.DECODE_SCOPE if native_comparison else LANE.MEMORY_SCOPE,
                "caps_gib": list(LANE.DECODE_CAPS if native_comparison or tile_comparison else LANE.ABBA),
                "native_decode_flags": list(LANE.TILE_FLAGS if tile_comparison else LANE.DECODE_FLAGS if native_comparison else (False,) * 4),
                "native_decode_tiles": list(LANE.TILE_TILES if tile_comparison else (8,) * 4),
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
            self.assertEqual(receipt["failure_stage"], "sample-worker")
            self.assertEqual(receipt["failed_sample_index"], 2)
            self.assertEqual(len(receipt["results"]), 1)
            self.assertNotIn("missing cleanup", json.dumps(receipt))

    def test_scopes_pin_typed_caps_and_decode_flags_without_cross_mode_fallback(self):
        for scope, caps, flags, tiles in ((LANE.MEMORY_SCOPE, LANE.ABBA, (False,) * 4, (8,) * 4),
                                          (LANE.DECODE_SCOPE, LANE.DECODE_CAPS, LANE.DECODE_FLAGS, (8,) * 4),
                                          (LANE.TILE_SCOPE, LANE.DECODE_CAPS, LANE.TILE_FLAGS, LANE.TILE_TILES)):
            original = {"scope": scope, "caps_gib": list(caps), "native_decode_flags": list(flags),
                        "native_decode_tiles": list(tiles)}
            self.assertEqual(LANE.experiment_schedule(original), tuple(zip(caps, flags)))
            for field, bad in (("scope", "unknown"), ("caps_gib", [52] * 3),
                               ("caps_gib", [float(cap) for cap in caps]),
                               ("caps_gib", [True] * 4),
                               ("native_decode_flags", [int(flag) for flag in flags]),
                               ("native_decode_flags", list(flags)[:-1]),
                               ("native_decode_flags", [not flag for flag in flags]),
                               ("native_decode_tiles", [True] * 4),
                               ("native_decode_tiles", [float(tile) for tile in tiles]),
                               ("native_decode_tiles", [4] * 4),
                               ("native_decode_tiles", list(tiles)[:-1])):
                plan = copy.deepcopy(original)
                plan[field] = bad
                with self.assertRaises(ValueError):
                    LANE.experiment_schedule(plan)
            swapped = copy.deepcopy(original)
            swapped["caps_gib"] = list(LANE.DECODE_CAPS if scope == LANE.MEMORY_SCOPE else LANE.ABBA)
            with self.assertRaises(ValueError):
                LANE.experiment_schedule(swapped)
            del original["native_decode_flags"]
            if scope == LANE.MEMORY_SCOPE:
                self.assertEqual(LANE.experiment_schedule(original), tuple(zip(caps, flags)))
            else:
                with self.assertRaises(ValueError):
                    LANE.experiment_schedule(original)

    def test_index_and_cap_select_one_fixed_child_and_environment(self):
        plan = {"scope": LANE.DECODE_SCOPE, "caps_gib": list(LANE.DECODE_CAPS),
                "native_decode_flags": list(LANE.DECODE_FLAGS), "model": {"path": "/original/blob"}}
        for index, native in enumerate(LANE.DECODE_FLAGS, 1):
            self.assertEqual(LANE.selected_sample(plan, index, 52), (52, native))
            env = LANE.worker_environment(plan, "/public/receipt", index)
            self.assertEqual(env["KUNO_LAB_NATIVE_DECODE"], "1" if native else "0")
            self.assertEqual(env["KUNO_LAB_NATIVE_DECODE_TILE"], "8")
        for index in (True, False, 0, 5, -1, 2.0, "2", None):
            with self.assertRaises(ValueError):
                LANE.selected_sample(plan, index)
        for cap in (8, 51, 53, 52.0, True):
            with self.assertRaises(ValueError):
                LANE.selected_sample(plan, 2, cap)
        with self.assertRaises(ValueError):
            LANE.worker_environment(plan, "/public/receipt")

    def test_indexed_guard_checks_real_limits_and_actual_decode_environment_before_exec(self):
        with tempfile.TemporaryDirectory() as directory:
            _, plan, cgroup, _ = self.supervisor_fixture(directory, True)
            plan.update(card="intel", executable="/public/frozen")
            for cap, index, env, actual_cap, accepted in (
                (52, 2, "1", 52, True), (52, 1, "0", 52, True),
                (52, 2, "0", 52, False), (52, 1, "1", 52, False),
                (8, 2, "1", 8, False), (52, 2, "1", 8, False),
                (52, True, "1", 52, False), (52, 0, "1", 52, False),
                (52, 2, "true", 52, False),
            ):
                (cgroup / "memory.max").write_text(str(actual_cap * 1024**3))
                with mock.patch.object(LANE, "load_plan", return_value=plan), \
                     mock.patch.object(LANE, "cgroup_path", return_value=cgroup), \
                     mock.patch.object(LANE.resource, "getrlimit", return_value=(0, 0)), \
                     mock.patch.dict("os.environ", {"KUNO_LAB_NATIVE_DECODE": env, "KUNO_LAB_NATIVE_DECODE_TILE": "8"}), \
                     mock.patch.object(LANE.os, "execv") as execute, \
                     contextlib.redirect_stdout(io.StringIO()) as output:
                    if accepted:
                        LANE.guard("/public/plan.json", cap, index)
                        execute.assert_called_once_with("/public/frozen", ["/public/frozen", "intel", "resident-rainwater"])
                        self.assertIn(f"sample_index={index} native_decode={int(env)} native_decode_tile=8", output.getvalue())
                    else:
                        with self.assertRaises(ValueError):
                            LANE.guard("/public/plan.json", cap, index)
                        execute.assert_not_called()

    def test_native_acceptance_needs_independent_mode_index_and_unchanged_audits(self):
        receipt = copy.deepcopy(self.receipt)
        receipt.update(native_decode=True, native_decode_steps=127, scalar_decode_steps=0)
        receipt["owned_zero_audits"][0].update(source_buffers=0, source_bytes=0)
        proof = "verified_performance_sample sample_index=2 native_decode=1 native_decode_tile=8\n"
        output = self.marker(52) + proof + self.output
        LANE.accepted_result(receipt, output, "intel", 52, 0, self.before, self.before,
                             native_decode=True, index=2)
        for mode, index, cap, text in ((False, 2, 52, output), (True, None, 52, output),
                                       (True, True, 52, output), (True, 3, 52, output),
                                       (True, 2, 8, self.marker(8) + proof + self.output),
                                       (True, 2, 52, self.marker(52) + self.output)):
            with self.assertRaises(ValueError):
                LANE.accepted_result(receipt, text, "intel", cap, 0, self.before, self.before,
                                     native_decode=mode, index=index)
        for owner in range(4):
            changed = copy.deepcopy(receipt)
            changed["owned_zero_audits"][owner]["main_bytes"] -= 1
            with self.assertRaises(ValueError):
                LANE.accepted_result(changed, output, "intel", 52, 0, self.before, self.before,
                                     native_decode=True, index=2)
        for field, bad in (("generation_ms", 0), ("generation_ms", True),
                           ("generation_ms", 1.0), ("prefill_ms", -1), ("prefill_ms", False)):
            changed = copy.deepcopy(receipt)
            changed[field] = bad
            with self.assertRaises(ValueError):
                LANE.accepted_result(changed, output, "intel", 52, 0, self.before, self.before,
                                     native_decode=True, index=2)

    def test_native_supervisor_stays_in_one_52g_unit_and_preserves_sample_modes(self):
        with tempfile.TemporaryDirectory() as directory:
            plan_path, plan, cgroup, setter = self.supervisor_fixture(directory, True)
            observed = []
            def child(current_plan, current_path, index, cap, current_cgroup):
                self.assertEqual((cgroup / "memory.max").read_text(), str(52 * 1024**3))
                self.assertIs(current_plan, plan)
                self.assertEqual(current_path, plan_path)
                self.assertEqual(current_cgroup, cgroup)
                native = LANE.selected_sample(plan, index, cap)[1]
                observed.append((index, cap, native))
                return {"accepted": True, "sample_index": index, "cap_gib": cap,
                        "native_decode": native, "decode_tokens_per_second": 2.0}
            with mock.patch.object(LANE, "load_plan", return_value=plan), \
                 mock.patch.object(LANE, "cgroup_path", return_value=cgroup), \
                 mock.patch.object(LANE.resource, "getrlimit", return_value=(0, 0)), \
                 mock.patch.object(LANE, "snapshot", return_value={"numeric": 1}), \
                 mock.patch.object(LANE.subprocess, "run", side_effect=setter), \
                 mock.patch.object(LANE, "run_child", side_effect=child), \
                 mock.patch.object(LANE, "model_identity", return_value=plan["model"]), \
                 contextlib.redirect_stdout(io.StringIO()):
                LANE.supervise(plan_path)
            self.assertEqual(observed, [(i, 52, native) for i, native in enumerate(LANE.DECODE_FLAGS, 1)])
            record = json.loads((pathlib.Path(directory) / "abba-receipt.json").read_text())
            self.assertTrue(record["accepted"])
            self.assertEqual(record["native_decode_flags"], list(LANE.DECODE_FLAGS))
            self.assertEqual(record["caps_gib"], [52] * 4)
            self.assertFalse(record["true_hot_full_request_implemented"])

    def test_native_supervisor_rejects_initial_8g_before_changing_any_unit(self):
        with tempfile.TemporaryDirectory() as directory:
            plan_path, plan, cgroup, _ = self.supervisor_fixture(directory, True)
            (cgroup / "memory.max").write_text(str(8 * 1024**3))
            with mock.patch.object(LANE, "load_plan", return_value=plan), \
                 mock.patch.object(LANE, "cgroup_path", return_value=cgroup), \
                 mock.patch.object(LANE.resource, "getrlimit", return_value=(0, 0)), \
                 mock.patch.object(LANE.subprocess, "run") as service, \
                 mock.patch.object(LANE, "run_child") as child:
                with self.assertRaises(ValueError):
                    LANE.supervise(plan_path)
                service.assert_not_called()
                child.assert_not_called()

    def full_decode_plan(self, directory):
        executable = pathlib.Path(directory) / "public-binary"
        executable.write_bytes(b"public fixture executable")
        executable.chmod(0o500)
        return {**LANE.FIXED_PLAN, "scope": LANE.DECODE_SCOPE, "caps_gib": [52] * 4,
                "native_decode_flags": list(LANE.DECODE_FLAGS), "card": "intel",
                "unit": "kuno-public-performance-fixture.service",
                "model": {"path": "/original/blob", "device": 1, "inode": 2,
                          "size": LANE.MODEL_SIZE, "mtime_ns": 3},
                "executable": str(executable), "executable_sha256": LANE.sha(executable.read_bytes()),
                "cargo_artifacts_sha256": "a" * 64, "cargo_executable_at_build": "/public/build/binary",
                "fixture_sha256": LANE.sha((HERE / "qwen38-baseline.json").read_bytes()),
                "reference_sha256": LANE.sha((HERE / "receipts/rust-full-resident-rainwater-intel-20261005.json").read_bytes()),
                "supervisor_sha256": LANE.sha((HERE / "run-native-performance-lab.py").read_bytes()),
                "source_head_at_freeze": "b" * 40}

    def test_new_plan_keys_types_fixture_and_artifact_identities_are_strict(self):
        with tempfile.TemporaryDirectory() as directory:
            plan = self.full_decode_plan(directory)
            path = pathlib.Path(directory) / "plan.json"
            path.write_text(json.dumps(plan))
            with mock.patch.object(LANE, "model_identity", return_value=plan["model"]):
                self.assertEqual(LANE.load_plan(path), plan)
                changed_values = [(key, None) for key in plan]
                changed_values.extend((
                    ("extra_host_limit", 100), ("prompt_tokens", 76), ("native_count", 1),
                    ("model_unloaded_before_each_child", 1), ("fixture_sha256", "c" * 64),
                    ("reference_sha256", "c" * 64), ("supervisor_sha256", "c" * 64),
                    ("executable_sha256", "c" * 64), ("source_head_at_freeze", "private-text"),
                ))
                for key, bad in changed_values:
                    changed = copy.deepcopy(plan)
                    changed[key] = bad
                    path.write_text(json.dumps(changed))
                    with self.assertRaises(ValueError):
                        LANE.load_plan(path)
                for key in plan:
                    changed = copy.deepcopy(plan)
                    del changed[key]
                    path.write_text(json.dumps(changed))
                    with self.assertRaises(ValueError):
                        LANE.load_plan(path)
                historical = copy.deepcopy(plan)
                historical.update(scope=LANE.MEMORY_SCOPE, caps_gib=list(LANE.ABBA))
                del historical["native_decode_flags"]
                path.write_text(json.dumps(historical))
                self.assertEqual(LANE.load_plan(path), historical)

    def test_child_records_selected_mode_numeric_samples_and_actual_completed_rate(self):
        with tempfile.TemporaryDirectory() as directory:
            plan_path, plan, cgroup, _ = self.supervisor_fixture(directory, True)
            plan["card"] = "intel"
            receipt = copy.deepcopy(self.receipt)
            receipt.update(native_decode=True, native_decode_steps=127, scalar_decode_steps=0,
                           generation_ms=63500)
            receipt["owned_zero_audits"][0].update(source_buffers=0, source_bytes=0)
            stdout = self.marker(52) + "verified_performance_sample sample_index=2 native_decode=1 native_decode_tile=8\n" + self.output
            state = {**self.before, "memory.current": 123}
            def launch(command, **options):
                self.assertEqual(command[-3:], [str(plan_path), "52", "2"])
                self.assertEqual(options["env"]["KUNO_LAB_NATIVE_DECODE"], "1")
                (pathlib.Path(directory) / "sample-2-cap-52.json").write_text(json.dumps(receipt))
                return SimpleNamespace(stdout=io.BytesIO(stdout.encode()), poll=lambda: 0,
                                       returncode=0, pid=123)
            with mock.patch.object(LANE.subprocess, "Popen", side_effect=launch), \
                 mock.patch.object(LANE, "snapshot", return_value=state), \
                 mock.patch.object(LANE.resource, "getrlimit", return_value=(0, 0)):
                result = LANE.run_child(plan, plan_path, 2, 52, cgroup)
            self.assertTrue(result["accepted"])
            self.assertIs(result["native_decode"], True)
            self.assertEqual(result["decode_tokens_per_second"], 2.0)
            self.assertEqual((result["native_decode_steps"], result["scalar_decode_steps"]), (127, 0))
            sample = json.loads((pathlib.Path(directory) / "sample-2-cap-52.cgroup.json").read_text())
            self.assertEqual(sample["samples"], [])
            self.assertNotIn("public_resident", json.dumps(sample))

    def test_launcher_selects_52g_only_for_explicit_native_comparison(self):
        for native, tile_comparison in ((False, False), (True, False), (False, True)):
            with tempfile.TemporaryDirectory() as directory:
                build = pathlib.Path(directory) / "target/release/kuno-text-lab"
                build.parent.mkdir(parents=True)
                build.write_bytes(b"fixed public test executable")
                transcript = pathlib.Path(directory) / "cargo-artifacts.jsonl"
                transcript.write_text(json.dumps({"reason": "compiler-artifact", "target": {"name": "kuno-text-lab"},
                                                  "executable": str(build)}) + "\n")
                args = SimpleNamespace(run_id="fixture", output=pathlib.Path(directory) / "records",
                                       native_decode_comparison=native, native_decode_tile_comparison=tile_comparison, cargo_artifacts=transcript,
                                       model=pathlib.Path("/original/blob"), card="intel")
                model = {"path": "/original/blob", "device": 1, "inode": 2,
                         "size": LANE.MODEL_SIZE, "mtime_ns": 3}
                with mock.patch.object(LANE.ACCEPTANCE, "cargo_binary",
                                       return_value=(build, LANE.sha(build.read_bytes()), LANE.sha(transcript.read_bytes()))), \
                     mock.patch.object(LANE, "model_identity", return_value=model), \
                     mock.patch.object(LANE.subprocess, "check_output", return_value="a" * 40), \
                     mock.patch.object(LANE.subprocess, "run") as service, \
                     contextlib.redirect_stdout(io.StringIO()):
                    LANE.launch(args)
                    plan = LANE.load_plan(args.output / "plan.json")
                self.assertEqual(plan["scope"], LANE.TILE_SCOPE if tile_comparison else LANE.DECODE_SCOPE if native else LANE.MEMORY_SCOPE)
                self.assertEqual(plan["native_decode_flags"], list(LANE.TILE_FLAGS if tile_comparison else LANE.DECODE_FLAGS if native else (False,) * 4))
                self.assertEqual(plan["native_decode_tiles"], list(LANE.TILE_TILES if tile_comparison else (8,) * 4))
                self.assertIn(f"--property=MemoryMax={52 if native or tile_comparison else 8}G", service.call_args.args[0])
                self.assertIn("--property=MemorySwapMax=0", service.call_args.args[0])
                self.assertIn("--property=LimitCORE=0", service.call_args.args[0])

    def test_process_io_exit_race_keeps_only_numeric_errno_and_does_not_mask_other_io(self):
        with tempfile.TemporaryDirectory() as directory:
            cgroup = pathlib.Path(directory)
            for name in ("memory.current", "memory.peak", "memory.swap.current", "memory.max"):
                (cgroup / name).write_text("0")
            (cgroup / "memory.stat").write_text("file 0\n")
            (cgroup / "memory.events").write_text("oom 0\n")
            (cgroup / "memory.pressure").write_text("some avg10=0.00 avg60=0.00 avg300=0.00 total=0\n")
            original_read = pathlib.Path.read_text
            for number in (errno.ESRCH, errno.ENOENT, errno.EACCES, errno.EPERM, errno.EIO):
                def read(path, *args, **options):
                    if str(path) == "/proc/123/io":
                        raise OSError(number, "raw error must not be retained")
                    return original_read(path, *args, **options)
                with mock.patch.object(LANE.pathlib.Path, "read_text", read):
                    if number == errno.EIO:
                        with self.assertRaises(OSError):
                            LANE.snapshot(cgroup, 123)
                    else:
                        value = LANE.snapshot(cgroup, 123)
                        self.assertIsNone(value["process_io"])
                        self.assertEqual(value["process_io_errno"], number)
                        self.assertNotIn("raw error", json.dumps(value))

    def test_tile_guard_requires_the_indexed_public_tile_before_execution(self):
        with tempfile.TemporaryDirectory() as directory:
            _, plan, cgroup, _ = self.supervisor_fixture(directory, tile_comparison=True)
            plan.update(card="intel", executable="/public/frozen")
            for index, raw_tile, permitted in ((1, "8", True), (2, "1", True), (3, "1", True),
                                               (4, "8", True), (1, "1", False), (2, "8", False),
                                               (2, "01", False), (2, None, False)):
                env = {"KUNO_LAB_NATIVE_DECODE": "1"}
                if raw_tile is not None:
                    env["KUNO_LAB_NATIVE_DECODE_TILE"] = raw_tile
                with mock.patch.object(LANE, "load_plan", return_value=plan), \
                     mock.patch.object(LANE, "cgroup_path", return_value=cgroup), \
                     mock.patch.object(LANE.resource, "getrlimit", return_value=(0, 0)), \
                     mock.patch.dict("os.environ", env, clear=True), \
                     mock.patch.object(LANE.os, "execv") as execute, \
                     contextlib.redirect_stdout(io.StringIO()):
                    if permitted:
                        LANE.guard("/public/plan", 52, index)
                        execute.assert_called_once()
                    else:
                        with self.assertRaises(ValueError):
                            LANE.guard("/public/plan", 52, index)
                        execute.assert_not_called()

    def test_tile_scope_requires_full_typed_plan_and_preserves_native_only_52g_schedule(self):
        with tempfile.TemporaryDirectory() as directory:
            plan = self.full_decode_plan(directory)
            plan.update(scope=LANE.TILE_SCOPE, native_decode_flags=list(LANE.TILE_FLAGS),
                        native_decode_tiles=list(LANE.TILE_TILES))
            path = pathlib.Path(directory) / "plan.json"
            path.write_text(json.dumps(plan))
            with mock.patch.object(LANE, "model_identity", return_value=plan["model"]):
                self.assertEqual(LANE.load_plan(path), plan)
                for index, tile in enumerate(LANE.TILE_TILES, 1):
                    self.assertEqual(LANE.selected_sample(plan, index), (52, True))
                    self.assertEqual(LANE.selected_tile(plan, index), tile)
                    self.assertEqual(LANE.worker_environment(plan, "/receipt", index)
                                     ["KUNO_LAB_NATIVE_DECODE_TILE"], str(tile))
                for changed in ({key: value for key, value in plan.items() if key != "native_decode_tiles"},
                                {**plan, "native_decode_flags": list(LANE.DECODE_FLAGS)},
                                {**plan, "native_decode_tiles": [8] * 4},
                                {**plan, "caps_gib": list(LANE.ABBA)}):
                    path.write_text(json.dumps(changed))
                    with self.assertRaises(ValueError):
                        LANE.load_plan(path)

    def test_tile_acceptance_cannot_follow_receipt_selected_tile_or_scalar_lane(self):
        receipt = copy.deepcopy(self.receipt)
        receipt.update(native_decode=True, native_decode_steps=127, scalar_decode_steps=0,
                       native_decode_projection_tile=1)
        receipt["owned_zero_audits"][0].update(source_buffers=0, source_bytes=0)
        proof = "verified_performance_sample sample_index=2 native_decode=1 native_decode_tile=1\n"
        output = self.marker(52) + proof + self.output + "\n" + "\n".join(LANE.ACCEPTANCE.TILE_ONE_COMPLETION)
        LANE.accepted_result(receipt, output, "intel", 52, 0, self.before, self.before,
                             native_decode=True, native_decode_tile=1, index=2)
        for selected in (8, True, 1.0, "1", 4):
            with self.assertRaises(ValueError):
                LANE.accepted_result(receipt, output, "intel", 52, 0, self.before, self.before,
                                     native_decode=True, native_decode_tile=selected, index=2)
        with self.assertRaises(ValueError):
            LANE.accepted_result(receipt, output, "intel", 52, 0, self.before, self.before,
                                 native_decode=False, native_decode_tile=1, index=2)

    def test_tile_child_retains_all_nine_exact_proofs_and_rejects_each_missing_recovery_proof(self):
        receipt = copy.deepcopy(self.receipt)
        receipt.update(native_decode=True, native_decode_steps=127, scalar_decode_steps=0,
                       native_decode_projection_tile=1, generation_ms=63500)
        receipt["owned_zero_audits"][0].update(source_buffers=0, source_bytes=0)
        proof = "verified_performance_sample sample_index=2 native_decode=1 native_decode_tile=1\n"
        complete = self.marker(52) + proof + self.output + "\n" + "\n".join(LANE.ACCEPTANCE.TILE_ONE_COMPLETION)
        for missing in LANE.ACCEPTANCE.TILE_ONE_COMPLETION:
            with self.assertRaises(ValueError):
                LANE.accepted_result(receipt, complete.replace(missing, ""), "intel", 52, 0,
                                     self.before, self.before, native_decode=True,
                                     native_decode_tile=1, index=2)
        with tempfile.TemporaryDirectory() as directory:
            plan_path, plan, cgroup, _ = self.supervisor_fixture(directory, tile_comparison=True)
            plan["card"] = "intel"
            state = {**self.before, "memory.current": 123}
            # Prefix/suffix payloads must not fill the bounded proof list.
            stdout = "\n".join(message + " private-test-fragment" for message in LANE.ACCEPTANCE.TILE_ONE_COMPLETION)
            stdout += "\n" + complete
            def launch(command, **options):
                self.assertEqual(options["env"]["KUNO_LAB_NATIVE_DECODE_TILE"], "1")
                (pathlib.Path(directory) / "sample-2-cap-52.json").write_text(json.dumps(receipt))
                return SimpleNamespace(stdout=io.BytesIO(stdout.encode()), poll=lambda: 0,
                                       returncode=0, pid=123)
            with mock.patch.object(LANE.subprocess, "Popen", side_effect=launch), \
                 mock.patch.object(LANE, "snapshot", return_value=state), \
                 mock.patch.object(LANE.resource, "getrlimit", return_value=(0, 0)):
                result = LANE.run_child(plan, plan_path, 2, 52, cgroup)
            self.assertTrue(result["accepted"])
            self.assertEqual(result["native_decode_steps"], 127)
            self.assertEqual(result["decode_tokens_per_second"], 2.0)
            self.assertNotIn("private-test-fragment", json.dumps(result))


if __name__ == "__main__":
    unittest.main()
