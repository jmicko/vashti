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


class SecondaryControls(unittest.TestCase):
    def test_actual_secondary_work_and_all_model_owner_closes_required(self):
        value={"secondary_experts":True,"secondary_expert_calls":35,"secondary_gpu_calls":20,
            "warm_expert_hits":2125,"completed_cpu_misses":1200,
            "secondary_bank_shards":[32,16],"secondary_bank_allocation_bytes":9*1024**3,
            "secondary_bank_setup_ms":2000,"secondary_weight_upload_bytes":1000000}
        output="\n".join(["public_secondary_closed=true owned_buffers=9 owned_bytes=1835008",
            "public_secondary_original_rows families=2 boundary_slots=4 numerical_and_full_owners=true"]*5)
        LAB.validate_secondary(value,True,output,False)
        for field,bad in [("secondary_expert_calls",0),("secondary_gpu_calls",36),
                         ("secondary_bank_shards",[48]),("secondary_bank_allocation_bytes",1),
                         ("secondary_weight_upload_bytes",0),("secondary_experts",False),
                         ("warm_expert_hits",2126),("completed_cpu_misses",1199)]:
            changed=dict(value);changed[field]=bad
            with self.assertRaises(ValueError):LAB.validate_secondary(changed,True,output,False)
        with self.assertRaises(ValueError):LAB.validate_secondary(value,True,output.split("\n",1)[1],False)
        with self.assertRaises(ValueError):LAB.validate_secondary(value,False,output,False)
        with self.assertRaises(ValueError):LAB.validate_secondary(value,True,output.replace("owned_bytes=1835008","owned_bytes=1"),False)
        with self.assertRaises(ValueError):LAB.validate_secondary(value,True,output.replace("families=2","families=1"),False)


def adaptive_receipt(direct=False, long=False):
    steps = 127 if long else 7
    return {
        "adaptive_experts": True, "secondary_experts": False,
        "adaptive_policy": "decaying-frequency-recency", "adaptive_capacity": 144,
        "adaptive_frequency_half_life_tokens": 16, "adaptive_score_scale": 256,
        "adaptive_admit_score": 384, "adaptive_budget": 4,
        "adaptive_promotions": 10, "adaptive_boundaries": steps,
        "adaptive_completed_batches": 3, "adaptive_private_peak": 150000,
        "adaptive_source_peak": 16777216, "adaptive_host_bytes": 105000,
        "warm_expert_hits": 2200, "completed_cpu_misses": steps*48*10 - 2200,
        "adaptive_upload_bytes": 1000000, "adaptive_read_ns": 10,
        "adaptive_upload_ns": 20, "adaptive_fence_ns": 30,
        "adaptive_wall_ns": 100, "adaptive_policy_ns": 10,
        "adaptive_fill_ns": 40, "adaptive_publication_ns": 20,
        "adaptive_direct_upload": direct,
        "adaptive_direct_batches": 3 if direct else 0,
        "adaptive_direct_components": 30 if direct else 0,
        "adaptive_direct_bytes": 1000000 if direct else 0,
    }


ADAPTIVE_COMPLETION = "\n".join(["public_adaptive_closed=true host_reservation=196608"]*5)


class AdaptiveControls(unittest.TestCase):
    def test_exact_partition_bounded_promotion_and_all_owner_closes(self):
        value = adaptive_receipt()
        output = ADAPTIVE_COMPLETION
        LAB.validate_adaptive(value,True,output,False)
        for field,bad in [("adaptive_capacity",145),("adaptive_budget",5),
            ("adaptive_promotions",29),("adaptive_completed_batches",2),
            ("adaptive_private_peak",196609),("adaptive_source_peak",41943041),("adaptive_promotions",0),("adaptive_host_bytes",196609),
            ("warm_expert_hits",2201),("adaptive_boundaries",6),("adaptive_fence_ns",0),
            ("secondary_experts",True)]:
            changed=dict(value);changed[field]=bad
            with self.assertRaises(ValueError): LAB.validate_adaptive(changed,True,output,False)
        with self.assertRaises(ValueError): LAB.validate_adaptive(value,True,output.split("\n",1)[1],False)
        with self.assertRaises(ValueError): LAB.validate_adaptive(value,False,output,False)

    def test_decay_metadata_is_required_and_exact_for_all_active_receipts(self):
        for direct in (False, True):
            for long in (False, True):
                value = adaptive_receipt(direct, long)
                LAB.validate_adaptive(value, True, ADAPTIVE_COMPLETION, long,
                                      adaptive_direct_upload=direct)
                for field in ("adaptive_frequency_half_life_tokens", "adaptive_score_scale", "adaptive_admit_score"):
                    for bad in (None, True, str(value[field]), float(value[field]), value[field] + 1):
                        with self.subTest(direct=direct, long=long, field=field, bad=bad):
                            wrong = dict(value, **{field: bad})
                            with self.assertRaises(ValueError):
                                LAB.validate_adaptive(wrong, True, ADAPTIVE_COMPLETION, long,
                                                      adaptive_direct_upload=direct)
                    missing = dict(value)
                    del missing[field]
                    with self.assertRaises(ValueError):
                        LAB.validate_adaptive(missing, True, ADAPTIVE_COMPLETION, long,
                                              adaptive_direct_upload=direct)

    def test_disjoint_phases_are_numeric_and_fully_enclosed_without_tolerance(self):
        value = adaptive_receipt()
        LAB.validate_adaptive(value, True, ADAPTIVE_COMPLETION, False)
        # Equality is accepted; losing even one nanosecond of enclosing time is not.
        with self.assertRaises(ValueError):
            LAB.validate_adaptive(dict(value, adaptive_wall_ns=99), True, ADAPTIVE_COMPLETION, False)
        LAB.validate_adaptive(dict(value, adaptive_wall_ns=101), True, ADAPTIVE_COMPLETION, False)
        for field in ("adaptive_wall_ns", "adaptive_policy_ns", "adaptive_fill_ns", "adaptive_publication_ns"):
            for bad in (None, True, -1, "10", 10.0):
                with self.subTest(field=field, bad=bad):
                    with self.assertRaises(ValueError):
                        LAB.validate_adaptive(dict(value, **{field: bad}), True, ADAPTIVE_COMPLETION, False)
            missing = dict(value)
            del missing[field]
            with self.assertRaises(ValueError):
                LAB.validate_adaptive(missing, True, ADAPTIVE_COMPLETION, False)
        LAB.validate_adaptive(dict(value, adaptive_wall_ns=30, adaptive_policy_ns=0,
                                   adaptive_fill_ns=0, adaptive_publication_ns=0),
                              True, ADAPTIVE_COMPLETION, False)

    def test_direct_proof_conserves_completed_batches_components_and_original_bytes(self):
        fields = ("adaptive_direct_batches", "adaptive_direct_components", "adaptive_direct_bytes")
        for direct in (False, True):
            value = adaptive_receipt(direct)
            LAB.validate_adaptive(value, True, ADAPTIVE_COMPLETION, False,
                                  adaptive_direct_upload=direct)
            for field in fields:
                for bad in (None, True, -1, str(value[field]), float(value[field]), value[field] + 1):
                    with self.subTest(direct=direct, field=field, bad=bad):
                        with self.assertRaises(ValueError):
                            LAB.validate_adaptive(dict(value, **{field: bad}), True, ADAPTIVE_COMPLETION, False,
                                                  adaptive_direct_upload=direct)
                missing = dict(value)
                del missing[field]
                with self.assertRaises(ValueError):
                    LAB.validate_adaptive(missing, True, ADAPTIVE_COMPLETION, False,
                                          adaptive_direct_upload=direct)
            for bad in (None, 1, "true", not direct):
                with self.assertRaises(ValueError):
                    LAB.validate_adaptive(dict(value, adaptive_direct_upload=bad), True,
                                          ADAPTIVE_COMPLETION, False, adaptive_direct_upload=direct)
            missing = dict(value)
            del missing["adaptive_direct_upload"]
            with self.assertRaises(ValueError):
                LAB.validate_adaptive(missing, True, ADAPTIVE_COMPLETION, False,
                                      adaptive_direct_upload=direct)

    def test_inactive_receipts_cannot_report_any_adaptive_proof(self):
        LAB.validate_adaptive({"adaptive_experts": False}, False, "", False)
        LAB.validate_adaptive({}, False, "", False)  # Historical inactive artifact.
        for field, value in adaptive_receipt().items():
            if field.startswith("adaptive_") and field != "adaptive_experts":
                for receipt_value in ({"adaptive_experts": False, field: value}, {field: value}):
                    with self.subTest(field=field, receipt=receipt_value):
                        with self.assertRaises(ValueError):
                            LAB.validate_adaptive(receipt_value, False, "", False)
        with self.assertRaises(ValueError):
            LAB.validate_adaptive({"adaptive_experts": False}, False, "", False,
                                  adaptive_direct_upload=True)
        for bad in (None, 1, "true"):
            with self.assertRaises(ValueError):
                LAB.validate_adaptive(adaptive_receipt(), True, ADAPTIVE_COMPLETION, False,
                                      adaptive_direct_upload=bad)

    def test_environment_does_not_inherit_accidental_opt_in(self):
        with mock.patch.dict(os.environ,{"KUNO_LAB_ADAPTIVE_EXPERTS":"1", "KUNO_LAB_ADAPTIVE_DIRECT_UPLOAD":"1"}):
            env=LAB.worker_environment(pathlib.Path("/original/model"),pathlib.Path("/receipt"))
            self.assertEqual(env["KUNO_LAB_ADAPTIVE_EXPERTS"],"0")
            self.assertEqual(env["KUNO_LAB_ADAPTIVE_DIRECT_UPLOAD"],"0")
            env=LAB.worker_environment(pathlib.Path("/original/model"),pathlib.Path("/receipt"),adaptive_experts=True)
            self.assertEqual(env["KUNO_LAB_ADAPTIVE_EXPERTS"],"1")
            self.assertEqual(env["KUNO_LAB_ADAPTIVE_DIRECT_UPLOAD"],"0")
            env=LAB.worker_environment("/original/model", "/receipt", adaptive_experts=True,
                                       adaptive_direct_upload=True)
            self.assertEqual(env["KUNO_LAB_ADAPTIVE_DIRECT_UPLOAD"],"1")
        with self.assertRaises(ValueError):
            LAB.worker_environment("/original/model", "/receipt", adaptive_direct_upload=True)
        for field in ("adaptive_experts", "adaptive_direct_upload"):
            for bad in (None, 1, "true"):
                with self.assertRaises(ValueError):
                    LAB.worker_environment("/original/model", "/receipt", **{field: bad})

    def test_cli_direct_flag_requires_adaptive_before_loading_artifacts_or_launching(self):
        args = ["lab", "--cargo-artifacts", "/absent/build", "--model", "/original/model",
                "--card", "intel", "--cpu-miss-cap", "10", "--run-id", "test",
                "--receipt", "/absent/receipt", "--adaptive-direct-upload"]
        with mock.patch.object(LAB.sys, "argv", args), \
             mock.patch.object(LAB.BASE, "cargo_binary") as load, \
             mock.patch.object(LAB.subprocess, "Popen") as launch, \
             mock.patch("sys.stderr", new_callable=io.StringIO) as errors:
            with self.assertRaises(SystemExit) as error:
                LAB.main()
        self.assertEqual(error.exception.code, 2)
        self.assertIn("requires --adaptive-experts", errors.getvalue())
        load.assert_not_called()
        launch.assert_not_called()

    def test_validator_forwards_explicit_direct_selection_to_adaptive_gate(self):
        # The existing full-model controls cover the downstream geometry and
        # lifecycle gates; this control isolates selection forwarding to the new gate.
        with mock.patch.object(LAB, "validate_adaptive") as gate:
            LAB.validate_receipt(receipt("intel", 10), "intel", 10, completion(10),
                                 adaptive_experts=True, adaptive_direct_upload=True)
        self.assertTrue(gate.call_args.args[1])
        self.assertTrue(gate.call_args.kwargs["adaptive_direct_upload"])

    def test_runner_forwards_direct_selection_to_environment_validator_and_run_record(self):
        for direct in (False, True):
            with self.subTest(direct=direct), tempfile.TemporaryDirectory() as directory:
                target = pathlib.Path(directory) / "receipt.json"
                build = pathlib.Path(directory) / "build.jsonl"
                build.write_text(json.dumps({"reason": "compiler-artifact",
                    "target": {"name": "kuno-text-lab"}, "executable": "/frozen", "features": []}) + "\n")
                command_seen = []

                def launch(command, **options):
                    command_seen.extend(command)
                    target.write_text(json.dumps(adaptive_receipt(direct)))
                    return SimpleNamespace(stdout=io.StringIO(ADAPTIVE_COMPLETION), wait=lambda: 0)

                def validate(*args, **selection):
                    # Only the adaptive gate is exercised here; independent
                    # full-model/fidelity/lifecycle controls remain below.
                    LAB.validate_adaptive(args[0], selection["adaptive_experts"], args[3], False,
                                          adaptive_direct_upload=selection["adaptive_direct_upload"])

                args = ["lab", "--cargo-artifacts", str(build), "--model", "/original/model",
                        "--card", "intel", "--cpu-miss-cap", "10", "--run-id", "test",
                        "--receipt", str(target), "--adaptive-experts", "--bank-profile", "intel144",
                        "--cpu-workers", "pooled", "--cpu-gpu-overlap", "--mapped-weights",
                        "--next-shared-kernels", "--fast-gpu", "--gpu-router", "--device-dispatch", "inline",
                        "--request-io-reuse", "--defer-router-io", "--compact-info",
                        "--info-batch-upload", "--inline-readback"]
                if direct:
                    args.append("--adaptive-direct-upload")
                with mock.patch.object(LAB.BASE, "cargo_binary", return_value=(pathlib.Path("/frozen"), "binary", "build")), \
                     mock.patch.object(LAB.subprocess, "Popen", side_effect=launch), \
                     mock.patch.object(LAB, "validate_receipt", side_effect=validate) as gate, \
                     mock.patch.object(LAB.sys, "argv", args), mock.patch("sys.stdout", new_callable=io.StringIO):
                    LAB.main()
                self.assertIs(gate.call_args.kwargs["adaptive_direct_upload"], direct)
                self.assertIn("--setenv=KUNO_LAB_ADAPTIVE_DIRECT_UPLOAD=" + ("1" if direct else "0"), command_seen)
                self.assertIn("--setenv=KUNO_LAB_ADAPTIVE_EXPERTS=1", command_seen)
                record = json.loads(target.with_suffix(".json.run.json").read_text())
                self.assertTrue(record["adaptive_experts"])
                self.assertIs(record["adaptive_direct_upload"], direct)


def steady_probe(adaptive=False, direct=False):
    windows = [{"first_step": index * 128, "steps": 128, "elapsed_ns": 10_000_000,
                "gpu_hits": 50000, "cpu_misses": 11440,
                "promotions": 100 if adaptive else 0,
                "upload_bytes": 1000000 if adaptive else 0,
                "promotions_wall_ns": 100000 if adaptive else 0,
                "resources": {key: 0 for key in ("minor_faults", "major_faults", "user_ticks",
                                                "system_ticks", "read_bytes", "write_bytes")}}
               for index in range(8)]
    return {"scope": "fixed-public-repeated-prefix-steady-diagnostic", "teacher_forced": True,
            "prefix_tokens": 128, "prefix_matches": True, "steps": 1024, "window_tokens": 128,
            "context_tokens": 1227, "seed": 424242, "adaptive_experts": adaptive,
            "adaptive_direct_upload": direct, "windows": windows, "greedy_ids": [9419] * 1024,
            "owned_io_cleanup_accepted": True,
            "owned_zero_audit": {"main_buffers": 1378, "main_bytes": 900000000,
                                 "source_buffers": 0, "source_bytes": 0}}


def steady_output(probe, original="original fixture"):
    audit = probe["owned_zero_audit"]
    extra = ["public_resident_owned_zero_audit main_buffers={main_buffers} main_bytes={main_bytes} "
             "source_buffers={source_buffers} source_bytes={source_bytes}".format(**audit)]
    if probe["adaptive_experts"]:
        extra.append(LAB.ADAPTIVE_CLOSED)
    return "\n".join([original, LAB.STEADY_BEGIN, *extra, LAB.STEADY_CLOSED])


STEADY_SELECTION = dict(long=True, banks="intel144", mapped=True, workers="pooled", overlap=True,
                        shared=True, dispatch="inline", request_io_reuse=True, deferred=True,
                        secondary_experts=False)


class SteadyControls(unittest.TestCase):
    def test_complete_fixed_and_adaptive_probes_split_original_only(self):
        for adaptive, direct in ((False, False), (True, False), (True, True)):
            value = steady_probe(adaptive, direct)
            output = steady_output(value)
            self.assertEqual(LAB.validate_steady_probe({"steady_probe": value}, True, output,
                adaptive_experts=adaptive, adaptive_direct_upload=direct), "original fixture")
            # No promotions is a valid settled window, even with adaptive enabled.
            for window in value["windows"]:
                window.update(promotions=0, upload_bytes=0, promotions_wall_ns=0)
            LAB.validate_steady_probe({"steady_probe": value}, True, output,
                adaptive_experts=adaptive, adaptive_direct_upload=direct)

    def test_every_fixed_schema_field_required_and_typed(self):
        original = steady_probe()
        for field in original:
            wrong = copy.deepcopy(original)
            del wrong[field]
            with self.subTest(field=field), self.assertRaises(ValueError):
                LAB.validate_steady_probe({"steady_probe": wrong}, True, steady_output(original))
        for field, bads in (("scope", (None, "private")), ("teacher_forced", (False, 1, "true")),
            ("prefix_matches", (False, 1)), ("prefix_tokens", (True, 127, "128")),
            ("steps", (1023, True)), ("window_tokens", (127, True)), ("context_tokens", (1228, True)),
            ("seed", (True, 0)), ("adaptive_experts", (True, 0)),
            ("adaptive_direct_upload", (True, 0)), ("owned_io_cleanup_accepted", (False, 1)),
            ("windows", (None, [], original["windows"][:-1])),
            ("greedy_ids", (None, [1] * 1023, [1] * 1025, [True] * 1024, [-1] * 1024,
                            [2**32] * 1024))):
            for bad in bads:
                with self.subTest(field=field, bad=type(bad)), self.assertRaises(ValueError):
                    LAB.validate_steady_probe({"steady_probe": dict(original, **{field: bad})},
                                              True, steady_output(original))
        wrong = dict(original, private_content="not allowed")
        with self.assertRaises(ValueError):
            LAB.validate_steady_probe({"steady_probe": wrong}, True, steady_output(original))

    def test_windows_conserve_complete_work_and_promotions(self):
        original = steady_probe(True)
        for field in original["windows"][0]:
            wrong = copy.deepcopy(original)
            del wrong["windows"][0][field]
            with self.subTest(field=field), self.assertRaises(ValueError):
                LAB.validate_steady_probe({"steady_probe": wrong}, True, steady_output(original),
                                          adaptive_experts=True)
        for field in ("first_step", "steps", "elapsed_ns", "gpu_hits", "cpu_misses", "promotions",
                      "upload_bytes", "promotions_wall_ns"):
            for bad in (None, True, "1", -1, 1.0):
                wrong = copy.deepcopy(original)
                wrong["windows"][3][field] = bad
                with self.subTest(field=field, bad=bad), self.assertRaises(ValueError):
                    LAB.validate_steady_probe({"steady_probe": wrong}, True, steady_output(original),
                                              adaptive_experts=True)
        for field, bad in (("first_step", 1), ("steps", 127), ("elapsed_ns", 0), ("gpu_hits", 49999),
            ("cpu_misses", 11441), ("promotions", 513), ("upload_bytes", 0), ("promotions_wall_ns", 0)):
            wrong = copy.deepcopy(original)
            wrong["windows"][0][field] = bad
            with self.subTest(field=field), self.assertRaises(ValueError):
                LAB.validate_steady_probe({"steady_probe": wrong}, True, steady_output(original),
                                          adaptive_experts=True)
        wrong = copy.deepcopy(original)
        wrong["windows"][0]["promotions"] = 0
        with self.assertRaises(ValueError):
            LAB.validate_steady_probe({"steady_probe": wrong}, True, steady_output(original), adaptive_experts=True)
        wrong = steady_probe()
        wrong["windows"][0].update(promotions=1, upload_bytes=1, promotions_wall_ns=1)
        with self.assertRaises(ValueError):
            LAB.validate_steady_probe({"steady_probe": wrong}, True, steady_output(wrong))

    def test_resources_are_exact_numeric_deltas(self):
        original = steady_probe()
        for field in original["windows"][0]["resources"]:
            for bad in (None, True, -1, "0", 0.0):
                wrong = copy.deepcopy(original)
                wrong["windows"][0]["resources"][field] = bad
                with self.subTest(field=field, bad=bad), self.assertRaises(ValueError):
                    LAB.validate_steady_probe({"steady_probe": wrong}, True, steady_output(original))
            wrong = copy.deepcopy(original)
            del wrong["windows"][0]["resources"][field]
            with self.assertRaises(ValueError):
                LAB.validate_steady_probe({"steady_probe": wrong}, True, steady_output(original))
        wrong = copy.deepcopy(original)
        wrong["windows"][0]["resources"]["private_routes"] = 0
        with self.assertRaises(ValueError):
            LAB.validate_steady_probe({"steady_probe": wrong}, True, steady_output(original))

    def test_extended_full_owner_is_separate_bounded_typed_and_matches_stdout(self):
        original = steady_probe()
        for field in original["owned_zero_audit"]:
            wrong = copy.deepcopy(original)
            del wrong["owned_zero_audit"][field]
            with self.assertRaises(ValueError):
                LAB.validate_steady_probe({"steady_probe": wrong}, True, steady_output(original))
            for bad in (None, True, -1, "0", 0.0):
                wrong = copy.deepcopy(original)
                wrong["owned_zero_audit"][field] = bad
                with self.subTest(field=field, bad=bad), self.assertRaises(ValueError):
                    LAB.validate_steady_probe({"steady_probe": wrong}, True, steady_output(original))
        for field, bad in (("main_buffers", 1377), ("main_bytes", 818413567),
                           ("main_bytes", 8 * 1024**3 + 1), ("source_buffers", 1), ("source_bytes", 1)):
            wrong = copy.deepcopy(original)
            wrong["owned_zero_audit"][field] = bad
            with self.assertRaises(ValueError):
                LAB.validate_steady_probe({"steady_probe": wrong}, True, steady_output(wrong))
        for byte_count in (818413568, 8 * 1024**3):
            value = copy.deepcopy(original)
            value["owned_zero_audit"]["main_bytes"] = byte_count
            LAB.validate_steady_probe({"steady_probe": value}, True, steady_output(value))
        with self.assertRaises(ValueError):
            LAB.validate_steady_probe({"steady_probe": original}, True,
                steady_output(original).replace("main_bytes=900000000", "main_bytes=900000001"))

    def test_delimiter_and_close_inventory_cannot_mix_with_original_acceptance(self):
        for adaptive in (False, True):
            value = steady_probe(adaptive)
            output = steady_output(value)
            malformed = [output.replace(LAB.STEADY_BEGIN, ""), output + "\n" + LAB.STEADY_BEGIN,
                output.replace(LAB.STEADY_CLOSED, ""), output + "\n" + LAB.STEADY_CLOSED,
                LAB.STEADY_CLOSED + "\n" + output,
                output.replace("public_resident_owned_zero_audit", "missing_owner"),
                output + "\npublic_resident_owned_zero_audit main_buffers=1378 main_bytes=900000000 source_buffers=0 source_bytes=0"]
            malformed.append(output + "\n" + LAB.ADAPTIVE_CLOSED)
            if adaptive:
                malformed.append(output.replace(LAB.ADAPTIVE_CLOSED, ""))
            for bad in malformed:
                with self.subTest(adaptive=adaptive, bad=bad), self.assertRaises(ValueError):
                    LAB.validate_steady_probe({"steady_probe": value}, True, bad, adaptive_experts=adaptive)
        LAB.validate_steady_probe({}, False, "ordinary output")
        for value, output in (({"steady_probe": steady_probe()}, ""), ({}, LAB.STEADY_BEGIN),
                              ({}, LAB.STEADY_CLOSED)):
            with self.assertRaises(ValueError):
                LAB.validate_steady_probe(value, False, output)

    def test_original_128_ids_and_lifecycle_gates_remain_independent(self):
        value = receipt("intel", 10)
        expected = json.loads((HERE / "receipts/rust-full-resident-rainwater-intel-20261005.json").read_text())["public_generated_ids"]
        value.update(fixture="resident-count1-hybrid-rainwater", prompt_tokens=75,
                     native_public_chunk_partition=[8] * 9 + [3], public_generated_ids=expected,
                     native_decode_steps=127, long_fixture=True, steady_probe=steady_probe())
        output = steady_output(value["steady_probe"], completion(10))
        # Independently tested protected selection is bypassed only to reuse the
        # existing synthetic owner geometry; every original model gate runs.
        with mock.patch.object(LAB, "validate_steady_selection"):
            LAB.validate_receipt(value, "intel", 10, output, long=True, steady_public=True)
            for field, bad in (("public_generated_ids", expected[:-1]), ("public_generated_ids", [9419] * 128),
                ("resident_private_owned_bytes", 900000000), ("owned_zero_audits", [])):
                wrong = copy.deepcopy(value)
                wrong[field] = bad
                with self.assertRaises(ValueError):
                    LAB.validate_receipt(wrong, "intel", 10, output, long=True, steady_public=True)
            marker = LAB.completion(10)[0]
            displaced = output.replace(marker, "") + "\n" + marker
            with self.assertRaises(ValueError):
                LAB.validate_receipt(value, "intel", 10, displaced, long=True, steady_public=True)

    def test_selection_is_explicit_and_requires_the_protected_mode(self):
        LAB.validate_steady_selection(True, **STEADY_SELECTION)
        for field, bad in (("long", False), ("banks", "intel128"), ("mapped", False),
            ("workers", "scoped"), ("overlap", False), ("shared", False), ("dispatch", "threaded"),
            ("request_io_reuse", False), ("deferred", False), ("secondary_experts", True),
            ("gpu_expert_limit", 9), ("card", "nvidia"), ("cap", 9)):
            with self.subTest(field=field), self.assertRaises(ValueError):
                LAB.validate_steady_selection(True, **dict(STEADY_SELECTION, **{field: bad}))
        for bad in (None, 1, "true"):
            with self.assertRaises(ValueError):
                LAB.validate_steady_selection(bad, **STEADY_SELECTION)
        environment = dict(STEADY_SELECTION, fast=True, gpu_router=True, compact_info=True,
                           info_batch_upload=True, inline_readback=True)
        with mock.patch.dict(os.environ, {"KUNO_LAB_STEADY_PUBLIC": "1"}):
            self.assertEqual(LAB.worker_environment("/original", "/receipt")["KUNO_LAB_STEADY_PUBLIC"], "0")
            for adaptive in (False, True):
                env = LAB.worker_environment("/original", "/receipt", steady_public=True,
                    adaptive_experts=adaptive, **environment)
                self.assertEqual(env["KUNO_LAB_STEADY_PUBLIC"], "1")
        with self.assertRaises(ValueError):
            LAB.worker_environment("/original", "/receipt", steady_public=True)

    def test_cli_rejects_wrong_mode_before_artifacts_or_launch(self):
        args = ["lab", "--cargo-artifacts", "/absent/build", "--model", "/original/model",
                "--card", "intel", "--cpu-miss-cap", "10", "--run-id", "test",
                "--receipt", "/absent/receipt", "--steady-public"]
        with mock.patch.object(LAB.sys, "argv", args), \
             mock.patch.object(LAB.BASE, "cargo_binary") as load, \
             mock.patch.object(LAB.subprocess, "Popen") as launch, \
             mock.patch("sys.stderr", new_callable=io.StringIO) as errors:
            with self.assertRaises(SystemExit) as error:
                LAB.main()
        self.assertEqual(error.exception.code, 2)
        self.assertIn("Steady public probe requires", errors.getvalue())
        load.assert_not_called()
        launch.assert_not_called()

    def test_runner_pins_selection_and_validates_before_record(self):
        for selected in (False, True):
            with self.subTest(selected=selected), tempfile.TemporaryDirectory() as directory:
                target = pathlib.Path(directory) / "receipt.json"
                build = pathlib.Path(directory) / "build.jsonl"
                build.write_text(json.dumps({"reason": "compiler-artifact", "target": {"name": "kuno-text-lab"},
                                             "executable": "/frozen", "features": []}) + "\n")
                value = {"steady_probe": steady_probe()} if selected else {}
                output = steady_output(value["steady_probe"]) if selected else "ordinary output"
                command_seen = []

                def launch(command, **options):
                    command_seen.extend(command)
                    target.write_text(json.dumps(value))
                    return SimpleNamespace(stdout=io.StringIO(output), wait=lambda: 0)

                def validate(*args, **selection):
                    LAB.validate_steady_probe(args[0], selection["steady_public"], args[3])

                args = ["lab", "--cargo-artifacts", str(build), "--model", "/original/model", "--card", "intel",
                        "--cpu-miss-cap", "10", "--run-id", "test", "--receipt", str(target)]
                if selected:
                    args.extend(["--steady-public", "--long-fixture", "--bank-profile", "intel144",
                                 "--cpu-workers", "pooled", "--cpu-gpu-overlap", "--mapped-weights",
                                 "--next-shared-kernels", "--fast-gpu", "--gpu-router", "--device-dispatch", "inline",
                                 "--request-io-reuse", "--defer-router-io", "--compact-info", "--info-batch-upload",
                                 "--inline-readback"])
                with mock.patch.object(LAB.BASE, "cargo_binary", return_value=(pathlib.Path("/frozen"), "binary", "build")), \
                     mock.patch.object(LAB.subprocess, "Popen", side_effect=launch), \
                     mock.patch.object(LAB, "validate_receipt", side_effect=validate) as gate, \
                     mock.patch.object(LAB.sys, "argv", args), mock.patch("sys.stdout", new_callable=io.StringIO):
                    LAB.main()
                self.assertIs(gate.call_args.kwargs["steady_public"], selected)
                self.assertIn("--setenv=KUNO_LAB_STEADY_PUBLIC=" + ("1" if selected else "0"), command_seen)
                record = json.loads(target.with_suffix(".json.run.json").read_text())
                self.assertIs(record["steady_public"], selected)


def oracle_probe():
    ids = json.loads((HERE / "receipts/rust-full-resident-rainwater-intel-20261005.json").read_text())["public_generated_ids"][:8]
    audit = {"main_buffers": 1378, "main_bytes": 818413568, "source_buffers": 0, "source_bytes": 0}
    return {"scope": "fixed-public-exact-expert-replay-ceiling", "prompt_tokens": 75, "prefix_tokens": 8,
            "decode_steps": 7, "context_tokens": 203, "seed": 424242, "train_prefix_matches": True,
            "replay_prefix_matches": True, "trace_membership_verified": True, "owned_io_cleanup_accepted": True,
            "expert_union_counts": [40] * 48, "preload_original_bytes": 1000000, "elapsed_ns": 100000000,
            "gpu_hits": 3360, "cpu_misses": 0, "expert_uploads": 0, "public_weight_upload_bytes": 0,
            "adaptive_promotions": 0, "train_greedy_ids": list(ids), "replay_greedy_ids": list(ids),
            "resources": {key: 0 for key in ("minor_faults", "major_faults", "user_ticks", "system_ticks", "read_bytes", "write_bytes")},
            "training_owned_zero_audit": dict(audit), "replay_owned_zero_audit": dict(audit)}


def oracle_receipt():
    probe = oracle_probe()
    return {"public_generated_ids": list(probe["train_greedy_ids"]), "oracle_probe": probe}


def oracle_output(original="original fixture"):
    audit = "public_resident_owned_zero_audit main_buffers=1378 main_bytes=818413568 source_buffers=0 source_bytes=0"
    return "\n".join([original, LAB.ORACLE_BEGIN, audit, LAB.ORACLE_TRACE_CLOSED,
                       audit, LAB.ORACLE_TRACE_CLOSED, LAB.ORACLE_CLOSED])


class OracleControls(unittest.TestCase):
    def test_fixed_ceiling_has_exact_work_and_two_separate_closes(self):
        value = oracle_receipt()
        self.assertEqual(LAB.validate_oracle_probe(value, True, oracle_output()), "original fixture")
        LAB.validate_oracle_probe({}, False, "ordinary output")
        for bad in (None, 1, "true"):
            with self.assertRaises(ValueError):
                LAB.validate_oracle_probe(value, bad, oracle_output())
        for field in value["oracle_probe"]:
            wrong = copy.deepcopy(value)
            del wrong["oracle_probe"][field]
            with self.subTest(field=field), self.assertRaises(ValueError):
                LAB.validate_oracle_probe(wrong, True, oracle_output())
        wrong = copy.deepcopy(value)
        wrong["oracle_probe"]["private_trace"] = []
        with self.assertRaises(ValueError):
            LAB.validate_oracle_probe(wrong, True, oracle_output())

    def test_typed_original_geometry_completed_flags_and_zero_work_counters(self):
        value = oracle_receipt()
        fields = ("prompt_tokens", "prefix_tokens", "decode_steps", "context_tokens", "seed", "gpu_hits",
                  "cpu_misses", "expert_uploads", "public_weight_upload_bytes", "adaptive_promotions")
        for field in fields:
            for bad in (None, True, "0", 0.0, -1, value["oracle_probe"][field] + 1):
                wrong = copy.deepcopy(value)
                wrong["oracle_probe"][field] = bad
                with self.subTest(field=field, bad=bad), self.assertRaises(ValueError):
                    LAB.validate_oracle_probe(wrong, True, oracle_output())
        for field in ("train_prefix_matches", "replay_prefix_matches", "trace_membership_verified", "owned_io_cleanup_accepted"):
            for bad in (False, 1, "true", None):
                wrong = copy.deepcopy(value)
                wrong["oracle_probe"][field] = bad
                with self.subTest(field=field, bad=bad), self.assertRaises(ValueError):
                    LAB.validate_oracle_probe(wrong, True, oracle_output())
        wrong = copy.deepcopy(value)
        wrong["oracle_probe"]["scope"] = "estimated ceiling"
        with self.assertRaises(ValueError):
            LAB.validate_oracle_probe(wrong, True, oracle_output())

    def test_union_covers_every_layer_and_uploads_and_elapsed_have_typed_bounds(self):
        value = oracle_receipt()
        for bad in (None, [], [10] * 47, [10] * 49, [9] * 48, [71] * 48, [True] * 48, ["10"] * 48):
            wrong = copy.deepcopy(value)
            wrong["oracle_probe"]["expert_union_counts"] = bad
            with self.subTest(bad=type(bad)), self.assertRaises(ValueError):
                LAB.validate_oracle_probe(wrong, True, oracle_output())
        for field, bads in (("elapsed_ns", (None, True, -1, 0, 1.0, "1")),
                           ("preload_original_bytes", (None, True, -1, "1", 1.0, 3360 * 6 * 1024**2 + 1))):
            for bad in bads:
                wrong = copy.deepcopy(value)
                wrong["oracle_probe"][field] = bad
                with self.subTest(field=field, bad=bad), self.assertRaises(ValueError):
                    LAB.validate_oracle_probe(wrong, True, oracle_output())
        for count, byte_count in ((10, 0), (70, 3360 * 6 * 1024**2)):
            good = copy.deepcopy(value)
            good["oracle_probe"].update(expert_union_counts=[count] * 48, preload_original_bytes=byte_count)
            LAB.validate_oracle_probe(good, True, oracle_output())

    def test_training_and_replay_match_original_ids_with_no_boolean_substitution(self):
        value = oracle_receipt()
        for field in ("train_greedy_ids", "replay_greedy_ids"):
            for bad in (None, [], value["public_generated_ids"][:-1], [0] * 8, [True] * 8, ["1"] * 8):
                wrong = copy.deepcopy(value)
                wrong["oracle_probe"][field] = bad
                with self.subTest(field=field, bad=bad), self.assertRaises(ValueError):
                    LAB.validate_oracle_probe(wrong, True, oracle_output())
        for bad in (None, [], [True] * 8, [-1] * 8, [2**32] * 8):
            with self.subTest(original=bad), self.assertRaises(ValueError):
                LAB.validate_oracle_probe(dict(value, public_generated_ids=bad), True, oracle_output())

    def test_resources_and_both_owner_audits_require_exact_fields_and_integers(self):
        original = oracle_receipt()
        for group in ("resources", "training_owned_zero_audit", "replay_owned_zero_audit"):
            for field in original["oracle_probe"][group]:
                wrong = copy.deepcopy(original)
                del wrong["oracle_probe"][group][field]
                with self.subTest(group=group, field=field), self.assertRaises(ValueError):
                    LAB.validate_oracle_probe(wrong, True, oracle_output())
                for bad in (None, True, -1, "0", 0.0):
                    wrong = copy.deepcopy(original)
                    wrong["oracle_probe"][group][field] = bad
                    with self.subTest(group=group, field=field, bad=bad), self.assertRaises(ValueError):
                        LAB.validate_oracle_probe(wrong, True, oracle_output())
                if group != "resources":
                    wrong = copy.deepcopy(original)
                    wrong["oracle_probe"][group][field] += 1
                    with self.assertRaises(ValueError):
                        LAB.validate_oracle_probe(wrong, True, oracle_output())
            wrong = copy.deepcopy(original)
            wrong["oracle_probe"][group]["unexpected"] = 0
            with self.assertRaises(ValueError):
                LAB.validate_oracle_probe(wrong, True, oracle_output())

    def test_probe_delimiter_and_inventory_are_exclusive_after_original_gates(self):
        value = oracle_receipt()
        output = oracle_output()
        audit = "public_resident_owned_zero_audit main_buffers=1378 main_bytes=818413568 source_buffers=0 source_bytes=0"
        malformed = [output.replace(LAB.ORACLE_BEGIN, ""), output + "\n" + LAB.ORACLE_BEGIN,
            output.replace(LAB.ORACLE_CLOSED, ""), output + "\n" + LAB.ORACLE_CLOSED,
            LAB.ORACLE_CLOSED + "\n" + output, LAB.ORACLE_TRACE_CLOSED + "\n" + output,
            output.replace(LAB.ORACLE_TRACE_CLOSED, "", 1), output + "\n" + LAB.ORACLE_TRACE_CLOSED,
            output.replace(audit, "", 1), output + "\n" + audit,
            output.replace("main_buffers=1378", "main_buffers=1377", 1),
            output + "\n" + LAB.ADAPTIVE_CLOSED, output + "\npublic_adaptive_closed=false",
            output + "\n" + LAB.STEADY_BEGIN, output + "\n" + LAB.STEADY_CLOSED]
        for bad in malformed:
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                LAB.validate_oracle_probe(value, True, bad)
        with self.assertRaises(ValueError):
            LAB.validate_oracle_probe(dict(value, steady_probe=steady_probe()), True, output)
        for receipt_value, text in ((value, ""), ({}, LAB.ORACLE_BEGIN), ({}, LAB.ORACLE_CLOSED), ({}, LAB.ORACLE_TRACE_CLOSED)):
            with self.assertRaises(ValueError):
                LAB.validate_oracle_probe(receipt_value, False, text)

    def test_original_128_ids_and_lifecycle_cannot_be_satisfied_by_oracle_records(self):
        value = receipt("intel", 10)
        expected = json.loads((HERE / "receipts/rust-full-resident-rainwater-intel-20261005.json").read_text())["public_generated_ids"]
        value.update(fixture="resident-count1-hybrid-rainwater", prompt_tokens=75,
                     native_public_chunk_partition=[8] * 9 + [3], public_generated_ids=expected,
                     native_decode_steps=127, long_fixture=True, oracle_probe=oracle_probe())
        output = oracle_output(completion(10))
        # Selection geometry is covered independently; keep all existing model
        # acceptance gates live with the historical synthetic owner inventory.
        with mock.patch.object(LAB, "validate_oracle_selection"):
            LAB.validate_receipt(value, "intel", 10, output, long=True, oracle_public=True)
            wrong = copy.deepcopy(value)
            wrong["public_generated_ids"][9] += 1
            with self.assertRaises(ValueError):
                LAB.validate_receipt(wrong, "intel", 10, output, long=True, oracle_public=True)
            marker = LAB.completion(10)[0]
            displaced = output.replace(marker, "") + "\n" + marker
            with self.assertRaises(ValueError):
                LAB.validate_receipt(value, "intel", 10, displaced, long=True, oracle_public=True)
            wrong = copy.deepcopy(value)
            wrong["owned_zero_audits"] = []
            with self.assertRaises(ValueError):
                LAB.validate_receipt(wrong, "intel", 10, output, long=True, oracle_public=True)

    def test_selection_excludes_other_probes_and_environment_is_always_explicit(self):
        LAB.validate_oracle_selection(True, **STEADY_SELECTION)
        for field, bad in (("steady_public", True), ("adaptive_experts", True), ("adaptive_direct_upload", True),
            ("long", False), ("banks", "intel128"), ("mapped", False), ("workers", "scoped"),
            ("overlap", False), ("shared", False), ("dispatch", "threaded"), ("request_io_reuse", False),
            ("deferred", False), ("secondary_experts", True), ("gpu_expert_limit", 9), ("card", "nvidia"), ("cap", 9)):
            with self.subTest(field=field), self.assertRaises(ValueError):
                LAB.validate_oracle_selection(True, **dict(STEADY_SELECTION, **{field: bad}))
        for bad in (None, 1, "true"):
            with self.assertRaises(ValueError):
                LAB.validate_oracle_selection(bad, **STEADY_SELECTION)
        settings = dict(STEADY_SELECTION, fast=True, gpu_router=True, compact_info=True,
                        info_batch_upload=True, inline_readback=True)
        with mock.patch.dict(os.environ, {"KUNO_LAB_ORACLE_PUBLIC": "1"}):
            self.assertEqual(LAB.worker_environment("/original", "/receipt")["KUNO_LAB_ORACLE_PUBLIC"], "0")
            env = LAB.worker_environment("/original", "/receipt", oracle_public=True, **settings)
            self.assertEqual(env["KUNO_LAB_ORACLE_PUBLIC"], "1")
            self.assertEqual(env["KUNO_LAB_STEADY_PUBLIC"], "0")
        for extra in ({"steady_public": True}, {"adaptive_experts": True}, {"adaptive_direct_upload": True}):
            with self.assertRaises(ValueError):
                LAB.worker_environment("/original", "/receipt", oracle_public=True, **dict(settings, **extra))

    def test_cli_rejects_wrong_or_mutually_exclusive_mode_before_launch(self):
        base = ["lab", "--cargo-artifacts", "/absent/build", "--model", "/original/model", "--card", "intel",
                "--cpu-miss-cap", "10", "--run-id", "test", "--receipt", "/absent/receipt", "--oracle-public"]
        protected = ["--long-fixture", "--bank-profile", "intel144", "--cpu-workers", "pooled", "--cpu-gpu-overlap",
                     "--mapped-weights", "--next-shared-kernels", "--fast-gpu", "--gpu-router", "--device-dispatch", "inline",
                     "--request-io-reuse", "--defer-router-io", "--compact-info", "--info-batch-upload", "--inline-readback"]
        for additions in ([], protected + ["--steady-public"], protected + ["--adaptive-experts"],
                          protected + ["--adaptive-direct-upload"], protected + ["--secondary-experts"]):
            with mock.patch.object(LAB.sys, "argv", base + additions), \
                 mock.patch.object(LAB.BASE, "cargo_binary") as load, \
                 mock.patch.object(LAB.subprocess, "Popen") as launch, \
                 mock.patch("sys.stderr", new_callable=io.StringIO):
                with self.assertRaises(SystemExit) as error:
                    LAB.main()
            self.assertEqual(error.exception.code, 2)
            load.assert_not_called()
            launch.assert_not_called()

    def test_runner_forwards_selection_to_environment_gate_and_run_record(self):
        for selected in (False, True):
            with self.subTest(selected=selected), tempfile.TemporaryDirectory() as directory:
                target = pathlib.Path(directory) / "receipt.json"
                build = pathlib.Path(directory) / "build.jsonl"
                build.write_text(json.dumps({"reason": "compiler-artifact", "target": {"name": "kuno-text-lab"},
                                             "executable": "/frozen", "features": []}) + "\n")
                value = oracle_receipt() if selected else {}
                output = oracle_output() if selected else "ordinary output"
                command_seen = []

                def launch(command, **options):
                    command_seen.extend(command)
                    target.write_text(json.dumps(value))
                    return SimpleNamespace(stdout=io.StringIO(output), wait=lambda: 0)

                def validate(*args, **selection):
                    LAB.validate_oracle_probe(args[0], selection["oracle_public"], args[3])

                args = ["lab", "--cargo-artifacts", str(build), "--model", "/original/model", "--card", "intel",
                        "--cpu-miss-cap", "10", "--run-id", "test", "--receipt", str(target)]
                if selected:
                    args.extend(["--oracle-public", "--long-fixture", "--bank-profile", "intel144",
                        "--cpu-workers", "pooled", "--cpu-gpu-overlap", "--mapped-weights", "--next-shared-kernels",
                        "--fast-gpu", "--gpu-router", "--device-dispatch", "inline", "--request-io-reuse",
                        "--defer-router-io", "--compact-info", "--info-batch-upload", "--inline-readback"])
                with mock.patch.object(LAB.BASE, "cargo_binary", return_value=(pathlib.Path("/frozen"), "binary", "build")), \
                     mock.patch.object(LAB.subprocess, "Popen", side_effect=launch), \
                     mock.patch.object(LAB, "validate_receipt", side_effect=validate) as gate, \
                     mock.patch.object(LAB.sys, "argv", args), mock.patch("sys.stdout", new_callable=io.StringIO):
                    LAB.main()
                self.assertIs(gate.call_args.kwargs["oracle_public"], selected)
                self.assertIn("--setenv=KUNO_LAB_ORACLE_PUBLIC=" + ("1" if selected else "0"), command_seen)
                record = json.loads(target.with_suffix(".json.run.json").read_text())
                self.assertIs(record["oracle_public"], selected)


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
                 count1_hybrid_cpu_gpu_overlap=False, observed_count1_hybrid_cpu_gpu_overlap=False, count1_hybrid_cpu_workers="scoped", observed_count1_hybrid_cpu_workers="scoped")
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
    def test_binding_capacity_is_bounded_and_explicit(self):
        for capacity in (0, True, 1024, 512):
            with self.assertRaises(ValueError):
                LAB.worker_environment("/original", "/receipt", binding_cache_capacity=capacity)
        selection=dict(mapped=True,deferred=True,compact_info=True,info_batch_upload=True,inline_readback=True,dispatch="inline",request_io_reuse=True,bind_group_reuse=True)
        env=LAB.worker_environment("/original", "/receipt", binding_cache_capacity=512, **selection)
        self.assertEqual(env["KUNO_LAB_BIND_GROUP_CAPACITY"], "512")
        self.assertEqual(LAB.worker_environment("/original", "/receipt")["KUNO_LAB_BIND_GROUP_CAPACITY"], "256")

    def test_integer_experts_require_explicit_fused_original_gpu_policy(self):
        for settings in ({}, {"fused_experts":True, "gpu_expert_limit":6, "mapped":True,"workers":"pooled","overlap":True}):
            with self.assertRaises(ValueError):
                LAB.worker_environment("/original", "/receipt", integer_experts=True, shared=True, fast=True, gpu_router=True, **settings)
        env=LAB.worker_environment("/original", "/receipt", integer_experts=True, fused_experts=True, shared=True, fast=True, gpu_router=True)
        self.assertEqual(env["KUNO_LAB_INTEGER_EXPERTS"], "1")
        self.assertEqual(LAB.worker_environment("/original", "/receipt")["KUNO_LAB_INTEGER_EXPERTS"], "0")

    def test_prefetch_counts_include_resident_cpu_work_without_changing_misses(self):
        value = receipt("intel",10)
        value.update(cpu_mapped_prefetch=True, observed_cpu_mapped_prefetch=True,
            warm_cpu_prefetch_calls=60, mapped_weights=True, observed_mapped_weights=True,
            count1_hybrid_cpu_workers="pooled", observed_count1_hybrid_cpu_workers="pooled",
            count1_hybrid_cpu_gpu_overlap=True, observed_count1_hybrid_cpu_gpu_overlap=True,
            gpu_expert_limit=6, observed_cpu_resident_expert_calls=3, observed_shared_overlap_calls=336)
        LAB.validate_receipt(value, "intel", 10, completion(10), workers="pooled", overlap=True,
            mapped=True, cpu_mapped_prefetch=True, gpu_expert_limit=6)
        for field, bad in (("warm_cpu_prefetch_calls", 51), ("observed_cpu_resident_expert_calls", 0),
            ("observed_cpu_resident_expert_calls", True), ("gpu_expert_limit", 10)):
            corrupted = copy.deepcopy(value); corrupted[field] = bad
            with self.assertRaises(ValueError):
                LAB.validate_receipt(corrupted, "intel", 10, completion(10), workers="pooled", overlap=True,
                    mapped=True, cpu_mapped_prefetch=True, gpu_expert_limit=6)

    def test_resident_balancing_requires_mapped_pooled_overlap(self):
        for limit in (-1, 11, True):
            with self.assertRaises(ValueError):
                LAB.worker_environment("/original", "/receipt", gpu_expert_limit=limit)
        with self.assertRaises(ValueError):
            LAB.worker_environment("/original", "/receipt", gpu_expert_limit=6)
        env = LAB.worker_environment("/original", "/receipt", gpu_expert_limit=6, mapped=True, workers="pooled", overlap=True)
        self.assertEqual(env["KUNO_LAB_GPU_EXPERT_LIMIT"], "6")
        self.assertEqual(LAB.worker_environment("/original", "/receipt")["KUNO_LAB_GPU_EXPERT_LIMIT"], "10")

    def test_fused_experts_requires_shared_and_keeps_control_explicit(self):
        with self.assertRaises(ValueError):
            LAB.worker_environment("/original", "/receipt", fused_experts=True)
        env = LAB.worker_environment("/original", "/receipt", fused_experts=True, shared=True, fast=True, gpu_router=True)
        self.assertEqual(env["KUNO_LAB_FUSED_EXPERTS"], "1")
        self.assertEqual(LAB.worker_environment("/original", "/receipt")["KUNO_LAB_FUSED_EXPERTS"], "0")

    def test_request_io_reuse_requires_actual_close_and_bounded_checkpoints(self):
        marker="public_request_io_reuse closed=true full_audit=true"
        value=dict(request_io_reuse=True, request_io_closed_and_audited=True,
                   observed_request_io_checkpoints=381, observed_request_io_reuses=1234, request_io_close_ms=42,
                   warm_info_cache_stats=[1000, 100, 20])
        LAB.validate_request_io_reuse(value, True, marker)
        for field, bads in [("request_io_reuse",[False,1,None]),
                            ("request_io_closed_and_audited",[False,1,None]),
                            ("observed_request_io_checkpoints",[0,True,None,-1,1000001]),
                            ("request_io_close_ms",[-1,True,None,900001]),
                            ("observed_request_io_reuses",[0,True,None,-1,10000001]),
                            ("warm_info_cache_stats",[None,[],[1,2],[0,1,2],[1,0,2],[1,2,0],[True,1,2],[1,-1,2],[1,2,10000001]])]:
            for bad in bads:
                wrong=dict(value);wrong[field]=bad
                with self.assertRaises(ValueError): LAB.validate_request_io_reuse(wrong,True,marker)
        with self.assertRaises(ValueError): LAB.validate_request_io_reuse(value,True,"")
        with self.assertRaises(ValueError): LAB.validate_request_io_reuse(value,False,marker)
        LAB.validate_request_io_reuse(dict(request_io_reuse=False,request_io_closed_and_audited=False,observed_request_io_checkpoints=0,observed_request_io_reuses=0,request_io_close_ms=0),False,"")

    def test_upload_transfer_receipts_conserve_original_bytes(self):
        marker = "public_request_io_reuse closed=true full_audit=true"
        value = dict(request_io_reuse=True, request_io_closed_and_audited=True,
                     observed_request_io_checkpoints=381, observed_request_io_reuses=1234,
                     request_io_close_ms=42, warm_info_cache_stats=[1000, 100, 20])
        for stats in ([32768, 0, 32768], [32768, 32768, 0], [32768, 10240, 22528]):
            LAB.validate_request_io_reuse(dict(value, warm_upload_transfer_bytes=stats), True, marker)
        for stats in (None, [], [1, 1], [1, 1, 1], [0, 0, 0], [32768, True, 32767],
                      [32768, -1, 32769], [2**64, 0, 2**64]):
            with self.assertRaises(ValueError):
                LAB.validate_request_io_reuse(dict(value, warm_upload_transfer_bytes=stats), True, marker)

    def test_binding_cache_selection_requires_request_and_rejects_inherited_enable(self):
        with mock.patch.dict(os.environ,{"KUNO_LAB_BIND_GROUP_REUSE":"1"}):
            self.assertEqual(LAB.worker_environment("/original","/receipt")["KUNO_LAB_BIND_GROUP_REUSE"],"0")
        with self.assertRaises(ValueError): LAB.worker_environment("/original","/receipt",bind_group_reuse=True)
        selection=dict(mapped=True,deferred=True,compact_info=True,info_batch_upload=True,inline_readback=True,dispatch="inline",request_io_reuse=True,bind_group_reuse=True)
        self.assertEqual(LAB.worker_environment("/original","/receipt",**selection)["KUNO_LAB_BIND_GROUP_REUSE"],"1")

    def test_private_upload_phases_are_paired_and_conserve_bytes(self):
        LAB.validate_private_upload_phases({})  # Historical frozen artifacts.
        fields = ("model_bind_private_upload_bytes", "prefill_private_upload_bytes")
        good = dict(zip(fields, ([32768, 0, 32768], [65536, 1024, 64512])))
        LAB.validate_private_upload_phases(good)
        LAB.validate_private_upload_phases(dict.fromkeys(fields, [0, 0, 0]))
        for key in fields:
            missing = dict(good)
            del missing[key]
            with self.assertRaises(ValueError):
                LAB.validate_private_upload_phases(missing)
            for bad in (None, [], [1, 1], [1, 1, 1], [True, 1, 0],
                        [32768, -1, 32769], [2**64, 0, 2**64]):
                wrong = dict(good, **{key: bad})
                with self.assertRaises(ValueError):
                    LAB.validate_private_upload_phases(wrong)

    def test_request_io_reuse_is_explicit_and_has_no_unsafe_mode(self):
        selection=dict(mapped=True,deferred=True,compact_info=True,info_batch_upload=True,
                       inline_readback=True,dispatch="inline",request_io_reuse=True)
        self.assertEqual(LAB.worker_environment("/original","/receipt",**selection)["KUNO_LAB_REQUEST_IO_REUSE"],"1")
        for field,value in [("mapped",False),("deferred",False),("info_batch_upload",False),
                            ("inline_readback",False),("gpu_upload_clear",True),("profile",True)]:
            invalid=dict(selection);invalid[field]=value
            with self.assertRaises(ValueError): LAB.worker_environment("/original","/receipt",**invalid)
        with mock.patch.dict(os.environ,{"KUNO_LAB_REQUEST_IO_REUSE":"1"}):
            self.assertEqual(LAB.worker_environment("/original","/receipt")["KUNO_LAB_REQUEST_IO_REUSE"],"0")

    def test_cpu_work_stealing_requires_pooled_policy_and_compiled_marker(self):
        value=receipt("intel",10)
        value.update(cpu_work_stealing=True,count1_hybrid_cpu_workers="pooled",observed_count1_hybrid_cpu_workers="pooled")
        output=completion(10)+"\ncpu_row_dispatch mode=work-stealing joined=true original_arithmetic=true"
        LAB.validate_receipt(value,"intel",10,output,workers="pooled",cpu_work_stealing=True)
        for bad in [False,1,None,"true"]:
            wrong=copy.deepcopy(value);wrong["cpu_work_stealing"]=bad
            with self.assertRaises(ValueError):LAB.validate_receipt(wrong,"intel",10,output,workers="pooled",cpu_work_stealing=True)
        with self.assertRaises(ValueError):LAB.validate_receipt(value,"intel",10,completion(10),workers="pooled",cpu_work_stealing=True)
        with self.assertRaises(ValueError):LAB.validate_receipt(value,"intel",10,output,cpu_work_stealing=True)
        with self.assertRaises(ValueError):LAB.worker_environment("/original","/receipt",cpu_work_stealing=True)
        self.assertEqual(LAB.worker_environment("/original","/receipt",workers="pooled",cpu_work_stealing=True)["KUNO_LAB_CPU_WORK_STEALING"],"1")
        with mock.patch.dict(os.environ,{"KUNO_LAB_CPU_WORK_STEALING":"1"}):
            self.assertEqual(LAB.worker_environment("/original","/receipt")["KUNO_LAB_CPU_WORK_STEALING"],"0")

    def test_grouped_status_requires_bounded_calls_active_sticky_probe_and_explicit_environment(self):
        value=receipt("intel",10);value.update(grouped_status=True,observed_grouped_status_calls=12000)
        output=completion(10)+"\npublic_grouped_status_fold groups=3,5,8 counts=1,3,8 active_and_sticky=true inactive_ignored=true"
        LAB.validate_receipt(value,"intel",10,output,grouped_status=True)
        for field,bads in [("grouped_status",[False,1,None]),("observed_grouped_status_calls",[0,True,None,-1,1000001])]:
            for bad in bads:
                wrong=copy.deepcopy(value);wrong[field]=bad
                with self.assertRaises(ValueError):LAB.validate_receipt(wrong,"intel",10,output,grouped_status=True)
        with self.assertRaises(ValueError):LAB.validate_receipt(value,"intel",10,completion(10),grouped_status=True)
        with self.assertRaises(ValueError):LAB.worker_environment("/original","/receipt",grouped_status=True)
        selection=dict(metadata=True,shared=True,fast=True,gpu_router=True,compact_info=True,batch_tasks=True)
        self.assertEqual(LAB.worker_environment("/original","/receipt",**selection,grouped_status=True)["KUNO_LAB_GROUPED_STATUS"],"1")
        with mock.patch.dict(os.environ,{"KUNO_LAB_GROUPED_STATUS":"1"}):
            self.assertEqual(LAB.worker_environment("/original","/receipt")["KUNO_LAB_GROUPED_STATUS"],"0")

    def test_inline_readback_requires_bounded_evidence_and_actual_policy(self):
        value=receipt("intel",10);value.update(inline_readback=True,observed_inline_readback_batches=12000)
        output=completion(10)+"\nprivate_inline_readback enabled=true checked_callbacks=true held_leases=true"
        LAB.validate_receipt(value,"intel",10,output,inline_readback=True)
        for field,bads in [("inline_readback",[False,1,None]),("observed_inline_readback_batches",[0,True,None,-1,1000001])]:
            for bad in bads:
                wrong=dict(value);wrong[field]=bad
                with self.assertRaises(ValueError):LAB.validate_receipt(wrong,"intel",10,output,inline_readback=True)
        with self.assertRaises(ValueError):LAB.validate_receipt(value,"intel",10,completion(10),inline_readback=True)
        for dispatch in ["threaded","inline"]:
            with self.assertRaises(ValueError):LAB.worker_environment("/original","/receipt",dispatch=dispatch,inline_readback=True)
        self.assertEqual(LAB.worker_environment("/original","/receipt",compact_info=True,info_batch_upload=True,dispatch="inline",inline_readback=True)["KUNO_LAB_INLINE_READBACK"],"1")

    def test_gpu_upload_clear_requires_bounded_evidence_and_actual_policy(self):
        value=receipt("intel",10);value.update(gpu_upload_clear=True,observed_gpu_upload_clears=12000)
        output=completion(10)+"\nprivate_primary_upload_clear enabled=true two_gpu_passes=true direct_full_cpu_inspection=true"
        LAB.validate_receipt(value,"intel",10,output,gpu_upload_clear=True)
        for field,bads in [("gpu_upload_clear",[False,1,None]),("observed_gpu_upload_clears",[0,True,None,-1,1000001])]:
            for bad in bads:
                wrong=dict(value);wrong[field]=bad
                with self.assertRaises(ValueError):LAB.validate_receipt(wrong,"intel",10,output,gpu_upload_clear=True)
        with self.assertRaises(ValueError):LAB.validate_receipt(value,"intel",10,completion(10),gpu_upload_clear=True)
        with self.assertRaises(ValueError):LAB.worker_environment("/original","/receipt",gpu_upload_clear=True)
        self.assertEqual(LAB.worker_environment("/original","/receipt",compact_info=True,info_batch_upload=True,gpu_upload_clear=True)["KUNO_LAB_GPU_UPLOAD_CLEAR"],"1")

    def test_readback_reuse_requires_bounded_evidence_and_actual_policy(self):
        value=receipt("intel",10);value.update(readback_reuse=True,observed_clean_readback_reuses=12000)
        output=completion(10)+"\nprivate_clean_readback_reuse enabled=true max_buffers=64 max_bytes=8388608"
        LAB.validate_receipt(value,"intel",10,output,readback_reuse=True)
        for field,bads in [("readback_reuse",[False,1,None]),("observed_clean_readback_reuses",[0,True,None,-1,1000001])]:
            for bad in bads:
                wrong=dict(value);wrong[field]=bad
                with self.assertRaises(ValueError):LAB.validate_receipt(wrong,"intel",10,output,readback_reuse=True)
        with self.assertRaises(ValueError):LAB.validate_receipt(value,"intel",10,completion(10),readback_reuse=True)
        with self.assertRaises(ValueError):LAB.worker_environment("/original","/receipt",readback_reuse=True)
        self.assertEqual(LAB.worker_environment("/original","/receipt",compact_info=True,info_batch_upload=True,readback_reuse=True)["KUNO_LAB_READBACK_REUSE"],"1")

    def test_compute_scratch_clear_requires_bounded_calls_full_probe_and_explicit_environment(self):
        value=receipt("intel",10);value.update(compute_scratch_clear=True,observed_compute_scratch_clear_calls=12000)
        output=completion(10)+"\npublic_compute_scratch_clear_full_owners=2,3,4 unselected_preserved=true padding=true"
        LAB.validate_receipt(value,"intel",10,output,compute_scratch_clear=True)
        for field,bads in [("compute_scratch_clear",[False,1,None]),("observed_compute_scratch_clear_calls",[0,True,None,-1,1000001])]:
            for bad in bads:
                wrong=copy.deepcopy(value);wrong[field]=bad
                with self.assertRaises(ValueError):LAB.validate_receipt(wrong,"intel",10,output,compute_scratch_clear=True)
        with self.assertRaises(ValueError):LAB.validate_receipt(value,"intel",10,completion(10),compute_scratch_clear=True)
        with self.assertRaises(ValueError):LAB.worker_environment("/original","/receipt",compute_scratch_clear=True)
        selection=dict(metadata=True,shared=True,fast=True,gpu_router=True,compact_info=True,batch_tasks=True)
        self.assertEqual(LAB.worker_environment("/original","/receipt",**selection,compute_scratch_clear=True)["KUNO_LAB_COMPUTE_SCRATCH_CLEAR"],"1")
        with self.assertRaises(ValueError):LAB.worker_environment("/original","/receipt",**selection,compute_scratch_clear=True,native_scratch_clear=True)
        with mock.patch.dict(os.environ,{"KUNO_LAB_COMPUTE_SCRATCH_CLEAR":"1"}):
            self.assertEqual(LAB.worker_environment("/original","/receipt")["KUNO_LAB_COMPUTE_SCRATCH_CLEAR"],"0")

    def test_cpu_threads_and_compiled_two_pass_require_actual_typed_operator_geometry(self):
        for threads in (4,8,16):
            value=receipt("intel",10);value.update(cpu_threads=threads,observed_cpu_threads=threads,cpu_q8_two_pass=True,observed_cpu_q8_residual_passes=2,cpu_q8_residual=True,observed_cpu_q8_residual=True,cpu_q8_activations=True,observed_cpu_q8_activations=True,count1_hybrid_cpu_workers="pooled",observed_count1_hybrid_cpu_workers="pooled")
            LAB.validate_receipt(value,"intel",10,completion(10),workers="pooled",cpu_q8=True,cpu_q8_residual=True,cpu_q8_two_pass=True,cpu_threads=threads)
            for field,bads in [("cpu_threads",[True,None,3]),("observed_cpu_threads",[True,None,3]),("cpu_q8_two_pass",[False,1,None]),("observed_cpu_q8_residual_passes",[True,3,None])]:
                for bad in bads:
                    wrong=copy.deepcopy(value);wrong[field]=bad
                    with self.assertRaises(ValueError):LAB.validate_receipt(wrong,"intel",10,completion(10),workers="pooled",cpu_q8=True,cpu_q8_residual=True,cpu_q8_two_pass=True,cpu_threads=threads)
        for threads in (True,0,3,17,None):
            with self.assertRaises(ValueError):LAB.worker_environment("/original","/receipt",cpu_threads=threads)
        with self.assertRaises(ValueError):LAB.worker_environment("/original","/receipt",cpu_q8_two_pass=True)
        with mock.patch.dict(os.environ,{"KUNO_LAB_CPU_THREADS":"4","KUNO_LAB_CPU_Q8_TWO_PASS":"1"}):
            env=LAB.worker_environment("/original","/receipt");self.assertEqual(env["KUNO_LAB_CPU_THREADS"],"16");self.assertEqual(env["KUNO_LAB_CPU_Q8_TWO_PASS"],"0")

    def test_gpu_source_pageout_requires_explicit_reclaim_and_typed_selection(self):
        value=receipt("intel",10);value.update(pageout_gpu_sources=True,reclaim_gpu_sources=True,gpu_source_mapping_advice="discard-clean-file-ptes",gpu_source_advised_bytes=19*1024**3,gpu_source_advice_ms=200,mapped_weights=True,observed_mapped_weights=True)
        LAB.validate_receipt(value,"intel",10,completion(10),mapped=True,reclaim_gpu_sources=True,pageout_gpu_sources=True)
        for bad in (False,1,None):
            wrong=copy.deepcopy(value);wrong["pageout_gpu_sources"]=bad
            with self.assertRaises(ValueError):LAB.validate_receipt(wrong,"intel",10,completion(10),mapped=True,reclaim_gpu_sources=True,pageout_gpu_sources=True)
        for mode in (None, "cold-pageout", "descriptor-only", True):
            wrong=copy.deepcopy(value);wrong["gpu_source_mapping_advice"]=mode
            with self.assertRaises(ValueError):LAB.validate_receipt(wrong,"intel",10,completion(10),mapped=True,reclaim_gpu_sources=True,pageout_gpu_sources=True)
        with self.assertRaises(ValueError):LAB.worker_environment("/original","/receipt",pageout_gpu_sources=True)
        with mock.patch.dict(os.environ,{"KUNO_LAB_PAGEOUT_GPU_SOURCES":"1"}):
            self.assertEqual(LAB.worker_environment("/original","/receipt")["KUNO_LAB_PAGEOUT_GPU_SOURCES"],"0")
        self.assertEqual(LAB.worker_environment("/original","/receipt",mapped=True,reclaim_gpu_sources=True,pageout_gpu_sources=True)["KUNO_LAB_PAGEOUT_GPU_SOURCES"],"1")

    def test_cpu_successful_host_phases_are_numeric_bounded_and_nested(self):
        value=receipt("intel",10);value["warm_cpu_operator_host_phase_ns"]=[20,30,40,100]
        LAB.validate_receipt(value,"intel",10,completion(10))
        for bad in (None, [20,30,40], [20,30,40,89], [True,0,0,1], [0,0,0,900_000_000_001], [0,-1,0,0]):
            wrong=copy.deepcopy(value);wrong["warm_cpu_operator_host_phase_ns"]=bad
            with self.assertRaises(ValueError):LAB.validate_receipt(wrong,"intel",10,completion(10))

    def test_gpu_profiling_is_bounded_to_one_explicit_native_decode(self):
        value=receipt("intel",10);value.update(kernel_profile=True,kernel_profile_scope="first-following-decode",profiled_native_decode_steps=1)
        LAB.validate_receipt(value,"intel",10,completion(10),profile=True)
        for field,bads in [("kernel_profile_scope",["prefill","full",None]),("profiled_native_decode_steps",[0,2,True,None])]:
            for bad in bads:
                wrong=copy.deepcopy(value);wrong[field]=bad
                with self.assertRaises(ValueError):LAB.validate_receipt(wrong,"intel",10,completion(10),profile=True)

    def test_buffered_cpu_requires_actual_typed_original_reads_without_prefetch(self):
        value=receipt("intel",10);value.update(cpu_buffered_weights=True,observed_cpu_buffered_weights=True,warm_cpu_buffered_reads=51,mapped_weights=True,observed_mapped_weights=True,count1_hybrid_cpu_workers="pooled",observed_count1_hybrid_cpu_workers="pooled")
        LAB.validate_receipt(value,"intel",10,completion(10),workers="pooled",mapped=True,cpu_buffered_weights=True)
        for field,bads in [("cpu_buffered_weights",[False,1,None]),("observed_cpu_buffered_weights",[False,1,None]),("warm_cpu_buffered_reads",[50,True,None,-1])]:
            for bad in bads:
                wrong=copy.deepcopy(value);wrong[field]=bad
                with self.assertRaises(ValueError):LAB.validate_receipt(wrong,"intel",10,completion(10),workers="pooled",mapped=True,cpu_buffered_weights=True)
        for kwargs in ({"workers":"scoped","mapped":True},{"workers":"pooled","mapped":False},{"workers":"pooled","mapped":True,"cpu_mapped_prefetch":True}):
            with self.assertRaises(ValueError):LAB.validate_receipt(value,"intel",10,completion(10),cpu_buffered_weights=True,**kwargs)

    def test_cpu_prefetch_requires_mapped_actual_typed_selected_source_counts(self):
        value=receipt("intel",10);value.update(cpu_mapped_prefetch=True,observed_cpu_mapped_prefetch=True,warm_cpu_prefetch_calls=51,mapped_weights=True,observed_mapped_weights=True,count1_hybrid_cpu_workers="pooled",observed_count1_hybrid_cpu_workers="pooled")
        LAB.validate_receipt(value,"intel",10,completion(10),workers="pooled",mapped=True,cpu_mapped_prefetch=True)
        for field,bads in [("cpu_mapped_prefetch",[False,1,None]),("observed_cpu_mapped_prefetch",[False,1,None]),("warm_cpu_prefetch_calls",[50,True,None,-1])]:
            for bad in bads:
                wrong=copy.deepcopy(value);wrong[field]=bad
                with self.assertRaises(ValueError):LAB.validate_receipt(wrong,"intel",10,completion(10),workers="pooled",mapped=True,cpu_mapped_prefetch=True)
        for kw in ({},{"mapped":True},{"workers":"pooled"}):
            with self.assertRaises(ValueError):LAB.worker_environment("/original","/receipt",cpu_mapped_prefetch=True,**kw)
        with mock.patch.dict(os.environ,{"KUNO_LAB_CPU_MAPPED_PREFETCH":"1"}):
            self.assertEqual(LAB.worker_environment("/original","/receipt")["KUNO_LAB_CPU_MAPPED_PREFETCH"],"0")
        self.assertEqual(LAB.worker_environment("/original","/receipt",workers="pooled",mapped=True,cpu_mapped_prefetch=True)["KUNO_LAB_CPU_MAPPED_PREFETCH"],"1")

    def test_residual_q8_requires_explicit_selected_and_observed_arithmetic(self):
        value=receipt("intel",10);value.update(cpu_q8_activations=True,observed_cpu_q8_activations=True,cpu_q8_residual=True,observed_cpu_q8_residual=True,count1_hybrid_cpu_workers="pooled",observed_count1_hybrid_cpu_workers="pooled")
        LAB.validate_receipt(value,"intel",10,completion(10),workers="pooled",cpu_q8=True,cpu_q8_residual=True)
        for field in ("cpu_q8_residual","observed_cpu_q8_residual"):
            for bad in (False,1,None):
                wrong=copy.deepcopy(value);wrong[field]=bad
                with self.assertRaises(ValueError):LAB.validate_receipt(wrong,"intel",10,completion(10),workers="pooled",cpu_q8=True,cpu_q8_residual=True)
        with self.assertRaises(ValueError):LAB.worker_environment("/original","/receipt",workers="pooled",cpu_q8_residual=True)
        with mock.patch.dict(os.environ,{"KUNO_LAB_CPU_Q8_RESIDUAL":"1"}):
            self.assertEqual(LAB.worker_environment("/original","/receipt")["KUNO_LAB_CPU_Q8_RESIDUAL"],"0")
        self.assertEqual(LAB.worker_environment("/original","/receipt",workers="pooled",cpu_q8=True,cpu_q8_residual=True)["KUNO_LAB_CPU_Q8_RESIDUAL"],"1")

    def test_native_scratch_clear_requires_completed_typed_evidence_and_explicit_environment(self):
        value=receipt("intel",10);value.update(native_scratch_clear=True,observed_native_scratch_clear_calls=18000)
        LAB.validate_receipt(value,"intel",10,completion(10),native_scratch_clear=True)
        for field,bads in [("native_scratch_clear",[False,1,None]),("observed_native_scratch_clear_calls",[0,True,None,-1,1000001])]:
            for bad in bads:
                wrong=copy.deepcopy(value);wrong[field]=bad
                with self.assertRaises(ValueError):LAB.validate_receipt(wrong,"intel",10,completion(10),native_scratch_clear=True)
        with mock.patch.dict(os.environ,{"KUNO_LAB_NATIVE_SCRATCH_CLEAR":"1"}):
            self.assertEqual(LAB.worker_environment("/original","/receipt")["KUNO_LAB_NATIVE_SCRATCH_CLEAR"],"0")
        self.assertEqual(LAB.worker_environment("/original","/receipt",native_scratch_clear=True)["KUNO_LAB_NATIVE_SCRATCH_CLEAR"],"1")

    def test_source_advice_requires_actual_bounded_evidence_and_static_bank_mapping(self):
        value=receipt("intel",10)
        value.update(reclaim_gpu_sources=True,gpu_source_advised_bytes=19*1024**3,gpu_source_advice_ms=150,mapped_weights=True,observed_mapped_weights=True)
        LAB.validate_receipt(value,"intel",10,completion(10),mapped=True,reclaim_gpu_sources=True)
        for field,bads in [("reclaim_gpu_sources",[False,1,"true",None]),("gpu_source_advised_bytes",[0,True,33*1024**3,None]),("gpu_source_advice_ms",[-1,True,None])]:
            for bad in bads:
                wrong=copy.deepcopy(value);wrong[field]=bad
                with self.assertRaises(ValueError):LAB.validate_receipt(wrong,"intel",10,completion(10),mapped=True,reclaim_gpu_sources=True)
        with self.assertRaises(ValueError):LAB.validate_receipt(value,"intel",5,completion(5),mapped=True,reclaim_gpu_sources=True)
        with self.assertRaises(ValueError):LAB.worker_environment("/original","/receipt",reclaim_gpu_sources=True)
        env=LAB.worker_environment("/original","/receipt",mapped=True,reclaim_gpu_sources=True)
        self.assertEqual(env["KUNO_LAB_RECLAIM_GPU_SOURCES"],"1")
        with mock.patch.dict(os.environ,{"KUNO_LAB_RECLAIM_GPU_SOURCES":"1"}):
            self.assertEqual(LAB.worker_environment("/original","/receipt")["KUNO_LAB_RECLAIM_GPU_SOURCES"],"0")
    def test_cpu_fma_arithmetic_is_explicit_typed_and_separate_from_q8(self):
        value=receipt("intel",10)
        value.update(cpu_fma_arithmetic=True,count1_hybrid_cpu_workers="pooled",observed_count1_hybrid_cpu_workers="pooled")
        LAB.validate_receipt(value,"intel",10,completion(10),workers="pooled",cpu_fma=True)
        for bad in (None,False,1,"true"):
            wrong=copy.deepcopy(value);wrong["cpu_fma_arithmetic"]=bad
            with self.assertRaises(ValueError):LAB.validate_receipt(wrong,"intel",10,completion(10),workers="pooled",cpu_fma=True)
        with self.assertRaises(ValueError):LAB.validate_receipt(value,"intel",10,completion(10),workers="pooled")
        with self.assertRaises(ValueError):LAB.validate_receipt(value,"intel",10,completion(10),workers="pooled",cpu_fma=True,cpu_q8=True)
        with self.assertRaises(ValueError):LAB.validate_receipt(value,"intel",10,completion(10),cpu_fma=True)
    def test_inline_dispatch_requires_compiled_runtime_evidence(self):
        value=receipt("intel",10)
        value.update(device_dispatch="inline",observed_inline_device_dispatch=True)
        output=completion(10)+"\ngpu_device_dispatch mode=inline blocking=true"
        LAB.validate_receipt(value,"intel",10,output,dispatch="inline")
        for change in ({"device_dispatch":"threaded"},{"observed_inline_device_dispatch":False}):
            bad=dict(value,**change)
            with self.assertRaises(ValueError):
                LAB.validate_receipt(bad,"intel",10,output,dispatch="inline")
        with self.assertRaises(ValueError):
            LAB.validate_receipt(value,"intel",10,completion(10),dispatch="inline")
        with self.assertRaises(ValueError):
            LAB.validate_receipt(value,"intel",0,output,dispatch="inline")
        with self.assertRaises(ValueError):
            LAB.worker_environment("/original/model","/public/receipt",dispatch="unknown")
        env=LAB.worker_environment("/original/model","/public/receipt",dispatch="inline")
        self.assertEqual(env["KUNO_LAB_DEVICE_DISPATCH"],"inline")

    def test_batched_info_requires_compact_slabs_and_independent_runtime_marker(self):
        value=receipt("intel",10)
        value.update(compact_info=True,info_slab_bytes=4096,info_batch_upload=True)
        output=completion(10)+"\nprivate_info_batch_upload enabled=true"
        LAB.validate_receipt(value,"intel",10,output,compact_info=True,info_batch_upload=True)
        with self.assertRaises(ValueError):
            LAB.validate_receipt(value,"intel",10,completion(10),compact_info=True,info_batch_upload=True)
        with self.assertRaises(ValueError):
            LAB.worker_environment("/original/model","/public/receipt",info_batch_upload=True)
        env=LAB.worker_environment("/original/model","/public/receipt",compact_info=True,info_batch_upload=True)
        self.assertEqual(env["KUNO_LAB_INFO_BATCH_UPLOAD"],"1")

    def test_shared_kernels_require_original_numeric_probes_and_all_dense_calls(self):
        value = receipt("intel", 10)
        value.update(fast_gpu=True, observed_fast_dense_calls=5264,
                     gpu_router=True, observed_gpu_router_calls=336,
                     next_shared_kernels=True, observed_shared_dense_calls=5264, observed_shared_gr_calls=679)
        markers = ("rust_gpu_router cases=6 ranks=10 numeric_and_owning_checks=true",
                   "next_shared_original_rows cases=6 numeric_and_owning_checks=true",
                   "next_shared_gpu_router cases=8 numeric_and_owning_checks=true",
                   "next_shared_gr_rms counts=1,3,8 cases=6 numeric_finite_status_and_owning_checks=true")
        output = completion(10) + "\n" + "\n".join(markers)
        LAB.validate_receipt(value, "intel", 10, output, fast=True, gpu_router=True, shared=True)
        for bad in (None, True, 0, 5263, 5265, -1):
            wrong = copy.deepcopy(value)
            wrong["observed_shared_dense_calls"] = bad
            with self.assertRaises(ValueError):
                LAB.validate_receipt(wrong, "intel", 10, output, fast=True, gpu_router=True, shared=True)
        for bad in (None, True, 0, -1, 1_000_001):
            wrong=copy.deepcopy(value);wrong["observed_shared_gr_calls"]=bad
            with self.assertRaises(ValueError):
                LAB.validate_receipt(wrong,"intel",10,output,fast=True,gpu_router=True,shared=True)
        for marker in markers:
            with self.assertRaises(ValueError):
                LAB.validate_receipt(value, "intel", 10, output.replace(marker, ""), fast=True, gpu_router=True, shared=True)
        for kwargs in ({"shared": True}, {"shared": True, "fast": True},
                       {"shared": 1, "fast": True, "gpu_router": True}):
            with self.assertRaises(ValueError):
                LAB.worker_environment("/original/model", "/public/receipt", **kwargs)
        combined=copy.deepcopy(value);combined.update(fast_gpu_fault=True,observed_fast_dense_fault_calls=5264)
        fused_output=output+"\n"+"\n".join(f"next_shared_packed_fused_fault dtype={d} cases=5 passed=true" for d in (0,1,30,12,6,14,8))
        LAB.validate_receipt(combined,"intel",10,fused_output,fast=True,gpu_router=True,shared=True,fast_fault=True)
        with self.assertRaises(ValueError):LAB.validate_receipt(combined,"intel",10,fused_output.replace("next_shared_packed_fused_fault dtype=14 cases=5 passed=true",""),fast=True,gpu_router=True,shared=True,fast_fault=True)
        LAB.worker_environment("/original/model", "/public/receipt", fast=True, gpu_router=True, shared=True,fast_fault=True)
        env = LAB.worker_environment("/original/model", "/public/receipt", fast=True, gpu_router=True, shared=True)
        self.assertEqual(env["KUNO_LAB_NEXT_SHARED"], "1")
        value.update(next_shared_gdn=True, observed_shared_gdn_calls=252)
        output += "\nnext_shared_gdn cases=3 numeric_and_owning_checks=true"
        LAB.validate_receipt(value, "intel", 10, output, fast=True, gpu_router=True, shared=True, shared_gdn=True)
        for bad in (0, 251, 253, True, None):
            wrong=copy.deepcopy(value);wrong["observed_shared_gdn_calls"]=bad
            with self.assertRaises(ValueError):
                LAB.validate_receipt(wrong, "intel", 10, output, fast=True, gpu_router=True, shared=True, shared_gdn=True)
        with self.assertRaises(ValueError):
            LAB.worker_environment("/original/model", "/public/receipt", shared_gdn=True)

    def test_deferred_epochs_require_actual_bridges_late_cancel_and_five_owning_audits(self):
        value=receipt("intel",10)
        value.update(compact_info=True,info_slab_bytes=4096,defer_router_io=True,observed_defer_router_io=True,observed_deferred_router_reads=336,io_token_checkpoint=True)
        value["owned_zero_audits"].append(copy.deepcopy(value["owned_zero_audits"][-1]))
        output=completion(10)+"\npublic_count1_hybrid_mid_token_cancel cap=10 layer=47 erased=true"
        LAB.validate_receipt(value,"intel",10,output,compact_info=True,deferred=True)
        for field,bads in (("defer_router_io",(False,1,None)),("observed_defer_router_io",(False,1,None)),("observed_deferred_router_reads",(335,337,True,None)),("io_token_checkpoint",(False,1,None))):
            for bad in bads:
                wrong=copy.deepcopy(value);wrong[field]=bad
                with self.assertRaises(ValueError):LAB.validate_receipt(wrong,"intel",10,output,compact_info=True,deferred=True)
        wrong=copy.deepcopy(value);wrong["owned_zero_audits"].pop()
        with self.assertRaises(ValueError):LAB.validate_receipt(wrong,"intel",10,output,compact_info=True,deferred=True)
        with self.assertRaises(ValueError):LAB.validate_receipt(value,"intel",10,completion(10),compact_info=True,deferred=True)
        with self.assertRaises(ValueError):LAB.validate_receipt(value,"intel",0,output,compact_info=True,deferred=True)
        with self.assertRaises(ValueError):LAB.worker_environment("/original/model","/public/receipt",deferred=True)
        env=LAB.worker_environment("/original/model","/public/receipt",compact_info=True,deferred=True)
        self.assertEqual(env["KUNO_LAB_DEFER_ROUTER_IO"],"1")
        self.assertEqual(env["KUNO_LAB_IO_TOKEN_CHECKPOINT"],"0") # Original flag cannot silently select protected policy.

    def test_fused_finite_requires_every_dispatch_and_all_format_fault_probes(self):
        value=receipt("intel",10)
        value.update(fast_gpu=True,observed_fast_dense_calls=5264,fast_gpu_fault=True,observed_fast_dense_fault_calls=5264)
        output=completion(10)+"\n"+"\n".join(f"rust_fast_packed_fused_fault dtype={dtype} cases=4 passed=true" for dtype in (0,1,30,12,6,14,8))
        LAB.validate_receipt(value,"intel",10,output,fast=True,fast_fault=True)
        for bad in (True,0,5263,5265,None):
            wrong=copy.deepcopy(value);wrong["observed_fast_dense_fault_calls"]=bad
            with self.assertRaises(ValueError):LAB.validate_receipt(wrong,"intel",10,output,fast=True,fast_fault=True)
        for dtype in (0,1,30,12,6,14,8):
            wrong=output.replace(f"rust_fast_packed_fused_fault dtype={dtype} cases=4 passed=true","")
            with self.assertRaises(ValueError):LAB.validate_receipt(value,"intel",10,wrong,fast=True,fast_fault=True)
        with self.assertRaises(ValueError):LAB.worker_environment("/original/model","/public/receipt",fast_fault=True)

    def test_compact_info_and_larger_submission_groups_require_typed_matching_bounds(self):
        for compact in (False,True):
            for batch in (False,True):
                value=receipt("intel",10)
                value.update(compact_info=compact,info_slab_bytes=4096 if compact else 0,batch_tasks=batch,runtime_tasks_max=256 if batch else 32)
                LAB.validate_receipt(value,"intel",10,completion(10),compact_info=compact,batch_tasks=batch)
                for field in ("compact_info","info_slab_bytes","batch_tasks","runtime_tasks_max"):
                    for bad in (None,"1",True if type(value[field]) is int else 1):
                        wrong=copy.deepcopy(value);wrong[field]=bad
                        with self.assertRaises(ValueError):LAB.validate_receipt(wrong,"intel",10,completion(10),compact_info=compact,batch_tasks=batch)
                env=LAB.worker_environment("/original/model","/public/receipt",compact_info=compact,batch_tasks=batch)
                self.assertEqual(env["KUNO_LAB_INFO_SLAB_BYTES"],"4096" if compact else "0")
                self.assertEqual(env["KUNO_LAB_BATCH_TASKS"],"1" if batch else "0")

    def test_q8_requires_pooled_workers_and_exact_completed_arithmetic_evidence(self):
        value = receipt("intel",10)
        value.update(count1_hybrid_cpu_workers="pooled",observed_count1_hybrid_cpu_workers="pooled",cpu_q8_activations=True,observed_cpu_q8_activations=True)
        LAB.validate_receipt(value,"intel",10,completion(10),workers="pooled",cpu_q8=True)
        for field in ("cpu_q8_activations","observed_cpu_q8_activations"):
            for bad in (None,False,1,"true"):
                wrong=copy.deepcopy(value);wrong[field]=bad
                with self.assertRaises(ValueError):
                    LAB.validate_receipt(wrong,"intel",10,completion(10),workers="pooled",cpu_q8=True)
        with self.assertRaises(ValueError):
            LAB.worker_environment("/original/model","/public/receipt",cpu_q8=True)
        env=LAB.worker_environment("/original/model","/public/receipt",workers="pooled",cpu_q8=True)
        self.assertEqual(env["KUNO_LAB_CPU_Q8"],"1")

    def test_larger_model_cache_requires_mapping_and_actual_bounded_no_swap_cgroup(self):
        spec=importlib.util.spec_from_file_location("protected_lab",HERE/"protected-lab-exec.py")
        guard=importlib.util.module_from_spec(spec);spec.loader.exec_module(guard)
        env={"KUNO_LAB_LARGE_WEIGHT_CACHE":"1","KUNO_LAB_MAPPED_WEIGHTS":"1"}
        args=["/frozen/binary","intel",LAB.MODE,"10"]
        self.assertEqual(guard.memory_limit(env,args),48*1024**3)
        for cap in range(11):
            self.assertEqual(guard.memory_limit(env,[*args[:3],str(cap)]),48*1024**3)
        self.assertEqual(guard.memory_limit({},args),8*1024**3)
        for bad in ("auto","48","",True):
            with self.assertRaises(ValueError):guard.memory_limit(dict(env,KUNO_LAB_LARGE_WEIGHT_CACHE=bad),args)
        for badargs in (args[:-1],[*args,"arbitrary"],[args[0],"cpu",LAB.MODE,"10"],[args[0],"intel","arbitrary","10"],*([*args[:3],cap] for cap in ("11","-1","01"," 1",True))):
            with self.assertRaises(ValueError):guard.memory_limit(env,badargs)
        with self.assertRaises(ValueError):guard.memory_limit(dict(env,KUNO_LAB_MAPPED_WEIGHTS="0"),args)
        value=receipt("intel",10);value.update(mapped_weights=True,observed_mapped_weights=True,large_weight_cache=True)
        output=completion(10).replace("8589934592","51539607552")
        LAB.validate_receipt(value,"intel",10,output,mapped=True,large_cache=True)
        for wrong in (completion(10),output.replace("51539607552","max"),output.replace("51539607552","55834574848")):
            with self.assertRaises(ValueError):LAB.validate_receipt(value,"intel",10,wrong,mapped=True,large_cache=True)
        with self.assertRaises(ValueError):LAB.worker_environment("/original/model","/public/receipt",large_cache=True)

    def test_target_rows_large_cache_is_exclusive_to_compiled_public_fixture(self):
        spec=importlib.util.spec_from_file_location("protected_target",HERE/"protected-lab-exec.py")
        guard=importlib.util.module_from_spec(spec);spec.loader.exec_module(guard)
        env={"KUNO_LAB_TARGET_ROWS_PUBLIC_ONLY":"1", "KUNO_LAB_STATIC_CACHE_BYTES":"8589934592", "KUNO_TEST_Q3_DIR":"/original", "KUNO_PUBLIC_TARGET_ROWS_REFERENCE":"/public/reference", "KUNO_PUBLIC_RECEIPT":"/public/receipt"}
        args=["/frozen/binary","intel","daslab-target-rows-probe"]
        self.assertEqual(guard.memory_limit(env,args),48*1024**3)
        for key in env:
            wrong=dict(env);wrong.pop(key)
            if key != "KUNO_LAB_TARGET_ROWS_PUBLIC_ONLY":
                with self.assertRaises(ValueError):guard.memory_limit(wrong,args)
            wrong=dict(env);wrong[key]=""
            with self.assertRaises(ValueError):guard.memory_limit(wrong,args)
        for key,value in (("KUNO_LAB_TARGET_ROWS_PUBLIC_ONLY","0"),("KUNO_LAB_STATIC_CACHE_BYTES","0"),("KUNO_LAB_LARGE_WEIGHT_CACHE","1"),("KUNO_LAB_MAPPED_WEIGHTS","0")):
            with self.assertRaises(ValueError):guard.memory_limit(dict(env,**{key:value}),args)
        for wrong in (args[:-1],[*args,"extra"],[args[0],"nvidia",args[2]],[args[0],"intel","probe"]):
            with self.assertRaises(ValueError):guard.memory_limit(env,wrong)

    def test_target_repair_large_cache_rejects_other_public_modes(self):
        spec=importlib.util.spec_from_file_location("protected_repair",HERE/"protected-lab-exec.py")
        guard=importlib.util.module_from_spec(spec);spec.loader.exec_module(guard)
        env={"KUNO_LAB_TARGET_REPAIR_PUBLIC_ONLY":"1", "KUNO_LAB_STATIC_CACHE_BYTES":"8589934592", "KUNO_TEST_Q3_DIR":"/original", "KUNO_PUBLIC_TARGET_ROWS_REFERENCE":"/public/reference", "KUNO_PUBLIC_RECEIPT":"/public/receipt"}
        args=["/frozen/binary","intel","daslab-target-repair-probe"]
        self.assertEqual(guard.memory_limit(env,args),48*1024**3)
        for wrong in (dict(env,KUNO_LAB_TARGET_ROWS_PUBLIC_ONLY="1"),dict(env,KUNO_LAB_TARGET_REPAIR_PUBLIC_ONLY="0"),dict(env,KUNO_LAB_LARGE_WEIGHT_CACHE="1")):
            with self.assertRaises(ValueError):guard.memory_limit(wrong,args)
        for mode in ("daslab-target-rows-probe","resident-mtp-probe"):
            with self.assertRaises(ValueError):guard.memory_limit(env,[*args[:2],mode])
        with self.assertRaises(ValueError):guard.memory_limit(env,[args[0],"nvidia",args[2]])

    def test_dual_gpu_speculation_keeps_exact_public_cache_and_artifact_gates(self):
        spec=importlib.util.spec_from_file_location("protected_speculation",HERE/"protected-lab-exec.py")
        guard=importlib.util.module_from_spec(spec);spec.loader.exec_module(guard)
        env={"KUNO_LAB_SPECULATIVE_PUBLIC_ONLY":"1", "KUNO_LAB_STATIC_CACHE_BYTES":"8589934592", "KUNO_TEST_Q3_DIR":"/original", "KUNO_TEST_MTP":"/original/draft", "KUNO_PUBLIC_TARGET_CARRY13_REFERENCE":"/public/reference", "KUNO_PUBLIC_RECEIPT":"/public/receipt"}
        args=["/frozen/binary","intel","daslab-speculative-probe"]
        self.assertEqual(guard.memory_limit(env,args),48*1024**3)
        for key in ("KUNO_TEST_MTP","KUNO_PUBLIC_TARGET_CARRY13_REFERENCE"):
            wrong=dict(env);wrong.pop(key)
            with self.assertRaises(ValueError):guard.memory_limit(wrong,args)
        for wrong in (dict(env,KUNO_LAB_TARGET_ROWS_PUBLIC_ONLY="1"),dict(env,KUNO_LAB_SPECULATIVE_PUBLIC_ONLY="0"),dict(env,KUNO_LAB_LARGE_WEIGHT_CACHE="1")):
            with self.assertRaises(ValueError):guard.memory_limit(wrong,args)
        for mode in ("daslab-target-repair-probe","resident-mtp-probe"):
            with self.assertRaises(ValueError):guard.memory_limit(env,[*args[:2],mode])
        with self.assertRaises(ValueError):guard.memory_limit(env,[args[0],"nvidia",args[2]])

    def test_verify3_operator_keeps_exact_small_public_envelope(self):
        spec=importlib.util.spec_from_file_location("protected_verify3_operator",HERE/"protected-lab-exec.py")
        guard=importlib.util.module_from_spec(spec);spec.loader.exec_module(guard)
        env={"KUNO_LAB_VERIFY3_DENSE_PUBLIC_ONLY":"1", "KUNO_PUBLIC_RECEIPT":"/public/numeric"}
        for card in ("intel", "nvidia"):
            args=["/frozen/binary",card,"verify3-dense-probe"]
            self.assertEqual(guard.memory_limit(env,args),8*1024**3)
            for wrong in (dict(env,KUNO_TEST_Q3_DIR="/model"), dict(env,KUNO_LAB_VERIFY3_DENSE_PUBLIC_ONLY="0"), dict(env,KUNO_PUBLIC_RECEIPT="")):
                with self.assertRaises(ValueError):guard.memory_limit(wrong,args)
            with self.assertRaises(ValueError):guard.memory_limit(env,[*args,"extra"])

    def test_verify3_finite_operator_keeps_exact_small_public_envelope(self):
        spec=importlib.util.spec_from_file_location("protected_verify3_finite",HERE/"protected-lab-exec.py")
        guard=importlib.util.module_from_spec(spec);spec.loader.exec_module(guard)
        env={"KUNO_LAB_VERIFY3_FINITE_PUBLIC_ONLY":"1", "KUNO_PUBLIC_RECEIPT":"/public/numeric"}
        for card in ("intel", "nvidia"):
            args=["/frozen/binary",card,"verify3-finite-probe"]
            self.assertEqual(guard.memory_limit(env,args),8*1024**3)
            for wrong in (dict(env,KUNO_TEST_Q3_DIR="/model"), dict(env,KUNO_LAB_VERIFY3_FINITE_PUBLIC_ONLY="0"), dict(env,KUNO_LAB_VERIFY3_DENSE_PUBLIC_ONLY="1"), dict(env,KUNO_PUBLIC_RECEIPT="")):
                with self.assertRaises(ValueError):guard.memory_limit(wrong,args)
            with self.assertRaises(ValueError):guard.memory_limit(env,[*args,"extra"])
            with self.assertRaises(ValueError):guard.memory_limit(env,[*args[:2],"verify3-dense-probe"])

    def test_verify3_dense_is_canonical_optional_and_remains_inside_public_target_modes(self):
        spec=importlib.util.spec_from_file_location("protected_verify3",HERE/"protected-lab-exec.py")
        guard=importlib.util.module_from_spec(spec);spec.loader.exec_module(guard)
        for gate,mode in (("KUNO_LAB_SPECULATIVE_PUBLIC_ONLY","daslab-speculative-probe"), ("KUNO_LAB_TARGET_ROWS_PUBLIC_ONLY","daslab-target-rows-probe"), ("KUNO_LAB_TARGET_REPAIR_PUBLIC_ONLY","daslab-target-repair-probe")):
            env={gate:"1", "KUNO_LAB_STATIC_CACHE_BYTES":"8589934592", "KUNO_TEST_Q3_DIR":"/original", "KUNO_PUBLIC_RECEIPT":"/public/receipt", "KUNO_LAB_TARGET_DENSE":"verify3"}
            if gate=="KUNO_LAB_SPECULATIVE_PUBLIC_ONLY":
                env.update(KUNO_TEST_MTP="/original/sidecar", KUNO_PUBLIC_TARGET_CARRY13_REFERENCE="/public/reference")
            else:
                env["KUNO_PUBLIC_TARGET_ROWS_REFERENCE"]="/public/reference"
            args=["/frozen/binary","intel",mode]
            self.assertEqual(guard.memory_limit(env,args),48*1024**3)
            self.assertEqual(guard.memory_limit(dict(env,KUNO_LAB_TARGET_DENSE="verify3-fused"),args),48*1024**3)
            self.assertEqual(guard.memory_limit(dict(env,KUNO_LAB_TARGET_IO="chunk"),args),48*1024**3)
            self.assertEqual(guard.memory_limit(dict(env,KUNO_LAB_TARGET_EXPERTS="prepared-verify3"),args),48*1024**3)
            self.assertEqual(guard.memory_limit(dict(env,KUNO_LAB_TARGET_FILL="direct-public"),args),48*1024**3)
            for invalid in ("", "auto", "0", "1", "direct-public ", True):
                with self.assertRaises(ValueError):guard.memory_limit(dict(env,KUNO_LAB_TARGET_FILL=invalid),args)
            for invalid in ("", "auto", "0", "1", "prepared-verify3 ", True):
                with self.assertRaises(ValueError):guard.memory_limit(dict(env,KUNO_LAB_TARGET_EXPERTS=invalid),args)
            if gate=="KUNO_LAB_SPECULATIVE_PUBLIC_ONLY":
                self.assertEqual(guard.memory_limit(dict(env,KUNO_LAB_MTP_SWEEPS="fused"),args),48*1024**3)
                self.assertEqual(guard.memory_limit(dict(env,KUNO_LAB_WARM_REPLAY="once"),args),48*1024**3)
                for invalid in ("", "auto", "0", "1", "once ", True):
                    with self.assertRaises(ValueError):guard.memory_limit(dict(env,KUNO_LAB_WARM_REPLAY=invalid),args)
                for invalid in ("", "auto", "0", "1", "fused ", True):
                    with self.assertRaises(ValueError):guard.memory_limit(dict(env,KUNO_LAB_MTP_SWEEPS=invalid),args)
            else:
                with self.assertRaises(ValueError):guard.memory_limit(dict(env,KUNO_LAB_MTP_SWEEPS="fused"),args)
                with self.assertRaises(ValueError):guard.memory_limit(dict(env,KUNO_LAB_WARM_REPLAY="once"),args)
            for invalid in ("", "auto", "0", "1", "chunk ", True):
                with self.assertRaises(ValueError):guard.memory_limit(dict(env,KUNO_LAB_TARGET_IO=invalid),args)
            for invalid in ("", "auto", "0", "1", "verify3 ", True):
                with self.assertRaises(ValueError):guard.memory_limit(dict(env,KUNO_LAB_TARGET_DENSE=invalid),args)
            with self.assertRaises(ValueError):guard.memory_limit(dict(env,KUNO_LAB_OTHER="1"),args)

    def test_resident_mtp_fused_sweeps_keep_fixed_public_artifact_envelope(self):
        spec=importlib.util.spec_from_file_location("protected_resident_mtp",HERE/"protected-lab-exec.py")
        guard=importlib.util.module_from_spec(spec);spec.loader.exec_module(guard)
        env={"KUNO_LAB_RESIDENT_MTP_PUBLIC_ONLY":"1", "KUNO_TEST_MTP":"/original/draft", "KUNO_PUBLIC_DASLAB_CARRY":"/public/carry", "KUNO_PUBLIC_MTP_REFERENCE":"/public/reference", "KUNO_PUBLIC_RECEIPT":"/public/receipt"}
        for card in ("intel","nvidia"):
            args=["/frozen/binary",card,"resident-mtp-probe"]
            self.assertEqual(guard.memory_limit(env,args),8*1024**3)
            self.assertEqual(guard.memory_limit(dict(env,KUNO_LAB_MTP_SWEEPS="fused"),args),8*1024**3)
            for invalid in ("", "auto", "0", "1", "fused ", True):
                with self.assertRaises(ValueError):guard.memory_limit(dict(env,KUNO_LAB_MTP_SWEEPS=invalid),args)
            for key in env:
                wrong=dict(env);wrong.pop(key)
                with self.assertRaises(ValueError):guard.memory_limit(wrong,args)
            with self.assertRaises(ValueError):guard.memory_limit(dict(env,KUNO_LAB_TARGET_EXPERTS="prepared-verify3"),args)
            with self.assertRaises(ValueError):guard.memory_limit(dict(env,KUNO_LAB_WARM_REPLAY="once"),args)
            with self.assertRaises(ValueError):guard.memory_limit(env,[*args,"extra"])

    def test_fast_gpu_requires_actual_dispatches_and_separate_long_selection(self):
        value = receipt("intel",10)
        value.update(fast_gpu=True,observed_fast_dense_calls=5264,long_fixture=False)
        LAB.validate_receipt(value,"intel",10,completion(10),fast=True)
        for field, bads in (("fast_gpu",(False,1,"true")),("observed_fast_dense_calls",(0,True,"5264",100001)),("long_fixture",(True,1,"false"))):
            for bad in bads:
                wrong=copy.deepcopy(value);wrong[field]=bad
                with self.assertRaises(ValueError):
                    LAB.validate_receipt(wrong,"intel",10,completion(10),fast=True)
        with mock.patch.dict(os.environ,{"KUNO_LAB_FAST_GPU":"auto","KUNO_LAB_LONG_FIXTURE":"auto"}):
            env=LAB.worker_environment("/original/model","/public/receipt",fast=True,long=False)
        self.assertEqual(env["KUNO_LAB_FAST_GPU"],"1")
        self.assertEqual(env["KUNO_LAB_LONG_FIXTURE"],"0")

    def test_mapping_and_profiling_are_explicit_observed_and_never_inherited(self):
        for mapped in (False, True):
            for profile in (False, True):
                value = receipt("intel", 10)
                value.update(mapped_weights=mapped, observed_mapped_weights=mapped, kernel_profile=profile, kernel_profile_scope="first-following-decode" if profile else "disabled", profiled_native_decode_steps=1 if profile else 0)
                LAB.validate_receipt(value, "intel", 10, completion(10), mapped=mapped, profile=profile)
                for field, selected in (("mapped_weights", mapped), ("observed_mapped_weights", mapped), ("kernel_profile", profile)):
                    for bad in (0, "false", None, not selected):
                        wrong = copy.deepcopy(value)
                        wrong[field] = bad
                        with self.assertRaises(ValueError):
                            LAB.validate_receipt(wrong, "intel", 10, completion(10), mapped=mapped, profile=profile)
                with mock.patch.dict(os.environ, {"KUNO_LAB_MAPPED_WEIGHTS":"auto","KUNO_LAB_KERNEL_PROFILE":"auto"}):
                    env = LAB.worker_environment("/original/model", "/public/receipt", mapped=mapped, profile=profile)
                self.assertEqual(env["KUNO_LAB_MAPPED_WEIGHTS"],"1" if mapped else "0")
                self.assertEqual(env["KUNO_LAB_KERNEL_PROFILE"],"1" if profile else "0")
        for bad in (1,"1",None):
            with self.assertRaises(ValueError):
                LAB.worker_environment("/original/model", "/public/receipt", mapped=bad)

    def test_gpu_routing_pins_device_controls_actual_calls_and_finite_fixture(self):
        value=receipt("intel",10)
        value.update(gpu_router=True,observed_gpu_router_calls=336)
        output=completion(10)+"\nrust_gpu_router cases=6 ranks=10 numeric_and_owning_checks=true\n"
        LAB.validate_receipt(value,"intel",10,output,gpu_router=True)
        for field,bads in (("gpu_router",(False,1,"true")),("observed_gpu_router_calls",(0,True,"336",335,337))):
            for bad in bads:
                wrong=copy.deepcopy(value);wrong[field]=bad
                with self.assertRaises(ValueError):LAB.validate_receipt(wrong,"intel",10,output,gpu_router=True)
        with self.assertRaises(ValueError):LAB.validate_receipt(value,"intel",10,completion(10),gpu_router=True)
        with self.assertRaises(ValueError):LAB.validate_receipt(value,"intel",0,output,gpu_router=True)
        with mock.patch.dict(os.environ,{"KUNO_LAB_GPU_ROUTER":"auto"}):
            self.assertEqual(LAB.worker_environment("/original/model","/public/receipt",gpu_router=False)["KUNO_LAB_GPU_ROUTER"],"0")
            self.assertEqual(LAB.worker_environment("/original/model","/public/receipt",gpu_router=True)["KUNO_LAB_GPU_ROUTER"],"1")

    def test_metadata_requires_real_warm_lookups_protected_mode_and_explicit_policy(self):
        value = receipt("intel", 10)
        value.update(retained_metadata=True, observed_retained_metadata_calls=1200)
        LAB.validate_receipt(value, "intel", 10, completion(10), metadata=True)
        for field, bads in (("retained_metadata", (False, 1, "true")),
                            ("observed_retained_metadata_calls", (None, True, "1200", 0, -1, 1000001))):
            for bad in bads:
                wrong=copy.deepcopy(value);wrong[field]=bad
                with self.assertRaises(ValueError):
                    LAB.validate_receipt(wrong,"intel",10,completion(10),metadata=True)
        with self.assertRaises(ValueError):LAB.validate_receipt(value,"intel",0,completion(0),metadata=True)
        with mock.patch.dict(os.environ, {"KUNO_LAB_RETAINED_METADATA":"auto"}):
            for selected in (False,True):
                self.assertEqual(LAB.worker_environment("/original/model","/public/receipt",metadata=selected)["KUNO_LAB_RETAINED_METADATA"],"1" if selected else "0")

    def test_expanded_banks_pin_the_entire_device_budget_without_changing_other_settings(self):
        for banks in ("intel72","intel96","intel128","intel144","intel160"):
            value=receipt("intel",10)
            value.update(count1_bank_profile=banks, expert_bank_slots={"intel72":72,"intel96":96,"intel128":128,"intel144":144,"intel160":160}[banks],
                         expert_bank_segments=[72,72,16] if banks=="intel160" else [72,72] if banks=="intel144" else [72] if banks=="intel72" else [64,64] if banks in ("intel128","intel144","intel160") else [64,32])
            if banks=="intel96":
                value.update(block_moe=True,explicit_static_cache=True,
                             expert_bank_accounted_limit_bytes=18*1024**3,static_cache_budget_bytes=6*1024**3)
            if banks in ("intel128","intel144","intel160"):
                value.update(block_moe=True,explicit_static_cache=True,expert_bank_accounted_limit_bytes=(27 if banks=="intel160" else 24)*1024**3,static_cache_budget_bytes=(3 if banks=="intel160" else 5)*1024**3,device_reserved_limit_bytes=31*1024**3)
            LAB.validate_receipt(value,"intel",10,completion(10),banks=banks)
            for field in ("expert_bank_slots","expert_bank_segments","expert_bank_accounted_limit_bytes","static_cache_budget_bytes","device_reserved_limit_bytes","explicit_static_cache","count1_bank_profile"):
                wrong=copy.deepcopy(value);wrong[field]=None
                with self.assertRaises(ValueError):LAB.validate_receipt(wrong,"intel",10,completion(10),banks=banks)
            for card,cap in (("nvidia",10),("intel",0)):
                with self.assertRaises(ValueError):LAB.validate_receipt(value,card,cap,completion(cap),banks=banks)
            env=LAB.worker_environment("/original/model","/public/receipt",banks=banks)
            self.assertEqual(env["KUNO_LAB_COUNT1_BANKS"],banks)
            self.assertEqual(env["KUNO_LAB_BLOCK_MOE"],"1" if banks in ("intel96","intel128","intel144","intel160") else "0")
            self.assertEqual(env.get("KUNO_LAB_STATIC_CACHE_BYTES"),"3221225472" if banks=="intel160" else "5368709120" if banks in ("intel128","intel144") else "6442450944" if banks=="intel96" else None)

    def test_both_caps_and_cards_require_fixed_inventory_and_actual_cpu_work(self):
        for card in ("intel", "nvidia"):
            for cap in range(11):
                LAB.validate_receipt(receipt(card, cap), card, cap, completion(cap))

    def test_intermediate_caps_require_matching_actual_completed_work_within_their_own_bound(self):
        for cap in range(1, 10):
            value = receipt("intel", cap)
            value.update(device_dispatch="inline", observed_inline_device_dispatch=True)
            output = completion(cap) + "\ngpu_device_dispatch mode=inline blocking=true"
            LAB.validate_receipt(value, "intel", cap, output, dispatch="inline")
            for field, limit in (("completed_cpu_misses", 7 * 48 * cap),
                                 ("fresh_completed_cpu_misses", 48 * cap)):
                for bad in (0, limit + 1, True, None):
                    wrong = copy.deepcopy(value); wrong[field] = bad
                    with self.assertRaises(ValueError):
                        LAB.validate_receipt(wrong, "intel", cap, output, dispatch="inline")
            with self.assertRaises(ValueError):
                LAB.validate_receipt(value, "intel", (cap % 10) + 1, output, dispatch="inline")

    def test_overlap_policy_pins_both_observations_and_cannot_enable_scoped_workers(self):
        for overlap in (False, True):
            value = receipt("intel", 10)
            value.update(count1_hybrid_cpu_workers="pooled", observed_count1_hybrid_cpu_workers="pooled",
                         count1_hybrid_cpu_gpu_overlap=overlap, observed_count1_hybrid_cpu_gpu_overlap=overlap, observed_shared_overlap_calls=336 if overlap else 0)
            LAB.validate_receipt(value, "intel", 10, completion(10), "pooled", overlap)
            for bad in (None, True, "336", 335, 337, 0 if overlap else 336):
                wrong = copy.deepcopy(value)
                wrong["observed_shared_overlap_calls"] = bad
                with self.assertRaises(ValueError):
                    LAB.validate_receipt(wrong, "intel", 10, completion(10), "pooled", overlap)
            for field in ("count1_hybrid_cpu_gpu_overlap", "observed_count1_hybrid_cpu_gpu_overlap"):
                for bad in (None, 0, "false", not overlap):
                    wrong = copy.deepcopy(value)
                    wrong[field] = bad
                    with self.assertRaises(ValueError):
                        LAB.validate_receipt(wrong, "intel", 10, completion(10), "pooled", overlap)
            with mock.patch.dict(os.environ, {"KUNO_LAB_CPU_GPU_OVERLAP": "auto"}):
                env = LAB.worker_environment("/original/model", "/public/receipt", "pooled", overlap)
            self.assertEqual(env["KUNO_LAB_CPU_GPU_OVERLAP"], "1" if overlap else "0")
        for workers, overlap in (("scoped", True), ("pooled", 1), ("pooled", "1")):
            with self.assertRaises(ValueError):
                LAB.worker_environment("/original/model", "/public/receipt", workers, overlap)

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
        for cap in (False, "0", 0.0, None, -1, 11):
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
                build = pathlib.Path(directory) / "build.jsonl"
                build.write_text(json.dumps({"reason":"compiler-artifact","target":{"name":"kuno-text-lab"},"executable":"/frozen","features":[]})+"\n")
                command_seen = []
                def launch(command, **options):
                    command_seen.extend(command)
                    value = receipt("intel", 10)
                    if not valid:
                        value["completed_cpu_misses"] = 0
                    target.write_text(json.dumps(value))
                    return SimpleNamespace(stdout=io.StringIO(completion(10)), wait=lambda: 0)
                args = ["lab", "--cargo-artifacts", str(build), "--model", "/original/model",
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
                self.assertIn("--setenv=KUNO_LAB_ADAPTIVE_DIRECT_UPLOAD=0", command_seen)
                record = target.with_suffix(".json.run.json")
                self.assertEqual(record.exists(), valid)
                if valid:
                    data = json.loads(record.read_text())
                    self.assertFalse(data["matched_performance_measurement"])
                    self.assertFalse(data["cpu_gpu_overlap"])
                    self.assertEqual(data["cpu_miss_cap"], 10)
                    self.assertFalse(data["adaptive_direct_upload"])


if __name__ == "__main__":
    unittest.main()
