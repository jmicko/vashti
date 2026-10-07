"""Strict public inventory, independent authority and corruption tests."""

import copy
import importlib.util
import json
import pathlib
import subprocess
import sys
import tempfile
import unittest


SCRIPT = pathlib.Path(__file__).with_name("validate-native-single-dense.py")
SPEC = importlib.util.spec_from_file_location("single_dense_validator", SCRIPT)
VALIDATOR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(VALIDATOR)
SHAPES = ((0, 7, 2560), (1, 7, 4), (30, 7, 2560), (12, 7, 2560),
          (6, 7, 160), (14, 7, 2560), (8, 7, 640))
INTEL, NVIDIA = 0x8086, 0x10DE


def page(words):
    size = 32768
    while size < words * 4:
        size *= 2
    return size


def receipt(shapes=SHAPES):
    # The fixture pins producer order independently, including the extra
    # count1 replay between ordinary count1 and count3 cases for every format.
    schedule = [(1, 1, False), (1, 4, False), (1, 8, False), (1, 1, True),
                (3, 1, False), (3, 4, False), (3, 8, False),
                (5, 1, False), (5, 4, False), (5, 8, False),
                (8, 1, False), (8, 4, False), (8, 8, False)]
    block_layout = {0: (1, 4), 1: (1, 2), 30: (1, 2), 12: (256, 144),
                    6: (32, 22), 14: (256, 210), 8: (32, 34)}
    cases = []
    epoch = 1
    for dtype, rows, columns in shapes:
        input_bytes = page(8 * (columns + 11) + 9)
        output_bytes = page(8 * (rows + 5) + 7)
        elements, block_bytes = block_layout[dtype]
        wire = rows * (columns // elements) * block_bytes + (12 if dtype == 0 else 3)
        wire = (wire + 3) // 4 * 4
        cache_bytes = 256
        while cache_bytes < wire:
            cache_bytes *= 2
        for count, tile, single in schedule:
            upload = input_bytes + 256 + (output_bytes if single else 0)
            uniform = 256
            readback = output_bytes + upload + uniform
            result = {
                "dtype": dtype, "rows": rows, "columns": columns, "count": count,
                "tile_tokens": tile, "max_abs_error": 0.,
                "main_full_span_bytes": input_bytes + output_bytes,
                "gaps_and_inactive_preserved": True, "main_erased": True,
                "public_cache_preserved": True,
                "io": {"epoch": epoch, "upload_bytes": upload,
                       "uniform_bytes": uniform, "readback_bytes": readback,
                       "inspected_bytes": upload + uniform + readback,
                       "inspected_buffers": 9 if single else 7},
                "main_audit": {"epoch": epoch + 1, "upload_bytes": 0,
                               "uniform_bytes": 0, "readback_bytes": input_bytes + output_bytes,
                               "inspected_bytes": input_bytes + output_bytes, "inspected_buffers": 2},
                "cache_audit": {"epoch": epoch + 2, "upload_bytes": 0,
                                "uniform_bytes": 0, "readback_bytes": cache_bytes,
                                "inspected_bytes": cache_bytes, "inspected_buffers": 1},
            }
            if single:
                result.update(member_capacity=1, max8_prefill_reused=True,
                              generic_max_abs_error=0.)
            cases.append(result)
            epoch += 3
    return {
        "scope": "fixed-public-synthetic-operator-test-only", "fixture": "packed-matmul-probe",
        "borrowed_original_gguf": True, "format_count": 7, "case_count": 91,
        "single_member_cases": 7, "cases": cases, "private_uploads_owned": True,
        "context_retired": False, "operator_timings": None,
        "proof_scope": "owned API-visible main and I/O spans; not opaque physical driver padding",
    }


class SingleDense(unittest.TestCase):
    def accepted(self, value, shapes=SHAPES, vendor=INTEL):
        self.assertTrue(VALIDATOR.validate_receipt(
            value, expected_shapes=shapes, expected_vendor=vendor))

    def rejected(self, value, shapes=SHAPES, vendor=INTEL):
        with self.assertRaisesRegex(ValueError, "Fixed-public single dense numeric receipt rejected"):
            VALIDATOR.validate_receipt(value, expected_shapes=shapes, expected_vendor=vendor)

    def test_complete_inventory_and_nonzero_reported_maxima_accept_for_both_vendors(self):
        value = receipt()
        self.assertEqual(len(value["cases"]), 91)
        self.assertEqual(sum("member_capacity" in row for row in value["cases"]), 7)
        self.assertEqual(value["cases"][0]["main_full_span_bytes"], 163840)
        for row in value["cases"]:
            row["max_abs_error"] = 1e-6
            if "generic_max_abs_error" in row:
                row["generic_max_abs_error"] = 2e-6
        for vendor in (INTEL, NVIDIA):
            self.accepted(value, vendor=vendor)

    def test_vendor_is_required_independent_authority_and_never_a_fabricated_receipt_field(self):
        for vendor in (None, "intel", True, float(INTEL), 0, 0x1002, -1, 65536):
            self.rejected(receipt(), vendor=vendor)
        value = receipt()
        value["vendor"] = INTEL
        self.rejected(value)
        with self.assertRaises(TypeError):
            VALIDATOR.validate_receipt(receipt(), expected_shapes=SHAPES)
        with self.assertRaises(TypeError):
            VALIDATOR.validate_receipt(receipt(), expected_vendor=INTEL)

    def test_shapes_are_independently_required_typed_bounded_and_format_ordered(self):
        for shapes in (None, [], SHAPES[:6], SHAPES + (SHAPES[0],), list(SHAPES),
                       tuple(reversed(SHAPES)), (SHAPES[1],) + SHAPES[1:]):
            self.rejected(receipt(), shapes=shapes)
        for bad in ((True, 7, 2560), (0., 7, 2560), [0, 7, 2560],
                    (0, True, 2560), (0, 0, 2560), (0, 8, 2560), (0, 7., 2560),
                    (0, 7, True), (0, 7, 0), (0, 7, 65537), (0, 7, 2560.),
                    (0, 7, 65536), (0, 7), (0, 7, 2560, 8)):
            self.rejected(receipt(), shapes=(bad,) + SHAPES[1:])
        for index, columns in ((3, 255), (4, 159), (5, 257), (6, 641)):
            shapes = list(SHAPES)
            dtype, rows, _ = shapes[index]
            shapes[index] = (dtype, rows, columns)
            self.rejected(receipt(), shapes=tuple(shapes))

    def test_alternate_original_shape_needs_independent_authority_and_exact_owning_formula(self):
        alternate = ((0, 3, 16),) + SHAPES[1:]
        value = receipt(alternate)
        self.rejected(value)
        self.accepted(value, shapes=alternate)
        value["cases"][0]["main_full_span_bytes"] = 163840
        self.rejected(value, shapes=alternate)

    def test_every_top_level_identity_flag_and_type_is_strict(self):
        for key, current in receipt().items():
            if key == "cases":
                continue
            value = receipt()
            del value[key]
            self.rejected(value)
            value = receipt()
            value[key] = (not current if type(current) is bool else
                          current + 1 if type(current) is int else "wrong")
            self.rejected(value)
            if type(current) in (bool, int):
                value[key] = int(current) if type(current) is bool else float(current)
                self.rejected(value)
        value = receipt()
        value["numeric_accepted"] = True
        self.rejected(value)

    def test_timings_are_not_numeric_authority_and_cannot_enter_this_mode(self):
        for timings in ({}, [], False, {"speed": 10}, "disabled", 0):
            value = receipt()
            value["operator_timings"] = timings
            self.rejected(value)

    def test_every_case_public_shape_count_tile_owner_and_guard_is_checked(self):
        fields = ("dtype", "rows", "columns", "count", "tile_tokens", "main_full_span_bytes",
                  "gaps_and_inactive_preserved", "main_erased", "public_cache_preserved")
        for index in range(91):
            for key in fields:
                value = receipt()
                current = value["cases"][index][key]
                value["cases"][index][key] = not current if type(current) is bool else current + 1
                self.rejected(value)
                value["cases"][index][key] = int(current) if type(current) is bool else True
                self.rejected(value)
        for key in receipt()["cases"][0]:
            value = receipt()
            del value["cases"][0][key]
            self.rejected(value)
        value = receipt()
        value["cases"][0]["padding_accepted"] = True
        self.rejected(value)

    def test_short_extra_duplicate_reordered_and_wrong_container_inventories_reject(self):
        for how in ("short", "extra", "duplicate", "swap", "tuple", "wrong_row"):
            value = receipt()
            rows = value["cases"]
            if how == "short":
                rows.pop()
            elif how == "extra":
                rows.append(copy.deepcopy(rows[-1]))
            elif how == "duplicate":
                rows[-1] = copy.deepcopy(rows[0])
            elif how == "swap":
                rows[0], rows[3] = rows[3], rows[0]
            elif how == "tuple":
                value["cases"] = tuple(rows)
            else:
                rows[0] = []
            self.rejected(value)

    def test_archived_84_cases_and_missing_replay_contract_never_fall_back(self):
        value = receipt()
        value["cases"] = [row for row in value["cases"] if "member_capacity" not in row]
        value["case_count"] = 84
        value["single_member_cases"] = 0
        self.rejected(value)
        for key in ("member_capacity", "max8_prefill_reused", "generic_max_abs_error"):
            value = receipt()
            for index in range(3, 91, 13):
                del value["cases"][index][key]
            self.rejected(value)

    def test_all_seven_replays_need_all_three_exact_candidate_fields(self):
        for index in range(3, 91, 13):
            for key, bad in (("member_capacity", 8), ("member_capacity", True),
                             ("member_capacity", 1.), ("max8_prefill_reused", False),
                             ("max8_prefill_reused", 1), ("generic_max_abs_error", None)):
                value = receipt()
                value["cases"][index][key] = bad
                self.rejected(value)
            for key in ("member_capacity", "max8_prefill_reused", "generic_max_abs_error"):
                value = receipt()
                del value["cases"][index][key]
                self.rejected(value)
                value = receipt()
                value["cases"][index - 3][key] = value["cases"][index][key]
                self.rejected(value)

    def test_reported_error_rejects_nonfinite_negative_boolean_and_non_numeric_values(self):
        for bad in (float("nan"), float("inf"), -float("inf"), -1., True, None,
                    "0", [], 10**400, VALIDATOR.F32_MAX * 2):
            for index, key in ((0, "max_abs_error"), (3, "max_abs_error"),
                               (3, "generic_max_abs_error")):
                value = receipt()
                value["cases"][index][key] = bad
                self.rejected(value)

    def test_audits_require_exact_key_sets_integer_types_and_nonempty_full_counts(self):
        for name in ("io", "main_audit", "cache_audit"):
            for key in receipt()["cases"][0][name]:
                value = receipt()
                del value["cases"][0][name][key]
                self.rejected(value)
                for bad in (True, 1., -1, None, "0"):
                    value = receipt()
                    value["cases"][0][name][key] = bad
                    self.rejected(value)
            value = receipt()
            value["cases"][0][name]["extra"] = 0
            self.rejected(value)
            value = receipt()
            value["cases"][0][name] = []
            self.rejected(value)
        for bad in (0, VALIDATOR.IO_LIMIT + 256, 257):
            value = receipt()
            value["cases"][0]["io"]["inspected_bytes"] = bad
            self.rejected(value)

    def test_io_counts_need_all_three_kinds_and_actual_verification_copy_charge(self):
        for key in ("upload_bytes", "uniform_bytes"):
            value = receipt()
            io = value["cases"][0]["io"]
            io[key] = 0
            io["readback_bytes"] = 32768 + io["upload_bytes"] + io["uniform_bytes"]
            io["inspected_bytes"] = sum(io[k] for k in ("upload_bytes", "uniform_bytes", "readback_bytes"))
            self.rejected(value)
        value = receipt()
        io = value["cases"][0]["io"]
        io["readback_bytes"] += 256
        io["inspected_bytes"] += 256
        self.rejected(value)
        for buffers in (0, 3, 6, 8, VALIDATOR.IO_LIMIT, True):
            value = receipt()
            value["cases"][0]["io"]["inspected_buffers"] = buffers
            self.rejected(value)
        value = receipt()
        io = value["cases"][0]["io"]
        io["upload_bytes"] += 1
        io["readback_bytes"] += 1
        io["inspected_bytes"] += 2
        self.rejected(value)

    def test_main_and_cache_audits_reject_partial_full_owner_or_fabricated_io_kinds(self):
        for name in ("main_audit", "cache_audit"):
            for index in (0, 3, 13, 29, 90):
                for how in ("short", "over", "upload", "uniform", "buffers"):
                    value = receipt()
                    item = value["cases"][index][name]
                    if how in ("short", "over"):
                        item["readback_bytes"] += -256 if how == "short" else 256
                    elif how == "buffers":
                        item["inspected_buffers"] += 1
                    else:
                        item[how + "_bytes"] = 256
                    item["inspected_bytes"] = sum(item[k] for k in (
                        "upload_bytes", "uniform_bytes", "readback_bytes"))
                    self.rejected(value)

    def test_one_fixture_epoch_sequence_rejects_replays_backward_or_overflow_epochs(self):
        for index, name, bad in ((0, "io", 0), (0, "io", 2**64),
                                 (90, "cache_audit", 2**64 - 1),
                                 (0, "main_audit", 1), (0, "cache_audit", 2),
                                 (1, "io", 3), (90, "cache_audit", 1)):
            value = receipt()
            value["cases"][index][name]["epoch"] = bad
            self.rejected(value)
        value = receipt()
        value["cases"][1]["io"] = copy.deepcopy(value["cases"][0]["io"])
        self.rejected(value)

    def test_full_owner_cap_and_source_slice_bounds_are_not_receipt_selected(self):
        for columns in (65537, 65536):
            shapes = ((0, 7, columns),) + SHAPES[1:]
            self.rejected(receipt(), shapes=shapes)
        value = receipt()
        io = value["cases"][0]["io"]
        io["upload_bytes"] = VALIDATOR.IO_LIMIT
        io["readback_bytes"] = VALIDATOR.IO_LIMIT + 32768 + 256
        io["inspected_bytes"] = sum(io[k] for k in ("upload_bytes", "uniform_bytes", "readback_bytes"))
        self.rejected(value)

    def test_loader_rejects_nested_duplicate_keys_nonfinite_json_and_oversized_data(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = pathlib.Path(temporary) / "receipt.json"
            path.write_text(json.dumps(receipt()))
            self.accepted(VALIDATOR.load_receipt(path))
            for data in ('{"cases":[],"cases":[]}', '{"io":{"epoch":1,"epoch":2}}',
                         '{"max":NaN}', '{"max":Infinity}', '{"max":-Infinity}',
                         " " * (VALIDATOR.RECEIPT_MAX_BYTES + 1)):
                path.write_text(data)
                with self.assertRaises(ValueError):
                    VALIDATOR.load_receipt(path)
            path.write_text('{"max":1e400}')
            self.rejected(VALIDATOR.load_receipt(path))

    def test_cli_requires_explicit_vendor_shapes_and_emits_only_fixed_scope(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = pathlib.Path(temporary) / "receipt.json"
            path.write_text(json.dumps(receipt()))
            base = [sys.executable, str(SCRIPT), "--receipt", str(path)]
            shape_args = [part for shape in SHAPES for part in ("--shape", ":".join(map(str, shape)))]
            for args in (base, base + ["--vendor", "intel"], base + shape_args):
                self.assertNotEqual(subprocess.run(args, capture_output=True).returncode, 0)
            for vendor in ("intel", "nvidia"):
                result = subprocess.run(base + ["--vendor", vendor] + shape_args,
                                        capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn(f"vendor={vendor} cases=91 legacy_grid_cases=84 single_member_cases=7", result.stdout)
                self.assertIn("whole_model_accepted=false performance_accepted=false", result.stdout)
            value = receipt()
            value["cases"][3]["max8_prefill_reused"] = False
            path.write_text(json.dumps(value))
            result = subprocess.run(base + ["--vendor", "intel"] + shape_args,
                                    capture_output=True, text=True)
            self.assertEqual(result.returncode, 1)
            self.assertEqual(result.stderr.strip(), "Fixed-public single dense numeric receipt rejected")
            self.assertEqual(result.stdout, "")


if __name__ == "__main__":
    unittest.main()
