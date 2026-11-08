"""Fixed Intel IQ4 decoder scope preserves the original public resource caps."""

import unittest

from test_protected_target_read_schedule import CONFLICTS, FIXTURES, GUARD, REQUIRED, target_scope

SELECTOR = "KUNO_LAB_TARGET_IQ4_DECODER"


class Iq4RegisterAdmission(unittest.TestCase):
    def rejected(self, environment, arguments):
        with self.assertRaises(ValueError):
            GUARD.memory_limit(environment, arguments)

    def test_original_and_selected_compositions_keep_exact_caps(self):
        for fixture in FIXTURES:
            environment, arguments, cap = target_scope(fixture, "intel")
            self.assertEqual(GUARD.memory_limit(environment, arguments), cap)
            environment[SELECTOR] = "packed-registers-public"
            for down in ("plain", "four", "integer"):
                selected = dict(environment)
                selected.update(KUNO_LAB_FROZEN_GU_MATH="integer-public", KUNO_LAB_FROZEN_GU_SCHEDULE="four-public", KUNO_LAB_TARGET_PLE_READS="grouped-public", KUNO_LAB_TARGET_STATUS_FOLDS="grouped-public", KUNO_LAB_MTP_Q8_HEAD="shared-public")
                if down == "four":
                    selected["KUNO_LAB_FROZEN_DOWN_ROWS"] = "four-public"
                elif down == "integer":
                    selected["KUNO_LAB_FROZEN_DOWN_MATH"] = "integer-public"
                before = dict(selected)
                self.assertEqual(GUARD.memory_limit(selected, arguments), cap)
                self.assertNotIn(SELECTOR, GUARD.selected_target_policy(selected, mtp=True))
                self.assertEqual(selected, before)

    def test_each_canonical_prerequisite_and_selector_is_required(self):
        for fixture in FIXTURES:
            environment, arguments, _ = target_scope(fixture, "intel")
            environment[SELECTOR] = "packed-registers-public"
            for key in (*REQUIRED, "KUNO_LAB_MTP_DEVICE", "KUNO_LAB_MTP_SWEEPS"):
                for value in (None, "", environment[key] + " "):
                    bad = dict(environment)
                    if value is None:
                        del bad[key]
                    else:
                        bad[key] = value
                    self.rejected(bad, arguments)
            for value in ("", "registers", "true", "PACKED-REGISTERS-PUBLIC", "packed-registers-public "):
                self.rejected(dict(environment, **{SELECTOR: value}), arguments)

    def test_other_placements_profiles_and_old_conflicting_policies_reject(self):
        for fixture in FIXTURES:
            environment, arguments, _ = target_scope(fixture, "intel")
            environment[SELECTOR] = "packed-registers-public"
            for key, value in dict(CONFLICTS, KUNO_LAB_MTP_PROFILE="strata-q2", KUNO_LAB_MTP_VOCAB="english-public", KUNO_LAB_MTP_Q8_HEAD="broadcast-public", KUNO_LAB_FROZEN_DOWN_SCHEDULE="four-public").items():
                self.rejected(dict(environment, **{key: value}), arguments)
            bad, args, _ = target_scope(fixture, "nvidia")
            self.rejected(dict(bad, **{SELECTOR: "packed-registers-public"}), args)
        for fixture in ("first-code", "warm-prose"):
            bad, args, _ = target_scope(fixture, "target-only")
            self.rejected(dict(bad, **{SELECTOR: "packed-registers-public"}), args)

    def test_no_generic_scope_extra_args_or_larger_caps(self):
        for fixture in FIXTURES:
            environment, arguments, _ = target_scope(fixture, "intel")
            environment[SELECTOR] = "packed-registers-public"
            self.rejected(environment, arguments + ["extra"])
            self.rejected(dict(environment, KUNO_LAB_STATIC_CACHE_BYTES="17179869184"), arguments)
            with self.assertRaises(ValueError):
                GUARD.selected_target_policy(environment, mtp=False)
        self.rejected({SELECTOR: "packed-registers-public"}, ["/frozen/binary", "intel", "resident-count1-hybrid-chat"])

    def test_exact_intel_compiled_c_values_only(self):
        args = ["/frozen/binary", "intel", "iq-quant-probe"]
        environment = {"KUNO_PUBLIC_RECEIPT": "/fixed/public/receipt.json"}
        self.assertEqual(GUARD.memory_limit(environment, args), 8 * 1024**3)
        for bad in ({}, {"KUNO_PUBLIC_RECEIPT": ""}, dict(environment, KUNO_TEST_GGUF="/arbitrary/model"), dict(environment, **{SELECTOR: "packed-registers-public"})):
            self.rejected(bad, args)
        self.rejected(environment, args + ["extra"])
        self.rejected(environment, [args[0], "nvidia", args[2]])


if __name__ == "__main__":
    unittest.main()
