"""Joined original cache repair keeps canonical Intel admission unchanged."""
import unittest

from test_protected_target_read_schedule import CONFLICTS, FIXTURES, GUARD, REQUIRED, target_scope

SELECTOR = "KUNO_LAB_MTP_REPAIR"


class MtpCacheRepairAdmission(unittest.TestCase):
    def rejected(self, environment, arguments):
        with self.assertRaises(ValueError):
            GUARD.memory_limit(environment, arguments)

    def test_scalar_default_and_batched_compositions_keep_caps_and_inputs(self):
        for fixture in FIXTURES:
            environment, arguments, cap = target_scope(fixture, "intel")
            self.assertEqual(GUARD.memory_limit(environment, arguments), cap)
            for down in ("original", "four", "integer"):
                selected = dict(environment)
                selected.update(KUNO_LAB_MTP_REPAIR="batch-public",
                                KUNO_LAB_MTP_Q8_HEAD="shared-public",
                                KUNO_LAB_FROZEN_GU_MATH="integer-public",
                                KUNO_LAB_FROZEN_GU_SCHEDULE="four-public",
                                KUNO_LAB_TARGET_PLE_READS="grouped-public")
                if down == "four":
                    selected["KUNO_LAB_FROZEN_DOWN_ROWS"] = "four-public"
                elif down == "integer":
                    selected["KUNO_LAB_FROZEN_DOWN_MATH"] = "integer-public"
                before = dict(selected)
                self.assertEqual(GUARD.memory_limit(selected, arguments), cap)
                self.assertNotIn(SELECTOR, GUARD.selected_target_policy(selected, mtp=True))
                self.assertEqual(selected, before)

    def test_each_original_prerequisite_and_selector_is_canonical(self):
        for fixture in FIXTURES:
            environment, arguments, _ = target_scope(fixture, "intel")
            environment[SELECTOR] = "batch-public"
            for key in (*REQUIRED, "KUNO_LAB_MTP_DEVICE", "KUNO_LAB_MTP_SWEEPS"):
                for value in (None, "", environment[key] + " "):
                    bad = dict(environment)
                    if value is None:
                        del bad[key]
                    else:
                        bad[key] = value
                    self.rejected(bad, arguments)
            for value in ("", "batch", "1", "scalar", "BATCH-PUBLIC", "batch-public "):
                self.rejected(dict(environment, **{SELECTOR: value}), arguments)

    def test_other_placements_profiles_and_conflicting_policies_reject(self):
        extra = dict(CONFLICTS, KUNO_LAB_MTP_PROFILE="strata-q2",
                     KUNO_LAB_MTP_VOCAB="english-public",
                     KUNO_LAB_MTP_Q8_HEAD="broadcast-public",
                     KUNO_LAB_FROZEN_BANK_FILL="profile-ranked-public",
                     KUNO_LAB_LANGUAGE_DECODE="target-only-public")
        for fixture in FIXTURES:
            environment, arguments, _ = target_scope(fixture, "intel")
            environment[SELECTOR] = "batch-public"
            for key, value in extra.items():
                self.rejected(dict(environment, **{key: value}), arguments)
            other, args, _ = target_scope(fixture, "nvidia")
            self.rejected(dict(other, **{SELECTOR: "batch-public"}), args)
        for fixture in ("first-code", "warm-prose"):
            other, args, _ = target_scope(fixture, "target-only")
            self.rejected(dict(other, **{SELECTOR: "batch-public"}), args)

    def test_no_generic_scope_extra_arguments_or_capacity_changes(self):
        for fixture in FIXTURES:
            environment, arguments, _ = target_scope(fixture, "intel")
            environment[SELECTOR] = "batch-public"
            self.rejected(environment, arguments + ["extra"])
            self.rejected(dict(environment, KUNO_LAB_STATIC_CACHE_BYTES="17179869184"), arguments)
            with self.assertRaises(ValueError):
                GUARD.selected_target_policy(environment, mtp=False)
        self.rejected({SELECTOR: "batch-public"}, ["/frozen/binary", "intel", "daslab-language-probe"])


if __name__ == "__main__":
    unittest.main()
