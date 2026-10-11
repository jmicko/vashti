"""Exact four-row admission retains original public inputs and all capacities."""
import unittest

from test_protected_target_read_schedule import CONFLICTS, FIXTURES, GUARD, REQUIRED, target_scope

SELECTOR = "KUNO_LAB_TARGET_LOOKAHEAD"


def selected_scope(fixture):
    environment, arguments, cap = target_scope(fixture, "intel")
    environment.update(KUNO_LAB_TARGET_LOOKAHEAD="four-public",
                       KUNO_LAB_FROZEN_GU_MATH="integer-public",
                       KUNO_LAB_FROZEN_DOWN_MATH="integer-public",
                       KUNO_LAB_MTP_SWEEPS="fused")
    return environment, arguments, cap


class FourLookaheadAdmission(unittest.TestCase):
    def rejected(self, environment, arguments):
        with self.assertRaises(ValueError):
            GUARD.memory_limit(environment, arguments)

    def test_original_defaults_and_selected_compositions_keep_caps_and_inputs(self):
        for fixture in FIXTURES:
            base, args, cap = target_scope(fixture, "intel")
            self.assertEqual(GUARD.memory_limit(base, args), cap)
            for stride in (False, True):
                for shared in (False, True):
                    selected, args, cap = selected_scope(fixture)
                    if stride:
                        selected["KUNO_LAB_FROZEN_GU_SCHEDULE"] = "four-public"
                    if shared:
                        selected["KUNO_LAB_MTP_Q8_HEAD"] = "shared-public"
                    selected.update(KUNO_LAB_TARGET_PLE_READS="grouped-public",
                                    KUNO_LAB_TARGET_QSA="chronological-public")
                    before = dict(selected)
                    self.assertEqual(GUARD.memory_limit(selected, args), cap)
                    self.assertNotIn(SELECTOR, GUARD.selected_target_policy(selected, mtp=True))
                    self.assertEqual(selected, before)

    def test_all_prerequisites_and_selector_are_canonical(self):
        for fixture in FIXTURES:
            environment, args, _ = selected_scope(fixture)
            for key in (*REQUIRED, "KUNO_LAB_MTP_DEVICE", "KUNO_LAB_MTP_SWEEPS",
                        "KUNO_LAB_FROZEN_GU_MATH", "KUNO_LAB_FROZEN_DOWN_MATH"):
                for value in (None, "", environment[key] + " "):
                    invalid = dict(environment)
                    if value is None:
                        del invalid[key]
                    else:
                        invalid[key] = value
                    self.rejected(invalid, args)
            for value in ("", "four", "4", "three-public", "FOUR-PUBLIC", "four-public "):
                self.rejected(dict(environment, **{SELECTOR: value}), args)

    def test_alternate_math_draft_and_repair_policies_reject(self):
        conflicts = dict(CONFLICTS, KUNO_LAB_MTP_PROFILE="strata-q2",
                         KUNO_LAB_MTP_VOCAB="english-public",
                         KUNO_LAB_MTP_Q8_HEAD="broadcast-public",
                         KUNO_LAB_MTP_REPAIR="batch-public",
                         KUNO_LAB_TARGET_GDN="three-streaming-public",
                         KUNO_LAB_TARGET_IQ4_DECODER="packed-registers-public",
                         KUNO_LAB_FROZEN_BANK_FILL="profile-ranked-public",
                         KUNO_LAB_FROZEN_DOWN_ROWS="four-public",
                         KUNO_LAB_FROZEN_DOWN_SCHEDULE="four-public",
                         KUNO_LAB_LANGUAGE_DECODE="target-only-public")
        for fixture in FIXTURES:
            environment, args, _ = selected_scope(fixture)
            for key, value in conflicts.items():
                self.rejected(dict(environment, **{key: value}), args)
            other, args, _ = target_scope(fixture, "nvidia")
            self.rejected(dict(other, **{SELECTOR: "four-public"}), args)
        for fixture in ("first-code", "warm-prose"):
            other, args, _ = target_scope(fixture, "target-only")
            self.rejected(dict(other, **{SELECTOR: "four-public"}), args)

    def test_no_generic_scope_capacity_change_or_extra_argument(self):
        for fixture in FIXTURES:
            environment, args, _ = selected_scope(fixture)
            self.rejected(environment, args + ["extra"])
            self.rejected(dict(environment, KUNO_LAB_STATIC_CACHE_BYTES="17179869184"), args)
            with self.assertRaises(ValueError):
                GUARD.selected_target_policy(environment, mtp=False)
        self.rejected({SELECTOR: "four-public"}, ["/frozen/binary", "intel", "daslab-language-probe"])

    def test_dedicated_math_operators_have_no_model_or_request_input_scope(self):
        for mode, gate in (("verify4-dense-probe", "KUNO_LAB_VERIFY4_DENSE_PUBLIC_ONLY"),
                           ("verify4-finite-probe", "KUNO_LAB_VERIFY4_FINITE_PUBLIC_ONLY")):
            environment = {gate: "1", "KUNO_PUBLIC_RECEIPT": "/public/numeric.json",
                           "KUNO_LAB_DEVICE_DISPATCH": "inline"}
            args = ["/frozen/binary", "intel", mode]
            self.assertEqual(GUARD.memory_limit(environment, args), 8 * 1024**3)
            for key in environment:
                bad = dict(environment)
                del bad[key]
                self.rejected(bad, args)
                self.rejected(dict(environment, **{key: ""}), args)
            self.rejected(environment, args + ["extra"])
            self.rejected(environment, [args[0], "nvidia", mode])
            for key, value in dict(CONFLICTS, KUNO_TEST_Q3_DIR="/model", KUNO_TEST_MTP="/draft",
                                   KUNO_LAB_TARGET_LOOKAHEAD="four-public").items():
                self.rejected(dict(environment, **{key: value}), args)


if __name__ == "__main__":
    unittest.main()
