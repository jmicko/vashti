"""Fixed Intel head admission preserves public fixtures and existing budgets."""
import itertools
import unittest

from test_protected_target_read_schedule import CONFLICTS, FIXTURES, GUARD, REQUIRED, target_scope

SELECTOR = "KUNO_LAB_TARGET_HEAD"
HEAD_REQUIRED = {
    "KUNO_LAB_MTP_DEVICE": "intel-public",
    "KUNO_LAB_MTP_SWEEPS": "fused",
    "KUNO_LAB_FROZEN_GU_MATH": "integer-public",
    "KUNO_LAB_FROZEN_DOWN_MATH": "integer-public",
    "KUNO_LAB_TARGET_PLE_READS": "grouped-public",
}


def head_scope(fixture):
    environment, arguments, cap = target_scope(fixture, "intel")
    environment.update(HEAD_REQUIRED)
    environment[SELECTOR] = "integer-public"
    return environment, arguments, cap


class IntegerHeadAdmission(unittest.TestCase):
    def rejected(self, environment, arguments):
        with self.assertRaises(ValueError):
            GUARD.memory_limit(environment, arguments)

    def test_default_and_three_four_repair_compositions_keep_inputs_and_caps(self):
        for fixture, four, repair, shared in itertools.product(FIXTURES, (False, True), (False, True), (False, True)):
            selected, args, cap = head_scope(fixture)
            selected["KUNO_LAB_FROZEN_GU_SCHEDULE"] = "four-public"
            selected["KUNO_LAB_TARGET_QSA"] = "chronological-public"
            if four:
                selected["KUNO_LAB_TARGET_LOOKAHEAD"] = "four-public"
            if repair:
                selected["KUNO_LAB_MTP_REPAIR"] = "batch-public"
            if shared:
                selected["KUNO_LAB_MTP_Q8_HEAD"] = "shared-public"
            before = dict(selected)
            self.assertEqual(GUARD.memory_limit(selected, args), cap)
            self.assertNotIn(SELECTOR, GUARD.selected_target_policy(selected, mtp=True))
            self.assertEqual(selected, before)
            del selected[SELECTOR]
            self.assertEqual(GUARD.memory_limit(selected, args), cap)

    def test_missing_or_noncanonical_dependencies_reject(self):
        for fixture in FIXTURES:
            selected, args, _ = head_scope(fixture)
            for key in (*REQUIRED, *HEAD_REQUIRED, SELECTOR):
                for value in (None, "", selected[key] + " "):
                    invalid = dict(selected)
                    if value is None:
                        if key == SELECTOR:
                            continue
                        del invalid[key]
                    else:
                        invalid[key] = value
                    self.rejected(invalid, args)
            for value in ("integer", "original", "INTEGER-PUBLIC"):
                self.rejected(dict(selected, **{SELECTOR: value}), args)

    def test_conflicting_models_math_and_unscoped_inputs_reject(self):
        conflicts = dict(CONFLICTS, KUNO_LAB_MTP_DEVICE="nvidia-public",
                         KUNO_LAB_MTP_PROFILE="strata-q2", KUNO_LAB_MTP_VOCAB="english-public",
                         KUNO_LAB_MTP_Q8_HEAD="broadcast-public",
                         KUNO_LAB_TARGET_IQ4_DECODER="packed-registers-public",
                         KUNO_LAB_FROZEN_BANK_FILL="profile-ranked-public",
                         KUNO_LAB_FROZEN_DOWN_ROWS="four-public",
                         KUNO_LAB_FROZEN_DOWN_SCHEDULE="four-public",
                         KUNO_LAB_LANGUAGE_DECODE="target-only-public",
                         KUNO_LAB_MTP_REPAIR="batch", KUNO_LAB_TARGET_LOOKAHEAD="four")
        for fixture in FIXTURES:
            selected, args, _ = head_scope(fixture)
            for key, value in conflicts.items():
                self.rejected(dict(selected, **{key: value}), args)
            self.rejected(selected, args + ["extra"])
            self.rejected(dict(selected, KUNO_LAB_STATIC_CACHE_BYTES="17179869184"), args)
            with self.assertRaises(ValueError):
                GUARD.selected_target_policy(selected, mtp=False)
        self.rejected({SELECTOR: "integer-public"}, ["/frozen/binary", "intel", "probe"])

    def test_existing_gdn_status_backlog_scopes_are_preserved(self):
        for fixture in FIXTURES:
            selected, args, cap = head_scope(fixture)
            selected.update(KUNO_LAB_TARGET_GDN="three-streaming-public",
                            KUNO_LAB_TARGET_STATUS_FOLDS="grouped-public",
                            KUNO_LAB_TARGET_SUBMISSION_BACKLOG="queued-2048-public")
            self.assertEqual(GUARD.memory_limit(selected, args), cap)
            self.rejected(dict(selected, KUNO_LAB_TARGET_LOOKAHEAD="four-public"), args)


if __name__ == "__main__":
    unittest.main()
