"""First-warm target profiling keeps the original CPU13 process cap and inputs."""
import itertools
import unittest

from test_protected_target_integer_head import HEAD_REQUIRED, head_scope
from test_protected_target_read_schedule import CONFLICTS, FIXTURES, GUARD, REQUIRED, target_scope

SELECTOR = "KUNO_LAB_TARGET_PROFILE"


def profile_scope():
    environment, arguments, cap = head_scope("warm-cpu13")
    del environment["KUNO_LAB_TARGET_HEAD"]
    environment[SELECTOR] = "first-warm-chunk"
    return environment, arguments, cap


class TargetProfileAdmission(unittest.TestCase):
    def rejected(self, environment, arguments):
        with self.assertRaises(ValueError):
            GUARD.memory_limit(environment, arguments)

    def test_default_and_three_four_head_repair_compositions_keep_48gib_and_input(self):
        for fixture in FIXTURES:
            base, arguments, cap = target_scope(fixture, "intel")
            self.assertEqual(GUARD.memory_limit(base, arguments), cap)
        for four, head, repair, shared in itertools.product((False, True), repeat=4):
            environment, arguments, cap = profile_scope()
            self.assertEqual(cap, 48 * 1024**3)
            if four:
                environment["KUNO_LAB_TARGET_LOOKAHEAD"] = "four-public"
            if head:
                environment["KUNO_LAB_TARGET_HEAD"] = "integer-public"
            if repair:
                environment["KUNO_LAB_MTP_REPAIR"] = "batch-public"
            if shared:
                environment.update(KUNO_LAB_MTP_Q8_HEAD="shared-public",
                                   KUNO_LAB_FROZEN_GU_SCHEDULE="four-public",
                                   KUNO_LAB_TARGET_QSA="chronological-public",
                                   KUNO_LAB_TARGET_STATUS_FOLDS="grouped-public",
                                   KUNO_LAB_TARGET_SUBMISSION_BACKLOG="queued-2048-public")
            before = dict(environment)
            self.assertEqual(GUARD.memory_limit(environment, arguments), cap)
            self.assertNotIn(SELECTOR, GUARD.selected_target_policy(environment, mtp=True))
            self.assertEqual(environment, before)
            del environment[SELECTOR]
            self.assertEqual(GUARD.memory_limit(environment, arguments), cap)

    def test_missing_noncanonical_dependencies_and_unknown_selector_reject(self):
        environment, arguments, _ = profile_scope()
        for key in (*REQUIRED, *HEAD_REQUIRED, "KUNO_LAB_WARM_REPLAY",
                    "KUNO_LAB_SPECULATIVE_PUBLIC_ONLY", "KUNO_LAB_STATIC_CACHE_BYTES"):
            for value in (None, "", environment[key] + " "):
                invalid = dict(environment)
                if value is None:
                    del invalid[key]
                else:
                    invalid[key] = value
                self.rejected(invalid, arguments)
        for value in ("", "first", "first-warm", "first-warm-chunk ", "FIRST-WARM-CHUNK", "1"):
            self.rejected(dict(environment, **{SELECTOR: value}), arguments)
        for value in ("original", "integer", "", "integer-public "):
            self.rejected(dict(environment, KUNO_LAB_TARGET_HEAD=value), arguments)

    def test_language_other_modes_devices_or_nonmtp_policy_cannot_enable_profile(self):
        environment, arguments, _ = profile_scope()
        for fixture in ("first-code", "warm-prose"):
            language, args, _ = head_scope(fixture)
            self.rejected(dict(language, **{SELECTOR: "first-warm-chunk"}), args)
        for wrong in (arguments + ["extra"], arguments[:2],
                      [arguments[0], "nvidia", arguments[2]],
                      [arguments[0], "intel", "daslab-language-probe"],
                      [arguments[0], "intel", "resident-mtp-probe"],
                      [arguments[0], "intel", "verify4-dense-probe"]):
            self.rejected(environment, wrong)
        self.rejected(dict(environment, KUNO_LAB_LANGUAGE_PUBLIC_ONLY="1"), arguments)
        self.rejected(dict(environment, KUNO_LAB_LANGUAGE_FIXTURE="first-code"), arguments)
        self.rejected({SELECTOR: "first-warm-chunk"}, arguments)
        with self.assertRaises(ValueError):
            GUARD.selected_target_policy(environment, mtp=False)

    def test_all_existing_integer_scope_exclusions_and_caps_remain(self):
        environment, arguments, _ = profile_scope()
        conflicts = dict(CONFLICTS, KUNO_LAB_MTP_DEVICE="nvidia-public",
                         KUNO_LAB_MTP_PROFILE="strata-q2", KUNO_LAB_MTP_VOCAB="english-public",
                         KUNO_LAB_MTP_Q8_HEAD="broadcast-public",
                         KUNO_LAB_TARGET_IQ4_DECODER="packed-registers-public",
                         KUNO_LAB_FROZEN_BANK_FILL="profile-ranked-public",
                         KUNO_LAB_FROZEN_DOWN_ROWS="four-public",
                         KUNO_LAB_FROZEN_DOWN_SCHEDULE="four-public",
                         KUNO_LAB_LANGUAGE_DECODE="target-only-public",
                         KUNO_LAB_MTP_REPAIR="batch", KUNO_LAB_TARGET_LOOKAHEAD="four",
                         KUNO_LAB_STATIC_CACHE_BYTES="17179869184")
        for key, value in conflicts.items():
            self.rejected(dict(environment, **{key: value}), arguments)
        three = dict(environment, KUNO_LAB_TARGET_GDN="three-streaming-public")
        self.assertEqual(GUARD.memory_limit(three, arguments), 48 * 1024**3)
        self.rejected(dict(three, KUNO_LAB_TARGET_LOOKAHEAD="four-public"), arguments)


if __name__ == "__main__":
    unittest.main()
