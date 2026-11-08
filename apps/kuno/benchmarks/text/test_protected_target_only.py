"""Target-only fixed language admission excludes draft engines and selectors."""

import unittest

from test_protected_frozen_members import GUARD, public_scope


def target_only(fixture):
    environment, arguments, cap = public_scope(fixture)
    environment.update(KUNO_LAB_LANGUAGE_DECODE="target-only-public", KUNO_LAB_TARGET_DENSE="verify3-fused")
    return environment, arguments, cap


class TargetOnlyAdmission(unittest.TestCase):
    def test_public_code_prose_keep_caps_and_all_target_dependencies(self):
        for fixture in ("first-code", "warm-prose"):
            base, arguments, cap = target_only(fixture)
            for integer, stride in ((False, False), (True, False), (True, True)):
                environment = dict(base)
                if integer:
                    environment["KUNO_LAB_FROZEN_GU_MATH"] = "integer-public"
                if stride:
                    environment["KUNO_LAB_FROZEN_GU_SCHEDULE"] = "four-public"
                environment.update(KUNO_LAB_FROZEN_DOWN_ROWS="four-public", KUNO_LAB_TARGET_QSA="chronological-public")
                self.assertEqual(GUARD.memory_limit(environment, arguments), cap)
                self.assertNotIn("KUNO_LAB_LANGUAGE_DECODE", GUARD.selected_target_policy(environment, mtp=True))
                self.assertEqual(environment["KUNO_LAB_LANGUAGE_DECODE"], "target-only-public")
            for key in ("KUNO_LAB_LANGUAGE_PUBLIC_ONLY", "KUNO_LAB_TARGET_DENSE", "KUNO_LAB_TARGET_SOURCES", "KUNO_LAB_FROZEN_PARTITION_EXIT", "KUNO_LAB_TARGET_ROUTER", "KUNO_LAB_TARGET_REUSE", "KUNO_LAB_TARGET_FILL", "KUNO_LAB_DEVICE_DISPATCH", "KUNO_LAB_TARGET_IO"):
                for value in (None, "", base[key] + " "):
                    broken = dict(base)
                    if value is None:
                        del broken[key]
                    else:
                        broken[key] = value
                    with self.assertRaises(ValueError):
                        GUARD.memory_limit(broken, arguments)

    def test_no_draft_selector_metadata_private_content_or_experimental_scope(self):
        conflicts = {
            "KUNO_LAB_MTP_SWEEPS": "fused", "KUNO_LAB_MTP_PROFILE": "strata-q2",
            "KUNO_LAB_MTP_Q8_HEAD": "shared-public", "KUNO_LAB_MTP_VOCAB": "english-public",
            "KUNO_LAB_MTP_DEVICE": "intel-public", "KUNO_PUBLIC_MTP_INTEL_TOTAL_BYTES": "34242297856",
            "KUNO_PUBLIC_MTP_INTEL_USED_BYTES": "50688000", "KUNO_PUBLIC_MTP_INTEL_DEVICE_ID": "57891",
            "KUNO_LAB_SPECULATIVE_PUBLIC_ONLY": "1", "KUNO_LAB_WARM_REPLAY": "once",
            "KUNO_LAB_TARGET_GDN": "three-streaming-public", "KUNO_LAB_FROZEN_TABLES": "private-public",
            "KUNO_LAB_FROZEN_DOWN_SCALES": "shared-public", "KUNO_LAB_FROZEN_GU_FACTORS": "shared-public",
            "KUNO_LAB_FROZEN_GU_ROWS": "two-public", "KUNO_LAB_FROZEN_HOST_STAGE": "packed-public",
            "KUNO_LAB_FROZEN_MEMBERS": "three-public", "KUNO_LAB_TARGET_EXPERTS": "prepared-verify3",
            "KUNO_LAB_TARGET_MOE": "live", "KUNO_LAB_PROMPT": "private", "KUNO_TEST_UNKNOWN": "1",
        }
        for fixture in ("first-code", "warm-prose"):
            base, arguments, _ = target_only(fixture)
            for key, value in conflicts.items():
                with self.assertRaises(ValueError):
                    GUARD.memory_limit(dict(base, **{key: value}), arguments)
            for key in ("KUNO_LAB_MTP_SWEEPS", "KUNO_LAB_MTP_DEVICE"):
                with self.assertRaises(ValueError):
                    GUARD.memory_limit(dict(base, **{key: ""}), arguments)

    def test_selector_cannot_escape_fixed_intel_language_or_canonical_mode(self):
        for fixture in ("first-code", "warm-prose"):
            base, arguments, _ = target_only(fixture)
            for value in ("", "target-only", "speculative", "target-only-public "):
                with self.assertRaises(ValueError):
                    GUARD.memory_limit(dict(base, KUNO_LAB_LANGUAGE_DECODE=value), arguments)
            for wrong in ([arguments[0], "nvidia", arguments[2]], [arguments[0], "intel", "daslab-speculative-probe"], arguments + ["extra"]):
                with self.assertRaises(ValueError):
                    GUARD.memory_limit(base, wrong)
            with self.assertRaises(ValueError):
                GUARD.selected_target_policy(base, mtp=False)
        base, arguments, _ = public_scope("warm-cpu13")
        with self.assertRaises(ValueError):
            GUARD.memory_limit(dict(base, KUNO_LAB_LANGUAGE_DECODE="target-only-public"), arguments)


if __name__ == "__main__":
    unittest.main()
