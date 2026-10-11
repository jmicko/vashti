"""Bounded public admission for the explicit two-Q8-plane frozen GU trial."""

import itertools
import unittest

from test_protected_frozen_members import GUARD, public_scope


class FrozenGuIntegerAdmission(unittest.TestCase):
    def scope(self, fixture):
        environment, arguments, cap = public_scope(fixture)
        environment.update({
            "KUNO_LAB_FROZEN_GU_MATH": "integer-public",
            "KUNO_LAB_TARGET_DENSE": "verify3-fused",
            "KUNO_LAB_MTP_SWEEPS": "fused",
        })
        return environment, arguments, cap

    def test_fixed_compositions_preserve_host_caps_and_original_defaults(self):
        for fixture in ("warm-cpu13", "first-code", "warm-prose"):
            base, arguments, cap = self.scope(fixture)
            for four, shared, qsa, gdn in itertools.product((False, True), repeat=4):
                environment = dict(base)
                for enabled, key, value in (
                    (four, "KUNO_LAB_FROZEN_DOWN_ROWS", "four-public"),
                    (shared, "KUNO_LAB_MTP_Q8_HEAD", "shared-public"),
                    (qsa, "KUNO_LAB_TARGET_QSA", "chronological-public"),
                    (gdn, "KUNO_LAB_TARGET_GDN", "three-streaming-public"),
                ):
                    if enabled:
                        environment[key] = value
                self.assertEqual(GUARD.memory_limit(environment, arguments), cap)
                self.assertNotIn("KUNO_LAB_FROZEN_GU_MATH", GUARD.selected_target_policy(environment, mtp=True))
                self.assertEqual(environment["KUNO_LAB_FROZEN_GU_MATH"], "integer-public")
            del base["KUNO_LAB_FROZEN_GU_MATH"]
            self.assertEqual(GUARD.memory_limit(base, arguments), cap)

    def test_every_required_policy_is_present_and_canonical(self):
        required = ("KUNO_LAB_FROZEN_GU_MATH", "KUNO_LAB_TARGET_SOURCES", "KUNO_LAB_FROZEN_PARTITION_EXIT", "KUNO_LAB_TARGET_ROUTER", "KUNO_LAB_TARGET_REUSE", "KUNO_LAB_TARGET_FILL", "KUNO_LAB_DEVICE_DISPATCH", "KUNO_LAB_TARGET_IO", "KUNO_LAB_TARGET_DENSE", "KUNO_LAB_MTP_SWEEPS")
        for fixture in ("warm-cpu13", "first-code", "warm-prose"):
            base, arguments, _ = self.scope(fixture)
            for key in required:
                for wrong in ("", base[key] + " "):
                    with self.assertRaises(ValueError):
                        GUARD.memory_limit(dict(base, **{key: wrong}), arguments)
                if key != "KUNO_LAB_FROZEN_GU_MATH":
                    missing = dict(base)
                    del missing[key]
                    with self.assertRaises(ValueError):
                        GUARD.memory_limit(missing, arguments)
            if fixture == "warm-cpu13":
                missing = dict(base)
                del missing["KUNO_LAB_WARM_REPLAY"]
                with self.assertRaises(ValueError):
                    GUARD.memory_limit(missing, arguments)

    def test_conflicting_math_profiles_and_nonpublic_modes_are_rejected(self):
        conflicts = (
            ("KUNO_LAB_FROZEN_TABLES", "private-public"),
            ("KUNO_LAB_FROZEN_TABLES", "workgroup-public"),
            ("KUNO_LAB_FROZEN_DOWN_SCALES", "shared-public"),
            ("KUNO_LAB_FROZEN_GU_FACTORS", "shared-public"),
            ("KUNO_LAB_FROZEN_GU_ROWS", "two-public"),
            ("KUNO_LAB_FROZEN_HOST_STAGE", "packed-public"),
            ("KUNO_LAB_FROZEN_MEMBERS", "three-public"),
            ("KUNO_LAB_FROZEN_BANK_PLACEMENT", "preserve-public"),
            ("KUNO_LAB_TARGET_EXPERTS", "prepared-verify3"),
            ("KUNO_LAB_TARGET_MOE", "live"),
            ("KUNO_LAB_MTP_PROFILE", "strata-q2"),
            ("KUNO_LAB_MTP_VOCAB", "english-public"),
            ("KUNO_LAB_MTP_Q8_HEAD", "broadcast-public"),
            ("KUNO_LAB_PROMPT", "private"),
        )
        for fixture in ("warm-cpu13", "first-code", "warm-prose"):
            base, arguments, _ = self.scope(fixture)
            for key, value in conflicts:
                with self.assertRaises(ValueError):
                    GUARD.memory_limit(dict(base, **{key: value}), arguments)
            with self.assertRaises(ValueError):
                GUARD.selected_target_policy(base, mtp=False)
            with self.assertRaises(ValueError):
                GUARD.memory_limit(dict(base, KUNO_LAB_SPECULATIVE_PUBLIC_ONLY="1", KUNO_LAB_LANGUAGE_PUBLIC_ONLY="1"), arguments)
            for wrong in ([arguments[0], "nvidia", arguments[2]], [arguments[0], "intel", "batch-router-gpu-probe"], [*arguments, "extra"]):
                with self.assertRaises(ValueError):
                    GUARD.memory_limit(base, wrong)


if __name__ == "__main__":
    unittest.main()
