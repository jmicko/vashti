"""Fixed public admission for the immutable English/code Q8 draft shortlist."""
import itertools
import unittest

from test_protected_frozen_members import GUARD, public_scope


class EnglishVocabularyAdmission(unittest.TestCase):
    def scope(self, fixture):
        environment, arguments, cap = public_scope(fixture)
        environment.update({
            "KUNO_LAB_TARGET_DENSE": "verify3-fused",
            "KUNO_LAB_MTP_SWEEPS": "fused",
            "KUNO_LAB_MTP_VOCAB": "english-public",
        })
        return environment, arguments, cap

    def test_fixed_compositions_keep_original_host_caps(self):
        for fixture in ("warm-cpu13", "first-code", "warm-prose"):
            base, arguments, cap = self.scope(fixture)
            for shared, members, gdn in itertools.product((False, True), repeat=3):
                environment = dict(base)
                if shared:
                    environment["KUNO_LAB_MTP_Q8_HEAD"] = "shared-public"
                if members:
                    environment["KUNO_LAB_FROZEN_MEMBERS"] = "three-public"
                if gdn:
                    environment["KUNO_LAB_TARGET_GDN"] = "three-streaming-public"
                self.assertEqual(GUARD.memory_limit(environment, arguments), cap)
                self.assertNotIn("KUNO_LAB_MTP_VOCAB", GUARD.selected_target_policy(environment, mtp=True))
            old = dict(base)
            del old["KUNO_LAB_MTP_VOCAB"]
            self.assertEqual(GUARD.memory_limit(old, arguments), cap)

    def test_required_scope_and_unsupported_profiles_are_rejected(self):
        for fixture in ("warm-cpu13", "first-code", "warm-prose"):
            base, arguments, _ = self.scope(fixture)
            keys = ("KUNO_LAB_TARGET_SOURCES", "KUNO_LAB_FROZEN_PARTITION_EXIT", "KUNO_LAB_TARGET_ROUTER", "KUNO_LAB_TARGET_REUSE", "KUNO_LAB_TARGET_FILL", "KUNO_LAB_DEVICE_DISPATCH", "KUNO_LAB_TARGET_IO", "KUNO_LAB_TARGET_DENSE", "KUNO_LAB_MTP_SWEEPS")
            for key in keys:
                for wrong in (None, "", base[key] + " "):
                    bad = dict(base)
                    if wrong is None:
                        del bad[key]
                    else:
                        bad[key] = wrong
                    with self.assertRaises(ValueError):
                        GUARD.memory_limit(bad, arguments)
            for key, value in (("KUNO_LAB_MTP_PROFILE", "strata-q2"), ("KUNO_LAB_MTP_Q8_HEAD", "broadcast-public"), ("KUNO_LAB_TARGET_MOE", "live"), ("KUNO_LAB_TARGET_EXPERTS", "prepared-verify3"), ("KUNO_LAB_MTP_VOCAB", "english-public ")):
                with self.assertRaises(ValueError):
                    GUARD.memory_limit(dict(base, **{key: value}), arguments)
            with self.assertRaises(ValueError):
                GUARD.selected_target_policy(base, mtp=False)
            with self.assertRaises(ValueError):
                GUARD.memory_limit(dict(base, KUNO_LAB_LANGUAGE_PUBLIC_ONLY="1", KUNO_LAB_SPECULATIVE_PUBLIC_ONLY="1"), arguments)


if __name__ == "__main__":
    unittest.main()
