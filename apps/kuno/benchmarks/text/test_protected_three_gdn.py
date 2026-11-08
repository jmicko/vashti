"""Admission for explicit original-arithmetic public three-token GDN."""
import itertools
import unittest

from test_protected_frozen_members import GUARD, public_scope


class ThreeGdnAdmission(unittest.TestCase):
    FIXTURES = ("warm-cpu13", "first-code", "warm-prose")

    def scope(self, fixture):
        environment, arguments, cap = public_scope(fixture)
        environment["KUNO_LAB_TARGET_DENSE"] = "verify3-fused"
        environment["KUNO_LAB_TARGET_GDN"] = "three-streaming-public"
        return environment, arguments, cap

    def rejected(self, environment, arguments):
        with self.assertRaises(ValueError):
            GUARD.memory_limit(environment, arguments)

    def test_fixed_public_compositions_preserve_caps_and_cold_defaults(self):
        for fixture in self.FIXTURES:
            base, arguments, cap = self.scope(fixture)
            for members, four, stage, profile in itertools.product(
                (False, True), (False, True), (False, True), (None, "strata-q2")
            ):
                with self.subTest(fixture=fixture, members=members, four=four, stage=stage, profile=profile):
                    environment = dict(base)
                    if members:
                        environment["KUNO_LAB_FROZEN_MEMBERS"] = "three-public"
                    if four:
                        environment["KUNO_LAB_FROZEN_DOWN_ROWS"] = "four-public"
                    if stage:
                        environment["KUNO_LAB_FROZEN_HOST_STAGE"] = "packed-public"
                    if profile:
                        environment["KUNO_LAB_MTP_PROFILE"] = profile
                    else:
                        environment["KUNO_LAB_MTP_Q8_HEAD"] = "shared-public"
                    self.assertEqual(GUARD.memory_limit(environment, arguments), cap)
                    selected = GUARD.selected_target_policy(environment, mtp=True)
                    self.assertNotIn("KUNO_LAB_TARGET_GDN", selected)
                    self.assertEqual(environment["KUNO_LAB_TARGET_GDN"], "three-streaming-public")
            old = dict(base)
            del old["KUNO_LAB_TARGET_GDN"]
            self.assertEqual(GUARD.memory_limit(old, arguments), cap)

    def test_every_required_scope_value_must_be_present_and_canonical(self):
        required = {
            "KUNO_LAB_TARGET_SOURCES": "frozen-host-public",
            "KUNO_LAB_FROZEN_PARTITION_EXIT": "1",
            "KUNO_LAB_TARGET_ROUTER": "device-frozen-public",
            "KUNO_LAB_TARGET_REUSE": "request",
            "KUNO_LAB_TARGET_FILL": "direct-public",
            "KUNO_LAB_DEVICE_DISPATCH": "inline",
            "KUNO_LAB_TARGET_IO": "chunk",
            "KUNO_LAB_TARGET_DENSE": "verify3-fused",
        }
        for fixture in self.FIXTURES:
            environment, arguments, _ = self.scope(fixture)
            for key, value in required.items():
                missing = dict(environment)
                del missing[key]
                self.rejected(missing, arguments)
                for wrong in ("", value + " ", value.upper(), "unknown"):
                    if wrong == value:
                        continue
                    with self.subTest(fixture=fixture, key=key, wrong=wrong):
                        self.rejected(dict(environment, **{key: wrong}), arguments)
            for wrong in ("", "three-streaming", "1", "three-streaming-public "):
                self.rejected(dict(environment, KUNO_LAB_TARGET_GDN=wrong), arguments)
            self.rejected(dict(environment, KUNO_LAB_TARGET_DENSE="verify3"), arguments)
            if fixture == "warm-cpu13":
                missing = dict(environment)
                del missing["KUNO_LAB_WARM_REPLAY"]
                self.rejected(missing, arguments)
                for wrong in ("", "once ", "twice"):
                    self.rejected(dict(environment, KUNO_LAB_WARM_REPLAY=wrong), arguments)

    def test_other_target_paths_and_wrong_commands_cannot_borrow_admission(self):
        for fixture in self.FIXTURES:
            environment, arguments, _ = self.scope(fixture)
            for key, value in (("KUNO_LAB_TARGET_MOE", "live"), ("KUNO_LAB_TARGET_EXPERTS", "prepared-verify3")):
                self.rejected(dict(environment, **{key: value}), arguments)
            for bad_args in ([arguments[0], "nvidia", arguments[2]], [arguments[0], "intel", "native-model-probe"], arguments + ["extra"]):
                self.rejected(environment, bad_args)
            self.rejected(dict(environment, KUNO_LAB_LANGUAGE_PUBLIC_ONLY="1", KUNO_LAB_SPECULATIVE_PUBLIC_ONLY="1"), arguments)
            private = {k: v for k, v in environment.items() if k not in ("KUNO_LAB_LANGUAGE_PUBLIC_ONLY", "KUNO_LAB_SPECULATIVE_PUBLIC_ONLY")}
            self.rejected(private, arguments)
            with self.assertRaises(ValueError):
                GUARD.selected_target_policy(environment, mtp=False)


if __name__ == "__main__":
    unittest.main()
