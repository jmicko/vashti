"""Admission of the internal fixed-public three-member arithmetic experiment."""

import importlib.util
import itertools
import pathlib
import unittest


SCRIPT = pathlib.Path(__file__).with_name("protected-lab-exec.py")
SPEC = importlib.util.spec_from_file_location("protected_frozen_members", SCRIPT)
GUARD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(GUARD)


def public_scope(fixture):
    environment = {
        "KUNO_LAB_STATIC_CACHE_BYTES": "8589934592",
        "KUNO_TEST_Q3_DIR": "/original",
        "KUNO_TEST_MTP": "/original/draft",
        "KUNO_PUBLIC_RECEIPT": "/public/numeric",
        "KUNO_LAB_DEVICE_DISPATCH": "inline",
        "KUNO_LAB_TARGET_IO": "chunk",
        "KUNO_LAB_TARGET_REUSE": "request",
        "KUNO_LAB_TARGET_FILL": "direct-public",
        "KUNO_LAB_TARGET_SOURCES": "frozen-host-public",
        "KUNO_LAB_FROZEN_PARTITION_EXIT": "1",
        "KUNO_LAB_TARGET_ROUTER": "device-frozen-public",
    }
    if fixture == "warm-cpu13":
        environment.update(
            KUNO_LAB_SPECULATIVE_PUBLIC_ONLY="1",
            KUNO_LAB_WARM_REPLAY="once",
            KUNO_PUBLIC_TARGET_CARRY13_REFERENCE="/public/reference",
        )
        mode = "daslab-speculative-probe"
        memory = 48 * 1024**3
    else:
        environment.update(
            KUNO_LAB_LANGUAGE_PUBLIC_ONLY="1",
            KUNO_LAB_LANGUAGE_FIXTURE=fixture,
        )
        mode = "daslab-language-probe"
        memory = 48 * 1024**3 + 196 * 1024**2
    return environment, ["/frozen/binary", "intel", mode], memory


class FrozenMembersAdmission(unittest.TestCase):
    FIXTURES = ("warm-cpu13", "warm-prose", "first-code")

    def rejected(self, environment, arguments):
        with self.assertRaises(ValueError):
            GUARD.memory_limit(environment, arguments)

    def test_all_fixed_scopes_compositions_keep_the_original_memory_limit(self):
        for fixture in self.FIXTURES:
            base, arguments, memory = public_scope(fixture)
            self.assertEqual(GUARD.memory_limit(base, arguments), memory)
            for four, staged, head in itertools.product(
                (False, True), (False, True), (None, "shared-public", "broadcast-public")
            ):
                with self.subTest(fixture=fixture, four=four, staged=staged, head=head):
                    environment = dict(base, KUNO_LAB_FROZEN_MEMBERS="three-public")
                    if four:
                        environment["KUNO_LAB_FROZEN_DOWN_ROWS"] = "four-public"
                    if staged:
                        environment["KUNO_LAB_FROZEN_HOST_STAGE"] = "packed-public"
                    if head:
                        environment["KUNO_LAB_MTP_Q8_HEAD"] = head
                    self.assertEqual(GUARD.memory_limit(environment, arguments), memory)
                    selected = GUARD.selected_target_policy(environment, mtp=True)
                    self.assertNotIn("KUNO_LAB_FROZEN_MEMBERS", selected)
                    self.assertEqual(environment["KUNO_LAB_FROZEN_MEMBERS"], "three-public")
            # The target specialization does not select or enlarge a draft profile.
            q2 = dict(base, KUNO_LAB_FROZEN_MEMBERS="three-public", KUNO_LAB_MTP_PROFILE="strata-q2")
            self.assertEqual(GUARD.memory_limit(q2, arguments), memory)
            self.rejected(dict(q2, KUNO_LAB_MTP_Q8_HEAD="shared-public"), arguments)

    def test_every_required_policy_is_present_and_canonical(self):
        corrupt = {
            "KUNO_LAB_TARGET_SOURCES": ("", "frozen-host", "FROZEN-HOST-PUBLIC"),
            "KUNO_LAB_FROZEN_PARTITION_EXIT": ("", "0", "true", "1 "),
            "KUNO_LAB_TARGET_ROUTER": ("", "device-warm-public", "device-frozen-public "),
            "KUNO_LAB_TARGET_REUSE": ("", "1", "request "),
            "KUNO_LAB_TARGET_FILL": ("", "direct", "direct-public "),
            "KUNO_LAB_DEVICE_DISPATCH": ("", "queued", "inline "),
            "KUNO_LAB_TARGET_IO": ("", "scalar", "chunk "),
        }
        for fixture in self.FIXTURES:
            base, arguments, _ = public_scope(fixture)
            environment = dict(base, KUNO_LAB_FROZEN_MEMBERS="three-public")
            for key, wrong_values in corrupt.items():
                with self.subTest(fixture=fixture, missing=key):
                    missing = dict(environment)
                    del missing[key]
                    self.rejected(missing, arguments)
                for wrong in wrong_values:
                    with self.subTest(fixture=fixture, key=key, wrong=wrong):
                        self.rejected(dict(environment, **{key: wrong}), arguments)
            for raw in ("", "true", "three", "3", "three-public ", "THREE-PUBLIC"):
                with self.subTest(fixture=fixture, members=raw):
                    self.rejected(dict(environment, KUNO_LAB_FROZEN_MEMBERS=raw), arguments)

    def test_incompatible_arithmetic_and_private_or_unknown_flags_are_rejected(self):
        incompatible = (
            ("KUNO_LAB_FROZEN_TABLES", "private-public"),
            ("KUNO_LAB_FROZEN_TABLES", "workgroup-public"),
            ("KUNO_LAB_FROZEN_GU_FACTORS", "shared-public"),
            ("KUNO_LAB_FROZEN_DOWN_SCALES", "shared-public"),
            ("KUNO_LAB_FROZEN_GU_ROWS", "two-public"),
            ("KUNO_LAB_TARGET_EXPERTS", "prepared-verify3"),
            ("KUNO_LAB_TARGET_MOE", "live"),
            ("KUNO_LAB_PROMPT", "private"),
            ("KUNO_LAB_OTHER", "1"),
            ("KUNO_TEST_OTHER", "1"),
            ("KUNO_PUBLIC_OTHER", "/public/other"),
        )
        for fixture in self.FIXTURES:
            base, arguments, _ = public_scope(fixture)
            for staged in (False, True):
                environment = dict(base, KUNO_LAB_FROZEN_MEMBERS="three-public")
                if staged:
                    environment["KUNO_LAB_FROZEN_HOST_STAGE"] = "packed-public"
                for key, value in incompatible:
                    with self.subTest(fixture=fixture, staged=staged, key=key, value=value):
                        self.rejected(dict(environment, **{key: value}), arguments)
            for key, raw in (
                ("KUNO_LAB_FROZEN_DOWN_ROWS", "four-public "),
                ("KUNO_LAB_FROZEN_HOST_STAGE", "packed-public "),
                ("KUNO_LAB_MTP_Q8_HEAD", "shared-public "),
            ):
                self.rejected(dict(base, KUNO_LAB_FROZEN_MEMBERS="three-public", **{key: raw}), arguments)

    def test_warm_replay_scope_is_required_only_for_the_speculative_fixture(self):
        base, arguments, _ = public_scope("warm-cpu13")
        environment = dict(base, KUNO_LAB_FROZEN_MEMBERS="three-public")
        for raw in (None, "", "1", "once "):
            wrong = dict(environment)
            wrong.pop("KUNO_LAB_WARM_REPLAY")
            if raw is not None:
                wrong["KUNO_LAB_WARM_REPLAY"] = raw
            self.rejected(wrong, arguments)
        self.rejected(dict(environment, KUNO_LAB_LANGUAGE_PUBLIC_ONLY="1"), arguments)
        for fixture in ("warm-prose", "first-code"):
            base, arguments, _ = public_scope(fixture)
            environment = dict(base, KUNO_LAB_FROZEN_MEMBERS="three-public")
            self.rejected(dict(environment, KUNO_LAB_WARM_REPLAY="once"), arguments)
            self.rejected(dict(environment, KUNO_LAB_SPECULATIVE_PUBLIC_ONLY="1"), arguments)
            self.rejected(dict(environment, KUNO_LAB_LANGUAGE_FIXTURE="private"), arguments)

    def test_selector_cannot_escape_to_another_probe_or_fallback_limit(self):
        for fixture in self.FIXTURES:
            base, arguments, _ = public_scope(fixture)
            environment = dict(base, KUNO_LAB_FROZEN_MEMBERS="three-public")
            for wrong in (
                [arguments[0], "nvidia", arguments[2]],
                [arguments[0], "intel", "daslab-target-rows-probe"],
                [arguments[0], "intel", "public-host-weight-cubecl-probe"],
                [*arguments, "extra"],
                arguments[:2],
            ):
                with self.subTest(fixture=fixture, arguments=wrong):
                    self.rejected(environment, wrong)
            with self.assertRaises(ValueError):
                GUARD.selected_target_policy(environment, mtp=False)
            unrelated = dict(environment)
            unrelated.pop("KUNO_LAB_SPECULATIVE_PUBLIC_ONLY", None)
            unrelated.pop("KUNO_LAB_LANGUAGE_PUBLIC_ONLY", None)
            self.rejected(unrelated, arguments)
        self.rejected(
            {"KUNO_LAB_FROZEN_MEMBERS": "three-public"},
            ["/frozen/binary", "intel", "batch-router-gpu-probe"],
        )


if __name__ == "__main__":
    unittest.main()
