"""Original grouped status admission with independent read/schedule choices."""

import itertools
import unittest

from test_protected_target_read_schedule import (
    CONFLICTS, DOWN, FIXTURES, GUARD, PLE, REQUIRED, select, target_scope,
)

STATUS = "KUNO_LAB_TARGET_STATUS_FOLDS"


def status_scope(fixture, placement="nvidia", policies=()):
    environment, arguments, cap = target_scope(fixture, placement)
    environment = select(environment, policies)
    environment[STATUS] = "grouped-public"
    return environment, arguments, cap


class GroupedStatusAdmission(unittest.TestCase):
    def rejected(self, environment, arguments):
        with self.assertRaises(ValueError):
            GUARD.memory_limit(environment, arguments)

    def test_status_is_independent_of_ple_down_and_retains_exact_caps(self):
        for fixture, policies, selected in itertools.product(
            FIXTURES, ((), (PLE,), (DOWN,), (PLE, DOWN)), (False, True),
        ):
            environment, arguments, cap = target_scope(fixture)
            environment = select(environment, policies)
            if selected:
                environment[STATUS] = "grouped-public"
            original = dict(environment)
            self.assertEqual(GUARD.memory_limit(environment, arguments), cap)
            remaining = GUARD.selected_target_policy(environment, mtp=True)
            self.assertFalse({STATUS, PLE, DOWN} & set(remaining))
            self.assertEqual(environment, original)
            self.assertEqual(cap, 48 * 1024**3 + (0 if fixture == "warm-cpu13" else 196 * 1024**2))

    def test_all_original_placement_gu_read_and_schedule_compositions(self):
        for fixture in FIXTURES:
            placements = ("nvidia", "intel") if fixture == "warm-cpu13" else ("nvidia", "intel", "target-only")
            for placement, gu, policies in itertools.product(
                placements, ("plain", "integer", "stride"), ((), (PLE,), (DOWN,), (PLE, DOWN)),
            ):
                with self.subTest(fixture=fixture, placement=placement, gu=gu, policies=policies):
                    environment, arguments, cap = status_scope(fixture, placement, policies)
                    environment.update(KUNO_LAB_FROZEN_DOWN_ROWS="four-public", KUNO_LAB_TARGET_QSA="chronological-public")
                    if placement != "target-only":
                        environment.update(KUNO_LAB_MTP_SWEEPS="fused", KUNO_LAB_MTP_Q8_HEAD="shared-public", KUNO_LAB_TARGET_GDN="three-streaming-public")
                    if gu != "plain":
                        environment["KUNO_LAB_FROZEN_GU_MATH"] = "integer-public"
                    if gu == "stride":
                        environment["KUNO_LAB_FROZEN_GU_SCHEDULE"] = "four-public"
                    self.assertEqual(GUARD.memory_limit(environment, arguments), cap)

    def test_original_draft_scopes_are_preserved_without_fabricated_prerequisites(self):
        for fixture, policies, q2, fused in itertools.product(
            FIXTURES, ((), (PLE,), (DOWN,), (PLE, DOWN)), (False, True), (False, True),
        ):
            environment, arguments, cap = status_scope(fixture, policies=policies)
            if q2:
                environment["KUNO_LAB_MTP_PROFILE"] = "strata-q2"
            if fused:
                environment["KUNO_LAB_MTP_SWEEPS"] = "fused"
            self.assertEqual(GUARD.memory_limit(environment, arguments), cap)
            if q2:
                self.rejected(dict(environment, KUNO_LAB_MTP_Q8_HEAD="shared-public"), arguments)
                self.rejected(dict(environment, KUNO_LAB_FROZEN_GU_MATH="integer-public"), arguments)
        for fixture in ("first-code", "warm-prose"):
            environment, arguments, cap = status_scope(fixture, "target-only", (PLE, DOWN))
            self.assertEqual(GUARD.memory_limit(environment, arguments), cap)
            for key, value in {"KUNO_LAB_MTP_SWEEPS": "fused", "KUNO_LAB_MTP_PROFILE": "strata-q2", "KUNO_LAB_MTP_Q8_HEAD": "shared-public", "KUNO_LAB_MTP_DEVICE": "intel-public", "KUNO_LAB_TARGET_GDN": "three-streaming-public"}.items():
                self.rejected(dict(environment, **{key: value}), arguments)
        # Actual Intel placement keeps its existing fused-sweep requirement.
        environment, arguments, _ = status_scope("first-code", "intel")
        del environment["KUNO_LAB_MTP_SWEEPS"]
        self.rejected(environment, arguments)

    def test_selector_and_every_original_requirement_are_canonical(self):
        for fixture in FIXTURES:
            environment, arguments, _ = status_scope(fixture)
            required = (*REQUIRED, "KUNO_LAB_STATIC_CACHE_BYTES", "KUNO_LAB_SPECULATIVE_PUBLIC_ONLY" if fixture == "warm-cpu13" else "KUNO_LAB_LANGUAGE_PUBLIC_ONLY", "KUNO_LAB_WARM_REPLAY" if fixture == "warm-cpu13" else "KUNO_LAB_LANGUAGE_FIXTURE")
            for key in required:
                for value in (None, "", environment[key] + " "):
                    with self.subTest(fixture=fixture, key=key, value=value):
                        invalid = dict(environment)
                        if value is None:
                            del invalid[key]
                        else:
                            invalid[key] = value
                        self.rejected(invalid, arguments)
            for value in ("", "true", "1", "grouped", "GROUPED-PUBLIC", "grouped-public "):
                self.rejected(dict(environment, **{STATUS: value}), arguments)
            self.rejected(dict(environment, KUNO_LAB_TARGET_ROUTER="device-warm-public"), arguments)
            self.rejected(dict(environment, KUNO_LAB_TARGET_DENSE="verify3"), arguments)

    def test_incompatible_math_private_content_unknown_flags_and_english_reject(self):
        for fixture in FIXTURES:
            environment, arguments, _ = status_scope(fixture)
            for key, value in CONFLICTS.items():
                with self.subTest(fixture=fixture, key=key):
                    self.rejected(dict(environment, **{key: value}), arguments)
            self.rejected(dict(environment, KUNO_LAB_FROZEN_TABLES="workgroup-public"), arguments)
            self.rejected(dict(environment, KUNO_LAB_FROZEN_GU_SCHEDULE="four-public"), arguments)
            self.rejected(dict(environment, KUNO_LAB_SPECULATIVE_PUBLIC_ONLY="1", KUNO_LAB_LANGUAGE_PUBLIC_ONLY="1"), arguments)
            self.rejected(dict(environment, KUNO_LAB_MTP_SWEEPS="fused", KUNO_LAB_MTP_VOCAB="english-public"), arguments)
            if fixture != "warm-cpu13":
                self.rejected(dict(environment, KUNO_LAB_WARM_REPLAY="once"), arguments)

    def test_status_cannot_escape_public_arguments_or_generic_caps(self):
        for fixture in FIXTURES:
            environment, arguments, _ = status_scope(fixture)
            for wrong in ([arguments[0], "nvidia", arguments[2]], [arguments[0], "intel", "batch-router-gpu-probe"], [arguments[0], "intel", "public-host-weight-cubecl-probe"], arguments + ["extra"], arguments[:2]):
                self.rejected(environment, wrong)
            with self.assertRaises(ValueError):
                GUARD.selected_target_policy(environment, mtp=False)
            self.rejected({STATUS: "grouped-public"}, ["/frozen/binary", "intel", "resident-count1-hybrid-chat"])
            self.rejected(dict(environment, KUNO_LAB_STATIC_CACHE_BYTES="17179869184"), arguments)
            self.rejected(dict(environment, KUNO_LAB_LANGUAGE_FIXTURE="private"), arguments)


if __name__ == "__main__":
    unittest.main()
