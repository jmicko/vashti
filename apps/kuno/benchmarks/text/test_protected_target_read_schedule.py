"""Explicit original target read/schedule admission without cap or draft changes."""

import itertools
import unittest

from test_protected_frozen_members import GUARD, public_scope
from test_protected_intel_mtp import intel_scope


PLE = "KUNO_LAB_TARGET_PLE_READS"
DOWN = "KUNO_LAB_FROZEN_DOWN_SCHEDULE"
SELECTORS = {PLE: "grouped-public", DOWN: "four-public"}
FIXTURES = ("warm-cpu13", "first-code", "warm-prose")
REQUIRED = (
    "KUNO_LAB_TARGET_SOURCES", "KUNO_LAB_FROZEN_PARTITION_EXIT",
    "KUNO_LAB_TARGET_ROUTER", "KUNO_LAB_TARGET_REUSE", "KUNO_LAB_TARGET_FILL",
    "KUNO_LAB_DEVICE_DISPATCH", "KUNO_LAB_TARGET_IO", "KUNO_LAB_TARGET_DENSE",
)
CONFLICTS = {
    "KUNO_LAB_FROZEN_TABLES": "private-public",
    "KUNO_LAB_FROZEN_DOWN_SCALES": "shared-public",
    "KUNO_LAB_FROZEN_GU_FACTORS": "shared-public",
    "KUNO_LAB_FROZEN_GU_ROWS": "two-public",
    "KUNO_LAB_FROZEN_HOST_STAGE": "packed-public",
    "KUNO_LAB_FROZEN_MEMBERS": "three-public",
    "KUNO_LAB_FROZEN_BANK_PLACEMENT": "preserve-public",
    "KUNO_LAB_TARGET_EXPERTS": "prepared-verify3",
    "KUNO_LAB_TARGET_MOE": "live",
    "KUNO_LAB_PROMPT": "forbidden-private-input",
    "KUNO_LAB_OTHER": "1", "KUNO_TEST_OTHER": "1",
    "KUNO_PUBLIC_OTHER": "/public/unknown",
}


def target_scope(fixture, placement="nvidia"):
    if placement == "intel":
        environment, arguments, cap = intel_scope(fixture)
    else:
        environment, arguments, cap = public_scope(fixture)
        environment["KUNO_LAB_TARGET_DENSE"] = "verify3-fused"
        if placement == "target-only":
            environment["KUNO_LAB_LANGUAGE_DECODE"] = "target-only-public"
    return environment, arguments, cap


def select(environment, policies):
    selected = dict(environment)
    for policy in policies:
        selected[policy] = SELECTORS[policy]
        if policy == DOWN:
            selected["KUNO_LAB_FROZEN_DOWN_ROWS"] = "four-public"
    return selected


class OriginalTargetReadScheduleAdmission(unittest.TestCase):
    def rejected(self, environment, arguments):
        with self.assertRaises(ValueError):
            GUARD.memory_limit(environment, arguments)

    def test_default_and_each_explicit_policy_keep_exact_caps_and_input(self):
        for fixture in FIXTURES:
            base, arguments, cap = target_scope(fixture)
            for policies in ((), (PLE,), (DOWN,), (PLE, DOWN)):
                environment = select(base, policies)
                original = dict(environment)
                self.assertEqual(GUARD.memory_limit(environment, arguments), cap)
                remaining = GUARD.selected_target_policy(environment, mtp=True)
                for key in SELECTORS:
                    self.assertNotIn(key, remaining)
                self.assertEqual(environment, original)
            self.assertEqual(cap, 48 * 1024**3 + (0 if fixture == "warm-cpu13" else 196 * 1024**2))

    def test_all_placement_and_plain_integer_stride_gu_compositions(self):
        for fixture in FIXTURES:
            placements = ("nvidia", "intel") if fixture == "warm-cpu13" else ("nvidia", "intel", "target-only")
            for placement, gu, policies in itertools.product(
                placements, ("plain", "integer", "stride"), ((PLE,), (DOWN,), (PLE, DOWN)),
            ):
                with self.subTest(fixture=fixture, placement=placement, gu=gu, policies=policies):
                    base, arguments, cap = target_scope(fixture, placement)
                    environment = select(base, policies)
                    environment.update(KUNO_LAB_FROZEN_DOWN_ROWS="four-public", KUNO_LAB_TARGET_QSA="chronological-public")
                    if placement != "target-only":
                        environment.update(KUNO_LAB_MTP_SWEEPS="fused", KUNO_LAB_MTP_Q8_HEAD="shared-public", KUNO_LAB_TARGET_GDN="three-streaming-public")
                    if gu != "plain":
                        environment["KUNO_LAB_FROZEN_GU_MATH"] = "integer-public"
                    if gu == "stride":
                        environment["KUNO_LAB_FROZEN_GU_SCHEDULE"] = "four-public"
                    self.assertEqual(GUARD.memory_limit(environment, arguments), cap)
                    remaining = GUARD.selected_target_policy(environment, mtp=True)
                    self.assertFalse(set(SELECTORS) & set(remaining))

    def test_no_draft_profile_or_sweep_dependency_is_fabricated(self):
        for fixture, policies, q2, fused in itertools.product(
            FIXTURES, ((PLE,), (DOWN,), (PLE, DOWN)), (False, True), (False, True),
        ):
            environment, arguments, cap = target_scope(fixture)
            environment = select(environment, policies)
            if q2:
                environment["KUNO_LAB_MTP_PROFILE"] = "strata-q2"
            if fused:
                environment["KUNO_LAB_MTP_SWEEPS"] = "fused"
            self.assertEqual(GUARD.memory_limit(environment, arguments), cap)
            if q2:
                self.rejected(dict(environment, KUNO_LAB_MTP_Q8_HEAD="shared-public"), arguments)
                self.rejected(dict(environment, KUNO_LAB_FROZEN_GU_MATH="integer-public"), arguments)
        for fixture in ("first-code", "warm-prose"):
            base, arguments, cap = target_scope(fixture, "target-only")
            environment = select(base, (PLE, DOWN))
            self.assertEqual(GUARD.memory_limit(environment, arguments), cap)
            for key, value in {"KUNO_LAB_MTP_SWEEPS": "fused", "KUNO_LAB_MTP_PROFILE": "strata-q2", "KUNO_LAB_MTP_Q8_HEAD": "shared-public", "KUNO_LAB_MTP_DEVICE": "intel-public", "KUNO_LAB_TARGET_GDN": "three-streaming-public"}.items():
                self.rejected(dict(environment, **{key: value}), arguments)

    def test_every_original_requirement_and_selector_is_canonical(self):
        for fixture, policy in itertools.product(FIXTURES, SELECTORS):
            base, arguments, _ = target_scope(fixture)
            environment = select(base, (policy,))
            required = (*REQUIRED, "KUNO_LAB_STATIC_CACHE_BYTES", "KUNO_LAB_SPECULATIVE_PUBLIC_ONLY" if fixture == "warm-cpu13" else "KUNO_LAB_LANGUAGE_PUBLIC_ONLY")
            if policy == DOWN:
                required += ("KUNO_LAB_FROZEN_DOWN_ROWS",)
            if fixture == "warm-cpu13":
                required += ("KUNO_LAB_WARM_REPLAY",)
            else:
                required += ("KUNO_LAB_LANGUAGE_FIXTURE",)
            for key in required:
                for value in (None, "", environment[key] + " "):
                    with self.subTest(fixture=fixture, policy=policy, key=key, value=value):
                        invalid = dict(environment)
                        if value is None:
                            del invalid[key]
                        else:
                            invalid[key] = value
                        self.rejected(invalid, arguments)
            for value in ("", "true", "1", "grouped", "four", SELECTORS[policy] + " ", SELECTORS[policy].upper()):
                self.rejected(dict(environment, **{policy: value}), arguments)
            self.rejected(dict(environment, KUNO_LAB_TARGET_ROUTER="device-warm-public"), arguments)
            self.rejected(dict(environment, KUNO_LAB_TARGET_DENSE="verify3"), arguments)

    def test_incompatible_target_math_private_content_and_unknown_flags_reject(self):
        for fixture, policy in itertools.product(FIXTURES, SELECTORS):
            base, arguments, _ = target_scope(fixture)
            environment = select(base, (policy,))
            for key, value in CONFLICTS.items():
                with self.subTest(fixture=fixture, policy=policy, key=key):
                    self.rejected(dict(environment, **{key: value}), arguments)
            self.rejected(dict(environment, KUNO_LAB_FROZEN_TABLES="workgroup-public"), arguments)
            self.rejected(dict(environment, KUNO_LAB_FROZEN_GU_SCHEDULE="four-public"), arguments)
            self.rejected(dict(environment, KUNO_LAB_SPECULATIVE_PUBLIC_ONLY="1", KUNO_LAB_LANGUAGE_PUBLIC_ONLY="1"), arguments)
            if fixture != "warm-cpu13":
                self.rejected(dict(environment, KUNO_LAB_WARM_REPLAY="once"), arguments)

    def test_ple_full_vocabulary_and_independent_down_draft_scope(self):
        # DOWN alone has no draft-arithmetic exclusion; the existing English
        # selector must still satisfy its own original fused Q8 prerequisites.
        for fixture in FIXTURES:
            base, arguments, cap = target_scope(fixture)
            environment = select(base, (DOWN,))
            environment.update(KUNO_LAB_MTP_SWEEPS="fused", KUNO_LAB_MTP_VOCAB="english-public")
            self.assertEqual(GUARD.memory_limit(environment, arguments), cap)
            self.rejected(dict(environment, **{PLE: "grouped-public"}), arguments)
            self.rejected(dict(environment, KUNO_LAB_MTP_VOCAB=""), arguments)

    def test_selectors_cannot_escape_to_other_modes_devices_or_generic_caps(self):
        for fixture, policy in itertools.product(FIXTURES, SELECTORS):
            base, arguments, _ = target_scope(fixture)
            environment = select(base, (policy,))
            for wrong in ([arguments[0], "nvidia", arguments[2]], [arguments[0], "intel", "batch-router-gpu-probe"], [arguments[0], "intel", "public-host-weight-cubecl-probe"], arguments + ["extra"], arguments[:2]):
                self.rejected(environment, wrong)
            with self.assertRaises(ValueError):
                GUARD.selected_target_policy(environment, mtp=False)
            self.rejected({policy: SELECTORS[policy]}, ["/frozen/binary", "intel", "resident-count1-hybrid-chat"])
            self.rejected(dict(environment, KUNO_LAB_STATIC_CACHE_BYTES="17179869184"), arguments)
            self.rejected(dict(environment, KUNO_LAB_LANGUAGE_FIXTURE="private"), arguments)


if __name__ == "__main__":
    unittest.main()
