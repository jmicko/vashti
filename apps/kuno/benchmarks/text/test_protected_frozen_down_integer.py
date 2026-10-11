"""Exact speculative full-Q8 scope for integer DOWN without allowance changes."""

import itertools
import unittest

from test_protected_target_read_schedule import CONFLICTS, FIXTURES, GUARD, REQUIRED, target_scope

DOWN_MATH = "KUNO_LAB_FROZEN_DOWN_MATH"


def down_scope(fixture, placement="nvidia"):
    environment, arguments, cap = target_scope(fixture, placement)
    environment.update(KUNO_LAB_FROZEN_DOWN_MATH="integer-public", KUNO_LAB_MTP_SWEEPS="fused")
    return environment, arguments, cap


class IntegerDownAdmission(unittest.TestCase):
    def rejected(self, environment, arguments):
        with self.assertRaises(ValueError):
            GUARD.memory_limit(environment, arguments)

    def test_each_placement_gu_and_independent_read_composition_keeps_exact_caps(self):
        for fixture, placement, gu, reads in itertools.product(
            FIXTURES, ("nvidia", "intel"), ("plain", "integer", "stride"), ((), ("ple",), ("status",), ("ple", "status")),
        ):
            environment, arguments, cap = down_scope(fixture, placement)
            environment.update(KUNO_LAB_TARGET_QSA="chronological-public", KUNO_LAB_MTP_Q8_HEAD="shared-public", KUNO_LAB_TARGET_GDN="three-streaming-public")
            if gu != "plain":
                environment["KUNO_LAB_FROZEN_GU_MATH"] = "integer-public"
            if gu == "stride":
                environment["KUNO_LAB_FROZEN_GU_SCHEDULE"] = "four-public"
            if "ple" in reads:
                environment["KUNO_LAB_TARGET_PLE_READS"] = "grouped-public"
            if "status" in reads:
                environment["KUNO_LAB_TARGET_STATUS_FOLDS"] = "grouped-public"
            before = dict(environment)
            self.assertEqual(GUARD.memory_limit(environment, arguments), cap)
            self.assertNotIn(DOWN_MATH, GUARD.selected_target_policy(environment, mtp=True))
            self.assertEqual(environment, before)
            self.assertEqual(cap, 48 * 1024**3 + (0 if fixture == "warm-cpu13" else 196 * 1024**2))

    def test_canonical_selector_and_each_original_target_q8_requirement(self):
        for fixture in FIXTURES:
            environment, arguments, _ = down_scope(fixture)
            keys = (*REQUIRED, "KUNO_LAB_MTP_SWEEPS", "KUNO_LAB_STATIC_CACHE_BYTES", "KUNO_LAB_SPECULATIVE_PUBLIC_ONLY" if fixture == "warm-cpu13" else "KUNO_LAB_LANGUAGE_PUBLIC_ONLY", "KUNO_LAB_WARM_REPLAY" if fixture == "warm-cpu13" else "KUNO_LAB_LANGUAGE_FIXTURE")
            for key in keys:
                for value in (None, "", environment[key] + " "):
                    bad = dict(environment)
                    if value is None:
                        del bad[key]
                    else:
                        bad[key] = value
                    self.rejected(bad, arguments)
            for value in ("", "integer", "true", "INTEGER-PUBLIC", "integer-public "):
                self.rejected(dict(environment, **{DOWN_MATH: value}), arguments)
            self.rejected(dict(environment, KUNO_LAB_TARGET_ROUTER="device-warm-public"), arguments)
            self.rejected(dict(environment, KUNO_LAB_TARGET_DENSE="verify3"), arguments)

    def test_old_down_math_other_draft_profiles_and_incompatible_targets_reject(self):
        for fixture in FIXTURES:
            environment, arguments, _ = down_scope(fixture)
            conflicts = dict(CONFLICTS, KUNO_LAB_FROZEN_DOWN_ROWS="four-public", KUNO_LAB_FROZEN_DOWN_SCHEDULE="four-public", KUNO_LAB_MTP_PROFILE="strata-q2", KUNO_LAB_MTP_VOCAB="english-public", KUNO_LAB_MTP_Q8_HEAD="broadcast-public")
            for key, value in conflicts.items():
                self.rejected(dict(environment, **{key: value}), arguments)
            self.rejected(dict(environment, KUNO_LAB_FROZEN_TABLES="workgroup-public"), arguments)
            self.rejected(dict(environment, KUNO_LAB_SPECULATIVE_PUBLIC_ONLY="1", KUNO_LAB_LANGUAGE_PUBLIC_ONLY="1"), arguments)
            if fixture != "warm-cpu13":
                self.rejected(dict(environment, KUNO_LAB_WARM_REPLAY="once"), arguments)

    def test_target_only_rejects_down_integer_but_preserves_original_gu_status_and_ple(self):
        for fixture in ("first-code", "warm-prose"):
            environment, arguments, cap = target_scope(fixture, "target-only")
            environment.update(KUNO_LAB_FROZEN_GU_MATH="integer-public", KUNO_LAB_FROZEN_GU_SCHEDULE="four-public", KUNO_LAB_TARGET_PLE_READS="grouped-public", KUNO_LAB_TARGET_STATUS_FOLDS="grouped-public")
            self.assertEqual(GUARD.memory_limit(environment, arguments), cap)
            self.rejected(dict(environment, **{DOWN_MATH: "integer-public"}), arguments)

    def test_selected_down_cannot_escape_public_args_or_generic_host_caps(self):
        for fixture in FIXTURES:
            environment, arguments, _ = down_scope(fixture)
            for wrong in ([arguments[0], "nvidia", arguments[2]], [arguments[0], "intel", "batch-router-gpu-probe"], arguments + ["extra"], arguments[:2]):
                self.rejected(environment, wrong)
            with self.assertRaises(ValueError):
                GUARD.selected_target_policy(environment, mtp=False)
            self.rejected({DOWN_MATH: "integer-public"}, ["/frozen/binary", "intel", "resident-count1-hybrid-chat"])
            self.rejected(dict(environment, KUNO_LAB_STATIC_CACHE_BYTES="17179869184"), arguments)


if __name__ == "__main__":
    unittest.main()
