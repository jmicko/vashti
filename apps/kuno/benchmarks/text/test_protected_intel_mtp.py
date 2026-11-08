"""Bounded same-card MTP admission requires real positive Xe metadata."""

import unittest

from test_protected_frozen_members import GUARD, public_scope


def intel_scope(fixture):
    environment, arguments, cap = public_scope(fixture)
    environment.update(KUNO_LAB_TARGET_DENSE="verify3-fused", KUNO_LAB_MTP_SWEEPS="fused", KUNO_LAB_MTP_DEVICE="intel-public", KUNO_PUBLIC_MTP_INTEL_TOTAL_BYTES="34242297856", KUNO_PUBLIC_MTP_INTEL_USED_BYTES="50688000", KUNO_PUBLIC_MTP_INTEL_DEVICE_ID="57891")
    return environment, arguments, cap


class IntelMtpAdmission(unittest.TestCase):
    def test_public_original_q8_and_q2_scopes_retain_exact_host_caps(self):
        for fixture in ("warm-cpu13", "first-code", "warm-prose"):
            base, arguments, cap = intel_scope(fixture)
            for profile in (None, "strata-q2"):
                environment = dict(base, KUNO_LAB_FROZEN_DOWN_ROWS="four-public", KUNO_LAB_TARGET_QSA="chronological-public")
                if profile:
                    environment["KUNO_LAB_MTP_PROFILE"] = profile
                else:
                    environment.update(KUNO_LAB_MTP_Q8_HEAD="shared-public", KUNO_LAB_FROZEN_GU_MATH="integer-public", KUNO_LAB_FROZEN_GU_SCHEDULE="four-public")
                self.assertEqual(GUARD.memory_limit(environment, arguments), cap)
                selected = GUARD.selected_target_policy(environment, mtp=True)
                self.assertNotIn("KUNO_LAB_MTP_DEVICE", selected)
                self.assertNotIn("KUNO_PUBLIC_MTP_INTEL_TOTAL_BYTES", selected)

    def test_every_device_metadata_and_target_requirement_is_canonical(self):
        for fixture in ("warm-cpu13", "first-code", "warm-prose"):
            base, arguments, _ = intel_scope(fixture)
            for key in ("KUNO_LAB_MTP_DEVICE", "KUNO_LAB_TARGET_SOURCES", "KUNO_LAB_FROZEN_PARTITION_EXIT", "KUNO_LAB_TARGET_ROUTER", "KUNO_LAB_TARGET_REUSE", "KUNO_LAB_TARGET_FILL", "KUNO_LAB_DEVICE_DISPATCH", "KUNO_LAB_TARGET_IO", "KUNO_LAB_TARGET_DENSE", "KUNO_LAB_MTP_SWEEPS", "KUNO_PUBLIC_MTP_INTEL_TOTAL_BYTES", "KUNO_PUBLIC_MTP_INTEL_USED_BYTES", "KUNO_PUBLIC_MTP_INTEL_DEVICE_ID"):
                for value in (None, "", base[key] + " "):
                    broken = dict(base)
                    if value is None:
                        del broken[key]
                    else:
                        broken[key] = value
                    with self.assertRaises(ValueError):
                        GUARD.memory_limit(broken, arguments)
            for key in ("KUNO_PUBLIC_MTP_INTEL_TOTAL_BYTES", "KUNO_PUBLIC_MTP_INTEL_USED_BYTES", "KUNO_PUBLIC_MTP_INTEL_DEVICE_ID"):
                for value in ("0", "-1", "+1", "01", "1.0", "1e3", "2" * 21, "１２"):
                    with self.assertRaises(ValueError):
                        GUARD.memory_limit(dict(base, **{key: value}), arguments)
            with self.assertRaises(ValueError):
                GUARD.memory_limit(dict(base, KUNO_PUBLIC_MTP_INTEL_USED_BYTES=base["KUNO_PUBLIC_MTP_INTEL_TOTAL_BYTES"]), arguments)

    def test_no_default_device_metadata_target_only_or_private_escape(self):
        conflicts = {"KUNO_LAB_LANGUAGE_DECODE": "target-only-public", "KUNO_LAB_MTP_VOCAB": "english-public", "KUNO_LAB_FROZEN_TABLES": "workgroup-public", "KUNO_LAB_FROZEN_DOWN_SCALES": "shared-public", "KUNO_LAB_FROZEN_GU_FACTORS": "shared-public", "KUNO_LAB_FROZEN_GU_ROWS": "two-public", "KUNO_LAB_FROZEN_HOST_STAGE": "packed-public", "KUNO_LAB_FROZEN_MEMBERS": "three-public", "KUNO_LAB_TARGET_EXPERTS": "prepared-verify3", "KUNO_LAB_TARGET_MOE": "live", "KUNO_LAB_PROMPT": "private"}
        for fixture in ("warm-cpu13", "first-code", "warm-prose"):
            base, arguments, _ = intel_scope(fixture)
            for key, value in conflicts.items():
                with self.assertRaises(ValueError):
                    GUARD.memory_limit(dict(base, **{key: value}), arguments)
            missing = dict(base)
            del missing["KUNO_LAB_MTP_DEVICE"]
            with self.assertRaises(ValueError):
                GUARD.memory_limit(missing, arguments)
            with self.assertRaises(ValueError):
                GUARD.selected_target_policy(base, mtp=False)
            with self.assertRaises(ValueError):
                GUARD.memory_limit(base, [arguments[0], "nvidia", arguments[2]])
            with self.assertRaises(ValueError):
                GUARD.memory_limit(base, [arguments[0], "intel", "batch-router-gpu-probe"])


if __name__ == "__main__":
    unittest.main()
