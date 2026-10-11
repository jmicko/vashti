"""Keep CPU/Intel overlap trials fixed and independently bounded."""
import unittest
from test_protected_target_read_schedule import GUARD


class HostCpuOverlapAdmission(unittest.TestCase):
    def setUp(self):
        self.environment = {
            "KUNO_LAB_HOST_CPU_OVERLAP_PUBLIC_ONLY": "1",
            "KUNO_LAB_DEVICE_DISPATCH": "inline",
            "KUNO_LAB_STATIC_CACHE_BYTES": "8589934592",
            "KUNO_TEST_Q3_DIR": "/fixed/public/model",
            "KUNO_PUBLIC_TARGET_CARRY13_REFERENCE": "/fixed/public/reference",
            "KUNO_PUBLIC_RECEIPT": "/fixed/public/numeric.json",
        }
        self.arguments = ["/frozen/binary", "intel", "public-host-cpu-overlap-probe"]

    def reject(self, environment=None, arguments=None):
        with self.assertRaises(ValueError):
            GUARD.memory_limit(self.environment if environment is None else environment,
                               self.arguments if arguments is None else arguments)

    def test_canonical_scope_preserves_exact_process_cap_and_input(self):
        original = dict(self.environment)
        self.assertEqual(GUARD.memory_limit(self.environment, self.arguments), 48 * 1024**3)
        self.assertEqual(self.environment, original)

    def test_each_required_marker_path_and_dispatch_are_checked(self):
        for key in self.environment:
            bad = dict(self.environment)
            del bad[key]
            self.reject(bad)
            self.reject(dict(self.environment, **{key: ""}))
        for key in ("KUNO_LAB_HOST_CPU_OVERLAP_PUBLIC_ONLY", "KUNO_LAB_DEVICE_DISPATCH",
                    "KUNO_LAB_STATIC_CACHE_BYTES"):
            self.reject(dict(self.environment, **{key: self.environment[key] + " "}))

    def test_other_device_mode_and_extra_arguments_are_not_admitted(self):
        for arguments in (
            self.arguments + ["extra"],
            ["/frozen/binary", "nvidia", "public-host-cpu-overlap-probe"],
            ["/frozen/binary", "intel", "daslab-language-probe"],
            ["/frozen/binary", "intel"],
        ):
            self.reject(arguments=arguments)

    def test_no_other_model_private_input_policy_or_larger_cap(self):
        for key, value in {
            "KUNO_TEST_MTP": "/another/model",
            "KUNO_LAB_HOST_CPU_BATCH_PUBLIC_ONLY": "1",
            "KUNO_LAB_STATIC_CACHE_BYTES": "17179869184",
            "KUNO_LAB_FROZEN_BANK_FILL": "profile-ranked-public",
            "KUNO_LAB_FROZEN_GU_MATH": "integer-public",
            "KUNO_LAB_LANGUAGE_PUBLIC_ONLY": "1",
            "KUNO_LAB_SPECULATIVE_PUBLIC_ONLY": "1",
            "KUNO_LAB_HOST_CPU_BATCH_WORKERS": "32",
            "KUNO_PUBLIC_PROMPT": "private",
            "KUNO_PUBLIC_CPU_HOST_BYTES": "2097152",
            "KUNO_LAB_STATIC_CACHE_BYTES": "6442450944",
        }.items():
            self.reject(dict(self.environment, **{key: value}))


if __name__ == "__main__":
    unittest.main()
