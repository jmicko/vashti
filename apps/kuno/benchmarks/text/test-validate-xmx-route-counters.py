"""Reject lost routes, wrong buckets and missing physical work independently."""
import copy
import importlib.util
import pathlib
import unittest

path = pathlib.Path(__file__).with_name("validate-xmx-route-counters.py")
spec = importlib.util.spec_from_file_location("route_counters", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class RouteInventory(unittest.TestCase):
    def setUp(self):
        self.p = {}
        for suffix, _, _ in module.BANDS:
            self.p["expert_groups_rows_" + suffix] = [0]
            self.p["expert_members_rows_" + suffix] = [0]
        #72 original rows *48 layers *10 experts; groups of8 members.
        g = 4320
        self.p.update({k: [v] for k, v in dict(
            expert_groups_rows_8_15=g, expert_members_rows_8_15=34560,
            expert_group_count=g, expert_resident_groups=2000,
            expert_streamed_groups=2320, expert_native_bulk_batches=48,
            expert_native_bulk_logical_phases=6*g,
            expert_native_bulk_tail_pieces=0, expert_native_bulk_wide_prefixes=0,
            expert_native_bulk_matrix_pieces=2*g, expert_native_bulk_launches=6*g,
            expert_native_scratch_groups=g, expert_native_scratch_slots=96).items()})

    def test_different_group_population_can_still_conserve_every_original_route(self):
        self.assertEqual(module.row(self.p, {"prompt_tokens": 73})["original_route_members"], 34560)

    def test_independent_work_omission_and_duplication_are_rejected(self):
        for name in ["expert_members_rows_8_15", "expert_group_count",
                     "expert_native_bulk_launches", "expert_native_bulk_matrix_pieces",
                     "expert_native_scratch_slots", "expert_native_bulk_batches",
                     "expert_native_bulk_logical_phases", "expert_streamed_groups"]:
            for delta in (-1, 1):
                with self.subTest(name=name, delta=delta):
                    p = copy.deepcopy(self.p)
                    p[name][0] += delta
                    with self.assertRaises(AssertionError):
                        module.row(p, {"prompt_tokens": 73})

    def test_bucket_cannot_hide_a_wrong_population_even_if_total_members_match(self):
        p = copy.deepcopy(self.p)
        p["expert_groups_rows_8_15"] = [2160]
        p["expert_groups_rows_1_7"] = [2160]
        p["expert_members_rows_8_15"] = [0]
        p["expert_members_rows_1_7"] = [34560]
        with self.assertRaises(AssertionError):
            module.row(p, {"prompt_tokens": 73})


if __name__ == "__main__":
    unittest.main()
