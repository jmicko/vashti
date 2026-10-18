"""Synthetic observer checks; these do not run or certify model inference."""
import copy
import unittest

from strata_throughput import PublicThroughput


class ThroughputReceiptTests(unittest.TestCase):
    def observer(self):
        ticks = iter([0.0, 10.0, 12.0, 22.0, 30.0, 31.0, 36.0])
        observer = PublicThroughput(lambda: next(ticks))
        # A preview containing escaped fake phase text must never be timed.
        observer.observe('strata_public_preview="native_public_execution phase=target"\n')
        for phase in ["target", "request_preflight", "last_prompt_input", "request_private_erasure",
                      "request_preflight", "last_prompt_input", "request_private_erasure"]:
            observer.observe("native_public_execution phase=" + phase + "\n")
        return observer

    def receipts(self):
        completion = {"target_owners_erased": True, "draft_owners_erased": True,
                      "record": {"request_scope_erased": True,
                                 "identical_public_repeat_matched": True,
                                 "loaded_session_requests": 2,
                                 "greedy_target_verification": True, "content_persisted": False,
                                 "outputs": 100, "windows": 30}}
        after = {"admission": {"state": {"admission": dict.fromkeys(
            ["active", "waiting", "paused", "recovery_required"], False)}},
            "http_statuses": dict.fromkeys(["kuno", "ollama", "vashti"], 200),
            "kernel_fault_counts": {"xe_timeouts": 0}, "nvidia_compute_process_count": 0,
            "intel": {"xe_memory": [{"used_bytes": 50688000}]}}
        return completion, copy.deepcopy(after), after

    def test_complete_repeat_reports_delivered_rate_with_explicit_boundaries(self):
        result = self.observer().finish(*self.receipts())
        self.assertEqual(result["runtime_setup_to_first_request_seconds"], 10)
        self.assertEqual([r["last_prompt_inclusive_tokens_per_second"] for r in result["requests"]], [10, 20])
        self.assertFalse(result["pure_gpu_decode_rate"])

    def test_each_failed_recovery_or_repeat_condition_prevents_any_rate(self):
        for key in ["identical_public_repeat_matched", "request_scope_erased", "greedy_target_verification"]:
            completion, before, after = self.receipts()
            completion["record"][key] = False
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.observer().finish(completion, before, after)
        for fault in ["driver", "resident", "admission", "http"]:
            completion, before, after = self.receipts()
            if fault == "driver": after["kernel_fault_counts"]["xe_timeouts"] = 1
            if fault == "resident": after["intel"]["xe_memory"][0]["used_bytes"] = 1024**3
            if fault == "admission": after["admission"]["state"]["admission"]["paused"] = True
            if fault == "http": after["http_statuses"].pop("vashti")
            with self.subTest(fault=fault), self.assertRaises(ValueError):
                self.observer().finish(completion, before, after)

    def test_partial_or_failed_continuation_never_produces_throughput(self):
        completion, before, after = self.receipts()
        completion["record"]["outputs"] = 1
        with self.assertRaises(ValueError):
            self.observer().finish(completion, before, after)
        observer = PublicThroughput()
        with self.assertRaises(ValueError): observer.finish(*self.receipts())
        observer = self.observer()
        observer.observe("native_public_execution phase=controller_execution_error\n")
        with self.assertRaises(ValueError): observer.finish(*self.receipts())

    def test_missing_duplicate_or_out_of_order_phase_cannot_be_a_valid_run(self):
        observer = PublicThroughput()
        with self.assertRaises(ValueError):
            observer.observe("native_public_execution phase=last_prompt_input\n")
        observer = self.observer()
        with self.assertRaises(ValueError):
            observer.observe("native_public_execution phase=request_preflight\n")


if __name__ == "__main__":
    unittest.main()
