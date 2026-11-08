"""CPU protocol/schedule fixtures only; no live API, model, GPU or authority."""
import unittest

from public_short_tail_cases import MODEL, TOTALS, run_public_short_tail_cases


class PublicApiFixture:
    def __init__(self, offset=22, mutation=None):
        self.offset = offset
        self.mutation = mutation
        self.counts = []

    def __call__(self, path, body):
        assert path == "/api/chat" and body["model"] == MODEL
        assert body["think"] is False and body["stream"] is False
        assert body["options"] == {"num_predict": 32, "num_ctx": 512,
                                   "temperature": 0, "seed": 0}
        assert len(body["messages"]) == 1
        assert body["messages"][0]["role"] == "user"
        prompt = body["messages"][0]["content"]
        assert prompt.startswith("Public reference: ")
        assert prompt.endswith("\nWhat is 17 times 19? Reply only with the integer.")
        fillers = prompt.count("test ")
        count = fillers + self.offset
        self.counts.append(count)
        value = {"done": True, "done_reason": "stop", "prompt_eval_count": count,
                 "eval_count": 4, "message": {"content": "323"},
                 "prompt_eval_duration": 10, "eval_duration": 20,
                 "load_duration": 0}
        if self.mutation is not None:
            self.mutation(value, len(self.counts))
        return value


class TailCases(unittest.TestCase):
    def test_exact_tail_schedule_and_content_free_receipt(self):
        api = PublicApiFixture()
        receipt = run_public_short_tail_cases(api, lambda: 123, 123)
        self.assertFalse(receipt["content_saved"])
        self.assertTrue(receipt["request_checks_passed_before_session_close"])
        self.assertEqual(len(receipt["calibration"]), 2)
        cases = receipt["cases"]
        self.assertEqual([c["prompt_tokens"] for c in cases],
                         list(TOTALS) + list(reversed(TOTALS)))
        self.assertEqual([c["terminal_window_rows"] for c in cases],
                         list(range(1, 9)) + list(range(8, 0, -1)))
        self.assertEqual([c["prefix_tokens"] for c in cases[:8]], list(range(49, 57)))
        self.assertTrue(all(c["known_prefix_windows"] == 7 for c in cases))
        # Deliberately no whole-session/physical-recovery passed assertion here.
        self.assertNotIn("passed", receipt)
        forbidden = {"prompt", "response", "content", "token_ids", "content_hash"}

        def check(value):
            if isinstance(value, dict):
                self.assertFalse(forbidden.intersection(value))
                for item in value.values():
                    check(item)
            elif isinstance(value, list):
                for item in value:
                    check(item)

        check(receipt)

    def test_first_calibration_can_be_reused_without_extra_inference(self):
        api = PublicApiFixture(offset=18)
        receipt = run_public_short_tail_cases(api, lambda: 123, 123)
        self.assertEqual(len(receipt["calibration"]), 1)
        self.assertEqual(api.counts, list(TOTALS) + list(reversed(TOTALS)))

    def test_wrong_arithmetic_truncation_reload_and_pid_rejected(self):
        for category, field, value in [
            ("wrong_arithmetic", "message", {"content": "324"}),
            ("incomplete_output", "done_reason", "length"),
            ("incomplete_output", "eval_count", 32),
            ("worker_reloaded", "load_duration", 1),
            ("prompt_count", "prompt_eval_count", True),
            ("duration_metadata", "prompt_eval_duration", -1),
        ]:
            with self.subTest(category=category):
                api = PublicApiFixture(mutation=lambda v, n: v.update({field: value}))
                with self.assertRaisesRegex(RuntimeError, "public_short_tail_" + category):
                    run_public_short_tail_cases(api, lambda: 123, 123)
        with self.assertRaisesRegex(RuntimeError, "worker_replaced"):
            run_public_short_tail_cases(PublicApiFixture(), lambda: 456, 123)

    def test_nonadjacent_tail_and_unbounded_calibration_rejected(self):
        def nonadjacent(value, request):
            if request == 3:
                value["prompt_eval_count"] += 1

        with self.assertRaisesRegex(RuntimeError, "nonadjacent_token_count"):
            run_public_short_tail_cases(PublicApiFixture(mutation=nonadjacent),
                                        lambda: 123, 123)
        api = PublicApiFixture(mutation=lambda value, n:
                               value.update(prompt_eval_count=49))
        with self.assertRaisesRegex(RuntimeError, "calibration_failed"):
            run_public_short_tail_cases(api, lambda: 123, 123)
        self.assertEqual(len(api.counts), 3)


if __name__ == "__main__":
    unittest.main()
