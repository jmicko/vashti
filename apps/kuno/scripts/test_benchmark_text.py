"""Stream/counter validation for the operator-only synthetic benchmark client."""

import contextlib
import importlib.util
import io
import json
import pathlib
import tempfile
import unittest
from unittest.mock import patch


SPEC = importlib.util.spec_from_file_location("benchmark_text", pathlib.Path(__file__).with_name("benchmark-text.py"))
BENCH = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BENCH)


class Stream:
    status = 200

    def __init__(self, records):
        self.records = records

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def __iter__(self):
        return iter(json.dumps(r).encode() + b"\n" for r in self.records)


class BenchmarkTests(unittest.TestCase):
    def run_fixture(self, records):
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "receipt.json"
            receipt = {"runs": []}
            with patch.object(BENCH, "request", return_value=Stream(records)), \
                    patch.object(BENCH, "metadata", return_value={"models": []}), \
                    contextlib.redirect_stdout(io.StringIO()):
                BENCH.generate("http://localhost", {}, "fixture", path, receipt)
            return json.loads(path.read_text())["runs"][0], path.read_text()

    def test_output_hash_is_independent_of_stream_chunking(self):
        first, serialized = self.run_fixture([
            {"response": "rain"}, {"response": "water"}, {"done": True}])
        second, _ = self.run_fixture([{"response": "rainwater", "done": True}])
        self.assertEqual(first["synthetic_output_sha256"], second["synthetic_output_sha256"])
        self.assertNotIn("rain", serialized)
        self.assertEqual(first["synthetic_output_bytes"], 9)

    def test_cached_tokens_are_excluded_from_prefill_rate(self):
        result, _ = self.run_fixture([{
            "done": True, "prompt_eval_count": 100, "prompt_eval_cached_count": 80,
            "prompt_eval_duration": 2_000_000_000,
            "eval_count": 40, "eval_duration": 4_000_000_000}])
        self.assertEqual(result["uncached_prefill_tokens_per_second"], 10)
        self.assertEqual(result["decode_tokens_per_second"], 10)

    def test_missing_completion_is_failure_without_replay(self):
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "receipt.json"
            with patch.object(BENCH, "request", return_value=Stream([{ "response": "partial" }])) as request, \
                    contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(ValueError, "missing_final_record"):
                    BENCH.generate("http://localhost", {}, "fixture", path, {"runs": []})
                self.assertEqual(request.call_count, 1)
            self.assertNotIn("partial", path.read_text())


if __name__ == "__main__":
    unittest.main()
