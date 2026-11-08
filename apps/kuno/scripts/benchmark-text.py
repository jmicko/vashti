#!/usr/bin/env python3
"""Operator-only public synthetic benchmark; never accepts a user prompt.

Store the reviewed fixture and allowlisted numeric receipts, never raw backend
errors, responses, thinking, request context or model templates. This is not a
production request logger. Generation has no wall-clock or idle deadline.
"""

import argparse
import datetime
import hashlib
import json
import pathlib
import sys
import time
import urllib.error
import urllib.request


COUNTS = ("prompt_eval_count", "prompt_eval_cached_count", "eval_count")
DURATIONS = ("total_duration", "load_duration", "prompt_eval_duration", "eval_duration")


def request(endpoint, path, body=None, timeout=15):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(endpoint + path, data=data,
                                 headers={"Content-Type": "application/json"})
    return urllib.request.urlopen(req, timeout=timeout)


def metadata(endpoint, path):
    with request(endpoint, path) as response:
        return json.load(response)


def save(path, receipt):
    tmp = path.with_suffix(".new")
    tmp.write_text(json.dumps(receipt, indent=2) + "\n")
    tmp.replace(path)


def generate(endpoint, body, label, receipt_path, receipt):
    started = time.monotonic()
    first_content = None
    first_record = None
    digests = {field: hashlib.sha256() for field in ("thinking", "response")}
    text_bytes = 0
    chunks = 0
    last_progress = started
    result = {"label": label, "outcome": "running"}
    receipt["runs"].append(result)
    save(receipt_path, receipt)
    print(json.dumps({"phase": label, "state": "started"}), flush=True)
    # A long model load or inference is valid. Do not impose a five-minute
    # timeout, or replay a request whose execution may have started.
    with request(endpoint, "/api/generate", body, timeout=None) as response:
        result["http_status"] = response.status
        for line in response:
            if len(line) > 8 * 1024 * 1024:
                raise ValueError("benchmark_stream_record_too_large")
            if not line.strip():
                continue
            record = json.loads(line)
            if first_record is None:
                first_record = time.monotonic() - started
            if "error" in record:
                raise ValueError("upstream_generation_error")
            for field in ("thinking", "response"):
                value = record.get(field, "")
                if value:
                    if first_content is None:
                        first_content = time.monotonic() - started
                    data = value.encode()
                    digests[field].update(data)
                    text_bytes += len(data)
            chunks += 1
            now = time.monotonic()
            if now - last_progress >= 30:
                print(json.dumps({"phase": label, "state": "streaming",
                                  "elapsed_seconds": round(now - started, 3),
                                  "records": chunks}), flush=True)
                last_progress = now
            if record.get("done"):
                result.update({"outcome": "complete",
                               "done_reason": record.get("done_reason"),
                               "wall_seconds": now - started,
                               "first_record_seconds": first_record,
                               "first_content_seconds": first_content,
                               "stream_records": chunks,
                               "synthetic_output_bytes": text_bytes,
                               "synthetic_output_sha256": {
                                   field: digest.hexdigest()
                                   for field, digest in digests.items()}})
                for name in COUNTS + DURATIONS:
                    value = record.get(name)
                    if isinstance(value, int) and value >= 0:
                        result[name] = value
                if result.get("eval_duration", 0):
                    result["decode_tokens_per_second"] = result.get("eval_count", 0) * 1e9 / result["eval_duration"]
                if result.get("prompt_eval_duration", 0):
                    count = max(0, result.get("prompt_eval_count", 0) - result.get("prompt_eval_cached_count", 0))
                    result["uncached_prefill_tokens_per_second"] = count * 1e9 / result["prompt_eval_duration"]
                result["resident_models_after"] = [
                    {k: m.get(k) for k in ("name", "digest", "size", "size_vram", "context_length")}
                    for m in metadata(endpoint, "/api/ps").get("models", [])]
                save(receipt_path, receipt)
                print(json.dumps(result), flush=True)
                return
    raise ValueError("benchmark_stream_missing_final_record")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--recipe", type=pathlib.Path, required=True)
    parser.add_argument("--receipt", type=pathlib.Path, required=True)
    parser.add_argument("--label", required=True)
    args = parser.parse_args()
    fixture = json.loads(args.recipe.read_text())
    if fixture.get("fixture_kind") != "public_synthetic_benchmark":
        raise ValueError("reviewed_synthetic_fixture_required")
    if args.receipt.exists():
        raise ValueError("receipt_already_exists")
    tags = metadata(args.endpoint, "/api/tags").get("models", [])
    match = next(m for m in tags if m["name"] == fixture["model"])
    if match["digest"] != fixture["expected_manifest_digest"]:
        raise ValueError("benchmark_artifact_changed")
    resident = metadata(args.endpoint, "/api/ps").get("models", [])
    if resident:
        raise ValueError("cold_benchmark_requires_all_models_unloaded")
    receipt = {"schema_version": 1, "fixture_kind": fixture["fixture_kind"],
               "label": args.label,
               "utc_started": datetime.datetime.now(datetime.timezone.utc).isoformat(),
               "endpoint": args.endpoint,
               "server_version": metadata(args.endpoint, "/api/version").get("version"),
               "fixture": fixture, "runs": [],
               "cold_definition": "no loaded models; OS disk page cache not flushed",
               "privacy": "public synthetic fixture only; no output text or raw errors retained"}
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    save(args.receipt, receipt)
    body = {k: fixture[k] for k in ("model", "stream", "raw", "think", "keep_alive", "prompt", "options")}
    try:
        generate(args.endpoint, body, "cold", args.receipt, receipt)
        generate(args.endpoint, body, "warm_same_prompt", args.receipt, receipt)
        body["prompt"] = fixture["alternate_prompt"]
        generate(args.endpoint, body, "warm_changed_prefix", args.receipt, receipt)
    except Exception as error:
        # Never persist exception strings; they can include backend payloads.
        if receipt["runs"]:
            receipt["runs"][-1]["outcome"] = "failed"
            receipt["runs"][-1]["failure_class"] = type(error).__name__
            if isinstance(error, urllib.error.HTTPError):
                receipt["runs"][-1]["http_status"] = error.code
        save(args.receipt, receipt)
        print("benchmark_failed_class=" + type(error).__name__, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as error:
        print("benchmark_setup_failed_class=" + type(error).__name__, file=sys.stderr)
        sys.exit(1)
