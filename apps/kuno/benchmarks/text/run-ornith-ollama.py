#!/usr/bin/env python3
"""Two fixed-public fresh-sequence Ornith samples; retain numeric receipts only."""
import argparse
import json
import pathlib
import time
import urllib.request

HERE = pathlib.Path(__file__).resolve().parent
URL = "http://127.0.0.1:11435"


def request(route, data=None):
    req = urllib.request.Request(URL + route,
        data=None if data is None else json.dumps(data).encode(),
        headers={"Content-Type": "application/json"})
    return urllib.request.urlopen(req, timeout=900)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--receipt", type=pathlib.Path, required=True)
    args = parser.parse_args()
    fixture = json.loads((HERE / "ornith35-baseline.json").read_text())
    body = {k: fixture[k] for k in ("model", "raw", "think", "prompt", "options")}
    body.update(stream=True, keep_alive="10m")
    samples = []
    try:
        with request("/api/generate", {"model": fixture["model"], "keep_alive": 0}) as response:
            response.read()
        for phase in ("cold_worker", "warm_weights_fresh_sequence"):
            start = time.monotonic()
            with request("/api/generate", body) as response:
                final = None
                for line in response:
                    item = json.loads(line)
                    if item.get("error"):
                        raise RuntimeError("Ollama fixed-public request failed")
                    if item.get("done"):
                        final = item
            if final is None:
                raise RuntimeError("Ollama final counters missing")
            with request("/api/ps") as response:
                resident = json.load(response)
            sample = {"phase": phase, "wall_seconds": time.monotonic() - start,
                **{k: final.get(k) for k in ("total_duration", "load_duration", "prompt_eval_count",
                    "prompt_eval_duration", "eval_count", "eval_duration")},
                "resident": [{k: model.get(k) for k in ("name", "size", "size_vram", "context_length")}
                    for model in resident.get("models", [])]}
            if isinstance(final.get("context"), list):
                sample["public_context_ids"] = final["context"]
            sample["decode_tokens_per_second"] = sample["eval_count"] / (sample["eval_duration"] / 1e9)
            samples.append(sample)
            print(json.dumps({k: v for k, v in sample.items() if k not in ("public_context_ids",)}), flush=True)
    finally:
        with request("/api/generate", {"model": fixture["model"], "keep_alive": 0}) as response:
            response.read()
    receipt = {"fixture": "ornith35-baseline.json", "seed": fixture["options"]["seed"],
        "temperature": 0, "model": fixture["model"], "model_blob": fixture["expected_model_blob"],
        "samples": samples, "warmth": "weights resident, fresh sequences, possible file cache warmth"}
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    with args.receipt.open("x") as stream:
        json.dump(receipt, stream, indent=2)
        stream.write("\n")


if __name__ == "__main__":
    main()
