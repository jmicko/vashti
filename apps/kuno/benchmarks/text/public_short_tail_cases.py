"""Fixed-public MAX8 terminal-window acceptance for a protected staged client.

This module owns no runtime or inference authority. The supplied callbacks must
use the ordinary staged API and inspect the real selected worker PID. Returned
metadata covers requests only; the caller still owns session close, erasure,
host/driver checks and actual physical recovery before accepting the candidate.
No prompt, reply, token ID, content hash or private field is returned or printed.
"""

MODEL = "kuno-qwen3.8-flash-next:iq3_s"
TOTALS = tuple(range(50, 58))
OUTPUT_LIMIT = 32
MAX_CALIBRATION = 3


def _require(condition, category):
    if not condition:
        raise RuntimeError(category)


def run_public_short_tail_cases(doc, worker_pid, expected_pid):
    """Run totals50..57 then57..50 on one real, already loaded staged worker.

    The final ordinary count1 input is excluded from the known prefix. Prefixes
    49..56 therefore exercise six complete MAX8 windows followed by every
    terminal count1..8, including a complete eighth-row terminal window.
    """
    _require(type(expected_pid) is int and expected_pid > 0,
             "public_short_tail_worker_identity")

    def request(fillers, expected=None):
        _require(type(fillers) is int and 1 <= fillers <= 128,
                 "public_short_tail_filler_bound")
        prompt = ("Public reference: " + "test " * fillers
                  + "\nWhat is 17 times 19? Reply only with the integer.")
        value = doc("/api/chat", {
            "model": MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "think": False,
            "stream": False,
            "options": {"num_predict": OUTPUT_LIMIT, "num_ctx": 512,
                        "temperature": 0, "seed": 0},
        })
        count = value.get("prompt_eval_count")
        outputs = value.get("eval_count")
        _require(type(count) is int and 0 < count < 512,
                 "public_short_tail_prompt_count")
        _require(value.get("done") is True and value.get("done_reason") == "stop"
                 and type(outputs) is int and 0 < outputs < OUTPUT_LIMIT,
                 "public_short_tail_incomplete_output")
        message = value.get("message")
        _require(isinstance(message, dict)
                 and isinstance(message.get("content"), str)
                 and message["content"].strip() == "323",
                 "public_short_tail_wrong_arithmetic")
        _require(worker_pid() == expected_pid,
                 "public_short_tail_worker_replaced")
        durations = {}
        for field in ["prompt_eval_duration", "eval_duration", "load_duration"]:
            duration = value.get(field)
            _require(type(duration) is int and duration >= 0,
                     "public_short_tail_duration_metadata")
            durations[field] = duration
        _require(durations["load_duration"] == 0,
                 "public_short_tail_worker_reloaded")
        if expected is not None:
            _require(count == expected, "public_short_tail_nonadjacent_token_count")
        return {
            "prompt_tokens": count,
            "output_tokens": outputs,
            "correct_arithmetic": True,
            "nontruncated": True,
            "same_worker": True,
            "prompt_ns": durations["prompt_eval_duration"],
            "eval_ns": durations["eval_duration"],
            "load_ns": durations["load_duration"],
        }

    calibration = []
    fillers = 32
    for attempt in range(MAX_CALIBRATION):
        first = request(fillers)
        calibration.append({"attempt": attempt + 1, **first})
        if first["prompt_tokens"] == TOTALS[0]:
            break
        fillers += TOTALS[0] - first["prompt_tokens"]
    else:
        raise RuntimeError("public_short_tail_calibration_failed")

    cases = []

    def record(value, expected, direction):
        _require(value["prompt_tokens"] == expected and expected in TOTALS,
                 "public_short_tail_inventory")
        prefix = expected - 1
        cases.append({
            **value,
            "expected_total_tokens": expected,
            "prefix_tokens": prefix,
            "terminal_window_rows": (prefix - 1) % 8 + 1,
            "known_prefix_windows": (prefix + 7) // 8,
            "direction": direction,
        })

    record(first, TOTALS[0], "ascending")
    for increment, target in enumerate(TOTALS[1:], 1):
        record(request(fillers + increment, target), target, "ascending")
    for increment in reversed(range(len(TOTALS))):
        target = TOTALS[increment]
        record(request(fillers + increment, target), target, "descending")

    _require([case["prompt_tokens"] for case in cases]
             == list(TOTALS) + list(reversed(TOTALS)),
             "public_short_tail_final_inventory")
    _require([case["terminal_window_rows"] for case in cases[:8]]
             == list(range(1, 9)), "public_short_tail_window_inventory")
    return {
        "scope": "fixed-public-known-prefix-terminal-counts-1-through-8",
        "request_checks_passed_before_session_close": True,
        "content_saved": False,
        "same_worker": True,
        "calibration": calibration,
        "cases": cases,
    }
