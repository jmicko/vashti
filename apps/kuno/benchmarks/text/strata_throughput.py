"""Content-free outer timing for an accepted, fixed-public retained-session run.

No engine/kernel profiling and no timing of generated text. The protected
runner supplies only its trusted static phase lines, then successful completion
and physical recovery receipts. Failed or incomplete runs cannot yield a rate.
"""

import math
import time


class PublicThroughput:
    def __init__(self, clock=time.monotonic):
        self.clock = clock
        self.setup = None
        self.requests = []
        self.failed = False

    def observe(self, line):
        # The caller must exclude the serialized preview before this method.
        prefix = "native_public_execution phase="
        if not line.startswith(prefix):
            return
        phase = line[len(prefix):].strip()
        if "error" in phase:
            self.failed = True
        if phase == "target":
            if self.setup is not None or self.requests:
                raise ValueError("duplicate runtime setup phase")
            self.setup = self.clock()
        elif phase == "request_preflight":
            if (self.setup is None or len(self.requests) >= 2
                    or (self.requests and len(self.requests[-1]) != 3)):
                raise ValueError("retained request timing order")
            self.requests.append([self.clock()])
        elif phase in ("last_prompt_input", "request_private_erasure"):
            expected = 1 if phase == "last_prompt_input" else 2
            if not self.requests or len(self.requests[-1]) != expected:
                raise ValueError("generation timing order")
            self.requests[-1].append(self.clock())

    def finish(self, completion, before, after):
        record = completion["record"]
        admission = after["admission"]["state"]["admission"]
        memory = after["intel"]["xe_memory"]
        if (self.failed or len(self.requests) != 2
                or any(len(r) != 3 for r in self.requests)
                or self.setup is None
                or not completion["target_owners_erased"]
                or not completion["draft_owners_erased"]
                or not record["request_scope_erased"]
                or not record["identical_public_repeat_matched"]
                or record["loaded_session_requests"] != 2
                or not record["greedy_target_verification"]
                or record["content_persisted"]
                or any(admission[k] for k in ("active", "waiting", "paused", "recovery_required"))
                or any(after["http_statuses"].get(k) != 200 for k in ("kuno", "ollama", "vashti"))
                or after["kernel_fault_counts"] != before["kernel_fault_counts"]
                or after["nvidia_compute_process_count"] != 0
                or not memory or any(m["used_bytes"] >= 160 * 1024**2 for m in memory)):
            raise ValueError("throughput requires completion, repeat and physical recovery")
        tokens = record["outputs"]
        if (not isinstance(tokens, int) or isinstance(tokens, bool)
                or not 32 <= tokens <= 512 or record["windows"] < 2):
            raise ValueError("checked delivered-token count")
        requests = []
        for start, last, erase in self.requests:
            if not all(math.isfinite(t) for t in (start, last, erase)) or not self.setup <= start < last < erase:
                raise ValueError("positive monotonic phase intervals")
            generation = erase - last
            requests.append({
                "delivered_tokens_including_eos": tokens,
                "prefix_and_preflight_seconds": last - start,
                "last_prompt_inclusive_generation_seconds": generation,
                "last_prompt_inclusive_tokens_per_second": tokens / generation,
                "request_before_erasure_seconds": erase - start,
                "request_before_erasure_tokens_per_second": tokens / (erase - start),
            })
        return {
            "scope": "accepted-fixed-public-retained-engine-throughput",
            "method": "outer-monotonic-static-phases",
            "runtime_setup_to_first_request_seconds": self.requests[0][0] - self.setup,
            "same_loaded_session": True,
            "identical_outputs_checked_in_ram": True,
            "generation_includes_final_prompt_input_and_first_prediction": True,
            "generation_excludes_request_erasure_and_final_model_close": True,
            "pure_gpu_decode_rate": False,
            "original_completion_and_physical_recovery_required": True,
            "requests": requests,
        }
