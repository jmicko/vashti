# Verified audit of the restart-loop report

2026-09-26, America/Chicago (CDT, UTC−05:00). Read-only investigation;
no inference requests, service changes, or driver resets were made in this audit.
The original `kuno-restart-loop-report.md` is preserved unchanged.

## Strongest additional finding: the morning retry cadence

Clark's preserved attempt records explain the approximately 20-minute cycle:
about 310 seconds spent on an unsuccessful request, followed by 900 seconds of
client retry backoff. Error timestamps coincide with the Intel engine resets.

Evidence is on Clark under
`/home/clark/dev/werd-ornith-35b/.chuggin/cycle-000335/`.
The timestamps below are file mtimes for matching `connection-NNN.json` and
`provider-error-NNN.json`, not embedded request timestamps.

| Sequence | Connection trace written | Provider error written | Elapsed | Saved retry delay |
| --- | --- | --- | --- | --- |
| 025 | 09:12:38.198 | 09:17:48.598 | 310.400 s | 900 s |
| 026 | 09:32:49.175 | 09:37:59.555 | 310.380 s | 900 s |
| 027 | 09:53:00.147 | 09:58:10.576 | 310.429 s | 900 s |
| 028 | 10:13:11.155 | 10:18:21.481 | 310.326 s | 900 s |
| 029 | 10:33:22.093 | 10:38:32.362 | 310.269 s | 900 s |

The actual inter-attempt waits were about 900.6 seconds. The records preserve
only the generic HTTP-5xx classification, `Provider temporarily unavailable`;
they do not preserve the exact HTTP code or internal Kuno error. Clark's request
timeout was 1,800 seconds, so that limit was not expiring after 310 seconds.

Clark source confirms the 900-second cap (`chuggin/src/provider.rs:85`) and the
connection/error trace placement (`chuggin/src/model.rs:211`, `:343`, `:371`).
Kuno has several 300-second bounds: configured response-header and stream-idle
timeouts, plus the managed worker HTTP client's timeout. A five-minute timeout
plus setup/cleanup is a strong lead, but which stage failed remains unproven.
Clark's workload used streaming; the report's non-streaming experiment therefore
cannot by itself identify this incident's failing stage.

This establishes the morning cadence without assuming a fixed descriptor-leak
rate. The NVIDIA leak was independently reproduced and fixed, but neither that
fact nor these timestamps proves the cause of every historical request failure.

## Journal checks

Fresh queries of the system journal confirmed:

- September 25: 62 automatic recovery starts and 62 completions.
- September 25: four recorded Intel engine resets. Each occurred approximately
  0.4 seconds after an automatic recovery began. This is meaningful correlation,
  but contradicts claiming a recorded Intel reset for every recovery.
- September 26: eleven recorded engine resets. The repeating morning resets
  occurred while the gateway process remained running from 08:57:13 until
  22:36:12. Whole-gateway restarts cannot explain those individual resets.
  Worker teardown within that process remains a plausible explanation.
- Intel scheduling failure was recorded at 12:00:13; the device was declared
  wedged at 13:01:35. These establish the driver failure sequence, not its initial
  application-level trigger.
- Across the examined events, the realtime-minus-monotonic clock offset varied
  by less than one millisecond. They do not substantiate the claimed 21-hour
  clock jump. No timeline correction should be applied on that claim alone.

## Corrections to other report claims

1. **The descriptor limit was not shown to have increased.** The current process
   and `systemctl` both report soft 1,024 / hard 524,288. `LimitNOFILE` alone
   reports the hard limit; `LimitNOFILESoft` reports the effective soft limit.
   `/etc/systemd/system.conf.d/50_linuxmint.conf` sets a shutdown timeout, not a
   descriptor limit. The running NVML fix removes the leak; raising the limit
   was not part of the deployment. The original exhausted process's FD count
   was captured, but it restarted before its descriptor-target histogram could
   be captured. The report overstates that historical snapshot.
2. **Metadata requests do not invoke the described teardown/latch path.** In
   `gateway.rs`, metadata skips both admission leases and `pool.stop_all()`.
   Unmanaged inference and opaque/catalog paths are different cases.
3. **A successful request after a client disconnect does not prove an abort.**
   The gateway first drains the abandoned response. Successful drainage gives
   the same visible result without killing a worker. Worker-PID or explicit
   lifecycle evidence is needed to identify which branch ran.
4. **The cleanup history is misstated.** Before `6993a25`, failed drains retained
   uncertainty rather than directly aborting the worker. SIGKILL-based cleanup
   already existed for other worker-management operations. The newer commit
   added direct abort handling for additional uncertain-request paths.
5. **The 300-second buffered-response concern is valid but not a complete
   diagnosis.** It may include all generation time before headers arrive, but
   the small-model timing experiment does not establish 35B throughput. Lack of
   response bytes alone also does not prove GPU work has stalled.
6. **The supervisor delay is not guaranteed to be at most 31 seconds.** Its
   interval is measured after the previous helper stops and also includes
   randomized delay, timer accuracy, and the helper's execution time.

## Consequence for the next investigation

Kuno's worker timeout and stop behavior is a serious suspect for the repeated
Intel resets. That is stronger than merely guessing from a twenty-minute service
restart pattern, but still does not prove why GPU work stopped completing or why
the driver ultimately wedged. The NVIDIA FD fix should not be described as a
proven solution to the Intel timeout/reset cycle.

After the Intel device is safely recovered, distinguish load/preload, prompt
processing, response-header wait, response drain, and worker shutdown. Use
content-free stage and stop-reason events plus worker PID/PCI identity and kernel
timestamps. A controlled graceful-stop versus forced-stop comparison is more
informative than another unattended loop. No user prompts, images, response
bodies, or request payloads were needed for this audit.

## Observation installed after the reboot

At 23:09 CDT on September 26, Vulkan enumeration again reported both the Intel
B70 and NVIDIA RTX 3060. At 23:18, the current boot's kernel journal contained
no Xe reset/wedge/timeout/fault messages. Both managed workers were started and
the gateway reported no resident models before instrumentation installation.

The boot log reports GuC firmware 70.44.1 while the Xe driver recommends 70.54.0.
This is a separate compatibility lead, not proof of the failure's cause. No
firmware, driver, timeout, placement, or shutdown policy was changed for this
observation run.

The installed native-image build adds `kuno_lifecycle` events covering managed
preparation, isolated/cached profile selection, reuse/preload, worker readiness,
preload headers/body, inference headers/body, disconnect drainage, abort,
SIGKILL, process-group exit proof, reap, and admission-recovery transitions.
Events contain fixed enum labels, elapsed milliseconds, optional worker PID and
validated PCI address. They contain no request IDs, model names/digests,
prompts, images, outputs, URLs, headers, or raw runtime errors.

Validation: 133 Kuno tests passed; seven hardware/native integration tests remain
ignored by the normal suite. Clippy passed with warnings denied. The expanded
subprocess privacy test checks private request/response/header/query/model/image
sentinels and the exact allowed event schema. Fake-worker tests exercise
timeouts and process-group cleanup without touching GPUs.

Build SHA-256: `829700c7daecbcba1e355aed6cdd533edf7bb8ab525856ce4485a4bfa6a1e065`.
The installer reported `kuno_upgraded_and_active`; the installed executable's
hash matches this build. Both managed workers passed readiness checks. A small,
synthetic, two-token streaming request completed with HTTP 200 in 4.083 seconds
on B70 worker PID 22529. Its trace measured preload at 3.127 seconds, subsequent
upstream-header wait at 0.889 seconds, and normal response-body completion.
No generated content was printed or saved by this verification.

The installation itself supplied an additional observation: systemd began
stopping the previous Kuno process at boot-relative +964.843330 seconds, and
the B70 logged a CCS engine reset at +965.441566 seconds, about 0.598 seconds
later. This precedes the new build's startup at +975.331208 seconds. The GPU
recovered and passed the above request without a further reset. No wedge,
timeout or fault appeared in the checked kernel events. This strengthens the
shutdown/reset correlation, but does not establish why repeated resets later
became a permanent wedge. The small-model check does not certify Clark's larger
workload or overnight stability.

The live journal command is documented in `apps/kuno/README.md` under
privacy/diagnostics; use boot-relative (`short-monotonic`) timestamps to correlate
with kernel events. Logging remains enabled without a terminal session.

## Clark recurrence: first-response timeout identified

Around midnight September 26–27, Clark reached `/api/version` with HTTP 200 in
3.1 ms. Kuno remained the same live gateway process, PID 22483. Its trace
identified three failures of the **inference response-header wait**, after
successful model preloads of roughly 7–8 seconds:

| Header wait began (boot seconds) | Timeout / SIGKILL | Intel engine reset |
| --- | --- | --- |
| 1117.819753 | 1417.821386 | 1418.404995 |
| 2336.280955 | 2636.282166 | 2636.857423 |
| 3016.976810 | 3316.978325 | 3317.551241 |

The first two resets follow explicit worker SIGKILL by approximately 0.58 seconds.
The header deadline, not a stalled model load or network-connectivity failure,
triggered these aborts. Clark's own request timeout was 1,800 seconds; its saved
errors collapse HTTP 5xx responses into a generic provider-unavailable message.

The current request has 181 messages and a 128,000-token context setting. Earlier
successful calls reported about 84,000 prompt tokens. At boot +3194.05 and
+3204.35 seconds, the B70 compute engine was 100% busy at 2,800 MHz; its idle
counter did not advance, energy consumption averaged approximately 216.5 W, and
the runner's CPU counters advanced by 6.44 CPU-seconds. This is activity while
the header wait is pending, not proof of useful kernel progress. Long cold
prompt processing remains the leading explanation until a complete request
can be observed beyond the old deadline.

The compatibility fix disables inference response-header and response-body idle
deadlines by default (`0`). Explicit administrator deadlines remain available.
Disconnect drainage likewise has no implicit total-duration cutoff; an opted-in
idle deadline resets on received frames. Queue, connection, request-upload,
worker-load and image-runtime limits are separate. A future stall detector must
not infer a hung backend merely from elapsed time or unavailable telemetry.

Ollama v0.34.2's `server/routes.go` constructs its HTTP server without read/write
timeouts. Its `envconfig/config.go` identifies `OLLAMA_LOAD_TIMEOUT` as model-load
stall detection; that is not a generation-response limit. The user explicitly
rejected replacing the five-minute response cutoff with another fixed default.

Prepared native build SHA-256:
`09d263c9ad44ee16ccf2da4cd422cee465199a09c0c9c71b576df466b44d8dd4`.
All 143 Kuno tests passed; Clippy passed with warnings denied. This includes
delayed headers, gaps between streamed chunks, disconnected response drainage,
explicit finite deadlines, and transactional configuration preservation.
Installation reported `kuno_timeout_configuration_active`; the installed hash
matches this build and both configured response limits are `0`. Managed worker
and image settings were retained.

### Full-context live acceptance passed

From Clark, replayed the same saved 181-message request with the same tools,
model, 128,000-token context setting, and streaming mode, changing only the
output limit to two tokens. The request stayed on Clark and was sent over HTTP;
the probe discarded generated content and printed only operational statistics.

- Start: 2026-09-27 05:08:18.375 UTC.
- HTTP 200 / first response bytes: 356.073 seconds after the client started.
- Completed successfully at 356.127 seconds.
- Prompt evaluation: 88,382 tokens in 347.651 seconds; output: two tokens.
- Kuno's exact header-wait span: 348.077 seconds, exceeding the former 300-second
  deadline. Model preparation was a separate 7.883 seconds.
- Same B70 worker PID 50293 and runner PID 51089 throughout. No new Xe reset,
  timeout, fault, or wedge during the run. Gateway PID 50247, zero automatic
  service restarts.

Read-only numeric snapshots from the owned runner's `/slots` endpoint also
showed `n_prompt_tokens_processed` increasing from 69,798 to 76,454 to 85,080,
with zero prompt-cache tokens during this cold run. The endpoint's
`n_prompt_tokens` field grew with scheduled batches; it was not a fixed total.
The completed response's `prompt_eval_count` supplies the final 88,382 count.
Full slot responses were never logged: they can contain generation parameters
and prompt content, so only an explicit numeric/boolean allowlist was printed.

This demonstrates useful inference continuing beyond five minutes and completing
successfully once Kuno's cutoff is removed. It does not certify every future
backend hang or establish the deeper driver defect behind resets after SIGKILL.
The user requested a retry in Chuggin. Its normal agent then resumed successfully:
cycle 339 responses 004, 005 and 006 completed at 05:15:54, 05:16:09 and 05:16:20
UTC, with backend totals of 10.368, 14.988 and 11.284 seconds. They reported
89,132/89,600/90,172 prompt tokens respectively, with 88,716/89,251/89,805 cached
tokens. A subsequent response was streaming. Kuno reused the same B70 worker,
with no new provider-error trace, service restart, or Intel reset during these
checks. These are real Chuggin calls, separate from the two-token replay test.
