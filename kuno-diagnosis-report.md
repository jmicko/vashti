# Kuno diagnosis and recovery report

**Date:** 2026-09-25  
**Repository state reviewed:** branch `codex/vashti-images`, commit `985b81a1f22b112253c88888dad6a5459d88bf51`  
**Scope:** Kuno service behavior, the `503 Service Unavailable` seen by Vashti, image-service authentication, and the recovery design.

## Executive summary

The current Vashti error is not a model-capacity error, an Ollama connectivity error, or a Kuno process crash. Kuno is deliberately returning:

```json
{"error":"upstream_recovery_required"}
```

The error is produced before a model is dispatched. Kuno has entered a fail-closed recovery state because an earlier dispatched upstream operation was not proven to have completed. The persistent admission journal is intentionally retained until an operator verifies that Kuno/Ollama/worker processes have stopped and clears the marker.

The safety rule is reasonable for GPU work, but the current implementation is too broad and has no automatic proof-based recovery path. A normal client disconnect, stream timeout, or incomplete Ollama response can therefore leave the entire service rejecting all future inference indefinitely. This is the main issue to fix.

There is also a separate historical startup failure: Kuno repeatedly failed when `/media/jmicko/faststore/ollama` was unavailable. That should be addressed separately from the admission recovery bug.

## User-visible symptom

Vashti reports:

```text
Ollama request failed: HTTP status server error (503 Service Unavailable)
for url (http://192.168.1.164:11434/api/chat)
```

The same request was reproduced directly against Kuno with a non-inference payload:

```bash
curl -i -X POST \
  -H 'content-type: application/json' \
  -d '{}' \
  http://192.168.1.164:11434/api/chat
```

Kuno returned:

```http
HTTP/1.1 503 Service Unavailable
{"error":"upstream_recovery_required"}
```

A request with a deliberately nonexistent model produced the same Kuno response, confirming rejection happens before model lookup or worker dispatch. The private Ollama upstream at `127.0.0.1:11435` remained healthy and returned normal `400`/`404` responses for the same payloads. Kuno's `/api/ps` returned an empty model list, and the GPU was healthy.

## Current service state

At the time of investigation:

- `kuno.service` was active with the same PID since 2026-09-23 19:36:33 CDT.
- `NRestarts=0` and `Result=success`.
- `ollama.service` was active.
- Kuno public listener: `[::]:11434`.
- Kuno control listener: `127.0.0.1:11432`.
- Displaced private Ollama upstream: `127.0.0.1:11435`.
- The installed Kuno configuration selected managed workers for the Arc B70 and RTX 3060, with native image work assigned to the RTX 3060.
- The reported 38–41 GiB service memory was predominantly read-only model-file cache. The Kuno process RSS was about 254 MiB, with zero cgroup OOM events.
- No current NVRM/Xid or kernel OOM messages matched the `upstream_recovery_required` condition.

The installed binary at `/usr/local/lib/kuno/kuno` was built at approximately 19:36, while `target/release/kuno` was rebuilt later. The exact deployed source revision should be confirmed before patching behavior; the source-level diagnosis below is consistent with the installed service's response and configuration.

## Root cause of the current 503

### Admission journal lifecycle

Kuno creates a durable in-flight marker before dispatching an admitted request:

- `apps/kuno/src/admission.rs:145-154` — `Lease::dispatched()` creates the journal marker.
- `apps/kuno/src/gateway.rs:638-642` — normal forwarding marks the lease dispatched before contacting the upstream.
- `apps/kuno/src/admission.rs:161-177` — dropping a dispatched but incomplete lease sets `recovery_required`.
- `apps/kuno/src/admission.rs:79-83` — subsequent requests are rejected with `upstream_recovery_required`.

The marker is intentionally content-free and survives a Kuno-only restart. `apps/kuno/src/journal.rs:67-95` only automatically reconciles an empty marker when the host boot identity changes, because a normal Kuno restart cannot prove that a local GPU/Ollama process stopped.

### Why a client disconnect can trigger it

Kuno ties admission to the lifetime of the upstream response body, not merely to response headers:

- `apps/kuno/src/gateway.rs:708-726` — the lease is completed only after the body finishes.
- `apps/kuno/src/gateway.rs:781-820` — if the downstream client disconnects, Kuno spawns a bounded drain of the upstream body.
- If that drain reaches EOF, Kuno calls `lease.complete()`.
- If the drain errors, stalls, or exceeds the idle deadline, the lease is dropped as incomplete and recovery remains required.

This behavior is visible in the current code and explains how a browser disconnect, Vashti timeout, VM network interruption, Ollama stream error, or image cancellation can leave the marker behind.

The exact initiating request cannot be identified from current logs. Kuno does not log request IDs or state transitions for rejected/disconnected inference requests. The image log entries at 01:15:39 and 01:36:20 are possible candidates, but they are not proof: the native image code emits similar source-location messages when a cancelled sampler returns an empty tensor. The 19:39 entry matches a deliberate disconnect/cleanup acceptance test.

## Why automatic recovery is not currently safe

Kuno's documented contract is that it owns admission through completion of the upstream operation. A closed client TCP connection does not prove that Ollama or a native image worker stopped using the GPU.

Automatically clearing the marker could allow:

- a second request to overlap work that is still running;
- duplicate generation or tool output;
- conflicting GPU reservations;
- an orphaned image child or worker;
- incorrect model residency and resource accounting.

The specification explicitly requires a successful drain to release admission, while stalled or failed drains retain uncertainty, and explicitly says not to automatically retry work that may already have executed (`docs/kuno-spec.md:203-212`). The implementation follows that fail-closed rule.

The problem is not the existence of a recovery latch. The problem is that the latch has no safe, automatic resolution path for common, recoverable disconnects. The service currently treats an ordinary client disconnect like an indeterminate process failure and provides no background supervisor that can prove the upstream stopped.

## Current recovery procedure

The supported operator recovery is:

```bash
sudo /usr/local/lib/kuno/kuno recover --yes
```

`apps/kuno/src/systemd.rs:850-877` shows that recovery:

1. stops Kuno;
2. stops the private Ollama service and runner group;
3. clears the empty admission and GPU-reservation markers;
4. restarts and verifies the private upstream;
5. restarts and verifies Kuno;
6. resumes admission.

Do not manually delete the marker. Recovery deliberately verifies service ownership and process state first. Recovery briefly interrupts inference and requires root because the normal Kuno service is intentionally unprivileged.

Useful root-level checks before recovery are:

```bash
sudo /usr/local/lib/kuno/kuno doctor --service ollama.service
sudo stat /var/lib/private/kuno/upstream-in-flight \
  /var/lib/private/kuno/host-boot.json
```

The state directory may be `/var/lib/private/kuno` because the service uses a systemd `DynamicUser`.

## Other Kuno findings

### Historical startup failure

The journal contains a real repeated startup failure around 02:56:

```text
kuno.service: Failed to set up mount namespacing:
/media/jmicko/faststore/ollama:/var/lib/kuno/models:
No such file or directory
```

The unit retries every few seconds while the path is unavailable. The faststore entry in `/etc/fstab` uses `nofail`, so a missing or late drive can leave Kuno unable to start even though the root filesystem is healthy. This is separate from the current admission latch but should be tested across reboot and drive-unavailable scenarios.

### Public listener liveness

During one investigation window, the public Kuno listener accepted connections but returned no bytes for repeated metadata requests, while the control listener responded immediately and the private Ollama upstream responded in milliseconds. The same Kuno PID later recovered without a restart. This suggests a transient request-forwarding/liveness problem in addition to the persistent recovery latch.

Relevant code includes:

- detached forwarding at `apps/kuno/src/gateway.rs:537-542`;
- the 300-second upstream-header timeout at `apps/kuno/src/gateway.rs:643-652`;
- absence of a service watchdog at `apps/kuno/src/main.rs:378-406`.

This needs a separate root/thread-dump capture if it recurs. It was not the cause of the currently reproduced 503, which is explicitly identified by the response body.

### Image backend log messages

The messages:

```text
image_backend_failure diffusion_engine.cpp:2662
image_backend_failure image.cpp:897
```

are generated by the native image error adapter when the sampler returns an empty tensor. Cancellation can produce the same signal as a real backend failure because the native shim discards the detailed message and retains only allowlisted source locations. The current logging is not sufficient to distinguish those cases for operators.

## Authentication/API design finding

Ordinary Ollama-compatible routes on the public Kuno listener are open and do not require the Kuno control token. The custom image route is the exception:

- `apps/kuno/src/gateway.rs:191-198` protects `/kuno/v1/images/generate` and image previews with the control bearer token on the compatibility listener.
- `apps/kuno/src/gateway.rs:201-213` protects the loopback control listener.
- `apps/kuno/src/main.rs:307-310` requires a control token to start the service at all.
- Vashti's image integration requires a stored token before advertising or using image tools.

This does not match the intended product model. Kuno should expose ordinary Ollama inference and image data-plane routes without requiring users or browsers to manage an administrative key. The control API can remain authenticated and loopback-only. The public image route should be governed by the same network exposure and queue/resource policy as ordinary Ollama, without exposing the privileged control token to the Vashti UI.

This change should be made alongside capability/readiness discovery. A host that is already configured for ordinary inference should be able to report whether image workflows are available without a second browser-facing credential.

## Vashti error-reporting gap

Vashti currently calls `error_for_status()` and formats the generic reqwest error:

- `apps/vashti/src/ollama/client.rs:56-68`
- `apps/vashti/src/chats/handlers.rs:785-801`
- `apps/vashti/src/private/handlers.rs:359-369`

Kuno's useful JSON error body is discarded. The UI should preserve and display the structured code, especially `upstream_recovery_required`, and offer a clear recovery/diagnostic action rather than making the user infer that the model is broken.

## Recommended design work

1. Separate ordinary client cancellation from uncertain process failure.
2. Give every admitted request a correlation ID and track its owned upstream operation.
3. On downstream disconnect, actively cancel the owned Ollama/native operation and wait for a definitive stopped/reaped state.
4. Automatically clear admission only after that proof; retain the latch when the upstream cannot be proven stopped.
5. Add a background recovery supervisor with bounded retries and a health/status endpoint.
6. Expose recovery state, last uncertain operation class, and whether automatic recovery is safe/waiting.
7. Add request-state logging without prompts, message content, images, or other inference data.
8. Add tests for:
   - client disconnect before headers;
   - client disconnect during streaming;
   - Ollama timeout and body error;
   - native image cancellation and forced child termination;
   - Kuno restart during cleanup;
   - automatic recovery only after verified worker shutdown;
   - no duplicate execution or overlapping GPU reservations.
9. Make the public image data plane keyless like ordinary Ollama while retaining authenticated loopback control operations.
10. Test boot and faststore-unavailable behavior separately from runtime recovery.

## Bottom line

The immediate 503 is a deliberately persisted `upstream_recovery_required` state, not a mysterious crash. The safety decision is sound in isolation, but ordinary chat disconnects currently have no automatic safe resolution path, so one uncertain request can take the whole inference service offline.

The highest-value fix is not “clear the marker automatically.” It is to make upstream cancellation and worker shutdown observable and provable, then automatically recover when proof succeeds while preserving fail-closed behavior for genuinely uncertain GPU work.

**Changes made while producing this report:** only this report was added. Kuno was not restarted, reconfigured, or otherwise modified.

## Follow-up implemented (2026-09-26)

Recommendations 1, 3, 4, 5, and 6 are now implemented, with the safety rules intact:

- A managed request that cannot prove upstream completion stops its owned worker
  process group and proves that group empty before the worker/device scope is
  released. A group that cannot be proven empty keeps its scope quarantined
  instead of reusing a device a runner may still hold.
- The automatic supervisor has a bounded budget: five attempts per episode and
  per boot with escalating backoff from 30 seconds to 30 minutes, recorded in the
  content-free `/var/lib/kuno-setup/auto-recovery.json`. It stops instead of
  restarting a failing gateway indefinitely, and `kuno recover` resets it.
- Automatic attempts are justified only by latched uncertainty or by an earlier
  interrupted attempt; a manual `systemctl stop` is never undone.
- `kuno doctor` reports the budget mode, remaining attempts, next retry, and
  whether an attempt is pending.

Still open: request correlation IDs, structured error propagation in Vashti
(`upstream_recovery_required` is currently flattened to a generic message), the
separate `faststore` startup failure, and the remaining image-API items. No live
host acceptance of the bounded supervisor has been performed yet.
