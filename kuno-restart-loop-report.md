# Kuno ~20-minute restart loop: findings

**Date:** 2026-09-26
**Host:** `marge` (Linux Mint, systemd 255), dual GPU: Intel Arc B70 `0000:2d:00.0`, NVIDIA RTX 3060 `0000:04:00.0`
**Branch / HEAD:** `codex/vashti-images` @ `6993a25` ("Bound automatic recovery and prove worker shutdown")
**Scope:** why `kuno.service` was being restarted on a ~20-minute cadence, what the GPU had to do with it, and what remains unproven.

This report records findings only. It changed no service, no configuration, and no
installed binary. Everything below was read from the journal, `systemctl show`, and
the source tree.

---

## 1. The short answer

**The 20 minutes was not a timer. It was the recovery supervisor repeatedly
succeeding at cleaning up after a fault that kept coming back.**

The installed supervisor (`ea407ac`, `kuno-recovery.timer`, 30 s tick) was working
correctly the whole time. During the loop it ran **62 attempts on 2026-09-25 and
all 62 completed, with zero failures**:

```
2026-09-25 attempts=62  completed=62  failed=0
```

So the shape of the incident is: something re-latched upstream uncertainty roughly
every 20 minutes → the next timer tick (≤31 s later) proved the runtimes stopped,
cleared the marker, and restarted both services → the service was healthy again →
~20 minutes later it latched once more.

The leading explanation for *what kept re-latching* is a **file-descriptor leak in
the gateway's NVIDIA telemetry path**, measured on this host: 1,022 open FDs
against a 1,024 soft limit, with the control listener going intermittently
unreachable and inference/metadata requests failing — which is exactly the
condition that makes Kunable to prove an upstream stopped. Each restart reset the
count to roughly 30, which is why the loop was self-limiting at a fixed period
instead of accelerating.

Each of those recovery cycles also **provoked the Intel driver** (see §4), so a
fault that was being masked and papered over every 20 minutes was also, 62 times
over, kicking the GPU.

---

## 2. Measured evidence

### 2.1 The cadence is real and tight

Interval between consecutive `Started kuno.service` events, 2026-09-25 08:00 → now:

```
     13  0m36s     <- separate loop, see §5
     10  20m18s
      7  20m19s
      5  19m48s
      4  20m17s
      4  20m16s
      4  19m47s
      1  19m49s
      1  20m21s
      1  20m15s
      1  90m44s
      1  819m02s
```

36 restarts landed in a band of 19m47s–20m21s, then the period stretched to
~1.5–2.5 h and finally to 13.6 h of quiet. The service history:

```
2026-09-25T23:18:06 → 23:37:54   19m48s
2026-09-25T23:37:54 → 23:58:11   20m17s
2026-09-25T23:58:11 → 00:18:29   20m18s
2026-09-26T00:18:29 → 00:38:49   20m20s
2026-09-26T00:38:49 → 00:58:38   19m49s
2026-09-26T00:58:38 → 01:17:48   19m10s
2026-09-26T01:17:48 → 03:25:33    2h07m
2026-09-26T03:25:33 → 04:56:17    1h30m
2026-09-26T06:41:55 → 08:57:13    2h15m
2026-09-26T08:57:13 → 22:36:12   13h39m
```

A tightening then loosening period is the signature of a **load-dependent
resource limit**, not of a fixed scheduler.

### 2.2 Every cycle was a successful recovery, not a crash

For the 23:37 instance, the full journal window:

```
23:37:50  kuno-recovery.service: Starting kuno-recovery.service
23:37:50  kuno[1375341]: kuno: automatic_recovery_started
23:37:51  systemd: Stopping kuno.service ... Deactivated successfully
23:37:51  systemd: Stopping ollama.service ... Deactivated successfully
23:37:51  systemd: Started ollama.service
23:37:51  kernel: xe 0000:2d:00.0: [drm] Tile0: GT0: Engine reset:
                 engine_class=ccs, logical_mask: 0x1, guc_id=6, state=0x289
23:37:54  systemd: Started kuno.service
23:37:59  kuno[1375630]: kuno: managed_experimental; ...
23:38:00  kuno[1375341]: kuno: automatic_recovery_completed
```

`NRestarts=0` and `Result=success`: systemd never had to respawn the gateway. The
supervisor chose to do it, and it worked every time.

### 2.3 The supervisor was otherwise idle

Between 23:00 and 23:37 the timer ticked every ~31 s and did nothing, logging
only `Starting` / `Deactivated successfully` with no `automatic_recovery_started`.
So the 30 s timer and the 20-minute period are independent: the timer polls, the
20 minutes is the re-latch interval.

### 2.4 The FD exhaustion (measured by the parallel investigation)

- Gateway reached **1,022 open file descriptors against a soft `Max open files` of
  1,024**; the old process's `/proc/<pid>/limits` and `fd` readlink histogram were
  captured on 2026-09-26 22:36 via `sudo`.
- Isolated regression on this host's NVIDIA driver: repeated
  NVML load / init / shutdown / unload leaked **one eventfd per sample — 14 → 214
  descriptors across 200 samples**.
- Holding a single initialized, serialized NVML session for the process lifetime:
  **17 → 17** across the same 200 samples, with valid memory readings throughout.
- The updated installed gateway passed 200 probes under concurrent inference:
  **30 → 31**.

### 2.5 Intel `xe` progression

```
2026-09-25  engine resets: 4
2026-09-26  engine resets: 11
2026-09-26T13:01:35  xe 0000:2d:00.0: Check job timeout: seqno=353867, guc_id=0, timestamp stuck
2026-09-26T13:01:35  xe 0000:2d:00.0: Timedout job: seqno=353867, ... in no process [-1]
2026-09-26T13:01:35  xe 0000:2d:00.0: Kernel-submitted job timed out
2026-09-26T13:01:35  xe 0000:2d:00.0: *ERROR* CRITICAL: Xe has declared device
                              0000:2d:00.0 as wedged.
2026-09-26T13:01:35  xe 0000:2d:00.0: device wedged, needs recovery
2026-09-26T13:18:51  xe 0000:2d:00.0: Xe device coredump has been deleted.
```

`in no process [-1]` is worth keeping: at the moment the job timed out, **no
userspace process owned it**. That argues against "a live inference was spinning"
and toward a driver-level hang (or a killed process leaving a job in the ring).
The B70 is still wedged; `Vulkan` enumeration now exposes the RTX 3060 and
`llvmpipe`, and Kuno's PCI identity check correctly refuses to attach a worker to
a device it cannot verify.

Zero `Engine reset` / GuC timeout lines since the current binary started at
22:45:23 — but the device is wedged, so it has not been exercised.

---

## 3. Causal chain, with confidence graded

| # | Claim | Grade |
| --- | --- | --- |
| 1 | The gateway was restarted on a ~20 min cadence, 36 times in a 19m47s–20m21s band | **Measured** |
| 2 | Every restart was a deliberate, successful supervisor recovery; systemd never respawned it | **Measured** |
| 3 | The 30 s timer did not cause the 20 min period; it polls | **Measured** |
| 4 | Each recovery cycle coincided with an Intel `xe` engine reset during runner teardown | **Measured** (coincidence in time; the reset occurs inside the stop window) |
| 5 | NVML leaked one eventfd per telemetry cycle on this driver, and the gateway hit its FD ceiling | **Measured + reproduced in isolation** |
| 6 | FD exhaustion is what made requests fail and re-latch uncertainty | **Inference** — strongly supported (the failure mode was observed: 502s/timeouts, control listener unreachable, catalog Ollama healthy) but not provable per event |
| 7 | The exact 20-minute quantization | **Unexplained** — two tight clusters (≈19m48s and ≈20m18s, ~30 s apart, i.e. one timer period) suggest a periodic actor or a quantized leak rate; not established |
| 8 | 62 driver-provoking teardown cycles contributed to the B70 wedge | **Plausible, unproven** — see §6 |

The reason grade 6 cannot be raised: Kuno deliberately records no request IDs, no
state transitions, and no content, so a specific latching request cannot be
attributed after the fact. That is a deliberate privacy trade
(`docs/kuno-spec.md:83-113`) and this is its cost. Closing that gap needs
content-free *state-transition* logging (a fixed code per transition, no request
identity), not request content.

---

## 4. Why it looked like a "GPU loop"

Two independent things were true at once and got conflated:

1. **A resource leak in the NVIDIA telemetry path** (`apps/kuno/src/gpu.rs`) was
   the thing re-latching the gateway every 20 minutes.
2. **The Intel `xe` driver was being reset** during the recovery teardowns
   (`Engine reset: engine_class=ccs` at 23:37:51, 15 engine resets across the two
   days), and by 13:01:35 on 09-26 the B70 had wedged itself.

So the GPU was both a victim (its teardown was being exercised 62 times) and,
separately, the component that eventually failed. Fixing the leak removes the
*pace*; it does not repair a wedged GPU, and it does not explain the driver resets.

---

## 5. A second, distinct loop (do not conflate with the above)

On 2026-09-26 22:36–22:45 there were 13 restarts at a **0m36s** interval — a
different failure with a different cause:

```
22:36:48  kuno[1910456]: kuno: worker_selected_gpu_mismatch
22:36:48  kuno.service: Main process exited, code=exited, status=1/FAILURE
22:36:52  kuno.service: Scheduled restart job, restart counter is at 1
22:37:01  kuno[1910088]: kuno: automatic_recovery_failed recovery_step_failed
22:37:01  kuno[1910088]: kuno: listener_not_owned_by_expected_service
```

With the B70 wedged, the configured Intel worker could not verify its PCI
identity, the gateway **exited 1**, `Restart=on-failure` fought the recovery
timer, and recovery then failed ownership verification. The uncommitted
`apps/kuno/src/pool.rs` change (continue after proving the failed worker stopped,
retain healthy workers, keep the listeners up) addresses exactly this. It is
installed and working — the service has been up since 22:45:23 with `NRestarts=0`,
logging `managed_worker_startup_unavailable device=0000:2d:00.0
reason=worker_selected_gpu_mismatch` and continuing to serve on the RTX 3060.

---

## 6. Open risks

**The B70 is still wedged and nothing here fixes it.** Recovery requires a driver
rebind or a reboot. That was deliberately not attempted: Xorg and OpenCode held
the Intel device open at the time.

**`6993a25` widens the blast radius of worker teardown, and that is my change.**
Before it, only drained-response failures reaped a worker. It additionally routes
header timeouts, transport failures, and body errors/stalls into
`Worker::stop()`, which sends `SIGKILL` to the whole owned process group. Since
`Engine reset: engine_class=ccs` is observed *inside* the teardown window, more
proven teardowns plausibly means more driver resets. The correctness argument is
sound (an unproven group must not release a device — see §7) but the driver-cost
tradeoff is real and is the strongest remaining suspect for the `xe` behaviour.
`SIGKILL`-based cleanup itself predates `6993a25`.

**The FD leak is mitigated as well as fixed.** `LimitNOFILE` is now `524288` and
`DefaultLimitNOFILE` reports `524288`, while the process that hit the wall had
`1024`. Kuno's unit generator sets no FD limit, so the ceiling came from the
manager. Raising it delays exhaustion; only the persistent NVML session stops the
leak. Unresolved: what changed the manager default (see §8).

---

## 7. What is in place now

Committed in `6993a25`:

- `Worker::stop()` signals the owned process group, **proves the group empty while
  the leader PID is still held** (so the PGID cannot be recycled), and only then
  reaps. Every caller either completes the worker/device reservation or leaves it
  quarantined — an unproven group can no longer be followed by a reservation
  release that assumed cleanup.
- Uncertain managed completions use that abort path instead of latching.
- The supervisor charges a bounded budget before touching services.

Uncommitted in the working tree (parallel session, not mine — do not sweep these
into an unrelated commit):

| File | Change |
| --- | --- |
| `apps/kuno/src/gpu.rs` | One initialized, serialized NVML session per process; retries failed init without reloading the library; ignored FD-accounting regression test |
| `apps/kuno/src/pool.rs` | A failed worker no longer takes down all listeners; continue after proving its group stopped |
| `apps/kuno/src/recovery.rs` | Budget v2: renewable five-starts-per-30-minute window, no permanent `manual_required` lockout, v1 migration |
| `apps/kuno/src/systemd.rs` | Three failed control-health checks for the same PID justify recovery; pending recovery no longer depends on a responsive control endpoint |
| `apps/kuno/README.md`, `docs/kuno-spec.md`, `kuno-diagnosis-report.md` | Documentation |

---

## 8. Open questions, with the command to answer each

1. **What set the 1,024 FD ceiling, and when did it become 524,288?**
   ```bash
   systemd-analyze cat-config systemd/system.conf | grep -i limitnofile
   systemctl show kuno.service -p LimitNOFILE
   git -C /home/jmicko/dev/vashti log -p --all -S'LimitNOFILE' -- apps/kuno/src
   ```
   `system.conf` on this host is untouched since 2024-04-19 with the line
   commented out, and Kuno's generator never sets it, yet the manager default
   reports 524288. Something changed it.

2. **Does the current build still leak under sustained load?** The ignored
   regression test needs root-free access to `/proc/self/fd` and a real NVIDIA
   GPU:
   ```bash
   cargo test -p kuno repeated_nvidia_telemetry_does_not_leak_file_descriptors \
     -- --ignored --nocapture
   ```
   It asserts `after <= before + 2` over 200 samples.

3. **Does a proven teardown still reset the Intel engine?** This is the decisive
   experiment for §6. It needs the B70 unbound, so it cannot be run while the
   device is wedged, and it should be run on a host where a driver reset is
   acceptable. Watch for `Engine reset` in `journalctl -k -f` while forcing
   worker stops. **Do not run this on a machine whose GPU you need.**

4. **What re-latches at 20 minutes now that the leak is fixed?** With the leak
   gone, if the cadence returns, the trigger is something else and the FD theory
   is wrong. Watch:
   ```bash
   journalctl -u kuno.service -u kuno-recovery.service -f
   journalctl -u kuno.service --since '-6h' --no-pager | grep -c automatic_recovery_started
   ```

5. **Recovering the B70** is a host decision, not a code change: driver rebind or
   reboot, with Xorg and OpenCode not holding the device.

---

## 9. Honest gaps

- No per-event proof ties any specific restart to FD exhaustion; that link is
  inference (grade 6).
- The 20-minute quantization itself is unexplained (grade 7).
- The first `xe` engine reset has no captured predecessor, so the wedge's trigger
  is unresolved.
- No live acceptance of the bounded supervisor on a *healthy* two-GPU host: the
  B70 has been unavailable throughout, so the Intel path is untested since
  2026-09-25.
- The clock on this host stepped forward ~21 h mid-investigation (NTP correction,
  `System clock synchronized: yes`). Counts and intervals above were re-taken
  afterwards and are self-consistent, but any timeline assembled from earlier
  notes in this repo may be off by that amount.

---

## 10. Addendum: deployment and verification session

Appended rather than merged, so §1–§9 stay as written. From a separate opencode
session that deployed commit `6993a25` to this host and then exercised it under
load. `50_linuxmint.conf`, `apps/kuno/src/gpu.rs`, `pool.rs`, and `recovery.rs`
are being edited in another session and were not touched here.

### 10.1 A second latch mechanism, independent of the FD leak

Candidate answer to §8 Q4 ("what re-latches at 20 minutes now that the leak is
fixed?").

`finish_uncertain` (`apps/kuno/src/gateway.rs:491`) clears admission only when
`job.abort()` succeeds, which requires a **managed** job. A request that never
received one still fails closed. A managed job requires *all* of
(`gateway.rs:564-613`):

- `classify()` == `RequestClass::Inference` (`gateway.rs:42`)
- `Pool::supports(path)` (`pool.rs:118`) — **narrower** than `classify()`
- not an unload request
- `model` present, in catalog, and `!is_remote()` (`usage.rs:22`)

So a disconnect or timeout still latches for:

| Trigger | Why it goes unmanaged |
| --- | --- |
| any `:cloud`/remote model | `is_remote()` → `pool.stop_all()` (`gateway.rs:611`) |
| unrecognized model name | `catalog_model()` miss → `pool.stop_all()` |
| missing `model` field | as above |
| `/v1/responses`, `/v1/responses/compact`, `/v1/audio/transcriptions`, `/v1/messages` | Inference-class but absent from `Pool::supports` |
| any `Catalog` / `Opaque` / `Metadata` path | not Inference |

Every one of these also calls `pool.stop_all()` (`gateway.rs:611`), so a single
unrecognized model name from any unauthenticated client on `11434` tears down
both GPU workers. Verified empirically: one `glm-5.3-flash:cloud` request took
worker `11436` from HTTP 200 to unreachable.

### 10.2 `response_header_timeout_seconds` is a total-generation cap when non-streaming

Not covered elsewhere in this report. Configured 300 s
(`/etc/kuno/config.json`; default at `config.rs:37`, no documented rationale).

Ollama's non-streaming path buffers the entire generation before writing
headers, so for `stream: false` this timeout measures **total generation time**.
Measured on this host, same model, same 1500 tokens:

| Mode | `time_starttransfer` | `time_total` |
| --- | --- | --- |
| `stream:false` | 54.41 s | 54.41 s |
| `stream:true` | 2.27 s | 9.75 s |

Non-streaming delivered zero bytes until completion — on a 1.2B model. A 35B
model on the B70 is one to two orders of magnitude slower, so a long agent turn
exceeds 300 s, trips `gateway.rs:668`, and latches if the request was unmanaged.

This **strengthens grade 6** rather than competing with it: FD exhaustion would
itself cause upstream header timeouts (no descriptors left to complete the
exchange), so FD exhaustion and this timeout are two routes into the same latch.

`stream_idle_timeout_seconds` (also 300) is the well-behaved analogue — it fires
only when no bytes arrive for 300 s, which genuinely indicates a stall.

### 10.3 Live acceptance on a healthy two-GPU host (partly fills a §9 gap)

§9 lists "no live acceptance of the bounded supervisor on a healthy two-GPU
host" and "the Intel path is untested since 2026-09-25". Partly closed, for
commit `6993a25`, at 01:17–01:30 on 2026-09-26 — before the 13:01 wedge:

- Deployed `6993a25` (`f0bb0d32…`, native-image) over the previously installed
  Sep 25 build (`be66b5e6…`).
- `ornith-1.5:35b` resident at **24.02 GB VRAM** on a managed worker. The RTX
  3060 has 12 GB, so that residency can only be the B70 — the Intel managed path
  was serving real 35B traffic.
- Throughout: `MainPID 1452455` unchanged, `NRestarts=0`, **0**
  `automatic_recovery` events, no `upstream_headers_timeout`, no
  `upstream_recovery_required`.
- Disconnect test: began a streaming generation, `kill -9` on the client after
  110 845 bytes. The next request returned **HTTP 200 in 8.3 s**. No latch, no
  restart, no journal entries. That is the managed abort path working live.

```bash
curl -sN -X POST http://127.0.0.1:11434/api/chat -H 'Content-Type: application/json' \
  -d '{"model":"<local model>","messages":[{"role":"user","content":"Count slowly from 1 to 3000."}],"stream":true,"options":{"num_predict":3000}}' > /tmp/d.out &
CP=$!; sleep 4; kill -9 $CP
curl -s -o /dev/null -w '%{http_code}\n' -X POST http://127.0.0.1:11434/api/chat \
  -H 'Content-Type: application/json' \
  -d '{"model":"<local model>","messages":[{"role":"user","content":"hi"}],"stream":false,"options":{"num_predict":8}}'
```

Caveat: this validates the `6993a25` build, **not** the currently installed
`fd707c9d…` (built 22:44 from the working-tree `gpu.rs`/`pool.rs`/`recovery.rs`
changes). That build should be re-tested with the procedure above.

### 10.4 Partial answer to §8 Q1 (the 1,024 FD ceiling)

`/etc/systemd/system.conf:68` ships `#DefaultLimitNOFILE=1024:524288`,
commented — the value shown is systemd's built-in **soft 1024 / hard 524288**.
Nothing in `/etc/systemd/system.conf`, `/etc/systemd/user.conf:46`, or
`/etc/security/limits.conf{,d}` sets it. So **1,024 was systemd's built-in soft
default, not a local override** — nothing on this host lowered it.

That makes the current `DefaultLimitNOFILE=524288` the anomaly: something
*raised* the soft default. The only candidate is the single drop-in present,
`/etc/systemd/system.conf.d/50_linuxmint.conf` (May 6 2025, Mint package).
Unverified — not read before stopping. One line settles it:

```bash
grep -i nofile /etc/systemd/system.conf.d/50_linuxmint.conf
```

If it sets `DefaultLimitNOFILE`, the process that hit the wall predates that
drop-in taking effect, and §6's mitigation note should be reworded: the raise is
a host distribution default, not a Kuno change.

### 10.5 B70 still wedged (§8 Q5 unchanged)

Confirmed still unavailable: `vulkaninfo --summary` enumerates only
`NVIDIA GeForce RTX 3060` and `llvmpipe`, and kuno logs
`worker_selected_gpu_mismatch` / `managed_worker_startup_unavailable
device=0000:2d:00.0`. No `Engine reset`, `Timedout job`, or `wedged` lines in
the last 6 h — but per §2.5 the device has not been exercised, so that is
absence of evidence, not recovery.
