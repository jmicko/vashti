# Fixed-public native memory-containment experiment

This laboratory compares one actual Cargo-selected release executable at host
memory limits 8/52/52/8 GiB in one persistent user systemd cgroup. File-cache
charges stay in that cgroup between children rather than being reparented when
each short-lived unit disappears. It does not flush global caches, move model
weights, alter production service limits, or accept arbitrary prompts.

The explicit `--native-decode-comparison` option instead fixes all four host caps
at 52 GiB and selects scalar/native/native/scalar continuation on one frozen
binary. Its separate `fixed-public-native-decode-abba-only` scope requires an
exact fixed-public plan, typed flags and caps. Each indexed child proves its
actual host/swap/core limits and selected decode environment before execution;
receipt validation independently requires the selected mode and observed
successful generation counts. All four MAIN audits, exact scalar source counts,
public IDs and cancellation/recovery checks remain mandatory. The ordinary
memory ABBA default and strict 8-GiB correctness lane retain their own limits.
The native comparison keeps the existing 64-slot Intel defaults and changes
only the continuation flag. No Ollama baseline runs are part of this option.

The separate `--native-decode-tile-comparison` option keeps all four children
native at52 GiB and selects public decode projection tiles8/1/1/8 on the same
frozen binary. Its `fixed-public-native-decode-tile-abba-only` scope requires
exact typed tile/cap/mode vectors. Each indexed guard proves its selected tile
environment before execution, and receipt acceptance independently requires
that tile alongside observed prefill tile8 and127 native decode steps. This
option retains the same64-slot defaults, full audits and numerical tolerances.
Tile1 acceptance also requires constant completion proofs for a fresh native
continuation matching the second public output and queued tile1 cancellation at
layers0/3; these diagnostics run after generation timing/count snapshots.
The two comparison options are mutually exclusive; ordinary memory and prior
native/scalar comparisons keep tile8.

Build once **before** measuring. Keep the actual Cargo JSON transcript. Stop
other inference and source compilation, verify the intended card is free, and
run one card at a time. Choose a new durable output directory outside the Cargo
target directory; existing directories/receipts cannot be overwritten:

```bash
python3 apps/kuno/benchmarks/text/run-native-performance-lab.py \
  --cargo-artifacts /path/to/completed-release-cargo-artifacts.jsonl \
  --model /media/jmicko/faststore/ollama/blobs/sha256-89059c880c2f000b24520919d4ba2edd2a8101327c7f8ab33b2280e01addda75 \
  --card intel --run-id intel-20261006-v1 \
  --output /media/jmicko/faststore/kuno-development/native-performance-intel-20261006-v1
```

The launcher copies the hash-verified selected executable and its Cargo transcript
outside the build target. The exact frozen executable is used for all four
samples. `plan.json` pins these identities, the original model's path/device/inode/
size/mtime, the synthetic fixture/reference identities and current source HEAD.
Source HEAD at freezing is not independent proof of the source used by an older
binary: preserve the associated build/source receipt as well. No 119-GB model
payload hash or duplicate is created.

Every child executes the compiled `resident-rainwater` 75-token public prompt,
seed 424242, greedy temperature 0, 128 output selections, native partition
`[8,8,8,8,8,8,8,8,8,3]`. Model context remains203. Intel defaults remain 64 expert
slots/8-GiB static GPU cache/26-GiB reserved device limit; NVIDIA remains
16 slots/6-GiB cache/10-GiB device limit. Owned transient I/O, cancellation
checks and complete owning-page zero inspection stay enabled. All optional
optimization overrides remain unchanged from the accepted public model fixture.

The supervisor changes only its declared enclosing
`kuno-public-performance-<run-id>.service` through user systemd. Actual
`memory.max`, zero swap usage/limit and zero soft/hard core limits are checked
inside every child. The correctness wrapper stays strict 8 GiB. Both lanes share
only the model/settings/public-ID/full-audit/cancellation validator; performance
has its own actual 8/52 guard. No guard output is rewritten.

The supervisor samples cgroup current/peak/stat/events/pressure and child `/proc`
I/O every 0.5 seconds. Numeric-only telemetry is flushed to
`sample-N-cap-C.telemetry.jsonl`; counters and completed model receipt identity
are retained separately in `sample-N-cap-C.cgroup.json`. `memory.peak` is
cumulative for the whole unit; use sampled per-child current peaks with that
limitation. Counter snapshots around each cap transition distinguish reclaim
while lowering 52→8 from pressure during actual inference. Check `file`,
`workingset_refault_file`, `pgscan*`, `pgsteal*`, `memory.events.max`, memory PSI
totals and child `read_bytes`, together with actual model read/upload timings.
An OOM, missing receipt, output mismatch or incomplete cleanup rejects the
sample and ends the sequence. Raw error payloads and generated text are drained
without persistence. Completed prior samples remain inspectable after failure.
If a child exits while its process-I/O counters are being sampled, only
ENOENT/ESRCH/EACCES/EPERM are recorded as numeric `process_io_errno` with absent
counters; unexpected I/O errors still reject. Aggregate failures record an
allowlisted stage and sample index without retaining exception text.

Each sample records its selected mode, completed prefill/generation milliseconds,
127 following decode steps and derived tokens/s alongside numeric telemetry.
These per-sample observations establish neither a 10-token/s result nor a
production speed claim. Historical memory plans without mode flags retain
their explicit scalar-only schedule. Historical memory/native plans without
tile fields mean tile8; tile comparison plans require their exact new keys.
Live model receipt validation remains strict, with archived receipt
compatibility available only through the separate offline validator option.

Each child starts without a resident native model. OS file/shader caches remain
unflushed and change naturally across ABBA; this is model-unloaded cold, not
cold storage. `model_bind_ms` excludes device setup/lazy model uploads. Prefill
includes fresh initialization; `generation_ms` measures 127 following graph steps
with warm in-process weight owners. Cancellation/fresh replay/audits happen after
those internal timings. These are instrumented explanatory observations, not
production worker throughput. A true second hot full request is not implemented.
The default experiment quantifies host-limit bias; the opt-in experiments compare
native/scalar continuation or public decode tiles at one fixed host cap. None establishes matched
Ollama performance or permits production native routing. The working native
target is 10 tokens/s; new or repeated Ollama baselines are outside this work.
The 3600-second public-unit limit is operator containment, not an inference
deadline. User-systemd/cgroup-V2 support is required; no sudo is needed.

Host-only checks:

```bash
python3 -m unittest discover -s apps/kuno/benchmarks/text -p test_native_performance_lab.py -v
```
