# Prompt source placement comparison, 2026-10-20

This is an internal engine investigation against pinned MIT Strata
`fb58e0dbc8399662c0e47c76578c6e878b14f6cf`. No API, model, admission limit or
Vashti consumer changes. CPU/source evidence is not a new hardware result.

## Actual selected sources

Both selected32K paths allocate independent prompt scratch and retain their
resident expert prefix. Kuno's `Session::load_inner` passes zero extra HOST grant
to `prepare_global_sources_before_prefill`, whose actual owned prompt stages
select `attach_prompt_stage`. Dormant `prompt_loan_quote` helpers do not prove
that this serving mode borrows expert roots. Upstream's selected32K default also
disables prompt borrowing; its actual reference INFO reports11,608 resident
expert pairs and22,556MiB of expert cache.

Upstream `GgufExpertSource::mirror` uses one `sycl::malloc_host` arena containing
complete `[gate|up|down]` expert blobs. The successful reference was run with
`STRATA_VERIFY_NO_HOST=1`, so every nonresident expert has a pinned mirror. Its
unpinned file/pageable-to-stager path is inactive. Kuno likewise reads misses
from immutable registered nonlocal `HOST_VISIBLE` Vulkan allocations, not from
files or a per-request CPU copy. `Sources::stage_prompt_group` copies each
original role with `vkCmdCopyBuffer` directly into the bounded raw ring. Neither
source inspection establishes comparative physical transfer bandwidth.

Both active pipelines have eight raw slots, two FP16 dequantized images and
eight-group lookahead, releasing raw sources after dequantization. Upstream
copies one contiguous expert blob; Kuno records three role ranges with the same
total original bytes. Kuno's ordinary transfer path does not synchronize before
each copy, and its host-read boundary applies only to `MAP_READ` destinations,
not the device raw ring. Native uses group modulo8 for slots, while upstream's
counter advances only on misses; potential overlap differences require actual
measurement rather than assuming a missing whole staging strategy.

## Capacity equation and new public receipt

Native residency is sized by the actual shared MAIN admission, not the retired
uniform240-slot fixture. Source arithmetic for the selected32K target+Q2 MTP
gives the following fixed components before the actual startup baseline:

| Component | Bytes |
| --- | ---: |
| Packed dense matrices and shared head |4,166,315,008|
| Actual target private allocation quote |2,579,169,280|
| I/O and runtime bounds |603,979,776|
| Public vector pages |33,554,432|
| Maximum source pages including shared-weight source window |98,566,144|
| Additional shared Q2 MTP capacity |1,355,853,824|
|48 source-address tables |1,572,864|
| Fixed subtotal |8,839,011,328|

`executor::begin_resident` therefore selects a budget of
`24,446,985,216 - actual bytes_reserved baseline`. Only the32KiB IQ table appears
to enter MAIN before admission on a fresh client. That predicts11,998 ranked
pairs and24,445,798,400 aligned arena bytes, about757MiB more than the upstream
reference. This is a source-derived prediction, not observed native placement;
allocator telemetry is authoritative. A baseline up to32MiB would still predict
11,982 pairs, so the reviewed source does not support a major under-residency
explanation.

The explicit `--public-prefill-profile` entry now emits one
`public_prefill_capacity` JSON line after the actual `Session::load` succeeds and
before any inference. It reads the retained original `Plan`, GPU cache roots,
registered HOST owner counters and actual prepared loan-shadow state. It reports
only numeric counts/bytes and booleans, including resident pairs, admitted budget,
actual GPU backing, HOST logical/backing counts, source tables and whether a
prompt-loan shadow was really prepared. It does not invent a new plan, retain a
startup telemetry field, inspect routes/usage or query/map/synchronize the GPU.
The original checked close still runs if diagnostic collection fails. Private
entry, ordinary checks and the terminal injected-failure check do not select it.

## Timing limits and next evidence

The native public fixture stages about26.9GB across39,225 role-copy ranges. Its
one-layer native timestamp diagnostic serializes operations and excludes gaps
before each timestamp. Upstream's87ms `wait copy` is the compute queue's residual
wait on copies already issued asynchronously, not total DMA time. Its319ms
`host staging` is host issuing/waiting, not a PCIe bandwidth measurement. These
numbers cannot be divided to claim a transfer speed ratio.

Next ordinary protected profile should collect the new actual capacity receipt
alongside the existing phase/copy counters. A matched copied-byte counter on the
upstream side or a narrowly scoped DMA benchmark would still be needed to
attribute transfer bandwidth. No additional GPU load or speed claim is made by
this source-only change. The focused CPU harness compiles the actual entry and
Cold-session admission functions with metadata-only stand-ins; full worker
compilation and protected execution remain separate parent acceptance gates.
