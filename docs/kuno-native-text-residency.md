# Resident Rust graph / private I/O review

Read-only source audit, 2026-10-05. No device actions, real model payload reads,
service changes or production edits. BoundGraph implementation is
`ce2bea0cfb8868ae5efd7dc167eeb8add9585846` (parent integrated as `2467ac7`).
Residual/arena module receipt is below. Existing public-fixture GPU acceptance
belongs to the root task; this document does not claim production privacy.

## Smallest next graph interfaces, in order

1. `ResidentSequence::new(BoundGraph, DeviceEpoch, context_limit, budget)` owns a
   PrivateArena plus opaque GDN/conv/QSA/PLE buffer IDs, CPU cursor and independent
   PLE history. Bind to `GraphBindingId` and device epoch. Reject invalid context
   or insufficient TOTAL padded budget before consuming public-cache memory.
   The scalar Sequence is never inspected/mutated. `step(token,cancel)` preserves
   scalar input validation, then marks busy/poisoned before its first mutation.
   Once any operator mutates, errors/cancellation discard/erase the WHOLE
   resident sequence; no replay/fallback of a partially executed token. Commit
   cursor/history only after complete finite/device-success acceptance.
2. `ResidentProjection::enqueue(bound TensorView, expert, input ID, output ID)`
   validates same model/device/arena epochs and exact logical rows; it enqueues
   the existing packed matvec without host vector materialization/readback.
   Read source unchanged before cached graph execution. Public cache owns only
   original weights. It charges actual cache pages after reserving private spans.
3. Use the four residual kernels for attention and FFN, preserving ORIGINAL R.
   Final output mixer has no injection/combine. Static folded gamma may live in
   public cache; runtime private activation/status buffers belong to arena.
4. Resident GDN: raw causal conv(10240 channels, kernel4,dilation1); SiLU; Q/K
   split 2048/2048 and value6144; L2 eps1e-6, Q/sqrt128; decay using stored A and
   dt; beta sigmoid; exact h%16 recurrence. Initially old/next state are distinct
   buffers. Follow with per-value-head RMS128 and sigmoid z, then output matvec.
   State persists across tokens only inside this request, reset/drop erases it.
5. Keep PLE hashing and token/embedding row reads on CPU initially. Borrow
   `BoundGraph::ple_hash()` and make per-request PleTokenHistory. Upload only one
   selected decoded embedding/16 bounded lexical rows. Their selection AND
   the resulting activation are private: recognizable model rows can reveal
   token IDs even though the original full model table is public.
   Port projected PLE via per-stream RMS, signed-sqrt dot gate, normalized gated
   value conv(kernel4,dilation3), R+gated+SiLU(conv), before layer1 attention.
6. Resident QSA separately: headwise interleaved Q+gate split24x512; direct
   gamma RMS; text NeoX RoPE first64; append causal KV/raw indexer K. Pool ONLY
   complete blocks of4 raw indexer keys BEFORE norm+RoPE at first block position;
   deterministic score/top-block order and exact incomplete tail. Use query
   head/12 for KV association, scale1/sqrt256, stable softmax, sigmoid output
   gate. Visible cursor is request-owned, never merely full allocated capacity.
7. Router can first read 10 IDs/probabilities/status per layer. Protect all
   selected expert banks through use and forbid eviction/replacement before
   queued consumers finish. Upload failures invalidate physical bank mappings.
   Use global router softmax, top10 renormalization and the separate shared
   expert (SiLU gate*up, down, sigmoid(dot(shared_gate_weight,x))). Do not log
   selected IDs/probabilities or persist LRU/access traces. Sampling initially
   returns only greedy token+status with ascending-ID tie rule.

Numerical acceptance: tiny independent operator fixtures first, then fixed
public multi-token core-oracle checkpoint/logit parity, then device reset/
cancellation/cleanup tests. This is a separate lab runner, not a broad rewrite
of model.rs or a production API decision.

Padded memory warning: at context4096, 36 GDN layers with old+next state use
72*4MiB=288MiB (logical state was 216MiB). Their conv history adds4.5MiB.
12 QSA layers use keys8MiB+values8MiB+indexer keys2MiB+positions32KiB each,
216.375MiB total. PLE conv history is0.5MiB. Thus ~509.375MiB before activation,
fault/status/routing/sampling scratch, transfers, or public weights/banks.
Positions could later be derived from scalar text cursor, but preserve the
initial explicit oracle shape. No hidden staging pool is included in this sum.

## Exact current runtime ownership and gaps

Source root:
`/home/jmicko/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/`.

- `cubecl-runtime-0.10.0/src/client.rs:287-300` creates a Vec from input;
  `212-243` clones it again into Bytes before queued server write. Erasing the
  caller's padded Upload does not erase these ordinary host copies.
- `cubecl-wgpu-0.10.0/src/compute/stream.rs:102-122` flushes prior encoder work
  before a scheduled write. `378-419` sends bytes with Queue::write_buffer or
  write_buffer_with; these produce another opaque native upload stage.
- `wgpu-core-29.0.4/src/resource.rs:1111-1217` owns mapped MAP_WRITE|COPY_SRC
  upload memory, then FlushedStagingBuffer. `1235-1242` only destroys its raw
  buffer on Drop. PendingWrites retains these buffers through submission
  (`device/queue.rs:379-381,1419-1469`), and completed-submission triage drops
  encoders/resources (`device/life.rs:298-324`). Destruction is NOT erasure.
- Readbacks use a distinct MAP_READ|COPY_DST pool
  (`cubecl-wgpu ... /mem_manager.rs:50-63,98-109`). They copy source to a pooled
  stage, flush, map the WHOLE buffer, and return a pinned Bytes controller
  (`stream.rs:137-229`). Logical Bytes length can be smaller than full page.
  Controller Drop drops view then unmaps; it does not zero
  (`controller.rs:13-23`). An abandoned read future also needs a cancellation
  guard to unmap/abort pending mapping, not merely drop callback receivers.
- Uniform storage is another ExclusivePages pool with UNIFORM|STORAGE|COPY_DST
  (`mem_manager.rs:65-81`). `reserve_uniform:117-131` retains handles until
  flush; `stream.rs:491-494` releases them after queue.submit, BEFORE GPU done.
  They may be reused; their bytes/padding are not erased. Schedule metadata Vec
  itself is dropped after creating uniform (`compute/schedule.rs:119-133`).
- `mem_manager.rs:135-137` memory_cleanup covers only MAIN storage. It never
  sanitizes staging or uniform pools. Their default ExclusivePages bins can own
  larger hidden pages than logical Handle sizes. Main power-of-two bins solve
  ONLY the arena-owned spans, not these other pools.
- `storage.rs:93-120` tracks full wgpu Buffer objects; dealloc just removes a
  Buffer reference. A buffer/map/queue/context Drop does not prove zeroing.
- Scheduler has stream access/alignment but no reviewed public selective
  teardown (`cubecl-runtime ... /stream/scheduler.rs:112-174`). A WgpuStream
  owns ALL three pools. Removing it loses public weights in that stream and
  still does not prove byte erasure. Do not substitute stream/context Drop for
  an explicit sanitizer.

## Preferred minimal feasible Rust runtime patch (proposal, not implemented)

Pin/vendor cubecl-runtime and cubecl-wgpu 0.10; preserve licenses and exact
source hashes. No foreign model engine or native graph is required. Keep the
public main-weight cache untouched. Add a checked, stream-local
`sanitize_transient_io()->Future<Result<ErasureReceipt>>`; unrelated streams
must not enqueue I/O during it, and caller must resolve/drop all readbacks.
This hook alone is insufficient unless uploads become explicitly owned.

1. Add private upload creation that MOVES a guarded Bytes allocation into the
   scheduled write without `to_vec` clones. A controller wipes full host
   allocation on Drop. No file-backed controller is allowed. Ordinary public
   weight APIs can keep their original behavior; private calls cannot fall back.
2. For secure upload/metadata writes, avoid opaque Queue::write_buffer stages.
   Create owned MAP_WRITE|COPY_SRC buffers with mapped_at_creation=true,
   initialized over full actual Buffer::size, then copy and unmap. wgpu-core
   `device/resource.rs:1138-1157` directly maps such buffers; creating a mapped
   buffer WITHOUT MAP_WRITE instead makes an opaque staging buffer. Enqueue
   copy_buffer_to_buffer on the same ordered encoder/queue before consumers;
   end an active compute pass before inserting copy commands. Track each full
   stage until GPU completion. After drain, map WRITE, wipe full span, drop
   mapped view, unmap (wgpu-core resource.rs:919-943 flushes noncoherent writes),
   then recycle/release. All map/error results reach sanitizer and retirement.
3. Track complete uniform and readback storage buffers, INCLUDING retired pages
   removed from allocator maps before final cleanup. A retained full Buffer
   object is needed until erasure receipt, not only a logical resource/slice.
   Add live-readback leases held across read future AND returned controller;
   cancellation guards abort/unmap pending maps. Sanitizer rejects outstanding
   leases instead of unmapping memory still borrowed by callers.
4. After drain/unmap, encoder.clear_buffer(full_buffer,0,None) clears every
   full readback/uniform page (both already COPY_DST); submit and wait for
   completion. Do not clear by last logical size. Release retired pages only
   after checked completion. Normal public-cache handles are unaffected.
5. Secure upload stages/retired pages need a separate checked total budget.
   Reclaim completed batches at existing router/readback fences or when bounded
   capacity requires it; retaining every token's uploads until EOS is unbounded.
   Errors must retire the shared device context and poison/close the sequence;
   clean failure has no silent fallback or success receipt.

Initial cleanup order: reject new sequence enqueues -> resolve/drop readbacks
and cancellation futures -> drain -> arena full main-span erasure -> sanitizer
owned upload/readback/uniform erasure -> checked final sync -> release buffers
and wipe CPU request scratch. Public weight cache may survive ONLY successful
cleanup, with no persisted request-derived expert/LRU/profile histories.
Keep swap/core dumps/trace disabled for the worker. Request tokens, positions,
route IDs/probabilities must be runtime GPU-buffer data, never #[comptime]
constants that could land in generated shader caches or debug source.

An alternative patch in wgpu-core's FlushedStagingBuffer Drop could zero raw
uploads after completed-submission retirement, but Drop has no checked return,
failure/cancel paths require separate proof, and it still leaves CubeCL host,
readback and uniform copies. It is not the preferred single fix.

Needed acceptance: owned buffers inspected after cleanup but before release;
padding/retired-page cases; readback retained/future-dropped cases; map/copy/
drain/erasure fault injection; no reuse after retirement; public packed weights
still valid and resident; live parity on both GPUs. This source proposal does
not establish any of those device receipts.

## Implemented module receipts

- `/tmp/kuno-rust-gpu-probe/src/residual_gpu.rs`
  SHA256 `97516841640041f66359851adceff5866d44f7c2955c7df9de8d3f906e9f24f3`.
  Four host tests and tiny/actual offline WGSL kernels pass with -D warnings.
- `/tmp/kuno-rust-gpu-probe/src/private_arena.rs`
  SHA256 `c596148f6f40afeec9185a576e78bce09db62163209312bc7154fb64f0f814ec`.
  Seven host tests/offline eraser WGSL pass. Required custom main pool config
  is ExclusivePages bins2^15..2^27. No GPU/private staging acceptance claimed.
- BoundGraph commit: all74 ordinary core tests pass; two installed-artifact
  tests remain ignored; fmt and all-target Clippy -D warnings pass. Accessors
  expose only admitted tensors/static vectors, exact geometry/borrowed PLE hash
  config, opaque binding identity and source-unchanged checks. No Sequence state.

## Connected owner and remaining CPU metadata gap

The public laboratory connects transactional GDN/QSA/all-ten MoE under one
whole-token owner; both cards pass four-token parity and cancellation after
queued layers0/3, followed by whole-arena erasure and poisoned retry rejection.
B70 also passes the eight-token chat. This establishes request-owned main
buffers and host PLE cleanup, not complete runtime staging sanitization.

An additional runtime audit found `cubecl-core0.10` scalar argument and INFO
thread-local builders retain ordinary Vec capacity after `clear()`. Slots,
probabilities and positions can pass through these runtime scalar arguments.
The minimal patch must therefore also wipe initialized and spare capacity of
scalar/INFO host builders on launch, reset, abandonment and panic. Guarded
runtime/wgpu buffers alone do not close that CPU-copy gap. An isolated pinned
three-crate patch is undergoing review and synthetic host fault tests; no
persistent private worker acceptance or device staging proof is claimed yet.
