# Borrowed original-F32 CPU expert batches

## Scope and status

This is a source-only, default-unused prototype for one to three members of a
single original expert layer. It reuses `cpu_expert`'s explicit persistent worker
pool and original packed F32 dots. It does not select CPU execution in a model,
map a GGUF or Vulkan allocation, own/copy model weights, reroute an expert,
multiply route probabilities, or claim a speed improvement. Existing count1
entries and defaults remain unchanged.

The later frozen-source parent must authenticate original tensor/expert identity,
source epoch, actual physical spans, and immutable lifetime of each RAM weight
view. A checked raw borrowed view proves geometry and bounds; it does **not**
prove that arbitrary caller bytes belong to that expert. The parent must retain
the authenticated source owner through every joined CPU and GPU consumer.

## Typed leaf contract

The additive API is `cpu_expert::batch`:

- `PackedExperts::borrow_expert(expert, [gate, up, down])` returns an `ExpertView`
  borrowing that original weights view and three exact immutable packed expert
  spans. No reader operation or payload allocation occurs. Expert ID, all three
  full lengths, original row bytes, kinds and geometry are checked.
- `Rank::new(original_rank, &ExpertView)` and `Member::new(input, ranks)` borrow
  caller-owned inputs and selected expert headers. At execution, one to three
  members must use the same original `PackedExperts` geometry identity. Each
  member has zero to ten HOST ranks in strictly increasing original-rank order,
  with no repeated expert within that member. The same expert may appear in
  different members and at different original ranks. Empty HOST subsets are
  valid; they do not stand in for a missing GPU result.
- `ExpertOperator::execute_borrowed_batch_with_private_limit(weights, members,
  cancel, private_limit)` requires an existing pooled Portable or AVX2 operator
  using original F32 arithmetic. Q8/FMA/source-read/phase-diagnostic options
  reject explicitly;
  nothing falls back to another backend. Input H, packed shapes, count, finite
  input, unique/ranked selections, and worker scratch bounds are checked before
  private fill or computation. Cancellation uses the existing CPU operator flag.
- `BatchPartials` exposes only read-only unweighted member/rank slices and the
  full three-member, ten-rank owning output. Missing ranks and inactive members
  remain bitwise positive zero. The caller must merge with its original route
  probability and original rank0..9 order. There is no mutable, Debug, Clone or
  serialization interface.

The leaf checks the complete input/rank view before executing either phase. GU
uses the existing `dot_row` twice and existing `ops::silu(gate) * up`. DOWN uses
the existing `dot_row` on those activations. All original weight scale/address
and F32 multiply/add/64-lane reduction operations remain unchanged. There is no
integer activation quantization, FMA, probability multiplication, expert
replacement, or decoded-weight cache. Equal source spans across members are
borrowed repeatedly without copying; multi-RHS decode reuse is not implemented.

## Capacity and transient ownership

The fixed request envelopes are three H-vector input slots, thirty FF-vector
activation slots, and thirty H-vector output slots. Every call precharges all
three, every configured worker's existing 32-KiB scratch reservation (with the
existing actual `Scratch::check_capacity` bound), existing complete row-job
header ownership, and bounded request/output metadata before allocating.
Actual vector capacities are checked together before copying private input.
The original 1-MiB private cap remains the hard ceiling; a caller may lower it.
No public packed-payload reservation is needed because the leaf allocates no
weights. Caller-owned immutable weight pages are not a new leaf reservation.

The existing synchronous pooled row phases join every consumer, including
worker errors/panics, before borrowed request owners can release. Call-owned
inputs, activations, compact-to-original rank placement metadata, and the full
output allocation are erased on error or unwind. The returned output remains
alive only for its caller and erases its complete owned allocation and private
selection metadata on drop. Caller-owned borrowed inputs/headers are the
caller's responsibility. No register, opaque Rayon/OS stack, allocator-history
or process-forensic erasure guarantee is made.

## Required source and eventual model gates

Focused public-fixture tests must compare unchanged count1 output against
batched counts1,2,3 for all seven actual DASLab format triples, covering all six
routed formats: IQ2_S, IQ3_S, IQ3_XXS, IQ4_XS, IQ4_NL, Q2_0. They must retain the
independent pinned-C decoded expectations and original64 reference, exercise
the same expert across members at different ranks, sparse/all-empty/maximum
rank subsets, exact inactive zero spans, finite errors, source identity and span
errors, count/rank/expert errors, capacity admission, persistent-pool completion,
and complete transient/output erasure. Tests must not read private model data.

Source tests do not establish hybrid speed or connected GPU correctness. A
future parent first measures full layer input handoff, CPU/GPU overlap, partial
upload and original-rank merge against direct HOST GPU execution. It then must
pass unchanged full original CPU13 absolute0.006 + relative0.0003 comparisons,
whole vocabulary/hidden vectors, exact expert identities, real speculative
accept/repair, owning fault/padding/epoch audits, closure and memory recovery.
CPU F32 plus accepted GPU integer GU/DOWN does not imply identical float order.

No runtime selector, service, API, installed binary or frontend change is part
of this leaf. Compilation/tests and physical acceptance remain separate receipts.

## Source validation receipt (2026-10-11)

Implemented on integration base `0d35d5c` in the isolated
`codex/kuno-cpu-batch-leaf` branch. The six public-fixture batch tests and one
complete-owner wipe test pass together with the existing expert suite:
**53 passed, zero failed, two existing timing probes ignored**. They establish
bitwise original CPU count1/batch equality for all seven DAS triples and
counts1..3, alongside the independently compiled pinned-C decoded expectations.
Both explicit Portable and AVX2 paths ran on this host. They do not establish
CPU/GPU mixed arithmetic or model performance.

The locked CPU manifest's all-target Clippy with warnings denied passes, as does
the locked native laboratory's all-target pure `direct-device` Clippy. Checks
used `-j2`, with test concurrency2, and NVMe caches. Formatting and whitespace
checks pass. No GPU, service, sudo, model mapping, installed binary, original
reference/tolerance or controller change accompanied this source receipt.

Reproduce CPU coverage with:

```text
cargo test --locked --manifest-path crates/kuno-text/Cargo.toml -j2 --lib cpu_expert:: -- --test-threads=2
cargo clippy --locked --manifest-path crates/kuno-text/Cargo.toml -j2 --all-targets -- -D warnings
cargo clippy --locked --manifest-path apps/kuno/benchmarks/native-text/Cargo.toml -j2 --all-targets --features direct-device -- -D warnings
```

The future parent still needs the authenticated RAM-view adapter, actual
per-layer GPU route/input export, CPU/GPU overlap, unweighted partial upload,
original-rank merge and unchanged complete CPU13/repair/closure gates. This
first API is synchronous and joined; it does not add a new GPU overlap wrapper
or a multi-RHS packed decoding kernel.
