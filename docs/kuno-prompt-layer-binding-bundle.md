# Borrowed public prompt-layer bindings — 2026-10-20

This is a bounded host-enqueue candidate, with no new GPU math, API, flight,
completion, persistence or scheduling path. Full compilation and protected
hardware acceptance remain parent-owned. Source formatting/patch checks alone
are not a timing or ownership result.

The selected profile attributes about1.879s to expert enqueue before native
capture, including0.578s source staging and0.498s dequant enqueue. These are
overlapping code buckets from one actual profile, not independent GPU costs.
Pinned MIT Strata `fb58e0d` selected Intel `prefill.cpp:3964–3981` directly reuses
its admitted expert format, raw device pointers, two FP16 image slots and GEMM
handle. The normal product reaches oneMKL via `gemm.dp.cpp:799–805`. It does not
revalidate original model FDs for every individual prototype binding.

Rust's old per-group dequant setup resolves the same three
`Sources::prototype(graph,layer,role)` values repeatedly. Each goes through
`validate_prompt_ready` and `effective_role`; a loan also checks its shadow before
resolving an ordinary cache/HOST source. Those paths call
`Plan::check_binding` repeatedly. That calls `BoundGraph::verify_unchanged`,
which locks and stats both original GGUF descriptors. Depending on whether the
prototype is shadowed, three prototype lookups perform12..18 file-stat calls per
group. At26,570 groups this is318,840..478,260 repeated metadata syscalls, plus
locks/source resolution. The top-level enqueue and each actual staged group
already validate the current graph/source boundaries separately.

A read-only CPU probe of200,000 `os.fstat` calls on those two original open model
files took0.2269s, about1,134ns/call including Python. Scaling just that loop to
the duplicate calls gives0.362..0.542s. This is evidence of a plausible cost,
**not** a Rust/GPU performance prediction, speed result or upper/lower bound.
It read no model content and changed no file or setting.

The candidate `strata_prompt_layer_sources.rs` borrows one actual immutable
`Sources` and `BoundGraph` for exactly one layer's enqueue. Entry invokes the
original graph/file/source/loan checks and resolves all three prototypes through
the original resolver. The immutable borrow prevents placement publication,
source closure, table rebuild or loan transition while the bundle lives. It is
explicitly not Send/Sync. Every prototype consumption still checks the actual
retired/loan-ready/mutation attachment state; it never uses a cached valid bit.

Every actual staged expert continues through the **unchanged**
`Sources::stage_prompt_group`: original graph FD identity, source plan/instance,
registered current HOST/cache/loan-shadow owner, raw-ring owner/stream/bounds,
and backend checked copy admission all remain. Every dequant launch still calls
the original `bind_prompt_image`, attaching the current real address table and
complete BDA read dependency set. No dependency is narrowed to the selected
expert, and no read-lease authority is manufactured. The same eight raw-ring
views are prepared once from the actual owner and original role extents, then
cloned for that enqueue only.

The bundle stores only public model handles, graph and layer. It stores no
prompt, route, expert-use profile, output, private map, validity flag or snapshot.
All request-derived group/map/selection lifetimes and wiping behavior stay
unchanged. The bundle is statically bounded to1KiB; eight ephemeral raw views
replace repeated equivalent temporary views. No reference escapes into a new
model cache or survives its original Sources borrow.

Further source leads, not implemented here: the optional CubeK adapter rebuilds
its forced public geometry/capability blueprint/config on every product; a
separately validated fixed-geometry template could amortize that while checking
every actual current owner/count. HOST role lookup also linearly scans its page
registry. Neither justifies retaining request geometry or weakening runtime
guards. Root should evaluate this small coherent candidate before changing
additional scheduling or ownership behavior.
