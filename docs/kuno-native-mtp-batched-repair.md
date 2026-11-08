# Bounded original-math MTP accepted-cache repair

This default-unused internal leaf is based on `9453a4b`. The public laboratory
controller may explicitly select `KUNO_LAB_MTP_REPAIR=batch-public`; that
controller, protected labels and physical acceptance are separate work. Existing
`rewind` and `append_kv` calls retain their original behavior.

`ResidentMtp::repair_accepted_cache(target_start, accepted_tokens,
preceding_carry, accepted_carries, cancel)` borrows one actual accepted target
prefix of one to three inputs. `accepted_carries` contains exactly that many
original target hidden rows. Token row `i > 0` is conditioned with hidden row
`i - 1`. The preceding carry is the original confirmed-token carry; only a
no-draft count-one repair appends that first cell. The caller must finish real
target acceptance/repair, retain its original full-vector/greedy checks, and drop
all guarded MTP outputs before calling this method. The leaf does not
authenticate arbitrary borrowed tokens as target outputs.

The initial cursor must describe either the completed first or second draft
(`target_start` or `target_start + 1`), or the no-draft count-one endpoint
(`target_start - 1`). Accepted count cannot exceed the actual drafted envelope.
The leaf checks the complete token/carry geometry, every finite input, context
bound, original full-Q8 profile, absence of English pruning, health, retirement
and output leases before its first mutation. Invalid preflight preserves the
previous completed cache and follow capability.

For a drafted round the leaf retains the first real cell, queues the original
whole-suffix rewind and complete private-owner guard, then queues each remaining
accepted scalar condition, HC mixer, key/value projection, RoPE and append in
original order. A no-draft endpoint queues the original count-one conditioning.
Explicit local cell positions drive RoPE and append; the stored cursor stays at
its old completed value until success. There is no FFN, head, draft proposal,
target math, weights conversion or cache reuse based merely on greedy equality.

The initial clear retains original behavior. Between queued cells the existing
37-role preserve-input clear is followed by complete original Token/Carry
clears. Thus every original scratch owner is cleared except Fault. Fault remains
sticky, including its complete physical padding, until one original full fault
read and audited transient-I/O fence after all guards. No intermediate completed
ledger fence or cursor publication occurs. Original source and physical-owner
checks remain. Cancellation, overflow, device error, nonzero fault or panic
after mutation uses the existing full-cache wipe/poison path. A successful call
publishes the final cursor and aggregate completed counters together.

There are no new GPU/host owners or capacities. All42 original owners and final
erasure remain unchanged. The at-most-one rewind and two scalar appends fit the
existing512-dispatch/INFO bound, including the non-fused sweep path; dispatch
count is not reset between cells. Existing64MiB runtime and16MiB host caps remain
mandatory. Borrowed input rows stay owned by their original target payload;
temporary carry conversion/upload/readback guards use the existing capacities.

`Traffic.batched_repair_calls`, `batched_repair_cells` and
`batched_repair_fault_gates` are completed-only aggregate u64 counters. Cells
count real appended cells, not retained or discarded draft cells; every
completed call has exactly one final full fault gate. No token, carry, cursor,
route or generated text is persisted by this leaf.

The v167 public control recorded138 full draft-packet reads and199 full fault
reads (69 rewinds plus130 appends). This leaf could reduce those repair fault
readbacks to69, saving130 readback/audit windows. These counts are source/receipt
evidence, not a speed claim. The1.829s repair phase includes five necessary target
recomputations and other work; it is not isolated readback cost. Unchanged
original CPU13 full vectors, actual rejection repair, target greedy and complete
owning closure/recovery gates plus a same-artifact normal timing comparison are
required before physical acceptance.

The integrated controller explicitly rejects profile-ranked bank filling for
this first comparison. The protected guard consumes only the canonical
`batch-public` selector and retains the original Intel memory metadata,
48GiB host cgroup (plus the existing196MiB language owners), no-swap/core rules
and all original target/draft capacities. Q2, English pruning, broadcast head,
target-only decoding and alternate source geometries reject. The fixed local
runner has three additive warm/code/prose labels ending in
`intel-mtp-batch-repair-down-integer-grouped-ple-gu-stride-shared-head-gpu-host`;
they retain Original bank fill, original512 backlog/IQ4 decoding and the
selected integer GU/DOWN, grouped PLE and shared full-Q8 head. Actual requested
selection and completed leaf counters are recorded separately. All271 existing
and additive protected-launcher tests and three actual helper scope/cap checks
pass. This source acceptance does not establish device or throughput acceptance.

Source validation uses only the existing
`cpu-batch-native-target-20261011` cache, the standalone native-text manifest,
`--no-default-features --features direct-device --bin kuno-text-lab -j2`, and
two test threads. All8 new repair tests,32 existing resident-MTP tests and9
original MTP kernel/math/binding tests pass. The new cache shader proof validates
Checked and Unchecked WGSL and emits SPIR-V with the actual runtime
`zero_initialize_workgroup_memory=None` setting. It covers the unchanged original
append/rewind kernels; it does not establish GPU runtime performance or fault
recovery. A lightweight source comparison also confirms16 original scalar,
condition/HC/projection, clear/guard and failure bodies unchanged. No model,
GPU, service or production integration is invoked by these checks. Strict
pure-direct all-target Clippy with `-j2 -- -D warnings`, scoped formatting and
diff checks also pass.


## Physical acceptance, 2026-10-11

Frozen v173 source970f8f4141adc903212fe102ada504b4a42edfe8 and pure-direct binary
69bc565a1396d85d9f9fdaa702620d94b44795c15b4541329f7f4d502c8ec0ec pass
unchanged original CPU13 cold/warm full vectors/carries/greedy and actual repairs.
Default cold counters remain0; warm actual6 calls/7 appended cells/6 fault gates
complete without new owners. Same-artifact compiler-idle code200 ordered pair
measures10.533053 original versus10.603973 batched real continuations/s,
+0.67percent. Both199 continuations,130of138 accepted drafts/69 rounds/five
real target recomputations. Batched69 calls/130 cells/69 gates reduce total MTP
reads337to207 and bytes15564800to11304960. Actual repair interval1.802to1.590s
includes necessary target recomputations and checks; do not credit it entirely
to readback reduction. All1382MAIN/828997632B,42 draft private/11698176B and432
HOST/36147609600B owners close. Same-boot physical memory/service/fault recovery
passes. This is one ordered code pair, not repeated/prose/independentCPU200 or
production acceptance. Exact aggregate evidence is checked in under
v173_intel_mtp_batched_repair_and_bank_backing_physical.
