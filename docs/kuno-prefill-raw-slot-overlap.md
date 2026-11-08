# Preserve each expert's actual copy endpoint

Optional backend continuation from accepted coupled source `de309ebf`.
Root owns the `strata-raw-slot-overlap` application feature, CLI registration,
composed build, hardware and acceptance. The vendor feature is
`cubecl-wgpu/kuno-raw-slot-overlap`, depending on the existing expert-cohort and
copy-cohort backend features. No API, model, configuration, allocation,
frontend, installation or release change is introduced; Vashti changes: none.

## Remaining active Intel scheduling difference

Pinned MIT Strata `fb58e0db` uses its ordinary FP16 experts on Intel. In
`sycl/src/prefill/prefill.cpp:3804-3844`, each genuine streamed blob waits on
the old reader of its own ring slot, then records its own copied endpoint.
The routed-only loop at `3993-4009` stages up to eight experts ahead and lets
each expert's compute wait on that expert's copied endpoint. Original raw
release is after both dequantizers at `3964-3976`.

Kuno's accepted eight-group lookahead keeps four consecutive legacy experts
in one host upload unit. Each expert already emits one actual `HostCopies`
command containing its real HOST role spans. Mixed-resident roles are omitted
by the unchanged source resolver. However, consecutive COPY nodes previously
shared one original Planner segment. Its wait became the maximum of every
slot's old reader. The accepted queue control explicitly shows replacement
expert8, whose slot was read by expert0, waiting for expert3's DOWN-image
decode. Likewise all initial eight uploads shared one signal, so expert0
could not begin DQ after just its own copy.

This is a concrete dependency distinction, not a measured attribution of
remaining wall time or a promise of a speed gain. Strata also materializes
both complete FP16 weight images and then queues GU, SwiGLU and DOWN
(`3964-3983`). Its inactive CUDA/MMQ/fused-INT8 alternatives do not explain
this Intel baseline.

## Complete source change

The existing full native expert grammar still validates the complete batch
before replay. `NativeStrataExpertCommand::HostCopies` resolves its original
retained bindings, then passes the genuine live batch scope to
`stage_public_host_copies_owned`. The original scope check moves into that
routine immediately before physical-range validation; it is not removed or
replaced with a caller boolean.

When the new vendor feature is selected, the graph verifies that the command's
actual appended nodes are retained COPY operations on one unchanged queue.
Only the final actual COPY node receives the existing `release_after` bit.
All of one expert's role copies therefore stay together; the next expert gets
a new segment and retains its own original range-derived wait. Ordinary
unscoped transfers and feature-off replay retain the previous policy. A
resident expert with no HOST command adds no fake copy or endpoint.

The original Planner, source inventory, copies, access intervals, queue tags,
three independent MAIN/COPY/SHARED waits and all compute phases remain intact.
In particular, the original streamed DOWN-image release and every original
final-DOWN unit marker still apply. Four-group host retirement/refill and the
same two admitted raw banks remain unchanged. Existing full-bank cohort DQ
descriptors still wait on every genuine conflicting bank range. This boundary
grants no copy readiness, GPU completion, overwrite or source authority.

The original safe segment compactor preserves an endpoint observed by a peer
or a following segment whose required waits increase. It therefore retains
the staggered slot waits. Real graph signatures already include each node's
`release_after` bit, so prepared replay identity retains the new endpoints.
No new node, field, owner, heap allocation, uncharged metadata or persistence
is created. Additional segment command buffers remain inside the original
node-count bound and existing actual metadata accounting and failure cleanup.

## Required acceptance

The original actual-Plan queue control remains selected with explicit old
policy, including its original maximum-DQ3 expectation. Three additional
controls exercise the original Planner:

1. `per_expert_copies_keep_each_original_slot_wait_before_and_after_compaction`
   checks replacement8/9/10/11 waits on original DQ0/1/2/3 respectively;
   initial DQ0 waits on copy0 rather than copy7; every expert's three roles
   remain together. Removing genuine early DQ release must restore the wait
   on that same slot's final math. The actual four-group Plan cursor, initial
   eight uploads, exact copy bytes and peak-live count remain covered.
2. `per_expert_copy_cuts_preserve_each_main_and_shared_reader_wait` requires
   each slot's complete distinct MAIN and SHARED reader waits, never a scalar
   maximum substituted for the separate queue dependencies.
3. `per_expert_copy_cuts_do_not_narrow_a_whole_bank_cohort_reader` requires
   every overwritten slot to wait on the actual full-bank reader and retains
   the full-bank footprint even across untouched slots.

All new controls check both uncompacted and compacted plans. Original whole-
bank, resident/no-copy future-wait, source/grammar, prepared selection,
arithmetic, ownership and recovery controls remain unchanged. No numerical
shader changes are introduced. Root must compose the complete candidate,
run the full original CPU/runtime/compiler/GPU populations and both original
full proof/recovery chains, inspect actual counters and physical recovery,
then compare completed whole-prompt timings. No tests, build, GPU work or
performance measurement was performed by the implementation subtask.
