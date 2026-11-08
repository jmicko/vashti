# Native expert batch allocator ownership

This is one connected part of the explicit `strata-native-expert-bulk` candidate,
not an installed engine or a new performance result. No API/Vashti/configuration
or private persistence changes. The selected installed worker remains3fffc266.

The scoped backend resolves original binding views once per layer. It must pass
the same `Arc<Vec<cubecl_runtime::server::Binding>>` to every original compute
and HOST-copy node through `Graph::push_owned`/`push_transfer_owned`. Retaining a
`WgpuResource` alone does not prevent allocator reuse; these are the original
allocator bindings too. Ordinary calls keep the wrappers with `None`.

Nodes own the bundle immediately. Before any fallible command recording or
submission, each actual Recording independently retains its referenced bundles,
deduplicating only the same Arc allocation. Equal values in distinct allocations
stay distinct. Actual binding-vector capacity, Vec/Arc headers and retained
pointer-vector capacity count against the existing64MiB metadata limit; the
limit is not raised. A bundle spanning original recording boundaries stays
alive in every referencing Recording until its own actual completion. This
conservative accounting can charge one shared allocation in multiple recordings.

An early failure drops only duplicate temporary references: Nodes still own the
bindings. Once submitted, clearing Node metadata leaves the Recording owners
intact. Existing real-fence completion and unknown-completion quarantine govern
the entire original Recording, including its bindings. No completion credit,
new descriptor authority, access-range cache, hazard change or reordered command
is introduced by this half.

Two CPU tests compile the actual production owner collector through the worker
harness: split recordings preserve the same owner until the last reference is
dropped; actual capacity, distinct equal allocations, empty bundles and budget
failures are checked. The whole worker test build with Intel/native32/paired/PV
features passes. Artifact under the development base:
`native-bulk-graph-owner-cpu-20261020.log`. An independent source review found no
clear lifetime/budget blocker in this half. This is not whole-batch acceptance:
the scoped client, backend binding/INFO preparation, original stream/lane checks,
GPU arithmetic, terminal-error recovery and complete-request tests remain gates
for the integrated candidate.
