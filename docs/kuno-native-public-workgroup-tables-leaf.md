# Public workgroup IQ table laboratory leaf

Source-only follow-up based on integration commit
`7364dc8df4dc645bb3df588f7a52b6e1b9751646`. This leaf is not selected by an
environment flag or launcher. It is not physically accepted or benchmarked.

The dedicated `grouped_gate_up_frozen_workgroup_grid` entrypoint calls the same
original64 frozen gate/up inner kernel. Its exact name selects public literal
table deduplication in the WGSL compiler. Identical gate/up codebooks share one
module workgroup array; all other entrypoints retain their existing lowering.
The compiler inserts a 64-way physical-lane switch immediately after the
existing direct uniform inactive-partition return guard. Each lane writes
literal indices and literal public words, then all lanes cross one barrier
before decoding. There is no dynamic by-value table lookup during initialization.

For the actual mixed-profile gate/up shaders, the table and total shared storage
are respectively 256 words / 5120 bytes for IQ3_XXS, 512 / 6144 for IQ3_S, and
2048 / 12288 for IQ2_S. IQ4_XS has no grid and retains 4096 bytes. This accounting
includes the original two 512-cell F32 reduction arrays. Storage bindings,
source owners, original ranks, status writes, input arithmetic and reduction
order are unchanged. Downward, finite and reduction entrypoints are untouched.

The immutable policy is `Geometry::with_workgroup_tables()` and requires frozen
sources plus inactive-partition exit; private and workgroup policies are mutually
exclusive. Integration hooks are `Sources::new_with_workgroup_tables(...)`,
`Sources::workgroup_tables()` and
`Workspace::frozen_host_source_workgroup_tables()`. Parent-side selection and
receipts remain pending; a future `workgroup-public` selector must explicitly
choose these hooks rather than silently changing the private-table experiment.

Focused CPU checks cover every actual mixed profile in Checked and Unchecked
mode. They validate WGSL and generated SPIR-V, all public initializer words and
their one-write lane assignment, deduplication, exact shared-storage sizes,
absence of Function/Private table arrays and their composite stores, unchanged
inference arithmetic and memory events after public initialization, original
barriers plus one initialization barrier, and unchanged downward/finite/reduce
expansions. An adversarial post-barrier element store must fail the immutability
proof. Geometry tests reject missing prerequisites and policy combinations.

Physical driver code, numerical/fault/owning acceptance, both-card behavior and
whole-model timing remain untested for this candidate. Literal initialization,
the extra barrier, shared-memory residency and repeated initialization per output
workgroup may outweigh any reduction in private-table costs. The demoted explicit
WGSL-family shared-table experiment does not establish this active CubeCL path's
performance in either direction. Existing GGML provenance and MIT notices remain
unchanged.
