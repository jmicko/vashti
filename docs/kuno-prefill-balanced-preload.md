# Two-bank K16 operands for the complete balanced expert bundle

Internal candidate on top of accepted full1425 balanced-global. Keep the entire
accepted variant selectable feature-off. `strata-balanced-preload` /
`--balanced-preload` requires `strata-balanced-global` / `--balanced-global`.
No new native ABI, role authority, GPU owners, geometry, resource capacity,
precision, API, configuration or Vashti behavior. No installed service changes.

Both macro32/64 and packed cohort16 call the SAME product selector. The old
single-bank product body is retained. The new path genuinely loads two K16
input-B matrices and four K16 weight-A matrices before consuming the current
bank. It then performs A00*B0+C0, A01*B0+C1, A10*B1+C0, A11*B1+C1 and advances
K32. Each output keeps the original chronological K16 accumulation order.
Actual admitted K=2560/640 are divisible by32; a workgroup-uniform divisibility
guard precedes all operand reads. The final next bank is K=columns-16, never a
speculative bank beyond the descriptor extent. Existing authentic input padding,
all route/header guards, independent8-half publication, checked activation and
pending ordered DOWN consumers remain identical. Tile8 and wide tails unchanged.

Separate profile counters `expert_balanced_preload_macro{32,64}_products` and
`expert_balanced_preload_cohort_{gu,down}_launches` are incremented only on real
new launches. Original balanced/semantic counters remain credited exactly as
before. Public fixed-numeric report carries `balanced_preload_selected`.

Original404 AppCPU names remain selected with the feature both absent/present;
one added bounds/chronology control makes405 with the feature. Existing actual
48-layout compiler proof selects exact one-bank/two-bank populations. New path
requires four distinct weight loads, two distinct input banks each used by its
own pair of products, all six loads before the first product, and actual second
K16 accumulator SSA chains back to the corresponding first-bank result. Existing
type/WG/owner/AtomicOr/publication negatives remain; additional real mutations
alias next B or A to the previous bank and crosswire accumulators. Original
single-bank source/proof branch stays intact feature-off. Native controls remain
18/73, metadata8+9, native10 because authority/ABI/geometry have not changed.

All1425 existing GPU controls must freshly execute with the new worker, retaining
independent F64 and complete-owner/status/fault/sentinel/source/erasure checks,
both roles and genuine packed eight-slot cases. No previous worker's numerical
receipt may stand in for this changed implementation. Both original18-script
whole-model/recovery chains must pass and new counters must equal their actual
balanced counterparts. Inspect the actual compiler's bank dataflow and spills;
do not infer prefetch survival or a speedup from source alone. The actual F64
checks are required to validate arithmetic and final K contributions, beyond
the CPU schedule law. No compilation/numerical/performance claim at composition.
