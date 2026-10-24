# Complete shape-aware local prefetch policy

The preceding complete balanced-wide bundle passed1534 GPU controls but its
new dense and expert kernels spill registers. Preserve that whole variant.
This new opt-in policy changes ALL four matrix consumers together: large dense,
large expert GU/DOWN, macro32/64, and eight-slot cohort16.

One shared Rust CMMA core alternates TWO weight-A fragments inside each K16
iteration. B inputs and A operands are assigned in that iteration; only C lives
across the loop. Prefetch the next row's A before the current row's products.
Large cells retain32 members x32 outputs/eight C/WG256 and both B fragments.
Small cells use16 members x32 outputs/four C, doubling N64->N128 at the original
macro/cohort WG128/256/64. Small inputs still use the original initialized
padding and exact six owners. Independent8-half publication, finite checker,
pending ordered DOWN, guards, raw-bank staging and erasure remain.

Small N128 kernels have distinct exact runtime role identities and actual N128
grids/counters. Do not widen legacy name matching, pretend old N64 grids are
N128, or reuse stale owner/admission state. Keep every older selectable variant
and its controls. Ordinary inference/API/privacy/Vashti/install behavior is
unchanged. No new dependencies or driver configuration.

Compose the whole policy before live speed testing. Validate actual generated
module geometry, storage operands, source origins, current-iteration lifetimes,
publication and negative mutations, then paired numerical/whole-model/recovery
controls. Preserve first/repeated comparison and original CLEAN helper. Inspect
actual Intel assembly for spills; estimated cycles are not throughput. Acceptance
and performance remain pending until measured on the complete frozen build.

The complete source policy is now composed, including the actual producer
selection and exact N128 runtime/cohort grammar. One shared core serves all
four matrix tiers. No live tests or speed claims yet. Planned421 full/418 off
AppCPU retains every preceding control;18/86 runtime includes nine new exact
role/grid/owner/phase/forgery controls. All88 new generated matrix layouts get
real operand-origin, current-iteration and publication/negative audits. Fresh
1747 GPU controls retain1534 and add93 large +116 small +4 multi-slot pairs.
Both original full-model/compiler/recovery chains and original CLEAN gate remain.
