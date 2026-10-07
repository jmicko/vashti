# Public single-member dense dispatch

This default-off source candidate removes the seven inactive token workgroups
from tile1 dense projections. Ordinary `MatrixGeometry::new` still has MAX8
dispatch capacity. `with_single_member()` accepts only public tile1, fixes the
grid to `(rows,1,1)`, and rejects any runtime count other than one before launch.
The packed shader, original 64-lane column arithmetic/reduction tree, runtime
count/bases/strides, and full owning buffer bindings do not change.

The protected native model selects this policy only at a completed full-audit
`SingleAll` or `SingleQ4All` boundary. Fresh executor admissions reset dispatch
capacity to eight. The executor exposes the observed capacity and rejects a
non-count1 call before weight/cache mutation or enqueue. Prefill and the other
profiles retain their previous policies. No private count enters a shader key
or chooses the public dispatch policy.

Inactive ordinary matrix workgroups never write storage output; their zero
stores affect workgroup-local reduction scratch only. Some floating scratch
owners intentionally retain prior inactive rows between uses. This candidate
preserves those exact bytes rather than asserting a new zero-tail invariant.
Existing GDN/QSA full scratch clears, outer/router/packet clears, GR status
clears, full projection finite/fault scans, owning spans, budgets, model
acceptance and full lifecycle erasure remain required.

`matmul-probe` retains its original 84 cases and adds seven fixed cases, one per
original format. Each uses the same full owners for real MAX8 prefill, ordinary
tile1 count1 reference and single-member count1 candidate. It restores the
accepted prefill span before the candidate, compares against independent CPU
expectations and the ordinary GPU reference with unchanged tolerances, checks
every inactive/gap word against actual prefill bits, and checks full padding and
post-reset erasure. The additional guarded full output snapshot is bounded to
32 KiB; no snapshots persist between cases. The receipt has 91 cases and seven
`member_capacity=1` entries with `max8_prefill_reused=true` and paired error.

Host tests cover public-policy/count rejection and MAX8-to-count1 nonzero tail
preservation. Compilation, operator/device lifecycle and performance acceptance
belong to the physical owner; source inspection establishes no speedup.
