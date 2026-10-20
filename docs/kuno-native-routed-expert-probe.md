# Native routed wide expert acceptance candidate

This is a compiled-public numerical diagnostic, with no inference API, private
request, model download, service change or performance result. The operator
implementation and runtime selection are separate from this acceptance code.

`strata_native_expert_probe.rs` calls the real
`native_expert_wide::try_expert` entry. It requires a genuine queued wide path
for counts 256, 257, 287, 288, 383 and 384. For 32, 33 and 64 it requires that
entry to decline without launching, then calls the actual original
`native::run_expert` dispatcher. It does not substitute a computed GPU result or
create a synthetic PromptScope. These explicit public cases test the original
primitive, while the production caller retains its existing prompt authority.

There are 26 reports, thirteen for each of GU and DOWN:

* Finite full-output checks for all nine counts above.
* Count257 with a NaN in weight feature zero, then another feature. Every member
  consumes the bad feature, covering both the genuine wide prefix and original
  one-member tail. Both expect the original fault128 and status policy.
* Count257 with one invalid route in the wide prefix, then one in the original
  tail. Each map is built and validated by the original full prompt grouper
  before exactly one selected member is deliberately corrupted.

The common fixture/oracle moved to `strata_routed_expert_probe.rs`.
`strata_cubek_expert_probe.rs` keeps its original fourteen cases, family name,
actual-library-launch requirement and report fields. Its finite/F64 output
oracle and unchanged `1e-4 + 3e-4 * abs(reference)` threshold are shared by the
native diagnostic, including the original FP16 subnormal row.

Every case retains authentic sorted, noncontiguous routes above20479, a genuine
256-byte weight descriptor offset, original map/status geometry and complete
GU/PART owners. The full output, status, fault, input, weight and map allocations
are scanned, including inactive owner regions and allocation padding. Original
readback leases are scrubbed, dropped and actually sanitized/audited after each
window. The actual arena reset/drain/erasure precedes every returned report,
including failed numerical checks; the outer caller must still close/audit its
arena. There is no shortened scanner or numerical threshold adjustment.

The following CPU/source check passed in the isolated worktree:

```sh
python3 apps/kuno/benchmarks/text/test-native-routed-expert-probe.py --deps-dir /media/jmicko/faststore/kuno-development/root-native-text-target-20261005/release/deps
rustfmt --edition 2024 --check apps/kuno/benchmarks/native-text/src/strata_cubek_expert_probe.rs apps/kuno/benchmarks/native-text/src/strata_native_expert_probe.rs apps/kuno/benchmarks/native-text/src/strata_routed_expert_probe.rs
git diff --check
```

The script checks the exact original14/native26 schedules and extracts the
actual pure fixture functions, actual original group ABI/builder/validator and
actual Handle-window function into a temporary Rust test. Three tests passed:
all counts and preserved subnormal inputs, original offset/overflow handling,
and exact single-member prefix/tail corruption plus nonfinite feature placement.
It links retained real dependencies without a Cargo build or GPU construction.
It does not compile the full runtime wrappers, prove actual wide kernel math,
recorded-prefix/tail authority or physical recovery. Those remain the combined
worker compilation and protected GPU acceptance gates owned by the root task.
