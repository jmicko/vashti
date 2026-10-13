# Frozen expert source compaction

This is an internal native-text laboratory scheduling optimization. It changes
neither the Kuno API nor installed inference defaults. It extends the existing
integer GU stride-four schedule for actual counts one through four; integer
DOWN uses the same plan when selected. Ordinary DOWN can still consume the
compacted GU output. Original weights, route identities, member order,
two-residual-Q8 arithmetic and acceptance tolerances remain unchanged.

## Plan and execution

GU preparation derives five lists from the immutable routed MAP, one for each
original source partition: first slots `[0,72,144,288,432]`, sizes
`[72,72,144,144,80]`. Lists contain ascending original group indices, never
replacement experts. The MAP's actual group count must fit the caller's
capacity (30 or 40); it need not equal that capacity.

The existing full 64 KiB GU owner reserves words 6144 through 6383 for five
48-word records. Each holds ready, count, GU dispatch `[80,min(4,count),1]`,
DOWN dispatch `[320,count,1]`, and 40 group indices. Unused indices stay zero.
A zero-group shard has a valid zero-work indirect dispatch. No extra model
buffer, host route readback or allocation-cap increase is introduced.

Preparation preserves the original complete frame/header/tail clearing:
three frames ending at word 4384 for count1..3, or four ending at 5824 for
count4. A uniform storage barrier orders all token-zero lanes' tail clearing
before the five record builders overwrite their disjoint records. Workgroup
barriers alone do not order these storage writes.

The parent authenticates the full owner and stream. The shared indirect-view
helper accepts only exactly 65536 backing bytes, zero numeric start/end trims,
and shard indices below five. CubeCL's allocator represents full owners with
`Some(0)` trims, which are equivalent to `None`. Nonzero trims and wrong backing
extents remain rejected before calling unchecked size subtraction. The GU and
DOWN dispatches use checked 12-byte views starting at
`24576 + 192*shard + 8` or `+20`. The native runtime issues indirect GPU
dispatches directly from these views.

GU retains 64 threads, stride-four codebook reuse and the original arithmetic
body. DOWN retains its original body and 64-thread grid. Kernel metadata must
match WGSL workgroup dimensions. After all consumers, the GU audit reconstructs
every record from MAP, checks all 40 index words including inactive ones, then
erases records after validation. Original full-owner, finite and transaction
checks remain in force. Receipts credit five compact GU dispatches and one
audit per selected layer, plus five compact DOWN dispatches only when integer
DOWN is independently selected.

## Validation and current status

The first inherited implementation failed before device dispatch because it
rejected allocator-produced zero trims. Review also found incorrect count1..3
reset coverage, a missing storage barrier, wrong count4 compact kernel metadata,
incomplete record-tail auditing and unconditional compact DOWN credit. The
first device retry then exposed a copied count4 guard in count1..3 DOWN, which
incorrectly skipped all ordinary compact DOWN work; its main guard now matches
the original count1..3 domain. These
are corrected together rather than treating the earlier source-only test pass
as proof of a working device path.

Regression coverage includes real pure allocator-produced handles, exact
indirect spans, malformed trims and shard indices, compiled WGSL dimensions,
original reset-body equivalence, shader lowering and independent DOWN policy
credit. Device acceptance still requires unchanged cold/warm CPU13 checks and
complete cleanup; performance requires a separate normal generation run.
Results and immutable artifact identities belong in the native-text handoff
and receipt file. No measured speedup is claimed by this specification.
