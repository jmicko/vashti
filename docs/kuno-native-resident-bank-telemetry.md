# Original resident expert-bank backing measurement

This is content-free ownership telemetry for the unchanged public DASLab
144-slot frozen source attachment. It changes no allocation, capacity,
arithmetic, source geometry, admission rule or private owner.

`Attachment::resident_bank_allocation()` is valid only while Attached and
returns a fixed scalar `ResidentBankAllocation`. The successful terminal
`CloseReceipt.resident_bank_allocation` captures the same resources after the
existing mandatory consumer join, before final source pins drop. A snapshot
does not grant release, erasure, device-local placement or admission credit.

The measurement visits the accepted immutable `Sources`'48 layers. Each layer
has15 role-major source bindings: two original resident VRAM buffers and three
imported HOST pages per role. Only indices0/1,5/6,10/11 contribute. The nine
HOST pages are excluded. Existing construction checks prove original full
resource extents, stream, registered classes and global physical-buffer
uniqueness. The measurement creates no source Handle clones, resource vectors,
new GPU owners, `get_resource`/heap queries, readbacks or submissions.

BankSegment handles, SourcePage handles and kernel bindings may reference the
same backing. They are counted once by the accepted source's distinct original
resident roots, not once per handle clone or dispatch. The snapshot reports:

- `layers`:48.
- `source_bindings`:288 original resident source bindings.
- `physical_buffers`:288 distinct accepted resident buffers.
- `logical_bytes`: original expert payload extent,14,144,716,800bytes.
- `physical_buffer_bytes`: sum of these actual retained Wgpu buffers' sizes.
- `conservative_bin_bytes`: the unchanged separately rounded admission upper
  bound,18,723,373,056bytes.
- `padding_bytes`: actual buffer backing minus logical payload.
- `admission_headroom_bytes`: conservative bound minus actual buffer backing.

Every included resource is rechecked as a full original binding at offset0,
with logical length and physical backing within its existing admitted bin.
All sums and differences are checked. Partial layers, wrong counts, changed
profile geometry and overflowing/out-of-bound extents cannot emit a snapshot.
The two fixed scalar snapshot headers are precharged within the unchanged
256KiB Attachment metadata allowance. No new persistent or model-sized owner
is introduced.

The admitted bank bound is not allocation telemetry. In the existing allocator,
exclusive buffers can receive backing larger than their request through a
rolling mean or retained free page. These measurements establish the actual
buffer difference. They do not measure opaque Vulkan requirements, driver
overhead, unrelated device allocations, HOST mirror placement or other models.
They also establish no speed improvement or additional safe capacity.

The controller owns eventual JSON receipt wiring. Source tests use the pinned
independent public tensor inventory to verify all288 original extents, HOST
exclusion, exact backing versus mixed aligned backing, separately rounded
bounds, completion counts and malformed/overflowing totals. Physical results
must come from root's frozen artifact and complete original-vector/closure
gates; source tests fabricate no GPU result.

Source receipt: all five new allocation metadata tests and four existing
Attachment capacity/phase/release tests pass. Locked all-target
`direct-device` Clippy with warnings denied (`-j2`), formatting and whitespace
checks pass. The isolated branch is based on root `22e72798`; no root checkout,
GPU, service or installed artifact was modified. Actual physical buffer sums
remain pending root's frozen-artifact receipt wiring and device acceptance.
