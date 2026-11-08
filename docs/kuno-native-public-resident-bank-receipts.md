# Public resident-bank allocation receipts

This additive laboratory receipt composes with the unchanged CPU13 and fixed
code/prose controllers. It adds no selector, owner, GPU work, model source,
admission rule or production API. It depends on the authenticated telemetry
leaf `859c2a5` and is implemented after the MTP controller `012320c`.

Both controllers call `Attachment::resident_bank_allocation()` while the actual
attachment is live, before closing the target. Original target/source close,
consumer drop and mandatory Attachment close retain their order. After those
successful closes, the controller requires exact equality with
`CloseReceipt.resident_bank_allocation`. Failed setup, target close or source
close cannot publish a successful receipt.

`record.resident_bank_allocation` and the nested
`record.frozen_host_close.resident_bank_allocation` contain the same authenticated
snapshot. They expose the leaf's eight scalar fields: `layers`,
`source_bindings`, `physical_buffers`, `logical_bytes`, `physical_buffer_bytes`,
`conservative_bin_bytes`, `padding_bytes` and `admission_headroom_bytes`.
The controller requires the original 48 layers, 288 distinct resident roots,
original full144 logical/bin extents, and checked differences for actual padding
and unused conservative allowance. It additionally records
`attached_close_snapshot_match=true`, `host_mirror_excluded=true`, and
`release_erasure_or_admission_credit=false`. When neither actual attachment nor
close snapshot exists, the allocation receipt is null; mismatched presence is
an error.

These are retained Wgpu buffer sizes, not opaque Vulkan memory requirements,
heap usage, newly available VRAM or release/erasure credit. Source and dispatch
handle clones do not add another backing allocation. The nine HOST source pages
per layer are excluded. Original full owner audits and fresh physical memory
observations remain the independent authorities for cleanup and admission.
JSON serialization occurs after the timed inference block. No prompts, tokens,
weights, expert IDs or routes are added to the receipt.

Source checks pass seven focused leaf/controller tests and all 39 target helper
tests (two overlap). They cover exact default null, both presence directions,
matched backing from logical extent through the old bin bound, all eight field
drift cases, invalid/overflowing byte relations, HOST exclusion and handle-clone
counting. Strict pure-direct all-target Clippy with `-j2 -- -D warnings`, scoped
formatting and diff checks are required before integration. Physical snapshots
and full unchanged CPU13 numerical/closure acceptance remain root's separate
frozen-artifact gates; these source tests claim no physical result or speed gain.
