# Native GDN retained metadata connection

This bounded internal correction connects `gdn_chunk::validate_bindings` to
the already admitted `resident_util::resources` policy. It adds no selector,
permission, counter, buffer, owner, budget or kernel. The metadata policy remains
disabled by default; its default branch performs the original ordered
`get_resource` calls. Existing canonical selected runs use the previously
authorized retained-metadata branch.

The validator uses exactly the same eight live `PrivateArena` handles, in the
same order. It keeps every original logical-word, duplicate-ID, zero-offset,
same-stream, resource-offset, full-size, full-backing and physical-nonalias
check. The fixed eight-handle vector is capacity-checked before population;
the returned resource vector must also have eight entries and eight header
slots before physical validation. No data read, fault proof or completion fence
is replaced. `ManagedResource` retains the original backing allocation through
validation, and the arena still owns every launched buffer.

The compile-time header inventory conservatively includes eight `Handle`,
eight `Binding`, sixteen `ManagedResource<WgpuResource>` and six vector headers.
It must fit within 4 KiB. The existing GDN stage reservation is 64 KiB, and its
pinned worst INFO inventory is 37 dispatches at two 512-byte records each;
that inventory plus the header peak must remain within the unchanged stage
reservation. The client converts exactly eight handles to bindings; the selected
server reserves exactly eight result slots before filling. There are no tensor
copies or additional persistent records. This is application metadata accounting,
not a claim about opaque driver memory.

## Source mechanism and evidence limit

Actual native `resident_gdn_batch` recurrence invokes this validator for ordinary
Chunk, Single/Three/FourStreaming and shared count1 execution. Previously its
eight direct `get_resource` calls bypassed the selected helper. The WGPU server's
`get_resource` invokes `Scheduler::execute_streams`, whose backend flush calls
`WgpuStream::flush_inner`. Pending tasks end their current compute pass and
submit queued projection/preparation work before recurrence. Empty subsequent
lookups do not imply additional actual submissions. The selected retained
method instead checks full same-stream retained bindings and active private I/O,
then inspects the same allocation metadata without a queue operation.

The v173 original and batched-repair normal records both contain 74 accepted
target chunks and 36 GDN layers. That implies 21,312 direct metadata lookups and
up to 2,664 needless mid-layer submission boundaries. It does not imply 21,312
submissions or removable GPU waits. Original/batched records respectively show
618/618 submission waits, 8.300911/8.286349 seconds of wait time, and 302/302
readback batches with 2.464642/2.470579 seconds of mapping. Those process-global
dual-device timers overlap target phases; they are neither additive costs nor
isolated GDN execution time.

Source regressions check the real validator's preserved guards, fixed header
shape, default/selected wrapper and backend paths, and validation before both
actual recurrence launch wrappers. Original GDN numerical and shader tests
remain required. CPU source checks establish no hardware gain. Root-owned
protected original CPU13/full-vector/greedy/repair and closing/recovery gates,
followed by compiler-idle frozen before/after artifact measurements, remain necessary
before any performance or physical acceptance claim.

## Source validation receipt

On the exact `26295f65bc4c78040731284bcf9b1d8c311e4539` base, the standalone
native-text manifest with `--no-default-features --features direct-device`,
`--bin kuno-text-lab -j2` and two test threads passes all 3 new metadata source
regressions, all 15 original GDN chunk/streaming arithmetic and offline shader
checks, and all 17 native resident-GDN owner/lifecycle/failure regressions.
Strict pure-direct all-target Clippy with warnings denied, scoped formatting
and whitespace checks pass. The verified existing cache was
`oracle-native-target-20261009`; no GPU or model runtime was constructed.

A source comparison also confirms that every non-validator production line in
`gdn_chunk.rs`, including all arithmetic, kernels and launch wrappers, is
byte-identical to the base. The test-module declaration is the only other
change to that file. These 35 CPU checks do not establish physical acceptance
or a performance improvement.

## Frozen comparison contract

There is deliberately no new GDN selector. The selected canonical metadata
policy was already enabled; this correction connects its omitted validator.
The before artifact is v177 on26295f6; the after artifact is v178 oncd638c7.
Their executable production source differs only in this validator; intervening
v177 receipt edits are documentation/aggregate data. Pin both binary/source
hashes, unchanged protected launcher/helper, original model/reference, Intel
placement, public prompt/seed and all selected policies. Compare the same
original or genuine-four configuration across these frozen artifacts. This is
an artifact comparison, not a claimed same-artifact toggle or isolated GPU
cost measurement. Compiler-idle paired normal timing is required; the existing
v177 three/four pair does not measure this correction.

## Physical correctness receipt, v178 (2026-10-11)

Frozen sourcecd638c70be6281f729bc0de7d2d23391bfbb313c, pure-direct binarySHA256
b53866265161a112da91bf9e94a1ed35db3c0f68fa70a874eeaec48fb0dcb63d,
passes both original and selected-four cold/warm full CPU13 vectors/carries/
greedies and real rejection repair using the unchanged original reference and
absolute0.006/relative0.0003 bounds. Selected work completes three genuine
count4 chunks/2256 equal Verify4 calls with original tail behavior; default/
cold/prefill count4 stays false/zero. The unchanged v174 numerical operators
are reused because this helper correction changes no arithmetic, shader or cap.
Complete original target/draft/432 HOST owner closures and same-boot baseline
50688000B, all HTTP200, idle/unpaused admission and zero classified GPU faults
pass after both runs. Aggregate receipts are in
v178_gdn_retained_metadata_cpu13_acceptance. Correctness work overlapped CPU
source compilation; its short-fixture rates carry no performance credit.
Compiler-idle frozen before/after normal timing is separately pending.

## Normal frozen before/after result

Compiler-idle selected-four code200 on v178 completes199 real continuations,
147of155 actual drafts/52 rounds/51 count4 chunks and five genuine target
recomputations, with zero repaired-vector maximum absolute error and all owner
closures/recovery. It measures10.704157 real continuations/s versus v177's
10.740576: about0.34percent slower in this one before/after comparison, so
there is no measured speed gain. Verify13.196322s versus13.227285s is nearly
unchanged; draft3.145481s versus3.049685s differs between runs. Global waits
608versus526 and wait/map timers are overlapping dual-runtime diagnostics,
not isolated GDN costs. The real accepted work/prefix counts are identical.
No repeatedABBA/prose/native30 or installed acceptance is claimed. Exact raw
aggregate/source/host receipts: v178_gdn_retained_metadata_four_normal.
