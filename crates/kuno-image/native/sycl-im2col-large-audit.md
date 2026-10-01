# Large SYCL IM2COL audit

This is a backend-only development diagnostic. It changes only the numeric
oracle, not patch 003, production selection, inference policy or services.
The existing native patch remains `c60d71b5af2a4e7f904a921b0317a28b3dc1f2f54639c41163e3d998e0f3d090`.

The coordinating task reported a successful 1248x832 full edit after bounded
launches, followed by a device-lost failure during 1184x1776 decoding. The latter
was detected at queue synchronization, so the faulting VAE operation is not
established. A reset or device-lost status alone does not prove OOM, watchdog
expiry, IM2COL indexing failure or a driver defect.

## Source arithmetic

The 2D IM2COL row, width, kernel-channel, source offset and destination offset
calculations are signed 64-bit. There is no source-level cast of the full tensor
index to 32 bits. The `dpct::dim3` dimensions narrow to unsigned 32-bit, but each
component is independently small and the bounded global product is at most
INT32_MAX. The helper guards `padded_channels * width` before multiplying.
For width 1184 and 288 channels, the exact cap is 644 rows, with chunks
644 + 644 + 488 at height 1776. For 144 channels the cap is 1180, with chunks
1180 + 596. Destination indices remain relative to the full output tensor;
chunking neither shrinks the allocation nor changes strides or precision.

The output sizes cross distinct boundaries:

| Oracle mode | Width x height | Channels | F16 elements | F16 bytes | Input F32 bytes |
| --- | --- | ---: | ---: | ---: | ---: |
| 5 | 1184 x 1776 | 144 | 2,725,208,064 | 5,450,416,128 | 1,211,203,584 |
| 6 | 1184 x 1776 | 288 | 5,450,416,128 | 10,900,832,256 | 2,422,407,168 |
| 7 | 1184 x 1399 | 288 | 4,293,430,272 | 8,586,860,544 | 1,908,191,232 |
| 8 | 1184 x 1400 | 288 | 4,296,499,200 | 8,592,998,400 | 1,909,555,200 |

Modes 7 and 8 bracket UINT32_MAX **element** indices while changing the height
by one row. This differs from the already accepted edit output, which crosses
4 GiB of bytes but stays below UINT32_MAX elements. The maximum destination
index in mode 6 is 5,450,416,127 and is valid in signed 64-bit arithmetic.

## Compiled addressing audit

An independent read-only extraction checked the actual failed worker with
SHA256 `8a6d07e18aefdc77fdbbbfface1e87085132378b82c6d8a06c11a7e7fdfb1a70`.
Its embedded F16 IM2COL SPIR-V uses Physical64 addressing. All 24 integer
multiply/add/subtract instructions in the 2D kernel and all 40 in the 3D kernel
use 64-bit integer types; destination pointer-access-chain indices are 64-bit,
with no narrowing conversions in either kernel. The binary also embeds the
`-ze-intel-greater-than-4GB-buffer-required` compile option. This excludes a
demonstrated source/SPIR-V UINT32 offset wrap in those kernels. It does not
establish final driver machine-code correctness, safe full-decoder residency,
or the cause of the asynchronous device loss.

## Independent oracle extension

The existing `sycl-im2col-test.cpp` now samples around 2, 4 and 8 GiB byte offsets,
signed/unsigned 32-bit element indices, and both sides of every bounded-launch
row boundary. It retains independent padding/stride/dilation reference math,
bounded 1 MiB input staging, no full-output host staging, and no saved data.
Modes 1 through 4 retain their earlier shapes; modes 5 through 8 are explicit
large GPU diagnostics, never part of the default CPU invocation.

Default CPU acceptance passes 272,704 exact values. Direct calls to the pinned
SDK host range helper accept 18 bounded ranges with complete row coverage,
including all four new geometries. It still rejects the original unbounded edit
range. GPU runs of the new modes remain scheduled and recorded by the root task:

```
sycl-im2col-test --device SYCL <physical PCI address> 5
sycl-im2col-test --device SYCL <physical PCI address> 6
sycl-im2col-test --device SYCL <physical PCI address> 7
sycl-im2col-test --device SYCL <physical PCI address> 8
```

Run one mode at a time under the established no-swap, core-dump and Landlock
policy, with admission quiesced and numeric-only output. If a device fault occurs,
stop subsequent diagnostics until the coordinating task establishes recovery.
The test is functional, not a cross-engine performance benchmark.

Standalone IM2COL writes directly into its destination. It does not reproduce
VAE layout-copy operations, vendor GEMM, retained backend pools or the full
post-sampling residency. In particular, mode 6's input size exceeds signed
INT32_MAX bytes; a separate review owns activation-copy offset arithmetic.
Passing these modes must not be treated as full-decoder acceptance.
