# Native matrix ISA and actual-device admission, 2026-10-20

The public native-wide operator capture demonstrates register spilling in the
actual selected dense kernel. It does not yet establish a faster replacement.
The installed accepted worker remains unchanged by this source preparation.

The root-controlled protected capture is under
`/media/jmicko/faststore/kuno-development/strata-parity-native-wide-20261020-operators/`:
`receipt.json` records 103 actual math controls; `compiler-public-stderr.txt`
contains only fixed-public numerical shaders and their compiler code. Child-only
`INTEL_DEBUG=cs MESA_SHADER_CACHE_DISABLE=true` produced the capture. Diagnostic
compilation/output is excluded from clean throughput comparisons.

| Input | Actual source hash | Compiler summary line | SIMD | Spills:fills | GRFs |
| --- | --- | --- | --- | --- | --- |
| BF16 | `0x78acc5b96524ea9b` | 27014 | 16 | 91:77 | 128 |
| FP16 | `0xe4d13532fa4e0eee` | 56964 | 16 | 91:77 | 128 |

Their compiler BLAKE3 hashes are respectively
`55369a7a48d70948b3e8f99c7b0f29f3d8015d423dd0d36d28b11a95c366b523` and
`8835a37075dd2474d65a61516104eec81baf1e844ac75f8b86aa2830e5681b97`.
Both show literal workgroup 256, required subgroup 16 and shared size 40,960 B.
The original numerical-case order, aligned 512x128 admission and unique compiled
geometry identify `dense_wide` rather than a QSA/GDN primitive. No individual
case header was emitted immediately before each compiler dump.

The two K32 loops, lines 27532–28165 and 57485–58117, each contain 27 scratch
loads, 40 scratch stores and 16 DPAS instructions. These are static ISA counts;
320 iterations at K=10,240 imply 21,440 scratch messages per hardware thread.
The compiler's estimated cycles and instruction counts are not measured latency.
The accepted native M32 tile actually uses 32 KiB shared memory, not just its
24 KiB paired input buffers. The device's 49,152-byte per-workgroup limit is
not a measurement of the entire hardware SLM pool or residency.

Installed Mesa is 26.2.5. No supported process-local pre-Xe3 switch to force
256 GRFs was found in its library strings or the inspected primary source.
Current [Mesa environment documentation](https://docs.mesa3d.org/envvars.html)
describes the newer threads-per-EU controls as Xe3-specific. They are not a
justified B70 tuning option. A supported lower-register tile is the next bounded
candidate; neither a guessed environment variable nor a SPIR-V register hint
is enabled by this change.

The native Graph now retains actual `vkGetPhysicalDeviceProperties2` subgroup
and compute limits. Before allocating a native shader/pipeline it reads the
selected compute entry's literal LocalSize, checks all dimensions and the
overflow-safe invocation product, verifies required width against actual device
width bounds and checks `ceil(invocations / required_width)` against actual
`maxComputeWorkgroupSubgroups`. Rust kernels also carry their original compiler
shared footprint for the actual device cap. Automatic-width kernels retain the
driver's choice; WGSL retains its original Wgpu/Naga shared-memory admission.
Specialized sizes and overriding WorkgroupSize builtins are rejected in this
bounded literal-only native path. Existing selected sources use neither.

This follows the actual [Vulkan subgroup properties](https://docs.vulkan.org/refpages/latest/refpages/source/VkPhysicalDeviceSubgroupSizeControlProperties.html)
and [required-width contract](https://docs.vulkan.org/refpages/latest/refpages/source/VkPipelineShaderStageRequiredSubgroupSizeCreateInfo.html).
No constant B70 limit replaces the physical query. Pipeline cache identity,
resource ranges, metadata caps, flight/nonce/fence ownership, rolling submission,
recovery quarantine and numerical controls are unchanged.

`test-native-program-device-limits.py` compiles the exact production parser and
arithmetic with rustc and nine metadata-only CPU tests. It covers 32-subgroup
admission/rejection, partial groups, existing shapes, automatic widths, overflow,
all device caps, selected entry, malformed extents and nonliteral overrides.
The fixtures are not executable shaders. Full worker compilation, original
shader controls, actual GPU correctness/recovery and clean whole-model timing
remain root-controlled acceptance gates; no new speed or installation is claimed.
