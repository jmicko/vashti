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

## Native 32-subgroup follow-up

The separately protected fixed-public capture in
`strata-parity-native-wide32-20261020-operators` now passes all 103 actual
operator controls and physical recovery. Its receipt identifies source
`3fffc266383917cada88fd176b273db8a6b737f4` and worker SHA256
`eac78ce0029205c9d503c78a1b2c868082801cf966b4e8a3d3bc7585ece80f4c`.
This follow-up establishes correct execution and actual spill elimination;
whole-model throughput is a separate pending measurement.

| Input | Actual source hash | Header / summary lines | K32 loop lines |
| --- | --- | --- | --- |
| BF16 | `0x06a70106bd9a861c` | 24588 / 24589 | 24674–24834 |
| FP16 | `0x0ac876bac0bc0d62` | 49581 / 49582 | 49670–49838 |

The respective compiler BLAKE3 identifiers are
`3577bf71f80bfd417ed0c2278a102914043549d53abae76822ec0f673ad75a02` and
`5a81aee046cb503f836148ef2d4a84c629aef473c09f73569e6fff492201bb27`.
Final NIR at lines 23148–23155 and 48137–48145 confirms workgroup 512x1x1,
required/min/max subgroup 16 and real shared allocation 49,152 B. The aligned
512x128 numerical-case selection and unique geometry attribute these shaders
to the native32 dense entry. BF16 runs first in the actual original probe;
the second shader also retains the FP16 float-control execution mode.

Both summaries report zero spills/fills and 128 GRFs. BF16 has 1,207 static
instructions, FP16 1,212, with 283 non-scratch sends and one K loop each.
The following counts are the actual static instructions inside one K32 loop,
per hardware thread/subgroup, not measured traffic or elapsed time:

| Instruction/message kind | Prior 16-subgroup tile | New 32-subgroup tile |
| --- | --- | --- |
| Scratch loads | 27 | 0 |
| Scratch stores | 40 | 0 |
| DPAS | 16 | 8 |
| Global loads | 8 | 4 |
| SLM loads | 40 | 24 |
| SLM stores | 4 | 2 |
| SLM fence | 1 | 1 |
| Barrier message | 1 | 1 |
| All sends, including scratch | 121 | 32 |

Workgroup totals preserve 256 DPAS instructions per K32 iteration
(`16*16 == 32*8`) and 128 global-load instructions (`16*8 == 32*4`).
Scratch sends drop from 1,072 per workgroup/iteration to zero. SLM load-message
counts rise from 640 to 768 after subgroup normalization; payload sizes and
hardware scheduling still matter, so instruction ratios are not a speedup
prediction. The complete accepted output tile and original ordered K16
arithmetic remain unchanged. No GPU generation/inference content was captured.
