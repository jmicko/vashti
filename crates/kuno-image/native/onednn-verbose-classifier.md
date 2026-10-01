# Synthetic oneDNN execution classifier

This counter-only helper is pinned to the private bundle's oneDNN 3.11.4 and
verbose format `v1`. It is a benchmark tool, not a content-worker logger. Raw
verbose must stay in RAM; retain only the returned numeric counters. Never
enable it while processing prompts, references, or generated images.

Build/run CPU regressions:

```bash
c++ -std=c++17 -O2 -Wall -Wextra -Werror \
  crates/kuno-image/native/onednn-verbose-classifier-test.cpp \
  -o "$PRIVATE_BUILD/onednn-verbose-classifier-test"
"$PRIVATE_BUILD/onednn-verbose-classifier-test"
```

For a fresh isolated synthetic process executing only the known linear
MatMul → scale → SoftMax → MatMul SDP graph, set `ONEDNN_VERBOSE=all` with no
component filter. Capture complete stdout in RAM, then feed it through the
tool's stdin. Set these flags only after verifying that capture was complete,
unfiltered, untruncated, and came from that dedicated known graph:

```bash
"$PRIVATE_BUILD/onednn-verbose-classifier-test" \
  --stdin --capture-complete --isolated-chain
```

The tool accepts bounded 16 KiB lines, retains empty CSV fields, handles optional
numeric timestamps, and returns numeric JSON. It never echoes records or stores
tensor metadata, IDs, or raw messages. Creation/dispatch/check/compile mentions
are not counted as executions. Unknown versions, malformed records, enabled
filters, unknown SDP implementations, reference implementations, GPU MatMul or
Softmax execution, and decomposed SDP graphs prevent optimized-path evidence.

`graph_sdp_paths_internal_unknown` counts exact strict GPU SDP graph executions
with implementation `sdp_primitive_kernel_t` and backend `dnnl_backend`. This
counter alone does not prove optimized fusion. The selected kernel runs the
`fuse_sdpa` pass, but that pass can succeed with no matching candidates.
[Dispatch](https://github.com/uxlfoundation/oneDNN/blob/v3.11.4/src/graph/backend/dnnl/kernels/sdp.hpp#L49),
[empty-candidate behavior](https://github.com/uxlfoundation/oneDNN/blob/v3.11.4/src/graph/backend/dnnl/passes/transform.cpp#L4378).

`optimized_graph_evidence=1` additionally requires the exact 3.11.4 version
banner and successful nested creation of GPU SDP implementation
`ocl:micro:reusable`, coupled with graph execution in the dedicated capture and
zero conflicting observations. `ONEDNN_VERBOSE=all` enables ordinary debuginfo;
the nested-creation proxy prints after successful creation. A creation record
alone proves no execution, and the aggregate predicate must not be used to
correlate unrelated graph instances or mixed-shape/mixed-workload tests.
[Nested creation](https://github.com/uxlfoundation/oneDNN/blob/v3.11.4/src/common/primitive_desc.hpp#L444),
[optimized implementation](https://github.com/uxlfoundation/oneDNN/blob/v3.11.4/src/gpu/intel/sdpa/micro.hpp#L109),
[reference implementation](https://github.com/uxlfoundation/oneDNN/blob/v3.11.4/src/gpu/intel/sdpa/ref.hpp#L35).

Do not require nested `primitive,exec,gpu,sdpa` records: the fused graph calls its
internal primitive directly, bypassing that execution logger. Successful device
waits, numerical validation, and a peak-memory bound remain mandatory independent
checks; the graph executable also ignores the internal SDP return status. The
predicate does not prove image quality, absence of every internal allocation,
or production performance.
[Graph SDP execution](https://github.com/uxlfoundation/oneDNN/blob/v3.11.4/src/graph/backend/dnnl/executables/sdpa.cpp#L200).

Without timestamps, graph execution has 14 fields: API 2, operation 3, engine 4,
partition kind 6, implementation 11, backend 12, elapsed 13. Primitive records
have 13 fields: API 2, operation 3, engine 4, primitive kind 5, implementation 6,
elapsed 12. Timestamps insert a field at index 2 and shift subsequent positions.
These are source-derived offsets; older documentation examples use `v0`.
[Record formatting](https://github.com/uxlfoundation/oneDNN/blob/v3.11.4/src/common/verbose.hpp#L89),
[graph field construction](https://github.com/uxlfoundation/oneDNN/blob/v3.11.4/src/graph/utils/verbose.cpp#L224).

CPU regressions cover 14 source-shaped records, timestamp and empty-field
handling, misleading dispatch/check/creation text, exact-version rejection,
malformed/nonfinite/truncated/oversized records, missing evidence, conflicting
GPU execution, and explicit capture/isolation flags. These CPU tests do not
establish GPU execution.
