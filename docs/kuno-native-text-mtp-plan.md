# Native Unsloth quant and Qwen4Exp MTP

The user approved the Unsloth UD-Q3_K_XL main model and self-contained Q8_0 MTP
head, and explicitly chose our Rust implementation. No replacement C/C++ text
engine is being introduced. This is internal laboratory work; production routing
and Vashti consumers are unchanged.

## Backend direction approved by the user

Kuno should prefer a supported native Rust backend and use Ollama when it has no
supported native implementation. Support must cover the actual model artifact,
geometry, quant formats and requested inference behavior, not just its model
name. Choose the backend before execution. A native runtime failure after work
has begun is not permission to replay silently in Ollama. Keep an explicit
Ollama selection for comparisons. Its exact naming/API contract remains future
shared-contract work; this milestone adds no routing or catalog override.

## Pinned artifacts and storage

Publisher: `unsloth/Qwen3.8-Flash-Next-GGUF`, revision
`766911a6b7369840a91dbcd95f9f997acaab6cd6`.

- UD-Q3_K_XL is three original GGUF shards totaling89,986,353,824bytes.
  Shard0 is metadata-only; shards1/2 contain402/822 tensors,1224 total.
- Its actual formats are F32(557), BF16(24), Q8_0(502), Q6_K(1),
  IQ3_XXS(94), IQ4_NL(44), IQ4_XS(2). Nominal Q3 does not imply uniform packing.
- The self-contained Q8 MTP head is4,137,429,120bytes with34 tensors:
  Q8_0(21), F32(11), BF16(2). It stores its own embedding and output head,
  simplifying a future independent RTX3060 owner.
- Verify complete SHA256 values before import or inference. Keep one payload on
  NVMe: Ollama's existing blob store plus same-filesystem named hardlinks for
  split views. An unreferenced MTP sidecar must not rely on Ollama's blob GC.
- Ollama0.35.1 supports local multi-shard `create`, despite the remote registry
  rejecting shard pulls. Its pinned b11232 backend lacks this MTP graph. The
  ordinary model remains a possible comparison; a DRAFT import alone would not
  establish supported MTP execution.

The public download receipt and bounded tensor inventories live under
`/media/jmicko/faststore/kuno-development` and the model import directory. They
contain public weight provenance only. No private content or expert histories
belong in them. Download/import completion is recorded separately after checks.

## First native MTP reference

`crates/kuno-text/src/mtp.rs` binds only the inspected self-contained Q8 head.
It independently translates the [pinned upstream graph](https://github.com/danielhanchen/llama.cpp/blob/6fcaa16f4b360649933a54d1f91ad40ed35c0e11/src/models/qwen4exp.cpp)
into Rust; no upstream engine is linked. Exact tensor names, dimensions and
types are whitelisted. Metadata requires49 blocks, one NextN block, four
2560-wide HC streams, LR320,512 experts/top10/FF640,24 query/2 KV heads,
256-wide heads,64 rotary dimensions, theta1e7 and RMS epsilon1e-6. All34 stored
tensors are checked, including four unused indexer tensors.

For each token, normalize the preceding10240-wide residual per HC stream,
normalize its2560-wide embedding, concatenate embedding then that stream's
hidden state, and apply the same5120→2560 EH projection four times. Perform
HC mixer → dense causal GQA → HC combine → HC mixer → routed and shared MoE →
HC combine. Return the resulting wide residual before the final HC head mixer
and248320-wide output projection. Attention Q and gate are interleaved per head.
Block48 uses dense attention, independent of ordinary trunk modulo scheduling
and the stored QSA indexer. Reference context is explicitly limited to2048.

Each sequence owns its KV cache and binding identity. A rejected preflight input
leaves state unchanged. An execution error clears/poisons it until reset. Draft
rewind clears the provisional KV suffix; target GDN/PLE rollback is a separate
unimplemented transaction. Returned logits/hidden and MTP-owned intermediate
buffers use wiping guards. Existing scalar normalization/HC helpers have not
received whole-stack error-path erasure acceptance; this reference remains a
public-fixture laboratory path, never a live HTTP/private-request engine.

Unit tests cover embedding-first per-stream conditioning, per-head Q/gate
ordering, independent dense-attention equations,4-query/2-KV grouping, all
visible positions past2048, cancellation, and observed cache wiping helpers.
An opt-in downloaded-head test checks strict real artifact binding, complete
graph scheduling with a synthetic executor, cross-binding rejection, failure
after KV publication, reset and fresh replay. It proves neither full-head
numerical parity nor useful speculative performance.

## Following implementation and acceptance

1. Complete native split ownership and original-packed IQ3_XXS/IQ4_NL/IQ4_XS
   scalar/GPU operators. Preserve original Q4 gates and independent packed
   reference fixtures. Do not expand all quant weights to F32 to claim support.
2. Bind and validate the complete smaller main artifact and obtain its own
   fixed public correctness fixture. Different quants cannot reuse Q4 token IDs
   as a quality/reference oracle. Validate complete head numerical output too.
3. Add an explicit target pre-final-mixer wide hidden tap and logits for every
   verification position. Existing MAX8 LastLogits only projects the last row.
4. Start with at most two drafts: confirmed token plus0–2 drafts uses a1–3-row
   target verification. Commit only the accepted prefix and produce the target
   correction/bonus token. Greedy equality first; exact sampled rejection is a
   separate milestone. Never substitute resident experts to manufacture speed.
5. Add device-owned accepted-prefix GDN recurrent **and raw convolution** states;
   preserve PLE history and QSA cursors. Merely changing sequence.position is
   wrong. Two additional GDN recurrent snapshots cost216MiB logical memory.
   Avoid whole-state host checkpoint copies on every speculative round.
6. Repair draft KV with real target hidden states after acceptance. Pair token
   at positionp with target hidden atp−1. A provisional draft's own hidden state
   must not remain authoritative in accepted catch-up history.
7. Build dedicated dense-attention and all-resident Q8 draft owners, explicit
   expert-buffer sharding and cross-device bridges. The existing main/secondary
   slot limits do not authorize a512-expert resident draft by changing a number.
8. Compare the same public fixture with MTP off, same-device draft, and RTX3060
   draft/B70 target. Include load/prefill, accepted-token throughput, acceptance
   rate, bridge/repair costs, context, physical disk I/O and GPU recovery. Do not
   present draft tokens/s as accepted response throughput.

See the [pinned speculative driver](https://github.com/danielhanchen/llama.cpp/blob/6fcaa16f4b360649933a54d1f91ad40ed35c0e11/common/speculative.cpp),
[rollback fix](https://github.com/ggml-org/llama.cpp/pull/28123), and
[publisher's MTP guide](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF/blob/766911a6b7369840a91dbcd95f9f997acaab6cd6/MTP/README.md).
Published speedups are not local acceptance. No GPU reset, driver/clock change,
installed binary/service change, release, API change, or Vashti work is included.
