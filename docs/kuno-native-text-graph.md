# Borrowed Qwen3.8-Flash-Next GGUF: implementable Rust graph contract

Prepared 2026-10-05 from pinned primary sources and installed artifact metadata.
This is the semantic implementation contract; acceptance receipts live in
`kuno-text-performance.md`. Formulas are not proof of complete engine acceptance.

## Sources and identity

- Workspace reviewed: `/home/jmicko/.codex/worktrees/kuno-image-speed/vashti`,
  branch `codex/kuno-native-text`, base `e9de15a`. Read `AGENTS.md`, active
  `docs/kuno-vashti-coordination.md`, research and Bonsai contracts.
- Actual metadata: `/tmp/kuno-qwen-text-research-93myra8q/ollama-qwen38-show-metadata.json`.
  Target `qwen3.8-flash-next:125b-a6b-q4_K_M`; 1,224 listed tensors;
  architecture `qwen4exp`; main manifest digest
  `f79436a1cc83fd1a9b7752cf62247a2e7ad9fd7775c67230da4254fcc1d07f01`.
- HF generated reference:
  `/tmp/kuno-qwen-text-research-93myra8q/modeling_qwen4_exp.py`, Transformers
  `0a896aa41bba78f92338db21cc468fe043888657`.
- Strata: `/tmp/strata-source-audit-nZgk5D/Strata`,
  `6f32ec070f23ced9f50e704d854d775da52591ab`.
- Additional fresh primary GGUF reference (downloaded source only): llama.cpp
  `806eee9841de5f2c20f9d43914117f157d2baacc`; files in
  `/tmp/kuno-qwen-gguf-reference/`, slash paths replaced with `__`.
  Relevant upstream paths: `conversion/qwen.py`, `conversion/qwen4exp.py`,
  `src/models/qwen4exp.cpp`, `src/llama-graph.cpp`, `src/llama-vocab.cpp`,
  `ggml/src/ggml-quants.c`. This pin is a contemporary semantic reference, not
  established provenance of the downloaded artifact's original conversion.

Primary-source permalink prefix for that extra reference:
`https://github.com/ggml-org/llama.cpp/blob/806eee9841de5f2c20f9d43914117f157d2baacc/`.

## Decisions that prevent plausible-looking wrong output

1. **Use GGUF conventions, not unadjusted HF tensor conventions.** GGUF norms
   already fold `(1 + HF gamma)`; apply their stored multiplier directly.
   `ssm_a` already contains `-exp(A_log)`; do not exponentiate/negate it again.
2. **GDN head pairing is modulo on this GGUF.** Value head `h` uses key/query
   head `h % 16`. HF uses `floor(h / 3)` before conversion; the converter
   permutes every corresponding value-head-associated tensor to tiled order.
3. **QSA uses a different pairing:** query head `h` uses KV head `h / 12`.
4. **GDN's closing gate is sigmoid**, not SiLU. The short QKV convolution uses
   SiLU. Expert gate activation and PLE convolution also use SiLU.
5. **QSA's Q projection is per-head `[query256, gate256]`**, repeated 24 times;
   not global `[all queries, all gates]`, and not alternating scalar Q/gate.
6. **PLE convolution consumes normalized gated values**; its history contains
   those normalized rows, not embeddings or raw gated values. Taps read lags
   `[9, 6, 3, 0]` in that order.
7. **PLE boundary 248044 differs from generation stop 248046.** Do not collapse
   them into a single `eos` variable.
8. **Select complete QSA blocks + exact incomplete tail.** Do not blindly port
   Strata's legacy fixed-width cell top-k routine; its `min(nkv, 2051)` width
   can admit partial extra complete blocks when the actual tail has <3 tokens.

## Concrete dimensions and tensor layout

Use model metadata for validation. Initial implementation may explicitly
support exactly this geometry and reject others instead of assuming defaults.

`D=2560`, `HC=4`, `HCD=10240`, `LR=320`, `NL=48`, `V=248320`, `E=512`,
`K=10`, expert intermediate `F=640`, shared intermediate `FS=640`.
QSA layers are zero-based `3,7,11,...,47`; all other layers are GDN. PLE executes
at zero-based layer 1, before its attention hyperconnection read. RMS epsilon
is `1e-6`. No separate traditional attention/FFN layer norms exist.

GGUF dimension zero is fastest. A tensor shape `[I,O]` is a row-major CPU
matrix `W[o][i]` with dot `y[o] = sum_i W[o][i] * x[i]`. A tensor
`[I,O,E]` is `W[e][o][i]`, so one expert is a contiguous run of `O` packed
rows. Matrix rows must be sized using the actual quant block size, not
`I * nominal_bits / 8`.

| GGUF name | GGUF shape | Rust meaning | Actual type on representative layer |
| --- | --- | --- | --- |
| `token_embd.weight` | `[2560,248320]` | `embedding[token][d]` | Q4_K |
| `output.weight` | `[2560,248320]` | `logits[token]=dot(output[token],mixed)` | Q6_K |
| `output_hc_norm.weight` | `[10240]` | grouped norm multiplier | F32 |
| `output_hc_down.weight` | `[10240,320]` | final mixer down | Q4_K |
| `output_hc_up.weight` | `[320,10240]` | final mixer up | Q5_0 |
| `blk.l.hc_attn_norm.weight` | `[10240]` | attention GR norm multiplier | F32 |
| `blk.l.hc_attn_down.weight` | `[10240,320]` | GR down | Q4_K |
| `blk.l.hc_attn_up.weight` | `[320,10240]` | GR up | Q5_0 |
| `blk.l.hc_attn_inject.weight` | `[10240,4]` | GR injection logits | Q4_K |
| `blk.l.hc_ffn_norm.weight` | `[10240]` | FFN GR norm multiplier | F32 |
| `blk.l.hc_ffn_down.weight` | `[10240,320]` | GR down | Q6_K |
| `blk.l.hc_ffn_up.weight` | `[320,10240]` | GR up | Q5_0 |
| `blk.l.hc_ffn_inject.weight` | `[10240,4]` | GR injection logits | Q4_K |
| `blk.l.ffn_gate_inp.weight` | `[2560,512]` | router | F32 |
| `blk.l.ffn_gate_inp_shexp.weight` | `[2560]` | shared expert scalar gate | F32 |
| `blk.l.ffn_gate_exps.weight` | `[2560,640,512]` | per-expert gate | Q4_K |
| `blk.l.ffn_up_exps.weight` | `[2560,640,512]` | per-expert up | Q4_K |
| `blk.l.ffn_down_exps.weight` | `[640,2560,512]` | per-expert down | Q8_0 or Q5_0 by layer |
| `blk.l.ffn_gate_shexp.weight` | `[2560,640]` | shared gate | Q4_K |
| `blk.l.ffn_up_shexp.weight` | `[2560,640]` | shared up | Q4_K |
| `blk.l.ffn_down_shexp.weight` | `[640,2560]` | shared down | Q8_0 or Q5_0 by layer |

Do not assume the representative type applies to every tensor with the same
role. Dispatch using each TensorInfo. Counts in actual metadata: Q6_K 49,
Q4_K 568, F32 388, Q5_0 146, Q8_0 48, F16 1, BF16 24. The corresponding
operator must support every such encoding without repacking the source files.

GDN-only tensors:

| Suffix after `blk.l.` | Shape | Meaning | Type |
| --- | --- | --- | --- |
| `attn_qkv.weight` | `[2560,10240]` | concatenated Q2048,K2048,V6144 | Q6_K |
| `attn_gate.weight` | `[2560,6144]` | output gate Z | Q4_K |
| `ssm_alpha.weight` | `[2560,48]` | decay step projection A | Q4_K |
| `ssm_beta.weight` | `[2560,48]` | write-strength projection B | Q4_K |
| `ssm_conv1d.weight` | `[4,10240]` | `kernel[channel][tap]`, tap fastest | F32 |
| `ssm_dt.bias` | `[48]` | decay bias | F32 |
| `ssm_a` | `[48]` | negative exp of original A_log | F32 |
| `ssm_norm.weight` | `[128]` | per-value-head norm multiplier, shared across heads | F32 |
| `ssm_out.weight` | `[6144,2560]` | output projection | Q4_K |

QSA-only tensors:

| Suffix | Shape | Meaning | Type |
| --- | --- | --- | --- |
| `attn_q.weight` | `[2560,12288]` | 24 `[Q256,G256]` heads | Q4_K |
| `attn_k.weight` | `[2560,512]` | 2 K heads of 256 | Q4_K |
| `attn_v.weight` | `[2560,512]` | 2 V heads of 256 | Q6_K |
| `attn_output.weight` | `[6144,2560]` | attention output | Q4_K |
| `attn_q_norm.weight` | `[256]` | per-head norm multiplier | F32 |
| `attn_k_norm.weight` | `[256]` | per-head norm multiplier | F32 |
| `indexer.q_proj.weight` | `[2560,512]` | 4 query heads of 128 | BF16 |
| `indexer.k_proj.weight` | `[2560,128]` | single raw key | BF16 |
| `indexer.q_norm.weight` | `[128]` | per-index-query-head norm multiplier | F32 |
| `indexer.k_norm.weight` | `[128]` | per-pooled-key norm multiplier | F32 |

PLE-only tensors:

| Name/suffix | Shape | Meaning | Type |
| --- | --- | --- | --- |
| `per_layer_token_embd.weight` | `[160,320001536]` | 160-wide gathered row | Q5_0 |
| `blk.1.ple_key.weight` | `[2560,10240]` | one key per residual stream | Q4_K |
| `blk.1.ple_value.weight` | `[2560,2560]` | value shared by all streams | Q4_K |
| `blk.1.ple_norm_key.weight` | `[10240]` | grouped direct multiplier | F32 |
| `blk.1.ple_norm_query.weight` | `[10240]` | grouped direct multiplier | F32 |
| `blk.1.ple_norm_conv.weight` | `[10240]` | grouped direct multiplier | F32 |
| `blk.1.ple_conv1d.weight` | `[4,10240]` | `kernel[channel][tap]` | F16 |

Tensor mapping evidence: HF modeling lines 461-520, 815-844, 900-997,
999-1034, 1179-1252, 1373-1514; actual show metadata. GGUF load shapes and
normalization/graph mapping: llama `src/models/qwen4exp.cpp:196-308,314-399`.

## Whole trunk graph

For every input token position t, gather D-wide embedding e and initialize
the residual stack `R[c][d]=e[d]` for each of four streams.
Process all positions in a prefill batch with per-position causality/state
equivalent to this sequential definition:

```
for l in 0..48:
    if l == 1: R = ple(R, original_token_ids, ple_state)
    (x, injection_logits) = gr_read(R, attention_hc_weights[l])
    a = gdn(x, gdn_state[l]) or qsa(x, positions, qsa_state[l])
    R = gr_write(R, a, injection_logits)
    (x, injection_logits) = gr_read(R, ffn_hc_weights[l])
    f = full_moe_with_shared(x, ffn_weights[l])
    R = gr_write(R, f, injection_logits)
mixed = gr_read_no_inject(R, output_hc_weights)
logits = output.weight @ mixed
```

The final mixer **is** the output norm. There is no extra output RMSNorm,
embedding sqrt(D) multiplier, residual sum in place of mean, or dropped expert.
Layer graph evidence: HF 1254-1307,1373-1500; llama 435-525.

## Exact gated residual (GR)

For each stream c, normalize over D features:
`XN[c,d]=R[c,d]/sqrt(mean_d(R[c,d]^2)+eps) * stored_gamma[c,d]`.

Flatten stream-major (`c*D+d`), then:

```
lo[r]   = silu(dot(W_down[r], XN) / HC)
gate[c,d] = sigmoid(dot(W_up[c*D+d], lo))
x[d]    = sum_c (XN[c,d] * gate[c,d]) / HC
j[c]    = dot(W_inject[c], XN)        # attention/FFN only
Rnew[c,d] = R[c,d] + block_out[d] * (2 * sigmoid(j[c] / HC))
```

Division by HC is **inside SiLU** on the low-rank projection and **inside
sigmoid** on the injection projection. Mixed-input gate has no division by HC.
`W_inject` is absent for final mixer. Keep the pre-block residual R until
gr_write; do not accidentally substitute normalized XN into the residual add.
HF 999-1034; llama 314-399; Strata gr.hpp 78-122.

## Exact GDN token operator

State per GDN layer, per independent sequence:
- Raw QKV convolution history: 3 previous vectors of 10240, initially zero.
  Choose Rust `[channel][history=3]` to match native source, oldest first.
- Recurrence: `S[value_head][key_dim=128][value_dim=128]` F32, initially zero.
  Strata uses transposed `[key_dim][value_head][value_dim]` for coalescing; either
  works if all readers/writers and tests use the same choice.

At a token's mixed input x:

```
p = W_qkv @ x                           # 10240
z = W_gate @ x                          # 48x128
a = W_alpha @ x                         # 48
b = W_beta @ x                          # 48
for channel:
    conv[channel] = sum_{tap=0..3} kernel[channel,tap] *
                      [history0,history1,history2,p[channel]][tap]
    history[channel] = [history1,history2,p[channel]]
u = silu(conv)                          # every Q/K/V channel
q = u[0..2048].reshape(16,128)
k = u[2048..4096].reshape(16,128)
v = u[4096..10240].reshape(48,128)
q[h] /= sqrt(sum_i q[h,i]^2 + 1e-6)
k[h] /= sqrt(sum_i k[h,i]^2 + 1e-6)
q[h] *= 1/sqrt(128)
beta[h] = sigmoid(b[h])
g[h] = stored_ssm_a[h] * softplus(a[h] + stored_dt[h])
decay[h] = exp(g[h])
for value_head h:
    kh = h % 16
    S[h] *= decay[h]                    # BEFORE read-to-update
    predicted[j] = sum_i S[h,i,j] * k[kh,i]
    delta[j] = (v[h,j] - predicted[j]) * beta[h]
    S[h,i,j] += k[kh,i] * delta[j]
    o[h,j] = sum_i S[h,i,j] * q[kh,i]   # UPDATED state
    y[h,j] = o[h,j] / sqrt(mean_j o[h,j]^2 + 1e-6) *
               ssm_norm[j] * sigmoid(z[h,j])
out = W_ssm_out @ flatten(y)
```

Stable softplus: `max(v,0)+ln1p(exp(-abs(v)))`; stable sigmoid branching avoids
overflow. L2 uses **sum**, RMS uses **mean**. No RoPE or KV cache in GDN.

For first correct prefill implementation, use exactly the recurrence in token
order with batched dense projections; that is the independent oracle for a
future chunk-GDN kernel. The official chunk rule uses chunk size 64 but matches
this recurrence mathematically. A prefill chunk beginning after a cached prefix
must carry the raw convolution history and S unchanged into the chunk; sequence
padding must not be committed into states. Compare 1-token, arbitrary chunk,
and all-at-once output/state. Different reductions/dtypes need stated numeric
tolerances, not bit-identical claims.

GGUF/HF permutation mapping: let HF value index `old=g*3+r` (g=keyhead0..15,
r=0..2); GGUF value index `new=r*16+g`. Converter permutes V rows of QKV and
convolution, Z rows, alpha/beta rows, A_log/dt vectors, and columns of output
projection. Do not reapply this permutation in the runtime. To compare against
HF intermediate activations, convert the logical indexing once in test adapters.

Evidence: HF 208-257,259-394,396-458,461-622; llama qwen4exp 1019-1143;
converter qwen.py 395-404,585-634; Strata gdn.cu 43-165. Original HF output gate
configuration is `output_gate_type="sigmoid"`, config.json line105; llama
qwen4exp 636-646 independently confirms artifact gate.

## Exact QSA token operator / prefill causality

Per layer, retain normalized/rotated K and raw V for every live token:
`K[t][2][256]`, `V[t][2][256]`. Keep raw indexer keys `IK[t][128]` or at
minimum completed pooled keys + the current incomplete 0..3 raw-key tail.
Position IDs must remain available for pooled-block start positions.

Compute from the same GR mixed input x:

```
q_full = Wq @ x               # 24x512
Q[h] = rms(q_full[h,0..256], qnorm, eps)
G[h] = q_full[h,256..512]
Kcur = rms((Wk @ x).reshape(2,256), knorm, eps)
Vcur = (Wv @ x).reshape(2,256)
Q, Kcur = rope64(Q,Kcur,current_position)
append Kcur,Vcur
IQ = rms((WIq @ x).reshape(4,128), iqnorm, eps)
IQ = rope64(IQ,current_position)
IKcur = WIk @ x               # raw128, no norm/rotation yet
append raw IKcur
```

For a query at position t, get only valid visible tokens at/before t for its
sequence. Single contiguous sequence: indices `0..=t`. Partition this visible
list into complete 4-member blocks and incomplete tail of length `(t+1)%4`.
For complete block b:

```
pool_raw[b] = mean of IK for its four visible members
PK[b] = rope64(rms(pool_raw[b], iknorm, eps), position_of_first_member)
score[b] = sum_{h=0..3} relu(dot(IQ[h],PK[b])) / sqrt(128)
selected = top min(512,complete_block_count) COMPLETE blocks, then exact tail
ids = union of selected blocks' four tokens plus every incomplete tail token
```

Keep pooling before normalization and rotation. Rotate at the block's first
token position, not last/mean/block-number. Score has ReLU per head before sum,
no indexer softmax, and no learned head weighting. The common factor 1/sqrt(128)
does not affect ordering but retain it in a reference score to independently
compare intermediate numbers. Define deterministic tie policy ascending block
index (HF torch topk does not guarantee this policy; degenerate ties require a
separate fixture rather than assuming exact HF IDs). Sort final selected token
IDs ascending for deterministic attention reduction if that is our chosen
reference; changing order can change last floating bits.

Actual selection length:
`4*min(512,floor(visible_count/4)) + visible_count%4`, <=2051.
All visible tokens are selected when visible_count<=2051; at 2052 there are
513 complete blocks and exactly2048 selected tokens. At 2053 selected count
is2049, 2054=>2050, 2055=>2051, 2056=>2048. Test these boundaries explicitly.

Attention for query head h uses `kv=h/12`:

```
logit[j] = dot(Q[h], K[ids[j],kv]) / sqrt(256)
a[j] = stable_softmax(logit)[j]
O[h,d] = sum_j a[j] * V[ids[j],kv,d]
Y[h,d] = O[h,d] * sigmoid(G[h,d])
out = Wo @ flatten(Y)
```

Cache precision F32 is a useful CPU reference; choose/cache-record explicit
F16/BF16 for GPU acceptance, then quantify its difference. V is never RoPE'd,
Q/K are normed before RoPE. Do not add Strata's optional Hadamard/KV compression
or activation quantization to this first graph.

RoPE text-only: first64 dimensions, NeoX half pairing `(i,i+32)` for i0..31;
frequency `theta_i=10_000_000^(-2*i/64)`, angle=`position*theta_i`.
`x'[i]=x[i]*cos-angle - x[i+32]*sin-angle`;
`x'[i+32]=x[i]*sin-angle+x[i+32]*cos-angle`;
dims64..head_dim unchanged. The model has interleaved MRoPE sections11,11,10;
for ordinary text all three spatial positions equal t, so simple text RoPE
above is exact. Do not claim vision equivalence until three-axis positions are
implemented. Initial benchmark uses no user RoPE overrides/scaling.

Correct prefill can run the token operator above sequentially. A batched QSA
implementation may compute Q/K/V/IQ/IK once for the entire chunk, but each query
must select only complete blocks of its own causal visible prefix; later tokens
must not complete a block for an earlier query. In a chunk crossing the boundary,
query t=2 uses tail[0,1,2]; query t=3 may pool[0,1,2,3]. Precompute global
completed-block keys only if every query separately masks unavailable blocks.
Padding/other sequences never complete this sequence's blocks. No cache-drop
optimization based solely on unused selection: a future query can select any
older block.

Evidence: HF indexer 669-775, attention815-898, RoPE77-146,631-665;
llama qwen4exp741-865,932-1017; Strata qsa.hpp94-113,222-312,
qsa.cu441-506. Strata's legacy cell selection discrepancy is source-confirmed
in qsa.hpp127-137 and qsa.cu315ff; newer block selector still uses fixed
`kStepWidth` in `qsa_select.cu:65-76,146-150`. Do not equate its name to the
official complete-block contract.

## PLE exact integer rows and injection

Consume original tokenizer IDs, not predicted embeddings or arbitrary reordered
position IDs. Text-only first implementation rejects embeddings-only/mixed
vision input. Per sequence keep previous2 token IDs, initialized to missing -1.
Those predecessor IDs are oldest-first; the current token's own EOS does not
cut its context, but any missing/EOS predecessor cuts itself and all older
predecessors to PLE boundary248044.

```
ctx[0] = current
cut = false
for distance s in 1..3:                  # s=1,2
    t = predecessors[2-s]              # newest then oldest
    cut |= t<0 || t==248044
    ctx[s] = if cut {248044} else {t}
mixed2 = u64(ctx0).wrapping_mul(mult0) ^ u64(ctx1).wrapping_mul(mult1)
mixed3 = mixed2 ^ u64(ctx2).wrapping_mul(mult2)
rows[h] = (if h<8 {mixed2} else {mixed3}) % vocab[h] + offset[h]
```

Multipliers `[23703573157769,20109073645365,8052911324071]`.
Vocab sizes:
`[20000003,20000023,20000033,20000047,20000059,20000063,20000069,20000077,
20000081,20000093,20000107,20000147,20000153,20000159,20000161,20000171]`.
Offsets:
`[0,20000003,40000026,60000059,80000106,100000165,120000228,140000297,
160000374,180000455,200000548,220000655,240000802,260000955,280001114,300001275]`.
Read these exact integer arrays from artifact metadata. Do not round through
f32/f64, regenerate with a guessed seed, use signed remainder, add instead of
XOR, or mask instead of modulo. Products for valid token IDs in this artifact
are below signed i64 max, so HF signed-integer path agrees; Rust wrapping u64
is the explicit GGUF contract for overflow.

Gather 16 Q5_0 rows of160 and concatenate head-major `emb[h*160+d]` into2560.
Each row is5 blocks of32,22 bytes/block,110 bytes. The actual data offset is
`GGUF data_section_start + tensor.offset + row_index*110`, not file_start+row.
Borrow store read-only; bounded row cache is model data but **keys/access
ordering reveal request history**, so clear those keys/order at request cleanup.

Q5_0 decode for each22-byte block: bytes0..1 LE f16 scale d, bytes2..5 LE u32
high-bit mask H, bytes6..21 sixteen packed nibbles qs. For j0..15:
`lo=((qs[j]&15)|(((H>>j)&1)<<4))-16`,
`hi=((qs[j]>>4)|(((H>>(j+16))&1)<<4))-16`;
output[j]=lo*d, output[j+16]=hi*d. Not interleaved output[2j],output[2j+1].

Inject before layer1 GR read. From HCD residual R and gathered emb:

```
key = grouped_rms(Wple_key @ emb, ple_norm_key, D=2560)
value = Wple_value @ emb                 # D, shared across streams
query = grouped_rms(R, ple_norm_query, D=2560)
s[c] = dot(key[c],query[c]) / sqrt(D)
sqrtgate[c] = sign(s[c]) * sqrt(max(abs(s[c]),1e-6))
gated[c,d] = sigmoid(sqrtgate[c]) * value[d]
normalized = grouped_rms(gated,ple_norm_conv,D)
conv[c,d] = silu(sum_{tap=0..3} Wple_conv[c,d,tap] *
                       normalized[t - (3-tap)*3,c,d])
Rout = R + gated + conv
```

Out-of-history normalized rows are zeros; remember last9 rows after each
token/chunk. sign(0)=0, so exactly zero dot yields sigmoid(0)=0.5 despite the
magnitude floor. Do not apply sqrt to an already-sigmoided score. PLE token
history and normalized convolution history are different buffers. EOS affects
hash context but **does not automatically zero convolution state** in these
references; only independent sequence/reset does. Chunked prefill must carry
all9 normalized rows even if the current chunk has1 token.

Evidence: HF1037-1176,1179-1252; llama qwen4exp1239-1282,1379-1466;
Strata ngram.cpp54-90, ple.hpp3-25; ggml-quants.c500-524 Q5_0 decode.

## Full routed + shared MoE

For x[D], router logits `L[e]=dot(router[e],x)` for all512; stable F32
softmax over all512; choose largest10 by probability. Fix descending score,
ascending expert-ID ties for reproducibility. Selected probabilities divide by
their sum (GGUF reference clamps denominator to at least2^-14; preserve it).
There are no group gates, no expert-bias correction tensor, no expert pruning,
and no renormalization to exclude CPU/disk misses.

```
for selected expert e with weight w:
    gate = Wgate[e] @ x                   # 640
    up   = Wup[e] @ x                     # 640
    a    = silu(gate) * up
    y_e  = Wdown[e] @ a                   # 2560
    routed += w * y_e
shared = Wshared_down @ (silu(Wshared_gate@x) * (Wshared_up@x))
shared *= sigmoid(dot(shared_scalar_gate,x))
out = routed + shared
```

Use one fixed accumulation order in reference; choose routed-probability order
or expert-ID order and document it. Official HF loop accumulates experts by
ascending expert ID, while reordered hardware execution may sum differently;
compare tolerances/top-logit margin rather than assert bit equality blindly.
Original model_config has norm_topk_prob=true; the GGUF graph always requests
norm_w=true for this architecture. No scaling factor metadata in current show;
reference default scale1.

Evidence HF900-997; llama qwen4exp1148-1198 and llama-graph.cpp2090-2100,
2170-2204; Strata router_top10.cu9-21 and shared_expert.cu70ff. Ignore Strata
engine-pack comments claiming router/shared gate BF16: the borrowed GGUF has
F32 there and must retain that type.

## Tokenizer, raw prompt and output

The actual artifact declares GPT2 byte-level BPE, pretokenizer `qwen35`,
vocab248320, add_bos_token=false, BOS/padding248044, EOS248046. Use tokenizer
tokens, types and ranked merge list from the borrowed GGUF metadata. Raw
generate must avoid applying a chat template; no automatic synthetic BOS when
the artifact says false. Explicit parse_special behavior needs parity with the
raw Ollama request used for baseline; expose/control that in the experimental
runner instead of silently choosing incompatible chat/raw parsing.

qwen35 pretokenizer pattern (requires lookahead + Unicode categories; plain
Rust regex does not support `(?!...)`, so choose a crate/implementation capable
of this exact operation):

```
(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+
```

GPT2 byte map: printable33..126,161..172,174..255 map to those codepoints;
remaining byte values ascending map to256+ordinal. After pretokenization map
each UTF8 byte; repeatedly merge adjacent pairs of lowest rank, leftmost tie;
do not merge across pretoken pieces. CONTROL type3 literals only match when
parse_special=true; USER_DEFINED type4 matches even when parse_special=false.
Longest matching special literal first. Incremental decode combines raw bytes
across token boundaries before UTF8 replacement/stream emission; a multibyte
character may span tokens. Chat templates and thinking/tool parsing are a later
explicit contract; do not need them for public raw-completion graph acceptance.

Final mixed output through untied output.weight yields V logits; do not apply
softmax unless sampling requests it. Initial oracle uses greedy argmax with
ascending token-ID ties. A seed alone does not establish sampled parity if
RNG/sampling transforms differ. The borrowed GGUF also advertises recommended
sampling temp1/top_k20/top_p.95; baseline explicitly freezes its request
sampling settings. Stop-token handling should be taken from the artifact and
requested stops, with248046 as actual tokenizer EOS; do not stop on PLE248044
merely because its metadata name contains eos.

Evidence llama-vocab.cpp392-399; Strata tools/strata_tokenizer.py22-141,
152-260; actual model metadata. Template text not read or printed in this audit.

## Small independent fixtures to implement before full model

1. Quant blocks: constructed zero/unit scales, exhaustive5-bit values through
   Q5_0 split-half/high-mask encoding, signed Q8_0 including -128, positive and
   negative f16 scale; compare exact f32 decoded values. For Q4_K/Q6_K verify
   subscale packing and scale/min signs separately before matvec. Reference
   upstream functions ggml-quants.c1530-1555,1940-1972,500-524,553ff.
2. Tensor orientation: nonsquare `[I=3,O=2]` synthetic matrix, distinct rows,
   plus two distinct experts `[3,2,2]`. A transposition must change output.
3. GR: two streams/two features, all down/up/inject zeros: mix equals0.5 times
   mean of independently normed streams; injection is1, so output adds block
   to both original streams. Nonzero down values distinguish silu(z/HC) from
   silu(z)/HC. Distinct per-stream gamma exposes whole-HCD RMS error.
4. GDN recurrence primitive: S=[[1,2],[3,4]], k=[1,0], q=[1/sqrt2,0],
   v=[1,2], decay0.25,beta0.5. DecayedS=[[.25,.5],[.75,1]],
   delta=[.375,.75], newS=[[.625,1.25],[.75,1]],
   out=[.625/sqrt2,1.25/sqrt2]. This exposes read-before/after-update and
   decay-before-correction errors. Separate hk2,hv4 fixture with distinct keys
   distinguishes modulo from interleaved head mapping. Never copy HF-head
   mapping into the GGUF fixture.
5. GDN conv: channels2, asymmetric taps[1,2,3,4], three raw past values1,2,3,
   current4 gives raw30, before SiLU. Store raw current4, not silu(30).
6. PLE exact hash fixture using actual constants: token42, previous[3,7]
   oldest-first gives ctx[42,7,3], mixed2=854821422343689,
   mixed3=875814630987004; all16 rows:
   [14120497,39300209,51890715,73518158,80628040,109664886,133220230,
   151294224,163952504,198468448,205404373,233797536,251057218,
   268317062,280737099,302837232]. Missing[-1,-1] and previous[3,248044]
   must agree: ctx[42,248044,248044], mixed2=4988649121201320102,
   mixed3=6810370438643746130. Current248044 with previous[3,7] does NOT
   agree: ctx[248044,7,3]. previous[0,7] proves token0 is valid, not missing.
7. PLE conv scalar normalized rows1..9 + current10, taps[1,2,3,4] => raw70
   (reads history indices0,3,6,current). Check zero dot gives gate0.5. Check
   convolution takes normalized rather than raw gated values with nonunitRMS.
8. QSA toy ratio2,budget2,one indexhead,keys dimension2, query=[1,0],
   raw keys token0=[1,0],1=[1,0],2=[-1,0],3=[-1,0],4=[100,0],
   ignore RoPE only in this toy selector: completeblock0 beats block1; selected
   IDs[0,1,4]. Token4 tail must not be pooled/normed for blockscore and must
   always be selected. Query at2 uses complete[0,1]+tail[2], never future3.
   Add four-index-head fixture whose dots[2,-3,1,-4] give relu sum3, not0.
9. Real QSA count boundaries2051..2056 above: selected IDs partition into512
   complete blocks + exact tail; no partial complete blocks, future/padding IDs,
   duplicate IDs or incomplete tail drops. Decoding/chunking must choose same
   IDs under deterministic ties and identical score precision.
10. RoPE p0 identity; p1 analytic half pairs; norm-before-RoPE; K sharedGQA
    h/12; two very different KV heads ensure modulo pairing fails.
11. Router zero logits over512: deterministic IDs0..9, each weight.1, no loss
    of mass; nonzero logits separate global softmax/topk/renorm. Sharedscalar
    gate0 contributes0.5*shared. Full sum must include all10routed+shared.
12. Whole tiny architecture fixture: independent equations (not same routine)
    with reducedD/heads and all GR/PLE/GDN/QSA/MoE stages; compare denseprefill,
    chunks[1,3,2], singledecode and state. Exact integerIDs, explicit f32
    tolerance intermediates/logits. Synthetic fixture data may be persisted;
    real private request state may not.
13. Tokenizer public Unicode corpus with combining marks, whitespace runs,
    single-digit pattern, apostrophes, emoji, control/userdefined literals;
    assert exact IDs against independent primary reference, not just roundtrip.

## Remaining uncertainties / boundaries before acceptance

- Exact original converter commit is not in inspected metadata. Above layout
  matches primary supported GGUF conventions, but full/model-layer acceptance
  against current Ollama is required to establish numerical equivalence.
- No `mtp`, `nextn`, or `draft` tensors appear; max block index47, no49th MTP
  block. Latest primary converter now supports Qwen MTP export, but this blob
  does not. Do not promise MTP speedup before a separately compatible artifact.
- Metadata lacks explicit GDN output activation key; sigmoid is confirmed by
  official model config + matching contemporary qwen4exp graph. Reject unknown
  architecture variants instead of assuming every future artifact shares it.
- Sampling RNG equality is unverified. Greedy correctness first; benchmark
  public prompts with fixed seed/settings, output checks and exact counts.
- Different Ollama CPU/GPU paths quantize activations/reduce sums differently.
  A Rust all-f32 dequant reference can be mathematically correct without exact
  sampled text equality. Agree top-logit margins/error bounds and independent
  tensor checkpoints before performance optimization changes precision.
- This review has not read tokenizer vocabulary/merges or chat template from
  full headers, only described their public declared contract. Root's bounded
  artifact parser should validate exact vocabulary/types/merges and effective
  request parameters without logging private custom templates/prompts.
- Benchmarking raw generate bypasses tools/thinking/vision wrappers but leaves
  the model's generated bytes untouched. Gateway native capabilities must only
  expose tested features; unsupported options must fail clearly, not silently
  switch to Ollama or change model.
- Single-sequence Rust graph first; batching requires independent state/visible
  causal lists per sequence. PLE predecessors must not cross conversations.
- Admission and cleanup are mandatory: bounded payload/file row/expert reads,
  no whole120GB allocation, no-swap/no-core transient buffers, finite checks,
  synchronize all async disk/CPU/GPU work before zeroing/releasing leases. A
  test/oracle capture is synthetic-only; do not export real token/router/PLE
  histories, state tensors or raw logits into diagnostic files.
