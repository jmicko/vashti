# Standalone Intel Strata comparison, 2026-10-09

## Current state: comparison installation removed

The user authorized teardown on 2026-10-09 after trying Strata. Its service was
stopped before removing the temporary checkout, build, virtual environment,
prepared dense arena and the specifically owned `localhost/strata-sycl-dev`
container image. Other containers and the original DASLab model shards were
preserved. Installation paths below describe the historical comparison.

Retained content-free receipts, installer/control source copies and public build
log are at
`/media/jmicko/faststore/kuno-development/strata-comparison-evidence-20261009`.
The verified draft sources, `mtp-q2_0.gguf`, runtime files and vocabulary subset
were moved without duplication to
`/media/jmicko/faststore/kuno-development/kuno-strata-mtp-q2-20261009` for the
native Rust implementation. The separate pinned research checkout remains.

Protected host check at 17:43 UTC verified B70 usage 50,688,000 bytes, no NVIDIA
compute process, Kuno admission unpaused with no recovery requirement, Kuno,
Ollama and Vashti HTTP 200, and zero classified Xe/NVIDIA faults. Strata is no
longer available on port 8085. No driver reset or production binary update was
performed.

The new native goal targets at least 30 accepted tokens/s. First work is GPU
expert routing and the alternate bound-target Q2_0 draft graph. The upstream
27.3/36.0 measurements remain separate from native performance and do not
establish a matched native comparison.

The user authorized installing upstream Strata to compare its experimental Intel
backend and asked to leave it running afterward. The native Rust optimization
goal stays paused. This installation is outside Kuno and changes no Kuno API,
Vashti connector, production model catalogue, driver, release or firmware.

## Source and model reuse

- MIT upstream: `Niko1221/Strata`, tag `v0.1.41`, source
  `fb58e0dbc8399662c0e47c76578c6e878b14f6cf`.
- Installation: `/media/jmicko/faststore/kuno-development/strata-intel-20261009`.
- Official agent guide: <https://github.com/Niko1221/Strata/blob/v0.1.41/docs/AI_SETUP.md>.
- Intel guide: <https://github.com/Niko1221/Strata/blob/v0.1.41/docs/INTEL.md>.
- Two DASLab IQ3_S shards are hard-linked from the existing pinned Ollama import;
  no model merge or second 83,617,662,656-byte shard copy. `iq_pack.py` stores the
  dense arena (1,538,035,200 bytes), tokenizer and index. Do not request the CUDA
  low-RAM `experts.bin` copy.
- Container `localhost/strata-sycl-dev`, built from upstream `sycl/tools`, image
  `63379d06dde9756ff1d2c97a34691f8829118a0122d1d4513aa237aede605d51`.
  Engine build target: `AOT=bmg-g31`, bounded eight compilation jobs.
- Local installer-only correction: `sycl_version()` delegates to
  `setup.source_version()`. The upstream numeric regex scans a variable-based
  SYCL CMake version and falsely reports zero. This describes our freshly built
  pinned source; it is not a stale-binary version workaround or weakened gate.
- Rootless build uses `--userns=keep-id -u 1000:1000`, preserving host ownership.
  First use copies/remaps image layers on the SATA container store and can take
  several minutes. Existing containers and their store must remain untouched.

## MTP format to support in a future Rust importer

Strata's draft model is a **different quantization and runtime format** from
Kuno's existing self-contained Unsloth Q8_0 MTP GGUF. Keep both options; neither
is a drop-in replacement for the other. Record this difference in comparisons.

1. `tools/mtp_fetch.py` fetches 31 original BF16 draft tensors, about 5.214 GB,
   at pinned Qwen revision `de4b8e4d43b917e7706784d8bb445c9af86a3540`.
   Its verified byte-range manifest and source hashes remain with the installation.
2. `tools/mtp_pack.py --experts q2_0` quantizes routed expert matrices to Q2_0
   and writes `mtp-q2_0.gguf`. Its code grid is `{-1, 0, 1, 2} * d`, with 17
   candidate scales minimizing block squared error. This is round-to-nearest
   quantization, **not GSQ**. Q2_0 GGUF type is 42. Quantization is lossy. Dense tensors in that
   intermediate GGUF retain BF16/F32 formats.
3. `tools/mtp_rt.py` losslessly rearranges the already-quantized Q2_0 expert
   blocks into `experts.bin`: 512 experts, 1,382,400 bytes each, 707,788,800
   bytes total. Hidden width 2560, expert width 640. Gate/up rows interleave
   (gate row r, up row r); down follows. Quantized codes and FP16 scales use
   separate planes. Q2_0 block size is 64 values / 18 bytes, distinct from Q2_K.
4. Runtime `dense.bin` is 116,099,072 bytes with a 29-entry `dense.txt` index:
   `name kind rows cols offset bytes`, tensors aligned to 256 bytes. Large
   projections become Q8_0 using GGML reference rounding. Router/hyperconnection
   tensors retain BF16; norm weights become F32 with Gemma RMSNorm `1 + w`.
   Do not apply that adjustment twice.
5. `draft_vocab.bin` supplies the selected output-head vocabulary subset, default
   CJK in this installation. The runtime borrows the compatible target embedding/
   head and per-layer lookup machinery; these three files alone are not a generic
   standalone draft model. Keep target/tokenizer identity and original tensor
   naming pinned when introducing a Rust importer. Source graph details also
   include global-width 10240 hidden RMS and draft cell `(R_c, T_{c+1})` at
   RoPE position `c`, predicting `T_{c+2}`; preserve that alignment. A missing
   vocabulary file uses the full-head fallback rather than rejecting startup.

Potential Rust work: reuse existing native raw-type42 Q2_0 arithmetic and add
this MTP runtime importer/profile with independent expert-relayout fixtures,
or read the intermediate GGUF directly without
requiring Strata's runtime files. Validate full draft vectors, target verification,
accepted-prefix repair and acceptance rate before selecting a default. A smaller
draft is a performance candidate, not proof that quantization preserves acceptance
or that a Strata measurement transfers to Kuno.

Primary references: upstream [packer](https://github.com/Niko1221/Strata/blob/fb58e0dbc8399662c0e47c76578c6e878b14f6cf/tools/mtp_pack.py),
[runtime conversion](https://github.com/Niko1221/Strata/blob/fb58e0dbc8399662c0e47c76578c6e878b14f6cf/tools/mtp_rt.py)
and [fetch manifest](https://github.com/Niko1221/Strata/blob/fb58e0dbc8399662c0e47c76578c6e878b14f6cf/tools/mtp_fetch.py).

## Verified runtime containment

Installed loopback web/OpenAI endpoint: `http://127.0.0.1:8085`.
Only the B70 is selected. Stock Strata does not move this draft to the RTX 3060.
Live process checks confirm a swap-free service cgroup, core dumps disabled, persistent SYCL cache off,
API monitor off, no server prompt/output log, Podman log driver `none`, and
original device-plan/no-host/alias safety checks. PLE initially uses direct reads;
published 30.5–40.8 tokens/s results used PLE RAM, which adds about 26.82 GiB.
A direct-read trial is not a matched reproduction of that published setup.

Fresh executable SHA256:
`821c0ed94f63fece74e1235d34a76435ac32a930dd58ff12f664e1af98d0066f`.
Ninja reports `BUILD EXIT 0`, errors0; its upstream shell wrapper nevertheless
returns1 because the final grep finds no FAILED lines. Executable existence,
AOT device, compiler version, source pin and actual inference independently pass.

Rootless containment required service `Delegate=yes` so Podman 4.9.3 sees a
user-owned cgroup. Neither `--cgroups=disabled` nor `cgroupfs` alone prevented
relocation. Initial starts were stopped before prompts; actual B70 memory returned
to 50,688,000 bytes and Kuno admission/HTTP recovered. A GPU-free cat container
proves corrected inheritance, and live server/Podman/conmon/Strata process checks
all remain in `/system.slice/kuno-strata-comparison.service` with
`memory.swap.max=0`, current 0, MemoryMax 56 GiB. No shared store/container/settings,
driver, GPU reset, frequency or global swap change.

## Public inference results

All use seed 424242, temperature 0, penalties disabled, thinking explicitly off,
short new chat prompts and default spec 4/min-p 0.5. Generation answers are never
stored; receipts hold only these public input fixtures and numeric/boolean results.

| Fixture | Generated tokens | Decode seconds | Decode tokens/s | Complete request seconds |
| --- | ---: | ---: | ---: | ---: |
| Integer sanity: 7 times 8, return only the integer | 3 | 0.2340 | 12.8 | 1.0264 |
| Merge two sorted integer lists; explain complexity | 200 | 7.3365 | 27.3 | 8.0451 |
| Explain hash tables, collisions, resizing, lookup complexity | 200 | 5.5495 | 36.0 | 6.1898 |

Integer answer matches 56 exactly; other outputs are nonempty and stop at the
requested 200-token cap. Neither200-token request reused a prompt cache. Draft
offered/accepted counts are 172/142 for code and 191/118 for prose. These are
ordinary upstream Strata API generations, not a matched native Kuno CPU-oracle
comparison, broad answer-quality acceptance or an Ollama benchmark. Workload,
MTP quantization and verification overhead differ from the native 13-token
controller. One low-token sanity result is not a sustained throughput estimate.

Detailed public evidence, config/hash provenance, initial failed containment
observation, successful containment and host snapshots:
`/media/jmicko/faststore/kuno-development/strata-intel-20261009/receipts`.
A corrected startup was observed ready by 56.7 seconds; this includes delayed
inspection and is an upper bound on readiness, not a precise load benchmark.

Protected shutdown/restart passes: B70 returns to 50,688,000 bytes, Kuno admission
resumes, Kuno/Ollama/Vashti HTTP return 200 and all classified GPU fault counts remain 0.
The final restart is observed ready at 34.0 seconds after GPU unload (file cache
remains warm). Sampled streaming at temperature 0.7/seed 424242 returns 45 tokens,
46 SSE chunks and the final DONE event, with no error: 16.4 decode tokens/s,
3.4825 seconds overall. Its 21-token prompt reuses no cache; 52 drafts offered,
25 accepted. Sampling/workload significantly changes speed, so 27.3–36.0 greedy
results should not be presented as a uniform default-chat throughput promise.
Final web page and health return 200; engine loaded, context 32768, logging off,
actual server/Podman/conmon/engine all in the protected cgroup, swap current 0.
B70 occupied 32,071,667,712 bytes at final snapshot, no NVIDIA compute process;
all classified GPU fault counts remain 0. Browser UI itself was not manually
interaction-tested; its API streaming path was tested.

## Operating the comparison

Strata is left running at `http://127.0.0.1:8085` for the user, as requested.
The OpenAI-compatible API is `http://127.0.0.1:8085/v1`, model ID
`qwen3.8-flash-next-iq3_s`. This is loopback only, not a LAN Kuno endpoint.
No inference key is configured. Strata has no automatic idle unload here.

Kuno inference admission is intentionally paused while Strata owns the B70;
Kuno/Ollama/Vashti listeners retain their addresses. Use these protected controls
rather than the upstream generated start script:

```sh
/media/jmicko/faststore/kuno-development/strata-intel-20261009/stop-strata.sh
/media/jmicko/faststore/kuno-development/strata-intel-20261009/start-strata.sh
```

The controls use ordinary sudo. Stop verifies memory release and restores Kuno
admission automatically; its privileged stop hook also runs if the service exits.
If termination or actual memory recovery fails, the marker/pause is retained
with an actionable control error. No driver reset is attempted. The comparison
service is transient and is not enabled for boot. The install and model files
remain, so start can be used again after a reboot.

Server log is `null`, API monitor is `false`, Podman log driver is `none`; core dumps and
swap are disabled for both API server and engine. Browser chat history is saved
in the browser's localStorage by upstream Strata; our checks used API-only calls
and retained no generated text. Rerunning upstream setup can overwrite the privacy
configuration and generated startup script, so reapply/inspect it before use.
The external install is for comparison, not certification of all upstream privacy
or correctness behavior. Required Vashti changes: none. Native Rust goal remains
paused; no native backend/default/API or Vashti consumer changed.
