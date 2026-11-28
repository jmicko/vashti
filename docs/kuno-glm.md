# Kuno GLM: GLM 5.3 Flash on Kuno's Own Engine

Kuno serves Z.ai's GLM 5.3 Flash (320B total, ~18B active per token; unsloth's
`UD-IQ3_XXS` GGUF, 120 GB) with its own Rust engine on the B70 (Vulkan) and the CPU. No
Ollama, no llama.cpp. Code: `crates/kuno-glm` (engine), `apps/kuno/src/glm.rs` (worker,
config), `apps/kuno/src/native_chat.rs` (`render_glm`, GLM tool calls).

## Install

```sh
sudo kuno configure-glm --models /media/jmicko/faststore/kuno-models/glm-5.3-flash/UD-IQ3_XXS [--gpu 0000:2d:00.0] [--context 32768] [--yes]
```

The directory holds the four `GLM-5.3-Flash-UD-IQ3_XXS-0000N-of-00004.gguf` shards
(`unsloth/GLM-5.3-Flash-GGUF` at `a38483c8`; `PINS.txt` lists their sha256). The service
sees it read-only at `/var/lib/kuno/glm-models` (a bind mount in its unit). The section
needs the managed pool. `kuno configure-glm --disable` removes it.

## API

The model is a native text model like the configured `native_text` one: the name
`kuno-glm-5.3-flash:iq3_xxs` appears in `/api/tags` and `/v1/models`
(`"kuno_backend": "rust-vulkan"`, family `glm5next`), `/api/show` describes it, and
`/api/ps` lists it while it is loaded. `/api/chat` and `/api/generate` take the same fields
and options as the other native model (`native_text::prepare`):

- `think`: `false`/absent (no reasoning), `true` or `"high"` (reasoning effort Max),
  `"medium"` (High), `"low"` (Low). Reasoning comes back as `message.thinking` (or
  `thinking` for generate), apart from the content, streamed as it is produced.
- `tools` and `tool` messages: GLM's own call format, parsed into Ollama's `tool_calls`
  (arguments typed by the tool's schema). Earlier assistant turns may carry `thinking`.
- Options: `temperature` (default 1.0), `top_p` (0.95), `top_k` (40, at most 64), `min_p`,
  `seed`, `num_predict` (1024), `num_ctx` (up to the configured context, default 32768).
  `temperature: 0` is greedy.
- `keep_alive: 0` unloads it. Images, `format` and `suffix` are refused with
  `native_request_feature_unsupported` (or `native_images_unsupported`).

## Behavior

- **Residency.** The worker needs most of the B70's memory and about 90 GB of RAM (the
  experts the B70 does not hold). Before it runs, the gateway stops image/video work and
  the managed Ollama workers, and the other native model; any other model's request stops
  it. It also stops after five idle minutes, so the RAM returns.
- **Load.** The engine is ready in about 10 s; the first request then reads every block's
  experts (about 50 s from a warm page cache, longer cold), shown as `load_duration`.
- **Speed.** About 13-15 tokens/s generating (two tokens drafted per step with the model's
  NextN block and checked together). A prompt of up to ~160 new tokens is read in small
  batches (~30 ms/token after ~1.3 s); longer prompts in batches of up to 2048 tokens
  (about 100-140 tokens/s, at least ~6-8 s per batch).
- **Conversations.** The worker keeps the sequence's state and a saved point at the end of
  the last prompt (in device memory only). A request whose rendered prompt starts with the
  last one (a chat's next turn, with the earlier turns unchanged) reads only the new part.
- **Seeds.** A seed fixes the sampler's random draws, but the experts move between the B70
  and the CPU as their use changes, and the two compute in slightly different precision,
  so the same seed can give slightly different text.
- **Privacy.** Content never reaches argv, logs or disk; the worker runs with core dumps
  off, no swap, and no writable filesystem (as the other workers). Expert-use counts that
  steer the placement are kept in memory only.

Errors are Ollama's `{"error": code}`: `native_context_exceeded` (prompt plus `num_predict`
over the context, 400), `native_load_failed`, `native_gpu_memory_busy`,
`native_worker_failed` (503, retryable; the next request starts a fresh worker).
