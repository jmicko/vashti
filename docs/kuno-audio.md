# Kuno Audio: Speech Recognition and Synthesis

Kuno serves LiquidAI's LFM3-Audio-2B (Q8_0 backbone, F16 encoder/decoder/codec) with its own
Rust engine: the backbone and depth decoder on a configured GPU (Vulkan), everything else on
the CPU, or all of it on the CPU. No Ollama, no llama.cpp. Code: `crates/kuno-audio` (engine),
`apps/kuno/src/audio.rs` (worker and runtime), `crates/kuno-protocol/src/audio.rs` (types).

## Install

```sh
sudo kuno fetch-models lfm3-audio-2b --directory /var/lib/kuno/audio-models   # 2.4 GB, pinned, sha256-checked
sudo kuno configure-audio --models /var/lib/kuno/audio-models [--gpu 0000:04:00.0] [--yes]
```

`--gpu` names the GPU by PCI address (`lspci -D`); the host's default is the RTX 3060
(`0000:04:00.0`), leaving the B70 to images, video and large language models. The worker
uses that GPU only when the model (about 2.6 GB) fits in its free memory at start, and
otherwise runs on the CPU; `capabilities.audio.device` shows which.

Without an `audio` config section, Kuno looks for the collection in the image models
directory. `kuno configure-audio --disable` removes the section. The files are
`LFM3-Audio-2B-Q8_0.gguf`, `mmproj-LFM3-Audio-2B-F16.gguf` and `LICENSE` (LFM Open License
v1.0) from `LiquidAI/LFM3-Audio-2B-GGUF` at `39670cb5`.

## API

Both routes are keyless on the trusted LAN, like images, and take `Content-Type:
application/json`. They reject requests with an `Origin` header, so Vashti calls them server
to server. Responses carry `Cache-Control: no-store`. Errors are `{"error": code}`.

`POST /kuno/v1/audio/transcriptions`

```json
{"audio": "<base64 WAV>", "language": "en"}
```

returns `{"text": "..."}`. The WAV can be PCM 8/16/24/32-bit or float 32/64, any sample
rate and any channel count (channels are averaged). `language` is optional: without it
the model detects the language. Audio of any length is accepted up to the 48 MiB request
bound (about 25 minutes of 16 kHz 16-bit mono); it is transcribed a minute at a time, cut at
the quietest 100 ms, and the pieces are joined with spaces. Numbers come back as digits
("platform 4").

`POST /kuno/v1/audio/speech`

```json
{"input": "text", "voice": "female", "language": "en", "seed": 42}
```

returns `audio/wav`, 24 kHz mono 16-bit. With `"stream": true` the WAV is sent as it is
generated: its header declares an unknown length (`0xFFFFFFFF`), then samples follow every
80 ms frame; an error after the first audio ends the stream early (before it, the request
fails with its status as usual). `voice` is `female` (default) or `male`;
`language` is a code from discovery (default `en`, American English); `seed` repeats a
rendering (default: random). Up to 20 000 characters; long text is spoken about 400
characters at a time, at sentence ends, and joined.

Discovery: `/kuno/v1/capabilities` has an `audio` section: `state` (`ready`, `unavailable`
with `reason`, or `disabled`), the two paths, `voices`, `languages`, `max_request_bytes`,
`max_speech_characters`, `speech_sample_rate`, `speech_stream` (true when `stream` is
accepted), and `device` (`"cpu"` or the GPU's name the model last loaded onto; absent until
the first request). It is absent from older Kuno versions.

Error codes: `audio_request_invalid`, `audio_request_too_large` (413),
`audio_request_timeout` (body slower than 120 s), `transcription_audio_invalid` (bad
base64), `transcription_language_invalid`, `audio_format_unsupported`, `audio_too_short`
(under 0.1 s), `speech_input_invalid` (empty or too long), `speech_voice_invalid`,
`speech_language_invalid`, `text_invalid` (400); `audio_not_configured`,
`audio_model_unavailable`, `audio_queue_full` (one running, four waiting),
`audio_worker_failed`, `audio_worker_start_timeout` (503). A wrong content type gets the
shared guard's `image_content_type_required` (415).

## Behaviour

- **Worker.** `kuno audio-worker` is started on first use, enters protection before any
  other thread exists (no core dumps, not dumpable, no swap, Landlock with no writable
  path), discards stderr and writes only protocol frames. Request content is zeroed after
  use. It stays while used and exits after five idle minutes, returning its memory to
  language models. A client that disconnects stops it; the next request starts a new one
  (about 1.5 s to load from the page cache).
- **Speed** (Ryzen 9 5900XT host):

  | Device | Speech | First audio | Recognition |
  |---|---|---|---|
  | RTX 3060 | 9.7x real time | 0.09 s | 0.43 s for 8.0 s, 0.78 s for 15.4 s |
  | Arc B70 (no language model loaded) | 12.8x real time | 0.04 s | 0.30 s for 8.0 s, 0.54 s for 15.4 s |
  | CPU, 16 cores | 0.97x real time | 1.1 s | 8.4 s for 11.2 s |

  (llama.cpp on the same host: 0.64x on the 3060, 1.42x on the B70.) On a GPU a whole
  frame is one submission: three depth-decoder passes, each followed by sampling on the
  device (the host draws the codebook order and the uniforms from the seed beforehand, in
  the order its own sampler uses them; the device sampler picked what the host sampler
  picks in 400 of 400 test cases), the backbone step and the codec's transformer (the
  last two side by side). Frames are pipelined: what a frame needs from the last one stays
  on the device, so the next frame is submitted before the last is read, and the device
  does not wait for the host. A frame (80 ms of audio) takes about 7.3 ms of device time on
  the RTX 3060 and 7.2 ms on the B70, most of it reading the backbone's 1.5 GB of weights
  (about 320 and 450 GB/s). Where the device has half-precision matrix units (16 x 16 x 16
  on NVIDIA, 8 x 16 x 16 on Intel) products of 2 to 16 positions (the depth decoder, the
  codec) and of prompt chunks use them; SwiGLU's two products and each attention gate are
  fused, so a frame is 446 steps (a step costs about 2.5 us on the 3060 and 5 us on the
  B70). The host does the
  inverse STFT (an FFT) and little else: about 0.3 s of CPU time for 8 s of speech.
  Recognition runs the encoder's blocks and greedy decoding on the device too (decoding a
  token ahead, the argmax on the device); the mel spectrogram and the encoder's
  convolutional front stay on the host. The CPU path uses one thread per physical core. The first request after the worker starts also loads the model (about 5.5 s with a
  GPU, every upload read back and compared; 1.5 s without).
- **Memory:** 2.5 GB of VRAM and 1.3 GB of RAM with a GPU; 2.4 GB of RAM without.
- **Fidelity** to llama.cpp (the reference): identical transcripts on the test clips, in
  both directions (Kuno's speech transcribed by llama.cpp and the reverse).
