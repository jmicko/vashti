# Kuno Audio: Speech Recognition and Synthesis

Kuno serves LiquidAI's LFM3-Audio-2B (Q8_0 backbone, F16 encoder/decoder/codec) with its own
Rust engine on the CPU: no GPU, no Ollama, no llama.cpp. Code: `crates/kuno-audio` (engine),
`apps/kuno/src/audio.rs` (worker and runtime), `crates/kuno-protocol/src/audio.rs` (types).

## Install

```sh
sudo kuno fetch-models lfm3-audio-2b --directory /var/lib/kuno/audio-models   # 2.4 GB, pinned, sha256-checked
sudo kuno configure-audio --models /var/lib/kuno/audio-models [--yes]
```

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

returns `audio/wav`, 24 kHz mono 16-bit. `voice` is `female` (default) or `male`;
`language` is a code from discovery (default `en`, American English); `seed` repeats a
rendering (default: random). Up to 20 000 characters; long text is spoken about 400
characters at a time, at sentence ends, and joined.

Discovery: `/kuno/v1/capabilities` has an `audio` section: `state` (`ready`, `unavailable`
with `reason`, or `disabled`), the two paths, `voices`, `languages`, `max_request_bytes`,
`max_speech_characters`, `speech_sample_rate`. It is absent from older Kuno versions.

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
  use. It stays while used and exits after five idle minutes (about 2.6 GB RSS returned to
  language models). A client that disconnects stops it; the next request starts a new one
  (about 1-3 s to load from cache).
- **Speed** (Ryzen 9 5900XT, 37 GB/s; the backbone's weights bound decoding): recognition
  about 0.6-0.85 s per second of audio; speech 0.6-0.8x real time, first audio after
  about 1 s (the API returns the whole file; streaming is possible later). A GPU path would
  beat real time; not done.
- **Fidelity** to llama.cpp (the reference): identical transcripts on the test clips, in
  both directions (Kuno's speech transcribed by llama.cpp and the reverse).
