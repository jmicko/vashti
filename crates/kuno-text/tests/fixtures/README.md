# Independent Qwen35 tokenizer fixture

`qwen35-python.json` contains only constructed public synthetic inputs and a
294-token synthetic vocabulary. It contains no downloaded vocabulary, private
template, user input or model weights. Expected token IDs were produced by
`tools/strata_tokenizer.py` from Strata revision
`6f32ec070f23ced9f50e704d854d775da52591ab`, using CPython 3.12 and
`regex==2025.11.3`. The reference's source/license is recorded in `../../NOTICE`.

The 256 initial token IDs deliberately equal raw byte values. Additional
normal tokens/merges include `abc`, `abab`, space-prefixed words, combining
marks, CJK, emoji, CRLF, contractions and a `12` token that must **not** form
across the pretokenizer's single-digit boundary. CONTROL, USER_DEFINED,
overlapping USER_DEFINED and UNUSED literals test their distinct semantics.
Every input has exact expected IDs with both values of `parse_special`.

This independent expected-ID fixture catches incorrect boundary/rank rules that
roundtrip tests cannot detect. The Python reference has no role in runtime
inference. Its tokenizer regex behavior on codepoints introduced after the
native Rust regex's Unicode tables is not established by these representative
cases; current installed-model benchmark acceptance covers the actual 75/80
token raw prompts separately.
