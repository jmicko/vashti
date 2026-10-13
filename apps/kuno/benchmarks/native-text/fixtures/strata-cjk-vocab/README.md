# Strata runtime draft vocabulary

Original public `rt/draft_vocab.bin` from the Q2 sidecar prepared on October9:
106,299 unique original Qwen token IDs. Original SHA256:
`b1e1d3a7a9e4bf862dcd5923ce661fb59bbd07907e594df5cf86a62ac235cb91`.

This fixture sorts the same set into strictly ascending little-endian u32 IDs
for the Rust indexed-head paging scheme. It adds/removes no token. SHA256:
`047861b9b153336c2b0e2d068af3819053622d0f7254e2f1d4df45167a78e97f`.
Generation/proposal logits are never persisted in this fixture. Strata is MIT;
see the source notice at `../../src/strata-port-NOTICE.txt`.

The port's internal older `EnglishVocabulary` names now refer to this set only
under `strata-rust-port`; the original engine still uses its40,525-row fixture.
Draft pruning affects proposed tokens only; the real full target evaluates and
verifies its complete vocabulary, including excluded draft tokens.
