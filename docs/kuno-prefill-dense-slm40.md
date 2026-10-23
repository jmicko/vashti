# Coupled dense pitch40 operand staging and cooperative readers

Optional source2026-10-23,requires complete direct-publication/tiny/GR/frontier
bundle. Hypothesis: original32-half row pitch may conflict in physical SLM banks.
No assertion of an observed hardware bank conflict or oneMKL's exact layout.
Keep original compact producer and finite consumer. Add actual40-half row
staging PLUS matching A/B cooperative loads together. Each original input and
weight element loaded once into same logical tile; eight padding halves per row
are explicitly zeroed by one real stage job. Both ping-pong buffers occupy
40960 real bytes; original49152B hardware requirement retained. No new global
payload allocation, no changed arithmetic,chronological K16 products,prefetch,
barriers,output publication,finite policy,owners,erasure or private persistence.

Compile all original32 composite layouts plus24 padded producer layouts.
Keep all264 CPU +three real stage/fragment/chronology/compiled-layout controls,
15 runtime/429WGSL/320 original matrix+16raw. Retain original12 compact-direct
GPU controls through a real compact selector; add12 paired pitch40/original
shared/F64 controls with full source/output/fault owners. Total417GPU, BOTH
unchanged original18-script chains,actual compiler no-spills and physical recovery
required before speed selection. One clean only if justified after full validation;
no favorable rerun,precision relaxation,install or release from diagnostics.

Source only; no build,GPU or speed claim. Previous complete27451ef9/worker61a8d8f6
normal8.3706s is neutral; Strata clean5.3515s remains unmet. No API/model/context/
config/frontend changes; required Vashti:none.

## Complete821bfd6a validation

Worker01188d21bb4e4838e097c7fe278d22b229f188b1f91f3c7ed1c4ac71933ad68c.
267CPU/15runtime/429WGSL/320matrix+16raw+32compact+24padded layouts, ALL417GPU
(all405 retained+12 padded),BOTH unchanged original18-script chains and physical
recovery PASS. Normal9.555901327/8.426700364s; profile9.627530733/8.520494850s.
Neutral/slightly longer than compact8.3706s; no gain,clean,staged,install or
Strata parity claimed. All three padded native programs SIMD16,359/454/362
instructions,79 sends,128GRF,0:0 spills/fills. Original compact/finite and QSA
zero-spill guards retained. Full public summary:
`/media/jmicko/faststore/kuno-development/strata-parity-dense-slm40-20261023-expert-cohorts-public-summary.json`.
No API/config/model/context/frontend changes; Vashti:none.
