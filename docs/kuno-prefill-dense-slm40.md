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
