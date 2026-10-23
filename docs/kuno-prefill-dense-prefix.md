# Real wide-M128 prefix with original exact remainder

Backend-only opt-in source2026-10-23,complete padded/direct/tiny/GR/frontier
bundle retained. Original general-wide selector requires count>=512 and an
exact multiple128. Our4244 processed-token benchmark ends with4096+148, so its
148 real members previously all use the smaller fallback. New source executes
128 real members in the wide kernel and20 in the original exact fallback.
No fabricated512/extra tokens,token omission,caching or changed context.

The new explicit prefix selector permits real count>=128,multiple128,with
original physical shared-memory/thread/subgroup/CMMA/vector/owner/grid guards.
Original general selector keeps minimum512. Full parent and suffix intervals
are checked BEFORE any prefix dispatch. Tail uses same original input root,
stream,end-offset with genuine additive prefix*columns*2 byte offset. Output
remains same full owner,first+prefix*stride,including holes. Suffix<128 uses
original whole-count dispatcher directly; no recursive split. Same chronological
K16 products,finite/fault behavior and full-owner erasure. No payload allocation,
extra host/device fence,Scope,private storage or API change.

Retain original gr_wide_tail_products semantics97; new prefix dispatches have
separate counters. Paired dense-product/finite counters continue to cover every
actual dense producer. All417 old GPU controls kept plus16 paired original
whole-fallback versus real prefix/remainder and original F64/full-owner cases:
BF16GR320/F16128,counts128/148/257/511/513/641,nonzero offsets/stride holes,
NaN-weight and NaN-input in the LAST real tail token. Total433 GPU. All267CPU
+three partition/real borrowed root/physical geometry controls;15runtime and
all previous shader layouts retained (no new shader arithmetic). BOTH complete
unchanged original18-script chains,native compiler and physical recovery
required before speed selection. No compilation during GPU timing.

Source only; no build/GPU/performance/install claim. No API/model/context/
config/frontend/release/reset/reboot change; Vashti:none. Strata5.3515s still
unmet; complete predecessor821bfd6a normal8.4267s is neutral.
