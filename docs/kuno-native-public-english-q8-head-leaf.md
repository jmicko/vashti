# Fixed public English Q8 draft-head leaf

This internal candidate computes 40,525 selected original Q8 head rows rather
than all 248,320 draft rows. Default full-vocabulary Q8 and Strata Q2 profiles
remain unchanged. Actual target verification, full target greedy and real repair
remain mandatory: the subset changes draft proposal coverage only.

The reviewed public artifact is Strata's `data/draft_vocab_en.bin` at MIT revision
`fb58e0dbc8399662c0e47c76578c6e878b14f6cf`, copied with its MIT license. It is
162,100 bytes of strictly sorted, unique little-endian 32-bit token IDs, all
within 0..248,320. Its SHA256 is
`369151522226a5edaa5f12cfd1e2ae7db8f4fbdbd222f3dcf327dced9597fb25`.
`EnglishVocabulary` accepts exact compiled artifact bytes after checking every
ID and ordering; a structurally valid different vocabulary is rejected. No
request-selected vocabulary or private ID map is admitted.

The parent retains every original packed head page and all 42 private owners.
The map adds one 196,608-byte public owner, with 162,100 logical bytes and exact
zero padding. It joins the same pinned client's complete public physical
ownership, no-alias, stream, source admission and close accounting. No extra
private owner is needed. Parent integration must prove its upload/source and
full padding before launch, clear the original full logits/candidate owners,
reject broadcast, and reject full `Output` APIs before any state mutation.
Shared Q8 scales may compose with the indexed entry.

The 16 original 16,384-row head segments select respectively
16,384/16,384/2,777/1,958/1,503/996/0/0/0/31/44/33/26/32/34/323 rows.
Their exact public intervals cover every compact row once. Empty segments are
skipped, retaining their original owners. A selected row's original token ID
is bounds/order checked before subtraction or any packed read. The source
page and complete packed extent, original 2,560 input values, count1 and the
full 248,320-logit binding must be valid. All four physical rows participate
in the original collective even at tails. Shared-scale initialization covers
all 320 cells before its uniform barrier, including inactive rows. Both paths
retain original signed-byte/half decoding, multiply order, low/high column
accumulators and original 64-virtual-lane F32 reduction. They use 128 physical
threads, original 512-byte reduction scratch, and at most 1,792 bytes of
shared memory with cooperative scales.

Compact greedy scans only 40,525 values in ten 4,096-row groups. Both chunk and
finish compare full mapped token IDs on ties and return the smallest original
ID. Finish reads only ten populated triples from the unchanged 183-word
candidate owner. Calling full-vocabulary greedy or the old 61-group finish
would admit cleared omitted rows or unset triples and is forbidden. The
last indexed page covers finite checks of the entire omitted/logical/physical
logit tail; the ordinary full-owner sweep remains required.

Selected original packed rows total 110,228,000 bytes per draft head, versus
675,430,400 bytes for the full head: 83.68% fewer row bytes and 6.13 times fewer
row MACs. These are source counts, not measured speed or proposal acceptance.
The full-model original target gate and compiler-idle normal-language timing
remain required. Source-only CPU/SIMT and Checked/Unchecked shader checks do
not establish physical GPU correctness, recovery or performance.

All four focused tests pass: exact artifact identity and all 16 intervals;
independent binary16/signed-Q8 equations against the original decoder and
64-lane tree for selected page rows; compact negative/tied greedy; and actual
Checked/Unchecked WGSL plus SPIR-V with workgroup initialization None. Shader
checks cover mapped-source/count guards on every packed read, original
collectives, shared extents and a deliberately hoisted packed-read rejection.
No C oracle is claimed. Strict main-and-test Clippy and formatting/diff checks
pass. The last CPU check completed before 2026-10-11 15:05:14 UTC under the
parent's cleared source-test window, with two build jobs and no GPU operations.
