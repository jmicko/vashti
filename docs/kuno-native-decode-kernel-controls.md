# Protected native decode kernel controls

This is an internal fixed-public laboratory contract. It changes no Kuno HTTP
API, production deadline, model arithmetic, owning page plan, or default policy.
Source controls and operator acceptance do not establish whole-model numerical,
lifecycle, or performance acceptance; those remain separate physical gates.

`KUNO_LAB_NATIVE_DECODE_KERNELS` accepts exactly these labels:

| Label | Grouped MoE | GDN recurrence |
| --- | --- | --- |
| `chunk` (absent/default) | Existing member tile | Existing chunk |
| `single-moe` | Single-member tile | Existing chunk |
| `single-gdn` | Existing member tile | Single-streaming recurrence |
| `single-all` | Single-member tile | Single-streaming recurrence |
| `single-q4-all` | Q4 block single-member tile | Single-streaming recurrence |

Any non-`chunk` label requires protected native decoding and projection tile 1:
`KUNO_LAB_NATIVE_DECODE=1`, `KUNO_LAB_NATIVE_DECODE_TILE=1`, native MAX8, owned
transient I/O, and complete owning zero audits on a compiled resident fixture.
Malformed or unprotected selections reject before device/model access. Prefill
still uses the existing chunk kernels and projection tile 8. A model applies the
selected profile only after its completed prefill and projection policy boundary.
Non-default profiles accept count 1 only, before graph scratch mutation. A fresh
owner is required to change from one non-default profile to another.

Model receipts require both string fields, independently equal to the caller's
selection: `native_decode_kernels` and `observed_native_decode_kernels`. The latter
comes from the actual workspace policies and is checked again after successful
generation. Successful generation-only native/scalar step counts, exact fixture
IDs, all four full owning audits, original source audit formulas and tile-1
recovery proofs remain mandatory.

Every non-default profile additionally requires all three exact completion lines
for the selected `PROFILE`:

```text
public_native_kernel_fresh_continuation profile=PROFILE matches=true
public_native_kernel_mid_token_cancel profile=PROFILE layer=0 erased=true
public_native_kernel_mid_token_cancel profile=PROFILE layer=3 erased=true
```

The fresh owner completes the original full prefill, applies the same profile,
performs a native continuation and matches the next fixed public output before
normal close/audit. Each partial owner completes one default native token,
applies the same profile, and proves that its requested layer cancellation fired
during a native continuation before erasure/audit. Prefixes, suffixes, missing
lines and another profile's markers do not satisfy acceptance.

`run-native-max8-lab.py --native-decode-kernels PROFILE` pins the child environment;
non-default selections also require `--native-decode --native-decode-tile 1`.
Live validators never accept missing profile fields. Explicit offline
`allow_legacy_decode=True` permits only both profile fields absent with expected
`chunk`; partial declarations and non-default profiles cannot use that fallback.
The lifecycle and existing minimal/default comparison runners explicitly pin
`chunk`, so an inherited environment cannot opt them into experimental kernels.

`run-native-performance-lab.py --native-decode-kernel-comparison` creates the
exact scope `fixed-public-native-decode-kernel-abba-only`: four 52-GiB, swap/core
disabled children with native decoding enabled, tile 1 throughout, and profiles
`chunk`, `single-all`, `single-all`, `chunk`. Plan, child environment, indexed
completion line and receipt must independently agree. Other comparison scopes
retain their existing schedules and pin `chunk`. The original 75/128 fixed
public fixture, seed 424242 and temperature 0 are unchanged. Numeric lifecycle
acceptance and matched timing acceptance must be reported separately.

`--native-decode-q4-comparison` instead creates the exact scope
`fixed-public-native-decode-q4-abba-only`, using profiles `single-all`,
`single-q4-all`, `single-q4-all`, `single-all`. All four children use the same
binary, 52-GiB cap, native decoding and tile 1. GDN remains single-streaming in
both profiles; the comparison changes only the public grouped gate/up policy
from generic single-member to the Q4 block candidate. Original Q4_K gate/up
metadata is required by the kernel owner. The same independent profile
declarations, exact three profile recovery markers, fixture IDs and four complete
owning audits apply. The older kernel comparison and other scopes retain their
original schedules. Source registration does not establish a timing improvement
or whole-model Q4 acceptance.
