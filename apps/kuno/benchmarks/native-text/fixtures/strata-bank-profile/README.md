# Fixed public Strata expert ranking

Pinned Strata commit `fb58e0dbc8399662c0e47c76578c6e878b14f6cf`.
Original `data/expert-profile.bin`: SHA256
`8f59b4aa8873209dff11c11e37bcda9529a1335b724a1afeea37bf6388975baf`.
`top144-u16le.bin`: SHA256 `7bb7a111617d13f3daeed4a3b406e0f7a34b2a7fb07a1ee3845161a08ab57a5d`.

The compact table filters the original pair list by layer, keeps each layer's
first144 IDs in original rank order, then concatenates layers0..47 as little
endian u16. No native routes or workload traces enter the derivation. Only the
13,824-byte compact table is compiled into the laboratory runtime. The full
196,632-byte original is an independent test fixture. Complete MIT notice in
[LICENSE](LICENSE). See the normative bank filler document for scope/caps.
