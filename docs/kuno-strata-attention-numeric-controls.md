# Active-prefix QSA device controls

Prepared against the active-prefix scoring/selection changeaa7ca05. The separate
`attention_checks` field of the existing compiled-public matrix check runs the
actual original `Op::Validate` and `Op::Select` primitives on the same private
arena. It does not fabricate a prompt scope, model state or capture identity.
Original matrix and other operator control inventories/counts are unchanged.

Six new cases retain8192 context blocks, full source row strides8192 and output
stride2051. Positive `(start,count)` geometries are `(0,3)`, `(2050,3)`,
`(4094,3)`, `(32764,4)`. Every visible block has a fixed binary-exact score,
including repeated ties and negative zero. Every future source score isNaN and
status is`u32::MAX`, including complete undispatched source rows. No causal
future/unused row may affect the result or finite/status fault packet.

An independent CPU sort ranks blocks by descending score/tie-ascending block,
charges full or partial visible cells, clips only the threshold block and sorts
the chosen cells chronologically. Device output must equal it exactly. Every
unused selected-row tail must contain the original context sentinel32768;
undispatched rows and the complete real owning padding must remain zero.

One negative gives a last-query active extent below its actual visible reach;
another poisons a visible score withNaN. Both must produce the real sticky
fault bit1 while complete fault padding and undispatched selected-owner tail
remain zero. No live-row oracle is claimed after a negative fault.

Every result/error first attempts actual `arena.reset()`. The existing worker
still requires its real final `arena.close()` and owned-zero audit before
publishing the combined receipt. A fault/erasure failure is never a pass.

`python3 apps/kuno/benchmarks/text/test-qsa-active-prefix-controls.py` compiles
the exact production oracle/checker with`rustc --test -D warnings`. Two tests
compare it with independent cell-level ranking and reject changed live/tail/
padding outputs. Structural checks confirm actual primitive calls, poison
inventories, full-owning read and reset-before-report. Rustfmt/diff checks pass.
No shared Cargo build, GPU, sudo, service or installation was performed for
these controls. They still require root's full compile and genuine device run;
CPU preparation does not prove selection, numerics, erasure or performance.
