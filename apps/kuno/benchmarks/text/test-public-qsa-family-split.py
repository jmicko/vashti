#!/usr/bin/env python3
"""CPU-only actual public QSA timer admission and stage call-site controls.

The unchanged admission method is compiled with metadata-only arena/timer
stand-ins to prove that disabled private collection never looks up a client or
starts a timer. No GPU/real fence is simulated or claimed. Full worker compile
and protected public hardware timing are the parent task's acceptance gates.
"""
from pathlib import Path
import subprocess
import tempfile

root = Path(__file__).resolve().parents[4]
native = root / 'apps/kuno/benchmarks/native-text/src'
profile = (native / 'strata_prefill_profile.rs').read_text()
source = (native / 'strata_prompt_qsa_full.rs').read_text()
marker = '// PUBLIC_QSA_TIMER_ADMISSION_'
method = profile.split(marker + 'BEGIN\n', 1)[1].split(marker + 'END', 1)[0]
assert method.index('if !ENABLED.load(Ordering::Acquire)') < method.index('arena.client()')
run = source[source.index('pub(crate) fn run('):source.index('fn join_flags(')]
labels = ['qsa_family_producers', 'qsa_family_score_select', 'qsa_family_attention_merge', 'qsa_family_gate']
marks = []
for label in labels:
    assert run.count('family_timing.mark("' + label + '")') == 1
    marks.append(run.index('family_timing.mark("' + label + '")'))
assert marks == sorted(marks)
start = run.index('Timer::for_public_arena(arena)?')
positions = run.index('Op::Positions,')
score_loop = run.index('for batch in score_batches(')
select = run.index('selection::select_active_prefix(')
attention_loop = run.index('for first in (0..count).step_by(32)')
attention = run.index('Kernel::PromptAttentionFault,')
merge = run.index('Kernel::PromptMergeFault,')
gate = run.index('Op::GateFault,')
assert run.rindex('span(&q[i], n)?;', 0, start) < start < positions < marks[0] < score_loop
assert score_loop < select < marks[1] < attention_loop < attention < merge < marks[2] < gate < marks[3]
assert run.index('join_flags(', run.index('Op::PoolRope,')) < marks[0]
assert 'client.sync' not in run and 'block_on' not in run
assert run.rstrip().endswith('family_timing.mark("qsa_family_gate")\n}')
# Admission is exact production source. These small stand-ins instrument only
# host control flow, never an actual device/client/arena or a fake sync receipt.
header = r"""#![allow(dead_code)]
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
static ENABLED: AtomicBool = AtomicBool::new(false);
static LOOKUPS: AtomicU64 = AtomicU64::new(0);
static STARTS: AtomicU64 = AtomicU64::new(0);
mod kuno_text { pub type Result<T> = ResultAlias<T>; type ResultAlias<T> = std::result::Result<T, &'static str>; }
mod resident_util { pub fn fault(value: &'static str) -> &'static str { value } }
mod private_arena {
    pub struct PrivateArena { pub closed: bool }
    impl PrivateArena {
        pub fn client(&self) -> Result<&'static u32, &'static str> {
            super::LOOKUPS.fetch_add(1, super::Ordering::Relaxed);
            if self.closed { Err("original_arena_closed") } else { Ok(&711) }
        }
    }
}
struct Timer(Option<u32>);
impl Timer {
    fn new(client: &u32) -> kuno_text::Result<Self> {
        STARTS.fetch_add(1, Ordering::Relaxed);
        Ok(Self(Some(*client)))
    }
"""
footer = r"""
}
#[test]
fn actual_disabled_gate_never_looks_up_private_arena_or_starts_sync_timer() {
    ENABLED.store(false, Ordering::Release);
    let before = (LOOKUPS.load(Ordering::Relaxed), STARTS.load(Ordering::Relaxed));
    let timer = Timer::for_public_arena(&private_arena::PrivateArena { closed: true }).unwrap();
    assert!(timer.0.is_none());
    assert_eq!((LOOKUPS.load(Ordering::Relaxed), STARTS.load(Ordering::Relaxed)), before);
}
#[test]
fn actual_enabled_gate_uses_original_live_arena_once_and_preserves_failure() {
    ENABLED.store(true, Ordering::Release);
    let before = (LOOKUPS.load(Ordering::Relaxed), STARTS.load(Ordering::Relaxed));
    let timer = Timer::for_public_arena(&private_arena::PrivateArena { closed: false }).unwrap();
    assert_eq!(timer.0, Some(711));
    assert_eq!((LOOKUPS.load(Ordering::Relaxed), STARTS.load(Ordering::Relaxed)), (before.0 + 1, before.1 + 1));
    assert_eq!(Timer::for_public_arena(&private_arena::PrivateArena { closed: true }).err(), Some("original_arena_closed"));
    assert_eq!((LOOKUPS.load(Ordering::Relaxed), STARTS.load(Ordering::Relaxed)), (before.0 + 2, before.1 + 1));
    ENABLED.store(false, Ordering::Release);
}
"""
with tempfile.TemporaryDirectory(prefix='kuno-public-qsa-family-cpu-') as directory:
    directory = Path(directory)
    code = directory / 'actual-admission.rs'
    binary = directory / 'actual-admission-tests'
    code.write_text(header + method + footer)
    subprocess.run(['rustc', '--edition=2024', '--test', '-D', 'warnings', str(code), '-o', str(binary)], check=True)
    subprocess.run([str(binary), '--test-threads=1'], check=True)
print('actual public timer admission and full QSA producer/selection/attention/gate boundaries passed')
