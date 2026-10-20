#!/usr/bin/env python3
"""CPU-only admission checks for the actual compiled-public capacity receipt.

The original two admission functions are compiled with metadata-only stand-ins;
no model, GPU, real owner, fence, memory telemetry or recovery is simulated.
The whole worker and an ordinary protected public run remain acceptance gates.
"""
from pathlib import Path
import subprocess
import tempfile

root = Path(__file__).resolve().parents[4]
native = root / 'apps/kuno/benchmarks/native-text/src'
worker = (native / 'strata_worker.rs').read_text()
engine = (native / 'strata_engine.rs').read_text()
sources = (native / 'strata_global_sources.rs').read_text()
host = (native / 'strata_global_host.rs').read_text()
profile = (native / 'strata_prefill_profile.rs').read_text()


def function(source, signature):
    start = source.index(signature)
    brace = source.index('{', start)
    depth = 1
    end = brace + 1
    while depth:
        depth += (source[end] == '{') - (source[end] == '}')
        end += 1
    return source[start:end]


entry = function(worker, 'fn compiled_public_capacity_entry(')
admission = function(engine, 'pub(crate) fn compiled_public_capacity(&self)')
actual_sources = function(sources, 'pub(crate) fn compiled_public_capacity(')
actual_host = function(host, 'pub(crate) fn compiled_public_capacity(')
load = worker.index('engine::Session::load(')
attempt = worker.index('let attempted: Result<()> = (|| {', load)
gate = worker.index('if compiled_public_capacity_entry(public_check, prefill_check)', attempt)
receipt = worker.index('eprintln!("public_prefill_capacity {capacity}")', gate)
request = worker.index('let result = session.run(', receipt)
close = worker.index('session.close().map_err', request)
assert load < attempt < gate < receipt < request < close
assert worker.count('eprintln!("public_prefill_capacity {capacity}")') == 1
assert worker.count('.compiled_public_capacity()') == 1
assert worker.index('std::env::remove_var(name)') < worker.index('strata_prefill_profile::enable_compiled_public()')
assert worker.index('strata_prefill_profile::enable_compiled_public()') < worker.index('cubecl::future::block_on(serve(')
assert 'ENABLED.load(Ordering::Acquire)' in function(profile, 'pub(crate) fn compiled_public_enabled()')
# Read the SAME retained original plan and physical owners, rather than creating
# a new plan with a nominal cap or sampling request-derived placement/usage.
for fragment in [
    'self.validate_graph_ready(graph)?', 'self.plan.slots.len()',
    'self.plan.budget', 'self.plan.arena_bytes', 'gpu.bytes()',
    'gpu.roots().len()', 'host.compiled_public_capacity()?',
    'self.prompt_stage.is_some()', 'self.loan_pending',
]:
    assert fragment in actual_sources
for fragment in ['self.logical', 'self.physical', 'self.pages.len()', 'self.shadow.is_some()']:
    assert fragment in actual_host
for forbidden in ['Plan::from_', 'prompt_loan_quote', 'route_map(', 'borrow_cpu(',
                  'retained_resource_metadata(', 'sync(', 'memory_usage(',
                  'std::env', 'read_range(', 'read_private', 'placement.']:
    assert forbidden not in actual_sources + actual_host + admission
assert actual_host.index('self.closed') < actual_host.index('serde_json::json!')

header = r'''#![allow(dead_code)]
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
static ENABLED: AtomicBool = AtomicBool::new(false);
static LOOKUPS: AtomicUsize = AtomicUsize::new(0);
mod serde_json { pub type Value = u64; }
mod strata_prefill_profile {
    pub fn compiled_public_enabled() -> bool { super::ENABLED.load(super::Ordering::Acquire) }
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Failure { Rejected, Inference }
#[derive(Clone, Copy, PartialEq, Eq)]
enum State { Loading, Cold, Running, Ready, Closed, Broken }
struct Sources;
impl Sources {
    fn compiled_public_capacity(&self, graph: u64) -> Result<u64, ()> {
        LOOKUPS.fetch_add(1, Ordering::Relaxed);
        if graph == 731 { Ok(89) } else { Err(()) }
    }
}
struct Executor { strata_global_sources: Option<Sources> }
struct Session {
    state: State,
    runtime_owned: bool,
    close_attempted: bool,
    executor: Option<Executor>,
    graph: u64,
}
impl Session {
'''
footer = r'''
fn session() -> Session {
    Session { state: State::Cold, runtime_owned: true, close_attempted: false,
        executor: Some(Executor { strata_global_sources: Some(Sources) }), graph: 731 }
}
#[test]
fn only_explicit_compiled_public_profile_entry_selects_capacity() {
    for public in [false, true] {
        for profile in [None, Some(false), Some(true)] {
            assert_eq!(compiled_public_capacity_entry(public, profile), public && profile == Some(true));
        }
    }
}
#[test]
fn private_and_non_cold_or_retired_owners_never_reach_source_metadata() {
    ENABLED.store(false, Ordering::Release);
    let before = LOOKUPS.load(Ordering::Relaxed);
    assert_eq!(session().compiled_public_capacity(), Err(Failure::Rejected));
    ENABLED.store(true, Ordering::Release);
    for state in [State::Loading, State::Running, State::Ready, State::Closed, State::Broken] {
        let mut owner = session(); owner.state = state;
        assert_eq!(owner.compiled_public_capacity(), Err(Failure::Rejected));
    }
    let mut owner = session(); owner.runtime_owned = false;
    assert_eq!(owner.compiled_public_capacity(), Err(Failure::Rejected));
    let mut owner = session(); owner.close_attempted = true;
    assert_eq!(owner.compiled_public_capacity(), Err(Failure::Rejected));
    assert_eq!(LOOKUPS.load(Ordering::Relaxed), before);
}
#[test]
fn actual_cold_entry_reaches_original_source_and_propagates_missing_or_changed_owner() {
    ENABLED.store(true, Ordering::Release);
    let before = LOOKUPS.load(Ordering::Relaxed);
    assert_eq!(session().compiled_public_capacity(), Ok(89));
    let mut owner = session(); owner.graph = 732;
    assert_eq!(owner.compiled_public_capacity(), Err(Failure::Inference));
    let mut owner = session(); owner.executor = None;
    assert_eq!(owner.compiled_public_capacity(), Err(Failure::Inference));
    let mut owner = session(); owner.executor.as_mut().unwrap().strata_global_sources = None;
    assert_eq!(owner.compiled_public_capacity(), Err(Failure::Inference));
    assert_eq!(LOOKUPS.load(Ordering::Relaxed), before + 2);
}
'''
with tempfile.TemporaryDirectory(prefix='kuno-public-capacity-cpu-') as directory:
    directory = Path(directory)
    code = directory / 'actual-admission.rs'
    binary = directory / 'actual-admission-tests'
    code.write_text(header + admission + '\n}\n' + entry + footer)
    subprocess.run(['rustc', '--edition=2024', '--test', '-D', 'warnings', str(code), '-o', str(binary)], check=True)
    subprocess.run([str(binary), '--test-threads=1'], check=True)
print('actual compiled-public capacity admission and static-only owner call sites passed')
