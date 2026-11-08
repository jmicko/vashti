//! Sealed fixed-public split/join control. No model or device API in this module.
use kuno_text::{Error, Result};
use std::mem::{MaybeUninit, size_of};
use std::sync::{
    Arc, Condvar, Mutex,
    atomic::{AtomicBool, Ordering},
};
use std::time::{Duration, Instant};
use zeroize::{Zeroize, Zeroizing};

pub(crate) const RANKS: usize = 10;
fn err() -> Error {
    Error::InvalidFormat("public CPU/GPU expert schedule rejected")
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mode {
    Concurrent,
    CpuFirst,
    GpuFirst,
}
struct Data {
    cpu: [usize; RANKS],
    gpu: [usize; RANKS],
    map: [u32; RANKS * 2],
    nc: usize,
    ng: usize,
}
impl Zeroize for Data {
    fn zeroize(&mut self) {
        self.cpu.zeroize();
        self.gpu.zeroize();
        self.map.zeroize();
        self.nc.zeroize();
        self.ng.zeroize();
    }
}
pub(crate) struct Split(Zeroizing<Data>);
impl Split {
    pub(crate) fn new(ids: &[usize; RANKS], misses: usize, cpu: usize) -> Result<Self> {
        if misses > RANKS || cpu > misses {
            return Err(err());
        }
        Self::from_actual_misses(ids, &ids[RANKS - misses..], cpu)
    }
    /// Canonical cache classification is supplied by the staged owner. Pick
    /// capped actual misses in original rank order, never a contiguous suffix.
    /// This value only describes ownership; it grants no bank/fill authority.
    pub(crate) fn from_actual_misses(
        ids: &[usize; RANKS],
        misses: &[usize],
        cpu_cap: usize,
    ) -> Result<Self> {
        Self::with_gpu_limit(ids, misses, cpu_cap, RANKS)
    }
    /// Explicit static-bank co-execution: CPU can also compute resident hits.
    /// Cache residency, actual misses and compute placement remain distinct.
    pub(crate) fn with_gpu_limit(
        ids: &[usize; RANKS],
        misses: &[usize],
        cpu_cap: usize,
        gpu_limit: usize,
    ) -> Result<Self> {
        if gpu_limit > RANKS || (gpu_limit < RANKS && cpu_cap != RANKS) {
            return Err(err());
        }
        if misses.len() > RANKS
            || cpu_cap > RANKS
            || ids
                .iter()
                .enumerate()
                .any(|(r, &v)| v >= 512 || ids[..r].contains(&v))
            || misses
                .iter()
                .enumerate()
                .any(|(r, id)| !ids.contains(id) || misses[..r].contains(id))
        {
            return Err(err());
        }
        let mut s = Self(Zeroizing::new(Data {
            cpu: [0; RANKS],
            gpu: [0; RANKS],
            map: [0; RANKS * 2],
            nc: 0,
            ng: 0,
        }));
        for (r, &id) in ids.iter().enumerate() {
            let c = (misses.contains(&id) && s.0.nc < cpu_cap) || s.0.ng >= gpu_limit;
            let i = if c {
                let i = s.0.nc;
                s.0.cpu[i] = id;
                s.0.nc += 1;
                i
            } else {
                let i = s.0.ng;
                s.0.gpu[i] = id;
                s.0.ng += 1;
                i
            };
            s.0.map[r * 2] = u32::from(c);
            s.0.map[r * 2 + 1] = i as u32;
        }
        Ok(s)
    }
    pub(crate) fn cpu(&self) -> &[usize] {
        &self.0.cpu[..self.0.nc]
    }
    pub(crate) fn gpu(&self) -> &[usize] {
        &self.0.gpu[..self.0.ng]
    }
    pub(crate) fn map(&self) -> &[u32; RANKS * 2] {
        &self.0.map
    }
}
#[derive(Clone, Copy, Debug)]
pub struct Times {
    pub cpu: Duration,
    pub gpu: Duration,
    pub joined: Duration,
}
struct Gate {
    state: Mutex<(bool, bool, bool)>,
    cv: Condvar,
}
impl Gate {
    fn new() -> Self {
        Self {
            state: Mutex::new((false, false, false)),
            cv: Condvar::new(),
        }
    }
    fn worker(&self) -> Result<()> {
        let mut s = self.state.lock().map_err(|_| err())?;
        s.0 = true;
        self.cv.notify_all();
        while !s.1 && !s.2 {
            s = self.cv.wait(s).map_err(|_| err())?;
        }
        if s.2 { Err(err()) } else { Ok(()) }
    }
    fn release(&self) -> Result<()> {
        let mut s = self.state.lock().map_err(|_| err())?;
        while !s.0 && !s.2 {
            s = self.cv.wait(s).map_err(|_| err())?;
        }
        if s.2 {
            return Err(err());
        }
        s.1 = true;
        self.cv.notify_all();
        Ok(())
    }
    fn abort(&self) {
        if let Ok(mut s) = self.state.lock() {
            s.2 = true;
            self.cv.notify_all();
        }
    }
}
// Local sealed retained-header owner follows core cpu_expert's proven
// MaybeUninit ownership pattern without exposing that private core helper.
// Consumers are standard non-panicking ScopedJoinHandle, Handle/resource tuples.
// Dropping a scope joins workers before borrowed inputs may release.
pub(crate) struct Metadata<T> {
    storage: MaybeUninit<Vec<MaybeUninit<T>>>,
    first: Zeroizing<usize>,
    end: Zeroizing<usize>,
    #[cfg(test)]
    witness: Option<Arc<(AtomicBool, AtomicBool)>>,
}
impl<T> Metadata<T> {
    pub(crate) fn new(n: usize) -> Result<Self> {
        Self::bytes_for(n)?;
        let mut slots = Vec::<MaybeUninit<T>>::new();
        slots.try_reserve_exact(n).map_err(|_| err())?;
        if slots.capacity() != n {
            return Err(err());
        }
        // SAFETY: full reserved aligned MaybeUninit capacity accepts all bits;
        // no T is live yet, and set_len only exposes MaybeUninit slots.
        unsafe {
            std::ptr::write_bytes(slots.as_mut_ptr().cast::<u8>(), 0, n * size_of::<T>());
            slots.set_len(n);
        }
        Ok(Self {
            storage: MaybeUninit::new(slots),
            first: Zeroizing::new(0),
            end: Zeroizing::new(0),
            #[cfg(test)]
            witness: None,
        })
    }
    pub(crate) fn bytes_for(n: usize) -> Result<usize> {
        if !(1..=24).contains(&n) {
            return Err(err());
        }
        let b = n
            .checked_mul(size_of::<T>())
            .and_then(|n| n.checked_add(size_of::<Self>()))
            .ok_or_else(err)?;
        if b > 32768 {
            return Err(err());
        }
        Ok(b)
    }
    fn slots(&self) -> &Vec<MaybeUninit<T>> {
        // SAFETY: constructor initializes Vec; only Drop ends it.
        unsafe { self.storage.assume_init_ref() }
    }
    fn slots_mut(&mut self) -> &mut Vec<MaybeUninit<T>> {
        // SAFETY: same proof, unique mutable owner.
        unsafe { self.storage.assume_init_mut() }
    }
    pub(crate) fn metadata_bytes(&self) -> Result<usize> {
        Self::bytes_for(self.slots().capacity())
    }
    pub(crate) fn len(&self) -> usize {
        *self.end
    }
    pub(crate) fn push(&mut self, t: T) -> Result<()> {
        let n = *self.end;
        if n >= self.slots().len() {
            return Err(err());
        }
        self.slots_mut()[n].write(t);
        *self.end += 1;
        Ok(())
    }
    pub(crate) fn get(&self, n: usize) -> Result<&T> {
        if n < *self.first || n >= *self.end {
            return Err(err());
        }
        // SAFETY: first..end initialized once and not yet consumed; immutable borrow pins it.
        Ok(unsafe { self.slots()[n].assume_init_ref() })
    }
    fn take_first(&mut self) -> Option<T> {
        let n = *self.first;
        if n >= *self.end {
            return None;
        }
        *self.first += 1;
        // SAFETY: consume exactly one initialized value; dead bits stay guarded until Drop.
        Some(unsafe { self.slots()[n].assume_init_read() })
    }
}
unsafe fn wipe(p: *mut u8, n: usize) {
    for i in 0..n {
        // SAFETY: caller proves uniquely owned dead span of n bytes.
        unsafe { std::ptr::write_volatile(p.add(i), 0) };
    }
    std::sync::atomic::compiler_fence(Ordering::SeqCst);
}
impl<T> Drop for Metadata<T> {
    fn drop(&mut self) {
        let first = Zeroizing::new(*self.first);
        let end = Zeroizing::new(*self.end);
        let slots = self.slots_mut();
        for slot in &mut slots[*first..*end] {
            // SAFETY: only initialized unconsumed T values drop once.
            unsafe { slot.assume_init_drop() };
        }
        let n = slots.capacity() * size_of::<T>();
        let p = slots.as_mut_ptr().cast::<u8>();
        // SAFETY: no T remains live; whole allocation remains uniquely owned.
        unsafe { wipe(p, n) };
        #[cfg(test)]
        if let Some(w) = &self.witness {
            // SAFETY: volatile wipe initialized every byte while allocation stays owned.
            let zeros = unsafe { std::slice::from_raw_parts(p, n) };
            w.0.store(zeros.iter().all(|&b| b == 0), Ordering::SeqCst);
        }
        // SAFETY: initialized Vec destroys only MaybeUninit slots and frees the erased allocation.
        unsafe { self.storage.assume_init_drop() };
        let p = std::ptr::addr_of_mut!(self.storage).cast::<u8>();
        // SAFETY: dead outer Vec header remains inside self until Drop returns.
        unsafe { wipe(p, size_of::<Vec<MaybeUninit<T>>>()) };
        #[cfg(test)]
        if let Some(w) = &self.witness {
            // SAFETY: wiped dead header is initialized bytes, still owned by self.
            let zeros = unsafe { std::slice::from_raw_parts(p, size_of::<Vec<MaybeUninit<T>>>()) };
            w.1.store(zeros.iter().all(|&b| b == 0), Ordering::SeqCst);
        }
    }
}
fn caught<T>(f: impl FnOnce() -> Result<T>) -> Result<T> {
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(f)).unwrap_or_else(|_| Err(err()))
}
fn checked<T>(cancel: &AtomicBool, f: impl FnOnce() -> Result<T>) -> Result<T> {
    let r = caught(f);
    if r.is_err() {
        cancel.store(true, Ordering::Release);
    }
    if cancel.load(Ordering::Acquire) {
        Err(err())
    } else {
        r
    }
}
/// Used by the real runner. GPU callback MUST complete its owning fence and
/// checked I/O before returning. Every CPU thread joins on either-side failure.
pub(crate) fn pair<T: Send, G>(
    mode: Mode,
    cancel: &AtomicBool,
    has_cpu: bool,
    cpu: impl FnOnce() -> Result<T> + Send,
    gpu: impl FnOnce() -> Result<G>,
) -> Result<(Option<T>, G, Times)> {
    if cancel.load(Ordering::Acquire) {
        return Err(err());
    }
    let start = Instant::now();
    if !has_cpu {
        let gs = Instant::now();
        let g = checked(cancel, gpu)?;
        return Ok((
            None,
            g,
            Times {
                cpu: Duration::ZERO,
                gpu: gs.elapsed(),
                joined: start.elapsed(),
            },
        ));
    }
    std::thread::scope(|scope| {
        let gate = Arc::new(Gate::new());
        let other = gate.clone();
        let mut slot = Metadata::new(1)?;
        if Metadata::<std::thread::ScopedJoinHandle<'_, Result<(T, Duration)>>>::bytes_for(1)?
            > 4096
        {
            return Err(err());
        }
        let spawned = std::thread::Builder::new()
            .name("kuno-public-expert".into())
            .spawn_scoped(scope, move || {
                let result = caught(|| {
                    other.worker()?;
                    let s = Instant::now();
                    let c = cpu()?;
                    Ok((c, s.elapsed()))
                });
                if result.is_err() {
                    cancel.store(true, Ordering::Release);
                    other.abort();
                }
                result
            });
        match spawned {
            Ok(h) => slot.push(h)?,
            Err(_) => {
                gate.abort();
                cancel.store(true, Ordering::Release);
                return Err(err());
            }
        }
        // Every mode uses the same scoped orchestrator, physical CPU workers
        // and join owner. Only release order changes in the serial controls.
        let mut early = None;
        if mode == Mode::CpuFirst {
            gate.release()?;
            let c = slot
                .take_first()
                .ok_or_else(err)?
                .join()
                .map_err(|_| err())
                .and_then(|v| v);
            if c.is_err() {
                cancel.store(true, Ordering::Release);
                gate.abort();
            }
            early = Some(c?);
        } else if mode == Mode::Concurrent {
            gate.release()?;
        }
        let gs = Instant::now();
        let g = caught(gpu);
        let gt = gs.elapsed();
        if g.is_err() {
            cancel.store(true, Ordering::Release);
            gate.abort();
        }
        if mode == Mode::GpuFirst && g.is_ok() {
            gate.release()?;
        }
        let c = if let Some(c) = early {
            Ok(c)
        } else {
            slot.take_first()
                .ok_or_else(err)?
                .join()
                .map_err(|_| err())
                .and_then(|v| v)
        };
        if c.is_err() {
            cancel.store(true, Ordering::Release);
            gate.abort();
        }
        let (c, ct) = c?;
        let g = g?;
        if cancel.load(Ordering::Acquire) {
            return Err(err());
        }
        Ok((
            Some(c),
            g,
            Times {
                cpu: ct,
                gpu: gt,
                joined: start.elapsed(),
            },
        ))
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn every_split_covers_original_ranks_exactly_and_miss_endpoints() {
        let ids = std::array::from_fn(|r| 511 - r * 17);
        for m in 0..=10 {
            for k in 0..=m {
                let s = Split::new(&ids, m, k).unwrap();
                assert_eq!(s.cpu().len(), k);
                assert_eq!(s.gpu().len(), 10 - k);
                for (r, &id) in ids.iter().enumerate() {
                    let i = s.map()[r * 2 + 1] as usize;
                    assert_eq!(
                        if s.map()[r * 2] == 1 {
                            s.cpu()[i]
                        } else {
                            s.gpu()[i]
                        },
                        id
                    );
                }
            }
        }
        assert!(Split::new(&ids, 11, 0).is_err());
        assert!(Split::new(&[0; 10], 1, 1).is_err());
    }
    #[test]
    fn production_pair_joins_and_drops_guard_before_gpu_error_or_panic_returns() {
        for panic in [false, true] {
            let live = AtomicBool::new(false);
            let c = AtomicBool::new(false);
            let result = pair(
                Mode::Concurrent,
                &c,
                true,
                || {
                    live.store(true, Ordering::Release);
                    while !c.load(Ordering::Acquire) {
                        std::thread::yield_now();
                    }
                    live.store(false, Ordering::Release);
                    Ok(Zeroizing::new(vec![3.0f32; 64]))
                },
                || -> Result<()> {
                    if panic {
                        panic!("public injected GPU panic");
                    }
                    Err(err())
                },
            );
            assert!(result.is_err());
            assert!(!live.load(Ordering::Acquire));
            assert!(c.load(Ordering::Acquire));
        }
    }
    #[test]
    fn actual_driver_serial_modes_and_no_cpu_endpoint() {
        for mode in [Mode::Concurrent, Mode::CpuFirst, Mode::GpuFirst] {
            let c = AtomicBool::new(false);
            let (c, g, _) = pair(mode, &c, true, || Ok(17), || Ok(23)).unwrap();
            assert_eq!((c, g), (Some(17), 23));
        }
        let c = AtomicBool::new(false);
        let (v, g, t) = pair(
            Mode::Concurrent,
            &c,
            false,
            || -> Result<()> { panic!("empty CPU was executed") },
            || Ok(2),
        )
        .unwrap();
        assert!(v.is_none());
        assert_eq!(g, 2);
        assert_eq!(t.cpu, Duration::ZERO);
    }
    #[test]
    fn every_mode_uses_scoped_orchestrator_and_rejects_cpu_error_panic_precancel() {
        let parent = std::thread::current().id();
        for mode in [Mode::CpuFirst, Mode::GpuFirst, Mode::Concurrent] {
            let c = AtomicBool::new(false);
            pair(
                mode,
                &c,
                true,
                || {
                    assert_ne!(std::thread::current().id(), parent);
                    Ok(())
                },
                || Ok(()),
            )
            .unwrap();
            for panic in [false, true] {
                let c = AtomicBool::new(false);
                let result = pair(
                    mode,
                    &c,
                    true,
                    || -> Result<()> {
                        if panic {
                            panic!("fixed public CPU panic");
                        }
                        Err(err())
                    },
                    || Ok(()),
                );
                assert!(result.is_err());
                assert!(c.load(Ordering::Acquire));
            }
        }
        let c = AtomicBool::new(true);
        assert!(
            pair(
                Mode::Concurrent,
                &c,
                true,
                || -> Result<()> { panic!("precancel CPU executed") },
                || -> Result<()> { panic!("precancel GPU executed") }
            )
            .is_err()
        );
    }
    #[test]
    fn metadata_erases_full_dead_and_spare_slots_and_outer_header() {
        let witness = Arc::new((AtomicBool::new(false), AtomicBool::new(false)));
        {
            let mut owner = Metadata::new(4).unwrap();
            owner.witness = Some(witness.clone());
            owner.push([0xabu8; 128]).unwrap();
            assert_eq!(owner.take_first().unwrap(), [0xab; 128]);
            owner.push([0xcdu8; 128]).unwrap();
            assert_eq!(owner.get(1).unwrap(), &[0xcd; 128]);
            assert!(owner.get(0).is_err());
            assert_eq!(
                Metadata::<[u8; 128]>::bytes_for(4).unwrap(),
                4 * 128 + size_of::<Metadata<[u8; 128]>>()
            );
        }
        assert!(witness.0.load(Ordering::Acquire));
        assert!(witness.1.load(Ordering::Acquire));
        assert!(Metadata::<u8>::new(0).is_err());
        assert!(Metadata::<u8>::new(25).is_err());
    }
}
