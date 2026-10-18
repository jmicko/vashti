//! MIT Strata fb58e0 Linux PLE reader: original eight-way packed cache,
//! sorted page jobs,256 slots,64 blocking direct readers and coordinator.
//! Kuno substitution: one wiped request lifetime, never a retained row trace.
use kuno_text::{
    Error, Result,
    gguf::DirectSource,
    model::{BoundGraph, GraphBindingId},
};
use std::{
    cell::RefCell,
    collections::VecDeque,
    sync::{
        Arc, Condvar, Mutex,
        atomic::{AtomicBool, AtomicUsize, Ordering},
        mpsc,
    },
    time::{Duration, Instant},
};
use zeroize::{Zeroize, Zeroizing};
const ROWS: usize = 1 << 20;
const WAYS: usize = 8;
const DEPTH: usize = 256;
const WORKERS: usize = 64;
const PAGE: usize = 4096;
const MAX_ROWS: usize = 4096 * 16;
const MAX_RB: usize = 110;
// Cache includes the largest supported original row format. Two caller plans/
// raw results can coexist with a prompt producer; all capacities are bounded.
const CACHE_HOST: usize = ROWS * (MAX_RB + 4) + ROWS / WAYS;
const WORKER_HOST: usize = WORKERS * 128 * 1024 + 288 * 1024;
const PLAN_HOST: usize = 2 * MAX_ROWS * (size_of::<Use>() + size_of::<Job>() + 4);
const RAW_HOST: usize = 2 * MAX_ROWS * MAX_RB;
const SLOT_HOST: usize = DEPTH * (2 * PAGE + size_of::<Mutex<Box<Aligned>>>());
pub(crate) const HOST_BYTES: usize =
    (CACHE_HOST + WORKER_HOST + PLAN_HOST + RAW_HOST + SLOT_HOST + 256 * 1024).div_ceil(65536)
        * 65536;
fn bad() -> Error {
    Error::InvalidFormat("native PLE reader owner/transition")
}
fn canceled(cancel: &AtomicBool) -> Result<()> {
    if cancel.load(Ordering::Acquire) {
        Err(bad())
    } else {
        Ok(())
    }
}
fn zeros<T: Zeroize + Default + Clone>(n: usize) -> Result<Zeroizing<Vec<T>>> {
    let mut v = Zeroizing::new(Vec::new());
    v.try_reserve_exact(n).map_err(|_| bad())?;
    if v.capacity() > n {
        return Err(bad());
    }
    v.resize(n, T::default());
    Ok(v)
}
struct Cache {
    keys: Zeroizing<Vec<u32>>,
    data: Zeroizing<Vec<u8>>,
    next: Zeroizing<Vec<u8>>,
    sets: usize,
    rb: usize,
}
impl Cache {
    fn new(rows: usize, rb: usize) -> Result<Self> {
        if rows == 0 || !rows.is_multiple_of(WAYS) || rows > ROWS || rb == 0 || rb > MAX_RB {
            return Err(bad());
        }
        let mut keys = zeros(rows)?;
        keys.fill(u32::MAX);
        Ok(Self {
            keys,
            data: zeros(rows.checked_mul(rb).ok_or(bad())?)?,
            next: zeros(rows / WAYS)?,
            sets: rows / WAYS,
            rb,
        })
    }
    fn set(&self, row: u32) -> usize {
        let x = u64::from(row).wrapping_mul(0x9E3779B97F4A7C15);
        ((x ^ (x >> 29)) % self.sets as u64) as usize
    }
    fn get(&self, row: u32) -> Option<&[u8]> {
        if row == u32::MAX {
            return None;
        }
        let first = self.set(row) * WAYS;
        (first..first + WAYS)
            .find(|&i| self.keys[i] == row)
            .map(|i| &self.data[i * self.rb..(i + 1) * self.rb])
    }
    fn insert(&mut self, row: u32, bytes: &[u8]) -> Result<()> {
        if bytes.len() != self.rb || row == u32::MAX {
            return Err(bad());
        }
        if self.get(row).is_some() {
            return Ok(());
        }
        let s = self.set(row);
        let index = s * WAYS + self.next[s] as usize;
        self.next[s] = (self.next[s] + 1) % WAYS as u8;
        self.keys[index] = row;
        self.data[index * self.rb..(index + 1) * self.rb].copy_from_slice(bytes);
        Ok(())
    }
    fn erase(&mut self) -> Result<()> {
        self.keys.as_mut_slice().zeroize();
        self.data.as_mut_slice().zeroize();
        self.next.as_mut_slice().zeroize();
        if self.keys.iter().any(|&x| x != 0)
            || self.data.iter().any(|&x| x != 0)
            || self.next.iter().any(|&x| x != 0)
        {
            return Err(bad());
        }
        Ok(())
    }
}
#[derive(Clone, Default)]
struct Use {
    row: u32,
    destination: u32,
    within: u32,
}
impl Zeroize for Use {
    fn zeroize(&mut self) {
        self.row.zeroize();
        self.destination.zeroize();
        self.within.zeroize();
    }
}
#[derive(Clone, Default)]
struct Job {
    offset: u64,
    length: usize,
    first: usize,
    end: usize,
}
impl Zeroize for Job {
    fn zeroize(&mut self) {
        self.offset.zeroize();
        self.length.zeroize();
        self.first.zeroize();
        self.end.zeroize();
    }
}
struct Plan {
    uses: Zeroizing<Vec<Use>>,
    jobs: Zeroizing<Vec<Job>>,
}
fn plan(rows: &[u32], offset: u64, nrows: u64, cache: &Cache, raw: &mut [u8]) -> Result<Plan> {
    if rows.is_empty() || rows.len() > MAX_ROWS || raw.len() != rows.len() * cache.rb {
        return Err(bad());
    }
    let mut uses = zeros::<Use>(rows.len())?;
    let mut count = 0;
    for (i, &row) in rows.iter().enumerate() {
        if row == u32::MAX || u64::from(row) >= nrows {
            return Err(bad());
        }
        if let Some(hit) = cache.get(row) {
            raw[i * cache.rb..(i + 1) * cache.rb].copy_from_slice(hit);
            continue;
        }
        let at = offset
            .checked_add(u64::from(row).checked_mul(cache.rb as u64).ok_or(bad())?)
            .ok_or(bad())?;
        uses[count] = Use {
            row,
            destination: i as u32,
            within: (at % PAGE as u64) as u32,
        };
        count += 1;
    }
    // Keep full initialized allocations for erasure; inactive tails stay zero.
    uses[..count].sort_unstable_by_key(|u| offset + u64::from(u.row) * cache.rb as u64);
    let mut jobs = zeros::<Job>(count)?;
    let mut nj = 0;
    for i in 0..count {
        let at = offset + u64::from(uses[i].row) * cache.rb as u64;
        let page = at / PAGE as u64 * PAGE as u64;
        let length = (uses[i].within as usize + cache.rb).div_ceil(PAGE) * PAGE;
        if nj > 0 && jobs[nj - 1].offset == page {
            jobs[nj - 1].length = jobs[nj - 1].length.max(length);
            jobs[nj - 1].end = i + 1;
        } else {
            jobs[nj] = Job {
                offset: page,
                length,
                first: i,
                end: i + 1,
            };
            nj += 1;
        }
    }
    // Only public lengths are shortened; wipe the full original capacity on drop.
    jobs[nj..].iter_mut().for_each(Zeroize::zeroize);
    jobs.truncate(nj);
    uses[count..].iter_mut().for_each(Zeroize::zeroize);
    uses.truncate(count);
    Ok(Plan { uses, jobs })
}
#[repr(align(4096))]
struct Aligned([u8; 2 * PAGE]);
impl Drop for Aligned {
    fn drop(&mut self) {
        self.0.zeroize();
    }
}
struct ReadJob {
    slot: usize,
    index: usize,
    offset: u64,
    length: usize,
}
impl Drop for ReadJob {
    fn drop(&mut self) {
        self.slot.zeroize();
        self.index.zeroize();
        self.offset.zeroize();
        self.length.zeroize();
    }
}
struct Done {
    slot: usize,
    index: usize,
    result: Result<usize>,
}
struct Queue {
    pending: VecDeque<ReadJob>,
    done: VecDeque<Done>,
    stop: bool,
    failed: bool,
    last_read: Option<Instant>,
}
struct Pool {
    queue: Mutex<Queue>,
    work: Condvar,
    done: Condvar,
    buffers: Vec<Mutex<Box<Aligned>>>,
}
impl Pool {
    fn new() -> Result<Self> {
        let buffers = (0..DEPTH)
            .map(|_| Mutex::new(Box::new(Aligned([0; 2 * PAGE]))))
            .collect();
        Ok(Self {
            queue: Mutex::new(Queue {
                pending: VecDeque::with_capacity(DEPTH),
                done: VecDeque::with_capacity(DEPTH),
                stop: false,
                failed: false,
                last_read: None,
            }),
            work: Condvar::new(),
            done: Condvar::new(),
            buffers,
        })
    }
    fn read_idle(&self) -> Result<Duration> {
        Ok(self
            .queue
            .lock()
            .map_err(|_| bad())?
            .last_read
            .map(|t| t.elapsed())
            .unwrap_or(Duration::MAX))
    }
    fn worker(&self, source: &DirectSource<'_>) -> Result<()> {
        loop {
            let job = {
                let mut q = self.queue.lock().map_err(|_| bad())?;
                while q.pending.is_empty() && !q.stop {
                    q = self.work.wait(q).map_err(|_| bad())?;
                }
                match q.pending.front_mut() {
                    Some(j) => {
                        let owned = ReadJob {
                            slot: j.slot,
                            index: j.index,
                            offset: j.offset,
                            length: j.length,
                        };
                        j.slot.zeroize();
                        j.index.zeroize();
                        j.offset.zeroize();
                        j.length.zeroize();
                        q.pending.pop_front();
                        q.last_read = Some(Instant::now());
                        owned
                    }
                    None => return Ok(()),
                }
            };
            let result = {
                let mut buffer = self.buffers[job.slot].lock().map_err(|_| bad())?;
                source.read_aligned(job.offset, &mut buffer.0[..job.length])
            };
            let mut q = self.queue.lock().map_err(|_| bad())?;
            if q.done.len() >= DEPTH {
                return Err(bad());
            }
            q.done.push_back(Done {
                slot: job.slot,
                index: job.index,
                result,
            });
            self.done.notify_one();
        }
    }
    fn run<T>(&self, source: &DirectSource<'_>, body: impl FnOnce() -> Result<T>) -> Result<T> {
        std::thread::scope(|scope| {
            let mut threads = Vec::with_capacity(WORKERS);
            let mut startup = None;
            for _ in 0..WORKERS {
                match std::thread::Builder::new()
                    .stack_size(96 * 1024)
                    .spawn_scoped(scope, || {
                        let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                            self.worker(source)
                        }));
                        if !matches!(r, Ok(Ok(()))) {
                            let mut q = self.queue.lock().unwrap_or_else(|e| e.into_inner());
                            q.failed = true;
                            self.done.notify_all();
                            self.work.notify_all();
                            return Err(bad());
                        }
                        Ok(())
                    }) {
                    Ok(t) => threads.push(t),
                    Err(_) => {
                        startup = Some(bad());
                        break;
                    }
                }
            }
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                if startup.is_some() {
                    Err(bad())
                } else {
                    body()
                }
            }));
            {
                let mut q = self.queue.lock().unwrap_or_else(|e| e.into_inner());
                q.stop = true;
                self.work.notify_all();
            }
            let mut failed = false;
            for t in threads {
                if !matches!(t.join(), Ok(Ok(()))) {
                    failed = true;
                }
            }
            // No worker can touch buffers/queues beyond this real join.
            for b in &self.buffers {
                let mut b = b.lock().unwrap_or_else(|e| e.into_inner());
                b.0.zeroize();
                if b.0.iter().any(|&v| v != 0) {
                    failed = true;
                }
            }
            let mut q = self.queue.lock().unwrap_or_else(|e| e.into_inner());
            if !q.pending.is_empty() || !q.done.is_empty() {
                failed = true;
                q.pending.clear();
                q.done.clear();
            }
            failed |= q.failed;
            drop(q);
            match result {
                Err(panic) => std::panic::resume_unwind(panic),
                Ok(_) if failed => Err(bad()),
                Ok(result) => result,
            }
        })
    }
    fn execute(
        &self,
        p: &Plan,
        raw: &mut [u8],
        cache: &mut Cache,
        cancel: &AtomicBool,
    ) -> Result<()> {
        let mut next = 0;
        let mut pending = 0;
        let mut failure = None;
        let mut active = Zeroizing::new([usize::MAX; DEPTH]);
        let mut free: Vec<_> = (0..DEPTH).rev().collect();
        loop {
            if cancel.load(Ordering::Acquire) && failure.is_none() {
                failure = Some(bad());
            }
            while failure.is_none() && next < p.jobs.len() && !free.is_empty() {
                let slot = free.pop().ok_or(bad())?;
                let j = &p.jobs[next];
                let mut q = self.queue.lock().map_err(|_| bad())?;
                if q.pending.len() >= DEPTH || q.stop {
                    return Err(bad());
                }
                active[slot] = next;
                pending += 1;
                q.pending.push_back(ReadJob {
                    slot,
                    index: next,
                    offset: j.offset,
                    length: j.length,
                });
                next += 1;
                self.work.notify_one();
            }
            if pending == 0 {
                break;
            }
            let c = {
                let mut q = self.queue.lock().map_err(|_| bad())?;
                while q.done.is_empty() && !q.failed {
                    q = self.done.wait(q).map_err(|_| bad())?;
                }
                if q.failed {
                    return Err(bad());
                }
                q.done.pop_front().ok_or(bad())?
            };
            if c.slot >= DEPTH || active[c.slot] != c.index {
                return Err(bad());
            }
            active[c.slot] = usize::MAX;
            pending -= 1;
            let mut buffer = self.buffers[c.slot].lock().map_err(|_| bad())?;
            match c.result {
                Ok(n) if failure.is_none() => {
                    let j = &p.jobs[c.index];
                    if n > j.length
                        || p.uses[j.first..j.end]
                            .iter()
                            .any(|u| u.within as usize + cache.rb > n)
                    {
                        failure = Some(bad());
                    } else {
                        for u in &p.uses[j.first..j.end] {
                            let bytes = &buffer.0[u.within as usize..u.within as usize + cache.rb];
                            let at = u.destination as usize * cache.rb;
                            raw[at..at + cache.rb].copy_from_slice(bytes);
                            if let Err(e) = cache.insert(u.row, bytes) {
                                failure = Some(e);
                                break;
                            }
                        }
                    }
                }
                Err(e) if failure.is_none() => {
                    failure = Some(e);
                }
                _ => (),
            }
            buffer.0.zeroize();
            free.push(c.slot);
        }
        // Every already submitted/queued read drained before return, even when
        // later requests were canceled or one direct read failed.
        if let Some(e) = failure {
            raw.zeroize();
            Err(e)
        } else {
            Ok(())
        }
    }
}
struct Permit(Arc<AtomicUsize>);
impl Drop for Permit {
    fn drop(&mut self) {
        self.0.fetch_sub(1, Ordering::AcqRel);
    }
}
/// Actual returned rows retain their allocation permit through decode/scatter.
/// No Clone, Debug, serialization or detached raw allocation is available.
pub(crate) struct OwnedRows {
    data: Zeroizing<Vec<u8>>,
    _permit: Arc<Permit>,
}
impl std::ops::Deref for OwnedRows {
    type Target = [u8];
    fn deref(&self) -> &[u8] {
        &self.data
    }
}
struct Gather {
    rows: Zeroizing<Vec<u32>>,
    permit: Arc<Permit>,
    reply: mpsc::SyncSender<Result<OwnedRows>>,
}
#[derive(Clone, Copy, PartialEq, Eq)]
enum Binding {
    Model(GraphBindingId),
    #[cfg(test)]
    Fixture(u64),
}
struct Endpoint {
    binding: Binding,
    rb: usize,
    tx: mpsc::SyncSender<Gather>,
    stop: Arc<AtomicBool>,
    outstanding: Arc<AtomicUsize>,
}
#[derive(Clone)]
pub(crate) struct Lease(Arc<Endpoint>);
/// Owned source operation. Dropping it closes delivery, but its permit remains
/// with actual queued/reading/returned rows until all of those owners end.
pub(crate) struct PendingRows {
    binding: Binding,
    rb: usize,
    count: usize,
    rx: mpsc::Receiver<Result<OwnedRows>>,
    _permit: Arc<Permit>,
    stop: Arc<AtomicBool>,
}
impl PendingRows {
    pub(crate) fn collect(
        self,
        binding: GraphBindingId,
        rb: usize,
        cancel: &AtomicBool,
    ) -> Result<OwnedRows> {
        self.collect_rows(Binding::Model(binding), rb, cancel)
    }
    fn collect_rows(self, binding: Binding, rb: usize, cancel: &AtomicBool) -> Result<OwnedRows> {
        canceled(cancel)?;
        if self.binding != binding || self.rb != rb || self.stop.load(Ordering::Acquire) {
            return Err(bad());
        }
        let result = self.rx.recv().map_err(|_| bad())??;
        canceled(cancel)?;
        if result.len() != self.count * rb || self.stop.load(Ordering::Acquire) {
            return Err(bad());
        }
        Ok(result)
    }
}
thread_local! {static CURRENT:RefCell<Option<Lease>>=const{RefCell::new(None)};}
pub(crate) fn current() -> Option<Lease> {
    CURRENT.with(|s| s.borrow().clone())
}
struct Installed;
impl Drop for Installed {
    fn drop(&mut self) {
        CURRENT.with(|s| *s.borrow_mut() = None);
    }
}
impl Lease {
    pub(crate) fn issue(
        &self,
        binding: GraphBindingId,
        rb: usize,
        rows: &[u32],
        cancel: &AtomicBool,
    ) -> Result<PendingRows> {
        self.issue_rows(Binding::Model(binding), rb, rows, cancel)
    }
    pub(crate) fn gather(
        &self,
        binding: GraphBindingId,
        rb: usize,
        rows: &[u32],
        cancel: &AtomicBool,
    ) -> Result<OwnedRows> {
        self.issue(binding, rb, rows, cancel)?
            .collect(binding, rb, cancel)
    }
    #[cfg(test)]
    fn read_rows(
        &self,
        binding: Binding,
        rb: usize,
        rows: &[u32],
        cancel: &AtomicBool,
    ) -> Result<OwnedRows> {
        self.issue_rows(binding, rb, rows, cancel)?
            .collect_rows(binding, rb, cancel)
    }
    fn issue_rows(
        &self,
        binding: Binding,
        rb: usize,
        rows: &[u32],
        cancel: &AtomicBool,
    ) -> Result<PendingRows> {
        canceled(cancel)?;
        if binding != self.0.binding
            || rb != self.0.rb
            || rows.is_empty()
            || rows.len() > MAX_ROWS
            || self.0.stop.load(Ordering::Acquire)
        {
            return Err(bad());
        }
        self.0
            .outstanding
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |n| {
                (n < 2).then_some(n + 1)
            })
            .map_err(|_| bad())?;
        let permit = Arc::new(Permit(self.0.outstanding.clone()));
        let mut owned = zeros(rows.len())?;
        owned.copy_from_slice(rows);
        // One ready result may wait for the real layer boundary; unlike a
        // rendezvous reply it never stalls the coordinator/keepalive itself.
        let (tx, rx) = mpsc::sync_channel(1);
        self.0
            .tx
            .send(Gather {
                rows: owned,
                permit: permit.clone(),
                reply: tx,
            })
            .map_err(|_| bad())?;
        Ok(PendingRows {
            binding,
            rb,
            count: rows.len(),
            rx,
            _permit: permit,
            stop: self.0.stop.clone(),
        })
    }
}
pub(crate) fn with_request<T>(
    graph: BoundGraph<'_, '_>,
    cancel: &AtomicBool,
    body: impl FnOnce() -> Result<T>,
) -> Result<T> {
    canceled(cancel)?;
    if CURRENT.with(|s| s.borrow().is_some()) {
        return Err(bad());
    }
    graph.verify_unchanged()?;
    let table = graph.tensor("per_layer_token_embd.weight")?;
    let rb = table.info().row_bytes() as usize;
    if table.info().dimensions() != [160, 320001536] || !matches!(rb, 90 | 110) {
        return Err(bad());
    }
    let source = table.positional_reader()?.direct_source(table)?;
    with_source(
        Binding::Model(graph.binding_id()),
        source,
        ROWS,
        cancel,
        body,
    )
}
fn with_source<T>(
    binding: Binding,
    source: DirectSource<'_>,
    cache_rows: usize,
    cancel: &AtomicBool,
    body: impl FnOnce() -> Result<T>,
) -> Result<T> {
    canceled(cancel)?;
    if CURRENT.with(|s| s.borrow().is_some()) {
        return Err(bad());
    }
    let rb = source.row_bytes() as usize;
    let (tx, rx) = mpsc::sync_channel::<Gather>(0);
    let (ready, wait) = mpsc::sync_channel(0);
    let stop = Arc::new(AtomicBool::new(false));
    let outstanding = Arc::new(AtomicUsize::new(0));
    std::thread::scope(|scope| {
        let stopped = stop.clone();
        let worker = std::thread::Builder::new()
            .stack_size(256 * 1024)
            .spawn_scoped(scope, move || -> Result<()> {
                let mut cache = match Cache::new(cache_rows, rb) {
                    Ok(c) => c,
                    Err(e) => {
                        let _ = ready.send(Err(bad()));
                        return Err(e);
                    }
                };
                let pool = match Pool::new() {
                    Ok(p) => p,
                    Err(e) => {
                        let _ = ready.send(Err(bad()));
                        return Err(e);
                    }
                };
                let result = pool.run(&source, || {
                    ready.send(Ok(())).map_err(|_| bad())?;
                    let mut last_issue = None;
                    let mut rng = 0x9E3779B97F4A7C15u64;
                    let mut keep = true;
                    while !stopped.load(Ordering::Acquire) && !cancel.load(Ordering::Acquire) {
                        match rx.recv_timeout(Duration::from_millis(10)) {
                            Ok(g) => {
                                last_issue = Some(Instant::now());
                                let result = (|| {
                                    canceled(cancel)?;
                                    source.verify_file_unchanged()?;
                                    let mut raw = zeros(g.rows.len() * rb)?;
                                    let p = plan(
                                        &g.rows,
                                        source.table_offset(),
                                        source.rows(),
                                        &cache,
                                        &mut raw,
                                    )?;
                                    pool.execute(&p, &mut raw, &mut cache, cancel)?;
                                    source.verify_file_unchanged()?;
                                    Ok(OwnedRows {
                                        data: raw,
                                        _permit: g.permit.clone(),
                                    })
                                })();
                                let failed = result.is_err();
                                let _ = g.reply.send(result);
                                if failed {
                                    return Err(bad());
                                }
                            }
                            Err(mpsc::RecvTimeoutError::Disconnected) => break,
                            Err(mpsc::RecvTimeoutError::Timeout) => {
                                if keep
                                    && last_issue.is_some_and(|i: Instant| {
                                        i.elapsed() < Duration::from_secs(60)
                                    })
                                    && pool.read_idle()? >= Duration::from_millis(100)
                                {
                                    rng ^= rng << 13;
                                    rng ^= rng >> 7;
                                    rng ^= rng << 17;
                                    let first = source.table_offset() / PAGE as u64;
                                    let end = (source.table_offset() + source.rows() * rb as u64)
                                        / PAGE as u64;
                                    let offset = (first + rng % (end - first).max(1)) * PAGE as u64;
                                    let p = Plan {
                                        uses: zeros(0)?,
                                        jobs: Zeroizing::new(vec![Job {
                                            offset,
                                            length: PAGE,
                                            first: 0,
                                            end: 0,
                                        }]),
                                    };
                                    let mut empty = [];
                                    if pool.execute(&p, &mut empty, &mut cache, cancel).is_err() {
                                        keep = false;
                                    }
                                }
                            }
                        }
                    }
                    Ok(())
                });
                // Cache still owns its full original allocations at inspection.
                let erased = cache.erase();
                erased?;
                result
            })
            .map_err(|_| bad())?;
        let startup = wait.recv().map_err(|_| bad())?;
        let result = match startup {
            Err(e) => Err(e),
            Ok(()) => {
                let lease = Lease(Arc::new(Endpoint {
                    binding,
                    rb,
                    tx,
                    stop: stop.clone(),
                    outstanding: outstanding.clone(),
                }));
                CURRENT.with(|s| *s.borrow_mut() = Some(lease));
                let installed = Installed;
                let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(body));
                drop(installed);
                stop.store(true, Ordering::Release);
                match result {
                    Ok(result) => result,
                    Err(panic) => {
                        let _ = worker.join();
                        std::panic::resume_unwind(panic)
                    }
                }
            }
        };
        stop.store(true, Ordering::Release);
        let joined = worker.join().map_err(|_| bad())?;
        joined?;
        if outstanding.load(Ordering::Acquire) != 0 {
            return Err(bad());
        }
        result
    })
}

#[cfg(all(test, target_os = "linux", target_arch = "x86_64"))]
#[path = "strata_ple_reader_tests.rs"]
mod tests;
