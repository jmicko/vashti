//! Request-scoped double-buffered prompt PLE source reads. Derived from MIT
//! Strata fb58e0 prefill.cpp's PLE prefetch: the next chunk is reconstructed
//! while the current chunk is consumed. No GPU client enters the worker.
use super::*;
use std::{cell::RefCell, sync::mpsc};

// At4096 rows, current and prefetched40MiB embeddings, runtime upload copy,
// <=6.88MiB original packed rows, wiping read plans and finite worker stacks
// coexist. Reserve128MiB; target embedding/other model copies are separately
// charged in resident_model_prompt. No stack-sized4096-row metadata arrays.
pub(crate) const HOST_BYTES: usize = 128 * 1024 * 1024;
struct Prepared {
    binding: GraphBindingId,
    start: usize,
    count: usize,
    rows: Zeroizing<Vec<[u32; 16]>>,
    data: Zeroizing<Vec<u8>>,
}
impl Drop for Prepared {
    fn drop(&mut self) {
        self.start.zeroize();
        self.count.zeroize();
    }
}
thread_local! {static INPUT:RefCell<Option<mpsc::Receiver<Result<Prepared>>>>=const{RefCell::new(None)};}
struct InputGuard;
impl InputGuard {
    fn new(rx: mpsc::Receiver<Result<Prepared>>) -> Result<Self> {
        INPUT.with(|s| {
            let mut s = s.borrow_mut();
            if s.is_some() {
                return Err(error());
            }
            *s = Some(rx);
            Ok(Self)
        })
    }
}
impl Drop for InputGuard {
    fn drop(&mut self) {
        // Disconnect before joining: a producer waiting on the bounded channel
        // cannot outlive the request or deadlock a failed consumer.
        INPUT.with(|s| *s.borrow_mut() = None);
    }
}
pub(super) fn take(
    graph: BoundGraph<'_, '_>,
    rows: &[[u32; 16]],
    start: usize,
) -> Result<Option<Zeroizing<Vec<u8>>>> {
    if rows.is_empty() || rows.len() > crate::strata_prompt::groups::CAPACITY {
        return Err(error());
    }
    INPUT.with(|s| {
        let s = s.borrow();
        let Some(rx) = s.as_ref() else {
            return Ok(None);
        };
        let mut prepared = rx.recv().map_err(|_| error())??;
        if prepared.binding != graph.binding_id()
            || prepared.start != start
            || prepared.count != rows.len()
            || prepared.rows[..rows.len()] != *rows
            || prepared.rows[rows.len()..]
                .iter()
                .flatten()
                .any(|&v| v != 0)
        {
            return Err(error());
        }
        graph.verify_unchanged()?;
        Ok(Some(std::mem::take(&mut prepared.data)))
    })
}
pub(crate) fn run<T>(
    graph: BoundGraph<'_, '_>,
    prompt: &[u32],
    chunk: usize,
    old: Zeroizing<[i32; 2]>,
    profile: OriginalPackedProfile,
    cancel: &AtomicBool,
    body: impl FnOnce() -> Result<T>,
) -> Result<T> {
    if !crate::strata_intel_core::enabled()
        || prompt.is_empty()
        || prompt.len() > 4096
        || !(1..=crate::strata_prompt::groups::CAPACITY).contains(&chunk)
        || prompt.iter().any(|&t| t >= 248320)
    {
        return Err(error());
    }
    graph_preflight(graph)?;
    // Rendezvous rather than a queued ready slot: exactly the current upload
    // and one next embedding may exist. The producer cannot prepare a third
    // chunk before the current chunk has completed and landed its successor.
    let (tx, rx) = mpsc::sync_channel(0);
    let stopped = AtomicBool::new(false);
    std::thread::scope(|scope| -> Result<T> {
        let stopped = &stopped;
        let job = std::thread::Builder::new()
            .stack_size(256 * 1024)
            .spawn_scoped(scope, move || {
                let mut previous = old;
                for start in (0..prompt.len()).step_by(chunk) {
                    if stopped.load(Ordering::Acquire) || cancel.load(Ordering::Acquire) {
                        return;
                    }
                    let count = (prompt.len() - start).min(chunk);
                    let result = (|| -> Result<Prepared> {
                        let mut rows =
                            Zeroizing::new(vec![[0; 16]; crate::strata_prompt::groups::CAPACITY]);
                        for t in 0..count {
                            crate::ple_chunk::rows_into(
                                graph.ple_hash().metadata(),
                                prompt[start + t],
                                &previous,
                                &mut rows[t],
                            )
                            .map_err(u::fault)?;
                            previous[0] = previous[1];
                            previous[1] = prompt[start + t] as i32;
                        }
                        let table = graph.tensor("per_layer_token_embd.weight")?;
                        profile.validate_ple(
                            table.info().dtype(),
                            table.info().dimensions(),
                            table.info().row_bytes(),
                        )?;
                        let data = native_embedding(graph, table, profile, &rows[..count], cancel)?;
                        Ok(Prepared {
                            binding: graph.binding_id(),
                            start,
                            count,
                            rows,
                            data,
                        })
                    })();
                    let failed = result.is_err();
                    if tx.send(result).is_err() || failed {
                        return;
                    }
                }
            })
            .map_err(|_| error())?;
        let input = InputGuard::new(rx)?;
        let result = body();
        stopped.store(true, Ordering::Release);
        drop(input);
        let joined = job.join();
        if joined.is_err() {
            return Err(error());
        }
        result
    })
}
