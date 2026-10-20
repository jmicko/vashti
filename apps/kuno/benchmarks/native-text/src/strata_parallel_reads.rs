//! Bounded scoped readers of disjoint, already-initialized PUBLIC model spans.
//! No GPU/client operations, raw pointers, path reopening or detached workers.
use std::panic::resume_unwind;

pub(super) const WORKERS: usize = 8;
const STACK_BYTES: usize = 256 * 1024;
// Requested stacks plus bounded per-worker bookkeeping/TLS allowance. Job
// vectors remain inside the original separately charged global metadata cap.
pub(super) const SETUP_BYTES: u64 = (WORKERS * (STACK_BYTES + 64 * 1024)) as u64;

#[derive(Debug)]
pub(super) enum Failure<E> {
    Geometry,
    Spawn,
    Read(E),
}

pub(super) fn fill<E: Send>(
    target: &mut [u8],
    item_bytes: usize,
    read: impl Fn(usize, &mut [u8]) -> Result<(), E> + Sync,
) -> Result<(), Failure<E>> {
    if item_bytes == 0 || target.is_empty() || !target.len().is_multiple_of(item_bytes) {
        return Err(Failure::Geometry);
    }
    let count = target.len() / item_bytes;
    let per_worker = count.div_ceil(WORKERS);
    let block_bytes = per_worker
        .checked_mul(item_bytes)
        .ok_or(Failure::Geometry)?;
    std::thread::scope(|scope| {
        let mut workers = Vec::with_capacity(WORKERS);
        let mut failure = None;
        let mut panic = None;
        for (block, bytes) in target.chunks_mut(block_bytes).enumerate() {
            let first = block * per_worker;
            let read = &read;
            match std::thread::Builder::new()
                .stack_size(STACK_BYTES)
                .spawn_scoped(scope, move || {
                    for (local, span) in bytes.chunks_mut(item_bytes).enumerate() {
                        read(first + local, span)?;
                    }
                    Ok(())
                }) {
                Ok(worker) => workers.push(worker),
                Err(_) => {
                    failure = Some(Failure::Spawn);
                    break;
                }
            }
        }
        // No early `?`: every successfully created worker must join before a
        // mapped span can flush/unmap/seal/register or a caller unwinds.
        for worker in workers {
            match worker.join() {
                Ok(Ok(())) => (),
                Ok(Err(error)) if failure.is_none() => failure = Some(Failure::Read(error)),
                Err(payload) if panic.is_none() => panic = Some(payload),
                _ => (),
            }
        }
        if let Some(payload) = panic {
            resume_unwind(payload);
        }
        failure.map_or(Ok(()), Err)
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::panic::{AssertUnwindSafe, catch_unwind};
    use std::sync::atomic::{AtomicUsize, Ordering};

    #[test]
    fn exact_disjoint_cells_and_initialized_padding_survive_all_short_tails() {
        for count in [1usize, 2, 7, 8, 9, 31, 63, 64, 65, 127, 128, 129, 511, 512] {
            let mut bytes = vec![0; count * 13 + 29];
            let visited = AtomicUsize::new(0);
            fill(&mut bytes[..count * 13], 13, |index, output| {
                output.fill((index % 251 + 1) as u8);
                visited.fetch_add(1, Ordering::Relaxed);
                Ok::<_, ()>(())
            })
            .unwrap();
            assert_eq!(visited.load(Ordering::Relaxed), count);
            for (index, cell) in bytes[..count * 13].chunks_exact(13).enumerate() {
                assert!(cell.iter().all(|&value| value == (index % 251 + 1) as u8));
            }
            assert!(bytes[count * 13..].iter().all(|&value| value == 0));
        }
    }

    #[test]
    fn rejects_invalid_geometry_before_any_reader_runs() {
        for (bytes, item) in [(0, 1), (8, 0), (8, 3)] {
            let mut output = vec![0; bytes];
            assert!(matches!(
                fill(&mut output, item, |_, _| -> Result<(), ()> {
                    panic!("reader must not run")
                }),
                Err(Failure::Geometry)
            ));
        }
        assert_eq!(SETUP_BYTES, 2_621_440);
    }

    #[test]
    fn source_failure_joins_every_other_scoped_worker_before_return() {
        let completed = AtomicUsize::new(0);
        let mut output = [0; 64];
        let result = fill(&mut output, 1, |index, target| {
            if index == 0 {
                return Err(73);
            }
            target[0] = 1;
            completed.fetch_add(1, Ordering::Relaxed);
            Ok(())
        });
        assert!(matches!(result, Err(Failure::Read(73))));
        assert_eq!(completed.load(Ordering::Relaxed), 56);
        assert!(output[8..].iter().all(|&value| value == 1));
    }

    #[test]
    fn worker_panic_joins_every_other_worker_before_mapped_owner_unwinds() {
        let completed = AtomicUsize::new(0);
        let mut output = [0; 64];
        assert!(
            catch_unwind(AssertUnwindSafe(|| {
                fill(&mut output, 1, |index, target| -> Result<(), ()> {
                    assert_ne!(index, 0, "fixed public worker fault");
                    target[0] = 1;
                    completed.fetch_add(1, Ordering::Relaxed);
                    Ok(())
                })
            }))
            .is_err()
        );
        assert_eq!(completed.load(Ordering::Relaxed), 56);
        assert!(output[8..].iter().all(|&value| value == 1));
    }
}
