//! Exclusive queued MTP window. No provisional state or token escapes until a
//! single final read/fault fence completes every cell. Original scalar APIs
//! retain their completed-forward authority and numerical paths.
use super::*;
use std::time::Instant;

pub(super) const EXTRA_BYTES: u64 = if cfg!(feature = "strata-rust-port") {
    32768
} else {
    0
};
pub(crate) struct DraftWindow {
    tokens: Zeroizing<Vec<u32>>,
    pub(crate) start: usize,
    pub(crate) binding: u64,
    pub(crate) last_enqueue_ns: u64,
}
impl DraftWindow {
    pub(crate) fn tokens(&self) -> &[u32] {
        &self.tokens
    }
}
impl Drop for DraftWindow {
    fn drop(&mut self) {
        self.start.zeroize();
        self.binding.zeroize();
        self.last_enqueue_ns.zeroize();
    }
}
#[cube(launch)]
fn save_proposal(
    greedy: &Array<u32>,
    packet: &mut Array<u32>,
    fault: &mut Array<Atomic<u32>>,
    index: usize,
) {
    if UNIT_POS_X == 0 {
        if index >= 3 || packet.len() < 4 || greedy.len() < 1 {
            fault[0].fetch_or(8);
        } else {
            let token = greedy[0];
            if token >= k::VOCAB as u32 {
                fault[0].fetch_or(1);
            } else {
                packet[index] = token;
            }
        }
    }
}
#[cube(launch)]
fn seal_window(packet: &mut Array<u32>, fault: &Array<Atomic<u32>>) {
    if UNIT_POS_X == 0 {
        packet[3] = fault[0].load();
    }
}
impl ResidentMtp<'_, '_> {
    pub(crate) fn draft_window(
        &mut self,
        token: u32,
        carry: &[f32],
        count: usize,
        cancel: &AtomicBool,
    ) -> Result<DraftWindow> {
        u::cancel(cancel)?;
        if !crate::strata_port::enabled()
            || !self.fused_sweeps
            || !(1..=3).contains(&count)
            || self.closed
            || self.poisoned
            || self.live_outputs.load(Ordering::Acquire) != 0
        {
            return Err(Error::InvalidFormat("MTP queued window scope"));
        }
        validate_input(token, carry, self.position)?;
        k::validate_position(
            self.position
                .checked_add(count - 1)
                .ok_or(Error::ArithmeticOverflow)?,
        )?;
        let id = self.draft_window.ok_or(Error::InvalidBufferLength)?;
        let start = self.position;
        let lease = OutputLease::acquire(&self.live_outputs)?;
        self.last_forward.take();
        // At most three bounded per-cell INFO sets may remain queued together.
        // Credit is retired only at the final completed fence, never on enqueue.
        self.ledger
            .precharge(&[count as u64 * INFO_BYTES])
            .map_err(error)?;
        let result =
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| -> Result<DraftWindow> {
                self.start(token, carry)?;
                u::zero(&self.arena, id)?;
                self.tick(1)?;
                let mut last_enqueue_ns = 0;
                for member in 0..count {
                    let clock = Instant::now();
                    self.forward()?;
                    self.finish_gpu()?;
                    save_proposal::launch::<WgpuRuntime>(
                        self.client()?,
                        CubeCount::Static(1, 1, 1),
                        CubeDim::new_1d(64),
                        self.a(B::Greedy)?,
                        u::arg(&self.arena, id)?,
                        self.a(B::Fault)?,
                        member,
                    );
                    self.tick(1)?;
                    last_enqueue_ns = u64::try_from(clock.elapsed().as_nanos())
                        .map_err(|_| Error::ArithmeticOverflow)?;
                    u::cancel(cancel)?;
                    if member + 1 < count {
                        // This copy consumes this window's preceding queued producer
                        // on the same stream. It is not a completed LastForward stamp.
                        copy_committed::launch::<WgpuRuntime>(
                            self.client()?,
                            grid(k::WIDE)?,
                            CubeDim::new_1d(64),
                            self.full(B::Residual2)?,
                            self.full(B::Greedy)?,
                            self.full(B::Carry)?,
                            self.full(B::Token)?,
                            self.full(B::Fault)?,
                        );
                        self.tick(1)?;
                        self.count_traffic(|t| {
                            t.followup_copy_calls = t
                                .followup_copy_calls
                                .checked_add(1)
                                .ok_or(Error::ArithmeticOverflow)?;
                            Ok(())
                        })?;
                        self.position += 1;
                        self.dispatches = 0;
                        self.clear_scratch_preserving_inputs(true)?;
                        self.guard_private(self.position)?;
                    }
                }
                seal_window::launch::<WgpuRuntime>(
                    self.client()?,
                    CubeCount::Static(1, 1, 1),
                    CubeDim::new_1d(64),
                    u::arg(&self.arena, id)?,
                    self.a(B::Fault)?,
                );
                self.tick(1)?;
                let packet = self.read_words_id(id)?;
                if packet[3] != 0
                    || packet[..count].iter().any(|&id| id >= k::VOCAB as u32)
                    || packet[count..3].iter().any(|&id| id != 0)
                {
                    return Err(Error::InvalidFormat(
                        "MTP completed queued window fault or token",
                    ));
                }
                self.verify_source()?;
                self.ledger
                    .completed_fence(sample(self.client()?)?)
                    .map_err(error)?;
                self.count_traffic(|t| {
                    t.draft_window_reads = t
                        .draft_window_reads
                        .checked_add(1)
                        .ok_or(Error::ArithmeticOverflow)?;
                    Ok(())
                })?;
                let tokens = Zeroizing::new(packet[..count].to_vec());
                Ok(DraftWindow {
                    tokens,
                    start,
                    binding: self.binding,
                    last_enqueue_ns,
                })
            }));
        drop(lease);
        match result {
            Ok(Ok(output)) => {
                if let Err(e) = u::cancel(cancel) {
                    drop(output);
                    self.fail();
                    return Err(e);
                }
                // All cells, full private checks, greedy and sticky fault have
                // completed. The only published follow stamp is the final cell.
                self.commit_forward();
                Ok(output)
            }
            Ok(Err(e)) => {
                self.fail();
                Err(e)
            }
            Err(_) => {
                self.fail();
                Err(Error::InvalidFormat("MTP queued window failure"))
            }
        }
    }
}
