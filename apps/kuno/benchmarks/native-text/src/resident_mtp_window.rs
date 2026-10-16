//! Exclusive queued MTP window. Proposals escape only after their original
//! packet/fault fence: once for zero threshold, per cell for confidence stops.
//! Original scalar APIs retain their completed-forward authority and math.
use super::*;
use crate::resident_model::batch::target_rows::DeviceCarry;
use std::time::Instant;

const ROWS: usize = crate::strata_port::verify_capacity();
pub(super) const PACKET_WORDS: usize = if cfg!(feature = "strata-intel-core") {
    2 * ROWS
} else {
    ROWS
};
const MAX_PROPOSALS: usize = ROWS - 1;
pub(super) const EXTRA_BYTES: u64 = if cfg!(feature = "strata-rust-port") {
    32768
} else {
    0
};
pub(crate) struct DraftWindow {
    tokens: Zeroizing<Vec<u32>>,
    probabilities: Zeroizing<Vec<f32>>,
    pub(crate) start: usize,
    pub(crate) binding: u64,
    pub(crate) last_enqueue_ns: u64,
}
impl DraftWindow {
    pub(crate) fn usable_window(&self, threshold: f32, maximum: usize) -> Result<usize> {
        if self.probabilities.len() != self.tokens.len() {
            return Err(Error::InvalidFormat("MTP window confidence unavailable"));
        }
        confidence_window(&self.probabilities, threshold, maximum)
    }
    pub(crate) fn joint_probability(&self, count: usize) -> Result<f64> {
        if count > self.probabilities.len()
            || self
                .probabilities
                .iter()
                .any(|p| !p.is_finite() || !(0. ..=1.).contains(p))
        {
            return Err(Error::InvalidFormat("MTP joint probability"));
        }
        Ok(self.probabilities[..count]
            .iter()
            .map(|&p| p as f64)
            .product())
    }
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
#[allow(clippy::len_zero)] // Cube Array API exposes len, not is_empty.
#[allow(clippy::manual_range_contains)] // Cube runtime comparisons must expand.
fn save_proposal(
    greedy: &Array<u32>,
    packet: &mut Array<u32>,
    fault: &mut Array<Atomic<u32>>,
    index: usize,
    confidence: &Array<u32>,
    #[comptime] with_probability: bool,
) {
    if UNIT_POS_X == 0 {
        if index >= MAX_PROPOSALS || packet.len() < PACKET_WORDS || greedy.len() < 1 {
            fault[0].fetch_or(8);
        } else {
            let token = greedy[0];
            if token >= k::VOCAB as u32 {
                fault[0].fetch_or(1);
            } else {
                packet[index] = token;
                if with_probability {
                    let bits = confidence[64];
                    let p = f32::reinterpret(bits);
                    if p >= 0.0 && p <= 1.0 {
                        packet[ROWS + index] = bits;
                    } else {
                        fault[0].fetch_or(1);
                    }
                }
            }
        }
    }
}
#[cube(launch)]
fn seal_window(packet: &mut Array<u32>, fault: &Array<Atomic<u32>>) {
    if UNIT_POS_X == 0 {
        packet[MAX_PROPOSALS] = fault[0].load();
    }
}
/// Borrowed actual target predictions/R rows. The controller owns their wiping
/// storage until this round has joined; this is never a fabricated carry/token.
pub(super) enum WindowStart<'a> {
    Input {
        token: u32,
        carry: &'a [f32],
    },
    Device {
        start: usize,
        tokens: &'a [u32],
        source: &'a DeviceCarry<'a>,
    },
    Accepted {
        start: usize,
        tokens: &'a [u32],
        carries: &'a [f32],
    },
}
impl WindowStart<'_> {
    fn position(&self, old: usize, completed: bool) -> Result<usize> {
        match self {
            Self::Input { token, carry } => {
                validate_input(*token, carry, old)?;
                Ok(old)
            }
            Self::Accepted {
                start,
                tokens,
                carries,
            } => {
                accepted_front_position(old, *start, tokens.len(), completed)?;
                if carries.len() != tokens.len() * k::WIDE {
                    return Err(Error::InvalidBufferLength);
                }
                for (row, &token) in tokens.iter().enumerate() {
                    validate_input(
                        token,
                        &carries[row * k::WIDE..(row + 1) * k::WIDE],
                        start.checked_add(row).ok_or(Error::ArithmeticOverflow)?,
                    )?;
                }
                Ok(start + tokens.len() - 1)
            }
            Self::Device {
                start,
                tokens,
                source,
            } => {
                source.validate_predictions(*start, tokens)?;
                device_front_position(old, *start, tokens.len(), completed)
            }
        }
    }
}

/// Native draft_first fronts exactly one verified final-prompt prediction
/// at the untouched teacher cursor. Accepted windows require completed front
/// authority; a missing head result is never fabricated into a full forward.
pub(super) fn device_front_position(
    old: usize,
    start: usize,
    count: usize,
    completed: bool,
) -> Result<usize> {
    if crate::strata_intel_core::enabled() && !completed && count == 1 && old == start {
        k::validate_position(start)?;
        return Ok(start);
    }
    accepted_front_position(old, start, count, completed)
}
pub(super) fn accepted_front_position(
    old: usize,
    start: usize,
    count: usize,
    completed: bool,
) -> Result<usize> {
    if !crate::strata_intel_core::enabled()
        || !completed
        || start == 0
        || !(1..=ROWS).contains(&count)
        || old > k::CONTEXT
        || old < start
        || old - start > MAX_PROPOSALS - 1
    {
        return Err(Error::InvalidFormat("MTP accepted-front boundary"));
    }
    let last = start
        .checked_add(count - 1)
        .ok_or(Error::ArithmeticOverflow)?;
    k::validate_position(last)?;
    Ok(last)
}

impl ResidentMtp<'_, '_> {
    pub(crate) fn draft_window_with_min_probability(
        &mut self,
        token: u32,
        carry: &[f32],
        count: usize,
        minimum: f32,
        cancel: &AtomicBool,
    ) -> Result<DraftWindow> {
        self.window_with_min_probability(
            &WindowStart::Input { token, carry },
            count,
            minimum,
            cancel,
        )
    }

    /// Pinned capture_round(T=a+1), called from the next round with exactly the
    /// accepted target predictions and residuals. Rejected rows never enter it.
    pub(crate) fn draft_accepted_window(
        &mut self,
        start: usize,
        tokens: &[u32],
        carries: &[f32],
        count: usize,
        minimum: f32,
        cancel: &AtomicBool,
    ) -> Result<DraftWindow> {
        self.window_with_min_probability(
            &WindowStart::Accepted {
                start,
                tokens,
                carries,
            },
            count,
            minimum,
            cancel,
        )
    }

    /// Original verifier R rows stay borrowed until the actual MTP packet and
    /// fault fence joins every copy/consumer, including early-stopped chains.
    pub(crate) fn draft_accepted_device_window(
        &mut self,
        start: usize,
        tokens: &[u32],
        source: &DeviceCarry<'_>,
        count: usize,
        minimum: f32,
        cancel: &AtomicBool,
    ) -> Result<DraftWindow> {
        let output = self.window_with_min_probability(
            &WindowStart::Device {
                start,
                tokens,
                source,
            },
            count,
            minimum,
            cancel,
        )?;
        // SAFETY: both confidence paths return ONLY after the original full
        // packet/fault owner read and completed ledger fence on this client.
        unsafe {
            source.completed();
        }
        Ok(output)
    }
    pub(super) fn window_start_position(
        &self,
        source: &WindowStart<'_>,
        count: usize,
    ) -> Result<usize> {
        if !(1..=MAX_PROPOSALS).contains(&count)
            || self.closed
            || self.poisoned
            || self.retired.load(Ordering::Acquire)
            || self.live_outputs.load(Ordering::Acquire) != 0
            || matches!(
                source,
                WindowStart::Accepted { .. } | WindowStart::Device { .. }
            ) && !self.native_confidence_supported()
        {
            return Err(Error::InvalidFormat("MTP window source scope"));
        }
        let completed = self
            .last_forward
            .as_ref()
            .is_some_and(|s| s.valid(self.binding, self.position))
            || matches!(source, WindowStart::Device { .. })
                && self
                    .last_front
                    .as_ref()
                    .is_some_and(|s| s.valid(self.binding, self.position));
        let start = source.position(self.position, completed)?;
        k::validate_position(
            start
                .checked_add(count - 1)
                .ok_or(Error::ArithmeticOverflow)?,
        )?;
        Ok(start)
    }

    pub(super) fn start_window(&mut self, source: &WindowStart<'_>) -> Result<()> {
        match source {
            WindowStart::Input { token, carry } => self.start(*token, carry),
            WindowStart::Device {
                start,
                tokens,
                source,
            } => {
                let last = tokens.len() - 1;
                if self.closed || self.poisoned || self.retired.load(Ordering::Acquire) {
                    return Err(Error::InvalidFormat("MTP resident poisoned"));
                }
                self.q8_head_policy_locked = true;
                self.verify_source()?;
                self.physical()?;
                self.ledger
                    .observe_before(sample(self.client()?)?)
                    .map_err(error)?;
                self.poisoned = true;
                self.dispatches = 0;
                self.clear_scratch()?;
                self.write(B::Token, &[tokens[last]])?;
                source.copy_rows(&self.arena, self.id(B::Carry), self.id(B::Fault), last, 1)?;
                self.tick(1)?;
                self.guard_private(self.position)?;
                self.rewind_cache(self.position, *start)?;
                self.guard_private(*start)?;
                self.position = start + last;
                Ok(())
            }
            WindowStart::Accepted {
                start,
                tokens,
                carries,
            } => {
                // Clear scratch/stage the actual selected token/carry at the OLD
                // complete boundary, then discard only speculative cache cells.
                let last = tokens.len() - 1;
                self.start_at(
                    tokens[last],
                    &carries[last * k::WIDE..(last + 1) * k::WIDE],
                    start + last,
                )?;
                self.rewind_cache(self.position, *start)?;
                self.guard_private(*start)?;
                self.position = start + last;
                Ok(())
            }
        }
    }
    fn prepare_window_first(
        &mut self,
        source: &WindowStart<'_>,
        cancel: &AtomicBool,
    ) -> Result<Option<PreparedNativeFront>> {
        match source {
            WindowStart::Input { .. } => Ok(None),
            WindowStart::Device {
                start,
                tokens,
                source,
            } => self
                .prepare_native_front(
                    tokens.len(),
                    Some(NativeFrontInputs::Device(tokens, source)),
                    *start,
                    cancel,
                )
                .map(Some),
            WindowStart::Accepted {
                start,
                tokens,
                carries,
            } => self
                .prepare_native_front(
                    tokens.len(),
                    Some(NativeFrontInputs::Host(tokens, carries)),
                    *start,
                    cancel,
                )
                .map(Some),
        }
    }
    fn forward_window_first(
        &mut self,
        source: &WindowStart<'_>,
        prepared: Option<PreparedNativeFront>,
        captured: bool,
        cancel: &AtomicBool,
    ) -> Result<()> {
        match source {
            WindowStart::Input { .. } if prepared.is_none() => self.forward(),
            WindowStart::Device { tokens, .. } | WindowStart::Accepted { tokens, .. } => {
                self.enqueue_prepared_native_front(
                    prepared.ok_or(Error::InvalidFormat("MTP first front not prepared"))?,
                    cancel,
                )?;
                if captured {
                    self.client()?
                        .set_native_strata_phase(cubecl_runtime::server::NativeStrataPhase::Draft {
                            index: 0,
                        })
                        .map_err(error)?;
                }
                self.native_selected_rest(tokens.len())
            }
            _ => Err(Error::InvalidFormat("MTP first front preparation source")),
        }
    }
    fn window_with_min_probability(
        &mut self,
        source: &WindowStart<'_>,
        count: usize,
        minimum: f32,
        cancel: &AtomicBool,
    ) -> Result<DraftWindow> {
        if !minimum.is_finite()
            || !(0. ..=1.).contains(&minimum)
            || !self.native_confidence_supported()
        {
            return Err(Error::InvalidFormat("MTP confidence-window scope"));
        }
        self.queued_window_with_minimum(source, count, minimum, cancel)
    }

    pub(crate) fn draft_window(
        &mut self,
        token: u32,
        carry: &[f32],
        count: usize,
        cancel: &AtomicBool,
    ) -> Result<DraftWindow> {
        self.queued_window_from(&WindowStart::Input { token, carry }, count, cancel)
    }
    fn queued_window_from(
        &mut self,
        source: &WindowStart<'_>,
        count: usize,
        cancel: &AtomicBool,
    ) -> Result<DraftWindow> {
        self.queued_window_with_minimum(source, count, 0., cancel)
    }
    fn queued_window_with_minimum(
        &mut self,
        source: &WindowStart<'_>,
        count: usize,
        minimum: f32,
        cancel: &AtomicBool,
    ) -> Result<DraftWindow> {
        // The native zero-threshold path joins the full chain once. Positive
        // thresholds observe ONLY the token/probability/fault packet after
        // each cell, never the10240-value hidden vector. Original carries stay
        // on GPU and feed the next cell through copy_committed in both modes.
        if !minimum.is_finite()
            || !(0. ..=1.).contains(&minimum)
            || minimum > 0. && !self.native_confidence_supported()
        {
            return Err(Error::InvalidFormat("MTP queued confidence scope"));
        }
        u::cancel(cancel)?;
        if !crate::strata_port::enabled()
            || !self.fused_sweeps
            || !(1..=MAX_PROPOSALS).contains(&count)
            || self.closed
            || self.poisoned
            || self.live_outputs.load(Ordering::Acquire) != 0
        {
            return Err(Error::InvalidFormat("MTP queued window scope"));
        }
        let start = self.window_start_position(source, count)?;
        let id = self.draft_window.ok_or(Error::InvalidBufferLength)?;
        let lease = OutputLease::acquire(&self.live_outputs)?;
        self.last_forward.take();
        self.last_front.take();
        // At most the compiled number of bounded per-cell INFO sets may remain queued together.
        // Credit is retired only at the final completed fence, never on enqueue.
        self.ledger
            .precharge(&[count as u64 * INFO_BYTES])
            .map_err(error)?;
        let result =
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| -> Result<DraftWindow> {
                self.start_window(source)?;
                u::zero(&self.arena, id)?;
                self.tick(1)?;
                // Original TOKEN upload and borrowed carry copy happen before
                // capture. The outer source grant stays live through the same
                // actual final packet/fault observation as before.
                let mut first_front = self.prepare_window_first(source, cancel)?;
                let mut last_enqueue_ns = 0;
                let mut observation = None;
                let coupled = crate::strata_sampler::request::coupled_draft_active(
                    self.client()?,
                    self.binding,
                )?;
                // The connected Intel controller owns stable target device rows.
                // Host/scalar entry points retain their original unkeyed path.
                let branch_rows = if cfg!(feature = "strata-intel-core") {
                    match source {
                        WindowStart::Device { tokens, .. } => {
                            Some(u8::try_from(tokens.len()).map_err(error)?)
                        }
                        _ => None,
                    }
                } else {
                    None
                };
                let front_origin = match source {
                    WindowStart::Device { start, .. } => *start,
                    _ => 0, // Unkeyed legacy paths never consume this origin.
                };
                let mut whole_branch = branch_rows
                    .filter(|_| minimum == 0.)
                    .map(|rows| {
                        cubecl_runtime::server::NativeStrataBranch::mtp_round(rows, count as u8)
                            .and_then(|key| key.with_model_owner(self.binding))
                            .map(|key| key.with_coupled_sampling(coupled))
                            .map_err(error)
                    })
                    .transpose()?;
                let mut whole_capture = None;
                if let Some(key) = whole_branch {
                    if self.try_native_frame_replay(key, front_origin, &mut first_front, cancel)? {
                        let WindowPayload {
                            tokens,
                            probabilities,
                        } = self.read_window_payload(id, count)?;
                        return Ok(DraftWindow {
                            tokens,
                            probabilities,
                            start,
                            binding: self.binding,
                            last_enqueue_ns: 0,
                        });
                    }
                    self.client()?
                        .begin_native_strata_invocation(
                            key,
                            cubecl_runtime::server::NativeStrataInvocation::mtp(
                                u32::try_from(front_origin).map_err(error)?,
                            )
                            .map_err(error)?,
                        )
                        .map_err(error)?;
                    whole_capture = Some(replay::Capture::begin(
                        key,
                        self.traffic.get(),
                        self.dispatches,
                    )?);
                }
                for member in 0..count {
                    let step_branch = branch_rows
                        .filter(|_| minimum > 0.)
                        .map(|rows| {
                            if member == 0 {
                                cubecl_runtime::server::NativeStrataBranch::mtp_round(rows, 1)
                            } else {
                                cubecl_runtime::server::NativeStrataBranch::mtp_step(member as u8)
                            }
                            .and_then(|key| key.with_model_owner(self.binding))
                            .map(|key| key.with_coupled_sampling(coupled))
                            .map_err(error)
                        })
                        .transpose()?;
                    let mut step_capture = None;
                    let step_replayed = if let Some(key) = step_branch {
                        self.try_native_frame_replay(
                            key,
                            if member == 0 {
                                front_origin
                            } else {
                                self.position
                            },
                            &mut first_front,
                            cancel,
                        )?
                    } else {
                        false
                    };
                    if let Some(key) = step_branch.filter(|_| !step_replayed) {
                        self.client()?
                            .begin_native_strata_invocation(
                                key,
                                cubecl_runtime::server::NativeStrataInvocation::mtp(
                                    u32::try_from(if member == 0 {
                                        front_origin
                                    } else {
                                        self.position
                                    })
                                    .map_err(error)?,
                                )
                                .map_err(error)?,
                            )
                            .map_err(error)?;
                        if member != 0 {
                            self.client()?
                                .set_native_strata_phase(
                                    cubecl_runtime::server::NativeStrataPhase::Draft {
                                        index: member as u8,
                                    },
                                )
                                .map_err(error)?;
                        }
                        step_capture = Some(replay::Capture::begin(
                            key,
                            self.traffic.get(),
                            self.dispatches,
                        )?);
                    }
                    let clock = Instant::now();
                    if !step_replayed {
                        if member == 0 {
                            self.forward_window_first(
                                source,
                                first_front.take(),
                                step_branch.is_some() || whole_branch.is_some(),
                                cancel,
                            )?;
                        } else {
                            self.forward()?;
                        }
                        self.finish_gpu()?;
                        save_proposal::launch::<WgpuRuntime>(
                            self.client()?,
                            CubeCount::Static(1, 1, 1),
                            CubeDim::new_1d(64),
                            self.a(B::Greedy)?,
                            u::arg(&self.arena, id)?,
                            self.a(B::Fault)?,
                            member,
                            self.a(B::Candidates)?,
                            self.native_confidence_supported(),
                        );
                        self.tick(1)?;
                        last_enqueue_ns = u64::try_from(clock.elapsed().as_nanos())
                            .map_err(|_| Error::ArithmeticOverflow)?;
                    } else {
                        last_enqueue_ns = 0;
                    }
                    u::cancel(cancel)?;
                    if minimum > 0. || member + 1 == count {
                        let mut captured = None;
                        if !step_replayed {
                            self.seal_window(id)?;
                            // Seal/join only the original producer graph. Actual
                            // private readback and confidence decisions are outside
                            // it, preserving each positive-threshold stop boundary.
                            if let Some(key) = step_branch.or_else(|| whole_branch.take()) {
                                self.client()?
                                    .end_native_strata_branch(key)
                                    .map_err(error)?;
                                captured = Some(
                                    step_capture
                                        .take()
                                        .or_else(|| whole_capture.take())
                                        .ok_or(Error::InvalidFormat(
                                            "MTP capture accounting missing",
                                        ))?
                                        .finish(self.traffic.get(), self.dispatches)?,
                                );
                            }
                        }
                        let payload = self.read_window_payload(id, member + 1)?;
                        if let Some(entry) = captured {
                            self.native_replay.insert(entry)?;
                        }
                        let stop = minimum > 0. && payload.probabilities[member] < minimum;
                        observation = Some(payload);
                        if stop {
                            break;
                        }
                    }
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
                        if member == 0
                            && let Some(capture) = whole_capture.as_mut()
                        {
                            capture.first_reset(self.dispatches)?;
                        }
                        self.position += 1;
                        self.dispatches = 0;
                        if whole_branch.is_some() {
                            self.client()?
                                .set_native_strata_phase(
                                    cubecl_runtime::server::NativeStrataPhase::Draft {
                                        index: (member + 1) as u8,
                                    },
                                )
                                .map_err(error)?;
                        }
                        self.clear_scratch_preserving_inputs(true)?;
                        self.guard_private_at_native(0)?;
                    }
                }
                if whole_branch.is_some() || first_front.is_some() {
                    return Err(Error::InvalidFormat("MTP unfinished native window capture"));
                }
                let WindowPayload {
                    tokens,
                    probabilities,
                } = observation.ok_or(Error::InvalidFormat("MTP window observation absent"))?;
                Ok(DraftWindow {
                    tokens,
                    probabilities,
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
                if let Err(e) = self.commit_forward() {
                    drop(output);
                    self.fail();
                    return Err(e);
                }
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
    /// Queue the original packet's sticky-fault seal after every producer.
    /// This is not an observation, output or completed-forward receipt.
    fn seal_window(&mut self, id: BufferId) -> Result<()> {
        seal_window::launch::<WgpuRuntime>(
            self.client()?,
            CubeCount::Static(1, 1, 1),
            CubeDim::new_1d(64),
            u::arg(&self.arena, id)?,
            self.a(B::Fault)?,
        );
        self.tick(1)
    }
    /// Actual readback follows branch closure. Full packet/padding, fault,
    /// source identity and the original ledger fence still gate observation.
    fn read_window_payload(&mut self, id: BufferId, count: usize) -> Result<WindowPayload> {
        if self.native_front_pending.load(Ordering::Acquire) {
            return Err(Error::InvalidFormat(
                "MTP observation has unconsumed front preparation",
            ));
        }
        let packet = self.read_words_id(id)?;
        let payload = decode_window(&packet, count, self.native_confidence_supported())?;
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
        Ok(payload)
    }
}

/// Native generator keeps ONLY drafts meeting the threshold. The first low
/// draft is returned by MTP for observation, but is not fed to the target.
pub(crate) fn confidence_window(
    probabilities: &[f32],
    threshold: f32,
    maximum: usize,
) -> Result<usize> {
    if !threshold.is_finite()
        || !(0. ..=1.).contains(&threshold)
        || !(1..=crate::strata_port::verify_capacity()).contains(&maximum)
        || probabilities.len() > MAX_PROPOSALS
        || probabilities
            .iter()
            .any(|p| !p.is_finite() || !(0. ..=1.).contains(p))
    {
        return Err(Error::InvalidFormat("MTP confidence prefix"));
    }
    Ok(1 + probabilities
        .iter()
        .take(maximum - 1)
        .take_while(|&&p| p >= threshold)
        .count())
}
struct WindowPayload {
    tokens: Zeroizing<Vec<u32>>,
    probabilities: Zeroizing<Vec<f32>>,
}
fn decode_window(packet: &[u32], count: usize, with_probability: bool) -> Result<WindowPayload> {
    let owned = BufferLayout::new(PACKET_WORDS).map_err(error)?.owned_words;
    if !(1..=MAX_PROPOSALS).contains(&count)
        || packet.len() != owned
        || packet[MAX_PROPOSALS] != 0
        || packet[..count].iter().any(|&id| id >= k::VOCAB as u32)
        || packet[count..MAX_PROPOSALS].iter().any(|&v| v != 0)
        || packet[if with_probability { ROWS + count } else { ROWS }..]
            .iter()
            .any(|&v| v != 0)
    {
        return Err(Error::InvalidFormat("MTP owning draft-window packet"));
    }
    let mut probabilities = Zeroizing::new(Vec::with_capacity(if with_probability {
        MAX_PROPOSALS
    } else {
        0
    }));
    if with_probability {
        for &bits in &packet[ROWS..ROWS + count] {
            let p = f32::from_bits(bits);
            if !p.is_finite() || !(0. ..=1.).contains(&p) {
                return Err(Error::InvalidFormat("MTP owning draft-window probability"));
            }
            probabilities.push(p);
        }
    }
    Ok(WindowPayload {
        tokens: Zeroizing::new(packet[..count].to_vec()),
        probabilities,
    })
}

#[cfg(test)]
mod confidence_tests {
    use super::*;
    #[test]
    fn native_grouped_round_covers_every_accepted_prefix_and_short_draft_chain() {
        if !crate::strata_intel_core::enabled() {
            return;
        }
        for n in 1..=ROWS {
            for start in [1, 2, 63, 2048, k::CONTEXT - n] {
                let tokens: Vec<_> = (0..n as u32).map(|i| i + 10).collect();
                let carries: Vec<_> = (0..n * k::WIDE).map(|i| (i / k::WIDE) as f32).collect();
                let source = WindowStart::Accepted {
                    start,
                    tokens: &tokens,
                    carries: &carries,
                };
                // A suffix-selected accepted window may be LONGER than the
                // actual confidence-stopped chain. Only old KV below start is kept.
                for produced in 1..=MAX_PROPOSALS {
                    let old = start - 1 + produced;
                    if old <= k::CONTEXT {
                        assert_eq!(source.position(old, true).unwrap(), start + n - 1);
                        assert!(source.position(old, false).is_err());
                    }
                }
                assert!(source.position(start - 1, true).is_err());
                assert!(source.position(start + MAX_PROPOSALS, true).is_err());
                // Native front row r consumes PREDICTION[r] and R[r] at cell
                // start+r, including the new confirmed last prediction. It is
                // not the prior accepted input/R[r-1] pair used by old repair.
                for row in 0..n {
                    validate_input(
                        tokens[row],
                        &carries[row * k::WIDE..(row + 1) * k::WIDE],
                        start + row,
                    )
                    .unwrap();
                    assert_eq!(carries[row * k::WIDE], row as f32);
                }
            }
        }
    }
    #[test]
    fn native_grouped_round_rejects_invalid_sources_before_consuming_follow_stamp() {
        let tokens = [1, 2];
        let mut carries = vec![0.; 2 * k::WIDE];
        let check = |start, ts: &[u32], rs: &[f32], old| {
            WindowStart::Accepted {
                start,
                tokens: ts,
                carries: rs,
            }
            .position(old, true)
        };
        if !crate::strata_intel_core::enabled() {
            assert!(check(1, &tokens, &carries, 1).is_err());
            return;
        }
        assert_eq!(check(1, &tokens, &carries, 1).unwrap(), 2);
        assert!(check(0, &tokens, &carries, 1).is_err());
        assert!(check(1, &[], &[], 1).is_err());
        assert!(check(1, &tokens, &carries[..k::WIDE], 1).is_err());
        assert!(check(1, &[k::VOCAB as u32, 2], &carries, 1).is_err());
        assert!(check(k::CONTEXT - 1, &tokens, &carries, k::CONTEXT).is_err());
        assert!(check(k::CONTEXT - 1, &[1], &carries[..k::WIDE], k::CONTEXT + 1).is_err());
        carries[k::WIDE] = f32::NAN;
        assert!(check(1, &tokens, &carries, 1).is_err());
        carries[k::WIDE] = f32::INFINITY;
        assert!(check(1, &tokens, &carries, 1).is_err());
        assert!(
            WindowStart::Input {
                token: 1,
                carry: &[0.; k::WIDE]
            }
            .position(1, false)
            .is_ok()
        );
    }
    #[test]
    fn native_confidence_prefix_excludes_first_low_draft_and_keeps_equality() {
        for maximum in 1..=crate::strata_port::verify_capacity() {
            let p = vec![0.5; MAX_PROPOSALS];
            assert_eq!(confidence_window(&p, 0., maximum).unwrap(), maximum);
            assert_eq!(confidence_window(&p, 0.5, maximum).unwrap(), maximum);
            assert_eq!(confidence_window(&p, 0.51, maximum).unwrap(), 1);
            for bad in 0..maximum - 1 {
                let mut p = p.clone();
                p[bad] = 0.;
                assert_eq!(confidence_window(&p, 0.2, maximum).unwrap(), bad + 1);
            }
        }
        assert_eq!(confidence_window(&[], 0.3, 1).unwrap(), 1);
        for bad in [f32::NAN, f32::INFINITY, -0.1, 1.1] {
            assert!(confidence_window(&[bad], 0., 2).is_err());
            assert!(confidence_window(&[1.], bad, 2).is_err());
        }
    }
    #[test]
    fn native_confidence_window_decoder_checks_tokens_probabilities_and_full_padding() {
        if !crate::strata_intel_core::enabled() {
            return;
        }
        let owned = BufferLayout::new(PACKET_WORDS).unwrap().owned_words;
        for count in 1..=MAX_PROPOSALS {
            let mut packet = vec![0; owned];
            for i in 0..count {
                packet[i] = i as u32;
                packet[ROWS + i] = (i as f32 / (count as f32)).to_bits();
            }
            let WindowPayload {
                tokens: t,
                probabilities: p,
            } = decode_window(&packet, count, true).unwrap();
            assert_eq!(&*t, &(0..count as u32).collect::<Vec<_>>());
            assert_eq!(p.len(), count);
            for i in [MAX_PROPOSALS, ROWS + MAX_PROPOSALS, owned - 1] {
                let mut bad = packet.clone();
                bad[i] = 1;
                assert!(decode_window(&bad, count, true).is_err());
            }
            for bits in [
                f32::NAN.to_bits(),
                f32::INFINITY.to_bits(),
                (-0.1f32).to_bits(),
                1.1f32.to_bits(),
            ] {
                let mut bad = packet.clone();
                bad[ROWS] = bits;
                assert!(decode_window(&bad, count, true).is_err());
            }
        }
    }
}

#[cfg(test)]
mod confidence_kernel_tests {
    use super::*;
    #[test]
    fn native_confidence_window_packet_kernels_lower_in_both_modes() {
        use cubecl::{Compiler, WgpuCompilationOptions, wgpu::WgslCompiler};
        for with_probability in [false, true] {
            for mode in [ExecutionMode::Checked, ExecutionMode::Unchecked] {
                let mut b = KernelBuilder::default();
                b.runtime_properties(WgpuRuntime::target_properties());
                AddressType::U32.register(&mut b.scope);
                let u = u32::as_type(&b.scope);
                let a = Atomic::<u32>::as_type(&b.scope);
                let token = b.input_array(u).into();
                let packet = b.output_array(u).into();
                let fault = b.output_array(a).into();
                let confidence = b.input_array(u).into();
                let index = b.scalar(u.storage_type()).into();
                save_proposal::expand(
                    &mut b.scope,
                    token,
                    packet,
                    fault,
                    index,
                    confidence,
                    with_probability,
                );
                let source = WgslCompiler::default()
                    .compile(
                        b.build(
                            KernelSettings::default()
                                .cube_dim(CubeDim::new_1d(64))
                                .kernel_name("native_mtp_confidence_window"),
                        ),
                        &WgpuCompilationOptions::default(),
                        mode,
                        AddressType::U32.unsigned_type(),
                    )
                    .unwrap()
                    .to_string();
                let module = wgpu::naga::front::wgsl::parse_str(&source)
                    .unwrap_or_else(|e| panic!("{}", e.emit_to_string(&source)));
                wgpu::naga::valid::Validator::new(
                    wgpu::naga::valid::ValidationFlags::all(),
                    wgpu::naga::valid::Capabilities::all(),
                )
                .validate(&module)
                .unwrap_or_else(|e| panic!("{}", e.emit_to_string(&source)));
            }
        }
    }
}
