//! Exclusive queued MTP window. No provisional state or token escapes until a
//! single final read/fault fence completes every cell. Original scalar APIs
//! retain their completed-forward authority and numerical paths.
use super::*;
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
impl ResidentMtp<'_, '_> {
    pub(crate) fn draft_window_with_min_probability(
        &mut self,
        token: u32,
        carry: &[f32],
        count: usize,
        minimum: f32,
        cancel: &AtomicBool,
    ) -> Result<DraftWindow> {
        if !minimum.is_finite()
            || !(0. ..=1.).contains(&minimum)
            || !(1..=MAX_PROPOSALS).contains(&count)
            || !self.native_confidence_supported()
        {
            return Err(Error::InvalidFormat("MTP confidence-window scope"));
        }
        if minimum == 0. {
            return self.draft_window(token, carry, count, cancel);
        }
        // Same stopping rule as pinned draft(): generate the first draft, then
        // launch j only if j-1's probability met the configured threshold.
        validate_input(token, carry, self.position)?;
        k::validate_position(
            self.position
                .checked_add(count - 1)
                .ok_or(Error::ArithmeticOverflow)?,
        )?;
        let start = self.position;
        let mut tokens = Zeroizing::new(Vec::with_capacity(MAX_PROPOSALS));
        let mut probabilities = Zeroizing::new(Vec::with_capacity(MAX_PROPOSALS));
        let mut last_enqueue_ns = 0;
        for step in 0..count {
            let clock = Instant::now();
            let output = if step == 0 {
                self.step_draft(token, carry, cancel)?
            } else {
                self.follow_draft(cancel)?
            };
            if output.start() != start + step || output.binding() != self.binding {
                drop(output);
                self.fail();
                return Err(Error::InvalidFormat("MTP confidence-step stamp"));
            }
            let probability = output.probability()?;
            tokens.push(output.greedy());
            probabilities.push(probability);
            drop(output);
            last_enqueue_ns =
                u64::try_from(clock.elapsed().as_nanos()).map_err(|_| Error::ArithmeticOverflow)?;
            if probability < minimum {
                break;
            }
        }
        Ok(DraftWindow {
            tokens,
            probabilities,
            start,
            binding: self.binding,
            last_enqueue_ns,
        })
    }

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
            || !(1..=MAX_PROPOSALS).contains(&count)
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
        // At most the compiled number of bounded per-cell INFO sets may remain queued together.
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
                        self.a(B::Candidates)?,
                        self.native_confidence_supported(),
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
                let WindowPayload {
                    tokens,
                    probabilities,
                } = decode_window(&packet, count, self.native_confidence_supported())?;
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
