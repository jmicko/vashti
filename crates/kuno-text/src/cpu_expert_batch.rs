//! Default-unused, borrowed original-F32 CPU expert batches, counts1..3.
//! Geometry-checked views do not authenticate weight origin or source epoch.
//! The eventual parent owns that proof and every borrowed source lifetime.
//! See docs/kuno-native-borrowed-cpu-batch.md; no mapping, model-weight copy,
//! routing, probability multiplication or GPU/model activation occurs here.
use super::*;

pub const MAX_MEMBERS: usize = 3;
pub const MAX_RANKS: usize = MAX_MEMBERS * MAX_SELECTED;

/// Exact original packed expert spans, borrowed from caller-owned immutable
/// weights. The enclosing normal weights view supplies geometry/source identity;
/// the caller must additionally authenticate payload origin, epoch and lifetime.
pub struct ExpertView<'a> {
    geometry: &'a Geometry,
    expert: usize,
    packed: [&'a [u8]; 3],
}
impl ExpertView<'_> {
    pub fn expert(&self) -> usize {
        self.expert
    }
}

impl PackedExperts<'_, '_> {
    /// No reader call, map, allocation, conversion or payload copy. This checks
    /// exact spans, not the caller's claim that these bytes belong to this ID.
    pub fn borrow_expert<'a>(
        &'a self,
        expert: usize,
        packed: [&'a [u8]; 3],
    ) -> Result<ExpertView<'a>> {
        if expert >= self.geometry.experts {
            return Err(Error::InvalidFormat("CPU batch original expert"));
        }
        for (part, span) in packed.iter().enumerate() {
            if span.len() != self.geometry.packed_bytes[part] {
                return Err(Error::InvalidBufferLength);
            }
        }
        Ok(ExpertView {
            geometry: &self.geometry,
            expert,
            packed,
        })
    }
}

/// One HOST-owned rank, retaining its original routing position. Caller-owned
/// borrowed selection headers must remain alive through the joined operation.
pub struct Rank<'a> {
    original: usize,
    expert: &'a ExpertView<'a>,
}
impl<'a> Rank<'a> {
    pub fn new(original: usize, expert: &'a ExpertView<'a>) -> Result<Self> {
        if original >= MAX_SELECTED {
            return Err(Error::InvalidBufferLength);
        }
        Ok(Self { original, expert })
    }
    pub fn original(&self) -> usize {
        self.original
    }
}

/// One input vector with an ordered sparse original-rank subset. Empty HOST
/// subsets are allowed; they never authorize a missing resident-GPU partial.
pub struct Member<'a> {
    input: &'a [f32],
    ranks: &'a [Rank<'a>],
}
impl<'a> Member<'a> {
    pub fn new(input: &'a [f32], ranks: &'a [Rank<'a>]) -> Result<Self> {
        if input.is_empty() || input.len() > MAX_HIDDEN || ranks.len() > MAX_SELECTED {
            return Err(Error::InvalidBufferLength);
        }
        for (index, rank) in ranks.iter().enumerate() {
            if index > 0 && ranks[index - 1].original >= rank.original {
                return Err(Error::InvalidFormat("CPU batch original rank order"));
            }
            if ranks[..index]
                .iter()
                .any(|prior| prior.expert.expert == rank.expert.expert)
            {
                return Err(Error::InvalidFormat("CPU batch repeated member expert"));
            }
        }
        Ok(Self { input, ranks })
    }
}

/// Full three-member/ten-rank output, unweighted and in original rank order.
/// No Clone, Debug, serialization or mutable view. Missing ranks are exact +0.
pub struct BatchPartials {
    hidden: usize,
    members: usize,
    masks: [u16; MAX_MEMBERS],
    values: Zeroizing<Vec<f32>>,
}
impl BatchPartials {
    pub fn members(&self) -> usize {
        self.members
    }
    pub fn hidden(&self) -> usize {
        self.hidden
    }
    pub fn ranks(&self, member: usize) -> Result<usize> {
        if member >= self.members {
            return Err(Error::InvalidBufferLength);
        }
        Ok(self.masks[member].count_ones() as usize)
    }
    pub fn rank(&self, member: usize, rank: usize) -> Result<Option<&[f32]>> {
        if member >= self.members || rank >= MAX_SELECTED {
            return Err(Error::InvalidBufferLength);
        }
        if self.masks[member] & (1 << rank) == 0 {
            return Ok(None);
        }
        let first = (member * MAX_SELECTED + rank) * self.hidden;
        Ok(Some(&self.values[first..first + self.hidden]))
    }
    pub fn member_full(&self, member: usize) -> Result<&[f32]> {
        if member >= self.members {
            return Err(Error::InvalidBufferLength);
        }
        let first = member * MAX_SELECTED * self.hidden;
        Ok(&self.values[first..first + MAX_SELECTED * self.hidden])
    }
    pub fn full(&self) -> &[f32] {
        &self.values[..MAX_RANKS * self.hidden]
    }
    pub fn owned_bytes(&self) -> usize {
        self.values.capacity() * size_of::<f32>()
    }
    fn wipe(&mut self) {
        // zeroed() initializes complete capacity, including allocator padding.
        self.values.as_mut_slice().zeroize();
        self.hidden.zeroize();
        self.members.zeroize();
        self.masks.zeroize();
    }
}
impl Drop for BatchPartials {
    fn drop(&mut self) {
        self.wipe();
    }
}

struct Plan {
    live: usize,
    selections: [usize; MAX_RANKS],
    destinations: [usize; MAX_RANKS],
    masks: [u16; MAX_MEMBERS],
}
impl Zeroize for Plan {
    fn zeroize(&mut self) {
        self.live.zeroize();
        self.selections.zeroize();
        self.destinations.zeroize();
        self.masks.zeroize();
    }
}

struct Buffers {
    inputs: Zeroizing<Vec<f32>>,
    activations: Zeroizing<Vec<f32>>,
    values: Zeroizing<Vec<f32>>,
}
impl Buffers {
    fn new(g: &Geometry) -> Result<Self> {
        Ok(Self {
            inputs: zeroed(MAX_MEMBERS * g.hidden)?,
            activations: zeroed(MAX_RANKS * g.feed_forward)?,
            values: zeroed(MAX_RANKS * g.hidden)?,
        })
    }
    fn owned_bytes(&self) -> Result<usize> {
        checked_sum(&[
            bytes(&self.inputs)?,
            bytes(&self.activations)?,
            bytes(&self.values)?,
        ])
    }
    fn wipe(&mut self) {
        self.inputs.as_mut_slice().zeroize();
        self.activations.as_mut_slice().zeroize();
        self.values.as_mut_slice().zeroize();
    }
}
impl Drop for Buffers {
    fn drop(&mut self) {
        self.wipe();
    }
}

impl ExpertOperator {
    fn borrowed_batch_fixed_bytes(&self, weights: &PackedExperts<'_, '_>) -> Result<usize> {
        if self.pool.is_none()
            || self.backend == Backend::Avx2Fma
            || self.q8_activations
            || self.q8_residual
            || self.mapped_prefetch
            || self.buffered_weights
            || self.phase_diagnostics
        {
            return Err(Error::InvalidFormat("CPU batch original pooled policy"));
        }
        validate_workers(self.workers.count)?;
        // The existing worker constructor and potential decoded-row allocation
        // use exactly this original scratch bound, not a new approximate budget.
        Scratch::check_capacity(weights.geometry.hidden.max(weights.geometry.feed_forward))?;
        checked_sum(&[
            MAX_WORKER_PRIVATE_BYTES
                .checked_mul(self.workers.count)
                .ok_or(Error::ArithmeticOverflow)?,
            worker_handle_metadata_bytes()?,
            size_of::<Plan>(),
            size_of::<Buffers>(),
            size_of::<BatchPartials>(),
        ])
    }

    /// Minimum full-capacity admission before allocating any request buffer.
    /// Actual allocator capacities are checked again together before private fill.
    pub fn borrowed_batch_private_bytes(&self, weights: &PackedExperts<'_, '_>) -> Result<usize> {
        let g = &weights.geometry;
        let words = checked_sum(&[
            MAX_MEMBERS * g.hidden,
            MAX_RANKS * g.feed_forward,
            MAX_RANKS * g.hidden,
        ])?;
        checked_sum(&[
            words
                .checked_mul(size_of::<f32>())
                .ok_or(Error::ArithmeticOverflow)?,
            self.borrowed_batch_fixed_bytes(weights)?,
        ])
    }

    /// Original-F32 packed dots over caller-authenticated immutable views.
    /// Both row phases join on every exit. No payload I/O or weights ownership.
    pub fn execute_borrowed_batch_with_private_limit(
        &self,
        weights: &PackedExperts<'_, '_>,
        members: &[Member<'_>],
        cancel: &AtomicBool,
        private_limit: usize,
    ) -> Result<BatchPartials> {
        self.execute_borrowed_batch_with(weights, members, cancel, private_limit, |_| {})
    }

    pub(super) fn execute_borrowed_batch_with(
        &self,
        weights: &PackedExperts<'_, '_>,
        members: &[Member<'_>],
        cancel: &AtomicBool,
        private_limit: usize,
        hook: impl Fn(usize) + Sync,
    ) -> Result<BatchPartials> {
        let g = &weights.geometry;
        let required = self.borrowed_batch_private_bytes(weights)?;
        if members.is_empty() || members.len() > MAX_MEMBERS {
            return Err(Error::InvalidBufferLength);
        }
        if private_limit > MAX_PRIVATE_HOST_BYTES || required > private_limit {
            return Err(Error::LimitExceeded("CPU batch private host capacity"));
        }
        let mut plan = Zeroizing::new(Plan {
            live: 0,
            selections: [0; MAX_RANKS],
            destinations: [0; MAX_RANKS],
            masks: [0; MAX_MEMBERS],
        });
        for (member, request) in members.iter().enumerate() {
            if request.input.len() != g.hidden
                || request.input.iter().any(|value| !value.is_finite())
            {
                return Err(Error::InvalidFormat("CPU batch original finite input"));
            }
            for (index, rank) in request.ranks.iter().enumerate() {
                if !std::ptr::eq(g, rank.expert.geometry) {
                    return Err(Error::InvalidFormat("CPU batch original source identity"));
                }
                let live = plan.live;
                plan.selections[live] = member * MAX_SELECTED + index;
                plan.destinations[live] = member * MAX_SELECTED + rank.original;
                plan.masks[member] |= 1 << rank.original;
                plan.live += 1;
            }
        }
        check_cancel(cancel)?;
        let mut buffers = Buffers::new(g)?;
        if checked_sum(&[
            buffers.owned_bytes()?,
            self.borrowed_batch_fixed_bytes(weights)?,
        ])? > private_limit
        {
            return Err(Error::LimitExceeded("CPU batch actual private capacity"));
        }
        for (member, request) in members.iter().enumerate() {
            buffers.inputs[member * g.hidden..(member + 1) * g.hidden]
                .copy_from_slice(request.input);
        }
        let live = plan.live;
        let width = g.hidden.max(g.feed_forward);
        if live != 0 {
            self.rows(
                &mut buffers.activations[..live * g.feed_forward],
                width,
                cancel,
                &hook,
                |index, scratch| {
                    let job = plan.selections[index / g.feed_forward];
                    let member = job / MAX_SELECTED;
                    let rank = job % MAX_SELECTED;
                    let expert = members[member].ranks[rank].expert;
                    let feature = index % g.feed_forward;
                    let input = &buffers.inputs[member * g.hidden..(member + 1) * g.hidden];
                    let gate = dot_row(
                        self.backend,
                        g.kinds[0],
                        row(expert.packed[0], feature, g.row_bytes[0])?,
                        input,
                        scratch,
                        cancel,
                    )?;
                    let up = dot_row(
                        self.backend,
                        g.kinds[1],
                        row(expert.packed[1], feature, g.row_bytes[1])?,
                        input,
                        scratch,
                        cancel,
                    )?;
                    finite(
                        crate::ops::silu(gate) * up,
                        "CPU expert non-finite activation",
                    )
                },
            )?;
            self.rows(
                &mut buffers.values[..live * g.hidden],
                width,
                cancel,
                &hook,
                |index, scratch| {
                    let compact = index / g.hidden;
                    let job = plan.selections[compact];
                    let expert = members[job / MAX_SELECTED].ranks[job % MAX_SELECTED].expert;
                    dot_row(
                        self.backend,
                        g.kinds[2],
                        row(expert.packed[2], index % g.hidden, g.row_bytes[2])?,
                        &buffers.activations
                            [compact * g.feed_forward..(compact + 1) * g.feed_forward],
                        scratch,
                        cancel,
                    )
                },
            )?;
        }
        check_cancel(cancel)?;
        // Destinations are strictly increasing and never below compact slots.
        // Moving backwards preserves every unweighted source row without a
        // second full output owner, then zeroes every missing original rank.
        for compact in (0..live).rev() {
            let first = compact * g.hidden;
            buffers.values.copy_within(
                first..first + g.hidden,
                plan.destinations[compact] * g.hidden,
            );
        }
        for slot in 0..MAX_RANKS {
            if plan.masks[slot / MAX_SELECTED] & (1 << (slot % MAX_SELECTED)) == 0 {
                buffers.values[slot * g.hidden..(slot + 1) * g.hidden].fill(0.);
            }
        }
        if buffers.values[MAX_RANKS * g.hidden..]
            .iter()
            .any(|value| value.to_bits() != 0)
            || buffers.activations[live * g.feed_forward..]
                .iter()
                .any(|value| value.to_bits() != 0)
            || buffers.inputs[members.len() * g.hidden..]
                .iter()
                .any(|value| value.to_bits() != 0)
        {
            return Err(Error::InvalidFormat("CPU batch inactive owning tail"));
        }
        let values = std::mem::replace(&mut buffers.values, Zeroizing::new(Vec::new()));
        Ok(BatchPartials {
            hidden: g.hidden,
            members: members.len(),
            masks: plan.masks,
            values,
        })
    }
}

#[cfg(test)]
mod owner_tests {
    use super::*;

    #[test]
    fn complete_initialized_buffer_and_output_capacities_are_erased() {
        let g = Geometry {
            hidden: 64,
            feed_forward: 32,
            experts: 1,
            kinds: [TensorType::F32; 3],
            row_bytes: [256, 256, 128],
            packed_bytes: [8192; 3],
        };
        let mut buffers = Buffers::new(&g).unwrap();
        buffers.inputs.fill(11.);
        buffers.activations.fill(23.);
        buffers.values.fill(37.);
        let bytes = buffers.owned_bytes().unwrap();
        buffers.wipe();
        assert_eq!(buffers.owned_bytes().unwrap(), bytes);
        for values in [&buffers.inputs, &buffers.activations, &buffers.values] {
            assert_eq!(values.len(), values.capacity());
            assert!(values.iter().all(|value| value.to_bits() == 0));
        }
        let mut output = BatchPartials {
            hidden: g.hidden,
            members: 3,
            masks: [1023; MAX_MEMBERS],
            values: std::mem::replace(&mut buffers.values, Zeroizing::new(Vec::new())),
        };
        output.values.fill(43.);
        let owned = output.owned_bytes();
        output.wipe();
        assert_eq!(output.owned_bytes(), owned);
        assert_eq!(output.values.len(), output.values.capacity());
        assert!(output.values.iter().all(|value| value.to_bits() == 0));
        assert_eq!(output.hidden, 0);
        assert_eq!(output.members, 0);
        assert_eq!(output.masks, [0; MAX_MEMBERS]);
    }
}
