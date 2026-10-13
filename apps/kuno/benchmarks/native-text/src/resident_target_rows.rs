//! Explicit fixed-public target output for one to three supplied rows.
//! This returns only after the parent's full chunk commits, not an accepted
//! speculative prefix. Complete packet leases use the original audited I/O.
#![allow(dead_code)] // Parent owns opt-in registration and physical acceptance.
use super::*;
use cubecl::server::Handle;
use kuno_text::model::GraphBindingId;

pub(crate) const MAX_ROWS: usize = 3;
pub(crate) const FOUR_ROWS: usize = crate::strata_port::verify_capacity();
const FOUR_LOGITS_WORDS: usize = FOUR_ROWS * VOCABULARY;
const FOUR_PACKET_WORDS: usize = HEADER_WORDS + FOUR_LOGITS_WORDS + FOUR_ROWS * HC_WIDTH;
pub(crate) const VOCABULARY: usize = 248320;
pub(crate) const HC_WIDTH: usize = 10240;
pub(crate) const HEADER_WORDS: usize = 8;
pub(crate) const LOGITS_WORDS: usize = MAX_ROWS * VOCABULARY;
pub(crate) const CARRY_WORDS: usize = MAX_ROWS * HC_WIDTH;
pub(crate) const PACKET_WORDS: usize = HEADER_WORDS + LOGITS_WORDS + CARRY_WORDS;
pub(crate) const OWNER_BYTES: usize = if cfg!(feature = "strata-eight-window") {
    8
} else {
    4
} * 1024
    * 1024;
pub(crate) const MAIN_BYTES: u64 = 2 * OWNER_BYTES as u64;
pub(crate) const HOST_CAP: usize = if cfg!(feature = "strata-eight-window") {
    16
} else {
    8
} * 1024
    * 1024;
pub(crate) const HOST_INCREMENT: usize = if cfg!(feature = "strata-eight-window") {
    12
} else {
    6
} * 1024
    * 1024;
pub(crate) const RUNTIME_INCREMENT: u64 = 2 * (OWNER_BYTES as u64 - 1024 * 1024) + 2 * 2048;
pub(crate) const RUNTIME_LIMIT: u64 = 128 * 1024 * 1024;

fn rejected() -> Error {
    Error::InvalidFormat("native target rows owning output")
}
pub(super) fn check_public_audit(reserved: bool, public_zero_audit: bool) -> Result<()> {
    if reserved && !public_zero_audit {
        return Err(rejected());
    }
    Ok(())
}
pub(super) fn coordinates_selected(count: usize, start: usize, four: bool) -> Result<()> {
    let maximum = if four { FOUR_ROWS } else { MAX_ROWS };
    if !(1..=maximum).contains(&count) || start.checked_add(count).is_none_or(|end| end > 4096) {
        return Err(rejected());
    }
    Ok(())
}
pub(crate) fn coordinates(count: usize, start: usize) -> Result<()> {
    if !(1..=MAX_ROWS).contains(&count) || start.checked_add(count).is_none_or(|end| end > 4096) {
        return Err(rejected());
    }
    Ok(())
}

/// Immutable request-derived output; no Debug, persistence or mutable payload.
/// The opaque public binding token is released on drop; payload capacities and
/// request-derived start/count are wiped. A row consumed start+row and predicts
/// the next token; hidden is the unnormalized pre-output_hc residual.
pub(crate) struct NativeTargetRows {
    payload: Payload,
    binding: Option<GraphBindingId>,
    // Field order retains the lease through payload destruction.
    _lease: OutputLease,
}
struct Payload {
    logits: Zeroizing<Vec<f32>>,
    greedy: Option<Zeroizing<Vec<u32>>>,
    hidden: Zeroizing<Vec<f32>>,
    coordinates: Zeroizing<[usize; 2]>,
}
impl NativeTargetRows {
    pub(crate) fn binding_id(&self) -> GraphBindingId {
        self.binding.expect("live target rows binding")
    }
    pub(crate) fn start(&self) -> usize {
        self.payload.coordinates[0]
    }
    pub(crate) fn count(&self) -> usize {
        self.payload.coordinates[1]
    }
    pub(crate) fn position(&self, row: usize) -> Result<usize> {
        if row >= self.count() {
            return Err(Error::OutOfBounds);
        }
        self.start()
            .checked_add(row)
            .ok_or(Error::ArithmeticOverflow)
    }
    pub(crate) fn logits(&self, row: usize) -> Result<&[f32]> {
        self.payload.row(row, false)
    }
    pub(crate) fn greedy(&self, row: usize) -> Result<u32> {
        if row >= self.count() {
            return Err(Error::OutOfBounds);
        }
        if let Some(tokens) = &self.payload.greedy {
            return tokens.get(row).copied().ok_or_else(rejected);
        }
        let logits = self.logits(row)?;
        let mut best = 0;
        for i in 1..logits.len() {
            if logits[i] > logits[best] {
                best = i;
            }
        }
        Ok(best as u32)
    }
    pub(crate) fn hidden(&self, row: usize) -> Result<&[f32]> {
        self.payload.row(row, true)
    }
}
impl Drop for NativeTargetRows {
    fn drop(&mut self) {
        // Finish wiping before the lease's release permits another chunk,
        // even if this result was moved to and dropped by another host thread.
        self.payload.logits.zeroize();
        self.payload.hidden.zeroize();
        self.payload.coordinates.zeroize();
        self.binding = None;
    }
}

struct OutputLease(Arc<AtomicBool>);
impl OutputLease {
    fn acquire(live: &Arc<AtomicBool>) -> Result<Self> {
        live.compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
            .map_err(|_| rejected())?;
        Ok(Self(live.clone()))
    }
}
impl Drop for OutputLease {
    fn drop(&mut self) {
        self.0.store(false, Ordering::Release);
    }
}
impl Payload {
    fn row(&self, row: usize, hidden: bool) -> Result<&[f32]> {
        if row >= self.coordinates[1] {
            return Err(Error::OutOfBounds);
        }
        let (values, width) = if hidden {
            (&self.hidden, HC_WIDTH)
        } else {
            (&self.logits, VOCABULARY)
        };
        values.get(row * width..(row + 1) * width).ok_or(rejected())
    }
}

fn word(bytes: &[u8], index: usize) -> Result<u32> {
    let first = index.checked_mul(4).ok_or(Error::ArithmeticOverflow)?;
    let end = first.checked_add(4).ok_or(Error::ArithmeticOverflow)?;
    let value = bytes.get(first..end).ok_or(rejected())?;
    Ok(u32::from_le_bytes(
        value.try_into().map_err(|_| rejected())?,
    ))
}
fn decode(bytes: &[u8], count: usize, start: usize) -> Result<Payload> {
    decode_selected(bytes, count, start, false)
}
fn decode_selected(bytes: &[u8], count: usize, start: usize, four: bool) -> Result<Payload> {
    let rows = if four { FOUR_ROWS } else { MAX_ROWS };
    let logits_words = rows * VOCABULARY;
    let packet_words = HEADER_WORDS + logits_words + rows * HC_WIDTH;
    coordinates_selected(count, start, four)?;
    if bytes.len() != OWNER_BYTES {
        return Err(rejected());
    }
    let expected = [
        1,
        count as u32,
        start as u32,
        VOCABULARY as u32,
        HC_WIDTH as u32,
        rows as u32,
        0,
        0,
    ];
    for (index, value) in expected.into_iter().enumerate() {
        if word(bytes, index)? != value {
            return Err(rejected());
        }
    }
    // Check the complete owning span before allocating either returned payload.
    for index in HEADER_WORDS..OWNER_BYTES / 4 {
        let bits = word(bytes, index)?;
        let active = if index < HEADER_WORDS + logits_words {
            index - HEADER_WORDS < count * VOCABULARY
        } else if index < packet_words {
            index - HEADER_WORDS - logits_words < count * HC_WIDTH
        } else {
            false
        };
        if (active && !f32::from_bits(bits).is_finite()) || (!active && bits != 0) {
            return Err(rejected());
        }
    }
    let mut logits = Zeroizing::new(Vec::new());
    let mut hidden = Zeroizing::new(Vec::new());
    logits
        .try_reserve_exact(count * VOCABULARY)
        .map_err(|_| rejected())?;
    hidden
        .try_reserve_exact(count * HC_WIDTH)
        .map_err(|_| rejected())?;
    let owned = logits
        .capacity()
        .checked_add(hidden.capacity())
        .and_then(|n| n.checked_mul(4))
        .and_then(|n| n.checked_add(bytes.len()))
        .and_then(|n| n.checked_add(std::mem::size_of::<NativeTargetRows>()))
        .ok_or(Error::ArithmeticOverflow)?;
    if logits.capacity() != count * VOCABULARY
        || hidden.capacity() != count * HC_WIDTH
        || owned > HOST_CAP
    {
        return Err(rejected());
    }
    for index in 0..count * VOCABULARY {
        logits.push(f32::from_bits(word(bytes, HEADER_WORDS + index)?));
    }
    for index in 0..count * HC_WIDTH {
        hidden.push(f32::from_bits(word(
            bytes,
            HEADER_WORDS + logits_words + index,
        )?));
    }
    Ok(Payload {
        logits,
        greedy: None,
        hidden,
        coordinates: Zeroizing::new([start, count]),
    })
}

const COMPACT_WORDS: usize = HEADER_WORDS + FOUR_ROWS + FOUR_ROWS * HC_WIDTH;
fn decode_compact(bytes: &[u8], count: usize, start: usize) -> Result<Payload> {
    coordinates_selected(count, start, true)?;
    if bytes.len() != COMPACT_WORDS * 4 {
        return Err(rejected());
    }
    for (i, v) in [
        2,
        count as u32,
        start as u32,
        VOCABULARY as u32,
        HC_WIDTH as u32,
        FOUR_ROWS as u32,
        0,
        0,
    ]
    .into_iter()
    .enumerate()
    {
        if word(bytes, i)? != v {
            return Err(rejected());
        }
    }
    let mut tokens = Zeroizing::new(Vec::with_capacity(count));
    for row in 0..FOUR_ROWS {
        let token = word(bytes, HEADER_WORDS + row)?;
        if row < count {
            if token >= VOCABULARY as u32 {
                return Err(rejected());
            }
            tokens.push(token);
        } else if token != 0 {
            return Err(rejected());
        }
    }
    let mut hidden = Zeroizing::new(Vec::with_capacity(count * HC_WIDTH));
    for i in 0..FOUR_ROWS * HC_WIDTH {
        let bits = word(bytes, HEADER_WORDS + FOUR_ROWS + i)?;
        if i < count * HC_WIDTH {
            let x = f32::from_bits(bits);
            if !x.is_finite() {
                return Err(rejected());
            }
            hidden.push(x);
        } else if bits != 0 {
            return Err(rejected());
        }
    }
    Ok(Payload {
        logits: Zeroizing::new(Vec::new()),
        greedy: Some(tokens),
        hidden,
        coordinates: Zeroizing::new([start, count]),
    })
}

#[cube(launch)]
#[allow(clippy::manual_range_contains)] // CubeCL expands explicit runtime comparisons.
fn target_packet(
    logits: &Array<f32>,
    carry: &Array<f32>,
    fault: &Array<u32>,
    output: &mut Array<u32>,
    count: usize,
    start: usize,
) {
    let index = ABSOLUTE_POS;
    if index < output.len() {
        let mut value = 0u32;
        if index == 0 {
            value = 1;
        } else if index == 1 {
            value = count as u32;
        } else if index == 2 {
            value = start as u32;
        } else if index == 3 {
            value = VOCABULARY as u32;
        } else if index == 4 {
            value = HC_WIDTH as u32;
        } else if index == 5 {
            value = MAX_ROWS as u32;
        } else if index == 6 {
            value = fault[0];
        } else if index >= HEADER_WORDS && index < HEADER_WORDS + LOGITS_WORDS {
            let source = index - HEADER_WORDS;
            if source < count * VOCABULARY {
                value = u32::reinterpret(logits[source]);
            }
        } else if index >= HEADER_WORDS + LOGITS_WORDS && index < PACKET_WORDS {
            let source = index - HEADER_WORDS - LOGITS_WORDS;
            if source < count * HC_WIDTH {
                value = u32::reinterpret(carry[source]);
            }
        }
        output[index] = value;
    }
}

#[cube(launch)]
#[allow(clippy::manual_range_contains)] // CubeCL expands explicit runtime comparisons.
fn target_four_packet(
    logits: &Array<f32>,
    carry: &Array<f32>,
    fault: &Array<u32>,
    output: &mut Array<u32>,
    count: usize,
    start: usize,
) {
    let index = ABSOLUTE_POS;
    if index < output.len() {
        let mut value = 0u32;
        if index == 0 {
            value = 1;
        } else if index == 1 {
            value = count as u32;
        } else if index == 2 {
            value = start as u32;
        } else if index == 3 {
            value = VOCABULARY as u32;
        } else if index == 4 {
            value = HC_WIDTH as u32;
        } else if index == 5 {
            value = FOUR_ROWS as u32;
        } else if index == 6 {
            value = fault[0];
        } else if index >= HEADER_WORDS && index < HEADER_WORDS + FOUR_LOGITS_WORDS {
            let source = index - HEADER_WORDS;
            if source < count * VOCABULARY {
                value = u32::reinterpret(logits[source]);
            }
        } else if index >= HEADER_WORDS + FOUR_LOGITS_WORDS && index < FOUR_PACKET_WORDS {
            let source = index - HEADER_WORDS - FOUR_LOGITS_WORDS;
            if source < count * HC_WIDTH {
                value = u32::reinterpret(carry[source]);
            }
        }
        output[index] = value;
    }
}

pub(super) struct Owners {
    ids: Zeroizing<[BufferId; 2]>,
    output_live: Arc<AtomicBool>,
    four_reserved: bool,
}
impl Owners {
    pub(super) fn allocation_requests() -> Result<[u64; 2]> {
        Ok([
            BufferLayout::new(LOGITS_WORDS)
                .map_err(u::fault)?
                .owned_bytes as u64,
            BufferLayout::new(PACKET_WORDS)
                .map_err(u::fault)?
                .owned_bytes as u64,
        ])
    }
    pub(super) fn allocate(arena: &mut PrivateArena) -> Result<Self> {
        Self::allocate_with_four_rows(arena, false)
    }
    pub(super) fn allocate_with_four_rows(
        arena: &mut PrivateArena,
        four_reserved: bool,
    ) -> Result<Self> {
        Ok(Self {
            ids: Zeroizing::new([
                arena
                    .allocate_zeroed(if four_reserved {
                        FOUR_LOGITS_WORDS
                    } else {
                        LOGITS_WORDS
                    })
                    .map_err(u::fault)?,
                arena
                    .allocate_zeroed(if four_reserved {
                        FOUR_PACKET_WORDS
                    } else {
                        PACKET_WORDS
                    })
                    .map_err(u::fault)?,
            ]),
            output_live: Arc::new(AtomicBool::new(false)),
            four_reserved,
        })
    }
    pub(super) fn check_no_live_output(&self) -> Result<()> {
        if self.output_live.load(Ordering::Acquire) {
            return Err(rejected());
        }
        Ok(())
    }
    pub(super) fn ids(&self) -> [BufferId; 2] {
        *self.ids
    }
    pub(super) fn logits(&self) -> BufferId {
        self.ids[0]
    }
    pub(super) fn four_rows_reserved(&self) -> bool {
        self.four_reserved
    }
    pub(super) fn prove(&self, arena: &PrivateArena) -> Result<[Handle; 2]> {
        let handles = [
            gr::full_private(
                arena,
                self.ids[0],
                if self.four_reserved {
                    FOUR_LOGITS_WORDS
                } else {
                    LOGITS_WORDS
                },
            )?,
            gr::full_private(
                arena,
                self.ids[1],
                if self.four_reserved {
                    FOUR_PACKET_WORDS
                } else {
                    PACKET_WORDS
                },
            )?,
        ];
        gr::physical(arena, &handles)?;
        Ok(handles)
    }
    #[allow(clippy::too_many_arguments)]
    pub(super) fn read(
        &self,
        graph: BoundGraph<'_, '_>,
        arena: &PrivateArena,
        admission: &ResidentAdmission,
        carry: BufferId,
        fault: BufferId,
        count: usize,
        start: usize,
        cancel: &AtomicBool,
    ) -> Result<NativeTargetRows> {
        let four = count > MAX_ROWS;
        if four && !self.four_reserved {
            return Err(rejected());
        }
        coordinates_selected(count, start, four)?;
        graph.verify_unchanged()?;
        let geometry = graph.geometry();
        if geometry.vocabulary != VOCABULARY
            || geometry.residual.width * geometry.residual.streams != HC_WIDTH
        {
            return Err(rejected());
        }
        let [logits, packet] = self.prove(arena)?;
        let carry = gr::full_private(arena, carry, MAX_BATCH * HC_WIDTH)?;
        let fault = gr::full_private(arena, fault, 1)?;
        gr::physical(
            arena,
            &[logits.clone(), packet.clone(), carry.clone(), fault.clone()],
        )?;
        let lease = OutputLease::acquire(&self.output_live)?;
        if crate::strata_port::enabled() {
            return batch_io::read_target_output(
                arena,
                admission,
                packet
                    .clone()
                    .offset_end((OWNER_BYTES - COMPACT_WORDS * 4) as u64),
                cancel,
                || {
                    crate::strata_packet::launch(
                        arena.client().map_err(u::fault)?,
                        [logits, carry, fault, packet],
                        count,
                        start,
                    );
                    Ok(())
                },
                |bytes| {
                    Ok(NativeTargetRows {
                        payload: decode_compact(bytes, count, start)?,
                        binding: Some(graph.binding_id()),
                        _lease: lease,
                    })
                },
            );
        }
        batch_io::read_target_output(
            arena,
            admission,
            packet.clone(),
            cancel,
            || {
                if four {
                    target_four_packet::launch::<WgpuRuntime>(
                        arena.client().map_err(u::fault)?,
                        u::grid(OWNER_BYTES / 4)?,
                        CubeDim::new_1d(64),
                        u::raw_arg(logits, FOUR_LOGITS_WORDS)?,
                        u::raw_arg(carry, MAX_BATCH * HC_WIDTH)?,
                        u::raw_arg(fault, 1)?,
                        u::raw_arg(packet, OWNER_BYTES / 4)?,
                        count,
                        start,
                    );
                } else {
                    target_packet::launch::<WgpuRuntime>(
                        arena.client().map_err(u::fault)?,
                        u::grid(OWNER_BYTES / 4)?,
                        CubeDim::new_1d(64),
                        u::raw_arg(logits, LOGITS_WORDS)?,
                        u::raw_arg(carry, MAX_BATCH * HC_WIDTH)?,
                        u::raw_arg(fault, 1)?,
                        u::raw_arg(packet, OWNER_BYTES / 4)?,
                        count,
                        start,
                    );
                }
                Ok(())
            },
            |bytes| {
                Ok(NativeTargetRows {
                    payload: decode_selected(bytes, count, start, four)?,
                    binding: Some(graph.binding_id()),
                    _lease: lease,
                })
            },
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn packet(count: usize, start: usize) -> Zeroizing<Vec<u8>> {
        packet_selected(count, start, false)
    }
    fn packet_selected(count: usize, start: usize, four: bool) -> Zeroizing<Vec<u8>> {
        let capacity = if four { FOUR_ROWS } else { MAX_ROWS };
        let logits_words = capacity * VOCABULARY;
        let mut bytes = Zeroizing::new(vec![0; OWNER_BYTES]);
        let header = [
            1,
            count as u32,
            start as u32,
            VOCABULARY as u32,
            HC_WIDTH as u32,
            capacity as u32,
            0,
            0,
        ];
        for (index, value) in header.into_iter().enumerate() {
            set(&mut bytes, index, value);
        }
        for row in 0..count {
            for column in 0..VOCABULARY {
                set(
                    &mut bytes,
                    HEADER_WORDS + row * VOCABULARY + column,
                    ((row + 1) as f32).to_bits(),
                );
            }
            for stream in 0..4 {
                for column in 0..2560 {
                    set(
                        &mut bytes,
                        HEADER_WORDS + logits_words + row * HC_WIDTH + stream * 2560 + column,
                        ((row * 4 + stream + 1) as f32).to_bits(),
                    );
                }
            }
        }
        bytes
    }
    fn set(bytes: &mut [u8], index: usize, value: u32) {
        bytes[index * 4..index * 4 + 4].copy_from_slice(&value.to_le_bytes());
    }
    #[test]
    fn four_rows_fit_original_owners_and_host_cap_with_tails_still_original() {
        assert_eq!(FOUR_LOGITS_WORDS * 4, 3973120);
        assert_eq!(FOUR_PACKET_WORDS * 4, 4136992);
        assert_eq!(
            OWNER_BYTES + (FOUR_ROWS * (VOCABULARY + HC_WIDTH)) * 4,
            8331264
        );
        assert!(
            OWNER_BYTES
                + (FOUR_ROWS * (VOCABULARY + HC_WIDTH)) * 4
                + std::mem::size_of::<NativeTargetRows>()
                < HOST_CAP
        );
        for count in [4, 2, 4] {
            let four = count == 4;
            let mut bytes = packet_selected(count, 11, four);
            let value = decode_selected(&bytes, count, 11, four).unwrap();
            assert_eq!(value.coordinates[1], count);
            assert!(
                value
                    .row(count - 1, true)
                    .unwrap()
                    .iter()
                    .all(|v| v.is_finite())
            );
            assert!(decode_selected(&bytes, count, 11, !four).is_err());
            if four {
                for (index, bits) in [
                    (6, 1),
                    (HEADER_WORDS + 3 * VOCABULARY, f32::NAN.to_bits()),
                    (
                        HEADER_WORDS + FOUR_LOGITS_WORDS + 3 * HC_WIDTH,
                        f32::INFINITY.to_bits(),
                    ),
                ] {
                    let original = word(&bytes, index).unwrap();
                    set(&mut bytes, index, bits);
                    assert!(decode_selected(&bytes, count, 11, true).is_err());
                    set(&mut bytes, index, original);
                }
            }
            set(&mut bytes, OWNER_BYTES / 4 - 1, 1);
            assert!(decode_selected(&bytes, count, 11, four).is_err());
        }
        assert!(coordinates(4, 0).is_err());
        assert!(coordinates_selected(4, 4093, true).is_err());
        assert!(coordinates_selected(4, 4092, true).is_ok());
    }
    #[test]
    fn exact_three_row_memory_and_info_charges() {
        assert_eq!(LOGITS_WORDS, 744960);
        assert_eq!(PACKET_WORDS, 775688);
        assert_eq!(Owners::allocation_requests().unwrap(), [4194304; 2]);
        assert_eq!(MAIN_BYTES, 8388608);
        assert_eq!(HOST_INCREMENT, 6291456);
        assert_eq!(RUNTIME_INCREMENT, 6295552);
        assert!(
            OWNER_BYTES
                + (LOGITS_WORDS + CARRY_WORDS) * 4
                + std::mem::size_of::<NativeTargetRows>()
                < HOST_CAP
        );
        for (reserved, audit) in [(false, false), (false, true), (true, true)] {
            assert!(check_public_audit(reserved, audit).is_ok());
        }
        assert!(check_public_audit(true, false).is_err());
    }
    #[test]
    fn live_output_blocks_another_chunk_until_guarded_result_destruction() {
        let live = Arc::new(AtomicBool::new(false));
        let lease = OutputLease::acquire(&live).unwrap();
        assert!(OutputLease::acquire(&live).is_err());
        let rows = NativeTargetRows {
            payload: decode(&packet(3, 11), 3, 11).unwrap(),
            binding: None, // No graph instance is created by this host test.
            _lease: lease,
        };
        assert_eq!((rows.start(), rows.count()), (11, 3));
        assert_eq!(rows.position(2).unwrap(), 13);
        assert!(rows.position(3).is_err());
        assert_eq!(rows.logits(2).unwrap().len(), VOCABULARY);
        assert_eq!(rows.hidden(2).unwrap().len(), HC_WIDTH);
        assert!(OutputLease::acquire(&live).is_err());
        drop(rows);
        let next = OutputLease::acquire(&live).unwrap();
        drop(next);
        assert!(!live.load(Ordering::Acquire));
        // Failed decoding never retains an exclusive output lease.
        let attempt = (|| -> Result<NativeTargetRows> {
            let lease = OutputLease::acquire(&live)?;
            Ok(NativeTargetRows {
                payload: decode(&[], 1, 0)?,
                binding: None,
                _lease: lease,
            })
        })();
        assert!(attempt.is_err());
        assert!(!live.load(Ordering::Acquire));
    }
    #[test]
    fn every_active_row_and_stream_has_its_own_bounded_original_position() {
        for count in 1..=3 {
            let bytes = packet(count, 11);
            let output = decode(&bytes, count, 11).unwrap();
            assert_eq!(*output.coordinates, [11, count]);
            for row in 0..count {
                assert!(
                    output
                        .row(row, false)
                        .unwrap()
                        .iter()
                        .all(|&v| v == (row + 1) as f32)
                );
                for (stream, values) in output
                    .row(row, true)
                    .unwrap()
                    .chunks_exact(2560)
                    .enumerate()
                {
                    assert!(values.iter().all(|&v| v == (row * 4 + stream + 1) as f32));
                }
            }
            assert!(output.row(count, false).is_err());
        }
    }
    #[test]
    fn wrong_header_fault_nonfinite_inactive_and_full_padding_are_rejected() {
        let original = packet(1, 0);
        for (index, bits) in [
            (0, 2),
            (1, 2),
            (2, 1),
            (3, 1),
            (4, 1),
            (5, 1),
            (6, 1),
            (7, 1),
            (HEADER_WORDS, f32::NAN.to_bits()),
            (HEADER_WORDS + LOGITS_WORDS, f32::INFINITY.to_bits()),
            (HEADER_WORDS + VOCABULARY, 1),
            (HEADER_WORDS + LOGITS_WORDS + HC_WIDTH, 1),
            (PACKET_WORDS, 1),
            (OWNER_BYTES / 4 - 1, 1),
        ] {
            let mut corrupt = original.clone();
            set(&mut corrupt, index, bits);
            assert!(decode(&corrupt, 1, 0).is_err());
        }
        assert!(decode(&original[..OWNER_BYTES - 4], 1, 0).is_err());
        assert!(decode(&original, 0, 0).is_err());
        assert!(decode(&original, 4, 0).is_err());
        assert!(coordinates(3, 4094).is_err());
        assert!(coordinates(1, usize::MAX).is_err());
    }
    #[test]
    fn owning_target_packet_validates_offline_with_runtime_count_and_start() {
        use cubecl::compute::KernelBuilder;
        use cubecl::ir::{ElemType, FloatKind, StorageType, Type, UIntKind};
        use cubecl::server::ExecutionMode;
        use cubecl::wgpu::WgslCompiler;
        use cubecl::{Compiler, KernelSettings, WgpuCompilationOptions};
        for mode in [ExecutionMode::Checked, ExecutionMode::Unchecked] {
            let mut b = KernelBuilder::default();
            let f = StorageType::Scalar(ElemType::Float(FloatKind::F32));
            let u = StorageType::Scalar(ElemType::UInt(UIntKind::U32));
            b.scope.register_type::<f32>(f);
            b.scope.register_type::<u32>(u);
            b.scope.register_type::<usize>(u);
            let logits = b.input_array(Type::new(f)).into();
            let carry = b.input_array(Type::new(f)).into();
            let fault = b.input_array(Type::new(u)).into();
            let output = b.output_array(Type::new(u)).into();
            let count = b.scalar(u).into();
            let start = b.scalar(u).into();
            target_packet::expand(&mut b.scope, logits, carry, fault, output, count, start);
            let shader = WgslCompiler::default()
                .compile(
                    b.build(
                        KernelSettings::default()
                            .cube_dim(CubeDim::new_1d(64))
                            .kernel_name("native_target_rows_packet"),
                    ),
                    &WgpuCompilationOptions::default(),
                    mode,
                    u,
                )
                .unwrap()
                .to_string();
            let module = wgpu::naga::front::wgsl::parse_str(&shader).unwrap();
            wgpu::naga::valid::Validator::new(
                wgpu::naga::valid::ValidationFlags::all(),
                wgpu::naga::valid::Capabilities::empty(),
            )
            .validate(&module)
            .unwrap();
            assert!(shader.contains("bitcast<u32>"));
            assert!(shader.contains("scalars"));
        }
    }
    #[test]
    fn four_owning_target_packet_validates_offline_without_old_carry_offset() {
        use cubecl::compute::KernelBuilder;
        use cubecl::ir::{ElemType, FloatKind, StorageType, Type, UIntKind};
        use cubecl::server::ExecutionMode;
        use cubecl::wgpu::WgslCompiler;
        use cubecl::{Compiler, KernelSettings, WgpuCompilationOptions};
        for mode in [ExecutionMode::Checked, ExecutionMode::Unchecked] {
            let mut b = KernelBuilder::default();
            let f = StorageType::Scalar(ElemType::Float(FloatKind::F32));
            let u = StorageType::Scalar(ElemType::UInt(UIntKind::U32));
            b.scope.register_type::<f32>(f);
            b.scope.register_type::<u32>(u);
            b.scope.register_type::<usize>(u);
            let logits = b.input_array(Type::new(f)).into();
            let carry = b.input_array(Type::new(f)).into();
            let fault = b.input_array(Type::new(u)).into();
            let output = b.output_array(Type::new(u)).into();
            let count = b.scalar(u).into();
            let start = b.scalar(u).into();
            target_four_packet::expand(&mut b.scope, logits, carry, fault, output, count, start);
            let shader = WgslCompiler::default()
                .compile(
                    b.build(
                        KernelSettings::default()
                            .cube_dim(CubeDim::new_1d(64))
                            .kernel_name("native_four_target_rows_packet"),
                    ),
                    &WgpuCompilationOptions::default(),
                    mode,
                    u,
                )
                .unwrap()
                .to_string();
            let module = wgpu::naga::front::wgsl::parse_str(&shader).unwrap();
            wgpu::naga::valid::Validator::new(
                wgpu::naga::valid::ValidationFlags::all(),
                wgpu::naga::valid::Capabilities::empty(),
            )
            .validate(&module)
            .unwrap();
            assert!(shader.contains("bitcast<u32>"));
            assert!(shader.contains("scalars"));
        }
    }
}
