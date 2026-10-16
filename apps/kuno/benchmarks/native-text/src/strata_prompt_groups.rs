//! MIT Strata Intel prompt grouping: logical expert order, stable token/rank
//! rows, one bounded host grouping per layer. All maps remain request-private.
use super::*;
use zeroize::Zeroizing;
// Raw packet/readback lease, grouped map/upload and validation scratch peak
// below2MiB at4096 rows; actual arena bytes are carved from priced owners.
pub(crate) const HOST_BYTES: usize = 2 * 1024 * 1024;
pub(crate) const RUNTIME_BYTES: u64 = 16 * 1024 * 1024;
pub(crate) const CAPACITY: usize = allocation::DEFAULT_ROWS;
// Separate prompt geometry from MAX8 verifier/decode and hardware workgroups.
pub(crate) const ROUTE_CAPACITY: usize = allocation::DEFAULT_ROWS;
pub(crate) const ROUTES: usize = ROUTE_CAPACITY * 10;
pub(crate) const LOGICAL: usize = 65536;
pub(crate) const RAW_WORDS: usize = 131072;
pub(crate) const MAP_WORDS: usize = 65536;
pub(crate) const PROB_WORDS: usize = 65536;
pub(crate) const STATUS_WORDS: usize = 131072;
pub(crate) const GU_BYTES: u64 = (CAPACITY * 1280 * 4) as u64;
pub(crate) const PART_BYTES: u64 = (ROUTES * 2560 * 4) as u64;
/// One compiled prompt ABI for actual Rust views and every native shader.
/// Decode/MAX8 packet dimensions and expert address-table sizes stay separate.
pub(crate) fn shader_prefix() -> String {
    format!(
        "const PROMPT_MAX_ROWS:u32={CAPACITY}u;const PROMPT_ROUTES:u32={ROUTES}u;const PROMPT_LOGICAL:u32={LOGICAL}u;const PROMPT_RAW_WORDS:u32={RAW_WORDS}u;const PROMPT_MAP_WORDS:u32={MAP_WORDS}u;const PROMPT_PROB_WORDS:u32={PROB_WORDS}u;const PROMPT_STATUS_WORDS:u32={STATUS_WORDS}u;const PROMPT_PART_WORDS:u32={}u;const PROMPT_WORKING_WORDS:u32={}u;\n",
        PART_BYTES / 4,
        storage::WORKING_BYTES / 4
    )
}
const MEMBERS: usize = 1540;
const LENGTHS: usize = 516;
const OFFSETS: usize = 1028;
pub(crate) fn group(raw: &[u32], count: usize) -> Result<Zeroizing<Vec<u32>>> {
    if !(1..=ROUTE_CAPACITY).contains(&count)
        || raw.len() != LOGICAL + count * 10
        || raw[..4] != [count as u32, 0, count as u32, 0]
    {
        return Err(crate::strata_prompt::invalid_buffer_length());
    }
    let mut lengths = Zeroizing::new([0usize; 512]);
    let mut cells = Zeroizing::new([512u32; 512]);
    let mut inverse = Zeroizing::new([512u32; 512]);
    for route in 0..count * 10 {
        let logical = raw[LOGICAL + route];
        let cell = raw[MEMBERS + route];
        if logical >= 512 || cell >= 512 {
            return Err(crate::strata_prompt::invalid_buffer_length());
        }
        for earlier in (route / 10) * 10..route {
            if raw[LOGICAL + earlier] == logical {
                return Err(crate::strata_prompt::invalid_buffer_length());
            }
        }
        let l = logical as usize;
        let c = cell as usize;
        if (cells[l] != 512 && cells[l] != cell) || (inverse[c] != 512 && inverse[c] != logical) {
            return Err(crate::strata_prompt::invalid_buffer_length());
        }
        cells[l] = cell;
        inverse[c] = logical;
        lengths[l] += 1;
    }
    let mut map = Zeroizing::new(vec![0u32; MAP_WORDS]);
    map[..3].copy_from_slice(&[count as u32, 0, count as u32]);
    let mut offsets = Zeroizing::new([0usize; 512]);
    let mut n = 0;
    let mut groups = 0;
    for logical in 0..512 {
        if lengths[logical] == 0 {
            continue;
        }
        map[4 + groups] = cells[logical];
        map[LENGTHS + groups] = lengths[logical] as u32;
        map[OFFSETS + groups] = n as u32;
        offsets[logical] = n;
        n += lengths[logical];
        groups += 1;
    }
    map[3] = groups as u32;
    for route in 0..count * 10 {
        let logical = raw[LOGICAL + route] as usize;
        map[MEMBERS + offsets[logical]] = route as u32;
        offsets[logical] += 1;
    }
    validate(&map, count)?;
    Ok(map)
}
pub(crate) fn validate(map: &[u32], count: usize) -> Result<usize> {
    if !(1..=ROUTE_CAPACITY).contains(&count)
        || map.len() != MAP_WORDS
        || map[..3] != [count as u32, 0, count as u32]
    {
        return Err(crate::strata_prompt::invalid_buffer_length());
    }
    let groups = map[3] as usize;
    if groups == 0 || groups > 512 || groups > count * 10 {
        return Err(crate::strata_prompt::invalid_buffer_length());
    }
    let mut seen = Zeroizing::new([false; ROUTES]);
    let mut seen_cells = Zeroizing::new([false; 512]);
    let mut cursor = 0;
    for g in 0..groups {
        let cell = map[4 + g] as usize;
        let length = map[LENGTHS + g] as usize;
        if cell >= 512
            || seen_cells[cell]
            || length == 0
            || length > count
            || map[OFFSETS + g] as usize != cursor
            || cursor + length > count * 10
        {
            return Err(crate::strata_prompt::invalid_buffer_length());
        }
        seen_cells[cell] = true;
        let mut previous = None;
        for i in 0..length {
            let route = map[MEMBERS + cursor + i] as usize;
            if route >= count * 10 || seen[route] || previous.is_some_and(|p| p >= route / 10) {
                return Err(crate::strata_prompt::invalid_buffer_length());
            }
            seen[route] = true;
            previous = Some(route / 10);
        }
        cursor += length;
    }
    if cursor != count * 10 || seen[..count * 10].iter().any(|&v| !v) {
        return Err(crate::strata_prompt::invalid_buffer_length());
    }
    Ok(groups)
}
/// Real prompt-only views. The raw DMA ring has an independent whole owner;
/// GU/PART and routing cells reuse the attention phase above the matrix image.
/// Decode's MAX8 roots and descriptors are never reinterpreted or enlarged.
#[derive(Clone)]
pub(crate) struct Views {
    pub(crate) raw: Handle,
    pub(crate) map: Handle,
    pub(crate) probability: Handle,
    pub(crate) status: Handle,
    pub(crate) gu: Handle,
    pub(crate) partial: Handle,
}
impl Views {
    /// Same phase region as PLE/GDN/QSA, after the live matrix working image.
    /// Transfer DMA uses a distinct whole raw-ring owner, never this region.
    pub(crate) fn new(root: Handle) -> Result<Self> {
        let root_bytes = allocation::whole_root_bytes(&root)?;
        let mut cursor = storage::WORKING_BYTES;
        let mut take = |bytes: u64| -> Result<Handle> {
            let first = cursor;
            let end = first.checked_add(bytes).ok_or(Error::ArithmeticOverflow)?;
            if bytes == 0 || !bytes.is_multiple_of(256) || end > root_bytes {
                return Err(crate::strata_prompt::invalid_buffer_length());
            }
            // Preserve Alloc::take's aligned256-byte gap between live views.
            cursor = end.checked_add(256).ok_or(Error::ArithmeticOverflow)?;
            if cursor > root_bytes {
                return Err(crate::strata_prompt::invalid_buffer_length());
            }
            Ok(root
                .clone()
                .offset_start(first)
                .offset_end(root_bytes - end))
        };
        Ok(Self {
            gu: take(GU_BYTES)?,
            partial: take(PART_BYTES)?,
            raw: take((RAW_WORDS * 4) as u64)?,
            map: take((MAP_WORDS * 4) as u64)?,
            probability: take((PROB_WORDS * 4) as u64)?,
            status: take((STATUS_WORDS * 4) as u64)?,
        })
    }
    pub(crate) fn handles(&self) -> [&Handle; 6] {
        [
            &self.raw,
            &self.map,
            &self.probability,
            &self.status,
            &self.gu,
            &self.partial,
        ]
    }
    /// Header/probability/role flags belong to this layer. Clear their actual
    /// views; stale proof words must not survive a failed router or expert.
    pub(crate) fn reset(&self, client: &ComputeClient<WgpuRuntime>) -> Result<()> {
        use cubecl::prelude::ArrayArg;
        for h in [&self.raw, &self.probability, &self.status] {
            let words =
                usize::try_from(h.size_in_used() / 4).map_err(|_| Error::ArithmeticOverflow)?;
            crate::private_arena::erase_full_span::launch::<WgpuRuntime>(
                client,
                CubeCount::Static(words.div_ceil(64) as u32, 1, 1),
                CubeDim::new_1d(64),
                // SAFETY: exact private view and actual word length, in-stream.
                unsafe { ArrayArg::from_raw_parts(h.clone(), words) },
            );
        }
        Ok(())
    }
}
pub(crate) fn views() -> Result<Views> {
    STAGES.with(|s| {
        s.borrow()
            .as_ref()
            .map(storage::Storage::moe)
            .ok_or(Error::InvalidBufferLength)
    })
}

#[cfg(test)]
mod native_route_layout_tests {
    use super::*;
    use cubecl_common::stream_id::StreamId;
    fn packet(count: usize) -> Zeroizing<Vec<u32>> {
        let mut raw = Zeroizing::new(vec![0u32; LOGICAL + count * 10]);
        raw[..3].copy_from_slice(&[count as u32, 0, count as u32]);
        for token in 0..count {
            for rank in 0..10 {
                let route = token * 10 + rank;
                let logical = ((token * 7 + rank * 37) % 512) as u32;
                raw[MEMBERS + route] = (logical * 13 + 17) % 512;
                raw[LOGICAL + route] = logical;
            }
        }
        raw
    }
    #[test]
    fn full_native_chunk_roundtrips_every_route_without_raw_packet_overlap() {
        for count in [1, 8, 256, 257, 1024, ROUTE_CAPACITY] {
            let raw = packet(count);
            assert!(MEMBERS + count * 10 <= LOGICAL);
            assert!(raw.len() <= RAW_WORDS);
            let grouped = group(&raw, count).unwrap();
            let groups = validate(&grouped, count).unwrap();
            for g in 0..groups {
                let cell = grouped[4 + g];
                let first = grouped[OFFSETS + g] as usize;
                let n = grouped[LENGTHS + g] as usize;
                for &route in &grouped[MEMBERS + first..MEMBERS + first + n] {
                    assert_eq!(raw[MEMBERS + route as usize], cell);
                }
            }
            assert!(count * 10 <= PROB_WORDS);
            assert!(ROUTES + count * 10 <= STATUS_WORDS);
            assert!((count * 1280 * 4) as u64 <= GU_BYTES);
            assert!((count * 10 * 2560 * 4) as u64 <= PART_BYTES);
        }
    }
    #[test]
    fn native_tail_corruption_and_duplicate_routes_cannot_pass_grouping() {
        let count = ROUTE_CAPACITY;
        let mut raw = packet(count);
        let tail = count * 10 - 1;
        let saved = raw[LOGICAL + tail];
        raw[LOGICAL + tail] = 512;
        assert!(group(&raw, count).is_err());
        raw[LOGICAL + tail] = saved;
        let mut grouped = group(&raw, count).unwrap();
        let groups = grouped[3] as usize;
        let last = groups - 1;
        let end = MEMBERS + (grouped[OFFSETS + last] + grouped[LENGTHS + last]) as usize;
        grouped[end - 1] = grouped[end - 2];
        assert!(validate(&grouped, count).is_err());
        raw[LOGICAL + tail] = raw[LOGICAL + tail - 1];
        raw[MEMBERS + tail] = raw[MEMBERS + tail - 1];
        assert!(group(&raw, count).is_err());
    }
    #[test]
    fn actual_moe_descriptors_are_disjoint_from_raw_ring_and_each_other() {
        let root = Handle::new(StreamId { value: 99 }, allocation::ROOT_BYTES);
        let v = Views::new(root.clone()).unwrap();
        let raw_ring = Handle::new(StreamId { value: 99 }, allocation::RAW_RING_BYTES);
        let working = root.offset_end(allocation::ROOT_BYTES - storage::WORKING_BYTES);
        let views = v.handles();
        for (i, h) in views.iter().enumerate() {
            assert!(!storage::overlaps(h, &raw_ring).unwrap());
            assert!(!storage::overlaps(h, &working).unwrap());
            for prior in &views[..i] {
                assert!(!storage::overlaps(h, prior).unwrap());
            }
            assert!(h.offset_start.unwrap() + h.size_in_used() <= allocation::ROOT_BYTES);
        }
        assert_eq!(v.raw.size_in_used(), (RAW_WORDS * 4) as u64);
        assert_eq!(v.map.size_in_used(), (MAP_WORDS * 4) as u64);
        assert_eq!(v.probability.size_in_used(), (PROB_WORDS * 4) as u64);
        assert_eq!(v.status.size_in_used(), (STATUS_WORDS * 4) as u64);
    }
}
