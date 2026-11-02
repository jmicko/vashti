//! MIT Strata Intel prompt grouping: logical expert order, stable token/rank
//! rows, one bounded host grouping per layer. All maps remain request-private.
use super::*;
use zeroize::Zeroizing;
// Raw packet/readback lease, grouped map/upload and validation scratch peak
// below2MiB at4096 rows; actual arena bytes are carved from priced owners.
pub(crate) const HOST_BYTES: usize = 2 * 1024 * 1024;
pub(crate) const RUNTIME_BYTES: u64 = 16 * 1024 * 1024;
pub(crate) const CAPACITY: usize = allocation::DEFAULT_ROWS;
// The vendored native expert grammar carries its own copy of the row count.
const _: () = assert!(
    CAPACITY == cubecl_runtime::server::native_expert_grammar::KUNO_PROMPT_ROWS
);
const _: () = assert!(
    allocation::ROOT_SIZES[0] == cubecl_runtime::server::native_expert_grammar::KUNO_PERSISTENT_ROOT_BYTES
        && allocation::ROOT_SIZES[1] == cubecl_runtime::server::native_expert_grammar::KUNO_PHASE_ROOT_BYTES
);
/// Whole-layer padded route rows (every route plus <=31 pad rows per expert).
pub(crate) const PADDED_ROWS: usize = ROUTES + 512 * 31;
// Separate prompt geometry from MAX8 verifier/decode and hardware workgroups.
pub(crate) const ROUTE_CAPACITY: usize = allocation::DEFAULT_ROWS;
pub(crate) const ROUTES: usize = ROUTE_CAPACITY * 10;
pub(crate) const LOGICAL: usize = 65536;
pub(crate) const RAW_WORDS: usize = 131072;
pub(crate) const MAP_WORDS: usize = 65536;
pub(crate) const PROB_WORDS: usize = 65536;
pub(crate) const STATUS_WORDS: usize = 131072;
/// Whole-layer pad prefixes follow the member list (43008 at 4096 rows).
pub(crate) const LAYER_PREFIX: usize = (MEMBERS + ROUTES).next_multiple_of(512);
pub(crate) const CAP_U32: u32 = CAPACITY as u32;
pub(crate) const LAYER_PADDED_ROWS: usize = PADDED_ROWS;
// Layer gather dispatches one workgroup row per padded row (grid Y <= 65535).
const _: () = assert!(LAYER_PADDED_ROWS <= 65535);
pub(crate) const LAYER_X_BYTES: u64 = (LAYER_PADDED_ROWS * 2560 * 2) as u64;
/// Int8 activations (Strata's numerics): per token 2560 codes (640 words) then
/// 80 f32 scales; per small-cohort row 640 hidden codes (160 words) + 20 scales.
pub(crate) const XQ_WORDS_PER_ROW: usize = 720;
pub(crate) const HQ_WORDS_PER_ROW: usize = 180;
pub(crate) const XQ_BYTES: u64 = (CAPACITY * XQ_WORDS_PER_ROW * 4) as u64;
pub(crate) const HQ_BYTES: u64 = (CAPACITY * HQ_WORDS_PER_ROW * 4) as u64;
pub(crate) const LAYER_Q_BYTES: u64 = XQ_BYTES + HQ_BYTES;
pub(crate) const GU_BYTES: u64 = (CAPACITY * 1280 * 4) as u64;
pub(crate) const PART_BYTES: u64 = (ROUTES * 2560 * 4) as u64;
/// One compiled prompt ABI for actual Rust views and every native shader.
/// Decode/MAX8 packet dimensions and expert address-table sizes stay separate.
pub(crate) fn shader_prefix() -> String {
    format!(
        "const PROMPT_MAX_ROWS:u32={CAPACITY}u;const PROMPT_PADDED_ROWS:u32={PADDED_ROWS}u;const PROMPT_LAYER_PREFIX:u32={LAYER_PREFIX}u;const PROMPT_ROUTES:u32={ROUTES}u;const PROMPT_LOGICAL:u32={LOGICAL}u;const PROMPT_RAW_WORDS:u32={RAW_WORDS}u;const PROMPT_MAP_WORDS:u32={MAP_WORDS}u;const PROMPT_PROB_WORDS:u32={PROB_WORDS}u;const PROMPT_STATUS_WORDS:u32={STATUS_WORDS}u;const PROMPT_PART_WORDS:u32={}u;const PROMPT_WORKING_WORDS:u32={}u;\n",
        PART_BYTES / 4,
        storage::WORKING_BYTES / 4
    )
}
const MEMBERS: usize = 1540;
const LENGTHS: usize = 516;
const OFFSETS: usize = 1028;
pub(crate) fn group(raw: &[u32], count: usize) -> Result<Zeroizing<Vec<u32>>> {
    group_with_order(raw, count, false)
}
/// Production whole-prefill policy. Original numerical fixtures and decode
/// retain logical order; the SAME authenticated raw packet builds all prefixes,
/// members and downstream inverse before expert staging can observe this map.
/// `streamed(cell)`: whether the layer's expert cell streams from HOST; it
/// only orders groups (copy/compute overlap) and never changes the math.
pub(crate) fn group_for_prefill(raw: &[u32], count: usize, streamed: &dyn Fn(u32) -> bool) -> Result<Zeroizing<Vec<u32>>> {
    let map = group_with_streamed(raw, count, cfg!(feature = "strata-cohort-row-order"), Some(streamed))?;
    if cfg!(feature = "strata-cohort-row-order") {
        crate::strata_prefill_profile::count("expert_row_order_layers", 1);
        crate::strata_prefill_profile::count("expert_row_order_groups", u64::from(map[3]));
    }
    Ok(map)
}
const ORDER_BYTES: usize = 512 * size_of::<u16>();
fn group_with_order(raw: &[u32], count: usize, row_order: bool) -> Result<Zeroizing<Vec<u32>>> {
    group_with_streamed(raw, count, row_order, None)
}
fn group_with_streamed(raw: &[u32], count: usize, row_order: bool, streamed: Option<&dyn Fn(u32) -> bool>) -> Result<Zeroizing<Vec<u32>>> {
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
    // Exactly1KiB, in place and erased on every return. Sorting allocates no
    // extra heap and does not retain any cross-request expert-use history.
    let mut order = Zeroizing::new(std::array::from_fn::<u16, 512, _>(|i| i as u16));
    if row_order {
        order.sort_unstable_by_key(|&id| {
            let n = lengths[usize::from(id)];
            (if n == 0 { usize::MAX } else { n }, id)
        });
    }
    if let (true, Some(streamed)) = (row_order, streamed) {
        let mut host = Zeroizing::new([false; 512]);
        for l in 0..512 {
            host[l] = lengths[l] != 0 && streamed(cells[l]);
        }
        overlap_order(&mut order, &lengths, &host);
    }
    let mut n = 0;
    let mut groups = 0;
    for &logical in order.iter() {
        let logical = usize::from(logical);
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
    #[cfg(feature="strata-layer-gather")]
    {
        let mut padded=0usize;
        for g in 0..groups {map[LAYER_PREFIX+g]=padded as u32;padded+=(map[LENGTHS+g] as usize).div_ceil(32)*32;}
        map[LAYER_PREFIX+512]=padded as u32;
    }
    validate(&map, count)?;
    Ok(map)
}
/// Order groups so the copy queue (HOST-streamed experts) and the compute
/// queue both stay busy: each tier is cut into cohort-sized runs that are
/// all-streamed or all-resident (ascending sizes kept), and runs are merged
/// greedily so cumulative copy time stays just ahead of cumulative compute.
/// Unit formation, outputs and arithmetic are unchanged (outputs are reduced
/// in route order). `order` must be sorted by length with unused experts last.
fn overlap_order(order: &mut [u16; 512], lengths: &[usize; 512], host: &[bool; 512]) {
    // Measured on B70 (layer 24, 4096 tokens): ~0.137 ms per streamed expert
    // copy; compute ~0.03 ms per group + 0.0004 ms per route, plus ~0.05 ms
    // FP16 dequantization for regular/large groups. Units: microseconds.
    let copy_us = |id: u16| if host[usize::from(id)] { 137 } else { 0 };
    let compute_us = |id: u16| {
        let n = lengths[usize::from(id)];
        30 + n * 4 / 10 + if n >= 32 { 50 } else { 0 }
    };
    let used = order.iter().take_while(|&&id| lengths[usize::from(id)] != 0).count();
    let sorted = *order;
    let tier = |id: u16| match lengths[usize::from(id)] { n if n < 32 => 0, n if n < 256 => 1, _ => 2 };
    let mut runs: Vec<Vec<u16>> = Vec::new();
    for t in 0..3 {
        for streamed in [true, false] {
            let ids: Vec<u16> = sorted[..used].iter().copied().filter(|&id| tier(id) == t && host[usize::from(id)] == streamed).collect();
            let mut i = 0;
            while i < ids.len() {
                let mut end = i + 1;
                if t < 2 {
                    end = (i + 8).min(ids.len());
                } else {
                    let mut rows = lengths[usize::from(ids[i])].div_ceil(32) * 32;
                    while end < ids.len() && end < i + 8 {
                        let next = lengths[usize::from(ids[end])].div_ceil(32) * 32;
                        if rows + next > CAPACITY { break; }
                        rows += next;
                        end += 1;
                    }
                }
                runs.push(ids[i..end].to_vec());
                i = end;
            }
        }
    }
    let cost = |run: &[u16]| (run.iter().map(|&id| copy_us(id)).sum::<usize>(), run.iter().map(|&id| compute_us(id)).sum::<usize>());
    // Copy-heavy runs first in each pick, compute-heavy runs when copies lead.
    let (mut copy_heavy, mut compute_heavy): (Vec<_>, Vec<_>) = runs.into_iter().partition(|r| { let (c, k) = cost(r); c > k });
    copy_heavy.reverse();
    compute_heavy.reverse();
    // Proportional merge: both queues finish together; copies lead by about
    // one raw bank (~2 ms) so compute rarely waits for the copy it needs.
    let total_copy = copy_heavy.iter().chain(&compute_heavy).map(|r| cost(r).0).sum::<usize>().max(1);
    let total_compute = copy_heavy.iter().chain(&compute_heavy).map(|r| cost(r).1).sum::<usize>().max(1);
    let lead = 2000usize;
    let (mut copied, mut computed, mut at) = (0usize, 0usize, 0usize);
    let mut out = [0u16; 512];
    while !copy_heavy.is_empty() || !compute_heavy.is_empty() {
        // copied/total_copy <= (computed + lead)/total_compute, cross-multiplied.
        let pick_copy = compute_heavy.is_empty()
            || (!copy_heavy.is_empty() && copied * total_compute <= (computed + lead) * total_copy);
        let run = if pick_copy { copy_heavy.pop().unwrap() } else { compute_heavy.pop().unwrap() };
        let (c, k) = cost(&run);
        copied += c;
        computed += k;
        out[at..at + run.len()].copy_from_slice(&run);
        at += run.len();
    }
    order[..used].copy_from_slice(&out[..used]);
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
#[cfg(feature="strata-layer-gather")]
pub(crate) fn validate_layer(map:&[u32],count:usize)->Result<(usize,usize)> {
 let groups=validate(map,count)?;let mut padded=0usize;
 for g in 0..groups {
  if map[LAYER_PREFIX+g] as usize!=padded {return Err(Error::InvalidBufferLength);}
  padded+=(map[LENGTHS+g] as usize).div_ceil(32)*32;
 }
 if padded>LAYER_PADDED_ROWS || map[LAYER_PREFIX+512] as usize!=padded || map[LAYER_PREFIX+groups..LAYER_PREFIX+512].iter().any(|&v|v!=0) {return Err(Error::InvalidBufferLength);}
 Ok((groups,padded))
}
/// Authenticated inverse of the original, fully checked grouped permutation.
/// RAW is dead after the audited router readback; its existing private view
/// holds this prefix until the layer's ordered consumer has finished.
pub(crate) fn ordered_inverse(map: &[u32], count: usize) -> Result<Zeroizing<Vec<u32>>> {
    validate(map, count)?;
    let total = count * 10;
    let mut inverse = Zeroizing::new(vec![u32::MAX; total]);
    for ordinal in 0..total {
        let route = map[MEMBERS + ordinal] as usize;
        if route >= total || inverse[route] != u32::MAX {
            return Err(crate::strata_prompt::invalid_buffer_length());
        }
        inverse[route] = ordinal as u32;
    }
    for (route, &ordinal) in inverse.iter().enumerate() {
        if ordinal as usize >= total || map[MEMBERS + ordinal as usize] as usize != route {
            return Err(crate::strata_prompt::invalid_buffer_length());
        }
    }
    Ok(inverse)
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
    fn ordered_inverse_authenticates_every_original_route_and_rejects_corruption() {
        for count in [1, 9, 148, 257, ROUTE_CAPACITY] {
            let mut map = group(&packet(count), count).unwrap();
            let inverse = ordered_inverse(&map, count).unwrap();
            assert_eq!(inverse.len(), count * 10);
            assert!(inverse.len() <= RAW_WORDS);
            for ordinal in 0..count * 10 {
                assert_eq!(inverse[map[MEMBERS + ordinal] as usize], ordinal as u32);
            }
            map[MEMBERS + count * 10 - 1] = map[MEMBERS];
            assert!(ordered_inverse(&map, count).is_err());
        }
        let map = group(&packet(9), 9).unwrap();
        assert!(ordered_inverse(&map, 8).is_err());
        assert!(ordered_inverse(&map[..MAP_WORDS - 1], 9).is_err());
    }
    #[cfg(feature="strata-layer-gather")]
    #[test]
    fn layer_gather_padded_prefix_preserves_routes_inverse_and_rejects_corruption() {
        for count in [1,149,4096] {
            let original=packet(count);
            for sorted in [false,true] {
                let map=group_with_order(&original,count,sorted).unwrap();
                let (n,padded)=validate_layer(&map,count).unwrap();
                let inverse=ordered_inverse(&map,count).unwrap();
                let mut cursor=0;
                for group in 0..n {
                    assert_eq!(map[LAYER_PREFIX+group] as usize,cursor);
                    let members=map[LENGTHS+group] as usize;
                    for member in 0..members {
                        let ordinal=map[OFFSETS+group] as usize+member;
                        let route=map[MEMBERS+ordinal] as usize;
                        assert_eq!(inverse[route] as usize,ordinal);
                        assert_eq!(original[MEMBERS+route],map[4+group]);
                    }
                    cursor+=members.div_ceil(32)*32;
                }
                assert_eq!(cursor,padded);assert!(padded<=LAYER_PADDED_ROWS);
                for index in [LAYER_PREFIX,LAYER_PREFIX+n-1,LAYER_PREFIX+512] {
                    let mut bad=map.clone();bad[index]=u32::MAX;
                    assert!(validate_layer(&bad,count).is_err());
                }
                if n<512 {let mut bad=map.clone();bad[LAYER_PREFIX+n]=32;assert!(validate_layer(&bad,count).is_err());}
            }
        }
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

#[cfg(all(test, feature = "strata-cohort-row-order"))]
#[path = "strata_prompt_group_order_tests.rs"]
pub(crate) mod row_order_tests;
