//! Carved prompt owners, borrowed from the three real admitted stage roots.
//! MIT Strata fb58e0 prefill.cpp::carve keeps GR/residual lifetime outside the
//! overlaid attention/MoE region. Matrix images remain below the shared phase
//! region; the asynchronous raw DMA ring has its own separately priced root.
use super::*;
// Persistent view tables, construction vectors and the largest live QSA
// argument bundle are charged together; all GPU bytes remain in old roots.
pub(crate) const HOST_BYTES: usize = 32 * 1024;
// Actual largest rounded input is4096 four-stream rows of2560 values; the
// largest converted head tile is16384x2560. Both need exactly80MiB, not128.
pub(super) const WORKING_BYTES: u64 = (allocation::DEFAULT_ROWS * 4 * 2560 * 2) as u64;
pub(crate) struct Storage {
    working: [Handle; 3],
    moe: groups::Views,
    gr: [Handle; 5], // packed BF16 norm, down, packed BF16 low, up, row scales
    mixed_images: [Handle; 2], // native BF16 and FP16 input images, persistent across each half
    mixed_rows: Option<usize>, // published only by the actual queued GR writer
    outer: [Handle; 12], // R/mixed/blocks/injection/router/shared/embedding, retained PLE tail
    ple: [Handle; 10],
    gdn: [Handle; 7],  // QKV, Z, H, readout, AB raw/controls/closing
    qsa: [Handle; 20], // projections/data, partials, pooled, compact norm/score flags and selections
}
impl Storage {
    pub(crate) fn new(roots: [Handle; 3]) -> Result<Self> {
        Self::at_capacity(roots, groups::CAPACITY)
    }
    fn at_capacity(roots: [Handle; 3], capacity: usize) -> Result<Self> {
        if capacity == 0 || capacity > allocation::DEFAULT_ROWS {
            return Err(crate::strata_prompt::invalid_buffer_length());
        }
        let sizes = [
            allocation::whole_root_bytes(&roots[0])?,
            allocation::whole_root_bytes(&roots[1])?,
            allocation::whole_root_bytes(&roots[2])?,
        ];
        if roots.iter().any(|h| h.stream != roots[0].stream)
            || roots.iter().enumerate().any(|(i, h)| {
                roots[..i].iter().any(|other| {
                    h.clone().binding().memory.allocation_id()
                        == other.clone().binding().memory.allocation_id()
                })
            })
        {
            return Err(crate::strata_prompt::invalid_buffer_length());
        }
        let carve = |root: &Handle, words: &[usize]| -> Result<Vec<Handle>> {
            let root_bytes = allocation::whole_root_bytes(root)?;
            let mut cursor = WORKING_BYTES;
            let mut views = Vec::with_capacity(words.len());
            for &n in words {
                // Native Alloc::take leaves256 bytes and aligns each cell to256.
                let bytes = (n * 4) as u64;
                let end = cursor.checked_add(bytes).ok_or(Error::ArithmeticOverflow)?;
                let next = end.checked_add(511).ok_or(Error::ArithmeticOverflow)? & !255;
                if next > root_bytes {
                    return Err(crate::strata_prompt::invalid_buffer_length());
                }
                views.push(
                    root.clone()
                        .offset_start(cursor)
                        .offset_end(root_bytes - end),
                );
                cursor = next;
            }
            Ok(views)
        };
        let mut persistent_words = [
            5120, 320, 160, 10240, 4, 1280, 1280, 10240, 2560, 2560, 2560, 4, 512, 640, 640, 2560,
            1, 2560,
        ]
        .map(|w| w * capacity)
        .to_vec();
        persistent_words.push(9 * 10240);
        let persistent = carve(&roots[0], &persistent_words)?;
        let gr: [Handle; 5] = std::array::from_fn(|i| persistent[i].clone());
        let mixed_images = std::array::from_fn(|i| persistent[i + 5].clone());
        let outer: [Handle; 12] = std::array::from_fn(|i| persistent[i + 7].clone());
        let ple: [Handle; 10] = carve(
            &roots[1],
            &[2560, 10240, 2560, 10240, 4, 10240, 4, 4, 4, 4].map(|w| w * capacity),
        )?
        .try_into()
        .map_err(|_| Error::InvalidBufferLength)?;
        let gdn: [Handle; 7] = carve(
            &roots[1],
            &[10240, 6144, 10240, 6144, 96, 96, 48].map(|w| w * capacity),
        )?
        .try_into()
        .map_err(|_| Error::InvalidBufferLength)?;
        let qsa: [Handle; 20] = carve(
            &roots[1],
            &[
                12288 * capacity,
                512 * capacity,
                512 * capacity,
                512 * capacity,
                128 * capacity,
                6144 * capacity,
                512 * capacity,
                512 * capacity,
                crate::strata_qsa::CACHE_WORDS * capacity,
                6144 * capacity,
                crate::strata_qsa::PARTIAL_WORDS * 32,
                crate::strata_qsa::STATS_WORDS * 32,
                (crate::qsa_gpu::MAX_CONTEXT / 4) * 128,
                crate::qsa_gpu::MAX_CONTEXT / 4,
                24 * capacity,
                2 * capacity,
                4 * capacity,
                (crate::qsa_gpu::MAX_CONTEXT / 4) * crate::strata_qsa::prompt::SELECT_BATCH,
                (crate::qsa_gpu::MAX_CONTEXT / 4) * crate::strata_qsa::prompt::SELECT_BATCH,
                2051 * capacity,
            ],
        )?
        .try_into()
        .map_err(|_| Error::InvalidBufferLength)?;
        let working = [
            roots[0].clone().offset_end(sizes[0] - WORKING_BYTES),
            roots[1].clone().offset_end(sizes[1] - WORKING_BYTES),
            roots[2].clone(),
        ];
        for (i, a) in persistent.iter().enumerate() {
            if overlaps(a, &working[0])?
                || persistent[..i]
                    .iter()
                    .any(|b| overlaps(a, b).unwrap_or(true))
            {
                return Err(crate::strata_prompt::invalid_buffer_length());
            }
        }
        for (i, a) in ple.iter().enumerate() {
            if overlaps(a, &working[1])? || ple[..i].iter().any(|b| overlaps(a, b).unwrap_or(true))
            {
                return Err(crate::strata_prompt::invalid_buffer_length());
            }
        }
        for (i, a) in gdn.iter().enumerate() {
            if overlaps(a, &working[1])? || gdn[..i].iter().any(|b| overlaps(a, b).unwrap_or(true))
            {
                return Err(crate::strata_prompt::invalid_buffer_length());
            }
        }
        for (i, a) in qsa.iter().enumerate() {
            if overlaps(a, &working[1])? || qsa[..i].iter().any(|b| overlaps(a, b).unwrap_or(true))
            {
                return Err(crate::strata_prompt::invalid_buffer_length());
            }
        }
        let moe = groups::Views::new(roots[1].clone())?;
        // Native Alloc::carve overlays mutually exclusive attention/MoE phases.
        // Raw ring DMA remains independent of every phase and matrix image.
        for h in moe.handles() {
            if overlaps(h, &working[1])? || overlaps(h, &roots[0])? || overlaps(h, &roots[2])? {
                return Err(crate::strata_prompt::invalid_buffer_length());
            }
        }
        Ok(Self {
            working,
            moe,
            gr,
            mixed_images,
            mixed_rows: None,
            outer,
            ple,
            gdn,
            qsa,
        })
    }
    pub(crate) fn moe(&self) -> groups::Views {
        self.moe.clone()
    }
    pub(crate) fn working(&self) -> [Handle; 3] {
        self.working.clone()
    }
    pub(crate) fn gr(&self) -> [Handle; 5] {
        self.gr.clone()
    }
    pub(crate) fn begin_gr(&mut self) -> [Handle; 2] {
        self.mixed_rows = None;
        self.mixed_images.clone()
    }
    pub(crate) fn publish_mixed(&mut self, source: &Handle, count: usize) {
        // Only the original full-prompt mixed view gets automatic reuse. Small
        // arena callers keep their existing explicit F32 conversion path.
        self.mixed_rows = (same_view(source, &self.outer[1])
            && (1..=allocation::DEFAULT_ROWS).contains(&count))
        .then_some(count);
    }
    pub(crate) fn mixed_input(
        &self,
        source: &Handle,
        count: usize,
        first: usize,
        stride: usize,
        bfloat: bool,
    ) -> Option<Handle> {
        (self.mixed_rows == Some(count)
            && first == 0
            && stride == 2560
            && same_view(source, &self.outer[1]))
        .then(|| self.mixed_images[usize::from(!bfloat)].clone())
    }
    pub(crate) fn gr_input(&self, h: &Handle, columns: usize) -> bool {
        match columns {
            10240 => same_view(h, &self.gr[0]),
            320 => same_view(h, &self.gr[2]),
            _ => false,
        }
    }
    pub(crate) fn outer(&self) -> [Handle; 12] {
        self.outer.clone()
    }
    pub(crate) fn ple(&self) -> [Handle; 10] {
        self.ple.clone()
    }
    pub(crate) fn gdn(&self) -> [Handle; 4] {
        std::array::from_fn(|i| self.gdn[i].clone())
    }
    pub(crate) fn gdn_controls(&self) -> [Handle; 3] {
        std::array::from_fn(|i| self.gdn[i + 4].clone())
    }
    pub(crate) fn qsa(&self) -> [Handle; 20] {
        self.qsa.clone()
    }
    pub(crate) fn output(&self, h: &Handle) -> bool {
        self.gr
            .iter()
            .chain(self.mixed_images.iter())
            .chain(self.outer.iter())
            .chain(self.ple.iter())
            .chain(self.gdn.iter())
            .chain(self.qsa.iter())
            .chain(self.moe.handles())
            .any(|v| same_view(h, v))
    }
}
pub(crate) fn same_view(a: &Handle, b: &Handle) -> bool {
    a.stream == b.stream
        && a.clone().binding().memory.allocation_id() == b.clone().binding().memory.allocation_id()
        && a.offset_start.unwrap_or(0) == b.offset_start.unwrap_or(0)
        && a.offset_end.unwrap_or(0) == b.offset_end.unwrap_or(0)
        && a.size_in_used() == b.size_in_used()
}
/// Select one writable descriptor only for an identical original view.
/// Partially overlapping views cannot use either in-place or disjoint kernels.
pub(crate) fn in_place_or_disjoint(source: &Handle, output: &Handle) -> Result<bool> {
    if same_view(source, output) {
        return Ok(true);
    }
    if overlaps(source, output)? {
        return Err(crate::strata_prompt::invalid_buffer_length());
    }
    Ok(false)
}
/// Full allocation identity alone is insufficient once native phase regions
/// are carved. Admit only genuinely disjoint ranges of the same retained root.
pub(crate) fn overlaps(a: &Handle, b: &Handle) -> Result<bool> {
    if a.stream != b.stream {
        return Err(crate::strata_prompt::invalid_buffer_length());
    }
    if a.clone().binding().memory.allocation_id() != b.clone().binding().memory.allocation_id() {
        return Ok(false);
    }
    let af = a.offset_start.unwrap_or(0);
    let bf = b.offset_start.unwrap_or(0);
    let ae = af
        .checked_add(a.size_in_used())
        .ok_or(Error::ArithmeticOverflow)?;
    let be = bf
        .checked_add(b.size_in_used())
        .ok_or(Error::ArithmeticOverflow)?;
    Ok(af < be && bf < ae)
}

#[cfg(test)]
mod owned_layout_tests {
    use super::*;
    use cubecl_common::stream_id::StreamId;
    #[test]
    fn actual_non_power_of_two_parent_sizes_carve_without_virtual_tail_bytes() {
        let stream = StreamId { value: 97 };
        let roots: [Handle; 3] = std::array::from_fn(|i| {
            Handle::new(stream, allocation::ROOT_BYTES - 256 * (i as u64 + 1))
        });
        let full = Storage::at_capacity(roots.clone(), allocation::DEFAULT_ROWS).unwrap();
        let end = |h: &Handle| h.offset_start.unwrap_or(0) + h.size_in_used();
        let required = [
            (full.gr.iter().chain(&full.outer).map(end).max().unwrap() + 511) & !255,
            (full
                .ple
                .iter()
                .chain(&full.gdn)
                .chain(&full.qsa)
                .chain(full.moe.handles())
                .map(end)
                .max()
                .unwrap()
                + 511)
                & !255,
            allocation::RAW_RING_BYTES,
        ];
        let exact: [Handle; 3] = std::array::from_fn(|i| Handle::new(stream, required[i]));
        let layout = Storage::at_capacity(exact, allocation::DEFAULT_ROWS).unwrap();
        for h in layout.gr.iter().chain(&layout.outer) {
            assert!(end(h) <= required[0]);
            assert_eq!(h.size(), required[0]);
        }
        for h in layout.ple.iter().chain(&layout.gdn).chain(&layout.qsa) {
            assert!(end(h) <= required[1]);
            assert_eq!(h.size(), required[1]);
        }
        for h in layout.moe.handles() {
            assert!(end(h) <= required[1]);
            assert_eq!(h.size(), required[1]);
        }
        for root in 0..3 {
            let mut short: [Handle; 3] = std::array::from_fn(|i| Handle::new(stream, required[i]));
            short[root] = Handle::new(stream, required[root] - 256);
            assert!(Storage::at_capacity(short, allocation::DEFAULT_ROWS).is_err());
        }
    }
    #[test]
    fn private_prompt_layout_rejects_aliases_slices_and_foreign_streams() {
        let stream = StreamId { value: 96 };
        let make = || allocation::ROOT_SIZES.map(|n| Handle::new(stream, n));
        let mut roots: [Handle; 3] = make();
        roots[1] = roots[0].clone();
        assert!(Storage::new(roots).is_err());
        let mut roots: [Handle; 3] = make();
        roots[2] = roots[2].clone().offset_end(256);
        assert!(Storage::new(roots).is_err());
        let mut roots: [Handle; 3] = make();
        roots[2] = Handle::new(StreamId { value: 95 }, allocation::ROOT_BYTES);
        assert!(Storage::new(roots).is_err());
        let mut roots: [Handle; 3] = make();
        roots[0] = Handle::new(stream, allocation::ROOT_BYTES + 256);
        assert!(Storage::new(roots).is_err());
    }
    #[test]
    fn native_default_persistent_and_overlaid_views_fit_actual_registered_roots() {
        let stream = StreamId { value: 98 };
        let roots = allocation::ROOT_SIZES.map(|n| Handle::new(stream, n));
        for count in [256, allocation::DEFAULT_ROWS] {
            let layout = Storage::at_capacity(roots.clone(), count).unwrap();
            let mut seen = Vec::new();
            for h in layout.gr.iter().chain(&layout.outer) {
                assert!(
                    h.size_in_used() <= crate::private_arena::MAX_SPAN_BYTES as u64
                        || !cfg!(feature = "strata-intel-core")
                );
                for prior in &seen {
                    assert!(!overlaps(h, prior).unwrap());
                }
                seen.push(h.clone());
            }
            for phase in [&layout.ple[..], &layout.gdn[..], &layout.qsa[..]] {
                for (i, h) in phase.iter().enumerate() {
                    assert!(!overlaps(h, &layout.working[1]).unwrap());
                    for other in &phase[..i] {
                        assert!(!overlaps(h, other).unwrap());
                    }
                }
            }
            // Actual packed inputs, full target residual and four-residual MTP
            // outputs must fit real views, independently of MAX8 verifier roots.
            assert!((count * 10240 * 4) as u64 <= layout.outer[0].size_in_used());
            assert!((count * 10240 * 4) as u64 <= layout.ple[3].size_in_used());
            assert!((count * 4 * 2560 * 2) as u64 <= layout.working[0].size_in_used());
            for h in layout.moe.handles() {
                assert!(layout.output(h));
            }
            assert_eq!(layout.working[0].size_in_used(), WORKING_BYTES);
            assert_eq!(layout.working[1].size_in_used(), WORKING_BYTES);
        }
    }
}

#[cfg(test)]
mod overlay_tests {
    use super::*;
    use crate::strata_queue_tests::{Access, NodeAccess, Planner};
    use cubecl_common::stream_id::StreamId;
    fn roots() -> [Handle; 3] {
        allocation::ROOT_SIZES.map(|n| Handle::new(StreamId { value: 712 }, n))
    }
    #[test]
    fn actual_phase_overlay_preserves_all_simultaneously_live_ranges_and_prices_real_raw_root() {
        let roots = roots();
        let s = Storage::new(roots.clone()).unwrap();
        assert_eq!(
            roots.iter().map(Handle::size_in_used).sum::<u64>(),
            allocation::MAIN_BYTES
        );
        assert_eq!(allocation::MAIN_BYTES, 2080 * 1024 * 1024);
        assert_eq!(roots[2].size_in_used(), 32 * 1024 * 1024);
        assert_eq!(WORKING_BYTES, 80 * 1024 * 1024);
        assert_eq!(groups::GU_BYTES, 20 * 1024 * 1024);
        assert_eq!(groups::PART_BYTES, 400 * 1024 * 1024);
        assert_eq!(s.qsa[17].size_in_used(), 8 * 1024 * 1024);
        assert_eq!(s.qsa[18].size_in_used(), 8 * 1024 * 1024);
        assert!((16384 * 2560 * 2) as u64 <= WORKING_BYTES);
        for h in s.moe.handles() {
            assert!(!overlaps(h, &roots[0]).unwrap());
            assert!(!overlaps(h, &roots[2]).unwrap());
            assert!(!overlaps(h, &s.working[1]).unwrap());
            assert!(s.qsa.iter().any(|a| overlaps(h, a).unwrap()));
        }
        for phase in [
            &s.ple[..],
            &s.gdn[..],
            &s.qsa[..],
            &s.moe.handles().map(Clone::clone)[..],
        ] {
            for (i, h) in phase.iter().enumerate() {
                assert!(!overlaps(h, &roots[0]).unwrap());
                assert!(!overlaps(h, &roots[2]).unwrap());
                for old in &phase[..i] {
                    assert!(!overlaps(h, old).unwrap());
                }
            }
        }
    }
    #[test]
    fn original_queue_planner_sees_actual_phase_aliases_and_independent_raw_dma() {
        let roots = roots();
        let s = Storage::new(roots.clone()).unwrap();
        let access = |h: &Handle, write| {
            let id = h.clone().binding().memory.allocation_id();
            let buffer = roots
                .iter()
                .position(|r| r.clone().binding().memory.allocation_id() == id)
                .unwrap() as u64
                + 1;
            Access {
                buffer,
                first: h.offset_start.unwrap_or(0),
                bytes: h.size_in_used(),
                write,
            }
        };
        let mut prefix = Planner::new(1, false).unwrap();
        prefix
            .push(&NodeAccess {
                copy: false,
                release_after: false,
                accesses: vec![access(&s.qsa[0], true)],
            })
            .unwrap();
        let prefix = prefix.finish().unwrap();
        assert!(prefix.conflicts(&[access(&s.moe.partial, true)]).unwrap());
        assert!(!prefix.conflicts(&[access(&roots[2], true)]).unwrap());
        let nodes = [
            NodeAccess {
                copy: false,
                release_after: false,
                accesses: vec![access(&s.qsa[0], true)],
            },
            NodeAccess {
                copy: true,
                release_after: false,
                accesses: vec![access(&roots[2], true)],
            },
            NodeAccess {
                copy: false,
                release_after: true,
                accesses: vec![access(&roots[2], false), access(&s.moe.gu, true)],
            },
            NodeAccess {
                copy: true,
                release_after: false,
                accesses: vec![access(&roots[2], true)],
            },
            NodeAccess {
                copy: false,
                release_after: false,
                accesses: vec![access(&s.moe.gu, false), access(&s.moe.partial, true)],
            },
            NodeAccess {
                copy: false,
                release_after: false,
                accesses: vec![access(&s.moe.partial, false)],
            },
            NodeAccess {
                copy: false,
                release_after: false,
                accesses: vec![access(&s.qsa[0], true)],
            },
        ];
        let mut p = Planner::new(nodes.len(), false).unwrap();
        for n in &nodes {
            p.push(n).unwrap();
        }
        let p = p.finish().unwrap();
        let seg = |i: usize| &p.segments[p.node_segments[i]];
        assert!(seg(1).wait < seg(0).signal);
        assert_eq!(seg(2).wait, seg(1).signal);
        assert_eq!(seg(3).wait, seg(2).signal);
        assert_ne!(p.node_segments[2], p.node_segments[4]);
        assert_eq!(p.node_segments[4], p.node_segments[6]); // original compute order retains the phase switch
    }
}

#[cfg(test)]
mod packed_gr_layout_tests {
    use super::*;
    use cubecl_common::stream_id::StreamId;
    #[test]
    fn original_gr_images_scales_and_outputs_have_disjoint_exact_native_extents() {
        let stream = StreamId { value: 919 };
        let roots = allocation::ROOT_SIZES.map(|n| Handle::new(stream, n));
        let s = Storage::new(roots).unwrap();
        let lengths = [5120, 320, 160, 10240, 4];
        for (h, words) in s.gr.iter().zip(lengths) {
            assert_eq!(
                h.size_in_used(),
                (words * allocation::DEFAULT_ROWS * 4) as u64
            );
        }
        let old_words = 10240 + 320 + 320 + 10240;
        assert_eq!(
            (old_words - lengths.iter().sum::<usize>()) * allocation::DEFAULT_ROWS * 4,
            86_441_984
        );
        assert!(s.gr_input(&s.gr[0], 10240));
        assert!(s.gr_input(&s.gr[2], 320));
        for h in
            s.gr.iter()
                .chain(&s.outer)
                .chain(&s.ple)
                .chain(&s.gdn)
                .chain(&s.qsa)
        {
            assert_eq!(s.gr_input(h, 10240), same_view(h, &s.gr[0]));
            assert_eq!(s.gr_input(h, 320), same_view(h, &s.gr[2]));
            assert!(!s.gr_input(h, 2560));
        }
        assert!(!s.gr_input(&s.gr[0].clone().offset_end(256), 10240));
        assert!(!s.gr_input(&Handle::new(stream, s.gr[0].size_in_used()), 10240));
        for count in [1, 2, 7, 8, 9, 255, 256, 257, 4095, 4096] {
            assert!(count * 10240 * 2 <= s.gr[0].size_in_used() as usize);
            assert!(count * 320 * 2 <= s.gr[2].size_in_used() as usize);
            assert!(count * 4 * 4 <= s.gr[4].size_in_used() as usize);
        }
        for (i, h) in s.gr.iter().enumerate() {
            for prior in s.gr[..i].iter().chain(&s.outer) {
                assert!(!overlaps(h, prior).unwrap());
            }
        }
    }
}

#[cfg(test)]
mod mixed_image_tests {
    use super::*;
    use cubecl_common::stream_id::StreamId;
    fn storage() -> Storage {
        Storage::new(allocation::ROOT_SIZES.map(|n| Handle::new(StreamId { value: 921 }, n)))
            .unwrap()
    }
    #[test]
    fn mixed_image_selection_requires_the_original_source_and_current_writer_geometry() {
        let mut s = storage();
        let source = s.outer[1].clone();
        assert!(s.mixed_input(&source, 4096, 0, 2560, true).is_none());
        let images = s.begin_gr();
        assert!(s.mixed_input(&source, 4096, 0, 2560, false).is_none());
        s.publish_mixed(&source, 4096);
        for (kind, image) in [(true, &images[0]), (false, &images[1])] {
            assert!(same_view(
                &s.mixed_input(&source, 4096, 0, 2560, kind).unwrap(),
                image
            ));
            for (count, first, stride) in [
                (0, 0, 2560),
                (1, 0, 2560),
                (4095, 0, 2560),
                (4096, 1, 2560),
                (4096, 0, 2561),
                (4097, 0, 2560),
            ] {
                assert!(s.mixed_input(&source, count, first, stride, kind).is_none());
            }
            for h in [
                &s.outer[0],
                &s.gr[0],
                &images[0],
                &images[1],
                &source.clone().offset_end(256),
                &Handle::new(source.stream, source.size_in_used()),
            ] {
                assert!(s.mixed_input(h, 4096, 0, 2560, kind).is_none());
            }
        }
        s.begin_gr();
        assert!(s.mixed_input(&source, 4096, 0, 2560, true).is_none());
        s.publish_mixed(&source, 1);
        assert!(s.mixed_input(&source, 1, 0, 2560, true).is_some());
        assert!(s.mixed_input(&source, 4096, 0, 2560, true).is_none());
        s.publish_mixed(&s.outer[0].clone(), 1);
        assert!(s.mixed_input(&source, 1, 0, 2560, true).is_none());
        for count in [0, allocation::DEFAULT_ROWS + 1] {
            s.publish_mixed(&source, count);
            assert!(s.mixed_input(&source, count, 0, 2560, true).is_none());
        }
    }
    #[test]
    fn mixed_images_are_retained_disjoint_native_views_inside_the_unchanged_admitted_parent() {
        let s = storage();
        assert_eq!(
            allocation::ROOT_SIZES,
            [1024 * 1024 * 1024, 1024 * 1024 * 1024, 32 * 1024 * 1024]
        );
        assert_eq!(
            s.mixed_images.iter().map(Handle::size_in_used).sum::<u64>(),
            40 * 1024 * 1024
        );
        for (i, image) in s.mixed_images.iter().enumerate() {
            assert!(s.output(image));
            assert_eq!(
                image.size_in_used(),
                (allocation::DEFAULT_ROWS * 2560 * 2) as u64
            );
            for other in
                s.gr.iter()
                    .chain(&s.outer)
                    .chain(&s.mixed_images[..i])
                    .chain(&s.working)
            {
                assert!(!overlaps(image, other).unwrap());
            }
            let access = |h: &Handle, write| crate::strata_queue_tests::Access {
                buffer: 1,
                first: h.offset_start.unwrap_or(0),
                bytes: h.size_in_used(),
                write,
            };
            let mut reader = crate::strata_queue_tests::Planner::new(1, false).unwrap();
            reader
                .push(&crate::strata_queue_tests::NodeAccess {
                    copy: false,
                    release_after: false,
                    accesses: vec![access(image, false)],
                })
                .unwrap();
            let reader = reader.finish().unwrap();
            assert!(reader.conflicts(&[access(image, true)]).unwrap());
            assert!(!reader.conflicts(&[access(&s.outer[1], true)]).unwrap());
            assert!(!reader.conflicts(&[access(&s.gr[0], true)]).unwrap());
        }
    }
}

#[cfg(test)]
mod native_ple_reuse_tests {
    use super::*;
    use cubecl_common::stream_id::StreamId;
    #[test]
    fn native_ple_unique_regions_reuse_query_after_gate_and_preserve_teacher_extents() {
        let s =
            Storage::new(allocation::ROOT_SIZES.map(|n| Handle::new(StreamId { value: 923 }, n)))
                .unwrap();
        assert_eq!(s.ple.len(), 10);
        for (h, words) in s
            .ple
            .iter()
            .zip([2560, 10240, 2560, 10240, 4, 10240, 4, 4, 4, 4])
        {
            assert_eq!(
                h.size_in_used(),
                (words * allocation::DEFAULT_ROWS * 4) as u64
            );
        }
        assert_eq!(2 * 10240 * allocation::DEFAULT_ROWS * 4, 320 * 1024 * 1024);
        for (i, a) in s.ple.iter().enumerate() {
            assert!(s.output(a));
            for b in &s.ple[..i] {
                assert!(!overlaps(a, b).unwrap());
            }
        }
        // The same query region is read by the gate then written by conv norm.
        // It must not conflict with the separately retained unnormalized gate.
        let access = |i: usize, write| crate::strata_queue_tests::Access {
            buffer: 1,
            first: s.ple[i].offset_start.unwrap_or(0),
            bytes: s.ple[i].size_in_used(),
            write,
        };
        let mut p = crate::strata_queue_tests::Planner::new(1, false).unwrap();
        p.push(&crate::strata_queue_tests::NodeAccess {
            copy: false,
            release_after: false,
            accesses: vec![access(1, false), access(3, false)],
        })
        .unwrap();
        let p = p.finish().unwrap();
        assert!(p.conflicts(&[access(1, true)]).unwrap());
        assert!(p.conflicts(&[access(3, true)]).unwrap());
        assert!(!p.conflicts(&[access(5, true)]).unwrap());
        for count in [1, 8, 9, 256, 4095, 4096] {
            // MTP teacher borrows these SAME original idle PLE views after the
            // target finishes. Four stream rows plus disjoint token/status data.
            assert!(count * 4 * 2560 * 4 <= s.ple[3].size_in_used() as usize);
            assert!(count * 4 <= s.ple[8].size_in_used() as usize);
            assert!(count * 4 * 4 <= s.ple[9].size_in_used() as usize);
        }
    }
}

#[cfg(test)]
mod residual_owner_tests {
    use super::*;
    use cubecl_common::stream_id::StreamId;
    #[test]
    fn original_view_selection_rejects_partial_aliases_and_foreign_streams() {
        let root = Handle::new(StreamId { value: 925 }, 4096);
        let first = root.clone().offset_end(2048);
        let next = root.clone().offset_start(2048);
        assert!(in_place_or_disjoint(&first, &first.clone()).unwrap());
        assert!(!in_place_or_disjoint(&first, &next).unwrap());
        assert!(
            in_place_or_disjoint(&first, &root.clone().offset_start(256).offset_end(2048)).is_err()
        );
        assert!(in_place_or_disjoint(&first, &first.clone().offset_end(256)).is_err());
        assert!(!in_place_or_disjoint(&first, &Handle::new(root.stream, 2048)).unwrap());
        assert!(in_place_or_disjoint(&first, &Handle::new(StreamId { value: 926 }, 2048)).is_err());
    }
    #[test]
    fn sole_target_carry_and_idle_teacher_residual_have_original_disjoint_full_chunk_owners() {
        let s =
            Storage::new(allocation::ROOT_SIZES.map(|n| Handle::new(StreamId { value: 927 }, n)))
                .unwrap();
        assert_eq!(s.outer.len(), 12);
        let words = [10240, 2560, 2560, 2560, 4, 512, 640, 640, 2560, 1, 2560];
        for (h, w) in s.outer.iter().zip(words) {
            assert_eq!(h.size_in_used(), (w * allocation::DEFAULT_ROWS * 4) as u64);
        }
        // Removing R1 removes160MiB logical payload, not physical root bytes.
        assert_eq!(10240 * allocation::DEFAULT_ROWS * 4, 160 * 1024 * 1024);
        let carry = &s.outer[0];
        let residual = &s.ple[1];
        let hn = &s.ple[3];
        for count in [1, 8, 9, 256, 4095, 4096] {
            for h in [carry, residual, hn] {
                assert!(count * 10240 * 4 <= h.size_in_used() as usize);
            }
        }
        for h in [residual, hn, &s.ple[8], &s.ple[9]] {
            assert!(!overlaps(carry, h).unwrap());
        }
        assert!(!overlaps(residual, hn).unwrap());
        for token_status in [&s.ple[8], &s.ple[9]] {
            assert!(!overlaps(residual, token_status).unwrap());
            assert!(!overlaps(hn, token_status).unwrap());
        }
        // A retained target R writer must precede its teacher reader; writing
        // independent PLE scratch must not conflict with that carry reader.
        use crate::strata_queue_tests::{Access, NodeAccess, Planner};
        let access = |h: &Handle, buffer, write| Access {
            buffer,
            first: h.offset_start.unwrap_or(0),
            bytes: h.size_in_used(),
            write,
        };
        let mut p = Planner::new(1, false).unwrap();
        p.push(&NodeAccess {
            copy: false,
            release_after: false,
            accesses: vec![access(carry, 1, false)],
        })
        .unwrap();
        let p = p.finish().unwrap();
        assert!(p.conflicts(&[access(carry, 1, true)]).unwrap());
        assert!(!p.conflicts(&[access(residual, 2, true)]).unwrap());
    }
}
