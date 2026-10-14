//! Carved prompt owners, borrowed from the three real admitted stage roots.
//! MIT Strata fb58e0 prefill.cpp::carve keeps GR/residual lifetime outside the
//! overlaid attention/MoE region. Here the working images are bounded to the
//! lower64 MiB and the connected GR intermediates occupy distinct upper views.
use super::*;
// Persistent view tables, construction vectors and the largest live QSA
// argument bundle are charged together; all GPU bytes remain in old roots.
pub(crate) const HOST_BYTES: usize = 32 * 1024;
const HALF: u64 = 64 * 1024 * 1024;
const ROOT: u64 = allocation::ROOT_BYTES;
pub(crate) struct Storage {
    working: [Handle; 3],
    moe: groups::Views,
    gr: [Handle; 4],     // normalized, down, low, up
    outer: [Handle; 13], // R/mixed/blocks/injection/router/shared/embedding, retained PLE tail
    ple: [Handle; 12],
    gdn: [Handle; 7],  // QKV, Z, H, readout, AB raw/controls/closing
    qsa: [Handle; 20], // projections/data, partials, pooled, compact norm/score flags and selections
}
impl Storage {
    pub(crate) fn new(roots: [Handle; 3]) -> Result<Self> {
        Self::at_capacity(roots, groups::CAPACITY)
    }
    fn at_capacity(roots: [Handle; 3], capacity: usize) -> Result<Self> {
        if capacity == 0 || capacity > allocation::DEFAULT_ROWS {
            return Err(Error::InvalidBufferLength);
        }
        if roots.iter().any(|h| {
            h.size_in_used() != ROOT
                || h.offset_start.unwrap_or(0) != 0
                || h.offset_end.unwrap_or(0) != 0
                || h.stream != roots[0].stream
        }) {
            return Err(Error::InvalidBufferLength);
        }
        let carve = |root: &Handle, words: &[usize]| -> Result<Vec<Handle>> {
            let mut cursor = HALF;
            let mut views = Vec::with_capacity(words.len());
            for &n in words {
                // Native Alloc::take leaves256 bytes and aligns each cell to256.
                let bytes = (n * 4) as u64;
                let end = cursor.checked_add(bytes).ok_or(Error::ArithmeticOverflow)?;
                let next = end.checked_add(511).ok_or(Error::ArithmeticOverflow)? & !255;
                if next > ROOT {
                    return Err(Error::InvalidBufferLength);
                }
                views.push(root.clone().offset_start(cursor).offset_end(ROOT - end));
                cursor = next;
            }
            Ok(views)
        };
        let mut persistent_words = [
            10240, 320, 320, 10240, 10240, 10240, 2560, 2560, 2560, 4, 512, 640, 640, 2560, 1, 2560,
        ]
        .map(|w| w * capacity)
        .to_vec();
        persistent_words.push(9 * 10240);
        let persistent = carve(&roots[0], &persistent_words)?;
        let gr: [Handle; 4] = std::array::from_fn(|i| persistent[i].clone());
        let outer: [Handle; 13] = std::array::from_fn(|i| persistent[i + 4].clone());
        let ple: [Handle; 12] = carve(
            &roots[1],
            &[2560, 10240, 2560, 10240, 10240, 4, 10240, 10240, 4, 4, 4, 4].map(|w| w * capacity),
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
                640 * capacity,
                6144 * capacity,
                crate::strata_qsa::PARTIAL_WORDS * 32,
                crate::strata_qsa::STATS_WORDS * 32,
                1024 * 128,
                1024,
                24 * capacity,
                2 * capacity,
                4 * capacity,
                1024 * capacity,
                1024 * capacity,
                2051 * capacity,
            ],
        )?
        .try_into()
        .map_err(|_| Error::InvalidBufferLength)?;
        let working = [
            roots[0].clone().offset_end(ROOT - HALF),
            roots[1].clone().offset_end(ROOT - HALF),
            roots[2].clone(),
        ];
        for (i, a) in persistent.iter().enumerate() {
            if overlaps(a, &working[0])?
                || persistent[..i]
                    .iter()
                    .any(|b| overlaps(a, b).unwrap_or(true))
            {
                return Err(Error::InvalidBufferLength);
            }
        }
        for (i, a) in ple.iter().enumerate() {
            if overlaps(a, &working[1])? || ple[..i].iter().any(|b| overlaps(a, b).unwrap_or(true))
            {
                return Err(Error::InvalidBufferLength);
            }
        }
        for (i, a) in gdn.iter().enumerate() {
            if overlaps(a, &working[1])? || gdn[..i].iter().any(|b| overlaps(a, b).unwrap_or(true))
            {
                return Err(Error::InvalidBufferLength);
            }
        }
        for (i, a) in qsa.iter().enumerate() {
            if overlaps(a, &working[1])? || qsa[..i].iter().any(|b| overlaps(a, b).unwrap_or(true))
            {
                return Err(Error::InvalidBufferLength);
            }
        }
        let moe = groups::Views::new(roots[2].clone())?;
        Ok(Self {
            working,
            moe,
            gr,
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
    pub(crate) fn gr(&self) -> [Handle; 4] {
        self.gr.clone()
    }
    pub(crate) fn outer(&self) -> [Handle; 13] {
        self.outer.clone()
    }
    pub(crate) fn ple(&self) -> [Handle; 12] {
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
            .chain(self.outer.iter())
            .chain(self.ple.iter())
            .chain(self.gdn.iter())
            .chain(self.qsa.iter())
            .chain(self.moe.handles())
            .any(|v| same_view(h, v))
    }
}
fn same_view(a: &Handle, b: &Handle) -> bool {
    a.stream == b.stream
        && a.clone().binding().memory.allocation_id() == b.clone().binding().memory.allocation_id()
        && a.offset_start.unwrap_or(0) == b.offset_start.unwrap_or(0)
        && a.offset_end.unwrap_or(0) == b.offset_end.unwrap_or(0)
        && a.size_in_used() == b.size_in_used()
}
/// Full allocation identity alone is insufficient once native phase regions
/// are carved. Admit only genuinely disjoint ranges of the same retained root.
pub(crate) fn overlaps(a: &Handle, b: &Handle) -> Result<bool> {
    if a.stream != b.stream {
        return Err(Error::InvalidBufferLength);
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
    fn native_default_persistent_and_overlaid_views_fit_actual_registered_roots() {
        let stream = StreamId { value: 98 };
        let roots = std::array::from_fn(|_| Handle::new(stream, allocation::ROOT_BYTES));
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
            assert_eq!(layout.working[0].size_in_used(), 64 * 1024 * 1024);
            assert_eq!(layout.working[1].size_in_used(), 64 * 1024 * 1024);
        }
    }
}
