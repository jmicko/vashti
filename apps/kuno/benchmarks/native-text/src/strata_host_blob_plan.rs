//! Immutable PUBLIC model-only HOST layout. Every expert stays in one root.
use super::*;
pub(crate) const PACKED_HOST_ROLE: usize = 3;
#[derive(Clone, Copy, Debug)]
pub(crate) struct HostRoot {
    pub layer: usize,
    pub role: usize,
    pub piece: usize,
    pub first: usize,
    pub count: usize,
    pub logical_bytes: u64,
    pub physical_bytes: u64,
}
#[derive(Clone, Debug)]
struct BlobLayer {
    resident: usize,
    roles: [u64; 3],
    stride: u64,
    chunks: Vec<HostRoot>,
}
#[derive(Clone, Debug)]
pub(crate) struct HostBlobs {
    layers: Vec<BlobLayer>,
    registered: u64,
    backing: u64,
}
impl HostBlobs {
    fn new(plan: &Plan, alignment: u64) -> Result<Self> {
        if !alignment.is_power_of_two() || !(ALIGNMENT..=1024 * 1024).contains(&alignment) {
            return Err(reject());
        }
        let mut layers = Vec::with_capacity(LAYERS);
        let (mut registered, mut backing) = (0u64, 0u64);
        for (layer, l) in plan.layers.iter().enumerate() {
            let bytes = l.role_expert_bytes.iter().try_fold(0u64, |n, &v| {
                n.checked_add(v).ok_or(Error::ArithmeticOverflow)
            })?;
            let stride = align(bytes, alignment)?;
            let max =
                usize::try_from(MAX_PIECE_BYTES / stride).map_err(|_| Error::ArithmeticOverflow)?;
            if max == 0 {
                return Err(reject());
            }
            let mut chunks = Vec::new();
            let mut first = l.resident;
            while first < EXPERTS {
                let count = (EXPERTS - first).min(max);
                let logical_bytes = (count as u64)
                    .checked_mul(stride)
                    .ok_or(Error::ArithmeticOverflow)?;
                let physical_bytes = align(logical_bytes, alignment)?;
                if physical_bytes > MAX_PIECE_BYTES {
                    return Err(reject());
                }
                chunks.push(HostRoot {
                    layer,
                    role: PACKED_HOST_ROLE,
                    piece: chunks.len(),
                    first,
                    count,
                    logical_bytes,
                    physical_bytes,
                });
                registered = registered
                    .checked_add(logical_bytes)
                    .ok_or(Error::ArithmeticOverflow)?;
                backing = backing
                    .checked_add(physical_bytes)
                    .ok_or(Error::ArithmeticOverflow)?;
                first += count;
            }
            layers.push(BlobLayer {
                resident: l.resident,
                roles: l.role_expert_bytes,
                stride,
                chunks,
            });
        }
        Ok(Self {
            layers,
            registered,
            backing,
        })
    }
    pub(super) fn source(&self, layer: usize, virtual_slot: usize, role: usize) -> Result<Source> {
        let l = self.layers.get(layer).ok_or_else(reject)?;
        let bytes = *l.roles.get(role).ok_or_else(reject)?;
        if virtual_slot < l.resident || virtual_slot >= EXPERTS {
            return Err(reject());
        }
        let root = l
            .chunks
            .iter()
            .find(|c| virtual_slot >= c.first && virtual_slot - c.first < c.count)
            .ok_or_else(reject)?;
        let prefix = l.roles[..role].iter().sum::<u64>();
        let offset = ((virtual_slot - root.first) as u64)
            .checked_mul(l.stride)
            .and_then(|v| v.checked_add(prefix))
            .ok_or(Error::ArithmeticOverflow)?;
        if offset
            .checked_add(bytes)
            .is_none_or(|v| v > root.logical_bytes)
        {
            return Err(reject());
        }
        Ok(Source::Host {
            piece: root.piece,
            offset,
            bytes,
        })
    }
}
impl Plan {
    pub(crate) fn has_host_blobs(&self) -> bool {
        self.host_blobs.is_some()
    }
    pub(crate) fn host_registered_bytes(&self) -> u64 {
        self.host_blobs
            .as_ref()
            .map_or(self.host_logical_bytes, |l| l.registered)
    }
    pub(crate) fn host_roots(&self) -> Result<Vec<HostRoot>> {
        if let Some(layout) = &self.host_blobs {
            return Ok(layout
                .layers
                .iter()
                .flat_map(|l| l.chunks.iter().copied())
                .collect());
        }
        let mut out = Vec::new();
        for (layer, l) in self.layers.iter().enumerate() {
            for (role, pieces) in l.roles.iter().enumerate() {
                for (piece, p) in pieces.iter().enumerate() {
                    if p.tier == Tier::Host {
                        out.push(HostRoot {
                            layer,
                            role,
                            piece,
                            first: p.first,
                            count: p.count,
                            logical_bytes: p.logical_bytes,
                            physical_bytes: p.physical_bytes,
                        });
                    }
                }
            }
        }
        Ok(out)
    }
    pub(crate) fn host_blob_stride(&self, layer: usize) -> Result<usize> {
        usize::try_from(
            self.host_blobs
                .as_ref()
                .ok_or_else(reject)?
                .layers
                .get(layer)
                .ok_or_else(reject)?
                .stride,
        )
        .map_err(|_| Error::ArithmeticOverflow)
    }
    pub(super) fn enable_host_blobs(&mut self, alignment: u64) -> Result<()> {
        if self.host_blobs.is_some() {
            return Err(reject());
        }
        let layout = HostBlobs::new(self, alignment)?;
        if layout.registered < self.host_logical_bytes {
            return Err(reject());
        }
        self.host_piece_bytes = layout.backing;
        self.host_blobs = Some(layout);
        Ok(())
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn host_blob_real_plan_preserves_resident_prefix_all_roles_and_root_bounds() {
        for budget in [0u64, 1, 24_414_413_824, u64::MAX / 2] {
            let mut p = Plan::from_roles(pinned_roles(), budget, ALIGNMENT).unwrap();
            let resident = p.slots.clone();
            let resident_bytes = p.resident_piece_bytes;
            let original_host = p.host_logical_bytes;
            let before = (0..LAYERS)
                .flat_map(|l| (0..EXPERTS).flat_map(move |e| (0..3).map(move |r| (l, e, r))))
                .map(|(l, e, r)| p.source(l, e, r).unwrap())
                .collect::<Vec<_>>();
            p.enable_host_blobs(ALIGNMENT).unwrap();
            assert_eq!(p.slots, resident);
            assert_eq!(p.resident_piece_bytes, resident_bytes);
            assert_eq!(p.host_logical_bytes, original_host);
            let roots = p.host_roots().unwrap();
            assert!(roots.iter().all(|r| r.role == 3
                && r.physical_bytes <= MAX_PIECE_BYTES
                && r.logical_bytes <= r.physical_bytes));
            let (mut logical, mut visited) = (0u64, 0usize);
            for l in 0..LAYERS {
                for e in 0..EXPERTS {
                    let sources = (0..3)
                        .map(|r| p.source(l, e, r).unwrap())
                        .collect::<Vec<_>>();
                    if let Source::Resident { .. } = sources[0] {
                        for (r, s) in sources.iter().enumerate() {
                            assert_eq!(*s, before[(l * EXPERTS + e) * 3 + r]);
                        }
                        continue;
                    }
                    let (mut root, mut end) = (None, 0);
                    for (r, s) in sources.iter().enumerate() {
                        let Source::Host {
                            piece,
                            offset,
                            bytes,
                        } = *s
                        else {
                            panic!("split tier")
                        };
                        assert_eq!(bytes, p.layers[l].role_expert_bytes[r]);
                        assert!(root.is_none_or(|old| old == piece));
                        if r > 0 {
                            assert_eq!(offset, end);
                        }
                        root = Some(piece);
                        end = offset + bytes;
                        logical += bytes;
                        let page = roots
                            .iter()
                            .find(|x| x.layer == l && x.piece == piece)
                            .unwrap();
                        assert!(end <= page.logical_bytes);
                    }
                    visited += 1;
                }
            }
            assert_eq!(visited, LAYERS * EXPERTS - resident.len());
            assert_eq!(logical, original_host);
            assert_eq!(
                p.host_registered_bytes(),
                roots.iter().map(|r| r.logical_bytes).sum::<u64>()
            );
        }
    }
    #[test]
    fn host_blob_pinned_role_ranges_preserve_original_exchange_alignment_and_quotes() {
        let mut p = Plan::from_roles(pinned_roles(), 24_414_413_824, ALIGNMENT).unwrap();
        p.enable_host_blobs(ALIGNMENT).unwrap();
        for l in 0..LAYERS {
            for e in 0..EXPERTS {
                for r in 0..3 {
                    let (offset, bytes) = match p.source(l, e, r).unwrap() {
                        Source::Host { offset, bytes, .. }
                        | Source::Resident { offset, bytes, .. } => (offset, bytes),
                    };
                    assert!(offset.is_multiple_of(ALIGNMENT));
                    assert!(bytes.is_multiple_of(ALIGNMENT));
                }
            }
        }
        let roots = p.host_roots().unwrap();
        assert!(roots.len() < cubecl_runtime::public_external::MAX_BUFFERS);
        assert_eq!(
            p.host_piece_bytes,
            roots.iter().map(|r| r.physical_bytes).sum::<u64>()
        );
        assert_eq!(p.host_logical_bytes, p.host_registered_bytes());
    }
    #[test]
    fn host_blob_padding_and_virtual_ids_are_disjoint_and_invalid_ranges_rejected() {
        let mut p = Plan::from_roles([[260, 516, 772]; LAYERS], 0, ALIGNMENT).unwrap();
        p.enable_host_blobs(ALIGNMENT).unwrap();
        assert_eq!(p.host_blob_stride(0).unwrap(), 1792);
        let mut spans = Vec::new();
        for e in 0..EXPERTS {
            for r in 0..3 {
                let Source::Host {
                    piece,
                    offset,
                    bytes,
                } = p.source(0, e, r).unwrap()
                else {
                    panic!()
                };
                spans.push((piece, offset, offset + bytes));
            }
        }
        spans.sort();
        for pair in spans.windows(2) {
            assert!(pair[0].0 != pair[1].0 || pair[0].2 <= pair[1].1);
        }
        for (l, e, r) in [(48, 0, 0), (0, 512, 0), (0, 0, 3)] {
            assert!(p.source(l, e, r).is_err());
        }
        assert!(p.enable_host_blobs(ALIGNMENT).is_err());
        let old = Plan::from_roles(pinned_roles(), 0, ALIGNMENT).unwrap();
        assert!(!old.has_host_blobs());
        assert!(old.host_blob_stride(0).is_err());
    }
}
