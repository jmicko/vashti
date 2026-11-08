//! Pure range coalescing AFTER the factory resolves every current source.
//! This grants no lookup/completion authority: the owning backend revalidates.
use super::*;
use cubecl_runtime::{
    public_external::{PublicHostCopy, validate_host_copies},
    server::Binding,
};
fn same_root(a: &Binding, b: &Binding) -> bool {
    a.memory.allocation_id() == b.memory.allocation_id()
        && a.stream == b.stream
        && a.size == b.size
        && a.offset_start.unwrap_or(0) == 0
        && b.offset_start.unwrap_or(0) == 0
        && a.offset_end.unwrap_or(0) == 0
        && b.offset_end.unwrap_or(0) == 0
}
pub(super) fn coalesce(
    copies: Vec<PublicHostCopy>,
    stream: cubecl_common::stream_id::StreamId,
) -> Result<Vec<PublicHostCopy>> {
    if copies.is_empty() {
        return Ok(copies);
    }
    validate_host_copies(&copies, stream).map_err(Error::InvalidFormat)?;
    let mut out: Vec<PublicHostCopy> = Vec::with_capacity(copies.len());
    for c in copies {
        if let Some(previous) = out.last_mut() {
            if same_root(&previous.source, &c.source)
                && same_root(&previous.destination, &c.destination)
                && previous.source_first.checked_add(previous.bytes) == Some(c.source_first)
                && previous.destination_first.checked_add(previous.bytes)
                    == Some(c.destination_first)
            {
                previous.bytes = previous
                    .bytes
                    .checked_add(c.bytes)
                    .ok_or(Error::ArithmeticOverflow)?;
                continue;
            }
        }
        out.push(c);
    }
    validate_host_copies(&out, stream).map_err(Error::InvalidFormat)?;
    Ok(out)
}
#[cfg(test)]
mod tests {
    use super::*;
    use cubecl_common::stream_id::StreamId;
    use cubecl_runtime::public_external::{
        PublicExternalBudget, PublicExternalHostSource, PublicExternalPool,
    };
    use cubecl_runtime::server::Handle;
    fn sources() -> (PublicExternalPool<u32>, Vec<Handle>) {
        let stream = StreamId::current();
        let mut p = PublicExternalPool::new(
            PublicExternalBudget {
                backing_bytes: 4096,
                buffers: 3,
                // The original registry reserves its bounded kernel-ID masks
                // even for this three-registration metadata-only fixture.
                metadata_bytes: cubecl_runtime::public_external::MAX_METADATA_BYTES,
            },
            stream,
        )
        .unwrap();
        // Metadata-only original public-source descriptors; no GPU authority/math is
        // claimed. Real source lookup/retirement remains the unchanged backend gate.
        let mut h = Vec::new();
        for i in 0..3u32 {
            let s = unsafe {
                PublicExternalHostSource::from_initialized_public_host(
                    i,
                    256,
                    256,
                    6,
                    0,
                    [i as u8 + 1; 32],
                )
            }
            .unwrap();
            h.push(p.register(s).unwrap().handle());
        }
        (p, h)
    }
    fn piece(s: &Handle, d: &Handle, a: u64, b: u64, n: u64) -> PublicHostCopy {
        PublicHostCopy {
            source: s.clone().binding(),
            destination: d.clone().binding(),
            source_first: a,
            destination_first: b,
            bytes: n,
        }
    }
    #[test]
    fn host_blob_copies_merge_only_actual_adjacent_same_root_ranges() {
        let (_pool, h) = sources();
        let d = Handle::new(StreamId::current(), 256);
        let out = coalesce(
            vec![
                piece(&h[0], &d, 16, 32, 32),
                piece(&h[0], &d, 48, 64, 48),
                piece(&h[0], &d, 96, 112, 16),
            ],
            d.stream,
        )
        .unwrap();
        assert_eq!(out.len(), 1);
        assert_eq!(
            (out[0].source_first, out[0].destination_first, out[0].bytes),
            (16, 32, 96)
        );
        assert_eq!(
            out[0].source.memory.allocation_id(),
            h[0].clone().binding().memory.allocation_id()
        );
    }
    #[test]
    fn host_blob_copies_preserve_gaps_order_sources_and_destinations() {
        let (_pool, h) = sources();
        let d = Handle::new(StreamId::current(), 256);
        let other = Handle::new(d.stream, 256);
        for copies in [
            vec![piece(&h[0], &d, 0, 0, 16), piece(&h[1], &d, 16, 16, 16)],
            vec![piece(&h[0], &d, 0, 0, 16), piece(&h[0], &d, 32, 16, 16)],
            vec![piece(&h[0], &d, 0, 0, 16), piece(&h[0], &d, 16, 32, 16)],
            vec![piece(&h[0], &d, 0, 0, 16), piece(&h[0], &other, 16, 16, 16)],
            vec![piece(&h[0], &d, 16, 16, 16), piece(&h[0], &d, 0, 0, 16)],
        ] {
            assert_eq!(coalesce(copies, d.stream).unwrap().len(), 2);
        }
    }
    #[test]
    fn host_blob_copies_reject_invalid_original_views_before_merging() {
        let (_pool, h) = sources();
        let d = Handle::new(StreamId::current(), 256);
        for kind in 0..7 {
            let mut c = piece(&h[0], &d, 0, 0, 32);
            match kind {
                0 => c.source = h[0].clone().offset_start(4).binding(),
                1 => c.destination = d.clone().offset_end(4).binding(),
                2 => c.source = d.clone().binding(),
                3 => c.destination = h[0].clone().binding(),
                4 => c.source_first = u64::MAX - 15,
                5 => c.bytes = 257,
                _ => {
                    c.source.stream = StreamId {
                        value: d.stream.value.wrapping_add(1),
                    }
                }
            };
            assert!(coalesce(vec![c], d.stream).is_err());
        }
        assert!(coalesce(Vec::new(), d.stream).unwrap().is_empty());
    }
}
