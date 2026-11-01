//! Fill final registered HOST roots from authenticated original role bytes.
use super::*;
fn fill_public_blob(
    roles: [u64; 3],
    span: &mut [u8],
    mut read: impl FnMut(usize, &mut [u8]) -> KResult<()>,
) -> KResult<()> {
    let bytes = roles.iter().try_fold(0usize, |n, &v| {
        if v == 0 || v > SOURCE_WINDOW {
            return Err(Error::InvalidBufferLength);
        }
        n.checked_add(usize::try_from(v).map_err(|_| Error::ArithmeticOverflow)?)
            .ok_or(Error::ArithmeticOverflow)
    })?;
    if bytes > span.len() {
        return Err(Error::InvalidBufferLength);
    }
    // Write the padding zeros here (parallel, write-only) rather than reading
    // back a pre-zeroed span from slow mapped host memory.
    span[bytes..].fill(0);
    let mut first = 0;
    for (role, bytes) in roles.into_iter().enumerate() {
        let end = first + bytes as usize;
        read(role, &mut span[first..end])?;
        first = end;
    }
    Ok(())
}
impl Host {
    pub(super) fn load_blob_pages(
        &mut self,
        plan: &GlobalPlan,
        graph: BoundGraph<'_, '_>,
        charges: &mut HeapCharges,
    ) -> KResult<()> {
        if !self.packed || !plan.has_host_blobs() {
            return Err(fail());
        }
        plan.check_binding(graph)?;
        #[cfg(feature = "strata-parallel-loading")]
        let reader = graph
            .tensor("blk.0.ffn_gate_exps.weight")?
            .positional_reader()?;
        for root in plan.host_roots()? {
            if root.role != crate::strata_global_plan::PACKED_HOST_ROLE {
                return Err(fail());
            }
            let l = plan.layers.get(root.layer).ok_or_else(fail)?;
            let tensors = [
                graph.tensor(&format!("blk.{}.ffn_gate_exps.weight", root.layer))?,
                graph.tensor(&format!("blk.{}.ffn_up_exps.weight", root.layer))?,
                graph.tensor(&format!("blk.{}.ffn_down_exps.weight", root.layer))?,
            ];
            let shard = tensors[0].info().shard_index();
            // The pinned model's ALL144 routed tensors are in shard0. Retain
            // the genuine original-shard identity; never label mixed sources
            // as a single authenticated file.
            if shard >= SOURCE_SHA256.len()
                || tensors.iter().any(|t| t.info().shard_index() != shard)
            {
                return Err(fail());
            }
            let stride = plan.host_blob_stride(root.layer)?;
            if root.count.checked_mul(stride) != Some(root.logical_bytes as usize)
                || root.physical_bytes > crate::strata_global_plan::MAX_PIECE_BYTES
                || l.role_expert_bytes
                    .iter()
                    .any(|&n| n == 0 || n > SOURCE_WINDOW)
            {
                return Err(fail());
            }
            self.allocate_original_page_filled(
                shadow::PageSpec {
                    layer: root.layer,
                    role: root.role,
                    piece: root.piece,
                    shard,
                    logical: root.logical_bytes,
                    backing_bound: envelope(root.logical_bytes)?,
                },
                charges,
                |mapped| {
                    let target = mapped
                        .get_mut(..root.logical_bytes as usize)
                        .ok_or(Error::InvalidBufferLength)?;
                    let fill = |ordinal: usize, span: &mut [u8]| -> KResult<()> {
                        let expert = l.expert_at(
                            root.first
                                .checked_add(ordinal)
                                .ok_or(Error::ArithmeticOverflow)?,
                        )?;
                        fill_public_blob(l.role_expert_bytes, span, |role, output| {
                            let source = (expert as u64)
                                .checked_mul(l.role_expert_bytes[role])
                                .ok_or(Error::ArithmeticOverflow)?;
                            #[cfg(feature = "strata-parallel-loading")]
                            reader.read_range(tensors[role], source, output)?;
                            #[cfg(not(feature = "strata-parallel-loading"))]
                            tensors[role].read_range(source, output)?;
                            Ok(())
                        })?;
                        Ok(())
                    };
                    #[cfg(feature = "strata-parallel-loading")]
                    {
                        parallel_reads::fill(target, stride, fill).map_err(|e| match e {
                            parallel_reads::Failure::Read(e) => e,
                            parallel_reads::Failure::Geometry => Error::InvalidBufferLength,
                            parallel_reads::Failure::Spawn => {
                                Error::LimitExceeded("HOST blob read workers")
                            }
                        })?;
                        reader.verify_file_unchanged()?;
                    }
                    #[cfg(not(feature = "strata-parallel-loading"))]
                    for (ordinal, span) in target.chunks_mut(stride).enumerate() {
                        fill(ordinal, span)?;
                    }
                    Ok(())
                },
            )?;
        }
        graph.verify_unchanged()?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn host_blob_final_span_has_original_role_bytes_zero_padding_and_bounded_reads() {
        // Start from non-zero bytes: the fill itself must write the padding.
        let mut span = [0xAAu8; 64];
        let mut calls = Vec::new();
        fill_public_blob([16, 20, 24], &mut span, |r, s| {
            calls.push((r, s.len()));
            s.fill((r + 1) as u8);
            Ok(())
        })
        .unwrap();
        assert_eq!(calls, [(0, 16), (1, 20), (2, 24)]);
        assert_eq!(&span[..16], &[1; 16]);
        assert_eq!(&span[16..36], &[2; 20]);
        assert_eq!(&span[36..60], &[3; 24]);
        assert_eq!(&span[60..], &[0; 4]);
        for roles in [[0, 4, 4], [SOURCE_WINDOW + 1, 4, 4], [32, 32, 32]] {
            let mut target = [0; 64];
            assert!(
                fill_public_blob(roles, &mut target, |_, _| panic!(
                    "invalid public source geometry must not read"
                ))
                .is_err()
            );
        }
        // Stale non-zero padding is overwritten with zeros, never published.
        span[63] = 7;
        fill_public_blob([16, 20, 24], &mut span, |_, s| {
            s.fill(9);
            Ok(())
        })
        .unwrap();
        assert_eq!(&span[60..], &[0; 4]);
    }
    #[cfg(feature = "strata-parallel-loading")]
    #[test]
    fn host_blob_composed_parallel_role_failure_joins_every_reader_before_publication() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        let done = AtomicUsize::new(0);
        let mut spans = [0u8; 8 * 64];
        let result = parallel_reads::fill(&mut spans, 64, |expert, span| {
            fill_public_blob([16, 20, 24], span, |role, bytes| {
                if expert == 0 && role == 1 {
                    return Err(Error::InvalidBufferLength);
                }
                bytes.fill((role + 1) as u8);
                done.fetch_add(1, Ordering::Relaxed);
                Ok(())
            })
        });
        assert!(matches!(result, Err(parallel_reads::Failure::Read(_))));
        assert_eq!(done.load(Ordering::Relaxed), 22);
        for span in spans[64..].chunks_exact(64) {
            assert_eq!(&span[..16], &[1; 16]);
            assert_eq!(&span[16..36], &[2; 20]);
            assert_eq!(&span[36..60], &[3; 24]);
            assert_eq!(&span[60..], &[0; 4]);
        }
    }
}
