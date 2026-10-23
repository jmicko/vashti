// Real complete-M128 prefix plus original exact remaining-member producer.
#[allow(clippy::too_many_arguments)]
pub(super) fn try_dense_split(
    client: &ComputeClient<WgpuRuntime>,
    h: [Handle; 4],
    count: usize,
    columns: usize,
    rows: usize,
    weight_first: usize,
    out_first: usize,
    out_stride: usize,
    bfloat: bool,
) -> Result<bool> {
    let Some((prefix, suffix)) = prefix_partition(count) else {
        return Ok(false);
    };
    let gr = gr_down_shape(bfloat, columns, rows, weight_first, out_first, out_stride);
    let hw = &client.properties().hardware;
    if !vector_views(&h)
        || !selected_min(
            prefix,
            columns,
            if gr {
                rows.div_ceil(BLOCK) * BLOCK
            } else {
                rows
            },
            weight_first,
            hw.max_shared_memory_size,
            hw.max_units_per_cube,
            hw.max_cube_dim,
            hw.plane_size_min,
            hw.plane_size_max,
            128,
        )
    {
        return Ok(false);
    }
    // Validate the entire real call before enqueuing its first prefix dispatch.
    // This does not advertise a larger input count or synthesize private tokens.
    check_bounds(
        &h,
        count,
        columns,
        rows,
        weight_first,
        out_first,
        out_stride,
    )?;
    let shifted = prefix_input_tail(&h[0], prefix, columns)?;
    let tail_first = out_first
        .checked_add(
            prefix
                .checked_mul(out_stride)
                .ok_or(Error::ArithmeticOverflow)?,
        )
        .ok_or(Error::ArithmeticOverflow)?;
    if suffix > 0 {
        check_bounds(
            &[shifted.clone(), h[1].clone(), h[2].clone(), h[3].clone()],
            suffix,
            columns,
            rows,
            weight_first,
            tail_first,
            out_stride,
        )?;
    }
    // Same physical hardware/CMMA/owners/grid checks, real count128 or more.
    // The original general selector retains its original minimum512 policy.
    if !try_dense_impl(
        client,
        h.clone(),
        prefix,
        columns,
        rows,
        weight_first,
        out_first,
        out_stride,
        bfloat,
        true,
        true,
    )? {
        return Ok(false);
    }
    if suffix > 0 {
        super::native::run_dense_original(
            client,
            [shifted, h[1].clone(), h[2].clone(), h[3].clone()],
            suffix,
            columns,
            rows,
            weight_first,
            tail_first,
            out_stride,
            bfloat,
        )?;
    }
    crate::strata_prefill_profile::count("dense_prefix_products", 1);
    crate::strata_prefill_profile::count("dense_prefix_tokens", prefix as u64);
    crate::strata_prefill_profile::count("dense_suffix_tokens", suffix as u64);
    Ok(true)
}
fn prefix_partition(count: usize) -> Option<(usize, usize)> {
    if count < 128 || (count >= 512 && count.is_multiple_of(128)) {
        return None;
    }
    let prefix = count / 128 * 128;
    Some((prefix, count - prefix))
}
fn prefix_input_tail(input: &Handle, prefix: usize, columns: usize) -> Result<Handle> {
    let bytes = checked_bytes(
        prefix
            .checked_mul(columns)
            .ok_or(Error::ArithmeticOverflow)?,
        2,
    )?;
    if bytes > input.size_in_used() || input.offset_start.unwrap_or(0).checked_add(bytes).is_none()
    {
        return Err(Error::InvalidBufferLength);
    }
    Ok(input.clone().offset_start(bytes))
}
#[cfg(test)]
mod prefix_tests {
    use super::*;
    #[test]
    fn all_actual_member_partitions_cover_once_without_extra_private_tokens() {
        for count in 0usize..=32768 * 4 {
            let p = prefix_partition(count);
            if count < 128 || (count >= 512 && count % 128 == 0) {
                assert_eq!(p, None);
                continue;
            }
            let (full, tail) = p.unwrap();
            assert!(full >= 128 && full % 128 == 0 && tail < 128);
            assert_eq!(full + tail, count);
            for i in [0, full - 1, count - 1] {
                assert_eq!(
                    usize::from(i < full) + usize::from(i >= full && i < count),
                    1
                );
            }
        }
        assert_eq!(prefix_partition(148), Some((128, 20)));
        assert_eq!(prefix_partition(511), Some((384, 127)));
        assert_eq!(prefix_partition(513), Some((512, 1)));
    }
    #[test]
    fn genuine_borrowed_tail_keeps_root_stream_end_and_absolute_offset() {
        use cubecl_common::stream_id::StreamId;
        let original = Handle::new(StreamId::current(), 64 * 1024 * 1024)
            .offset_start(512)
            .offset_end(64);
        for cols in [320, 2560, 10240] {
            for prefix in [128, 256, 384, 512, 640] {
                let tail = prefix_input_tail(&original, prefix, cols).unwrap();
                assert_eq!(tail.offset_start, Some(512 + (prefix * cols * 2) as u64));
                assert_eq!(tail.offset_end, original.offset_end);
                assert_eq!(tail.stream, original.stream);
                assert_eq!(
                    tail.size_in_used() + (prefix * cols * 2) as u64,
                    original.size_in_used()
                );
                assert_eq!(
                    tail.clone().binding().memory.allocation_id(),
                    original.clone().binding().memory.allocation_id()
                );
            }
        }
        assert!(prefix_input_tail(&original, usize::MAX, 10240).is_err());
        assert!(prefix_input_tail(&original, 32768, 10240).is_err());
    }
    #[test]
    fn real_prefix_geometry_changes_only_minimum_preserving_original_general_policy() {
        for count in [128, 256, 384, 512, 640] {
            assert!(selected_min(
                count,
                10240,
                384,
                0,
                SHARED_BYTES,
                THREADS,
                (THREADS, 1, 1),
                16,
                32,
                128
            ));
            assert_eq!(
                selected(
                    count,
                    10240,
                    384,
                    0,
                    SHARED_BYTES,
                    THREADS,
                    (THREADS, 1, 1),
                    16,
                    32
                ),
                count >= 512
            );
            assert!(!selected_min(
                count,
                10240,
                384,
                0,
                SHARED_BYTES - 1,
                THREADS,
                (THREADS, 1, 1),
                16,
                32,
                128
            ));
            assert!(!selected_min(
                count,
                10240,
                384,
                0,
                SHARED_BYTES,
                THREADS - 1,
                (THREADS, 1, 1),
                16,
                32,
                128
            ));
            assert!(!selected_min(
                count,
                10240,
                384,
                0,
                SHARED_BYTES,
                THREADS,
                (THREADS, 1, 1),
                32,
                32,
                128
            ));
        }
        assert!(!selected_min(
            148,
            10240,
            384,
            0,
            SHARED_BYTES,
            THREADS,
            (THREADS, 1, 1),
            16,
            32,
            128
        ));
        assert!(!selected_min(
            128,
            10240,
            320,
            0,
            SHARED_BYTES,
            THREADS,
            (THREADS, 1, 1),
            16,
            32,
            128
        ));
    }
}
