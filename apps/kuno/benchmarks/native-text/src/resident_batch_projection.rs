//! Shared original-packed MAX8 projection seam for the future chunk owner.
//! Uses the existing admitted model cache; no public/decoded weight duplication,
//! state commit, synchronization, arbitrary input or device initialization.
use crate::{
    private_arena::{BufferId, PrivateArena},
    quant_matvec::matmul::{BatchSpans, MAX_BATCH},
    resident_gdn::BoundProjection,
    resident_util as u,
};
use kuno_text::{Error, Result};
use std::sync::atomic::AtomicBool;

/// Full original row width is checked independently of bounded row tiles.
/// Returning an end proves all active tokens; it never narrows a storage owner.
pub fn validate_view(
    count: usize,
    first: usize,
    stride: usize,
    width: usize,
    words: usize,
) -> Result<usize> {
    if !(1..=MAX_BATCH).contains(&count) || width == 0 || stride < width {
        return Err(Error::InvalidBufferLength);
    }
    let end = (count - 1)
        .checked_mul(stride)
        .and_then(|n| first.checked_add(n))
        .and_then(|n| n.checked_add(width))
        .ok_or(Error::ArithmeticOverflow)?;
    if end > words
        || [first, stride, width, words, end]
            .into_iter()
            .any(|n| n > u32::MAX as usize)
    {
        return Err(Error::InvalidBufferLength);
    }
    Ok(end)
}

#[allow(dead_code)] // Pending full causal-owner composition; no default selection.
pub trait BatchedResidentProjector {
    /// Sealed stage owners must fold projections into their SAME model fault.
    fn global_fault(&self) -> BufferId;
    #[allow(clippy::too_many_arguments)]
    fn enqueue_batch(
        &mut self,
        projection: BoundProjection<'_>,
        arena: &PrivateArena,
        input: BufferId,
        output: BufferId,
        spans: &BatchSpans,
        cancel: &AtomicBool,
    ) -> Result<()>;
}

#[allow(dead_code)]
pub struct ResidentBatchProjection<'e, 'f> {
    pub executor: &'e mut crate::executor::GpuExecutor<'f>,
    pub fault: BufferId,
}
impl BatchedResidentProjector for ResidentBatchProjection<'_, '_> {
    fn global_fault(&self) -> BufferId {
        self.fault
    }
    fn enqueue_batch(
        &mut self,
        projection: BoundProjection<'_>,
        arena: &PrivateArena,
        input: BufferId,
        output: BufferId,
        spans: &BatchSpans,
        cancel: &AtomicBool,
    ) -> Result<()> {
        u::cancel(cancel)?;
        self.executor
            .check_client(arena.client().map_err(u::fault)?)?;
        if input == output
            || input == self.fault
            || output == self.fault
            || u::words(arena, self.fault)? != 1
        {
            return Err(Error::InvalidBufferLength);
        }
        let input_words = u::words(arena, input)?;
        let output_words = u::words(arena, output)?;
        validate_view(
            spans.count(),
            spans.input_first(),
            spans.input_stride(),
            projection.matrix.inputs(),
            input_words,
        )?;
        validate_view(
            spans.count(),
            spans.output_first(),
            spans.output_stride(),
            projection.matrix.outputs(),
            output_words,
        )?;
        // SAFETY: live distinct arena IDs are exclusive actual physical owners;
        // the enclosing owner retains them through completion and erasure.
        let fused = unsafe {
            self.executor.enqueue_matrix_batch(
                projection.tensor,
                &u::handle(arena, input)?,
                input_words,
                &u::handle(arena, output)?,
                output_words,
                &u::handle(arena, self.fault)?,
                spans,
                cancel,
            )?
        };
        // Preserve the whole-token output fault decision. Unused owned logical
        // columns are zero-initialized; a prior failed chunk cannot be reused.
        if !fused {
            u::validate(arena, output, self.fault)?;
        }
        u::cancel(cancel)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn large_head_and_odd_stride_views_cover_every_active_original_row() {
        for count in [1, 3, 5, 8] {
            for width in [48, 320, 640, 2560, 10240, 248320] {
                let first = 7;
                let stride = width + 13;
                let words = first + 8 * stride;
                let end = validate_view(count, first, stride, width, words).unwrap();
                for token in 0..count {
                    for column in [0, width - 1] {
                        assert!(first + token * stride + column < end);
                    }
                }
                assert!(validate_view(count, first, stride, width, end - 1).is_err());
            }
        }
    }
    #[test]
    fn overlapping_rows_empty_counts_and_shader_overflow_reject() {
        for x in [
            (0, 0, 7, 7, 7),
            (9, 0, 7, 7, 100),
            (1, 0, 0, 0, 1),
            (8, 0, 6, 7, 100),
            (8, usize::MAX, usize::MAX, 1, usize::MAX),
            (1, 0, 1, 1, u32::MAX as usize + 1),
        ] {
            assert!(validate_view(x.0, x.1, x.2, x.3, x.4).is_err());
        }
    }
}
