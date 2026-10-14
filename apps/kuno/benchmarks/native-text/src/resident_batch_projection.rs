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
    /// Prompt output is an exact carved view authenticated by the active owner,
    /// never an arbitrary writable handle or an immutable model-cache buffer.
    #[allow(clippy::too_many_arguments)]
    fn enqueue_prompt_view(
        &mut self,
        _projection: BoundProjection<'_>,
        _arena: &PrivateArena,
        _input: cubecl_runtime::server::Handle,
        _output: cubecl_runtime::server::Handle,
        _spans: &BatchSpans,
        _cancel: &AtomicBool,
    ) -> Result<()> {
        Err(Error::InvalidFormat("carved prompt projection unavailable"))
    }
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
    #[allow(clippy::too_many_arguments)]
    fn enqueue_pair(
        &mut self,
        projections: [BoundProjection<'_>; 2],
        arena: &PrivateArena,
        input: BufferId,
        outputs: [BufferId; 2],
        spans: &BatchSpans,
        cancel: &AtomicBool,
    ) -> Result<()> {
        for (p, out) in projections.into_iter().zip(outputs) {
            self.enqueue_batch(p, arena, input, out, spans, cancel)?;
        }
        Ok(())
    }
    /// Consecutive projections borrow one immutable input epoch. Distinct
    /// output owners and original per-projection spans remain authoritative.
    fn enqueue_bundle(
        &mut self,
        jobs: &[(BoundProjection<'_>, BufferId, BatchSpans)],
        arena: &PrivateArena,
        input: BufferId,
        cancel: &AtomicBool,
    ) -> Result<()> {
        for (projection, output, spans) in jobs {
            self.enqueue_batch(*projection, arena, input, *output, spans, cancel)?;
        }
        Ok(())
    }
    fn enqueue_native_gr(
        &mut self,
        _projections: [BoundProjection<'_>; 2],
        _injection: Option<BoundProjection<'_>>,
        _request: crate::strata_gr::Read<'_>,
        _cancel: &AtomicBool,
    ) -> Result<bool> {
        Ok(false)
    }
    fn enqueue_native_gdn_ab(
        &mut self,
        _projections: [BoundProjection<'_>; 2],
        _request: crate::strata_gdn::Ab<'_>,
        _cancel: &AtomicBool,
    ) -> Result<bool> {
        Ok(false)
    }
    /// Prompt-only native FP16 producer on the actual admitted activation stage.
    fn enqueue_prompt_prepacked(
        &mut self,
        _projection: BoundProjection<'_>,
        _arena: &PrivateArena,
        _input: cubecl_runtime::server::Handle,
        _output: BufferId,
        _spans: &BatchSpans,
        _cancel: &AtomicBool,
    ) -> Result<()> {
        Err(Error::InvalidFormat(
            "prompt prepacked projection unavailable",
        ))
    }
    /// The ordinary path ignores this non-reusable Final-only capability.
    /// true means the caller must queue a full scratch clear after all reads.
    #[allow(clippy::too_many_arguments)]
    fn enqueue_final_head(
        &mut self,
        projection: BoundProjection<'_>,
        arena: &PrivateArena,
        input: BufferId,
        output: BufferId,
        spans: &BatchSpans,
        _scratch: crate::resident_gr::batch::FinalHeadScratch<'_>,
        cancel: &AtomicBool,
    ) -> Result<bool> {
        self.enqueue_batch(projection, arena, input, output, spans, cancel)?;
        Ok(false)
    }
    /// Separate sealed all-row scratch. The default retains original F32 head
    /// arithmetic; true requires the GR owner to clear its entire NORM page.
    #[allow(clippy::too_many_arguments)]
    fn enqueue_batched_final_head(
        &mut self,
        projection: BoundProjection<'_>,
        arena: &PrivateArena,
        input: BufferId,
        output: BufferId,
        spans: &BatchSpans,
        _scratch: crate::resident_gr::batch::BatchedFinalHeadScratch<'_>,
        cancel: &AtomicBool,
    ) -> Result<bool> {
        self.enqueue_batch(projection, arena, input, output, spans, cancel)?;
        Ok(false)
    }
}

#[allow(dead_code)]
pub struct ResidentBatchProjection<'e, 'f> {
    pub executor: &'e mut crate::executor::GpuExecutor<'f>,
    pub fault: BufferId,
    pub dense_q8: Option<BufferId>,
}
impl BatchedResidentProjector for ResidentBatchProjection<'_, '_> {
    fn enqueue_prompt_view(
        &mut self,
        projection: BoundProjection<'_>,
        arena: &PrivateArena,
        input: cubecl_runtime::server::Handle,
        output: cubecl_runtime::server::Handle,
        spans: &BatchSpans,
        cancel: &AtomicBool,
    ) -> Result<()> {
        u::cancel(cancel)?;
        crate::strata_prompt::validate_output(&output)?;
        if crate::strata_prompt::storage::overlaps(&input, &output)? {
            return Err(Error::InvalidBufferLength);
        }
        self.executor
            .check_client(arena.client().map_err(u::fault)?)?;
        self.executor.enqueue_strata_prompt_projection(
            projection.tensor,
            &input,
            &output,
            &u::handle(arena, self.fault)?,
            spans,
            cancel,
        )
    }
    fn enqueue_prompt_prepacked(
        &mut self,
        projection: BoundProjection<'_>,
        arena: &PrivateArena,
        input: cubecl_runtime::server::Handle,
        output: BufferId,
        spans: &BatchSpans,
        cancel: &AtomicBool,
    ) -> Result<()> {
        u::cancel(cancel)?;
        self.executor
            .check_client(arena.client().map_err(u::fault)?)?;
        self.executor.enqueue_strata_prompt_prepacked_projection(
            projection.tensor,
            &input,
            &u::handle(arena, output)?,
            &u::handle(arena, self.fault)?,
            spans,
            cancel,
        )
    }
    fn global_fault(&self) -> BufferId {
        self.fault
    }
    fn enqueue_native_gdn_ab(
        &mut self,
        projections: [BoundProjection<'_>; 2],
        request: crate::strata_gdn::Ab<'_>,
        cancel: &AtomicBool,
    ) -> Result<bool> {
        if !crate::strata_intel_core::enabled() {
            return Ok(false);
        }
        u::cancel(cancel)?;
        if self.fault != request.fault {
            return Err(Error::InvalidBufferLength);
        }
        self.executor
            .check_client(request.arena.client().map_err(u::fault)?)?;
        for p in projections {
            if p.matrix.inputs() != request.columns || p.matrix.outputs() != request.heads {
                return Err(Error::InvalidBufferLength);
            }
        }
        let weights = self
            .executor
            .strata_gdn_ab_weights(projections.map(|p| p.tensor))?;
        if crate::strata_prompt::active() {
            crate::strata_prompt::ab(request, weights)?;
        } else {
            crate::strata_gdn::ab(request, weights)?;
        }
        u::cancel(cancel)?;
        Ok(true)
    }
    fn enqueue_native_gr(
        &mut self,
        projections: [BoundProjection<'_>; 2],
        injection: Option<BoundProjection<'_>>,
        request: crate::strata_gr::Read<'_>,
        cancel: &AtomicBool,
    ) -> Result<bool> {
        if !crate::strata_intel_core::enabled() {
            return Ok(false);
        }
        u::cancel(cancel)?;
        if self.fault != request.fault {
            return Err(Error::InvalidBufferLength);
        }
        self.executor
            .check_client(request.arena.client().map_err(u::fault)?)?;
        let weights = self
            .executor
            .strata_gr_weights(projections.map(|p| p.tensor), injection.map(|p| p.tensor))?;
        if crate::strata_prompt::active() {
            crate::strata_prompt::gr(request, weights)?;
        } else {
            crate::strata_gr::read(request, weights)?;
        }
        u::cancel(cancel)?;
        Ok(true)
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
        if crate::strata_prompt::active() {
            self.executor.enqueue_strata_prompt_projection(
                projection.tensor,
                &u::handle(arena, input)?,
                &u::handle(arena, output)?,
                &u::handle(arena, self.fault)?,
                spans,
                cancel,
            )?;
            return u::cancel(cancel);
        }
        if crate::strata_intel_core::enabled()
            && (matches!(
                projection.tensor.info().name(),
                "blk.1.ple_key.weight" | "blk.1.ple_value.weight"
            ) || projection
                .tensor
                .info()
                .name()
                .ends_with(".ffn_gate_inp.weight")
                || projection
                    .tensor
                    .info()
                    .name()
                    .ends_with(".ffn_gate_inp_shexp.weight"))
            && self.executor.enqueue_strata_bf16_projection(
                projection.tensor,
                &u::handle(arena, input)?,
                &u::handle(arena, output)?,
                &u::handle(arena, self.fault)?,
                spans,
                cancel,
            )?
        {
            return u::cancel(cancel);
        }
        if let Some(planes) = self.dense_q8 {
            if planes == input || planes == output || planes == self.fault {
                return Err(Error::InvalidBufferLength);
            }
            if crate::strata_dense::supports_count(
                projection.tensor.info().dtype().gguf_id(),
                projection.matrix.inputs(),
                arena.handle(planes).map_err(u::fault)?.size_in_used(),
                spans.count(),
            ) {
                self.executor.enqueue_strata_dense(
                    projection.tensor,
                    &u::handle(arena, input)?,
                    &arena.handle(planes).map_err(u::fault)?.clone(),
                    &u::handle(arena, output)?,
                    &u::handle(arena, self.fault)?,
                    spans,
                    cancel,
                    true,
                )?;
                // The parent retains the authentic private scratch through
                // all consumers, before its next explicitly ordered reuse.
                u::cancel(cancel)?;
                return Ok(());
            }
        }
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
    fn enqueue_bundle(
        &mut self,
        jobs: &[(BoundProjection<'_>, BufferId, BatchSpans)],
        arena: &PrivateArena,
        input: BufferId,
        cancel: &AtomicBool,
    ) -> Result<()> {
        if !crate::strata_intel_core::enabled()
            || self.dense_q8.is_none()
            || crate::strata_prompt::active()
        {
            for (p, output, spans) in jobs {
                self.enqueue_batch(*p, arena, input, *output, spans, cancel)?;
            }
            return Ok(());
        }
        u::cancel(cancel)?;
        self.executor
            .check_client(arena.client().map_err(u::fault)?)?;
        let planes = self.dense_q8.ok_or(Error::InvalidBufferLength)?;
        let Some((first, _, first_spans)) = jobs.first() else {
            return Err(Error::InvalidBufferLength);
        };
        let inputs = first.matrix.inputs();
        let roots = [input, planes, self.fault];
        if !(1..=5).contains(&jobs.len())
            || roots
                .iter()
                .enumerate()
                .any(|(i, id)| roots[..i].contains(id))
            || u::words(arena, self.fault)? != 1
        {
            return Err(Error::InvalidBufferLength);
        }
        for (i, (p, output, spans)) in jobs.iter().enumerate() {
            if roots.contains(output)
                || jobs[..i].iter().any(|(_, out, _)| out == output)
                || p.matrix.inputs() != inputs
                || spans.count() != first_spans.count()
                || spans.input_first() != first_spans.input_first()
                || spans.input_stride() != first_spans.input_stride()
            {
                return Err(Error::InvalidBufferLength);
            }
            validate_view(
                spans.count(),
                spans.input_first(),
                spans.input_stride(),
                inputs,
                u::words(arena, input)?,
            )?;
            validate_view(
                spans.count(),
                spans.output_first(),
                spans.output_stride(),
                p.matrix.outputs(),
                u::words(arena, *output)?,
            )?;
        }
        let scratch_bytes = arena.handle(planes).map_err(u::fault)?.size_in_used();
        let mut prepared = false;
        for (p, output, spans) in jobs {
            if crate::strata_dense::supports_count(
                p.tensor.info().dtype().gguf_id(),
                inputs,
                scratch_bytes,
                spans.count(),
            ) {
                self.executor.enqueue_strata_dense(
                    p.tensor,
                    &u::handle(arena, input)?,
                    arena.handle(planes).map_err(u::fault)?,
                    &u::handle(arena, *output)?,
                    &u::handle(arena, self.fault)?,
                    spans,
                    cancel,
                    !prepared,
                )?;
                prepared = true;
            } else {
                // The ordinary leaf writes only its distinct output and fault;
                // it cannot alter this input or the borrowed packed scratch.
                self.enqueue_batch(*p, arena, input, *output, spans, cancel)?;
            }
        }
        u::cancel(cancel)
    }
    #[allow(clippy::too_many_arguments)]
    fn enqueue_pair(
        &mut self,
        projections: [BoundProjection<'_>; 2],
        arena: &PrivateArena,
        input: BufferId,
        outputs: [BufferId; 2],
        spans: &BatchSpans,
        cancel: &AtomicBool,
    ) -> Result<()> {
        if crate::strata_prompt::active() {
            for (p, out) in projections.into_iter().zip(outputs) {
                self.enqueue_batch(p, arena, input, out, spans, cancel)?;
            }
            return Ok(());
        }
        let supported = self.dense_q8.is_some_and(|id| {
            projections.iter().all(|p| {
                arena.handle(id).is_ok_and(|h| {
                    crate::strata_dense::supports_count(
                        p.tensor.info().dtype().gguf_id(),
                        p.matrix.inputs(),
                        h.size_in_used(),
                        spans.count(),
                    )
                })
            })
        });
        if !supported {
            for (p, out) in projections.into_iter().zip(outputs) {
                self.enqueue_batch(p, arena, input, out, spans, cancel)?;
            }
            return Ok(());
        }
        u::cancel(cancel)?;
        self.executor
            .check_client(arena.client().map_err(u::fault)?)?;
        let planes = self.dense_q8.ok_or(Error::InvalidBufferLength)?;
        let ids = [input, outputs[0], outputs[1], planes, self.fault];
        if ids.iter().enumerate().any(|(i, id)| ids[..i].contains(id))
            || u::words(arena, self.fault)? != 1
            || projections[0].matrix.inputs() != projections[1].matrix.inputs()
        {
            return Err(Error::InvalidBufferLength);
        }
        for (p, out) in projections.iter().zip(outputs) {
            validate_view(
                spans.count(),
                spans.input_first(),
                spans.input_stride(),
                p.matrix.inputs(),
                u::words(arena, input)?,
            )?;
            validate_view(
                spans.count(),
                spans.output_first(),
                spans.output_stride(),
                p.matrix.outputs(),
                u::words(arena, out)?,
            )?;
        }
        for (ordinal, (p, out)) in projections.into_iter().zip(outputs).enumerate() {
            self.executor.enqueue_strata_dense(
                p.tensor,
                &u::handle(arena, input)?,
                arena.handle(planes).map_err(u::fault)?,
                &u::handle(arena, out)?,
                &u::handle(arena, self.fault)?,
                spans,
                cancel,
                ordinal == 0,
            )?;
        }
        u::cancel(cancel)
    }
    fn enqueue_final_head(
        &mut self,
        projection: BoundProjection<'_>,
        arena: &PrivateArena,
        input: BufferId,
        output: BufferId,
        spans: &BatchSpans,
        scratch: crate::resident_gr::batch::FinalHeadScratch<'_>,
        cancel: &AtomicBool,
    ) -> Result<bool> {
        if crate::strata_prompt::active() {
            self.enqueue_batch(projection, arena, input, output, spans, cancel)?;
            return Ok(false);
        }
        // Prefill keeps its original public tile8 policy. Only the completed
        // owner's explicitly selected member1/tile1 decode policy activates
        // integer arithmetic; never select from a private active token count.
        if !self.executor.q6_integer_head
            || self.executor.native_projection_member_capacity() != 1
            || self.executor.native_projection_tile() != 1
        {
            self.enqueue_batch(projection, arena, input, output, spans, cancel)?;
            return Ok(false);
        }
        u::cancel(cancel)?;
        self.executor
            .check_client(arena.client().map_err(u::fault)?)?;
        let planes = scratch.into_planes(arena, input, output, self.fault)?;
        let ids = [input, planes, output, self.fault];
        let mut owners = Vec::with_capacity(ids.len());
        for id in ids {
            owners.push(crate::resident_gr::batch::full_private(
                arena,
                id,
                u::words(arena, id)?,
            )?);
        }
        crate::resident_gr::batch::physical(arena, &owners)?;
        if u::words(arena, self.fault)? != 1 {
            return Err(Error::InvalidBufferLength);
        }
        // SAFETY: consumed Final-only capability, live distinct physical full
        // owners on the admitted stream. No private allocation/upload occurs;
        // original whole-model completion/commit/erase remains the caller's.
        unsafe {
            self.executor.enqueue_q6_integer_final_head(
                projection.tensor,
                &owners[0],
                &owners[1],
                &owners[2],
                &owners[3],
                spans,
                cancel,
            )?;
        }
        u::cancel(cancel)?;
        Ok(true)
    }
    fn enqueue_batched_final_head(
        &mut self,
        projection: BoundProjection<'_>,
        arena: &PrivateArena,
        input: BufferId,
        output: BufferId,
        spans: &BatchSpans,
        scratch: crate::resident_gr::batch::BatchedFinalHeadScratch<'_>,
        cancel: &AtomicBool,
    ) -> Result<bool> {
        if crate::strata_prompt::active() {
            self.enqueue_batch(projection, arena, input, output, spans, cancel)?;
            return Ok(false);
        }
        if !self.executor.q6_integer_verify_head {
            self.enqueue_batch(projection, arena, input, output, spans, cancel)?;
            return Ok(false);
        }
        u::cancel(cancel)?;
        self.executor
            .check_client(arena.client().map_err(u::fault)?)?;
        let (planes, four_reserved) =
            scratch.into_planes(arena, input, output, self.fault, spans)?;
        if four_reserved != self.executor.target_four_rows_reserved
            || u::words(arena, self.fault)? != 1
        {
            return Err(Error::InvalidBufferLength);
        }
        let ids = [input, planes, output, self.fault];
        let mut owners = Vec::with_capacity(ids.len());
        for id in ids {
            owners.push(crate::resident_gr::batch::full_private(
                arena,
                id,
                u::words(arena, id)?,
            )?);
        }
        crate::resident_gr::batch::physical(arena, &owners)?;
        // SAFETY: consumed authentic all-row capability, actual distinct full
        // private owners on the admitted stream, exact immutable constructor
        // capacity, and original GR completion/erasure retained by the caller.
        unsafe {
            self.executor.enqueue_q6_integer_verify_head(
                projection.tensor,
                &owners[0],
                &owners[1],
                &owners[2],
                &owners[3],
                spans,
                cancel,
            )?;
        }
        u::cancel(cancel)?;
        Ok(true)
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
