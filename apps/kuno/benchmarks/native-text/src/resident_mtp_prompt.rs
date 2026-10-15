//! MIT Strata fb58e0 prefill.cpp::draft_kv, Intel default FP16/Q8 GEMMs.
//! Borrows the completed target chunk's device carry and now-idle phase region.
use super::*;
use crate::{strata_mtp_front as f, strata_prompt as p};
impl ResidentMtp<'_, '_> {
    pub(crate) fn append_prompt_device(
        &mut self,
        tokens: &[u32],
        source_client: &ComputeClient<WgpuRuntime>,
        carry: Handle,
        cancel: &AtomicBool,
    ) -> Result<()> {
        let count = tokens.len();
        if !p::active()
            || self.graph.profile() != MtpProfile::StrataQ2
            || self.poisoned
            || self.closed
            || self.retired.load(Ordering::Acquire)
            || self.live_outputs.load(Ordering::Acquire) != 0
            || !(1..=p::groups::CAPACITY).contains(&count)
            || self
                .position
                .checked_add(count)
                .is_none_or(|end| end > k::CONTEXT)
            || tokens.iter().any(|&t| t as usize >= k::VOCAB)
            || carry.size_in_used() < (count * k::WIDE * 4) as u64
            || !std::ptr::eq(source_client.info(), self.client()?.info())
        {
            return Err(crate::strata_prompt::invalid_buffer_length());
        }
        p::validate_output(&carry)?;
        self.verify_source()?;
        self.physical()?;
        u::cancel(cancel)?;
        self.last_forward.take();
        self.last_front.take();
        self.poisoned = true;
        self.dispatches = 0;
        let attempted = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| -> Result<()> {
            self.clear_scratch()?;
            self.queue_prompt_device(tokens, carry, cancel)?;
            self.guard_private(self.position + count)?;
            self.complete_fault()?;
            self.verify_source()?;
            u::cancel(cancel)
        }));
        match attempted {
            Ok(Ok(())) => {
                self.position += count;
                self.poisoned = false;
                Ok(())
            }
            Ok(Err(e)) => {
                self.fail();
                Err(e)
            }
            Err(_) => {
                self.fail();
                Err(Error::InvalidFormat("large MTP teacher failure"))
            }
        }
    }
    fn prompt_project(
        &mut self,
        name: &'static str,
        input: Handle,
        output: Handle,
        count: usize,
        cancel: &AtomicBool,
    ) -> Result<()> {
        let spec = *matrices_for(self.graph.profile())
            .iter()
            .find(|s| s.name == name)
            .ok_or(Error::InvalidBufferLength)?;
        if spec.dtype != quant_matvec::Q8_0 {
            return Err(crate::strata_prompt::invalid_buffer_length());
        }
        let fault = u::handle(&self.arena, self.id(B::Fault))?;
        let pages = self.public()?.pages(name)?.len();
        for page in self.public()?.pages(name)? {
            u::cancel(cancel)?;
            p::project(
                self.client()?,
                spec.dtype,
                spec.columns,
                page.rows,
                spec.columns / 32 * 34,
                page.handle.clone(),
                input.clone(),
                output.clone(),
                fault.clone(),
                count,
                0,
                spec.columns,
                page.first,
                spec.rows,
                false,
                false,
            )?;
        }
        self.tick(pages * 3)
    }
    fn queue_prompt_device(
        &mut self,
        tokens: &[u32],
        carry: Handle,
        cancel: &AtomicBool,
    ) -> Result<()> {
        let count = tokens.len();
        let start = self.position;
        let v = p::outer_workspace()?;
        let phase = p::ple_workspace()?;
        // Outer R1 is the immutable target carry. R0 becomes the MTP residual;
        // the phase region is no longer used by target attention or PLE.
        let embedding = v[11].clone();
        let en = v[2].clone();
        let e2 = v[3].clone();
        let hn = phase[3].clone();
        let residual = v[0].clone();
        let token_view = phase[10].clone();
        let status = phase[9].clone();
        let key = v[6].clone();
        let value = v[11].clone();
        let rotated = v[9].clone();
        let fault = u::handle(&self.arena, self.id(B::Fault))?;
        let ids = *self
            .native_front
            .as_ref()
            .ok_or(Error::InvalidBufferLength)?
            .ids;
        let positions = u::handle(&self.arena, ids[f::POS])?;
        let mut upload = Zeroizing::new(vec![0u8; count * 4]);
        for (word, t) in upload.chunks_exact_mut(4).zip(tokens) {
            word.copy_from_slice(&t.to_le_bytes());
        }
        // Same shared native client and active target prompt scope; the
        // scalar-token owner is scratch during whole-prefix teacher forcing.
        p::stage_private_view(&self.arena, self.id(B::Token), token_view.clone(), &upload)?;
        let info = [count as u32, start as u32, 0, 0, 0, 0, EPS.to_bits(), 0];
        let pages = self.public()?.pages(EMBEDDING.name)?.len();
        for page in self.public()?.pages(EMBEDDING.name)? {
            let mut q = info;
            q[2] = k::VOCAB as u32;
            q[3] = page.rows as u32;
            q[4] = page.first as u32;
            f::launch_prompt(
                &self.arena,
                f::Kernel::Embed,
                [40, count, 1],
                vec![
                    page.handle.clone(),
                    token_view.clone(),
                    embedding.clone(),
                    fault.clone(),
                ],
                q,
            )?;
        }
        self.tick(pages)?;
        for (input, output, gamma, columns) in [
            (embedding, en.clone(), "blk.48.nextn.enorm.weight", k::D),
            (carry, hn.clone(), "blk.48.nextn.hnorm.weight", k::WIDE),
        ] {
            let mut q = info;
            q[2] = columns as u32;
            f::launch_prompt(
                &self.arena,
                f::Kernel::Norm,
                [count, 1, 1],
                vec![input, self.front_public(gamma)?, output, fault.clone()],
                q,
            )?;
        }
        self.tick(2)?;
        self.prompt_project(strata::FC_EMBEDDING, en, e2.clone(), count, cancel)?;
        self.prompt_project(
            strata::FC_HIDDEN,
            hn,
            residual.clone(),
            count * k::HC,
            cancel,
        )?;
        f::launch_prompt(
            &self.arena,
            f::Kernel::Broadcast,
            [160, count, 1],
            vec![e2, residual.clone(), fault.clone()],
            info,
        )?;
        self.tick(1)?;
        p::gr_handles(
            p::GrHandles {
                client: self.client()?,
                residual,
                previous: None,
                gamma: self.front_public("blk.48.hc_attn_norm.weight")?,
                injection: v[5].clone(),
                mixed: v[2].clone(),
                fault: fault.clone(),
                count,
                first: 0,
                epsilon: EPS,
                inject: true,
            },
            [
                self.front_public("blk.48.hc_attn_down.weight")?,
                self.front_public("blk.48.hc_attn_up.weight")?,
                self.front_public("blk.48.hc_attn_inject.weight")?,
            ],
        )?;
        self.tick(9)?;
        self.prompt_project(
            "blk.48.attn_k.weight",
            v[2].clone(),
            key.clone(),
            count,
            cancel,
        )?;
        self.prompt_project(
            "blk.48.attn_v.weight",
            v[2].clone(),
            value.clone(),
            count,
            cancel,
        )?;
        f::launch_prompt(
            &self.arena,
            f::Kernel::Positions,
            [count.div_ceil(64), 1, 1],
            vec![positions.clone()],
            info,
        )?;
        crate::strata_qsa::launch(
            &self.arena,
            crate::strata_qsa::Kernel::PromptNorm,
            [k::KVH, count, 1],
            vec![
                key,
                self.front_public("blk.48.attn_k_norm.weight")?,
                positions,
                rotated.clone(),
                status.clone(),
            ],
            [
                count as u32,
                start as u32,
                k::CONTEXT as u32,
                k::KV as u32,
                k::KV as u32,
                k::KVH as u32,
                k::KVH as u32,
                k::HD as u32,
                0,
                EPS.to_bits(),
                crate::qsa_gpu::THETA.powf(-2. / 64.).to_bits(),
                0,
            ],
        )?;
        p::head_flags(self.client()?, status, fault.clone(), count, k::KVH)?;
        f::launch_prompt(
            &self.arena,
            f::Kernel::Append,
            [4, count, 1],
            vec![
                rotated,
                value,
                u::handle(&self.arena, self.id(B::Keys))?,
                u::handle(&self.arena, self.id(B::Values))?,
                fault,
            ],
            info,
        )?;
        self.tick(4)?;
        u::cancel(cancel)
    }
}
