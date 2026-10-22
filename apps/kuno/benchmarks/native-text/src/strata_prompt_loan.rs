//! Joined prompt-only source publication around three genuine cache owners.
//! The session never grants decode/adaptation access to private scratch roots.
use super::*;
use crate::private_arena::{BufferId, OwnedZeroAudit, PrivateArena};
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};

pub(crate) struct BorrowedPrompt {
    instance: u64,
    indices: [usize; 3],
    pub(super) arena: Option<PrivateArena>,
    pub(super) ids: [BufferId; 3],
    expected: OwnedZeroAudit,
    old_prompt_stage: Option<Handle>,
    retired: Arc<AtomicBool>,
    completed: bool,
}
impl BorrowedPrompt {
    /// Scope cannot outlive the protected same-worker session (unsafe creation
    /// contract below); actual IDs belong to this registered arena, not MAIN.
    pub(crate) fn scope(&self) -> Result<crate::strata_prompt::Scope> {
        if self.completed || self.retired.load(Ordering::Acquire) {
            return Err(reject());
        }
        crate::strata_prompt::Scope::new(self.arena.as_ref().ok_or_else(reject)?, self.ids)
    }
}
impl Drop for BorrowedPrompt {
    fn drop(&mut self) {
        if !self.completed {
            // An abandoned session cannot publish original model residency.
            // Attempt private cleanup while preserving the source exclusion;
            // owner close retains the cache on this shared retired context.
            if let Some(a) = self.arena.as_mut() {
                let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| a.close()));
            }
            self.retired.store(true, Ordering::Release);
        }
    }
}
impl Sources {
    pub(crate) fn prompt_loan_quote(&self, graph: BoundGraph<'_, '_>) -> Result<([usize; 3], u64)> {
        self.validate_graph_ready(graph)?;
        single_prompt_raw_bank_only(self.prompt_secondary_stage.as_ref())?;
        let roots = self.gpu.as_ref().ok_or_else(reject)?.roots();
        // Native tail lending, skipping only an undersized physical final root.
        // Selection is based on real contiguous owners and actual carve checks.
        for first in (0..roots.len().saturating_sub(2)).rev() {
            let indices = [first, first + 1, first + 2];
            if crate::strata_prompt::storage::Storage::new(indices.map(|i| roots[i].clone()))
                .is_err()
            {
                continue;
            }
            let quote = Host::loan_shadow_layout(graph, &self.plan, indices)?.reservation()?;
            return Ok((indices, quote));
        }
        Err(kuno_text::Error::LimitExceeded(
            "native prompt cache loan has no fitting tail",
        ))
    }
    /// Install one persistent fallback for this model's chosen lending roots.
    /// # Safety
    /// Complete authenticated original model and exact extra HOST reservation
    /// are held through Sources close. Same sole protected worker/client.
    pub(crate) unsafe fn prepare_prompt_loan(
        &mut self,
        graph: BoundGraph<'_, '_>,
        indices: [usize; 3],
        reserved: u64,
    ) -> Result<()> {
        self.validate_graph_ready(graph)?;
        single_prompt_raw_bank_only(self.prompt_secondary_stage.as_ref())?;
        let layout = Host::loan_shadow_layout(graph, &self.plan, indices)?;
        if reserved != layout.reservation()? {
            return Err(reject());
        }
        let gpu = self.gpu.as_ref().ok_or_else(reject)?;
        let roots: [Handle; 3] = indices
            .iter()
            .map(|&i| gpu.roots().get(i).cloned().ok_or_else(reject))
            .collect::<Result<Vec<_>>>()?
            .try_into()
            .map_err(|_| reject())?;
        crate::strata_prompt::storage::Storage::new(roots)?;
        self.mutation_pending = true;
        let prepared = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| unsafe {
            self.host.as_mut().ok_or_else(reject)?.create_loan_shadow(
                graph,
                &self.plan,
                layout,
                &self.placement,
                reserved,
            )
        }));
        if !matches!(prepared, Ok(Ok(()))) {
            self.closing_failed = true;
            self.retired.store(true, Ordering::Release);
            return Err(reject());
        }
        self.mutation_pending = false;
        Ok(())
    }
    /// Start real registered private ownership, after redirecting all lent
    /// physical-cell pointers/prototypes to initialized original HOST weights.
    /// # Safety
    /// Same protected serialized worker holds all precharges and owns the full
    /// session. No raw handle, Scope, ID or replay may escape its finish call.
    pub(crate) unsafe fn begin_prompt_loan(
        &mut self,
        graph: BoundGraph<'_, '_>,
    ) -> Result<BorrowedPrompt> {
        self.validate_graph_ready(graph)?;
        single_prompt_raw_bank_only(self.prompt_secondary_stage.as_ref())?;
        if crate::strata_prompt::active() {
            return Err(reject());
        }
        let indices = self
            .host
            .as_ref()
            .and_then(Host::loan_shadow_indices)
            .ok_or_else(reject)?;
        let roots: [Handle; 3] = indices
            .iter()
            .map(|&i| {
                self.gpu
                    .as_ref()
                    .ok_or_else(reject)?
                    .roots()
                    .get(i)
                    .cloned()
                    .ok_or_else(reject)
            })
            .collect::<Result<Vec<_>>>()?
            .try_into()
            .map_err(|_| reject())?;
        crate::strata_prompt::storage::Storage::new(roots.clone())?;
        let expected = OwnedZeroAudit {
            bytes: roots.iter().map(Handle::size_in_used).sum(),
            buffers: 3,
        };
        cubecl_common::future::block_on(self.client.sync()).map_err(crate::resident_util::fault)?;
        self.loan_pending = true;
        self.mutation_pending = true;
        self.loan_ready = false;
        let old_prompt_stage = self.prompt_stage.take();
        let prepared = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| -> Result<_> {
            unsafe {
                self.host.as_mut().ok_or_else(reject)?.refresh_loan_shadow(
                    graph,
                    &self.plan,
                    &self.placement,
                )?;
            }
            self.build_tables(graph, &self.device.clone())?;
            cubecl_common::future::block_on(self.client.sync())
                .map_err(crate::resident_util::fault)?;
            // Exclusion and tables are published BEFORE a single scratch byte
            // is initialized. All physical BDA read sets exclude these roots.
            self.prompt_stage = Some(roots[2].clone());
            unsafe {
                PrivateArena::register_cache_roots_on_configured(
                    self.client.clone(),
                    roots,
                    self.retired.clone(),
                )
            }
            .map_err(crate::resident_util::fault)
        }));
        let (arena, ids) = match prepared {
            Ok(Ok(v)) => v,
            _ => {
                self.closing_failed = true;
                self.retired.store(true, Ordering::Release);
                return Err(reject());
            }
        };
        self.loan_ready = true;
        Ok(BorrowedPrompt {
            instance: self.plan.instance(),
            indices,
            arena: Some(arena),
            ids,
            expected,
            old_prompt_stage,
            retired: self.retired.clone(),
            completed: false,
        })
    }
    /// Erasure/inspection and exact current-placement restoration precede any
    /// ordinary model pointer/read-set publication. Error/panic leaves exclusion.
    pub(crate) fn finish_prompt_loan(
        &mut self,
        graph: BoundGraph<'_, '_>,
        mut loan: BorrowedPrompt,
    ) -> Result<()> {
        if self.unavailable()
            || !self.loan_pending
            || !self.loan_ready
            || !self.mutation_pending
            || crate::strata_prompt::active()
            || loan.instance != self.plan.instance()
            || !Arc::ptr_eq(&loan.retired, &self.retired)
            || self.host.as_ref().and_then(Host::loan_shadow_indices) != Some(loan.indices)
        {
            return Err(reject());
        }
        self.plan.check_binding(graph)?;
        self.loan_ready = false;
        let completed = finish_borrowed_roots(
            loan.expected,
            || {
                let a = loan.arena.as_mut().ok_or_else(reject)?;
                a.close().map_err(crate::resident_util::fault)?;
                a.owned_zero_audit_receipt().ok_or_else(reject)
            },
            || {
                self.gpu.as_ref().ok_or_else(reject)?.restore_chunks(
                    graph,
                    &self.plan,
                    loan.indices,
                    &self.placement,
                )
            },
        );
        if completed.is_err() {
            self.closing_failed = true;
            self.retired.store(true, Ordering::Release);
            return Err(reject());
        }
        // Closed arena is idempotent and cannot erase the restored model.
        loan.arena.take();
        self.prompt_stage = loan.old_prompt_stage.take();
        // All observers remain excluded by mutation_pending until the ordinary
        // tables and queue completion are restored. Unknown partial writes retire.
        self.loan_pending = false;
        let published = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| -> Result<()> {
            self.build_tables(graph, &self.device.clone())?;
            cubecl_common::future::block_on(self.client.sync())
                .map_err(crate::resident_util::fault)?;
            graph.verify_unchanged()
        }));
        if !matches!(published, Ok(Ok(()))) {
            self.closing_failed = true;
            self.retired.store(true, Ordering::Release);
            return Err(reject());
        }
        self.mutation_pending = false;
        loan.completed = true;
        Ok(())
    }
}
