//! Fixed-public real model HOST -> original enrolled raw-ring copy oracle.
//! Cold session only. Never accepts request text or saves model/payload bytes.
use super::*;
use zeroize::{Zeroize, Zeroizing};
struct Scrub(cubecl::bytes::Bytes);
impl Drop for Scrub {
    fn drop(&mut self) {
        self.0.zeroize();
    }
}
impl Sources {
    fn blob_audit_io(&self) -> Result<u64> {
        let r = cubecl::future::block_on(self.client.sanitize_transient_io_audited())
            .map_err(crate::resident_util::fault)?;
        let n = r
            .upload_bytes
            .checked_add(r.uniform_bytes)
            .and_then(|v| v.checked_add(r.readback_bytes))
            .ok_or(Error::ArithmeticOverflow)?;
        if n == 0 || r.inspected_buffers == 0 || r.inspected_bytes != n {
            return Err(reject());
        }
        Ok(n)
    }
    fn blob_audit_read(&self, raw: &Handle, first: u64, n: usize) -> Result<Scrub> {
        let end = first
            .checked_add(n as u64)
            .ok_or(Error::ArithmeticOverflow)?;
        if n == 0 || end > raw.size_in_used() {
            return Err(Error::InvalidBufferLength);
        }
        self.client
            .read_one(
                raw.clone()
                    .offset_start(first)
                    .offset_end(raw.size_in_used() - end),
            )
            .map(Scrub)
            .map_err(crate::resident_util::fault)
    }
    pub(crate) fn compiled_public_host_blob_probe(
        &self,
        graph: BoundGraph<'_, '_>,
    ) -> Result<serde_json::Value> {
        if !crate::strata_prefill_profile::compiled_public_enabled()
            || !self.plan.has_host_blobs()
            || self.loan_pending
        {
            return Err(reject());
        }
        self.validate_graph_ready(graph)?;
        cubecl::future::block_on(self.client.sync()).map_err(crate::resident_util::fault)?;
        let (cpu_bytes, padding, cpu_experts, cpu_maps) = self
            .host
            .as_ref()
            .ok_or_else(reject)?
            .compiled_public_blob_cpu_audit(&self.plan, graph)?;
        let roots = [
            self.prompt_stage.as_ref().ok_or_else(reject)?.clone(),
            self.prompt_secondary_stage
                .as_ref()
                .ok_or_else(reject)?
                .clone(),
        ];
        validate_prompt_raw_bank_shapes(&roots[0], Some(&roots[1]), roots[0].stream, false)?;
        let mut checked = 0u64;
        let mut gpu_bytes = 0u64;
        let mut negatives = 0u64;
        let mut io_bytes = 0u64;
        let mut reference = Zeroizing::new(vec![0u8; 1024 * 1024]);
        let attempt = (|| -> Result<()> {
            for root in self.plan.host_roots()? {
                let l = &self.plan.layers[root.layer];
                let role_bytes = l.role_expert_bytes.map(|v| v as usize);
                let total = role_bytes.iter().sum::<usize>();
                let slot = crate::strata_prompt::allocation::RAW_RING_BYTES / 8;
                if total as u64 + 16 > slot {
                    return Err(reject());
                }
                let mut cells = vec![
                    l.expert_at(root.first)?,
                    l.expert_at(root.first + root.count - 1)?,
                ];
                cells.dedup();
                for cell in cells {
                    let raw = &roots[(checked as usize) % 2];
                    let first = (7 * slot) as usize;
                    self.client
                        .clear_private_owners(vec![raw.clone()])
                        .map_err(crate::resident_util::fault)?;
                    if self
                        .stage_prompt_group(
                            graph,
                            root.layer,
                            cell,
                            role_bytes,
                            &raw.clone().offset_start(16),
                            first,
                        )
                        .is_ok()
                    {
                        return Err(reject());
                    }
                    negatives += 1;
                    let mut foreign = raw.clone();
                    foreign.stream.value = foreign.stream.value.wrapping_add(1);
                    if self
                        .stage_prompt_group(graph, root.layer, cell, role_bytes, &foreign, first)
                        .is_ok()
                    {
                        return Err(reject());
                    }
                    negatives += 1;
                    if self
                        .stage_prompt_group(graph, root.layer, EXPERTS, role_bytes, raw, first)
                        .is_ok()
                    {
                        return Err(reject());
                    }
                    negatives += 1;
                    if self
                        .stage_prompt_group(
                            graph,
                            root.layer,
                            cell,
                            role_bytes,
                            raw,
                            raw.size_in_used() as usize - 16,
                        )
                        .is_ok()
                    {
                        return Err(reject());
                    }
                    negatives += 1;
                    let mut wrong = role_bytes;
                    wrong[0] += 16;
                    if self
                        .stage_prompt_group(graph, root.layer, cell, wrong, raw, first)
                        .is_ok()
                    {
                        return Err(reject());
                    }
                    negatives += 1;
                    let foreign_owner = self
                        .host
                        .as_ref()
                        .ok_or_else(reject)?
                        .role(graph, &self.plan, root.layer, cell, 0)?
                        .ok_or_else(reject)?
                        .0;
                    if self
                        .stage_prompt_group(
                            graph,
                            root.layer,
                            cell,
                            role_bytes,
                            &foreign_owner,
                            first,
                        )
                        .is_ok()
                    {
                        return Err(reject());
                    }
                    negatives += 1;
                    self.stage_prompt_group(graph, root.layer, cell, role_bytes, raw, first)?;
                    let mut at = first as u64;
                    for (role, n) in role_bytes.into_iter().enumerate() {
                        let name = [
                            "ffn_gate_exps.weight",
                            "ffn_up_exps.weight",
                            "ffn_down_exps.weight",
                        ][role];
                        let tensor = graph.tensor(&format!("blk.{}.{name}", root.layer))?;
                        let mut done = 0;
                        while done < n {
                            let take = (n - done).min(reference.len());
                            tensor.read_range((cell * n + done) as u64, &mut reference[..take])?;
                            {
                                let got = self.blob_audit_read(raw, at + done as u64, take)?;
                                if got.0.len() != take || got.0[..] != reference[..take] {
                                    return Err(Error::InvalidFormat(
                                        "public raw-ring blob bytes differ",
                                    ));
                                }
                            }
                            io_bytes += self.blob_audit_io()?;
                            done += take;
                            gpu_bytes += take as u64;
                        }
                        at += n as u64;
                    }
                    for edge in [first as u64 - 16, first as u64 + total as u64] {
                        {
                            let got = self.blob_audit_read(raw, edge, 16)?;
                            if got.0.iter().any(|&v| v != 0) {
                                return Err(reject());
                            }
                        }
                        io_bytes += self.blob_audit_io()?;
                    }
                    checked += 1;
                }
            }
            Ok(())
        })();
        // Restore ALL original raw owners even if a public comparison failed.
        self.client
            .clear_private_owners(roots.to_vec())
            .map_err(crate::resident_util::fault)?;
        cubecl::future::block_on(self.client.sync()).map_err(crate::resident_util::fault)?;
        let mut zeroed = 0u64;
        for raw in &roots {
            for first in (0..raw.size_in_used()).step_by(1024 * 1024) {
                let n = (raw.size_in_used() - first).min(1024 * 1024) as usize;
                {
                    let got = self.blob_audit_read(raw, first, n)?;
                    if got.0.iter().any(|&v| v != 0) {
                        return Err(reject());
                    }
                }
                io_bytes += self.blob_audit_io()?;
                zeroed += n as u64;
            }
        }
        attempt?;
        self.validate_graph_ready(graph)?;
        Ok(
            serde_json::json!({"compiled_public_fixture_only":true,"content_saved":false,"all_loaded_host_role_bytes_equal_original_gguf":true,"checked_host_role_bytes":cpu_bytes,"checked_logical_blob_padding_bytes":padding,"checked_host_experts":cpu_experts,"actual_scoped_cpu_map_count":cpu_maps,"actual_original_registered_host_to_enrolled_raw_bank_cases":checked,"checked_gpu_copy_payload_bytes":gpu_bytes,"actual_rejected_factory_cases":negatives,"raw_bank_owners":roots.len(),"full_raw_bank_zero_bytes_after_probe":zeroed,"transient_io_zero_audited_bytes":io_bytes,"extra_gpu_root_allocations":0,"gpu_rebase_performed":false,"original_rebase_root_and_inventory_paths_retained":true}),
        )
    }
}
