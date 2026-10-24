//! Compiled-public cold-owner audit of EVERY loaded HOST expert role byte.
use super::*;
impl Host {
    pub(crate) fn compiled_public_blob_cpu_audit(
        &self,
        plan: &GlobalPlan,
        graph: BoundGraph<'_, '_>,
    ) -> KResult<(u64, u64, u64, u64)> {
        if !crate::strata_prefill_profile::compiled_public_enabled()
            || !self.packed
            || !plan.has_host_blobs()
        {
            return Err(fail());
        }
        plan.check_binding(graph)?;
        let mut reference = Zeroizing::new(vec![0u8; SOURCE_WINDOW as usize]);
        let (mut bytes, mut padding, mut experts, mut maps) = (0u64, 0u64, 0u64, 0u64);
        for (layer, l) in plan.layers.iter().enumerate() {
            let tensors = [
                graph.tensor(&format!("blk.{layer}.ffn_gate_exps.weight"))?,
                graph.tensor(&format!("blk.{layer}.ffn_up_exps.weight"))?,
                graph.tensor(&format!("blk.{layer}.ffn_down_exps.weight"))?,
            ];
            let cells = (l.resident..crate::strata_global_plan::EXPERTS)
                .map(|v| l.expert_at(v))
                .collect::<KResult<Vec<_>>>()?;
            for batch in cells.chunks(80) {
                let view = self.borrow_cpu(graph, plan, layer, batch)?;
                if view.maps.len() > batch.len() {
                    return Err(fail());
                }
                maps += view.maps.len() as u64;
                for &cell in batch {
                    for (role, tensor) in tensors.iter().enumerate() {
                        let actual = view.role(graph, plan, layer, cell, role)?;
                        if actual.len() as u64 != l.role_expert_bytes[role] {
                            return Err(fail());
                        }
                        for (first, part) in actual.chunks(SOURCE_WINDOW as usize).enumerate() {
                            let target = &mut reference[..part.len()];
                            tensor.read_range(
                                (cell as u64) * l.role_expert_bytes[role]
                                    + (first as u64) * SOURCE_WINDOW,
                                target,
                            )?;
                            if part != target {
                                return Err(Error::InvalidFormat(
                                    "public HOST blob original bytes differ",
                                ));
                            }
                            bytes += part.len() as u64;
                        }
                    }
                    let data = l.role_expert_bytes.iter().sum::<u64>();
                    let stride = plan.host_blob_stride(layer)? as u64;
                    if stride < data {
                        return Err(fail());
                    }
                    if stride > data {
                        let GlobalSource::Host { piece, offset, .. } =
                            plan.source(layer, cell, 0)?
                        else {
                            return Err(fail());
                        };
                        let map = view
                            .maps
                            .iter()
                            .find(|(r, p, _)| {
                                *r == crate::strata_global_plan::PACKED_HOST_ROLE && *p == piece
                            })
                            .ok_or_else(fail)?;
                        let gap = map
                            .2
                            .span(offset + data, stride - data)
                            .map_err(Error::InvalidFormat)?;
                        if gap.iter().any(|&v| v != 0) {
                            return Err(fail());
                        }
                        padding += gap.len() as u64;
                    }
                    experts += 1;
                }
            }
        }
        if bytes != plan.host_logical_bytes
            || bytes.checked_add(padding) != Some(plan.host_registered_bytes())
        {
            return Err(fail());
        }
        graph.verify_unchanged()?;
        Ok((bytes, padding, experts, maps))
    }
}
