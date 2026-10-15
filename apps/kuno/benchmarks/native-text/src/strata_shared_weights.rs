//! Original target embedding/head roots, retained once for both target and MTP.
//! MIT Strata fb58e0: NativeEmbed / NativeHead and MtpDrafter::bind.
//! These are public model bytes only; no request data enters this owner.
use crate::{executor::ResidentAdmission, quant_matvec, resident_util as u};
use cubecl::{prelude::ComputeClient, wgpu::WgpuRuntime};
use cubecl_runtime::server::Handle;
use kuno_text::{
    Error, Result,
    model::{BoundGraph, GraphBindingId},
};
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};

pub(crate) const ROWS: usize = 248320;
pub(crate) const COLUMNS: usize = 2560;
pub(crate) const TILE_ROWS: usize = 16384;
pub(crate) const ALIGNMENT: u64 = 4096;
pub(crate) const SOURCE_BYTES: u64 = 64 * 1024 * 1024;
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Tensor {
    Embedding,
    Head,
}
impl Tensor {
    pub(crate) fn name(self) -> &'static str {
        match self {
            Self::Embedding => "token_embd.weight",
            Self::Head => "output.weight",
        }
    }
    pub(crate) fn dtype(self) -> u32 {
        match self {
            Self::Embedding => quant_matvec::IQ4_XS,
            Self::Head => quant_matvec::Q6_K,
        }
    }
    pub(crate) fn named(name: &str) -> Option<Self> {
        [Self::Embedding, Self::Head]
            .into_iter()
            .find(|t| t.name() == name)
    }
    pub(crate) fn row_bytes(self) -> usize {
        quant_matvec::packed_row_bytes(COLUMNS, self.dtype())
    }
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Tile {
    pub first: usize,
    pub rows: usize,
    pub logical: usize,
    pub owned: usize,
}
pub(crate) fn tiles(tensor: Tensor) -> Vec<Tile> {
    (0..ROWS)
        .step_by(TILE_ROWS)
        .map(|first| {
            let rows = TILE_ROWS.min(ROWS - first);
            let logical = rows * tensor.row_bytes();
            let owned = (logical + ALIGNMENT as usize - 1) & !(ALIGNMENT as usize - 1);
            Tile {
                first,
                rows,
                logical,
                owned,
            }
        })
        .collect()
}
pub(crate) fn public_bytes() -> u64 {
    [Tensor::Embedding, Tensor::Head]
        .into_iter()
        .flat_map(tiles)
        .map(|t| t.owned as u64)
        .sum()
}
fn rejected() -> Error {
    Error::InvalidFormat("native shared target weights")
}
fn validate_payload(tensor: Tensor, bytes: &[u8], tile: Tile) -> Result<()> {
    let (block, scale) = match tensor {
        Tensor::Embedding => (136, 0),
        Tensor::Head => (210, 208),
    };
    if bytes.len() != tile.owned
        || tile.logical == 0
        || tile.logical > tile.owned
        || tile.rows == 0
        || tile.rows > TILE_ROWS
        || tile.first.checked_add(tile.rows).is_none_or(|n| n > ROWS)
        || tile.rows.checked_mul(tensor.row_bytes()) != Some(tile.logical)
        || !tile.logical.is_multiple_of(block)
        || bytes[tile.logical..].iter().any(|&b| b != 0)
    {
        return Err(rejected());
    }
    // All remaining packed fields are integers. Finite original half scales
    // bound every decoded value well inside F32 for both original formats.
    if bytes[..tile.logical]
        .chunks_exact(block)
        .any(|b| u16::from_le_bytes([b[scale], b[scale + 1]]) & 0x7c00 == 0x7c00)
    {
        return Err(rejected());
    }
    Ok(())
}
pub(crate) struct Page {
    pub handle: Handle,
    pub tile: Tile,
}
// Token staging uses an original FULL private owner, including zero padding.
// Never pass a sliced logical token view to the strict private upload API.
fn token_upload(tokens: &[u32], owned: usize) -> Result<zeroize::Zeroizing<Vec<u8>>> {
    if tokens.is_empty()
        || tokens.len() > crate::strata_prompt::groups::CAPACITY
        || tokens.iter().any(|&t| t as usize >= ROWS)
        || owned == 0
        || owned > 1024 * 1024
        || !owned.is_multiple_of(4)
        || tokens.len().checked_mul(4).is_none_or(|n| n > owned)
    {
        return Err(rejected());
    }
    let mut bytes = zeroize::Zeroizing::new(Vec::new());
    bytes
        .try_reserve_exact(owned)
        .map_err(|_| Error::LimitExceeded("native token upload owner"))?;
    bytes.resize(owned, 0);
    for (cell, token) in bytes.chunks_exact_mut(4).zip(tokens) {
        cell.copy_from_slice(&token.to_le_bytes());
    }
    Ok(bytes)
}

pub(crate) struct Weights {
    client: ComputeClient<WgpuRuntime>,
    binding: GraphBindingId,
    retired: Arc<AtomicBool>,
    embedding: Vec<Page>,
    head: Vec<Page>,
}
impl Weights {
    pub(crate) fn validate_graph(graph: BoundGraph<'_, '_>) -> Result<()> {
        graph.verify_unchanged()?;
        for tensor in [Tensor::Embedding, Tensor::Head] {
            let t = graph.tensor(tensor.name())?;
            if t.info().dimensions() != [COLUMNS as u64, ROWS as u64]
                || t.info().dtype().gguf_id() != tensor.dtype()
                || t.info().row_bytes() != tensor.row_bytes() as u64
            {
                return Err(rejected());
            }
        }
        Ok(())
    }
    pub(crate) fn load(
        graph: BoundGraph<'_, '_>,
        client: &ComputeClient<WgpuRuntime>,
        admission: &ResidentAdmission,
    ) -> Result<Arc<Self>> {
        Self::validate_graph(graph)?;
        admission.check_client(client)?;
        let alignment = client.properties().memory.alignment;
        if alignment == 0 || !alignment.is_power_of_two() || alignment > ALIGNMENT {
            return Err(rejected());
        }
        let mut owner = Self {
            client: client.clone(),
            binding: graph.binding_id(),
            retired: admission.retired_context(),
            embedding: Vec::new(),
            head: Vec::new(),
        };
        for tensor in [Tensor::Embedding, Tensor::Head] {
            for tile in tiles(tensor) {
                graph.verify_unchanged()?;
                let mut data = vec![0; tile.owned];
                graph.tensor(tensor.name())?.read_range(
                    (tile.first * tensor.row_bytes()) as u64,
                    &mut data[..tile.logical],
                )?;
                validate_payload(tensor, &data, tile)?;
                let source =
                    crate::main_admission::page_bytes(tile.owned as u64).map_err(u::fault)?;
                if source > SOURCE_BYTES {
                    return Err(rejected());
                }
                let handle =
                    admission.allocate_public_roots(&[tile.owned as u64, source], || {
                        crate::expert_banks::exact_persistent::persistent_scope(
                            |mode| {
                                unsafe { client.allocation_mode(mode) };
                                Ok(())
                            },
                            &owner.retired,
                            || Ok(client.create_public_owned(data)),
                        )
                    })?;
                let metadata = client
                    .retained_resource_metadata(vec![handle.clone()])
                    .map_err(u::fault)?;
                if handle.size() != tile.owned as u64
                    || metadata.len() != 1
                    || metadata[0].resource().offset != 0
                    || metadata[0].resource().size != tile.owned as u64
                    || metadata[0].resource().buffer.size() != tile.owned as u64
                {
                    return Err(rejected());
                }
                match tensor {
                    Tensor::Embedding => &mut owner.embedding,
                    Tensor::Head => &mut owner.head,
                }
                .push(Page { handle, tile });
                cubecl::future::block_on(client.sync()).map_err(u::fault)?;
                admission.shared_observation_checkpoint()?;
            }
        }
        graph.verify_unchanged()?;
        Ok(Arc::new(owner))
    }
    pub(crate) fn check(
        &self,
        client: &ComputeClient<WgpuRuntime>,
        binding: GraphBindingId,
    ) -> Result<()> {
        if self.binding != binding
            || !std::ptr::eq(client.info(), self.client.info())
            || self.retired.load(Ordering::Acquire)
        {
            return Err(rejected());
        }
        Ok(())
    }
    pub(crate) fn pages(&self, tensor: Tensor) -> &[Page] {
        match tensor {
            Tensor::Embedding => &self.embedding,
            Tensor::Head => &self.head,
        }
    }
    pub(crate) fn page(&self, tensor: Tensor, first: usize, rows: usize) -> Result<&Page> {
        self.pages(tensor)
            .iter()
            .find(|p| p.tile.first == first && p.tile.rows == rows)
            .ok_or_else(rejected)
    }
}

impl Weights {
    /// Gather once into D, then broadcast to the four HC streams. All request
    /// bytes live in original, fully erased private owners. No new root/view
    /// acquires erasure authority, and no token-dependent shader key is used.
    #[allow(clippy::too_many_arguments)] // Original private owners and current request geometry.
    pub(crate) fn gather(
        &self,
        graph: BoundGraph<'_, '_>,
        arena: &crate::private_arena::PrivateArena,
        token_owner: crate::private_arena::BufferId,
        fault_owner: crate::private_arena::BufferId,
        tokens: &[u32],
        embedding: Handle,
        residual: Handle,
        prompt: bool,
        cancel: &AtomicBool,
    ) -> Result<()> {
        let client = arena.client().map_err(u::fault)?;
        self.check(client, graph.binding_id())?;
        let cap = if prompt {
            crate::strata_prompt::groups::CAPACITY
        } else {
            8
        };
        if tokens.is_empty()
            || tokens.len() > cap
            || tokens.iter().any(|&t| t as usize >= ROWS)
            || embedding.size_in_used() < (tokens.len() * COLUMNS * 4) as u64
            || residual.size_in_used() < (tokens.len() * COLUMNS * 4 * 4) as u64
        {
            return Err(rejected());
        }
        let source = u::handle(arena, token_owner)?;
        let fault = u::handle(arena, fault_owner)?;
        if source.size_in_used() < (tokens.len() * 4) as u64 || fault.size_in_used() != 4 {
            return Err(rejected());
        }
        let handles = [
            source.clone(),
            fault.clone(),
            embedding.clone(),
            residual.clone(),
        ];
        for (i, h) in handles.iter().enumerate() {
            if !(arena.contains_private_view(h).map_err(u::fault)?
                || i >= 2 && prompt && crate::strata_prompt::validate_output(h).is_ok())
            {
                return Err(rejected());
            }
            for other in &handles[..i] {
                if h.clone().binding().memory.allocation_id()
                    == other.clone().binding().memory.allocation_id()
                {
                    let range = |v: &Handle| {
                        v.offset_start.unwrap_or(0)..v.size() - v.offset_end.unwrap_or(0)
                    };
                    let a = range(h);
                    let b = range(other);
                    if a.start < b.end && b.start < a.end {
                        return Err(rejected());
                    }
                }
            }
        }
        u::cancel(cancel)?;
        let upload_owner = arena.handle(token_owner).map_err(u::fault)?.clone();
        let owned = arena.layout(token_owner).map_err(u::fault)?.owned_bytes;
        if upload_owner.offset_start.unwrap_or(0) != 0
            || upload_owner.offset_end.unwrap_or(0) != 0
            || upload_owner.size_in_used() != owned as u64
        {
            return Err(rejected());
        }
        let mut upload = token_upload(tokens, owned)?;
        client
            .write_private(upload_owner, std::mem::take(&mut *upload))
            .map_err(u::fault)?;
        let launch = |kernel, grid, bindings, info| {
            if prompt {
                crate::strata_mtp_front::launch_prompt(arena, kernel, grid, bindings, info)
            } else {
                crate::strata_mtp_front::launch(arena, kernel, grid, bindings, info)
            }
        };
        for page in &self.embedding {
            launch(
                crate::strata_mtp_front::Kernel::Embed,
                [COLUMNS.div_ceil(64), tokens.len(), 1],
                vec![
                    page.handle.clone(),
                    source.clone(),
                    embedding.clone(),
                    fault.clone(),
                ],
                [
                    tokens.len() as u32,
                    0,
                    ROWS as u32,
                    page.tile.rows as u32,
                    page.tile.first as u32,
                    0,
                    0,
                    0,
                ],
            )?;
        }
        launch(
            crate::strata_mtp_front::Kernel::TargetBroadcast,
            [(COLUMNS * 4).div_ceil(64), tokens.len(), 1],
            vec![embedding, residual, fault],
            [tokens.len() as u32, 0, 0, 0, 0, 0, 0, 0],
        )?;
        u::cancel(cancel)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn native_shared_token_upload_covers_the_full_private_owner_and_zeroes_unused_bytes() {
        for count in [1, 8, crate::strata_prompt::groups::CAPACITY] {
            let tokens = (0..count)
                .map(|i| (i * 29 % ROWS) as u32)
                .collect::<Vec<_>>();
            let bytes = token_upload(&tokens, 1024 * 1024).unwrap();
            assert_eq!(bytes.len(), 1024 * 1024);
            for (cell, token) in bytes.chunks_exact(4).zip(&tokens) {
                assert_eq!(u32::from_le_bytes(cell.try_into().unwrap()), *token);
            }
            assert!(bytes[count * 4..].iter().all(|&b| b == 0));
            assert!(token_upload(&tokens, count * 4 - 1).is_err());
        }
        for (tokens, size) in [
            (vec![], 32768),
            (vec![ROWS as u32], 32768),
            (vec![u32::MAX], 32768),
            (vec![1], 0),
            (vec![1], 5),
            (vec![1], 1024 * 1024 + 4),
            (vec![1; crate::strata_prompt::groups::CAPACITY + 1], 32768),
        ] {
            assert!(token_upload(&tokens, size).is_err());
        }
        assert!(token_upload(&[ROWS as u32 - 1], 4).is_ok());
    }

    #[test]
    fn native_shared_weights_original_tiles_cover_every_row_once_with_exact_roots() {
        assert_eq!(Tensor::Embedding.row_bytes(), 1360);
        assert_eq!(Tensor::Head.row_bytes(), 2100);
        assert_eq!(public_bytes(), 859_189_248);
        for role in [Tensor::Embedding, Tensor::Head] {
            let all = tiles(role);
            assert_eq!(all.len(), 16);
            assert_eq!(all.last().unwrap().rows, 2560);
            let mut next = 0;
            for p in all {
                assert_eq!(p.first, next);
                assert_eq!(p.logical, p.rows * role.row_bytes());
                assert!(p.logical <= p.owned && p.owned - p.logical < 4096);
                assert!(p.owned.is_multiple_of(4096));
                assert!(crate::main_admission::page_bytes(p.owned as u64).unwrap() <= SOURCE_BYTES);
                next += p.rows;
            }
            assert_eq!(next, ROWS);
        }
    }
    #[test]
    fn native_shared_weights_validate_every_original_half_scale_and_padding() {
        for role in [Tensor::Embedding, Tensor::Head] {
            let tile = *tiles(role).last().unwrap();
            let mut data = vec![0; tile.owned];
            let (block, scale) = match role {
                Tensor::Embedding => (136, 0),
                Tensor::Head => (210, 208),
            };
            for start in [scale, tile.logical - block + scale] {
                for bits in [0u16, 0x8000, 0x7bff, 0xfbff] {
                    data[start..start + 2].copy_from_slice(&bits.to_le_bytes());
                    validate_payload(role, &data, tile).unwrap();
                }
                for bits in [0x7c00u16, 0xfc00, 0x7fff, 0xffff] {
                    data[start..start + 2].copy_from_slice(&bits.to_le_bytes());
                    assert!(validate_payload(role, &data, tile).is_err());
                }
                data[start..start + 2].fill(0);
            }
            if tile.owned > tile.logical {
                data[tile.logical] = 1;
                assert!(validate_payload(role, &data, tile).is_err());
            }
            assert!(validate_payload(role, &data[..data.len() - 1], tile).is_err());
        }
    }
}

#[cfg(test)]
mod backlog_tests {
    use crate::target_lab_build::native_shared_backlog_record as record;
    use cubecl_runtime::server::PublicTargetSubmissionBacklog as P;
    #[test]
    fn native_shared_weights_backlog_uses_the_same_observed_policy_at_entry_and_exit() {
        for selected in [P::Default512, P::Queued2048] {
            if !crate::strata_intel_core::enabled() {
                assert!(record(selected, Some(selected), true).is_err());
                continue;
            }
            let initial = record(selected, Some(selected), true).unwrap();
            let closed = record(selected, Some(selected), true).unwrap();
            assert_eq!(initial, closed);
            assert_eq!(initial["actual_task_window"], selected.task_window());
            assert!(record(selected, None, true).is_err());
            assert!(record(selected, Some(selected), false).is_err());
            let foreign = if selected == P::Default512 {
                P::Queued2048
            } else {
                P::Default512
            };
            assert!(record(selected, Some(foreign), true).is_err());
        }
    }
}
