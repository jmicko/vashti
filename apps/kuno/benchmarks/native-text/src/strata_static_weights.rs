//! Pinned Strata fb58e0 Intel: load immutable dense weights before sizing the
//! ranked expert prefix. The selected target has one complete tile per matrix;
//! the old fixed-capacity LRU is not a native inference weight owner.
use super::*;
use kuno_text::model::{BoundGraph, GraphBindingId};

pub(crate) const HOST_BYTES: usize = 512 * 1024;
const COUNT: usize = 784;
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct Tile {
    inputs: usize,
    pub(super) rows: usize,
    row_bytes: usize,
    dtype: u32,
    pub(super) payload: usize,
    pub(super) owned: usize,
    pub(super) source: u64,
}
impl Tile {
    fn new(
        inputs: usize,
        rows: usize,
        row_bytes: usize,
        dtype: u32,
        alignment: u64,
    ) -> Result<Self> {
        if alignment == 0 || !alignment.is_power_of_two() || alignment > 4096 {
            return Err(rejected());
        }
        let alignment = alignment.max(256) as usize;
        // Every selected matrix must be the same single complete tile in the
        // original GR, prompt, packed decode and scalar-gate consumers. Never
        // silently reuse first=0 with a different cached row count.
        let expanded = inputs.checked_mul(2).ok_or(Error::ArithmeticOverflow)?;
        if inputs == 0
            || rows == 0
            || row_bytes == 0
            || expanded > TILE_BYTES
            || rows > 16384
            || row_bytes > TILE_BYTES
            || rows > TILE_BYTES / row_bytes
            || rows > TILE_BYTES / expanded
        {
            return Err(rejected());
        }
        let (block, packed) = quant_matvec::block_layout(dtype).ok_or_else(rejected)?;
        if !inputs.is_multiple_of(block) || (inputs / block).checked_mul(packed) != Some(row_bytes)
        {
            return Err(rejected());
        }
        let bytes = rows
            .checked_mul(row_bytes)
            .ok_or(Error::ArithmeticOverflow)?;
        let payload = bytes.checked_add(3).ok_or(Error::ArithmeticOverflow)? / 4 * 4;
        let owned = payload
            .checked_add(alignment - 1)
            .ok_or(Error::ArithmeticOverflow)?
            / alignment
            * alignment;
        let source = main_admission::page_bytes(owned as u64).map_err(admission_error)?;
        if source > crate::strata_shared_weights::SOURCE_BYTES {
            return Err(rejected());
        }
        Ok(Self {
            inputs,
            rows,
            row_bytes,
            dtype,
            payload,
            owned,
            source,
        })
    }
}
fn rejected() -> Error {
    Error::InvalidFormat("native dense weight inventory")
}
struct Geometry {
    name: String,
    inputs: usize,
    rows: usize,
}
fn geometries() -> Vec<Geometry> {
    let mut out = Vec::with_capacity(COUNT);
    let mut add = |name: String, inputs, rows| out.push(Geometry { name, inputs, rows });
    for layer in 0..48 {
        for family in ["attn", "ffn"] {
            for (role, i, o) in [
                ("down", 10240, 320),
                ("up", 320, 10240),
                ("inject", 10240, 4),
            ] {
                add(format!("blk.{layer}.hc_{family}_{role}.weight"), i, o);
            }
        }
        for (role, i, o) in [
            ("ffn_gate_inp", 2560, 512),
            ("ffn_gate_shexp", 2560, 640),
            ("ffn_up_shexp", 2560, 640),
            ("ffn_down_shexp", 640, 2560),
            ("ffn_gate_inp_shexp", 2560, 1),
        ] {
            add(format!("blk.{layer}.{role}.weight"), i, o);
        }
        let roles: &[(&str, usize, usize)] = if layer % 4 == 3 {
            &[
                ("attn_q", 2560, 12288),
                ("attn_k", 2560, 512),
                ("attn_v", 2560, 512),
                ("attn_output", 6144, 2560),
                ("indexer.q_proj", 2560, 512),
                ("indexer.k_proj", 2560, 128),
            ]
        } else {
            &[
                ("attn_qkv", 2560, 10240),
                ("attn_gate", 2560, 6144),
                ("ssm_alpha", 2560, 48),
                ("ssm_beta", 2560, 48),
                ("ssm_out", 6144, 2560),
            ]
        };
        for &(role, i, o) in roles {
            add(format!("blk.{layer}.{role}.weight"), i, o);
        }
    }
    for (name, i, o) in [
        ("output_hc_down.weight", 10240, 320),
        ("output_hc_up.weight", 320, 10240),
        ("blk.1.ple_key.weight", 2560, 10240),
        ("blk.1.ple_value.weight", 2560, 2560),
    ] {
        add(name.into(), i, o);
    }
    out
}
fn excluded(graph: BoundGraph<'_, '_>, name: &str) -> bool {
    matches!(
        name,
        "token_embd.weight" | "output.weight" | "per_layer_token_embd.weight"
    ) || name.ends_with("_exps.weight")
        || graph.static_vectors().any(|(n, _)| n == name)
}
pub(super) struct Plan {
    binding: GraphBindingId,
    tiles: BTreeMap<String, Tile>,
    pub(super) bytes: usize,
    pub(super) sealed: bool,
}
impl Plan {
    pub(super) fn from_graph(graph: BoundGraph<'_, '_>, alignment: u64) -> Result<Self> {
        if !crate::strata_intel_core::enabled() {
            return Err(rejected());
        }
        graph.verify_unchanged()?;
        let mut tiles = BTreeMap::new();
        let mut bytes = 0usize;
        for g in geometries() {
            let t = graph.tensor(&g.name)?;
            if strata_projection_shape(t)? != [g.inputs, g.rows, 1]
                || crate::daslab_profile::expected_dtype(&g.name)
                    != Some(t.info().dtype().gguf_id())
            {
                return Err(rejected());
            }
            let tile = Tile::new(
                g.inputs,
                g.rows,
                usize::try_from(t.info().row_bytes()).map_err(|_| Error::ArithmeticOverflow)?,
                t.info().dtype().gguf_id(),
                alignment,
            )?;
            bytes = bytes
                .checked_add(tile.owned)
                .ok_or(Error::ArithmeticOverflow)?;
            if tiles.insert(g.name, tile).is_some() {
                return Err(rejected());
            }
        }
        if tiles.len() != COUNT
            || graph
                .tensor_names()
                .any(|name| !tiles.contains_key(name) && !excluded(graph, name))
        {
            return Err(rejected());
        }
        Ok(Self {
            binding: graph.binding_id(),
            tiles,
            bytes,
            sealed: false,
        })
    }
    pub(super) fn total(&self) -> Result<u64> {
        (self.bytes as u64)
            .checked_add(crate::strata_shared_weights::public_bytes())
            .ok_or(Error::ArithmeticOverflow)
    }
    pub(super) fn tile(
        &self,
        t: TensorView<'_>,
        expert: usize,
        first: usize,
        rows: usize,
    ) -> Result<Tile> {
        let tile = *self.tiles.get(t.info().name()).ok_or_else(rejected)?;
        if (expert, first, rows) != (0, 0, tile.rows)
            || strata_projection_shape(t)? != [tile.inputs, tile.rows, 1]
            || t.info().row_bytes() != tile.row_bytes as u64
            || t.info().dtype().gguf_id() != tile.dtype
        {
            return Err(rejected());
        }
        Ok(tile)
    }
}
impl GpuExecutor<'_> {
    pub(crate) fn prepare_native_static_weights(
        &mut self,
        graph: BoundGraph<'_, '_>,
    ) -> Result<()> {
        let result = self.prepare_native_static_weights_inner(graph);
        if result.is_err() {
            self.poison_resident();
        }
        result
    }
    fn prepare_native_static_weights_inner(&mut self, graph: BoundGraph<'_, '_>) -> Result<()> {
        graph.verify_unchanged()?;
        let plan = self.native_static.as_ref().ok_or_else(rejected)?;
        let admission = self
            .resident_admission
            .as_ref()
            .ok_or_else(rejected)?
            .clone();
        if plan.binding != graph.binding_id()
            || admission.binding != graph.binding_id()
            || !self.banks.is_empty()
            || self.strata_global_sources.is_some()
        {
            return Err(rejected());
        }
        if plan.sealed {
            return Err(rejected());
        }
        if !self.cache.is_empty() || self.cache_bytes != 0 {
            return Err(rejected());
        }
        // Public names/geometry only. Peak plan + iterator metadata is charged
        // in CORE_CACHE_HOST, independent of private prompt/output ownership.
        let inventory = plan
            .tiles
            .iter()
            .map(|(n, t)| (n.clone(), *t))
            .collect::<Vec<_>>();
        for (name, tile) in inventory {
            graph.verify_unchanged()?;
            let t = graph.tensor(&name)?;
            if !t.belongs_to(self.model) {
                return Err(rejected());
            }
            self.weight(t, 0, 0, tile.rows)?;
            // Each actual source upload is joined/released before another is
            // allocated. No invented source fence or accumulated staging debt.
            cubecl::future::block_on(self.client.sync()).map_err(|_| rejected())?;
            admission.shared_observation_checkpoint()?;
        }
        graph.verify_unchanged()?;
        let plan = self.native_static.as_mut().ok_or_else(rejected)?;
        if self.cache.len() != plan.tiles.len() || self.cache_bytes != plan.bytes {
            return Err(rejected());
        }
        plan.sealed = true;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn complete_selected_inventory_includes_every_family_and_no_request_or_expert_source() {
        let g = geometries();
        let names = g
            .iter()
            .map(|g| g.name.as_str())
            .collect::<std::collections::BTreeSet<_>>();
        assert_eq!(g.len(), COUNT);
        assert_eq!(names.len(), COUNT);
        assert_eq!(g.iter().filter(|g| g.name.contains(".hc_")).count(), 288);
        assert_eq!(
            g.iter()
                .filter(|g| g.name.ends_with(".attn_qkv.weight"))
                .count(),
            36
        );
        assert_eq!(
            g.iter()
                .filter(|g| g.name.ends_with(".indexer.q_proj.weight"))
                .count(),
            12
        );
        assert_eq!(
            g.iter()
                .filter(|g| g.name.ends_with(".ffn_gate_inp_shexp.weight"))
                .count(),
            48
        );
        for g in &g {
            assert!(
                !g.name.contains("_exps.")
                    && !g.name.contains("token_embd")
                    && g.name != "output.weight"
            );
            assert!(g.name.capacity() <= 64);
            // Worst supported original16-bit matrix: all actual selected
            // geometries fit the same complete tile in prompt and decode.
            let tile = Tile::new(g.inputs, g.rows, g.inputs * 2, 30, 256).unwrap();
            assert_eq!(tile.payload, g.inputs * g.rows * 2);
            assert!(tile.source <= 64 * 1024 * 1024);
        }
        assert!(
            2 * COUNT * (std::mem::size_of::<Tile>() + std::mem::size_of::<String>() + 64 + 128)
                < HOST_BYTES
        );
    }
    #[test]
    fn original_shard_header_receipt_covers_every_dense_matrix_without_omission_or_conversion() {
        let receipt: serde_json::Value = serde_json::from_str(include_str!(
            "../../text/receipts/daslab-iq3-artifact-20261009.json"
        ))
        .unwrap();
        let inventory = receipt["tensor_inventory"].as_array().unwrap();
        let mut actual = inventory
            .iter()
            .map(|t| (t["name"].as_str().unwrap(), t))
            .collect::<BTreeMap<_, _>>();
        assert_eq!(actual.len(), crate::daslab_profile::TENSOR_COUNT);
        assert_eq!(actual.len(), inventory.len());
        let mut bytes = 0u64;
        for g in geometries() {
            let t = actual.remove(g.name.as_str()).unwrap();
            let dims = t["dimensions"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_u64().unwrap() as usize)
                .collect::<Vec<_>>();
            let expected = if g.name.ends_with("ffn_gate_inp_shexp.weight") {
                vec![g.inputs]
            } else {
                vec![g.inputs, g.rows]
            };
            assert_eq!(dims, expected);
            let dtype = t["type_id"].as_u64().unwrap() as u32;
            assert_eq!(Some(dtype), crate::daslab_profile::expected_dtype(&g.name));
            for alignment in [256, 4096] {
                let tile = Tile::new(
                    g.inputs,
                    g.rows,
                    t["row_bytes"].as_u64().unwrap() as usize,
                    dtype,
                    alignment,
                )
                .unwrap();
                assert_eq!(tile.payload as u64, t["tensor_bytes"].as_u64().unwrap());
                assert!(
                    tile.owned >= tile.payload && tile.owned - tile.payload < alignment as usize
                );
                if alignment == 256 {
                    bytes += tile.owned as u64;
                }
            }
        }
        assert_eq!(bytes, 3_644_840_960);
        assert_eq!(actual.len(), 440);
        assert_eq!(
            actual
                .keys()
                .filter(|n| n.ends_with("_exps.weight"))
                .count(),
            144
        );
        for (name, t) in actual {
            if matches!(
                name,
                "token_embd.weight" | "output.weight" | "per_layer_token_embd.weight"
            ) || name.ends_with("_exps.weight")
            {
                continue;
            }
            // Every remaining original tensor is a separately prewarmed norm,
            // control or convolution vector, never an omitted projection.
            let dims = t["dimensions"].as_array().unwrap();
            assert!(
                dims.len() == 1
                    || name.ends_with("ssm_conv1d.weight")
                    || name == "blk.1.ple_conv1d.weight"
            );
            assert!(dims.iter().map(|v| v.as_u64().unwrap()).product::<u64>() <= 65536);
        }
    }
    #[test]
    fn pinned_original_formats_fit_identical_complete_prompt_and_decode_tiles() {
        let mut bytes = 0u64;
        let mut payload = 0u64;
        let mut peak = 0u64;
        for g in geometries() {
            let dtype = crate::daslab_profile::expected_dtype(&g.name).unwrap();
            let (block, packed) = quant_matvec::block_layout(dtype).unwrap();
            assert!(g.inputs.is_multiple_of(block));
            let row = g.inputs / block * packed;
            let tile = Tile::new(g.inputs, g.rows, row, dtype, 256).unwrap();
            bytes += tile.owned as u64;
            payload += tile.payload as u64;
            peak = peak.max(tile.source);
            assert_eq!(tile.dtype, dtype);
            assert!(Tile::new(g.inputs, g.rows, row + 1, dtype, 256).is_err());
        }
        let total = bytes + crate::strata_shared_weights::public_bytes();
        assert!(total < 5 * 1024 * 1024 * 1024);
        assert!(payload <= bytes && bytes - payload < (COUNT * 256) as u64);
        assert!(peak <= crate::strata_shared_weights::SOURCE_BYTES);
        println!(
            "public_native_static_inventory matrices={} payload_bytes={} owned_bytes={} shared_head_bytes={} total_dense_bytes={} source_peak_bytes={}",
            COUNT,
            payload,
            bytes,
            crate::strata_shared_weights::public_bytes(),
            total,
            peak
        );
    }
    #[test]
    fn physical_padding_and_one_source_page_are_priced_without_a_five_gib_allowance() {
        for alignment in [1, 16, 256, 4096] {
            let tile = Tile::new(2560, 1, 5120, 30, alignment).unwrap();
            let mut ledger =
                main_admission::MainAdmission::new(tile.owned as u64 + tile.source, 0).unwrap();
            assert_eq!(
                ledger
                    .precharge_persistent(&[tile.owned as u64, tile.source], alignment)
                    .unwrap(),
                tile.owned as u64 + tile.source
            );
            assert_eq!(tile.payload, 5120);
            assert_eq!(tile.owned, if alignment == 4096 { 8192 } else { 5120 });
            let old_fixed = 5 * 1024 * 1024 * 1024u64;
            let fixed = [
                (tile.owned as u64) + crate::strata_shared_weights::public_bytes(),
                tile.source,
            ];
            assert!(
                main_admission::shared_expert_capacity(old_fixed, &fixed, u64::MAX).unwrap()
                    > old_fixed / 2
            );
        }
        for (i, o, rb, a) in [
            (0, 1, 1, 256),
            (2560, 0, 5120, 256),
            (2560, 16384, 5120, 256),
            (2560, 1, 0, 256),
            (2560, 1, 5120, 0),
            (2560, 1, 5120, 3),
            (2560, 1, 5120, 8192),
            (usize::MAX, 1, 5120, 256),
        ] {
            assert!(Tile::new(i, o, rb, 30, a).is_err());
        }
    }
}
