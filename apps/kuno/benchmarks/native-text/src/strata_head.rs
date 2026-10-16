//! Whole original Q6 heads across authentic bounded GPU pages. Vulkan's32KiB
//! immutable address metadata is owned/charged beside each head, never a copy of
//! weights. Actual native recordings retain every original source dependency.
use cubecl::{
    ir::{ElemType, StorageType, UIntKind},
    prelude::{ComputeClient, CubeCount, CubeDim},
    wgpu::{AutoCompiler, WgpuRuntime},
};
use cubecl_runtime::{
    compiler::{CompilationError, CubeTask},
    id::KernelId,
    kernel::{CompiledKernel, KernelMetadata},
    server::{ExecutionMode, Handle, KernelArguments, MetadataBindingInfo},
};
use kuno_text::{Error, Result};
pub(crate) const TABLE_BYTES: usize = 32768;
pub(crate) const COLUMNS: usize = 2560;
pub(crate) const ROW_BYTES: usize = 2100;
pub(crate) const TILE_ROWS: usize = 16384;
pub(crate) const ADDRESS_KIND: u32 = u32::MAX - 3;
fn fail() -> Error {
    Error::InvalidFormat("whole original Q6 head")
}
pub(crate) struct Page {
    pub handle: Handle,
    pub first: usize,
    pub rows: usize,
    pub logical: usize,
    pub owned: usize,
}
fn encode(
    rows: usize,
    pages: &[(usize, usize, usize, usize)],
    addresses: &[[u64; 2]],
) -> Result<Vec<u8>> {
    if ![
        crate::strata_shared_weights::ROWS,
        crate::resident_mtp::cjk::ROWS,
    ]
    .contains(&rows)
        || pages.len() != rows.div_ceil(TILE_ROWS)
        || addresses.len() != pages.len()
    {
        return Err(fail());
    }
    let mut output = vec![0u8; TABLE_BYTES];
    for (i, (&(first, count, logical, owned), &[address, bytes])) in
        pages.iter().zip(addresses).enumerate()
    {
        if first != i * TILE_ROWS
            || count != TILE_ROWS.min(rows - first)
            || logical != count * ROW_BYTES
            || owned != logical.next_multiple_of(4096)
            || !owned.is_multiple_of(16)
            || bytes != owned as u64
            || address == 0
            || !address.is_multiple_of(16)
            || address.checked_add(bytes).is_none()
        {
            return Err(fail());
        }
        for (at, value) in [
            address as u32,
            (address >> 32) as u32,
            (logical / 4) as u32,
            count as u32,
        ]
        .into_iter()
        .enumerate()
        {
            output[i * 16 + at * 4..i * 16 + at * 4 + 4].copy_from_slice(&value.to_le_bytes());
        }
    }
    Ok(output)
}
pub(crate) fn metadata(
    client: &ComputeClient<WgpuRuntime>,
    rows: usize,
    pages: &[Page],
) -> Result<Vec<u8>> {
    let first = pages.first().ok_or_else(fail)?;
    for (i, p) in pages.iter().enumerate() {
        if p.handle.stream != first.handle.stream
            || p.handle.offset_start.unwrap_or(0) != 0
            || p.handle.offset_end.unwrap_or(0) != 0
            || p.handle.size_in_used() != p.owned as u64
            || pages[..i].iter().any(|q| {
                q.handle.clone().binding().memory.allocation_id()
                    == p.handle.clone().binding().memory.allocation_id()
            })
        {
            return Err(fail());
        }
    }
    let addresses = client
        .native_strata_source_addresses(pages.iter().map(|p| p.handle.clone()).collect())
        .map_err(crate::resident_util::fault)?;
    encode(
        rows,
        &pages
            .iter()
            .map(|p| (p.first, p.rows, p.logical, p.owned))
            .collect::<Vec<_>>(),
        &addresses,
    )
}
pub(crate) struct Head {
    rows: usize,
    client: ComputeClient<WgpuRuntime>,
    pages: Vec<Page>,
    table: Handle,
}
impl Head {
    pub(crate) fn new(
        client: &ComputeClient<WgpuRuntime>,
        rows: usize,
        pages: Vec<Page>,
        table: Handle,
    ) -> Result<Self> {
        if ![
            crate::strata_shared_weights::ROWS,
            crate::resident_mtp::cjk::ROWS,
        ]
        .contains(&rows)
            || table.size_in_used() != TABLE_BYTES as u64
            || table.offset_start.unwrap_or(0) != 0
            || table.offset_end.unwrap_or(0) != 0
            || pages.len() != rows.div_ceil(TILE_ROWS)
            || pages.iter().any(|p| {
                p.handle.stream != table.stream
                    || p.handle.clone().binding().memory.allocation_id()
                        == table.clone().binding().memory.allocation_id()
            })
        {
            return Err(fail());
        }
        Ok(Self {
            rows,
            client: client.clone(),
            pages,
            table,
        })
    }
    pub(crate) fn table(&self) -> &Handle {
        &self.table
    }
    #[allow(clippy::too_many_arguments)] // Original private/output geometry plus same owning model/client.
    pub(crate) fn project(
        &self,
        client: &ComputeClient<WgpuRuntime>,
        planes: Handle,
        output: Handle,
        fault: Handle,
        first: usize,
        stride: usize,
        count: usize,
    ) -> Result<()> {
        if !std::ptr::eq(client.info(), self.client.info())
            || count == 0
            || count > 8
            || first > u32::MAX as usize
            || stride > u32::MAX as usize
            || planes.size_in_used() < (count * (COLUMNS / 2 + COLUMNS / 16) * 4) as u64
            || fault.size_in_used() != 4
            || stride < self.rows
            || first
                .checked_add((count - 1) * stride)
                .and_then(|n| n.checked_add(self.rows))
                .is_none_or(|n| n as u64 * 4 > output.size_in_used())
            || [&planes, &output, &fault]
                .iter()
                .any(|h| h.stream != self.table.stream)
        {
            return Err(fail());
        }
        let mut args = KernelArguments::new()
            .with_buffer(self.pages[0].handle.clone().binding())
            .with_buffer(planes.binding())
            .with_buffer(output.binding())
            .with_buffer(fault.binding())
            .with_buffer(self.table.clone().binding());
        args = args
            .with_read_dependencies(
                self.pages
                    .iter()
                    .map(|p| p.handle.clone().binding())
                    .collect(),
            )
            .with_info(MetadataBindingInfo::custom(vec![
                first as u64 | ((stride as u64) << 32),
                (COLUMNS / 2 + COLUMNS / 16) as u64 | ((count as u64) << 32),
            ]));
        unsafe {
            client.launch_unchecked(
                Box::new(Kernel {
                    rows: self.rows,
                    members: count,
                }),
                CubeCount::Static(self.rows.div_ceil(4) as u32, 1, 1),
                args,
            );
        }
        crate::strata_dense::record_projection_call();
        Ok(())
    }
}
#[derive(Clone, Copy, Hash, PartialEq, Eq, Debug)]
struct Kernel {
    rows: usize,
    members: usize,
}
impl KernelMetadata for Kernel {
    fn name(&self) -> &'static str {
        "StrataWholeOriginalQ6Head"
    }
    fn id(&self) -> KernelId {
        KernelId::new::<Self>().info(*self)
    }
    fn address_type(&self) -> StorageType {
        StorageType::Scalar(ElemType::UInt(UIntKind::U32))
    }
}
impl CubeTask<AutoCompiler> for Kernel {
    fn compile(
        &self,
        _: &mut AutoCompiler,
        _: &cubecl::WgpuCompilationOptions,
        _: ExecutionMode,
        _: StorageType,
    ) -> core::result::Result<CompiledKernel<AutoCompiler>, CompilationError> {
        Ok(CompiledKernel {
            entrypoint_name: "main".into(),
            debug_info: None,
            debug_name: None,
            source: crate::strata_dense::whole_head_source(self.rows, self.members),
            repr: None,
            cube_dim: CubeDim::new_1d(128),
        })
    }
}
pub(crate) fn sources() -> Vec<CompiledKernel<AutoCompiler>> {
    [
        crate::strata_shared_weights::ROWS,
        crate::resident_mtp::cjk::ROWS,
    ]
    .into_iter()
    .flat_map(|rows| (1..=8).map(move |members| Kernel { rows, members }))
    .map(|k| {
        k.compile(
            &mut AutoCompiler::Wgsl(Default::default()),
            &Default::default(),
            ExecutionMode::Unchecked,
            StorageType::Scalar(ElemType::UInt(UIntKind::U32)),
        )
        .unwrap()
    })
    .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    type Geometry = (usize, usize, usize, usize);
    fn fixture(rows: usize) -> (Vec<Geometry>, Vec<[u64; 2]>) {
        let pages = (0..rows)
            .step_by(TILE_ROWS)
            .map(|first| {
                let n = TILE_ROWS.min(rows - first);
                (
                    first,
                    n,
                    n * ROW_BYTES,
                    (n * ROW_BYTES).next_multiple_of(4096),
                )
            })
            .collect::<Vec<_>>();
        let addresses = pages
            .iter()
            .enumerate()
            .map(|(i, p)| [0x12_0000_0000 + i as u64 * 0x1000_0000, p.3 as u64])
            .collect();
        (pages, addresses)
    }
    #[test]
    fn whole_original_tables_cover_every_target_and_compact_row_with_exact_owner_padding() {
        for rows in [
            crate::strata_shared_weights::ROWS,
            crate::resident_mtp::cjk::ROWS,
        ] {
            let (pages, addresses) = fixture(rows);
            let bytes = encode(rows, &pages, &addresses).unwrap();
            assert_eq!(bytes.len(), TABLE_BYTES);
            assert!(bytes[pages.len() * 16..].iter().all(|b| *b == 0));
            for row in 0..rows {
                let index = row / TILE_ROWS;
                let words = bytes[index * 16..index * 16 + 16]
                    .chunks_exact(4)
                    .map(|b| u32::from_le_bytes(b.try_into().unwrap()))
                    .collect::<Vec<_>>();
                let pointer = u64::from(words[0]) | u64::from(words[1]) << 32;
                assert_eq!(pointer, addresses[index][0]);
                assert_eq!(words[2] as usize * 4, pages[index].2);
                assert_eq!(words[3] as usize, pages[index].1);
                let local_row = row % TILE_ROWS;
                assert!(local_row < words[3] as usize);
                assert!((local_row + 1) * ROW_BYTES <= words[2] as usize * 4);
                for block in 0..10 {
                    let base = local_row * ROW_BYTES + block * 210;
                    for lane in 0..8 {
                        let qh = 2 * (lane / 4) + lane % 2;
                        for at in [base + 16 * lane, base + 128 + 16 * qh] {
                            // Original A2 may read the next containing vector.
                            let vector_end = at / 16 * 16 + if at % 16 == 0 { 16 } else { 32 };
                            assert!(vector_end <= pages[index].3);
                        }
                    }
                    // half_at reads two original bytes through their containing
                    // vec4; it does not call the two-vector A2 helper.
                    assert!((base + 209) / 16 * 16 + 16 <= pages[index].3);
                }
            }
            assert!(rows.div_ceil(4) <= 65535);
        }
    }
    #[test]
    fn whole_head_encoder_rejects_missing_and_malformed_original_page_geometry() {
        for rows in [
            crate::strata_shared_weights::ROWS,
            crate::resident_mtp::cjk::ROWS,
        ] {
            let (pages, addresses) = fixture(rows);
            assert!(encode(rows - 1, &pages, &addresses).is_err());
            assert!(encode(rows, &pages[1..], &addresses).is_err());
            assert!(encode(rows, &pages, &addresses[1..]).is_err());
            let mut extra = pages.clone();
            extra.push(pages[0]);
            assert!(encode(rows, &extra, &addresses).is_err());
            for i in [0, pages.len() - 1] {
                for field in 0..4 {
                    let mut bad = pages.clone();
                    match field {
                        0 => bad[i].0 += 1,
                        1 => bad[i].1 += 1,
                        2 => bad[i].2 -= 4,
                        _ => bad[i].3 += 4096,
                    }
                    assert!(encode(rows, &bad, &addresses).is_err());
                }
                for replacement in [
                    [0, addresses[i][1]],
                    [addresses[i][0] + 4, addresses[i][1]],
                    [u64::MAX - 15, addresses[i][1]],
                    [addresses[i][0], addresses[i][1] - 16],
                ] {
                    let mut bad = addresses.clone();
                    bad[i] = replacement;
                    assert!(encode(rows, &pages, &bad).is_err());
                }
            }
        }
    }
    #[cfg(feature = "strata-intel-core")]
    #[test]
    fn actual_whole_head_shaders_lower_original_pointer_loads_for_both_heads_and_all_rhs() {
        use wgpu::naga as n;
        let compiled = sources();
        assert_eq!(compiled.len(), 16);
        for shader in compiled {
            let source = &shader.source;
            let module = n::front::wgsl::parse_str(source).unwrap();
            n::valid::Validator::new(
                n::valid::ValidationFlags::all(),
                n::valid::Capabilities::all(),
            )
            .validate(&module)
            .unwrap();
            assert_eq!(module.entry_points[0].workgroup_size, [128, 1, 1]);
            let mut bindings = module
                .global_variables
                .iter()
                .filter_map(|(_, v)| v.binding.as_ref().map(|b| b.binding))
                .collect::<Vec<_>>();
            bindings.sort_unstable();
            assert_eq!(bindings, [0, 1, 2, 3, 4, 5]);
            assert!(source.contains("let b=(row%16384u)*ROW_BYTES+block*210u;"));
            assert!(source.contains("let src=addresses[page]"));
            assert!(!source.contains("arrayLength(&weight)"));
            assert!(!source.contains("arrayLength(&gate0)"));
            assert!(source.contains("if valid_row && lane==0u"));
            let gate = module
                .global_variables
                .iter()
                .find(|(_, v)| v.name.as_deref() == Some("gate0"))
                .unwrap()
                .1;
            assert_eq!(
                gate.space,
                n::AddressSpace::Storage {
                    access: n::StorageAccess::LOAD
                }
            );
            assert!(matches!(
                module.types[gate.ty].inner,
                n::TypeInner::Array { stride: 16, .. }
            ));
            let words = cubecl::wgpu::native_expert_shader_words(source).unwrap();
            assert_eq!(words[0], 0x07230203);
            // Check real native rewrite emits a physical pointer conversion,
            // rather than merely accepting the logical dummy descriptor.
            let mut at = 5;
            let mut pointers = 0;
            while at < words.len() {
                let count = (words[at] >> 16) as usize;
                assert!(count > 0);
                if words[at] & 0xffff == 120 {
                    pointers += 1;
                } // OpConvertUToPtr
                at += count;
            }
            assert!(pointers > 0);
        }
    }
}
