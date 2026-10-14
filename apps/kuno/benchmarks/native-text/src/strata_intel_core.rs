//! Connected Intel core port. Shared original IQ tables are public weights;
//! they are loaded once, read directly, and never contain request content.
//! MIT Strata fb58e0dbc8399662c0e47c76578c6e878b14f6cf, iq_kernels.dp.cpp.
use cubecl::{prelude::ComputeClient, wgpu::WgpuRuntime};
use cubecl_runtime::server::{Handle, KernelArguments};
use kuno_text::{Error, Result};
use std::cell::RefCell;

pub(crate) const PUBLIC_TABLE_BYTES: u64 = 32768;
const TABLE_WORDS: usize = 8192;
pub(crate) const ADDRESS_TABLE_BYTES: u64 = 48 * 32768;
struct AddressTable {
    roots: Vec<(u64, u64)>,
    handle: Handle,
}
struct AddressState {
    device: wgpu::Device,
    tables: Vec<AddressTable>,
}
thread_local! {static TABLE:RefCell<Option<Handle>>=const {RefCell::new(None)};static ADDRESSES:RefCell<Option<AddressState>>=const{RefCell::new(None)};}
pub(crate) const fn enabled() -> bool {
    cfg!(feature = "strata-intel-core")
}
fn reject() -> Error {
    Error::InvalidFormat("Strata Intel public table scope")
}
pub(crate) struct Tables {
    client: ComputeClient<WgpuRuntime>,
}
impl Tables {
    pub(crate) fn new(
        client: &ComputeClient<WgpuRuntime>,
        device: Option<&wgpu::Device>,
    ) -> Result<Option<Self>> {
        if !enabled() {
            return Ok(None);
        }
        if TABLE.with(|t| t.borrow().is_some()) {
            return Err(reject());
        }
        let mut words = Vec::with_capacity(TABLE_WORDS);
        for dtype in [
            crate::quant_matvec::IQ3_XXS,
            crate::quant_matvec::IQ3_S,
            crate::quant_matvec::IQ2_S,
        ] {
            words.extend(crate::quant_matvec::iq_moe_integer::original_grid_words(
                dtype,
            ));
        }
        if words.len() != 2816 {
            return Err(reject());
        }
        words.resize(TABLE_WORDS, 0);
        let bytes = words.into_iter().flat_map(u32::to_le_bytes).collect();
        let handle = client.create_public_owned(bytes);
        let r = client
            .retained_resource_metadata(vec![handle.clone()])
            .map_err(crate::resident_util::fault)?;
        if handle.size_in_used() != PUBLIC_TABLE_BYTES
            || r.len() != 1
            || r[0].resource().offset != 0
            || r[0].resource().buffer.size() != PUBLIC_TABLE_BYTES
        {
            return Err(reject());
        }
        let device = device.ok_or_else(reject)?;
        ADDRESSES.with(|t| {
            *t.borrow_mut() = Some(AddressState {
                device: device.clone(),
                tables: Vec::new(),
            })
        });
        TABLE.with(|t| *t.borrow_mut() = Some(handle));
        Ok(Some(Self {
            client: client.clone(),
        }))
    }
}
impl Drop for Tables {
    fn drop(&mut self) {
        // Queue completion precedes dropping the last shared public model handle.
        let _ = self.client.flush();
        TABLE.with(|t| *t.borrow_mut() = None);
        ADDRESSES.with(|t| *t.borrow_mut() = None);
    }
}
pub(crate) fn bind(
    args: KernelArguments,
    stream: cubecl_common::stream_id::StreamId,
) -> Result<KernelArguments> {
    if !enabled() {
        return Ok(args);
    }
    TABLE.with(|t| {
        let t = t.borrow();
        let h = t.as_ref().ok_or_else(reject)?;
        if h.stream != stream || h.size_in_used() != PUBLIC_TABLE_BYTES {
            return Err(reject());
        }
        Ok(args.with_buffer(h.clone().binding()))
    })
}
pub(crate) fn shared_table(mut source: String, binding: usize, dtype: u32) -> String {
    if !enabled() {
        return source;
    }
    let n = crate::quant_matvec::iq_moe_integer::original_grid_words(dtype).len();
    if n > 0 {
        let decl = format!("var<workgroup> iq3_grid:array<u32,{n}>;");
        assert!(source.contains(&decl), "original shared grid declaration");
        source = source.replace(&decl, "");
        let start = source
            .find("switch local {")
            .expect("original public grid initializer");
        let end = crate::strata_port::function_end(&source, start);
        source.replace_range(start..end, "");
        let offset = match dtype {
            crate::quant_matvec::IQ3_XXS => 0,
            crate::quant_matvec::IQ3_S => 256,
            crate::quant_matvec::IQ2_S => 768,
            _ => unreachable!(),
        };
        source = source.replace("iq3_grid[", &format!("strata_grid[{offset}u+"));
    }
    // Declared for every GU format: actual owner shape is a dynamic guard, so
    // even formats without an IQ grid retain the same full descriptor layout.
    source.push_str(&format!(
        "\n@group(0) @binding({binding}) var<storage,read> strata_grid:array<u32>;\n"
    ));
    let guard = "    if arrayLength(&strata_grid)!=8192u {return false;}\n";
    let start = source.find("fn valid").expect("original owning guards");
    let brace = start + source[start..].find('{').unwrap() + 1;
    source.insert_str(brace, guard);
    source
}

pub(crate) fn bind_addresses(
    client: &ComputeClient<WgpuRuntime>,
    pages: &[Handle],
    args: KernelArguments,
) -> Result<KernelArguments> {
    if !enabled() {
        return Ok(args);
    }
    if pages.len() != 15 {
        return Err(reject());
    }
    let resources = client
        .retained_resource_metadata(pages.to_vec())
        .map_err(crate::resident_util::fault)?;
    ADDRESSES.with(|state| {
        let mut state = state.borrow_mut();
        let state = state.as_mut().ok_or_else(reject)?;
        let dev = unsafe { state.device.as_hal::<wgpu::hal::api::Vulkan>() }.ok_or_else(reject)?;
        let mut words = Vec::with_capacity(8192);
        let mut roots = Vec::with_capacity(15);
        for r in &resources {
            let r = r.resource();
            if r.offset != 0 || r.size != r.buffer.size() {
                return Err(reject());
            }
            let h = unsafe { r.buffer.as_hal::<wgpu::hal::api::Vulkan>() }.ok_or_else(reject)?;
            let address = unsafe {
                dev.raw_device().get_buffer_device_address(
                    &ash::vk::BufferDeviceAddressInfo::default().buffer(h.raw_buffer_handle()),
                )
            };
            if address == 0 || address % 16 != 0 {
                return Err(reject());
            }
            words.extend([
                address as u32,
                (address >> 32) as u32,
                (r.size / 4) as u32,
                0,
            ]);
        }
        if let Some(table) = state.tables.iter().find(|t| t.roots == roots) {
            return Ok(args.with_buffer(table.handle.clone().binding()));
        }
        if state.tables.len() >= 48 {
            return Err(reject());
        }
        words.resize(8192, 0);
        let handle =
            client.create_public_owned(words.into_iter().flat_map(u32::to_le_bytes).collect());
        state.tables.push(AddressTable {
            roots,
            handle: handle.clone(),
        });
        Ok(args.with_buffer(handle.binding()))
    })
}
