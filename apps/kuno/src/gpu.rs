//! Linux PCI discovery and read-only driver telemetry. No GPU contexts or models.
use crate::resources::{Gpu, Memory};
use std::{
    ffi::{CStr, CString, c_char, c_void},
    fs,
    path::Path,
};
type Result<T> = std::result::Result<T, &'static str>;

pub(crate) fn collect() -> Result<Vec<Gpu>> {
    let devices = fs::read_dir("/sys/bus/pci/devices").map_err(|_| "pci_inventory_unavailable")?;
    let mut gpus = Vec::new();
    for entry in devices {
        let path = entry.map_err(|_| "pci_inventory_unavailable")?.path();
        let Some(class) = hex(&path.join("class")) else {
            continue;
        };
        if class >> 16 != 3 {
            continue;
        }
        let pci = path
            .file_name()
            .and_then(|p| p.to_str())
            .ok_or("pci_identity_invalid")?;
        if !valid_pci(pci) {
            continue;
        }
        let vendor_id = hex(&path.join("vendor"))
            .and_then(|v| u16::try_from(v).ok())
            .ok_or("pci_identity_invalid")?;
        let device_id = hex(&path.join("device"))
            .and_then(|v| u16::try_from(v).ok())
            .ok_or("pci_identity_invalid")?;
        let driver = fs::read_link(path.join("driver"))
            .ok()
            .and_then(|p| p.file_name().map(|n| n.to_string_lossy().into_owned()));
        let render_node = fs::read_dir(path.join("drm")).ok().and_then(|entries| {
            entries.filter_map(|e| e.ok()).find_map(|e| {
                let name = e.file_name().to_string_lossy().into_owned();
                name.strip_prefix("renderD")
                    .filter(|s| !s.is_empty() && s.bytes().all(|b| b.is_ascii_digit()))
                    .map(|_| format!("/dev/dri/{name}"))
            })
        });
        let mut gpu = Gpu {
            id: format!("pci:{pci}"),
            pci_address: pci.into(),
            vendor_id,
            device_id,
            driver,
            name: None,
            uuid: None,
            render_node,
            memory: None,
            errors: vec![],
        };
        let result = match gpu.driver.as_deref() {
            Some("nvidia") => nvidia(&mut gpu),
            Some("xe") => {
                let result = gpu
                    .render_node
                    .as_deref()
                    .ok_or("xe_render_node_missing")
                    .and_then(xe_memory);
                match result {
                    Ok(memory) => {
                        if memory.free_bytes.is_none() {
                            gpu.errors
                                .push("xe_memory_usage_unavailable_or_restricted".into());
                        }
                        gpu.memory = Some(memory);
                        Ok(())
                    }
                    Err(e) => Err(e),
                }
            }
            Some("amdgpu") => {
                let total = number(&path.join("mem_info_vram_total"));
                let used = number(&path.join("mem_info_vram_used"));
                match (total, used) {
                    (Some(total), Some(used)) if total > 0 && used <= total => {
                        gpu.memory = Some(Memory {
                            total_bytes: total,
                            used_bytes: Some(used),
                            free_bytes: Some(total - used),
                            reserved_bytes: None,
                            source: "amdgpu_sysfs".into(),
                            estimated: true,
                        });
                        Ok(())
                    }
                    _ => Err("amdgpu_memory_unavailable"),
                }
            }
            _ => Err("gpu_driver_telemetry_unsupported"),
        };
        if let Err(error) = result {
            gpu.errors.push(error.into());
        }
        gpus.push(gpu);
    }
    gpus.sort_by(|a, b| a.pci_address.cmp(&b.pci_address));
    Ok(gpus)
}

fn hex(path: &Path) -> Option<u32> {
    u32::from_str_radix(
        fs::read_to_string(path).ok()?.trim().strip_prefix("0x")?,
        16,
    )
    .ok()
}
fn number(path: &Path) -> Option<u64> {
    fs::read_to_string(path).ok()?.trim().parse().ok()
}
fn valid_pci(pci: &str) -> bool {
    pci.len() == 12
        && pci.bytes().enumerate().all(|(i, b)| match i {
            4 | 7 => b == b':',
            10 => b == b'.',
            _ => b.is_ascii_hexdigit(),
        })
}

#[repr(C)]
#[derive(Default)]
struct XeQuery {
    extensions: u64,
    query: u32,
    size: u32,
    data: u64,
    reserved: [u64; 2],
}
const _: () = assert!(std::mem::size_of::<XeQuery>() == 40);

fn xe_memory(node: &str) -> Result<Memory> {
    use std::os::{
        fd::AsRawFd,
        unix::fs::{FileTypeExt, OpenOptionsExt},
    };
    let file = fs::OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_NOFOLLOW | libc::O_CLOEXEC)
        .open(node)
        .map_err(|_| "xe_render_node_not_accessible")?;
    if !file
        .metadata()
        .map_err(|_| "xe_render_node_not_accessible")?
        .file_type()
        .is_char_device()
    {
        return Err("xe_render_node_invalid");
    }
    // Same MEM_REGIONS query proven by itop/src/collect.rs::xe_memory. Stable
    // Linux Xe UAPI: _IOWR('d', 0x40, drm_xe_device_query), query 1.
    const IOCTL: libc::c_ulong = 0xc0286440;
    let mut query = XeQuery {
        query: 1,
        ..Default::default()
    };
    // SAFETY: the repr(C) query has the 40-byte kernel ABI and writable storage.
    if unsafe { libc::ioctl(file.as_raw_fd(), IOCTL, &mut query) } < 0 {
        return Err("xe_memory_query_failed");
    }
    if !(8..=65536).contains(&query.size) {
        return Err("xe_memory_reply_invalid");
    }
    let capacity = query.size as usize;
    let mut data = vec![0u8; capacity];
    query.data = data.as_mut_ptr() as u64;
    // SAFETY: data owns a writable allocation of the requested size for the
    // duration of the ioctl; all reserved fields are zero and no GPU work runs.
    if unsafe { libc::ioctl(file.as_raw_fd(), IOCTL, &mut query) } < 0 {
        return Err("xe_memory_query_failed");
    }
    if query.size as usize > capacity {
        return Err("xe_memory_reply_invalid");
    }
    parse_xe_memory(&data[..query.size as usize])
}

fn parse_xe_memory(data: &[u8]) -> Result<Memory> {
    if data.len() < 8 {
        return Err("xe_memory_reply_invalid");
    }
    let count = u32::from_ne_bytes(data[..4].try_into().unwrap()) as usize;
    if count > 64 || data.len() < 8 + count * 88 {
        return Err("xe_memory_reply_invalid");
    }
    let (mut total, mut used) = (0u64, 0u64);
    let mut instances = std::collections::BTreeSet::new();
    for region in data[8..8 + count * 88].chunks_exact(88) {
        if u16::from_ne_bytes(region[..2].try_into().unwrap()) != 1 {
            continue;
        }
        if !instances.insert(u16::from_ne_bytes(region[2..4].try_into().unwrap())) {
            return Err("xe_memory_reply_invalid");
        }
        let size = u64::from_ne_bytes(region[8..16].try_into().unwrap());
        let allocated = u64::from_ne_bytes(region[16..24].try_into().unwrap());
        if allocated > size {
            return Err("xe_memory_reply_invalid");
        }
        total = total.checked_add(size).ok_or("xe_memory_reply_invalid")?;
        used = used
            .checked_add(allocated)
            .ok_or("xe_memory_reply_invalid")?;
    }
    if total == 0 {
        return Err("no_dedicated_vram_reported");
    }
    // Like itop, accept nonzero accounting but never mistake privilege-masked
    // zero for a fully empty card. Different kernels restrict this differently.
    Ok(Memory {
        total_bytes: total,
        used_bytes: (used > 0).then_some(used),
        free_bytes: (used > 0).then_some(total - used),
        reserved_bytes: None,
        source: "xe_drm_memory_regions".into(),
        estimated: true,
    })
}

#[repr(C)]
#[derive(Default)]
struct NvmlMemory {
    total: u64,
    free: u64,
    used: u64,
}
type Device = *mut c_void;
type Init = unsafe extern "C" fn() -> u32;
type Handle = unsafe extern "C" fn(*const c_char, *mut Device) -> u32;
type ReadMemory = unsafe extern "C" fn(Device, *mut NvmlMemory) -> u32;
#[repr(C)]
#[derive(Default)]
struct NvmlMemoryV2 {
    version: u32,
    total: u64,
    reserved: u64,
    free: u64,
    used: u64,
}
type ReadMemoryV2 = unsafe extern "C" fn(Device, *mut NvmlMemoryV2) -> u32;
const _: () = assert!(std::mem::size_of::<NvmlMemoryV2>() == 40);

type ReadString = unsafe extern "C" fn(Device, *mut c_char, u32) -> u32;

struct Nvml {
    library: *mut c_void,
    shutdown: Init,
}
impl Nvml {
    fn open() -> Result<Self> {
        // SAFETY: fixed system driver library, RTLD_LOCAL, valid NUL-terminated
        // symbol names and documented NVML C signatures. No GPU context created.
        unsafe {
            let library = libc::dlopen(
                c"libnvidia-ml.so.1".as_ptr(),
                libc::RTLD_NOW | libc::RTLD_LOCAL,
            );
            if library.is_null() {
                return Err("nvml_library_unavailable");
            }
            let initialized = (|| {
                let init: Init = symbol(library, c"nvmlInit_v2")?;
                let shutdown: Init = symbol(library, c"nvmlShutdown")?;
                if init() != 0 {
                    return Err("nvml_initialization_failed");
                }
                Ok(Self { library, shutdown })
            })();
            if initialized.is_err() {
                libc::dlclose(library);
            }
            initialized
        }
    }
}
impl Drop for Nvml {
    fn drop(&mut self) {
        // SAFETY: successfully initialized NVML; library outlives every symbol.
        unsafe {
            (self.shutdown)();
            libc::dlclose(self.library);
        }
    }
}
unsafe fn symbol<T: Copy>(library: *mut c_void, name: &CStr) -> Result<T> {
    assert_eq!(std::mem::size_of::<T>(), std::mem::size_of::<*mut c_void>());
    // SAFETY: callers supply a valid loaded handle and the documented function
    // pointer type matching the fixed C symbol. It is used before dlclose.
    let address = unsafe { libc::dlsym(library, name.as_ptr()) };
    if address.is_null() {
        return Err("nvml_function_unavailable");
    }
    Ok(unsafe { std::mem::transmute_copy(&address) })
}

fn nvidia(gpu: &mut Gpu) -> Result<()> {
    let nvml = Nvml::open()?;
    let pci = CString::new(gpu.pci_address.as_str()).map_err(|_| "pci_identity_invalid")?;
    // SAFETY: all functions use the documented NVML ABI. The handle is obtained
    // by physical PCI address, not an enumeration index that can change order.
    unsafe {
        let get: Handle = symbol(nvml.library, c"nvmlDeviceGetHandleByPciBusId_v2")?;
        let mut device = std::ptr::null_mut();
        if get(pci.as_ptr(), &mut device) != 0 || device.is_null() {
            return Err("nvml_device_unavailable");
        }
        let mut measured = None;
        if let Ok(read) = symbol::<ReadMemoryV2>(nvml.library, c"nvmlDeviceGetMemoryInfo_v2") {
            let mut memory = NvmlMemoryV2 {
                version: (2 << 24) | 40,
                ..Default::default()
            };
            if read(device, &mut memory) == 0 {
                let accounted = memory
                    .used
                    .checked_add(memory.free)
                    .and_then(|n| n.checked_add(memory.reserved))
                    .ok_or("nvml_memory_invalid")?;
                if memory.total == 0 || accounted > memory.total {
                    return Err("nvml_memory_invalid");
                }
                measured = Some(Memory {
                    total_bytes: memory.total,
                    used_bytes: Some(memory.used),
                    free_bytes: Some(memory.free),
                    reserved_bytes: Some(memory.reserved),
                    source: "nvml_v2".into(),
                    estimated: true,
                });
            }
        }
        if measured.is_none() {
            let read: ReadMemory = symbol(nvml.library, c"nvmlDeviceGetMemoryInfo")?;
            let mut memory = NvmlMemory::default();
            if read(device, &mut memory) != 0 {
                return Err("nvml_memory_unavailable");
            }
            let accounted = memory
                .used
                .checked_add(memory.free)
                .ok_or("nvml_memory_invalid")?;
            if memory.total == 0 || accounted > memory.total {
                return Err("nvml_memory_invalid");
            }
            // v1 includes reserved memory in used; it does not expose a separate
            // reserved counter. Never invent a zero reserved value from it.
            measured = Some(Memory {
                total_bytes: memory.total,
                used_bytes: Some(memory.used),
                free_bytes: Some(memory.free),
                reserved_bytes: None,
                source: "nvml_v1_includes_reserved".into(),
                estimated: true,
            });
        }
        gpu.memory = measured;
        for (symbol_name, target) in [
            (c"nvmlDeviceGetName", &mut gpu.name),
            (c"nvmlDeviceGetUUID", &mut gpu.uuid),
        ] {
            let read: ReadString = symbol(nvml.library, symbol_name)?;
            let mut buffer = [0u8; 128];
            if read(device, buffer.as_mut_ptr().cast(), buffer.len() as u32) == 0 {
                *target = CStr::from_bytes_until_nul(&buffer)
                    .ok()
                    .and_then(|s| s.to_str().ok())
                    .filter(|s| !s.chars().any(char::is_control))
                    .map(str::to_owned);
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    fn reply(regions: &[(u16, u16, u64, u64)]) -> Vec<u8> {
        let mut bytes = vec![0; 8 + regions.len() * 88];
        bytes[..4].copy_from_slice(&(regions.len() as u32).to_ne_bytes());
        for (i, (class, instance, total, used)) in regions.iter().enumerate() {
            let r = &mut bytes[8 + i * 88..8 + (i + 1) * 88];
            r[..2].copy_from_slice(&class.to_ne_bytes());
            r[2..4].copy_from_slice(&instance.to_ne_bytes());
            r[8..16].copy_from_slice(&total.to_ne_bytes());
            r[16..24].copy_from_slice(&used.to_ne_bytes());
        }
        bytes
    }
    #[test]
    fn xe_counts_only_vram_and_never_treats_masked_zero_as_free() {
        let memory = parse_xe_memory(&reply(&[
            (0, 0, 99999, 9999),
            (1, 1, 1000, 100),
            (1, 2, 2000, 500),
        ]))
        .unwrap();
        assert_eq!(memory.total_bytes, 3000);
        assert_eq!(memory.used_bytes, Some(600));
        assert_eq!(memory.free_bytes, Some(2400));
        let restricted = parse_xe_memory(&reply(&[(1, 1, 1000, 0)])).unwrap();
        assert_eq!(restricted.total_bytes, 1000);
        assert!(restricted.used_bytes.is_none());
        assert!(restricted.free_bytes.is_none());
    }
    #[test]
    fn invalid_xe_layouts_fail_without_inventing_capacity() {
        assert!(parse_xe_memory(&reply(&[(1, 1, 100, 101)])).is_err());
        assert!(parse_xe_memory(&reply(&[(1, 1, 100, 1), (1, 1, 100, 1)])).is_err());
        assert!(parse_xe_memory(&reply(&[(1, 1, u64::MAX, 1), (1, 2, 100, 1)])).is_err());
        assert!(parse_xe_memory(&[0; 7]).is_err());
        let mut short = reply(&[(1, 1, 100, 1)]);
        short.pop();
        assert!(parse_xe_memory(&short).is_err());
    }
}
