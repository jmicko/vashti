//! Level Zero, Intel's low-level GPU interface, called directly: it is the one interface that
//! accepts machine code we wrote ourselves. The loader library is opened at run time, so a
//! machine without it simply has no fast path. Only what the kernels need is bound: device
//! memory imported from Vulkan, native modules, and launching a kernel and waiting for it.
use std::{
    ffi::{CString, c_char, c_void},
    os::fd::{AsRawFd, OwnedFd},
    ptr::{null, null_mut},
    sync::{Arc, Mutex},
};

type Handle = *mut c_void;
type Code = u32;

#[repr(C)]
struct ContextDesc {
    stype: u32,
    next: *const c_void,
    flags: u32,
}
#[repr(C)]
struct QueueDesc {
    stype: u32,
    next: *const c_void,
    ordinal: u32,
    index: u32,
    flags: u32,
    mode: u32,
    priority: u32,
}
#[repr(C)]
struct ListDesc {
    stype: u32,
    next: *const c_void,
    ordinal: u32,
    flags: u32,
}
#[repr(C)]
struct ModuleDesc {
    stype: u32,
    next: *const c_void,
    format: u32,
    size: usize,
    input: *const u8,
    flags: *const c_char,
    constants: *const c_void,
}
#[repr(C)]
struct KernelDesc {
    stype: u32,
    next: *const c_void,
    flags: u32,
    name: *const c_char,
}
#[repr(C)]
struct DeviceMemDesc {
    stype: u32,
    next: *const c_void,
    flags: u32,
    ordinal: u32,
}
#[repr(C)]
struct ImportFd {
    stype: u32,
    next: *const c_void,
    flags: u32,
    fd: i32,
}
#[repr(C)]
struct GroupCount {
    x: u32,
    y: u32,
    z: u32,
}
const CONTEXT_DESC: u32 = 0xd;
const QUEUE_DESC: u32 = 0xe;
const LIST_DESC: u32 = 0xf;
const DEVICE_MEM_DESC: u32 = 0x15;
const IMPORT_FD: u32 = 0x19;
const MODULE_DESC: u32 = 0x1b;
const KERNEL_DESC: u32 = 0x1d;
const DMA_BUF: u32 = 2;
const FORMAT_NATIVE: u32 = 1;
const INIT_GPU_ONLY: u32 = 1;

macro_rules! api {
    ($($field:ident = $name:literal: fn($($arg:ty),*);)*) => {
        struct Api {
            $($field: unsafe extern "C" fn($($arg),*) -> Code,)*
            _library: libloading::Library,
        }
        impl Api {
            fn load() -> Result<Self, &'static str> {
                // SAFETY: the loader is a plain C library; each symbol is bound with the
                // signature its header declares, and the library outlives the pointers.
                unsafe {
                    let library = libloading::Library::new("libze_loader.so.1")
                        .map_err(|_| "gpu_unavailable")?;
                    Ok(Self {
                        $($field: *library
                            .get::<unsafe extern "C" fn($($arg),*) -> Code>($name)
                            .map_err(|_| "gpu_unavailable")?,)*
                        _library: library,
                    })
                }
            }
        }
    };
}
api! {
    init = b"zeInit\0": fn(u32);
    driver_get = b"zeDriverGet\0": fn(*mut u32, *mut Handle);
    device_get = b"zeDeviceGet\0": fn(Handle, *mut u32, *mut Handle);
    context_create = b"zeContextCreate\0": fn(Handle, *const ContextDesc, *mut Handle);
    context_destroy = b"zeContextDestroy\0": fn(Handle);
    queue_create = b"zeCommandQueueCreate\0": fn(Handle, Handle, *const QueueDesc, *mut Handle);
    queue_destroy = b"zeCommandQueueDestroy\0": fn(Handle);
    queue_execute = b"zeCommandQueueExecuteCommandLists\0": fn(Handle, u32, *mut Handle, Handle);
    queue_synchronize = b"zeCommandQueueSynchronize\0": fn(Handle, u64);
    list_create = b"zeCommandListCreate\0": fn(Handle, Handle, *const ListDesc, *mut Handle);
    list_destroy = b"zeCommandListDestroy\0": fn(Handle);
    list_close = b"zeCommandListClose\0": fn(Handle);
    list_reset = b"zeCommandListReset\0": fn(Handle);
    list_launch = b"zeCommandListAppendLaunchKernel\0": fn(Handle, Handle, *const GroupCount, Handle, u32, *mut Handle);
    list_barrier = b"zeCommandListAppendBarrier\0": fn(Handle, Handle, u32, *mut Handle);
    mem_alloc_device = b"zeMemAllocDevice\0": fn(Handle, *const DeviceMemDesc, usize, usize, Handle, *mut *mut c_void);
    mem_free = b"zeMemFree\0": fn(Handle, *mut c_void);
    module_create = b"zeModuleCreate\0": fn(Handle, Handle, *const ModuleDesc, *mut Handle, *mut Handle);
    module_destroy = b"zeModuleDestroy\0": fn(Handle);
    kernel_create = b"zeKernelCreate\0": fn(Handle, *const KernelDesc, *mut Handle);
    kernel_destroy = b"zeKernelDestroy\0": fn(Handle);
    kernel_group_size = b"zeKernelSetGroupSize\0": fn(Handle, u32, u32, u32);
    kernel_argument = b"zeKernelSetArgumentValue\0": fn(Handle, u32, usize, *const c_void);
}

fn check(code: Code, error: &'static str) -> Result<(), &'static str> {
    if code == 0 { Ok(()) } else { Err(error) }
}

struct Core {
    api: Api,
    context: Handle,
    device: Handle,
    queue: Handle,
    list: Handle,
    /// The command list and queue are used by one thread at a time.
    turn: Mutex<()>,
}
// SAFETY: the handles are opaque driver objects that may be used from any thread; the one
// stateful pair (list and queue) is only touched while holding `turn`.
unsafe impl Send for Core {}
unsafe impl Sync for Core {}
impl Drop for Core {
    fn drop(&mut self) {
        // SAFETY: every `Memory` and `Kernel` holds an `Arc<Core>`, so none is alive.
        unsafe {
            (self.api.list_destroy)(self.list);
            (self.api.queue_destroy)(self.queue);
            (self.api.context_destroy)(self.context);
        }
    }
}

/// The first Level Zero GPU, with one queue.
#[derive(Clone)]
pub struct LevelZero {
    core: Arc<Core>,
}
/// Device memory another interface allocated, visible to kernels at `address`.
pub struct Memory {
    core: Arc<Core>,
    address: *mut c_void,
}
// SAFETY: the address is only ever handed to the driver as a kernel argument.
unsafe impl Send for Memory {}
unsafe impl Sync for Memory {}
impl Drop for Memory {
    fn drop(&mut self) {
        // SAFETY: launches wait for completion, so no kernel is using the memory.
        unsafe {
            (self.core.api.mem_free)(self.core.context, self.address);
        }
    }
}
/// One kernel of a native module, run one hardware thread per group.
pub struct Kernel {
    core: Arc<Core>,
    module: Handle,
    kernel: Handle,
    pointers: usize,
}
// SAFETY: as for `Core`; arguments are set and the kernel launched while holding `turn`.
unsafe impl Send for Kernel {}
unsafe impl Sync for Kernel {}
impl Kernel {
    /// Work items per group (16 per hardware thread for the machine-code kernels).
    pub fn set_group(&self, group: [u32; 3]) -> Result<(), &'static str> {
        // SAFETY: a live kernel handle; group sizes are plain values the driver validates.
        unsafe {
            check(
                (self.core.api.kernel_group_size)(self.kernel, group[0], group[1], group[2]),
                "gpu_kernel",
            )
        }
    }
}
impl Drop for Kernel {
    fn drop(&mut self) {
        // SAFETY: as for `Memory`.
        unsafe {
            (self.core.api.kernel_destroy)(self.kernel);
            (self.core.api.module_destroy)(self.module);
        }
    }
}

impl LevelZero {
    pub fn open() -> Result<Self, &'static str> {
        let api = Api::load()?;
        // SAFETY: plain setup calls with out-parameters that live across each call; the
        // handles are owned by `Core`, which destroys them.
        unsafe {
            check((api.init)(INIT_GPU_ONLY), "gpu_unavailable")?;
            let (mut count, mut driver, mut device) = (1u32, null_mut(), null_mut());
            check((api.driver_get)(&mut count, &mut driver), "gpu_unavailable")?;
            if count == 0 || driver.is_null() {
                return Err("gpu_unavailable");
            }
            count = 1;
            check(
                (api.device_get)(driver, &mut count, &mut device),
                "gpu_unavailable",
            )?;
            if count == 0 || device.is_null() {
                return Err("gpu_unavailable");
            }
            let mut context = null_mut();
            let desc = ContextDesc {
                stype: CONTEXT_DESC,
                next: null(),
                flags: 0,
            };
            check(
                (api.context_create)(driver, &desc, &mut context),
                "gpu_unavailable",
            )?;
            let (mut queue, mut list) = (null_mut(), null_mut());
            let desc = QueueDesc {
                stype: QUEUE_DESC,
                next: null(),
                ordinal: 0,
                index: 0,
                flags: 0,
                mode: 0,
                priority: 0,
            };
            if (api.queue_create)(context, device, &desc, &mut queue) != 0 {
                (api.context_destroy)(context);
                return Err("gpu_unavailable");
            }
            let desc = ListDesc {
                stype: LIST_DESC,
                next: null(),
                ordinal: 0,
                flags: 0,
            };
            if (api.list_create)(context, device, &desc, &mut list) != 0 {
                (api.queue_destroy)(queue);
                (api.context_destroy)(context);
                return Err("gpu_unavailable");
            }
            Ok(Self {
                core: Arc::new(Core {
                    api,
                    context,
                    device,
                    queue,
                    list,
                    turn: Mutex::new(()),
                }),
            })
        }
    }
    /// Maps a dma-buf of `bytes` bytes into this device's address space. The descriptor can
    /// be closed afterwards: the mapping keeps the memory alive.
    pub fn import(&self, fd: &OwnedFd, bytes: u64) -> Result<Memory, &'static str> {
        let core = &self.core;
        let import = ImportFd {
            stype: IMPORT_FD,
            next: null(),
            flags: DMA_BUF,
            fd: fd.as_raw_fd(),
        };
        let desc = DeviceMemDesc {
            stype: DEVICE_MEM_DESC,
            next: (&raw const import).cast(),
            flags: 0,
            ordinal: 0,
        };
        let mut address = null_mut();
        // SAFETY: both descriptors live across the call.
        let code = unsafe {
            (core.api.mem_alloc_device)(
                core.context,
                &desc,
                bytes as usize,
                64,
                core.device,
                &mut address,
            )
        };
        check(code, "gpu_memory")?;
        Ok(Memory {
            core: core.clone(),
            address,
        })
    }
    /// Loads a native module (see `zebin`) holding the kernel `name`, which takes `pointers`
    /// memory arguments.
    /// Kernel `name` of a native module compiled from OpenCL (by `ocloc`), run with workgroups
    /// of `group` work-items. Its arguments are given per launch (`Arg`).
    pub fn compiled_kernel(
        &self,
        native: &[u8],
        name: &str,
        group: [u32; 3],
    ) -> Result<Kernel, &'static str> {
        let core = &self.core;
        let name = CString::new(name).map_err(|_| "gpu_kernel")?;
        let desc = ModuleDesc {
            stype: MODULE_DESC,
            next: null(),
            format: FORMAT_NATIVE,
            size: native.len(),
            input: native.as_ptr(),
            flags: c"".as_ptr(),
            constants: null(),
        };
        // SAFETY: as for `kernel`.
        unsafe {
            let mut module = null_mut();
            check(
                (core.api.module_create)(core.context, core.device, &desc, &mut module, null_mut()),
                "gpu_kernel",
            )?;
            let desc = KernelDesc {
                stype: KERNEL_DESC,
                next: null(),
                flags: 0,
                name: name.as_ptr(),
            };
            let mut kernel = null_mut();
            if (core.api.kernel_create)(module, &desc, &mut kernel) != 0 {
                (core.api.module_destroy)(module);
                return Err("gpu_kernel");
            }
            let out = Kernel {
                core: core.clone(),
                module,
                kernel,
                pointers: usize::MAX,
            };
            check(
                (core.api.kernel_group_size)(kernel, group[0], group[1], group[2]),
                "gpu_kernel",
            )?;
            Ok(out)
        }
    }
    /// A command list to fill with launches (`Batch::launch`), run in order with one wait.
    pub fn batch(&self) -> Result<Batch<'_>, &'static str> {
        let core = &self.core;
        let desc = ListDesc {
            stype: LIST_DESC,
            next: null(),
            ordinal: 0,
            flags: 0,
        };
        let mut list = null_mut();
        // SAFETY: the list is destroyed by `Batch`.
        unsafe {
            check(
                (core.api.list_create)(core.context, core.device, &desc, &mut list),
                "gpu_failed",
            )?;
        }
        Ok(Batch { fast: self, list })
    }
    pub fn kernel(
        &self,
        native: &[u8],
        name: &str,
        pointers: usize,
    ) -> Result<Kernel, &'static str> {
        let core = &self.core;
        let name = CString::new(name).map_err(|_| "gpu_kernel")?;
        let desc = ModuleDesc {
            stype: MODULE_DESC,
            next: null(),
            format: FORMAT_NATIVE,
            size: native.len(),
            input: native.as_ptr(),
            flags: c"".as_ptr(),
            constants: null(),
        };
        // SAFETY: descriptors and their strings live across the calls; a module that fails
        // to yield its kernel is destroyed here.
        unsafe {
            let mut module = null_mut();
            check(
                (core.api.module_create)(core.context, core.device, &desc, &mut module, null_mut()),
                "gpu_kernel",
            )?;
            let desc = KernelDesc {
                stype: KERNEL_DESC,
                next: null(),
                flags: 0,
                name: name.as_ptr(),
            };
            let mut kernel = null_mut();
            if (core.api.kernel_create)(module, &desc, &mut kernel) != 0 {
                (core.api.module_destroy)(module);
                return Err("gpu_kernel");
            }
            let out = Kernel {
                core: core.clone(),
                module,
                kernel,
                pointers,
            };
            check((core.api.kernel_group_size)(kernel, 1, 1, 1), "gpu_kernel")?;
            Ok(out)
        }
    }
    /// Runs `kernel` once per group with `memory` as its arguments and waits for it.
    pub fn launch(
        &self,
        kernel: &Kernel,
        memory: &[&Memory],
        groups: [u32; 3],
    ) -> Result<(), &'static str> {
        let core = &self.core;
        if memory.len() != kernel.pointers {
            return Err("gpu_kernel");
        }
        let _turn = core.turn.lock().map_err(|_| "gpu_failed")?;
        let count = GroupCount {
            x: groups[0],
            y: groups[1],
            z: groups[2],
        };
        // SAFETY: arguments are copied by the driver during each call; the list is reset
        // whether or not the launch succeeded, after the queue has drained.
        unsafe {
            for (index, memory) in memory.iter().enumerate() {
                check(
                    (core.api.kernel_argument)(
                        kernel.kernel,
                        index as u32,
                        8,
                        (&raw const memory.address).cast(),
                    ),
                    "gpu_failed",
                )?;
            }
            let mut list = core.list;
            let result = check(
                (core.api.list_launch)(core.list, kernel.kernel, &count, null_mut(), 0, null_mut()),
                "gpu_failed",
            )
            .and_then(|()| check((core.api.list_close)(core.list), "gpu_failed"))
            .and_then(|()| {
                check(
                    (core.api.queue_execute)(core.queue, 1, &mut list, null_mut()),
                    "gpu_failed",
                )
            });
            let waited = check(
                (core.api.queue_synchronize)(core.queue, u64::MAX),
                "gpu_failed",
            );
            let reset = check((core.api.list_reset)(core.list), "gpu_failed");
            result.and(waited).and(reset)
        }
    }
}

/// A kernel argument.
#[derive(Clone, Copy)]
pub enum Arg<'a> {
    Memory(&'a Memory),
    U32(u32),
    F32(f32),
}
/// Launches recorded into one command list, each after the previous has finished.
pub struct Batch<'a> {
    fast: &'a LevelZero,
    list: Handle,
}
impl Batch<'_> {
    /// Appends `kernel` with `args` over `groups` workgroups, after everything before it.
    pub fn launch(
        &mut self,
        kernel: &Kernel,
        args: &[Arg],
        groups: [u32; 3],
    ) -> Result<(), &'static str> {
        let core = &self.fast.core;
        if kernel.pointers != usize::MAX && args.len() != kernel.pointers {
            return Err("gpu_kernel");
        }
        let _turn = core.turn.lock().map_err(|_| "gpu_failed")?;
        let count = GroupCount {
            x: groups[0],
            y: groups[1],
            z: groups[2],
        };
        // SAFETY: argument values are copied when the launch is appended.
        unsafe {
            for (index, arg) in args.iter().enumerate() {
                let (size, value): (usize, *const c_void) = match arg {
                    Arg::Memory(memory) => (8, (&raw const memory.address).cast()),
                    Arg::U32(value) => (4, (value as *const u32).cast()),
                    Arg::F32(value) => (4, (value as *const f32).cast()),
                };
                check(
                    (core.api.kernel_argument)(kernel.kernel, index as u32, size, value),
                    "gpu_kernel",
                )?;
            }
            check(
                (core.api.list_launch)(self.list, kernel.kernel, &count, null_mut(), 0, null_mut()),
                "gpu_failed",
            )?;
            check(
                (core.api.list_barrier)(self.list, null_mut(), 0, null_mut()),
                "gpu_failed",
            )
        }
    }
    /// Runs the list and waits for it.
    pub fn run(self) -> Result<(), &'static str> {
        let core = &self.fast.core;
        let _turn = core.turn.lock().map_err(|_| "gpu_failed")?;
        // SAFETY: the list is closed, executed and waited for before `Drop` destroys it.
        unsafe {
            let mut list = self.list;
            check((core.api.list_close)(self.list), "gpu_failed")?;
            check(
                (core.api.queue_execute)(core.queue, 1, &mut list, null_mut()),
                "gpu_failed",
            )?;
            check(
                (core.api.queue_synchronize)(core.queue, u64::MAX),
                "gpu_failed",
            )
        }
    }
}
impl Drop for Batch<'_> {
    fn drop(&mut self) {
        let core = &self.fast.core;
        // SAFETY: a list that was executed has been waited for in `run`; on any failure the
        // queue is drained first.
        unsafe {
            let _ = (core.api.queue_synchronize)(core.queue, u64::MAX);
            (core.api.list_destroy)(self.list);
        }
    }
}
