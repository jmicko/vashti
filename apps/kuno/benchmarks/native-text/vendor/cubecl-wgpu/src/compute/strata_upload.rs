//! Original upload ownership and synchronous native host writes. No data cache.
use std::cell::Cell;
use wgpu::hal::Device as _;

/// These states are bookkeeping, NOT fence authority. Only the original stream
/// may grant NativeReady, after joining actual GPU readers/copies of the owner.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum State {
    Idle,
    WgpuMapped,
    NativeReady,
    NativeWriting,
    NativeWritten,
}
impl State {
    pub fn reusable(self) -> bool {
        matches!(self, Self::WgpuMapped | Self::NativeReady)
    }
    pub fn staged(self) -> bool {
        matches!(self, Self::WgpuMapped | Self::NativeWritten)
    }
    pub fn begin_native_write(&mut self) -> Result<(), &'static str> {
        if !matches!(self, Self::NativeReady | Self::NativeWritten) {
            return Err("native upload original write grant absent");
        }
        *self = Self::NativeWriting; // Before fallible map/write/flush/unmap.
        Ok(())
    }
    pub fn finish_native_write(&mut self) -> Result<(), &'static str> {
        if *self != Self::NativeWriting {
            return Err("native upload original writer changed");
        }
        *self = Self::NativeWritten;
        Ok(())
    }
    /// A blocked owner gets no new writer. Revoke an unused native grant;
    /// an unexpected WG map or partial/staged writer on a blocked buffer fails.
    pub fn after_selected_join(&mut self, admitted: bool) -> Result<(), &'static str> {
        if admitted {
            self.after_joined()
        } else {
            match self {
                Self::Idle | Self::NativeReady => {
                    *self = Self::Idle;
                    Ok(())
                }
                _ => Err("native excluded upload has a mapped or unsubmitted writer"),
            }
        }
    }
    /// Actual original GPU readers/copies have joined; never call on enqueue,
    /// a peer fence, retained metadata or merely an empty Wgpu encoder.
    pub fn after_joined(&mut self) -> Result<(), &'static str> {
        match self {
            Self::Idle | Self::NativeReady => *self = Self::NativeReady,
            Self::WgpuMapped => {} // Unused mapped-at-creation owner.
            Self::NativeWritten | Self::NativeWriting => {
                return Err("native upload has unsubmitted writes");
            }
        }
        Ok(())
    }
}

/// Both Wgpu write-only mapped views and native writes use this exact renderer.
/// No readable mapped slice or pointer is exposed to record/data producers.
pub(super) fn render_bounded<T, E>(
    bytes: usize,
    mut put: impl FnMut(usize, &[u8]),
    render: impl FnOnce(&mut dyn FnMut(usize, &[u8])) -> Result<T, E>,
    fault: impl FnOnce(&'static str) -> E,
) -> Result<T, E> {
    let failed = Cell::new(false);
    let result = render(&mut |at, data| {
        if failed.get() || at.checked_add(data.len()).is_none_or(|end| end > bytes) {
            failed.set(true);
            return;
        }
        put(at, data);
    });
    if failed.get() {
        Err(fault("native upload write outside original full owner"))
    } else {
        result
    }
}

struct Mapping<'a> {
    device: &'a wgpu::hal::vulkan::Device,
    buffer: &'a wgpu::hal::vulkan::Buffer,
    failed: &'a Cell<bool>,
}
impl Drop for Mapping<'_> {
    fn drop(&mut self) {
        if std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| unsafe {
            self.device.unmap_buffer(self.buffer);
        }))
        .is_err()
        {
            self.failed.set(true);
        }
    }
}

/// # Safety
/// Original stream has joined this exact owner's previous GPU read/copy and
/// granted NativeReady, or owns its still-unsubmitted NativeWritten append page.
/// Buffer is an original tracked full MAP_WRITE/COPY_SRC owner, Wgpu-unmapped.
/// No GPU use, Wgpu map, reuse or erasure may touch it during this local call.
/// Partial writes and errors retain original owner and retire the caller's I/O.
pub(super) unsafe fn write_joined<T, E>(
    device: &wgpu::Device,
    buffer: &wgpu::Buffer,
    clear: bool,
    render: impl FnOnce(&mut dyn FnMut(usize, &[u8])) -> Result<T, E>,
    mut fault: impl FnMut(&'static str) -> E,
) -> Result<T, E> {
    let bytes = buffer.size();
    if bytes < 256
        || bytes > 128 * 1024 * 1024
        || !bytes.is_power_of_two()
        || !buffer
            .usage()
            .contains(wgpu::BufferUsages::MAP_WRITE | wgpu::BufferUsages::COPY_SRC)
    {
        return Err(fault("native upload original mapped owner"));
    }
    let device = unsafe { device.as_hal::<wgpu::hal::api::Vulkan>() }
        .ok_or_else(|| fault("native upload original Vulkan device"))?;
    let buffer = unsafe { buffer.as_hal::<wgpu::hal::api::Vulkan>() }
        .ok_or_else(|| fault("native upload original Vulkan buffer"))?;
    let mapped = unsafe { device.map_buffer(&buffer, 0..bytes) }
        .map_err(|_| fault("native upload original host map"))?;
    let failed = Cell::new(false);
    let mapping = Mapping {
        device: &device,
        buffer: &buffer,
        failed: &failed,
    };
    if clear {
        // SAFETY: original full write grant; initialize padding, too. No read.
        unsafe {
            std::ptr::write_bytes(mapped.ptr.as_ptr(), 0, bytes as usize);
        }
    }
    let result = render_bounded(
        bytes as usize,
        |at, data| unsafe {
            // SAFETY: checked actual full-owner range. Producers receive no raw
            // pointer/read view; their input bytes cannot alias this private map.
            std::ptr::copy_nonoverlapping(data.as_ptr(), mapped.ptr.as_ptr().add(at), data.len());
        },
        render,
        &mut fault,
    );
    if !mapped.is_coherent {
        // Flush even partial/error writes before guarded unmap. No GPU copy is
        // authorized on error; original terminal wipe still owns every byte.
        unsafe {
            device.flush_mapped_ranges(&buffer, std::iter::once(0..bytes));
        }
    }
    drop(mapping);
    if failed.get() {
        return Err(fault("native upload original host unmap"));
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn upload_state_separates_unjoined_reusable_and_staged_original_owners() {
        let mut state = State::Idle;
        assert!(!state.reusable());
        assert!(!state.staged());
        assert!(state.begin_native_write().is_err());
        state.after_joined().unwrap(); // Synthetic host outcome, not a fence.
        assert_eq!(state, State::NativeReady);
        assert!(state.reusable());
        assert!(!state.staged());
        state.begin_native_write().unwrap();
        assert!(!state.reusable());
        assert!(!state.staged()); // Map/write/flush/unmap have not succeeded.
        assert!(state.after_joined().is_err());
        state.finish_native_write().unwrap();
        assert!(state.staged());
        state.begin_native_write().unwrap(); // Same unsubmitted append page.
        state.finish_native_write().unwrap();
        assert!(state.after_joined().is_err()); // May not recycle staged input.
        state = State::Idle; // Production only after actual source unmap/seal.
        state.after_joined().unwrap();
        let mut unused = State::WgpuMapped;
        assert!(unused.reusable() && unused.staged());
        assert!(unused.begin_native_write().is_err());
        unused.after_joined().unwrap();
        assert_eq!(unused, State::WgpuMapped);
    }
    #[test]
    fn bounded_writer_preserves_bits_append_and_rejects_overflow_without_later_writes() {
        let mut bytes = [0u8; 256];
        render_bounded(
            bytes.len(),
            |at, data| bytes[at..at + data.len()].copy_from_slice(data),
            |put| {
                put(0, &[0, 0, 0, 128, 1, 0, 192, 127]); // Exact signed-zero/NaN payload bits.
                put(252, &[1, 2, 3, 4]);
                Ok::<_, &'static str>(())
            },
            |e| e,
        )
        .unwrap();
        assert_eq!(&bytes[..8], &[0, 0, 0, 128, 1, 0, 192, 127]);
        assert_eq!(&bytes[252..], &[1, 2, 3, 4]);
        assert!(bytes[8..252].iter().all(|&b| b == 0));
        let mut writes = 0;
        assert!(
            render_bounded(
                256,
                |_, _| writes += 1,
                |put| {
                    put(0, &[1]);
                    put(usize::MAX, &[2]);
                    put(1, &[3]); // Must not continue after invalid original span.
                    Ok::<_, &'static str>(())
                },
                |e| e
            )
            .is_err()
        );
        assert_eq!(writes, 1);
        assert!(
            render_bounded(
                256,
                |_, _| unreachable!(),
                |put| {
                    put(255, &[1, 2]);
                    Ok::<_, &'static str>(())
                },
                |e| e
            )
            .is_err()
        );
    }
}
