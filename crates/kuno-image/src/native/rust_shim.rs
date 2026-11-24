//! The C++ shim's entry points for a worker built without the C++ engine (`rust-image`):
//! every model runs in Rust, so the session is an empty token and only cancellation is
//! real. SIGTERM or SIGINT during a request sets the flag the Rust engines poll between
//! steps (the C++ shim's cooperative shutdown, without restoring handlers per session:
//! the worker process holds one session for its life).
use std::{
    ffi::{c_char, c_void},
    sync::atomic::{AtomicBool, Ordering},
};

static CANCELLED: AtomicBool = AtomicBool::new(false);
/// A non-null token for the session pointer.
static SESSION: u8 = 0;

const SIGINT: i32 = 2;
const SIGTERM: i32 = 15;
unsafe extern "C" {
    fn signal(signum: i32, handler: extern "C" fn(i32)) -> usize;
}
extern "C" fn request_cancel(_: i32) {
    CANCELLED.store(true, Ordering::Relaxed);
}

#[unsafe(no_mangle)]
extern "C" fn kuno_qwen_session_new(
    directory: *const c_char,
    pci: *const c_char,
    out: *mut *mut c_void,
) -> i32 {
    if directory.is_null() || pci.is_null() || out.is_null() {
        return 5;
    }
    CANCELLED.store(false, Ordering::Relaxed);
    // SAFETY: `request_cancel` only stores to an atomic, which is async-signal-safe.
    unsafe {
        signal(SIGTERM, request_cancel);
        signal(SIGINT, request_cancel);
    }
    // SAFETY: the caller passes a valid out pointer (checked non-null above).
    unsafe { *out = (&raw const SESSION).cast_mut().cast() };
    0
}
#[unsafe(no_mangle)]
extern "C" fn kuno_qwen_session_free(_session: *mut c_void) {}
#[unsafe(no_mangle)]
extern "C" fn kuno_qwen_session_release(_session: *mut c_void) {}
#[unsafe(no_mangle)]
extern "C" fn kuno_image_cancel_requested() -> bool {
    CANCELLED.load(Ordering::Relaxed)
}
/// The C++ engine's own image and video paths: absent from this build.
#[unsafe(no_mangle)]
extern "C" fn kuno_qwen_session_generate() -> i32 {
    9
}
#[unsafe(no_mangle)]
extern "C" fn kuno_minimax_session_generate() -> i32 {
    9
}
