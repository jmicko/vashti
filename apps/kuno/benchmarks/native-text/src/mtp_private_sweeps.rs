//! Fixed original MTP owner sweeps. Only launch count changes: every complete
//! owner word keeps the original finite/prefix/padding predicate. The caller
//! proves full extent, source role and physical disjointness before enqueue.
use cubecl::prelude::*;

pub const INTEGER: u32 = 0;
pub const FLOAT: u32 = 1;
pub const CACHE: u32 = 2;

#[cube]
#[allow(clippy::manual_range_contains)]
fn check_word(
    values: &Array<u32>,
    fault: &mut Array<Atomic<u32>>,
    i: usize,
    cache_position: usize,
    #[comptime] logical: usize,
    #[comptime] role: u32,
) {
    // Actual span branch precedes every load, even in unchecked lowering.
    if i < values.len() {
        let bits = values[i];
        let mut outside = i >= logical;
        if comptime!(role == CACHE) {
            outside = outside || i >= cache_position * crate::mtp_resident_kernels::KV;
        }
        if outside {
            if bits != 0 {
                fault[0].fetch_or(4);
            }
        } else if comptime!(role != INTEGER) {
            let value = f32::reinterpret(bits);
            if !(value >= -3.4028235e38_f32 && value <= 3.4028235e38_f32) {
                fault[0].fetch_or(1);
            }
        }
    }
}

macro_rules! guard_kernel {
    ($name:ident, $(($a:ident, $logical:ident, $role:ident)),+) => {
        #[cube(launch)]
        #[allow(clippy::too_many_arguments)]
        pub fn $name(
            $($a: &Array<u32>,)+
            fault: &mut Array<Atomic<u32>>,
            cache_position: usize,
            $(#[comptime] $logical: usize, #[comptime] $role: u32,)+
        ) {
            let i = ABSOLUTE_POS;
            $(check_word($a, fault, i, cache_position, $logical, $role);)+
        }
    };
}
guard_kernel!(
    guard6,
    (a0, n0, r0),
    (a1, n1, r1),
    (a2, n2, r2),
    (a3, n3, r3),
    (a4, n4, r4),
    (a5, n5, r5)
);
guard_kernel!(
    guard5,
    (a0, n0, r0),
    (a1, n1, r1),
    (a2, n2, r2),
    (a3, n3, r3),
    (a4, n4, r4)
);

/// Fault is its own atomic binding, never re-bound as a read-only u32 array.
/// Original final fault-role sweep had live0, rejecting every nonzero word.
/// Run after all role guards, so their sticky bits receive the same bit4 fold.
#[cube(launch)]
#[allow(clippy::collapsible_if)]
pub fn guard_fault(fault: &mut Array<Atomic<u32>>) {
    let i = ABSOLUTE_POS;
    if i < fault.len() {
        if fault[i].load() != 0 {
            fault[0].fetch_or(4);
        }
    }
}

macro_rules! clear_kernel {
    ($name:ident, $($a:ident),+) => {
        #[cube(launch)]
        pub fn $name($($a: &mut Array<u32>,)+) {
            let i = ABSOLUTE_POS;
            $(if i < $a.len() { $a[i] = 0; })+
        }
    };
}
// Exact remaining widths avoid dummy/repeated owner bindings or reflection
// mismatch. Explicit start clears40 owners; GPU followup preserves3 of them.
clear_kernel!(clear7, a0, a1, a2, a3, a4, a5, a6);
clear_kernel!(clear5, a0, a1, a2, a3, a4);
clear_kernel!(clear2, a0, a1);
