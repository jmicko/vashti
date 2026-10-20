//! Shared compiled-public original routed-expert oracle and complete-owner audit.
//! Launch selection remains in each actual backend's small diagnostic wrapper.
use crate::private_arena::{BufferId, PrivateArena};
use crate::resident_util as u;
use crate::strata_prompt::groups as abi;
use cubecl::{prelude::ComputeClient, server::Handle, wgpu::WgpuRuntime};
use half::f16;
use zeroize::{Zeroize, Zeroizing};
pub(super) type ProbeResult<T> = std::result::Result<T, Box<dyn std::error::Error>>;
pub(super) type Launcher = fn(
    &ComputeClient<WgpuRuntime>,
    [Handle; 6],
    usize,
    usize,
    usize,
    usize,
    usize,
    bool,
) -> ProbeResult<bool>;
const WINDOW_BYTES: usize = 16 * 1024 * 1024;
const TOTAL: usize = 4096;
const PREFIX: usize = 128;
const STATUS_GAP: u32 = 0xa513579b;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Control {
    Finite,
    NonfiniteFirst,
    NonfiniteOther,
    InvalidRoute,
    InvalidRoutePrefix,
}
impl Control {
    fn name(self) -> &'static str {
        match self {
            Self::Finite => "finite-ieee",
            Self::NonfiniteFirst => "nonfinite-feature-zero",
            Self::NonfiniteOther => "nonfinite-other-feature",
            Self::InvalidRoute => "invalid-route-bound",
            Self::InvalidRoutePrefix => "invalid-route-prefix-bound",
        }
    }
    fn bad_feature(self) -> Option<usize> {
        match self {
            Self::NonfiniteFirst => Some(0),
            Self::NonfiniteOther => Some(4),
            _ => None,
        }
    }
    fn bad_member(self, count: usize) -> Option<usize> {
        match self {
            Self::InvalidRoute => Some(count - 1),
            Self::InvalidRoutePrefix => Some(0),
            _ => None,
        }
    }
}

struct Fixture {
    map: Zeroizing<Vec<u32>>,
    routes: Zeroizing<Vec<usize>>,
    group: usize,
    status: Zeroizing<Vec<u32>>,
    input: Zeroizing<Vec<f32>>,
    input_image: Zeroizing<Vec<u32>>,
    weight: Zeroizing<Vec<f32>>,
    weight_image: Zeroizing<Vec<u32>>,
    columns: usize,
    rows: usize,
}

fn fixture(count: usize, down: bool, control: Control) -> ProbeResult<Fixture> {
    if ![32, 33, 64, 256, 257, 287, 288, 383, 384].contains(&count) {
        return Err("public CubeK expert count".into());
    }
    let (columns, rows) = if down { (640, 2560) } else { (2560, 1280) };
    let mut raw = Zeroizing::new(vec![0u32; abi::LOGICAL + TOTAL * 10]);
    raw[..3].copy_from_slice(&[TOTAL as u32, 0, TOTAL as u32]);
    let mut routes = Zeroizing::new(Vec::with_capacity(count));
    for token in 0..TOTAL {
        for rank in 0..10 {
            let route = token * 10 + rank;
            let chosen = token >= TOTAL - count && rank == token % 10;
            let logical = if chosen { 400 } else { 1 + rank };
            raw[abi::LOGICAL + route] = logical as u32;
            raw[1540 + route] = ((logical * 13 + 17) % 512) as u32;
            if chosen {
                routes.push(route);
            }
        }
    }
    let mut map = abi::group(&raw, TOTAL)?;
    let groups = abi::validate(&map, TOTAL)?;
    let group = groups - 1;
    let offset = map[1028 + group] as usize;
    if routes.len() != count
        || map[516 + group] as usize != count
        || map[1540 + offset..1540 + offset + count]
            .iter()
            .zip(&*routes)
            .any(|(&a, &b)| a as usize != b)
        || routes.iter().any(|&r| r <= 20479)
        || offset == 0
    {
        return Err("public CubeK expert authentic map".into());
    }
    if let Some(member) = control.bad_member(count) {
        map[1540 + offset + member] = (TOTAL * 10) as u32;
    }
    let mut status = Zeroizing::new(vec![STATUS_GAP; abi::STATUS_WORDS]);
    for (member, &route) in routes.iter().enumerate() {
        status[route] = u32::from(member % 3 != 1);
    }
    let mut input = Zeroizing::new(vec![0f32; count * columns]);
    let mut input_image = Zeroizing::new(vec![0u32; input.len() / 2]);
    for (i, value) in input.iter_mut().enumerate() {
        let v = f16::from_f32(0.125 + ((i * 7 + i / columns * 13) % 43) as f32 / 19.0);
        *value = v.to_f32();
        input_image[i / 2] |= u32::from(v.to_bits()) << (16 * (i % 2));
    }
    let mut weight = Zeroizing::new(vec![0f32; rows * columns]);
    let mut weight_image = Zeroizing::new(vec![0u32; (PREFIX + weight.len()) / 2]);
    // Prefix is a genuine descriptor-aligned offset, not a pointer fiction.
    weight_image[..PREFIX / 2].fill(0x35553555);
    for (i, value) in weight.iter_mut().enumerate() {
        let row = i / columns;
        let v = if row == 2 {
            f16::from_bits(128)
        } else {
            f16::from_f32((((i * 11 + row * 17) % 67) as f32 - 33.0) / 41.0)
        };
        *value = v.to_f32();
        let at = PREFIX + i;
        weight_image[at / 2] |= u32::from(v.to_bits()) << (16 * (at % 2));
    }
    if let Some(row) = control.bad_feature() {
        let at = PREFIX + row * columns;
        weight[row * columns] = f32::NAN;
        weight_image[at / 2] = (weight_image[at / 2] & 0xffff0000) | 0x7e00;
    }
    Ok(Fixture {
        map,
        routes,
        group,
        status,
        input,
        input_image,
        weight,
        weight_image,
        columns,
        rows,
    })
}

// Guard the genuine returned mapped lease even if a numerical/bounds check
// fails. The backend's actual readback sanitizer is joined after every window.
struct Scrub<'a>(&'a mut [u8]);
impl Drop for Scrub<'_> {
    fn drop(&mut self) {
        self.0.zeroize();
    }
}
fn read_window(root: &Handle, start: usize, len: usize) -> ProbeResult<Handle> {
    let used = root
        .size()
        .checked_sub(root.offset_start.unwrap_or(0))
        .and_then(|n| n.checked_sub(root.offset_end.unwrap_or(0)))
        .ok_or("public CubeK expert original window owner")?;
    let end = start
        .checked_add(len)
        .ok_or("public CubeK expert window overflow")?;
    if len == 0 || !start.is_multiple_of(4) || !len.is_multiple_of(4) || end as u64 > used {
        return Err("public CubeK expert original window bounds".into());
    }
    // Handle slicing ADDS relative offsets. Retain both original boundaries.
    Ok(root
        .clone()
        .offset_start(start as u64)
        .offset_end(used - end as u64))
}
fn scan(
    arena: &PrivateArena,
    id: BufferId,
    mut check: impl FnMut(usize, u32) -> ProbeResult<()>,
) -> ProbeResult<usize> {
    let client = arena.client()?;
    let layout = arena.layout(id)?;
    let root = arena.handle(id)?;
    if root.size_in_used() != layout.owned_bytes as u64 {
        return Err("public CubeK expert owner extent".into());
    }
    let mut scanned = 0;
    for start in (0..layout.owned_bytes).step_by(WINDOW_BYTES) {
        let len = (layout.owned_bytes - start).min(WINDOW_BYTES);
        let view = read_window(root, start, len)?;
        let mut lease = client.read_one(view)?;
        let result: ProbeResult<()> = (|| {
            let guarded = Scrub(&mut lease);
            if guarded.0.len() != len {
                return Err("public CubeK expert read extent".into());
            }
            for (i, word) in guarded.0.chunks_exact(4).enumerate() {
                check(start / 4 + i, u32::from_le_bytes(word.try_into()?))?;
            }
            Ok(())
        })();
        drop(lease);
        let audit = cubecl::future::block_on(client.sanitize_transient_io_audited())?;
        let bytes = audit
            .upload_bytes
            .checked_add(audit.uniform_bytes)
            .and_then(|v| v.checked_add(audit.readback_bytes))
            .ok_or("public CubeK expert I/O count")?;
        if bytes == 0 || audit.inspected_bytes != bytes || audit.inspected_buffers == 0 {
            return Err("public CubeK expert I/O audit".into());
        }
        result?;
        scanned += len / 4;
    }
    Ok(scanned)
}

fn one(
    client: &ComputeClient<WgpuRuntime>,
    arena: &mut PrivateArena,
    count: usize,
    down: bool,
    control: Control,
    family: &'static str,
    launch_field: &'static str,
    launch: Launcher,
) -> ProbeResult<serde_json::Value> {
    let f = fixture(count, down, control)?;
    let input = arena.upload_u32(&f.input_image)?;
    let weight = arena.upload_u32(&f.weight_image)?;
    let output =
        arena.allocate_zeroed((if down { abi::PART_BYTES } else { abi::GU_BYTES }) as usize / 4)?;
    let map = arena.upload_u32(&f.map)?;
    let status = arena.upload_u32(&f.status)?;
    let fault = arena.allocate_zeroed(1)?;
    let handles = [input, weight, output, map, status, fault]
        .map(|id| u::handle(arena, id))
        .into_iter()
        .collect::<std::result::Result<Vec<_>, _>>()?;
    let handles: [Handle; 6] = handles
        .try_into()
        .map_err(|_| "public CubeK expert six owners")?;
    let actual_launch = launch(
        client, handles, count, f.columns, f.rows, PREFIX, f.group, down,
    )?;
    // Independent full output oracle. Only public fixed arithmetic is computed;
    // original IEEE tolerance remains unchanged, including preserved subnormals.
    let mut expected = Zeroizing::new(vec![0f32; count * f.rows]);
    for member in 0..count {
        if control.bad_member(count) == Some(member) {
            continue;
        }
        for row in 0..f.rows {
            if control.bad_feature() == Some(row) {
                continue;
            }
            expected[member * f.rows + row] = f.weight[row * f.columns..][..f.columns]
                .iter()
                .zip(&f.input[member * f.columns..][..f.columns])
                .map(|(&w, &x)| f64::from(w) * f64::from(x))
                .sum::<f64>() as f32;
        }
    }
    let mut by_route = Zeroizing::new(vec![usize::MAX; abi::ROUTES]);
    for (member, &route) in f.routes.iter().enumerate() {
        if control.bad_member(count) != Some(member) {
            by_route[route] = member;
        }
    }
    let mut mismatches = 0;
    let mut max_abs = 0f32;
    let output_words = scan(arena, output, |at, bits| {
        let member = if down {
            by_route.get(at / f.rows).copied().unwrap_or(usize::MAX)
        } else {
            at / f.rows
        };
        if member >= count {
            if bits != 0 {
                return Err("public CubeK expert inactive owner/padding".into());
            }
        } else {
            let row = at % f.rows;
            let reference = expected[member * f.rows + row];
            let value = f32::from_bits(bits);
            let error = (value - reference).abs();
            mismatches += usize::from(!value.is_finite() || error > 1e-4 + 3e-4 * reference.abs());
            max_abs = max_abs.max(error);
            if (control.bad_feature() == Some(row) || control.bad_member(count) == Some(member))
                && bits != 0
            {
                return Err("public CubeK expert fault replacement".into());
            }
        }
        Ok(())
    })?;
    let mut expected_status = f.status.clone();
    if down {
        for (member, &route) in f.routes.iter().enumerate() {
            if control.bad_member(count) == Some(member) {
                continue;
            }
            expected_status[abi::ROUTES + route] =
                u32::from(control != Control::NonfiniteFirst && f.status[route] == 1);
        }
    }
    let status_words = scan(arena, status, |at, bits| {
        if bits != expected_status.get(at).copied().unwrap_or(0) {
            return Err("public CubeK expert full status owner".into());
        }
        Ok(())
    })?;
    let expected_fault = if control == Control::Finite { 0 } else { 128 };
    scan(arena, fault, |at, bits| {
        if bits != if at == 0 { expected_fault } else { 0 } {
            return Err("public CubeK expert full fault owner".into());
        }
        Ok(())
    })?;
    for (id, image) in [
        (input, &*f.input_image),
        (weight, &*f.weight_image),
        (map, &*f.map),
    ] {
        scan(arena, id, |at, bits| {
            if bits != image.get(at).copied().unwrap_or(0) {
                return Err("public CubeK expert original source/padding changed".into());
            }
            Ok(())
        })?;
    }
    if mismatches != 0 {
        return Err("public CubeK expert unchanged IEEE parity".into());
    }
    let mut report = serde_json::json!({"family":family,"down":down,"count":count,"rows":f.rows,"columns":f.columns,
        "control":control.name(),"mismatches":mismatches,"max_abs":max_abs,
        "full_output_words_checked":output_words,"full_status_words_checked":status_words,
        "full_owner_status_fault_sources_and_tail_checked":true,"original_map_builder_and_validator":true,
        "deliberately_mutated_after_original_validation":control.bad_member(count).is_some(),
        "noncontiguous_routes_above_20479":true,"real_weight_offset_half_elements":PREFIX});
    report[launch_field] = actual_launch.into();
    Ok(report)
}

pub(super) fn cases(
    client: &ComputeClient<WgpuRuntime>,
    arena: &mut PrivateArena,
    family: &'static str,
    launch_field: &'static str,
    controls: &[(usize, Control)],
    launch: Launcher,
) -> ProbeResult<Vec<serde_json::Value>> {
    // No knob/private call can silently gain public readback/replay authority.
    // The public diagnostic owner must select this original policy at creation.
    if !arena.public_owned_zero_audit_enabled()
        || arena.buffer_count() != 0
        || !std::ptr::eq(client.info(), arena.client()?.info())
    {
        return Err("public CubeK expert requires original empty public-audit arena".into());
    }
    let mut reports = Vec::new();
    for down in [false, true] {
        for &(count, control) in controls {
            let checked = one(
                client,
                arena,
                count,
                down,
                control,
                family,
                launch_field,
                launch,
            );
            // Real reset/drain/erasure precedes every returned report, including
            // a failed numerical check. Outer explicit close remains required.
            arena.reset()?;
            reports.push(checked?);
        }
    }
    Ok(reports)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn read_windows_retain_actual_owner_offsets_and_reject_overflow() {
        let root = Handle::new(cubecl_common::stream_id::StreamId::current(), 4096)
            .offset_start(256)
            .offset_end(512);
        let view = read_window(&root, 256, 512).unwrap();
        assert_eq!(view.size(), root.size());
        assert_eq!(view.offset_start, Some(512));
        assert_eq!(view.offset_end, Some(3072));
        assert_eq!(view.size_in_used(), 512);
        let whole = read_window(&root, 0, 3328).unwrap();
        assert_eq!(whole.offset_start, root.offset_start);
        assert_eq!(whole.offset_end, root.offset_end);
        for (start, len) in [(0, 0), (1, 4), (0, 3), (3328, 4), (usize::MAX, 4)] {
            assert!(read_window(&root, start, len).is_err());
        }
    }
    #[test]
    fn authentic_public_groups_cover_high_sorted_disjoint_routes_and_original_status() {
        for down in [false, true] {
            for count in [32, 33, 64, 256, 257, 287, 288, 383, 384] {
                let f = fixture(count, down, Control::Finite).unwrap();
                assert_eq!(abi::validate(&f.map, TOTAL).unwrap(), 11);
                assert!(f.routes.windows(2).all(|v| v[0] / 10 < v[1] / 10));
                assert!(f.routes.windows(2).any(|v| v[1] - v[0] != 10));
                assert!(f.input.iter().all(|v| v.is_finite()));
                assert!(f.weight.iter().all(|v| v.is_finite()));
                assert!(
                    f.weight[2 * f.columns..3 * f.columns]
                        .iter()
                        .all(|&v| v > 0.0 && v < f16::MIN_POSITIVE.to_f32())
                );
            }
        }
    }

    #[test]
    fn wide_prefix_and_original_tail_faults_mutate_only_the_selected_authentic_member() {
        let good = fixture(257, false, Control::Finite).unwrap();
        let offset = good.map[1028 + good.group] as usize;
        for control in [Control::InvalidRoutePrefix, Control::InvalidRoute] {
            let bad = fixture(257, false, control).unwrap();
            let member = control.bad_member(257).unwrap();
            assert_eq!(
                member,
                if control == Control::InvalidRoute {
                    256
                } else {
                    0
                }
            );
            assert!(abi::validate(&bad.map, TOTAL).is_err());
            let changed: Vec<_> = good
                .map
                .iter()
                .zip(&*bad.map)
                .enumerate()
                .filter_map(|(i, (a, b))| (a != b).then_some(i))
                .collect();
            assert_eq!(changed, [1540 + offset + member]);
            assert_eq!(bad.map[changed[0]], (TOTAL * 10) as u32);
            assert_eq!(*good.routes, *bad.routes);
            assert_eq!(*good.status, *bad.status);
        }
        for down in [false, true] {
            for control in [Control::NonfiniteFirst, Control::NonfiniteOther] {
                let bad = fixture(257, down, control).unwrap();
                assert!(abi::validate(&bad.map, TOTAL).is_ok());
                let row = control.bad_feature().unwrap();
                assert!(bad.weight[row * bad.columns].is_nan());
                assert_eq!(bad.weight.iter().filter(|v| v.is_nan()).count(), 1);
                assert_eq!(
                    bad.weight_image[(PREFIX + row * bad.columns) / 2] & 0xffff,
                    0x7e00
                );
                assert_eq!(bad.routes.len(), 257);
                assert_eq!(control.bad_member(257), None);
            }
        }
    }
}
