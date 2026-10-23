//! Additive paired original-rank versus genuine ordered-PART producer controls.
//! Keeps independent F64, complete original owners and original route mutations.
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
    let original = [32, 33, 64, 256, 257, 287, 288, 383, 384].contains(&count);
    let medium = cfg!(feature = "strata-native-expert-medium")
        && [128, 129, 159, 160, 191, 192, 255].contains(&count);
    let narrow =
        cfg!(feature = "strata-narrow-global") && [1, 3, 7, 8, 9, 15, 16, 17, 31].contains(&count);
    let macro64 =
        cfg!(feature = "strata-expert-macro64") && [64, 65, 127, 128, 129, 255].contains(&count);
    if !original && !medium && !narrow && !macro64 {
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
pub(super) fn scan(
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

pub(super) fn one(
    client: &ComputeClient<WgpuRuntime>,
    arena: &mut PrivateArena,
    count: usize,
    down: bool,
    control: Control,
    family: &'static str,
    launch_field: &'static str,
    launch: Launcher,
    ordered: bool,
) -> ProbeResult<serde_json::Value> {
    let f = fixture(count, down, control)?;
    let mut image = f.input_image.clone();
    #[cfg(feature = "strata-narrow-global")]
    if count < 32 {
        let tile = if count <= 8 { 8 } else { 16 };
        image.resize(count.div_ceil(tile) * tile * f.columns / 2, 0);
    }
    let input = arena.upload_u32(&image)?;
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
    #[cfg(feature = "strata-expert-cohorts")]
    let cohort = family.contains("cohort");
    #[cfg(not(feature = "strata-expert-cohorts"))]
    let cohort = false;
    let actual_launch = if cohort {
        #[cfg(feature = "strata-expert-cohorts")]
        {
            // Reuse THIS original fixture and unchanged independent F64 oracle.
            // Only repack its authentic rounded weight image into the selected
            // complete GU+DOWN image layout; every new owned byte is audited.
            let mut packed = Zeroizing::new(vec![
                0u32;
                crate::strata_prompt::experts::cohorts::IMAGE_BYTES
                    as usize
                    / 4
            ]);
            let offset = if down { 1280 * 2560 / 2 } else { 0 };
            packed[offset..offset + f.rows * f.columns / 2]
                .copy_from_slice(&f.weight_image[PREFIX / 2..]);
            let packed_id = arena.upload_u32(&packed)?;
            let mut selected = handles.clone();
            selected[1] = u::handle(arena, packed_id)?;
            let mut words = Zeroizing::new([0u32; 48]);
            words[0] = TOTAL as u32;
            words[1] = f.columns as u32;
            words[2] = f.rows as u32;
            words[4] = f.group as u32;
            words[5] = if down { 2 } else { 0 };
            words[6] = 100;
            words[7] = 100;
            words[8] = 300;
            words[9] = 1;
            words[10] = count as u32;
            words[11] = if count <= 8 { 8 } else { 16 };
            words[16] = f.group as u32;
            words[17] = count as u32;
            words[19] = 8;
            #[cfg(feature="strata-small-direct-publish")]
            let small_cohort=family=="small-direct-cohort" && ordered;
            #[cfg(not(feature="strata-small-direct-publish"))]
            let small_cohort=false;
            if small_cohort {
                #[cfg(feature="strata-small-direct-publish")]
                crate::strata_matrix::cohorts::small_direct::launch(client,selected,&words,down,words[11]as usize)?;
            } else if ordered && down {
                crate::strata_matrix::cohorts::launch_ordered(
                    client,
                    selected,
                    &words,
                    down,
                    words[11] as usize,
                )?;
            } else {
                crate::strata_matrix::cohorts::launch(
                    client,
                    selected,
                    &words,
                    down,
                    words[11] as usize,
                )?;
            }
            scan(arena, packed_id, |at, bits| {
                if bits != packed.get(at).copied().unwrap_or(0) {
                    return Err("public cohort original repacked image owner changed".into());
                }
                Ok(())
            })?;
            if family=="small-direct-cohort" {ordered}else{true}
        }
        #[cfg(not(feature = "strata-expert-cohorts"))]
        {
            unreachable!()
        }
    } else {
        launch(
            client, handles, count, f.columns, f.rows, PREFIX, f.group, down,
        )?
    };
    let prefix = f.map[1028 + f.group] as usize;
    let wide_prefix = if !cohort
        && count
            >= if cfg!(feature = "strata-native-expert-medium") {
                128
            } else {
                256
            } {
        count / 128 * 128
    } else {
        0
    };
    // Only the separately named additive GU/direct controls select this
    // original raw-publication contract; existing cases retain their policy.
    let small_selected=family.starts_with("small-direct-") && ordered;
    // New global-register controls deliberately use the same complete8 direct
    // tail publication as the production path. All old families keep policy.
    let global_direct_tail=matches!(family,"global-register-direct-gu"|"global-register-ordered-down");
    let direct_gu = family == "shared-reuse-direct-gu" || family=="global-register-direct-gu" || (small_selected && !down);
    let direct_member = |member: usize| {
        (ordered || direct_gu)
            && if small_selected || global_direct_tail {member/8*8+8<=count} else {member < wide_prefix}
            && control
                .bad_member(count)
                .is_none_or(|bad| bad / 8 != member / 8)
    };
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
        let member = if ordered && down {
            (at / f.rows)
                .checked_sub(prefix)
                .filter(|&m| m < count && control.bad_member(count) != Some(m))
                .unwrap_or(usize::MAX)
        } else if down {
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
            if direct_member(member) && control.bad_feature() == Some(row) {
                if !value.is_nan() {
                    return Err("ordered direct pending nonfinite result".into());
                }
            } else {
                mismatches +=
                    usize::from(!value.is_finite() || error > 1e-4 + 3e-4 * reference.abs());
                max_abs = max_abs.max(error);
                if (control.bad_feature() == Some(row) || control.bad_member(count) == Some(member))
                    && bits != 0
                {
                    return Err("public CubeK expert fault replacement".into());
                }
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
            expected_status[abi::ROUTES + route] = if direct_member(member) {
                2
            } else {
                u32::from(control != Control::NonfiniteFirst && f.status[route] == 1)
            };
        }
    }
    let status_words = scan(arena, status, |at, bits| {
        if bits != expected_status.get(at).copied().unwrap_or(0) {
            return Err("public CubeK expert full status owner".into());
        }
        Ok(())
    })?;
    let expected_fault = if control.bad_member(count).is_some()
        || (control.bad_feature().is_some() && (0..count).any(|m| !direct_member(m)))
    {
        128
    } else {
        0
    };
    scan(arena, fault, |at, bits| {
        if bits != if at == 0 { expected_fault } else { 0 } {
            return Err("public CubeK expert full fault owner".into());
        }
        Ok(())
    })?;
    for (id, image) in [(input, &*image), (weight, &*f.weight_image), (map, &*f.map)] {
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
    report["expert_ordered_part"] = (ordered && down).into();
    if family.starts_with("small-direct-") {
        report["actual_small_direct_publication"]=small_selected.into();
        report["raw_gu_pending_finite"]= (small_selected && !down).into();
        report["complete_original8_fragment_only"]=true.into();
    }
    report["raw_pending2_checked_before_consumer"] = (down && ordered && if small_selected {(0..count).any(direct_member)} else {wide_prefix > 0}).into();
    report["paired_original_rank_case_retained"] = true.into();
    if cohort {
        report["original_fixture_prefix_repacked_half_elements"] = PREFIX.into();
        report["real_weight_offset_half_elements"] = (if down { 1280 * 2560 } else { 0 }).into();
        report["actual_cohort_image_weight_half_offset"] =
            (if down { 1280 * 2560 } else { 0 }).into();
    }
    Ok(report)
}

fn original(
    client: &ComputeClient<WgpuRuntime>,
    h: [Handle; 6],
    count: usize,
    columns: usize,
    rows: usize,
    weight_first: usize,
    group: usize,
    down: bool,
) -> ProbeResult<bool> {
    crate::strata_matrix::native::run_expert(
        client,
        h,
        count,
        columns,
        rows,
        weight_first,
        group,
        down,
    )?;
    Ok(false)
}
fn selected(
    client: &ComputeClient<WgpuRuntime>,
    h: [Handle; 6],
    count: usize,
    columns: usize,
    rows: usize,
    weight_first: usize,
    group: usize,
    down: bool,
) -> ProbeResult<bool> {
    crate::strata_matrix::native::ordered::run_expert(
        client,
        h,
        count,
        columns,
        rows,
        weight_first,
        group,
        down,
    )?;
    Ok(true)
}
pub(crate) fn cases(
    client: &ComputeClient<WgpuRuntime>,
    arena: &mut PrivateArena,
) -> ProbeResult<Vec<serde_json::Value>> {
    if !arena.public_owned_zero_audit_enabled()
        || arena.buffer_count() != 0
        || !std::ptr::eq(client.info(), arena.client()?.info())
    {
        return Err("ordered expert requires empty original public audit arena".into());
    }
    use Control::*;
    let mut controls = vec![
        (32, Finite),
        (33, Finite),
        (64, Finite),
        (256, Finite),
        (257, Finite),
        (256, NonfiniteFirst),
        (257, NonfiniteOther),
        (257, InvalidRoute),
        (257, InvalidRoutePrefix),
    ];
    #[cfg(feature = "strata-narrow-global")]
    controls.extend([
        (1, Finite),
        (7, NonfiniteFirst),
        (9, Finite),
        (17, NonfiniteOther),
        (31, InvalidRoute),
    ]);
    #[cfg(feature = "strata-native-expert-medium")]
    controls.extend([
        (128, Finite),
        (129, Finite),
        (129, NonfiniteFirst),
        (255, InvalidRoutePrefix),
    ]);
    #[cfg(feature = "strata-expert-macro64")]
    controls.extend([(65, Finite), (127, NonfiniteOther)]);
    let mut reports = Vec::new();
    for (ordered, launch) in [(false, original as Launcher), (true, selected as Launcher)] {
        for &(count, control) in &controls {
            let result = one(
                client,
                arena,
                count,
                true,
                control,
                "paired-native-ordered-expert",
                "actual_ordered_producer",
                launch,
                ordered,
            );
            arena.reset()?;
            reports.push(result?);
        }
        #[cfg(feature = "strata-expert-cohorts")]
        for (count, control) in [
            (3, Finite),
            (8, NonfiniteFirst),
            (9, NonfiniteOther),
            (17, InvalidRoute),
        ] {
            let result = one(
                client,
                arena,
                count,
                true,
                control,
                "paired-native-ordered-cohort",
                "actual_ordered_producer",
                launch,
                ordered,
            );
            arena.reset()?;
            reports.push(result?);
        }
    }
    Ok(reports)
}
