//! One additive fixed-public edge proof for the native half conversion path.
//! The retained software helpers and an independent host half implementation
//! are exact finite-value oracles. NaN payloads are not part of publication:
//! scale NaNs are displayed as zero, and a nonfinite eight-value group is zero.
use super::*;
use crate::{
    private_arena::{BufferId, PrivateArena},
    resident_util as u,
};
use zeroize::Zeroizing;

type Checked<T> = std::result::Result<T, Box<dyn std::error::Error>>;
const RECORD_WORDS: usize = 9;
const RESULT_WORDS: usize = 12;
const WITNESS_WORDS_PER_WG: usize = 4 * 128;
const MAX_RECORDS: usize = 200_000;
const MAX_OWNED_BYTES: usize = 512 * 1024 * 1024;
const SCALE_RECORDS: usize = 65_536;
const MIDPOINT_INTERVALS: usize = 0x7bff;
const CONTROLS: [u32; 4] = [0x3c00_3c00, 0, 0x3f00_0000, 0x3e80_0000];
const WITNESS_BITS: u32 = 0x3f40_0000; // dynamic0.5+0.25, plus dynamic1*0

#[cube(launch_unchecked)]
pub(crate) fn quant_half_edges(
    records: &Array<u32>,
    results: &mut Array<u32>,
    witness: &mut Array<f32>,
    count: u32,
    control_first: u32,
) {
    let count = usize::cast_from(count);
    let controls = usize::cast_from(control_first);
    let groups = (count + 63) / 64;
    // All lanes see the same guard. In particular, the final partial record
    // group still reaches the genuine subgroup CMMA and both shared barriers.
    if count == 0
        || count > 200_000
        || controls != count * 9
        || records.len() != controls + 4
        || results.len() != count * 12
        || witness.len() != groups * 512
        || usize::cast_from(CUBE_POS_X) >= groups
    {
        terminate!();
    }
    let index = usize::cast_from(ABSOLUTE_POS);
    if index < count {
        let first = index * 9;
        let output = index * 12;
        let scale_bits = records[first] & 65535;
        let native_scale =
            crate::strata_matrix::quant_cohort::decoder::native_half_to_float(scale_bits);
        let software_scale = crate::strata_matrix::quant_cohort::decoder::half_to_float(scale_bits);
        let mut native_pairs = Vector::<u32, Const<4>>::new(0);
        let mut software_pairs = Vector::<u32, Const<4>>::new(0);
        let mut native_good = true;
        let mut software_good = true;
        #[unroll]
        for pair in 0..4usize {
            let coefficient_x = f32::reinterpret(records[first + 1 + pair * 2]);
            let coefficient_y = f32::reinterpret(records[first + 2 + pair * 2]);
            let nx = native_scale * coefficient_x;
            let ny = native_scale * coefficient_y;
            let sx = software_scale * coefficient_x;
            let sy = software_scale * coefficient_y;
            let np = crate::strata_matrix::quant_cohort::decoder::native_pack_pair(nx, ny);
            let hx = crate::strata_matrix::quant_cohort::decoder::half_bits(sx);
            let hy = crate::strata_matrix::quant_cohort::decoder::half_bits(sy);
            native_pairs[pair] = np;
            software_pairs[pair] = hx | (hy << 16);
            native_good = native_good
                && nx.abs() <= f32::MAX
                && ny.abs() <= f32::MAX
                && (np & 31744) != 31744
                && ((np >> 16) & 31744) != 31744;
            software_good = software_good
                && sx.abs() <= f32::MAX
                && sy.abs() <= f32::MAX
                && (hx & 31744) != 31744
                && (hy & 31744) != 31744;
        }
        // Do not require internal signaling/quiet NaN payload preservation.
        // These are the only published scale bits; original bad-group
        // publication below also removes every unobservable NaN half payload.
        let mut ns = u32::reinterpret(native_scale);
        let mut ss = u32::reinterpret(software_scale);
        if (ns & 2147483647) > 2139095040 {
            ns = 0;
        }
        if (ss & 2147483647) > 2139095040 {
            ss = 0;
        }
        results[output] = ns;
        results[output + 6] = ss;
        results[output + 5] = select(native_good, 1u32, 0u32);
        results[output + 11] = select(software_good, 1u32, 0u32);
        #[unroll]
        for pair in 0..4usize {
            results[output + 1 + pair] = select(native_good, native_pairs[pair], 0u32);
            results[output + 7 + pair] = select(software_good, software_pairs[pair], 0u32);
        }
    }

    // This entry genuinely takes the same cooperative SG16/WG64 native HAL
    // path as the production producer. Dynamic original packed half controls
    // avoid a different scalar-F32-to-F16 conversion or a constant-folded CMMA.
    let sg = usize::cast_from(PLANE_POS);
    let lane = usize::cast_from(UNIT_POS_PLANE);
    let av = Vector::<f16, Const<2>>::reinterpret(records[controls]);
    let bv = Vector::<f16, Const<2>>::reinterpret(records[controls + 1]);
    let seed = f32::reinterpret(records[controls + 2]) + f32::reinterpret(records[controls + 3]);
    let mut a = SharedMemory::<f16>::new(4usize * 128);
    let mut b = SharedMemory::<f16>::new(4usize * 256);
    #[unroll]
    for round in 0..8usize {
        a[sg * 128 + lane + round * 16] = av[0];
    }
    #[unroll]
    for round in 0..16usize {
        b[sg * 256 + lane + round * 16] = bv[0];
    }
    sync_plane();
    let am = unsafe {
        cmma::Matrix::<f16>::uninitialized(
            cmma::MatrixIdent::A,
            8usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
        )
    };
    let bm = unsafe {
        cmma::Matrix::<f16>::uninitialized(
            cmma::MatrixIdent::B,
            8usize,
            16usize,
            16usize,
            cmma::MatrixLayout::ColMajor,
        )
    };
    let acc = cmma::Matrix::<f32>::from_value(
        cmma::MatrixIdent::Accumulator,
        8usize,
        16usize,
        16usize,
        cmma::MatrixLayout::Undefined,
        seed,
    );
    cmma::load(&am, &a.to_slice().slice(sg * 128, a.len()), 16);
    cmma::load(&bm, &b.to_slice().slice(sg * 256, b.len()), 16);
    cmma::execute::<f16, f16, f32, f32>(&am, &bm, &acc, &acc);
    let at = (usize::cast_from(CUBE_POS_X) * 4 + sg) * 128;
    cmma::store(
        &mut witness.slice_mut(at, witness.len()),
        &acc,
        16,
        cmma::MatrixLayout::RowMajor,
    );
    sync_plane();
}

/// The exact generated entry definition is also compiled by the separate
/// eight-layout native-conversion proof. This does not extend old populations.
pub(crate) fn definition() -> KernelDefinition {
    let mut builder = KernelBuilder::default();
    AddressType::U32.register(&mut builder.scope);
    let records = builder.input_array(u32::as_type(&builder.scope));
    let results = builder.output_array(u32::as_type(&builder.scope));
    let witness = builder.output_array(f32::as_type(&builder.scope));
    let count = builder.scalar(u32::as_type(&builder.scope).storage_type());
    let control_first = builder.scalar(u32::as_type(&builder.scope).storage_type());
    quant_half_edges::expand(
        &mut builder.scope,
        records.into(),
        results.into(),
        witness.into(),
        count.into(),
        control_first.into(),
    );
    builder.build(KernelSettings::default().cube_dim(CubeDim::new_1d(64)))
}

fn append_record(records: &mut Vec<u32>, scale: u16, values: [u32; 8]) {
    records.push(u32::from(scale));
    records.extend(values);
}

struct Corpus {
    records: Zeroizing<Vec<u32>>,
    count: usize,
    midpoint_values: usize,
    explicit_edges: usize,
    poisoned_lanes: usize,
}
fn corpus() -> Checked<Corpus> {
    let mut records = Zeroizing::new(Vec::new());
    records.try_reserve_exact(MAX_RECORDS * RECORD_WORDS + CONTROLS.len())?;
    // Every half encoding is decoded, including both zeros, every subnormal,
    // both infinities and every quiet/signaling NaN bit pattern. The decoded
    // value really participates in the same F32 multiply then all-eight gate.
    let coefficients = [-1.0f32, 1.0, -0.5, 0.5, -0.0, 0.0, -1.5, 1.5].map(f32::to_bits);
    for scale in 0u32..=u16::MAX as u32 {
        append_record(&mut records, scale as u16, coefficients);
    }
    // Each boundary between adjacent nonnegative finite half encodings is
    // exactly representable in F32. Test the midpoint and its immediate F32
    // neighbours for both signs. Scale1 makes these exact converter inputs.
    let mut points = Zeroizing::new(Vec::new());
    points.try_reserve_exact(MIDPOINT_INTERVALS * 6 + 8)?;
    for low in 0u16..0x7bff {
        let a = f16::from_bits(low).to_f32();
        let b = f16::from_bits(low + 1).to_f32();
        let midpoint = ((a + b) * 0.5).to_bits();
        for positive in [midpoint - 1, midpoint, midpoint + 1] {
            points.push(positive);
            points.push(positive | 0x8000_0000);
        }
    }
    let midpoint_values = points.len();
    while points.len() % 8 != 0 {
        points.push(0);
    }
    for values in points.chunks_exact(8) {
        append_record(&mut records, 0x3c00, values.try_into()?);
    }
    drop(points);
    // Independent explicit cases cover F32 subnormals/normal boundaries and
    // the finite-half overflow midpoint, which has no finite upper half code.
    let edges = [
        0x0000_0000u32,
        0x0000_0001,
        0x007f_ffff,
        0x0080_0000,
        0x32ff_ffff,
        0x3300_0000,
        0x3300_0001,
        0x337f_ffff,
        0x3380_0000,
        0x3380_0001,
        0x387f_bfff,
        0x387f_c000,
        0x387f_c001,
        0x387f_ffff,
        0x3880_0000,
        0x3880_0001,
        0x477f_e000,
        0x477f_efff,
        0x477f_f000,
        0x477f_f001,
        0x4780_0000,
        0x7f7f_ffff,
        0x7f80_0000,
        0x7f80_0001,
        0x7fc0_1234,
    ];
    let mut explicit_edges = 0;
    for positive in edges {
        for value in [positive, positive | 0x8000_0000] {
            append_record(&mut records, 0x3c00, [value; 8]);
            explicit_edges += 1;
        }
    }
    // Every lane independently poisons its complete original-eight group.
    // These include finite F32 values whose rounded half becomes infinity.
    let poisons = [
        0x7f80_0000u32,
        0xff80_0000,
        0x7f80_0001,
        0x7fc0_1234,
        0xffc0_1234,
        0x477f_f000,
        0xc77f_f000,
        0x7f7f_ffff,
        0xff7f_ffff,
    ];
    let mut poisoned_lanes = 0;
    for lane in 0..8 {
        for poison in poisons {
            let mut values = [0x3f00_0000; 8];
            values[lane] = poison;
            append_record(&mut records, 0x3c00, values);
            poisoned_lanes += 1;
        }
    }
    let count = records.len() / RECORD_WORDS;
    if records.len() % RECORD_WORDS != 0
        || count > MAX_RECORDS
        || midpoint_values != MIDPOINT_INTERVALS * 6
        || poisoned_lanes != poisons.len() * 8
    {
        return Err("native half edge bounded exhaustive corpus".into());
    }
    records.extend(CONTROLS);
    Ok(Corpus {
        records,
        count,
        midpoint_values,
        explicit_edges,
        poisoned_lanes,
    })
}

fn sanitize(arena: &PrivateArena) -> Checked<()> {
    let receipt = cubecl::future::block_on(arena.client()?.sanitize_transient_io_audited())?;
    let bytes = receipt
        .upload_bytes
        .checked_add(receipt.uniform_bytes)
        .and_then(|v| v.checked_add(receipt.readback_bytes))
        .ok_or("native half edge IO audit overflow")?;
    if bytes == 0 || receipt.inspected_bytes != bytes || receipt.inspected_buffers == 0 {
        return Err("native half edge complete transient IO erasure".into());
    }
    Ok(())
}
fn read_owner(arena: &PrivateArena, id: BufferId) -> Checked<Zeroizing<Vec<u32>>> {
    let layout = arena.layout(id)?;
    let owner = arena.handle(id)?.clone();
    if owner.offset_start.unwrap_or(0) != 0
        || owner.offset_end.unwrap_or(0) != 0
        || owner.size_in_used() != layout.owned_bytes as u64
        || owner.size() != layout.owned_bytes as u64
        || layout.owned_bytes > MAX_OWNED_BYTES
    {
        return Err("native half edge complete owning span".into());
    }
    let data = u::read_handle_prefix_u32(arena, owner, layout.owned_words)?;
    sanitize(arena)?;
    Ok(data)
}
fn erase_owner(
    client: &ComputeClient<WgpuRuntime>,
    arena: &PrivateArena,
    id: BufferId,
) -> Checked<()> {
    let layout = arena.layout(id)?;
    let owner = arena.handle(id)?.clone();
    // The same original full-span eraser uses the actual arena-owned extent.
    // It is followed by a complete owning read, including allocator padding.
    crate::private_arena::erase_full_span::launch::<WgpuRuntime>(
        client,
        CubeCount::Static(layout.workgroups_x, layout.workgroups_y, 1),
        CubeDim::new_1d(64),
        unsafe { ArrayArg::from_raw_parts(owner, layout.owned_words) },
    );
    cubecl::future::block_on(client.sync()).map_err(u::fault)?;
    if read_owner(arena, id)?.iter().any(|v| *v != 0) {
        return Err("native half edge full owner not zero after actual erasure".into());
    }
    Ok(())
}
fn check_record(record: &[u32], actual: &[u32], index: usize) -> Checked<bool> {
    let scale = f16::from_bits(record[0] as u16).to_f32();
    let scale_bits = if scale.is_nan() { 0 } else { scale.to_bits() };
    let mut pairs = [0u32; 4];
    let mut good = true;
    for (pair, values) in record[1..].chunks_exact(2).enumerate() {
        let x = scale * f32::from_bits(values[0]);
        let y = scale * f32::from_bits(values[1]);
        let hx = f16::from_f32(x);
        let hy = f16::from_f32(y);
        good &= x.is_finite() && y.is_finite() && hx.is_finite() && hy.is_finite();
        pairs[pair] = u32::from(hx.to_bits()) | (u32::from(hy.to_bits()) << 16);
    }
    if !good {
        pairs.fill(0);
    }
    let expected = [
        scale_bits,
        pairs[0],
        pairs[1],
        pairs[2],
        pairs[3],
        u32::from(good),
    ];
    // Exact finite bits and final group publication, never raw NaN payloads.
    for version in 0..2 {
        for field in 0..6 {
            let got = actual[version * 6 + field];
            let want = expected[field];
            if got != want {
                return Err(format!("native half edge exact publication mismatch public_record={index} implementation={version} field={field} expected={want:08x} actual={got:08x}").into());
            }
        }
    }
    Ok(good)
}
fn one(
    client: &ComputeClient<WgpuRuntime>,
    arena: &mut PrivateArena,
) -> Checked<serde_json::Value> {
    let corpus = corpus()?;
    let groups = corpus.count.div_ceil(64);
    let witness_words = groups * WITNESS_WORDS_PER_WG;
    let control_first = corpus.count * RECORD_WORDS;
    let logical_words = [
        corpus.records.len(),
        corpus.count * RESULT_WORDS,
        witness_words,
    ];
    let planned_owned_bytes = logical_words.iter().try_fold(0usize, |sum, words| {
        sum.checked_add(crate::private_arena::BufferLayout::new(*words)?.owned_bytes)
            .ok_or(crate::private_arena::ArenaError::InvalidLayout)
    })?;
    if planned_owned_bytes > MAX_OWNED_BYTES || groups > 65535 || control_first > u32::MAX as usize
    {
        return Err("native half edge preallocation total owned budget".into());
    }
    let ids = [
        arena.upload_u32(&corpus.records)?,
        // Even an entirely bad group must perform all twelve publications;
        // zero-initialized output could otherwise conceal an omitted record.
        arena.upload_u32(&Zeroizing::new(vec![
            0xa5c3_5e71u32;
            corpus.count * RESULT_WORDS
        ]))?,
        // A skipped or partially omitted cooperative store cannot pass as a
        // zero-initialized witness. Every logical position begins as a NaN.
        arena.upload_u32(&Zeroizing::new(vec![0x7fc0_4321u32; witness_words]))?,
    ];
    let owned_bytes = ids.iter().try_fold(0usize, |sum, id| {
        sum.checked_add(arena.layout(*id)?.owned_bytes)
            .ok_or(crate::private_arena::ArenaError::InvalidLayout)
    })?;
    if owned_bytes != planned_owned_bytes {
        return Err("native half edge total bounded owning allocation".into());
    }
    // SAFETY: all three arrays are exact logical views in this same retained
    // arena; the guard checks their true lengths. Source and padding remain
    // owned until the completed kernel and complete erasure have been audited.
    unsafe {
        quant_half_edges::launch_unchecked::<WgpuRuntime>(
            client,
            CubeCount::Static(groups as u32, 1, 1),
            CubeDim::new_1d(64),
            u::arg(arena, ids[0])?,
            u::arg(arena, ids[1])?,
            u::arg(arena, ids[2])?,
            corpus.count as u32,
            control_first as u32,
        );
    }
    cubecl::future::block_on(client.sync()).map_err(u::fault)?;
    let results = read_owner(arena, ids[1])?;
    let mut good_records = 0;
    for (index, record) in corpus.records[..control_first]
        .chunks_exact(RECORD_WORDS)
        .enumerate()
    {
        good_records += usize::from(check_record(
            record,
            &results[index * RESULT_WORDS..(index + 1) * RESULT_WORDS],
            index,
        )?);
    }
    if results[corpus.count * RESULT_WORDS..]
        .iter()
        .any(|v| *v != 0)
    {
        return Err("native half edge output owner padding changed".into());
    }
    drop(results);
    let source = read_owner(arena, ids[0])?;
    if source[..corpus.records.len()] != corpus.records[..]
        || source[corpus.records.len()..].iter().any(|v| *v != 0)
    {
        return Err("native half edge complete source owner changed".into());
    }
    drop(source);
    let witness = read_owner(arena, ids[2])?;
    if witness[..witness_words].iter().any(|v| *v != WITNESS_BITS)
        || witness[witness_words..].iter().any(|v| *v != 0)
    {
        return Err("native half edge every actual subgroup CMMA witness".into());
    }
    drop(witness);
    for id in ids {
        erase_owner(client, arena, id)?;
    }
    Ok(serde_json::json!({
        "family":"quant-half-native-exact-edges",
        "records":corpus.count,
        "scale_encodings":SCALE_RECORDS,
        "finite_half_midpoint_intervals":MIDPOINT_INTERVALS,
        "signed_midpoint_and_adjacent_f32_values":corpus.midpoint_values,
        "explicit_signed_edge_records":corpus.explicit_edges,
        "individually_poisoned_lane_records":corpus.poisoned_lanes,
        "good_records":good_records,
        "bad_records":corpus.count-good_records,
        "actual_native_and_original_software_helpers":true,
        "independent_host_half_reference":true,
        "finite_bits_signed_zero_subnormals_exact":true,
        "all_eight_good_flags_and_zero_publication_checked":true,
        "nan_scale_publication_zero_and_internal_nan_payload_unobserved":true,
        "actual_cooperative_sg16_wg64_product":true,
        "dynamic_zero_product_preserves_nonzero_seed":true,
        "dynamic_cmma_witness_words_checked":witness_words,
        "full_source_output_witness_owners_and_padding_checked":true,
        "all_full_owners_erased_and_zero_verified":true,
        "total_owned_bytes":owned_bytes,
        "bit_mismatches":0
    }))
}

pub(crate) fn cases(
    client: &ComputeClient<WgpuRuntime>,
    arena: &mut PrivateArena,
) -> Checked<Vec<serde_json::Value>> {
    if !arena.public_owned_zero_audit_enabled() || arena.buffer_count() != 0 {
        return Err("native half edge empty fixed-public audited arena required".into());
    }
    let result = one(client, arena);
    // Even a numerical failure must leave normal whole-owner erasure to the
    // original arena. No failed case can emit a successful public receipt.
    arena.reset()?;
    Ok(vec![result?])
}
pub(crate) use cases as public_numeric;
