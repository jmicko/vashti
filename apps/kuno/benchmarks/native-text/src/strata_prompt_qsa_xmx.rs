//! Connected QK/softmax/PV CMMA attention for full prompt batches.
//! Design: MIT Strata fb58e0 qsa_prompt_attn_xmx.dp.cpp v2. Original public
//! scalar/short paths remain intact. Only device-local scratch is introduced;
//! original full query/cache/selection/partial/stats/fault owners are retained.
use cubecl::{prelude::*, server::Handle, wgpu::WgpuRuntime};
use half::f16;
use kuno_text::{Error, Result};

pub(crate) const THREADS: u32 = 128;
pub(crate) const SHARED_BYTES: usize =
    8192 * 2 + 4096 * 2 + 2048 * 2 + (1024 + 768) * 4 + 64 * 4 + 12 * 4;

static LAUNCHES: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
pub(crate) fn launch_count() -> u64 {
    LAUNCHES.load(std::sync::atomic::Ordering::Relaxed)
}
#[cube]
fn code(cache: &Array<u32>, token: usize, offset: usize) -> f32 {
    let b = (cache[token * 264 + offset / 4] >> u32::cast_from((offset % 4) * 8)) & 255;
    let mut c = i32::cast_from(b);
    if b >= 128 {
        c -= 256;
    }
    f32::cast_from(c)
}
#[cube]
fn scale(cache: &Array<u32>, token: usize, offset: usize) -> f32 {
    let pair = cache[token * 264 + 256 + offset / 128];
    let bits = (pair >> u32::cast_from(((offset / 64) % 2) * 16)) & 65535;
    let exponent = (bits >> 10) & 31;
    let fraction = bits & 1023;
    let sign = (bits & 32768) << 16;
    let mut result = sign | ((exponent + 112) << 23) | (fraction << 13);
    if exponent == 31 {
        result = sign | 2139095040u32 | (fraction << 13);
    }
    let mut value = f32::reinterpret(result);
    if exponent == 0 {
        value = f32::cast_from(fraction) * 5.960464477539063e-8;
        if sign != 0 {
            value = -value;
        }
    }
    value
}

/// Exact IEEE binary16 RTE, returned as binary32 bits. Integer operations
/// make the high-part rounding observable BEFORE residual subtraction; an
/// over-precise F16 register cannot turn the residual into x-x.
#[cube]
fn rounded_half_bits(bits: u32) -> u32 {
    let sign = bits & 0x80000000u32;
    let mag = bits & 0x7fffffffu32;
    let e = mag >> 23;
    let fraction = mag & 0x7fffffu32;
    let mut rounded = mag;
    if mag >= 0x7f800000u32 {
        rounded = mag;
        if fraction != 0 {
            rounded = (mag | 0x400000u32) & 0x7fffe000u32;
        }
    } else if mag >= 0x477ff000u32 {
        rounded = 0x7f800000u32;
    } else if mag >= 0x38800000u32 {
        rounded = (mag + 0xfffu32 + ((mag >> 13) & 1)) & 0xffffe000u32;
    } else {
        rounded = 0;
        if e >= 102 {
            let shift = 126 - e;
            let mantissa = 0x800000u32 | fraction;
            let mut q = mantissa >> shift;
            let remainder = mantissa & ((1u32 << shift) - 1);
            let halfway = 1u32 << (shift - 1);
            if remainder > halfway || (remainder == halfway && (q & 1) != 0) {
                q += 1;
            }
            if q > 0 {
                let mut n = q;
                let mut leading = 0u32;
                while n > 1 {
                    n >>= 1;
                    leading += 1;
                }
                rounded = ((leading + 103) << 23) | ((q - (1u32 << leading)) << (23 - leading));
            }
        }
    }
    sign | rounded
}
#[cube]
fn stage_split_raw(hi: &mut SharedMemory<f16>, lo: &mut SharedMemory<f16>, i: usize, x: f32) {
    let h = f16::cast_from(x);
    hi[i] = h;
    lo[i] = f16::cast_from(x - f32::cast_from(h));
}
#[cube]
fn rounded_high(x: f32) -> f32 {
    f32::reinterpret(rounded_half_bits(u32::reinterpret(x)))
}
#[cube]
fn stage_split(hi: &mut SharedMemory<f16>, lo: &mut SharedMemory<f16>, i: usize, x: f32) {
    let rounded = rounded_high(x);
    hi[i] = f16::cast_from(rounded);
    lo[i] = f16::cast_from(x - rounded);
}
#[path = "strata_qsa_xmx_stage_probe.rs"]
mod stage_probe;
pub(crate) use stage_probe::definition as stage_definition;
pub(crate) use stage_probe::run as public_stage_probe;

#[cube(launch_unchecked)]
fn attention_xmx(
    query: &Array<f32>,
    cache: &Array<u32>,
    selected: &Array<u32>,
    partial: &mut Array<f32>,
    stats: &mut Array<f32>,
    fault: &mut Array<Atomic<u32>>,
    count: usize,
    start: usize,
    capacity: usize,
    query_offset: usize,
) {
    let tid = usize::cast_from(UNIT_POS);
    let lane = usize::cast_from(UNIT_POS_PLANE);
    let sg = usize::cast_from(PLANE_POS);
    let kvh = usize::cast_from(CUBE_POS_Y);
    let t = usize::cast_from(CUBE_POS_Z);
    let qr = t + query_offset;
    let visible = start + qr + 1;
    let cells = visible.min(2051);
    if t < count && kvh < 2 {
        // Keep all query heads across chunks. Stage one64-dimension scale
        // group at a time, retaining full64-cell matrix products in registers.
        let mut qh = SharedMemory::<f16>::new(4096usize);
        let mut ql = SharedMemory::<f16>::new(4096usize);
        let mut kv = SharedMemory::<Vector<f16, Const<4>>>::new(1024usize);
        let mut ph = SharedMemory::<Vector<f16, Const<4>>>::new(256usize);
        let mut pl = SharedMemory::<Vector<f16, Const<4>>>::new(256usize);
        let mut tile = SharedMemory::<f32>::new(1024usize);
        let mut prob = SharedMemory::<f32>::new(768usize);
        let mut rows = SharedMemory::<u32>::new(64usize);
        // Pad this final scalar stage to16-byte size, like every other stage.
        // Explicit-layout allocation may visit stages in different orders;
        // no vector alignment hole may exceed the authenticated shared bound.
        let mut reduction = SharedMemory::<f32>::new(12usize);
        let mut qm = 0.0f32;
        let mut i = tid;
        while i < 3072 {
            let x = query[qr * 6144 + kvh * 3072 + i];
            if x.abs() <= f32::MAX {
                qm = max(qm, x.abs());
            } else {
                fault[0].fetch_or(1);
            }
            i += 128;
        }
        qm = plane_max(qm);
        if lane == 0 {
            reduction[sg] = qm;
        }
        sync_cube();
        qm = 0.0;
        for j in 0..8usize {
            qm = max(qm, reduction[j]);
        }
        let mut up = 1.0f32;
        if qm > 0.0 {
            let e = (u32::reinterpret(qm) >> 23) & 255;
            let ue = (268u32 - e).min(254u32);
            up = f32::reinterpret(ue << 23);
        }
        if tid == 0 {
            reduction[8] = (1.0 / up) / 16.0;
        }
        i = tid;
        while i < 4096 {
            let mut x = 0.0f32;
            if i < 3072 {
                let v = query[qr * 6144 + kvh * 3072 + i];
                if v.abs() <= f32::MAX {
                    x = v * up;
                }
            }
            stage_split(&mut qh, &mut ql, i, x);
            i += 128;
        }
        sync_cube();
        let ah = unsafe {
            cmma::Matrix::<f16>::uninitialized(
                cmma::MatrixIdent::A,
                8usize,
                16usize,
                16usize,
                cmma::MatrixLayout::RowMajor,
            )
        };
        let al = unsafe {
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
        let mut chunk = 0usize;
        while chunk * 64 < cells {
            let here = (cells - chunk * 64).min(64);
            let slot = kvh * 33 + chunk;
            if tid < 64 {
                let mut tok = capacity;
                if tid < here {
                    tok = usize::cast_from(selected[qr * 2051 + chunk * 64 + tid]);
                    if tok >= visible || tok >= capacity {
                        fault[0].fetch_or(1);
                        tok = capacity;
                    }
                }
                rows[tid] = u32::cast_from(tok);
            }
            i = tid;
            while i < 768 {
                prob[i] = 0.0;
                i += 128;
            }
            sync_cube();
            for g in 0..4usize {
                i = tid;
                while i < 1024 {
                    let col = i / 16;
                    let d = g * 64 + (i % 16) * 4;
                    let tok = usize::cast_from(rows[col]);
                    let mut packed = Vector::<f16, Const<4>>::new(f16::new(0.0));
                    #[unroll]
                    for j in 0..4usize {
                        let mut x = 0.0f32;
                        if tok < capacity {
                            x = code(cache, tok, kvh * 256 + d + j);
                        }
                        packed[j] = f16::cast_from(x);
                    }
                    kv[i] = packed;
                    i += 128;
                }
                sync_cube();
                let cell = (sg % 4) * 16;
                let head = (sg / 4) * 8;
                let c = cmma::Matrix::<f32>::from_value(
                    cmma::MatrixIdent::Accumulator,
                    8usize,
                    16usize,
                    16usize,
                    cmma::MatrixLayout::Undefined,
                    0.0,
                );
                for kk in 0..4usize {
                    let k = kk * 16;
                    cmma::load(
                        &ah,
                        &qh.to_slice().slice(head * 256 + g * 64 + k, qh.len()),
                        256,
                    );
                    cmma::load(
                        &al,
                        &ql.to_slice().slice(head * 256 + g * 64 + k, ql.len()),
                        256,
                    );
                    cmma::load(&bm, &kv.to_slice().slice((cell * 64 + k) / 4, kv.len()), 64);
                    cmma::execute::<f16, f16, f32, f32>(&ah, &bm, &c, &c);
                    cmma::execute::<f16, f16, f32, f32>(&al, &bm, &c, &c);
                }
                cmma::store(
                    &mut tile.slice_mut(sg * 128, sg * 128 + 128),
                    &c,
                    16,
                    cmma::MatrixLayout::RowMajor,
                );
                sync_cube();
                let tok = usize::cast_from(rows[cell + lane]);
                let mut ks = 0.0f32;
                if tok < capacity {
                    ks = scale(cache, tok, kvh * 256 + g * 64);
                    if !(ks.abs() <= f32::MAX) {
                        fault[0].fetch_or(1);
                        ks = 0.0;
                    }
                }
                i = lane;
                while i < 128 {
                    let h = head + i / 16;
                    if h < 12 {
                        prob[h * 64 + cell + i % 16] += tile[sg * 128 + i] * ks;
                    }
                    i += 16;
                }
                sync_cube();
            }
            let mut h = sg;
            while h < 12 {
                let mut maximum = f32::cast_from(-f32::MAX);
                for j in 0..4usize {
                    let col = lane + j * 16;
                    let mut score = f32::cast_from(-f32::MAX);
                    if col < here && usize::cast_from(rows[col]) < capacity {
                        score = prob[h * 64 + col] * reduction[8];
                        if !(score.abs() <= f32::MAX) {
                            fault[0].fetch_or(1);
                        }
                    }
                    prob[h * 64 + col] = score;
                    maximum = max(maximum, score);
                }
                maximum = plane_max(maximum);
                let mut total = 0.0f32;
                for j in 0..4usize {
                    let col = lane + j * 16;
                    let mut v = 0.0f32;
                    if col < here && usize::cast_from(rows[col]) < capacity {
                        v = (prob[h * 64 + col] - maximum).exp();
                    }
                    prob[h * 64 + col] = v;
                    total += v;
                }
                total = plane_sum(total);
                if lane == 0 {
                    stats[t * 1584 + (slot * 12 + h) * 2] = maximum;
                    stats[t * 1584 + (slot * 12 + h) * 2 + 1] = total;
                }
                h += 8;
            }
            sync_cube();
            let vg = sg % 4;
            let mut vmax = 0.0f32;
            for j in 0..4usize {
                let col = lane + j * 16;
                let tok = usize::cast_from(rows[col]);
                if tok < capacity {
                    let x = scale(cache, tok, 512 + kvh * 256 + vg * 64);
                    if x.abs() <= f32::MAX {
                        vmax = max(vmax, x.abs());
                    } else {
                        fault[0].fetch_or(1);
                    }
                }
            }
            vmax = plane_max(vmax);
            if lane == 0 && sg < 4 {
                reduction[vg] = vmax;
            }
            sync_cube();
            // Stage one full64-cell V scale group, accumulate its complete
            // P.V in registers, then initialize the original partial once.
            for g in 0..4usize {
                i = tid;
                while i < 1024 {
                    let d = g * 64 + i / 16;
                    let col = (i % 16) * 4;
                    let mut packed = Vector::<f16, Const<4>>::new(f16::new(0.0));
                    #[unroll]
                    for j in 0..4usize {
                        let tok = usize::cast_from(rows[col + j]);
                        let mut x = 0.0f32;
                        if tok < capacity {
                            x = code(cache, tok, 512 + kvh * 256 + d);
                        }
                        packed[j] = f16::cast_from(x);
                    }
                    kv[i] = packed;
                    i += 128;
                }
                i = tid;
                while i < 256 {
                    let h = i / 16;
                    let col = (i % 16) * 4;
                    let mut hi = Vector::<f16, Const<4>>::new(f16::new(0.0));
                    let mut lo = Vector::<f16, Const<4>>::new(f16::new(0.0));
                    #[unroll]
                    for j in 0..4usize {
                        let tok = usize::cast_from(rows[col + j]);
                        let mut x = 0.0f32;
                        if h < 12 && tok < capacity && reduction[g] > 0.0 {
                            let vs = scale(cache, tok, 512 + kvh * 256 + g * 64);
                            if vs.abs() <= f32::MAX {
                                x = prob[h * 64 + col + j] * (vs * (16384.0 / reduction[g]));
                            }
                        }
                        let rounded = rounded_high(x);
                        hi[j] = f16::cast_from(rounded);
                        lo[j] = f16::cast_from(x - rounded);
                    }
                    ph[i] = hi;
                    pl[i] = lo;
                    i += 128;
                }
                sync_cube();
                let head = (sg / 4) * 8;
                let dim = (sg % 4) * 16;
                let c = cmma::Matrix::<f32>::from_value(
                    cmma::MatrixIdent::Accumulator,
                    8usize,
                    16usize,
                    16usize,
                    cmma::MatrixLayout::Undefined,
                    0.0,
                );
                for kk in 0..4usize {
                    let k = kk * 16;
                    cmma::load(&ah, &ph.to_slice().slice((head * 64 + k) / 4, ph.len()), 64);
                    cmma::load(&al, &pl.to_slice().slice((head * 64 + k) / 4, pl.len()), 64);
                    cmma::load(&bm, &kv.to_slice().slice((dim * 64 + k) / 4, kv.len()), 64);
                    cmma::execute::<f16, f16, f32, f32>(&ah, &bm, &c, &c);
                    cmma::execute::<f16, f16, f32, f32>(&al, &bm, &c, &c);
                }
                cmma::store(
                    &mut tile.slice_mut(sg * 128, sg * 128 + 128),
                    &c,
                    16,
                    cmma::MatrixLayout::RowMajor,
                );
                sync_cube();
                i = lane;
                while i < 128 {
                    let h = head + i / 16;
                    if h < 12 {
                        let at = t * 202752 + (slot * 12 + h) * 256 + g * 64 + dim + i % 16;
                        let v = tile[sg * 128 + i] * (reduction[g] / 16384.0);
                        partial[at] = v;
                        if !(v.abs() <= f32::MAX) {
                            fault[0].fetch_or(1);
                        }
                    }
                    i += 16;
                }
                sync_cube();
            }
            chunk += 1;
        }
    }
}

pub(crate) fn try_launch(
    arena: &crate::private_arena::PrivateArena,
    h: &[Handle],
    p: [u32; 12],
    grid: [usize; 3],
) -> Result<bool> {
    use cubecl::ir::{ElemType, FloatKind, features::MmaConfig};
    if !(16..=32).contains(&p[0]) {
        return Ok(false);
    }
    let client = arena.client().map_err(crate::resident_util::fault)?;
    let hw = &client.properties().hardware;
    let kind = ElemType::Float(FloatKind::F16);
    if hw.max_shared_memory_size < SHARED_BYTES
        || hw.max_units_per_cube < THREADS
        || !client
            .properties()
            .features
            .matmul
            .cmma
            .contains(&MmaConfig {
                a_type: kind.into(),
                b_type: kind.into(),
                cd_type: ElemType::Float(FloatKind::F32).into(),
                m: 8,
                n: 16,
                k: 16,
            })
    {
        return Ok(false);
    }
    let count = p[0] as usize;
    let start = p[1] as usize;
    let cap = p[2] as usize;
    let offset = p[11] as usize;
    if cap > crate::qsa_gpu::MAX_CONTEXT
        || start
            .checked_add(offset)
            .and_then(|n| n.checked_add(count))
            .is_none_or(|end| end > cap)
        || offset.checked_add(count).is_none_or(|end| end > 4096)
        || grid != [33, 2, count]
        || h.len() != 7
        || h.iter().any(|x| x.stream != h[0].stream)
    {
        return Err(Error::InvalidBufferLength);
    }
    let required = [
        (offset + count) * 6144,
        cap * 264,
        0,
        (offset + count) * 2051,
        count * 202752,
        count * 1584,
        1,
    ];
    for (v, n) in h.iter().zip(required) {
        if v.size_in_used() < (n * 4) as u64
            || !arena
                .contains_private_view(v)
                .map_err(crate::resident_util::fault)?
        {
            return Err(Error::InvalidBufferLength);
        }
    }
    if h[6].size_in_used() != 4 {
        return Err(Error::InvalidBufferLength);
    }
    let arg =
        |i: usize| crate::resident_util::raw_arg(h[i].clone(), h[i].size_in_used() as usize / 4);
    // SAFETY: genuine full private owners, all offsets/grids, device CMMA/SLM
    // capacity and exact fault owner checked. Each source index is also causal.
    unsafe {
        attention_xmx::launch_unchecked::<WgpuRuntime>(
            &client,
            CubeCount::Static(1, 2, count as u32),
            CubeDim::new_1d(THREADS),
            arg(0)?,
            arg(1)?,
            arg(3)?,
            arg(4)?,
            arg(5)?,
            arg(6)?,
            count,
            start,
            cap,
            offset,
        );
    }
    LAUNCHES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    Ok(true)
}

pub(crate) fn definition() -> KernelDefinition {
    let mut b = KernelBuilder::default();
    cubecl::ir::AddressType::U32.register(&mut b.scope);
    let q = b.input_array(f32::as_type(&b.scope));
    let c = b.input_array(u32::as_type(&b.scope));
    let s = b.input_array(u32::as_type(&b.scope));
    let out = b.output_array(f32::as_type(&b.scope));
    let stats = b.output_array(f32::as_type(&b.scope));
    let f = b.output_array(
        cubecl::ir::StorageType::Atomic(cubecl::ir::ElemType::UInt(cubecl::ir::UIntKind::U32))
            .into(),
    );
    let a: [cubecl::ir::ManagedVariable; 4] =
        std::array::from_fn(|_| b.scalar(u32::as_type(&b.scope).storage_type()));
    attention_xmx::expand(
        &mut b.scope,
        q.into(),
        c.into(),
        s.into(),
        out.into(),
        stats.into(),
        f.into(),
        a[0].clone().into(),
        a[1].clone().into(),
        a[2].clone().into(),
        a[3].clone().into(),
    );
    b.build(KernelSettings::default().cube_dim(CubeDim::new_1d(THREADS)))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn actual_compiled_split_retains_half_rounding_before_f32_residual() {
        use cubecl::wgpu::{AutoCompiler, AutoRepresentation};
        use cubecl::{
            Compiler, WgpuCompilationOptions,
            ir::{ElemType, StorageType, UIntKind},
        };
        let mut options = WgpuCompilationOptions {
            supports_vulkan: true,
            ..Default::default()
        };
        options.vulkan.max_spirv_version = (1, 6);
        let AutoRepresentation::SpirV(kernel) = Compiler::compile(
            &mut AutoCompiler::SpirV(Default::default()),
            stage_definition(true),
            &options,
            cubecl::server::ExecutionMode::Unchecked,
            StorageType::Scalar(ElemType::UInt(UIntKind::U32)),
        )
        .unwrap() else {
            panic!("actual matrix SPIRV");
        };
        let w = &kernel.assembled_module;
        let mut widths = std::collections::BTreeMap::new();
        let mut conversions = std::collections::BTreeMap::new();
        let mut pairs = 0;
        let mut at = 5;
        while at < w.len() {
            let n = (w[at] >> 16) as usize;
            let op = w[at] & 65535;
            let a = &w[at + 1..at + n];
            if op == 22 {
                widths.insert(a[0], a[1]);
            }
            if op == 115 {
                conversions.insert(a[1], (widths.get(&a[0]).copied(), a[2]));
            }
            if op == 131 && widths.get(&a[0]) == Some(&32) {
                if let Some((Some(32), half)) = conversions.get(&a[3]) {
                    if let Some((Some(16), original)) = conversions.get(half) {
                        if *original == a[2] {
                            pairs += 1;
                        }
                    }
                }
            }
            at += n;
        }
        assert_eq!(
            pairs, 1,
            "original raw stage retains its half conversion chain before GPU lowering"
        );
    }
    #[test]
    fn integer_rounding_matches_independent_half_reference_at_all_half_boundaries() {
        for bits in 0..=65535u16 {
            let x = f16::from_bits(bits).to_f32();
            if !x.is_finite() {
                continue;
            }
            let f = x.to_bits();
            for candidate in [
                f,
                f.wrapping_add(1),
                f.wrapping_sub(1),
                f.wrapping_add(0xfff),
                f.wrapping_add(0x1000),
                f.wrapping_add(0x1001),
                f.wrapping_sub(0x1000),
            ] {
                let value = f32::from_bits(candidate);
                if !value.is_finite() {
                    continue;
                }
                assert_eq!(
                    rounded_half_bits(candidate),
                    f16::from_f32(value).to_f32().to_bits(),
                    "input bits {candidate:08x}"
                );
            }
        }
    }
    #[test]
    fn every_cooperative_tile_has_exact_one_writer_and_stays_inside_original_slots() {
        assert_eq!(SHARED_BYTES, 36144);
        assert!(SHARED_BYTES <= 49152);
        assert_eq!(THREADS, 8 * 16);
        let mut outputs = std::collections::BTreeSet::new();
        let mut scores = std::collections::BTreeSet::new();
        for sg in 0..8usize {
            for i in 0..128usize {
                let h = (sg / 4) * 8 + i / 16;
                let col = (sg % 4) * 16 + i % 16;
                assert!(scores.insert(h * 64 + col));
            }
        }
        for sg in 0..8usize {
            for i in 0..128usize {
                let h = (sg / 4) * 8 + i / 16;
                for g in 0..4usize {
                    let d = g * 64 + (sg % 4) * 16 + i % 16;
                    if h < 12 {
                        assert!(outputs.insert(h * 256 + d));
                    }
                }
            }
        }
        assert_eq!(scores.len(), 16 * 64);
        assert_eq!(outputs.len(), 12 * 256);
        for t in 0..32usize {
            for kv in 0..2usize {
                for chunk in 0..33usize {
                    let slot = kv * 33 + chunk;
                    assert!((t * 202752 + (slot * 12 + 11) * 256 + 255) < 32 * 202752);
                    assert!((t * 1584 + (slot * 12 + 11) * 2 + 1) < 32 * 1584);
                }
            }
        }
    }
    #[test]
    fn persistent_chunks_publish_each_original_owned_partial_exactly_once() {
        for visible in [1usize, 31, 32, 33, 63, 64, 65, 2048, 2051, 32768] {
            let cells = visible.min(2051);
            for kv in 0..2usize {
                let mut initialized = vec![false; 33 * 12 * 256];
                let mut selected = vec![0u8; cells];
                let mut chunk = 0;
                while chunk * 64 < cells {
                    let here = (cells - chunk * 64).min(64);
                    for col in 0..here {
                        selected[chunk * 64 + col] += 1;
                    }
                    for sg in 0..8usize {
                        for lane in 0..16usize {
                            for i in (lane..128).step_by(16) {
                                let h = (sg / 4) * 8 + i / 16;
                                if h >= 12 {
                                    continue;
                                }
                                for g in 0..4usize {
                                    let dim = g * 64 + (sg % 4) * 16 + i % 16;
                                    let at = (chunk * 12 + h) * 256 + dim;
                                    assert!(!std::mem::replace(&mut initialized[at], true));
                                    assert!(((kv * 33 + chunk) * 12 + h) * 256 + dim < 202752);
                                }
                            }
                        }
                    }
                    chunk += 1;
                }
                assert!(selected.iter().all(|&visits| visits == 1));
                let used = chunk * 12 * 256;
                assert!(initialized[..used].iter().all(|&v| v));
                assert!(initialized[used..].iter().all(|&v| !v));
            }
        }
    }
    #[test]
    fn compiled_attention_shared_envelope_is_exact_for_every_layout() {
        use cubecl::ir::{ElemType, StorageType, UIntKind};
        use cubecl::wgpu::{AutoCompiler, AutoRepresentation};
        use cubecl::{Compiler, WgpuCompilationOptions};
        let mut actual = Vec::new();
        for mask in 0u8..8 {
            let mut options = WgpuCompilationOptions {
                supports_vulkan: true,
                ..Default::default()
            };
            options.vulkan.max_spirv_version = (1, 6);
            options.vulkan.supports_uniform_standard_layout = mask & 1 != 0;
            options.vulkan.supports_uniform_unsized_array = mask & 2 != 0;
            options.vulkan.supports_explicit_smem = mask & 4 != 0;
            let AutoRepresentation::SpirV(kernel) = Compiler::compile(
                &mut AutoCompiler::SpirV(Default::default()),
                definition(),
                &options,
                cubecl::server::ExecutionMode::Unchecked,
                StorageType::Scalar(ElemType::UInt(UIntKind::U32)),
            )
            .unwrap() else {
                panic!("actual matrix SPIRV");
            };
            actual.push(kernel.shared_size);
        }
        assert_eq!(actual, vec![SHARED_BYTES; 8]);
    }
    #[test]
    fn packed_stages_cover_all_scalars_and_cooperative_views_stay_in_bounds() {
        let mut keys = vec![0u8; 4096];
        let mut values = vec![0u8; 4096];
        let mut probs = vec![0u8; 1024];
        for tid in 0..128usize {
            for job in (tid..1024).step_by(128) {
                for j in 0..4usize {
                    let col = job / 16;
                    let dim = (job % 16) * 4 + j;
                    keys[col * 64 + dim] += 1;
                    let dim = job / 16;
                    let col = (job % 16) * 4 + j;
                    values[dim * 64 + col] += 1;
                }
            }
            for job in (tid..256).step_by(128) {
                for j in 0..4usize {
                    probs[(job / 16) * 64 + (job % 16) * 4 + j] += 1;
                }
            }
        }
        assert!(
            keys.iter()
                .chain(values.iter())
                .chain(probs.iter())
                .all(|&v| v == 1)
        );
        for sg in 0..8usize {
            let h = (sg / 4) * 8;
            let tile = (sg % 4) * 16;
            for g in 0..4usize {
                for kk in 0..4usize {
                    let k = kk * 16;
                    assert!((h * 256 + g * 64 + k) % 4 == 0);
                    assert!(h * 256 + g * 64 + k + 7 * 256 + 15 < 4096);
                    assert!((tile * 64 + k) % 4 == 0);
                    assert!(tile * 64 + k + 15 * 64 + 15 < 4096);
                    assert!((h * 64 + k) % 4 == 0);
                    assert!(h * 64 + k + 7 * 64 + 15 < 1024);
                }
            }
        }
    }
    #[test]
    fn power_of_two_query_scaling_and_hi_lo_preserve_normal_f32_precision() {
        for e in 1..255u32 {
            let up = f32::from_bits((268 - e).min(254) << 23);
            for fraction in [0, 1, 0x123456, 0x7fffff] {
                let x = f32::from_bits(e << 23 | fraction);
                let staged = x * up;
                assert!(staged.is_finite() && staged.abs() < 65504.0);
                let hi = f16::from_f32(staged);
                let lo = f16::from_f32(staged - hi.to_f32());
                let reconstructed =
                    (f64::from(hi.to_f32()) + f64::from(lo.to_f32())) / f64::from(up);
                assert!((reconstructed - f64::from(x)).abs() <= f64::from(x) * 1e-6 + 1e-44);
            }
        }
    }
}
