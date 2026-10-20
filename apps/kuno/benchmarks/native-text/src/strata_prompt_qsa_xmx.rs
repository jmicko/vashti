//! Connected QK/softmax/PV CMMA attention for full prompt batches.
//! Design: MIT Strata fb58e0 qsa_prompt_attn_xmx.dp.cpp v2. Original public
//! scalar/short paths remain intact. Only device-local scratch is introduced;
//! original full query/cache/selection/partial/stats/fault owners are retained.
use cubecl::{prelude::*, server::Handle, wgpu::WgpuRuntime};
use half::f16;
use kuno_text::{Error, Result};

pub(crate) const THREADS: u32 = 128;
pub(crate) const SHARED_BYTES: usize = 8192 * 2 + 16384 * 2 + 2048 * 4 + 64 * 4 + 256 * 4 + 9 * 4;

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
    let chunk = usize::cast_from(CUBE_POS_X);
    let kvh = usize::cast_from(CUBE_POS_Y);
    let t = usize::cast_from(CUBE_POS_Z);
    let qr = t + query_offset;
    let visible = start + qr + 1;
    let cells = visible.min(2051);
    if t < count && kvh < 2 && chunk * 64 < cells {
        let here = (cells - chunk * 64).min(64);
        let slot = kvh * 33 + chunk;
        let mut qh = SharedMemory::<f16>::new(4096usize);
        let mut ql = SharedMemory::<f16>::new(4096usize);
        let mut kv = SharedMemory::<f16>::new(16384usize);
        let mut tile = SharedMemory::<f32>::new(1024usize);
        let mut prob = SharedMemory::<f32>::new(1024usize);
        let mut rows = SharedMemory::<u32>::new(64usize);
        let mut scales = SharedMemory::<f32>::new(256usize);
        let mut reduction = SharedMemory::<f32>::new(9usize);
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
        sync_cube();
        qm = 0.0;
        for j in 0..8usize {
            qm = max(qm, reduction[j]);
        }
        // Exact power-of-two scaling before hi/lo splitting. Clamp the exponent
        // for finite subnormal inputs; no input is narrowed directly to half.
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
            let hi = f16::cast_from(x);
            qh[i] = hi;
            ql[i] = f16::cast_from(x - f32::cast_from(hi));
            i += 128;
        }
        // Col-major K: a complete64-cell,256-feature code image. All padding
        // and invalid cells are zero; source indices are checked before reads.
        i = tid;
        while i < 16384 {
            let c = i / 256;
            let d = i % 256;
            let tok = usize::cast_from(rows[c]);
            let mut x = 0.0f32;
            if tok < capacity {
                x = code(cache, tok, kvh * 256 + d);
            }
            kv[i] = f16::cast_from(x);
            i += 128;
        }
        i = tid;
        while i < 256 {
            let tok = usize::cast_from(rows[i / 4]);
            let mut x = 0.0f32;
            if tok < capacity {
                x = scale(cache, tok, kvh * 256 + (i % 4) * 64);
                if !(x.abs() <= f32::MAX) {
                    fault[0].fetch_or(1);
                    x = 0.0;
                }
            }
            scales[i] = x;
            i += 128;
        }
        i = tid;
        while i < 1024 {
            prob[i] = 0.0;
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
        let b = unsafe {
            cmma::Matrix::<f16>::uninitialized(
                cmma::MatrixIdent::B,
                8usize,
                16usize,
                16usize,
                cmma::MatrixLayout::ColMajor,
            )
        };
        let head = (sg / 4) * 8;
        let cell = (sg % 4) * 16;
        for g in 0..4usize {
            let c = cmma::Matrix::<f32>::from_value(
                cmma::MatrixIdent::Accumulator,
                8usize,
                16usize,
                16usize,
                cmma::MatrixLayout::Undefined,
                0.0,
            );
            for kk in 0..4usize {
                let k = g * 64 + kk * 16;
                cmma::load(&ah, &qh.to_slice().slice(head * 256 + k, qh.len()), 256);
                cmma::load(&al, &ql.to_slice().slice(head * 256 + k, ql.len()), 256);
                cmma::load(&b, &kv.to_slice().slice(cell * 256 + k, kv.len()), 256);
                cmma::execute::<f16, f16, f32, f32>(&ah, &b, &c, &c);
                cmma::execute::<f16, f16, f32, f32>(&al, &b, &c, &c);
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
                let col = cell + i % 16;
                let at = h * 64 + col;
                prob[at] += tile[sg * 128 + i] * scales[col * 4 + g];
                i += 16;
            }
            sync_cube();
        }
        // Same per64-cell softmax and original external partial/stat slots.
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
        // All QK matrix reads are complete before replacing K storage with V
        // and Q storage with the scaled probability hi/lo matrices.
        i = tid;
        while i < 16384 {
            let d = i / 64;
            let col = i % 64;
            let tok = usize::cast_from(rows[col]);
            let mut x = 0.0f32;
            if tok < capacity {
                x = code(cache, tok, 512 + kvh * 256 + d);
            }
            kv[i] = f16::cast_from(x);
            i += 128;
        }
        i = tid;
        while i < 256 {
            let tok = usize::cast_from(rows[i / 4]);
            let mut x = 0.0f32;
            if tok < capacity {
                x = scale(cache, tok, 512 + kvh * 256 + (i % 4) * 64);
                if !(x.abs() <= f32::MAX) {
                    fault[0].fetch_or(1);
                    x = 0.0;
                }
            }
            scales[i] = x;
            i += 128;
        }
        sync_cube();
        for g in 0..4usize {
            let mut vmax = 0.0f32;
            for col in 0..64usize {
                vmax = max(vmax, scales[col * 4 + g].abs());
            }
            let mut vup = 0.0f32;
            if vmax > 0.0 {
                vup = 16384.0 / vmax;
            }
            let vdown = vmax / 16384.0;
            i = tid;
            while i < 1024 {
                let hh = i / 64;
                let col = i % 64;
                let mut x = 0.0f32;
                if hh < 12 {
                    x = prob[i] * (scales[col * 4 + g] * vup);
                }
                let hi = f16::cast_from(x);
                qh[i] = hi;
                ql[i] = f16::cast_from(x - f32::cast_from(hi));
                i += 128;
            }
            sync_cube();
            let dim = g * 64 + (sg % 4) * 16;
            let c = cmma::Matrix::<f32>::from_value(
                cmma::MatrixIdent::Accumulator,
                8usize,
                16usize,
                16usize,
                cmma::MatrixLayout::Undefined,
                0.0,
            );
            for k in 0..4usize {
                cmma::load(&ah, &qh.to_slice().slice(head * 64 + k * 16, qh.len()), 64);
                cmma::load(&al, &ql.to_slice().slice(head * 64 + k * 16, ql.len()), 64);
                cmma::load(&b, &kv.to_slice().slice(dim * 64 + k * 16, kv.len()), 64);
                cmma::execute::<f16, f16, f32, f32>(&ah, &b, &c, &c);
                cmma::execute::<f16, f16, f32, f32>(&al, &b, &c, &c);
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
                let hh = head + i / 16;
                if hh < 12 {
                    let x = tile[sg * 128 + i] * vdown;
                    partial[t * 202752 + (slot * 12 + hh) * 256 + dim + i % 16] = x;
                    if !(x.abs() <= f32::MAX) {
                        fault[0].fetch_or(1);
                    }
                }
                i += 16;
            }
            sync_cube();
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
            CubeCount::Static(33, 2, count as u32),
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
    fn every_cooperative_tile_has_exact_one_writer_and_stays_inside_original_slots() {
        assert!(SHARED_BYTES <= 65536);
        assert_eq!(THREADS, 8 * 16);
        let mut outputs = std::collections::BTreeSet::new();
        let mut scores = std::collections::BTreeSet::new();
        for sg in 0..8usize {
            for i in 0..128 {
                let h = (sg / 4) * 8 + i / 16;
                let c = (sg % 4) * 16 + i % 16;
                assert!(h < 16 && c < 64 && scores.insert(h * 64 + c));
                for g in 0..4 {
                    if h < 12 {
                        let d = g * 64 + (sg % 4) * 16 + i % 16;
                        assert!(d < 256 && outputs.insert(h * 256 + d));
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
