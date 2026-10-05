//! Fixed-public multi-record INFO ownership acceptance, no arbitrary input.
//! Exercises distinct runtime scalars across automatic submissions and epochs.
use crate::private_arena::{BufferId, PrivateArena};
use cubecl::prelude::*;
use cubecl::wgpu::WgpuRuntime;
use cubecl_runtime::transient::TransientIoReceipt;
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};
use zeroize::{Zeroize, Zeroizing};
type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;
const RECORDS: usize = 67;
const WORDS: usize = 37;
const BASE: usize = 5;
const STRIDE: usize = 43;
const LOGICAL: usize = BASE + RECORDS * STRIDE + 13;

#[cube(launch)]
fn runtime_records(
    input: &Array<u32>,
    output: &mut Array<u32>,
    salt: u32,
    base: usize,
    count: usize,
) {
    let i = ABSOLUTE_POS;
    if i < count {
        output[base + i] = input[i] ^ salt;
    }
}
fn salt(epoch: usize, rank: usize) -> u32 {
    0x52a6_b3e1 ^ (rank as u32).wrapping_mul(0x7fa3_621d) ^ (epoch as u32).wrapping_mul(0x19e3_785b)
}
fn validate_span(base: usize, count: usize, input: usize, output: usize) -> Result<()> {
    if count == 0
        || count > input
        || base.checked_add(count).is_none_or(|end| end > output)
        || input > u32::MAX as usize
        || output > u32::MAX as usize
    {
        return Err("public INFO fixture span".into());
    }
    Ok(())
}
fn launch(arena: &PrivateArena, x: BufferId, y: BufferId, epoch: usize) -> Result<()> {
    if arena.layout(x)?.logical_words != WORDS || arena.layout(y)?.logical_words != LOGICAL {
        return Err("public INFO fixture owning geometry".into());
    }
    let client = arena.client()?;
    let input = arena.handle(x)?;
    let output = arena.handle(y)?;
    if input.stream != output.stream || x == y {
        return Err("public INFO fixture same-stream disjoint owners".into());
    }
    let left = client.get_resource(input.clone())?;
    let right = client.get_resource(output.clone())?;
    let left = left.resource();
    let right = right.resource();
    if left.offset != 0
        || right.offset != 0
        || left.size != arena.layout(x)?.owned_bytes as u64
        || right.size != arena.layout(y)?.owned_bytes as u64
        || left.size != left.buffer.size()
        || right.size != right.buffer.size()
        || left.buffer == right.buffer
    {
        return Err("public INFO fixture physical owning-span alias/extent".into());
    }
    // Preflight all logical regions BEFORE the first enqueue.
    for rank in 0..RECORDS {
        validate_span(BASE + rank * STRIDE, WORDS, WORDS, LOGICAL)?;
    }
    for rank in 0..RECORDS {
        // SAFETY: same client/stream, distinct complete arena owners; each
        // preflighted runtime region fits the supplied whole logical arrays.
        runtime_records::launch::<WgpuRuntime>(
            client,
            CubeCount::Static(1, 1, 1),
            CubeDim::new_1d(64),
            unsafe { ArrayArg::from_raw_parts(input.clone(), WORDS) },
            unsafe { ArrayArg::from_raw_parts(output.clone(), LOGICAL) },
            salt(epoch, rank),
            BASE + rank * STRIDE,
            WORDS,
        );
    }
    Ok(())
}
struct Scrub<'a>(&'a mut [u8]);
impl Drop for Scrub<'_> {
    fn drop(&mut self) {
        self.0.zeroize();
    }
}
fn verify(bytes: &[u8], owned: usize, pattern: &[u32], epoch: usize) -> Result<()> {
    if pattern.len() != WORDS
        || bytes.len() != owned
        || owned < LOGICAL * 4
        || !owned.is_multiple_of(4)
    {
        return Err("public INFO fixture full-span read".into());
    }
    for (i, b) in bytes.chunks_exact(4).enumerate() {
        let expected = if i >= BASE && (i - BASE) / STRIDE < RECORDS && (i - BASE) % STRIDE < WORDS
        {
            pattern[(i - BASE) % STRIDE] ^ salt(epoch, (i - BASE) / STRIDE)
        } else {
            0
        };
        if u32::from_le_bytes(b.try_into().map_err(|_| "public INFO fixture word")?) != expected {
            return Err(
                "public INFO fixture distinct runtime records or owning tail changed".into(),
            );
        }
    }
    Ok(())
}
fn audit(client: &ComputeClient<WgpuRuntime>) -> Result<TransientIoReceipt> {
    let r = cubecl::future::block_on(client.sanitize_transient_io_audited())?;
    let full = r
        .upload_bytes
        .checked_add(r.uniform_bytes)
        .and_then(|n| n.checked_add(r.readback_bytes))
        .ok_or("public INFO audit overflow")?;
    if full == 0 || full != r.inspected_bytes || r.inspected_buffers == 0 {
        return Err("public INFO audit incomplete owning spans".into());
    }
    Ok(r)
}
fn finish<T>(attempt: std::thread::Result<Result<T>>, clean: bool, retired: bool) -> Result<T> {
    if !clean || retired {
        return Err("public INFO owning cleanup or retired context".into());
    }
    attempt.map_err(|_| "public INFO probe panicked")?
}
pub fn run(client: &ComputeClient<WgpuRuntime>) -> Result<()> {
    // Original public bytes may persist; private main/INFO cleanup preserves them.
    let public = client.create_from_slice(&[0x53; 128]);
    let retired = Arc::new(AtomicBool::new(false));
    let mut arena = PrivateArena::new_transient(
        client.clone(),
        1024 * 1024,
        64 * 1024 * 1024,
        retired.clone(),
    )?;
    arena.enable_public_owned_zero_audit()?;
    let attempt = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| -> Result<_> {
        let pattern = Zeroizing::new(
            (0..WORDS)
                .map(|i| 0xa596_317b ^ (i as u32).wrapping_mul(0x13_7821))
                .collect::<Vec<_>>(),
        );
        let x = arena.upload_u32(&pattern)?;
        let y = arena.allocate_zeroed(LOGICAL)?;
        let owned = arena.layout(y)?.owned_bytes;
        let retained = arena.handle(y)?.clone();
        let mut reports = Vec::new();
        for epoch in 0..2 {
            launch(&arena, x, y, epoch)?;
            let mut bytes = client.read_one(retained.clone())?;
            {
                let scrub = Scrub(&mut bytes);
                verify(scrub.0, owned, &pattern, epoch)?;
                if cubecl::future::block_on(client.sanitize_transient_io()).is_ok() {
                    return Err("public INFO cleanup accepted retained read lease".into());
                }
            }
            drop(bytes);
            let r = audit(client)?;
            if r.upload_bytes == 0 || r.uniform_bytes == 0 || r.readback_bytes == 0 {
                return Err("public INFO probe missing complete I/O kind".into());
            }
            reports.push(r);
        }
        // Cancellation/abandonment leaves many queued records to owning close.
        // A dropped read future must release its lease; no epoch is accepted here.
        launch(&arena, x, y, 2)?;
        let pending = client.read_async(vec![retained.clone()]);
        if cubecl::future::block_on(client.sanitize_transient_io()).is_ok() {
            return Err("public INFO cleanup accepted pending lease".into());
        }
        drop(pending);
        Ok((reports, retained, owned, y))
    }));
    let close = arena.close();
    let (reports, retained, owned, stale) =
        finish(attempt, close.is_ok(), retired.load(Ordering::Acquire))?;
    let main = arena
        .owned_zero_audit_receipt()
        .ok_or("public INFO missing full main zero audit")?;
    if arena.handle(stale).is_ok() {
        return Err("public INFO stale owner accepted after close".into());
    }
    let erased = client.read_one(retained)?;
    if erased.len() != owned || erased.iter().any(|&b| b != 0) {
        return Err("public INFO abandoned main owner not erased".into());
    }
    drop(erased);
    let after = audit(client)?;
    let original = client.read_one(public)?;
    if original.len() != 128 || original.iter().any(|&b| b != 0x53) {
        return Err("public INFO erasure changed public weight fixture".into());
    }
    drop(original);
    audit(client)?;
    if retired.load(Ordering::Acquire) {
        return Err("public INFO context retired".into());
    }
    if let Some(path) = std::env::var_os("KUNO_PUBLIC_RECEIPT") {
        use std::io::Write;
        let epochs = reports.into_iter().map(|r| serde_json::json!({"epoch":r.epoch,"upload_bytes":r.upload_bytes,"info_target_bytes":r.uniform_bytes,"readback_bytes":r.readback_bytes,"inspected_bytes":r.inspected_bytes,"inspected_buffers":r.inspected_buffers})).collect::<Vec<_>>();
        let r = serde_json::json!({"scope":"fixed-public-owned-multi-record-INFO-operator-only","records_per_epoch":RECORDS,"numeric_epochs":2,"runtime_scalars_distinct":true,"all_gaps_and_owned_tails_unchanged":true,"retained_readback_rejected":true,"pending_readback_drop":true,"abandoned_queued_records_cleaned":true,"public_weight_fixture_preserved":true,"main_erased_buffers":main.buffers,"main_erased_bytes":main.bytes,"epochs":epochs,"after_cleanup_inspected_bytes":after.inspected_bytes,"context_retired":false,"full_graph_or_performance_accepted":false});
        let mut out = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(path)?;
        out.write_all(serde_json::to_string_pretty(&r)?.as_bytes())?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn full_fixture_guards_gaps_tails_distinct_records_and_spans() {
        let pattern = (0..WORDS).map(|i| i as u32 * 7).collect::<Vec<_>>();
        let owned = 32768;
        let mut bytes = vec![0; owned];
        for rank in 0..RECORDS {
            for (i, &v) in pattern.iter().enumerate() {
                let start = (BASE + rank * STRIDE + i) * 4;
                bytes[start..start + 4].copy_from_slice(&(v ^ salt(0, rank)).to_le_bytes());
            }
        }
        verify(&bytes, owned, &pattern, 0).unwrap();
        assert!(verify(&bytes, owned, &pattern, 1).is_err());
        for bad in [0, (BASE + WORDS) * 4, LOGICAL * 4, owned - 1] {
            bytes[bad] ^= 1;
            assert!(verify(&bytes, owned, &pattern, 0).is_err());
            bytes[bad] ^= 1;
        }
        assert!(verify(&bytes[..owned - 4], owned, &pattern, 0).is_err());
        assert!(verify(&bytes, owned, &pattern[..WORDS - 1], 0).is_err());
        assert!(validate_span(usize::MAX, 1, 1, usize::MAX).is_err());
        assert!(validate_span(BASE, 0, WORDS, LOGICAL).is_err());
        assert!(validate_span(BASE, WORDS + 1, WORDS, LOGICAL).is_err());
        assert!(validate_span(LOGICAL, WORDS, WORDS, LOGICAL).is_err());
        assert_eq!(finish(Ok(Ok(7)), true, false).unwrap(), 7);
        assert!(finish(Ok(Ok(7)), false, false).is_err());
        assert!(finish(Ok(Ok(7)), true, true).is_err());
        assert!(finish::<u32>(Ok(Err("rejected".into())), true, false).is_err());
        assert!(finish::<u32>(Err(Box::new("panic")), true, false).is_err());
    }
    #[test]
    fn offline_info_values_stay_runtime_and_storage_binding_has_no_private_key() {
        use cubecl::compute::KernelBuilder;
        use cubecl::ir::{ElemType, StorageType, Type, UIntKind};
        use cubecl::server::ExecutionMode;
        use cubecl::wgpu::WgslCompiler;
        use cubecl::{Compiler, KernelSettings, WgpuCompilationOptions};
        let u = StorageType::Scalar(ElemType::UInt(UIntKind::U32));
        for mode in [ExecutionMode::Checked, ExecutionMode::Unchecked] {
            let mut b = KernelBuilder::default();
            b.scope.register_type::<u32>(u);
            b.scope.register_type::<usize>(u);
            let input = b.input_array(Type::new(u)).into();
            let output = b.output_array(Type::new(u)).into();
            let salt = b.scalar(u).into();
            let base = b.scalar(u).into();
            let count = b.scalar(u).into();
            runtime_records::expand(&mut b.scope, input, output, salt, base, count);
            let source = WgslCompiler::default()
                .compile(
                    b.build(
                        KernelSettings::default()
                            .cube_dim(CubeDim::new_1d(64))
                            .kernel_name("info_slab_probe"),
                    ),
                    &WgpuCompilationOptions::default(),
                    mode,
                    u,
                )
                .unwrap()
                .to_string();
            assert!(source.contains("var<storage, read> info"));
            for i in 0..3 {
                assert!(source.contains(&format!("scalars_u32[{i}]")));
            }
            assert!(!source.contains("scalars_u32[3]"));
        }
    }
}
