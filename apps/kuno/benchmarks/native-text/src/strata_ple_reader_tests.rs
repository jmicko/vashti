use super::*;
use std::{
    fs::{File, OpenOptions},
    io::Write,
    path::PathBuf,
    sync::atomic::AtomicU64,
};
struct Fixture {
    path: PathBuf,
    model: kuno_text::GgufFile,
    rb: usize,
}
impl Fixture {
    fn new(rows: u64) -> Self {
        static ID: AtomicU64 = AtomicU64::new(0);
        let path = std::env::temp_dir().join(format!(
            "kuno-ple-reader-public-{}-{}.gguf",
            std::process::id(),
            ID.fetch_add(1, Ordering::Relaxed)
        ));
        let mut header = b"GGUF".to_vec();
        header.extend(3u32.to_le_bytes());
        header.extend(1u64.to_le_bytes());
        header.extend(0u64.to_le_bytes());
        header.extend(4u64.to_le_bytes());
        header.extend(b"rows");
        header.extend(2u32.to_le_bytes());
        header.extend(160u64.to_le_bytes());
        header.extend(rows.to_le_bytes());
        header.extend(20u32.to_le_bytes());
        header.extend(0u64.to_le_bytes());
        header.resize(header.len().div_ceil(32) * 32, 0);
        let mut file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&path)
            .unwrap();
        file.write_all(&header).unwrap();
        for row in 0..rows as u32 {
            file.write_all(&bytes(row, 90)).unwrap();
        }
        drop(file);
        let model = kuno_text::GgufFile::open(&path, kuno_text::GgufLimits::default()).unwrap();
        Self {
            path,
            model,
            rb: 90,
        }
    }
    fn source(&self) -> DirectSource<'_> {
        let t = self.model.tensor("rows").unwrap();
        t.positional_reader().unwrap().direct_source(t).unwrap()
    }
    fn expected(&self, rows: &[u32]) -> Vec<u8> {
        let t = self.model.tensor("rows").unwrap();
        let mut out = vec![0; rows.len() * self.rb];
        for (&row, b) in rows.iter().zip(out.chunks_exact_mut(self.rb)) {
            t.read_row(row as u64, b).unwrap();
        }
        out
    }
}
impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.path);
        let _ = std::fs::remove_file(self.path.with_extension("old"));
    }
}
fn bytes(row: u32, rb: usize) -> Vec<u8> {
    let mut v: Vec<_> = (0..rb)
        .map(|i| (i * 17 + row as usize * 31) as u8)
        .collect();
    v[..4].copy_from_slice(&row.to_le_bytes());
    v
}
#[test]
fn eight_way_original_hash_replacement_hits_and_owned_erasure() {
    let mut c = Cache::new(16, 90).unwrap();
    let rows: Vec<_> = (0..100u32).filter(|&r| c.set(r) == 0).take(9).collect();
    assert_eq!(rows.len(), 9);
    for &row in &rows[..8] {
        c.insert(row, &bytes(row, 90)).unwrap();
    }
    let cursor = c.next[0];
    c.insert(rows[1], &bytes(rows[1], 90)).unwrap();
    assert_eq!(c.next[0], cursor);
    c.insert(rows[8], &bytes(rows[8], 90)).unwrap();
    assert!(c.get(rows[0]).is_none());
    for &row in &rows[1..] {
        assert_eq!(c.get(row).unwrap(), bytes(row, 90));
    }
    let owners = (
        c.keys.as_ptr(),
        c.data.as_ptr(),
        c.next.as_ptr(),
        c.keys.capacity(),
        c.data.capacity(),
        c.next.capacity(),
    );
    assert!(c.data.iter().any(|&v| v != 0));
    c.erase().unwrap();
    assert_eq!(
        owners,
        (
            c.keys.as_ptr(),
            c.data.as_ptr(),
            c.next.as_ptr(),
            c.keys.capacity(),
            c.data.capacity(),
            c.next.capacity()
        )
    );
    assert!(
        c.keys.iter().all(|&v| v == 0)
            && c.data.iter().all(|&v| v == 0)
            && c.next.iter().all(|&v| v == 0)
    );
    assert!(Cache::new(0, 90).is_err());
    assert!(Cache::new(17, 90).is_err());
    assert!(Cache::new(16, 111).is_err());
}
#[test]
fn original_default_cache_and_complete_host_reservation_are_bounded() {
    let mut c = Cache::new(ROWS, 90).unwrap();
    let actual = c.keys.capacity() * 4 + c.data.capacity() + c.next.capacity();
    assert_eq!(actual, 98_697_216);
    assert!(actual + WORKER_HOST + PLAN_HOST + RAW_HOST + SLOT_HOST + 256 * 1024 <= HOST_BYTES);
    assert!(HOST_BYTES.is_multiple_of(65536));
    assert_eq!((WORKERS, DEPTH), (64, 256));
    c.insert(17, &bytes(17, 90)).unwrap();
    c.erase().unwrap();
}
#[test]
fn page_grouping_direct_reads_straddles_duplicates_tail_and_real_warm_hits() {
    let f = Fixture::new(512);
    let source = f.source();
    let pool = Pool::new().unwrap();
    let cancel = AtomicBool::new(false);
    let mut c = Cache::new(1024, 90).unwrap();
    let rows = [511, 91, 0, 44, 45, 46, 1, 44];
    let mut raw = zeros(rows.len() * 90).unwrap();
    let p = plan(&rows, source.table_offset(), source.rows(), &c, &mut raw).unwrap();
    assert!(p.jobs.iter().any(|j| j.length == 8192));
    assert!(p.jobs.len() < p.uses.len());
    pool.run(&source, || pool.execute(&p, &mut raw, &mut c, &cancel))
        .unwrap();
    assert_eq!(&*raw, &f.expected(&rows));
    let mut warm = zeros(rows.len() * 90).unwrap();
    let p = plan(&rows, source.table_offset(), source.rows(), &c, &mut warm).unwrap();
    assert!(p.jobs.is_empty());
    assert_eq!(warm, raw);
    c.erase().unwrap();
}
#[test]
fn original_descriptor_reopen_survives_path_replacement_and_checks_alignment() {
    let f = Fixture::new(64);
    let expected = f.expected(&[0, 63]);
    let old = f.path.with_extension("old");
    std::fs::rename(&f.path, &old).unwrap();
    File::create(&f.path)
        .unwrap()
        .write_all(b"replacement")
        .unwrap();
    let source = f.source();
    source.verify_file_unchanged().unwrap();
    let mut c = Cache::new(128, 90).unwrap();
    let pool = Pool::new().unwrap();
    let cancel = AtomicBool::new(false);
    let mut raw = zeros(180).unwrap();
    let p = plan(&[0, 63], source.table_offset(), source.rows(), &c, &mut raw).unwrap();
    pool.run(&source, || pool.execute(&p, &mut raw, &mut c, &cancel))
        .unwrap();
    assert_eq!(&*raw, &expected);
    let mut b = Box::new(Aligned([0; 8192]));
    assert!(source.read_aligned(1, &mut b.0[..4096]).is_err());
    assert!(source.read_aligned(0, &mut b.0[1..4097]).is_err());
    assert!(source.read_aligned(4096, &mut b.0[..8192]).is_err());
    c.erase().unwrap();
}
#[test]
fn queued_failure_and_cancellation_drain_all_real_readers_and_buffers() {
    let f = Fixture::new(4096);
    let source = f.source();
    let pool = Pool::new().unwrap();
    let cancel = AtomicBool::new(false);
    let mut c = Cache::new(128, 90).unwrap();
    let rows: Vec<_> = (0..4096).collect();
    let mut raw = zeros(rows.len() * 90).unwrap();
    let mut p = plan(&rows, source.table_offset(), source.rows(), &c, &mut raw).unwrap();
    assert!(p.jobs.len() > 64);
    p.jobs[7].offset = 1;
    assert!(
        pool.run(&source, || pool.execute(&p, &mut raw, &mut c, &cancel))
            .is_err()
    );
    assert!(
        pool.buffers
            .iter()
            .all(|b| b.lock().unwrap().0.iter().all(|&v| v == 0))
    );
    {
        let q = pool.queue.lock().unwrap();
        assert!(q.pending.is_empty() && q.done.is_empty());
    }
    c.erase().unwrap();
    let pool = Pool::new().unwrap();
    cancel.store(true, Ordering::Release);
    let p = plan(&rows, source.table_offset(), source.rows(), &c, &mut raw).unwrap();
    assert!(
        pool.run(&source, || pool.execute(&p, &mut raw, &mut c, &cancel))
            .is_err()
    );
    assert!(raw.iter().all(|&v| v == 0));
    c.erase().unwrap();
}
#[test]
fn request_scope_shares_prompt_and_decode_and_rejects_stale_or_cross_model_leases() {
    let f = Fixture::new(512);
    let cancel = AtomicBool::new(false);
    let mut stale = None;
    let binding = Binding::Fixture(1);
    with_source(binding, f.source(), 1024, &cancel, || {
        let lease = current().unwrap();
        stale = Some(lease.clone());
        assert!(
            lease
                .read_rows(Binding::Fixture(2), 90, &[0], &cancel)
                .is_err()
        );
        assert!(lease.read_rows(binding, 110, &[0], &cancel).is_err());
        assert!(lease.read_rows(binding, 90, &[], &cancel).is_err());
        let rows = [0, 44, 511, 45];
        assert_eq!(
            &*lease.read_rows(binding, 90, &rows, &cancel)?,
            &f.expected(&rows)
        );
        // This is the actual clone passed to a prompt producer; main TLS is
        // intentionally absent in that child, so it cannot discover another owner.
        std::thread::scope(|scope| {
            let child = scope.spawn(|| {
                assert!(current().is_none());
                lease.read_rows(binding, 90, &rows, &cancel)
            });
            let decode = lease.read_rows(binding, 90, &[91, 46], &cancel).unwrap();
            assert_eq!(&*decode, &f.expected(&[91, 46]));
            assert_eq!(&*child.join().unwrap().unwrap(), &f.expected(&rows));
        });
        Ok(())
    })
    .unwrap();
    assert!(current().is_none());
    assert!(
        stale
            .unwrap()
            .read_rows(binding, 90, &[0], &cancel)
            .is_err()
    );
}
#[test]
fn original_stat_changes_are_rejected_even_after_cache_hits_and_tls_is_cleared() {
    let f = Fixture::new(512);
    let cancel = AtomicBool::new(false);
    let binding = Binding::Fixture(3);
    assert!(
        with_source(binding, f.source(), 1024, &cancel, || {
            let lease = current().unwrap();
            lease.read_rows(binding, 90, &[0, 44], &cancel)?;
            OpenOptions::new()
                .write(true)
                .open(&f.path)
                .unwrap()
                .set_len(0)
                .unwrap();
            assert!(lease.read_rows(binding, 90, &[0, 44], &cancel).is_err());
            Ok(())
        })
        .is_err()
    );
    assert!(current().is_none());
}
#[test]
fn body_error_and_unwind_join_pool_and_remove_the_request_owner() {
    let f = Fixture::new(64);
    let cancel = AtomicBool::new(false);
    assert!(
        with_source(
            Binding::Fixture(4),
            f.source(),
            16,
            &cancel,
            || -> Result<()> { Err(bad()) }
        )
        .is_err()
    );
    assert!(current().is_none());
    let panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _: Result<()> = with_source(Binding::Fixture(5), f.source(), 16, &cancel, || {
            panic!("public reader fixture unwind")
        });
    }));
    assert!(panic.is_err());
    assert!(current().is_none());
    let pool = Pool::new().unwrap();
    let source = f.source();
    let panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _: Result<()> = pool.run(&source, || panic!("public coordinator fixture unwind"));
    }));
    assert!(panic.is_err());
    assert!(
        pool.buffers
            .iter()
            .all(|b| b.lock().unwrap().0.iter().all(|&v| v == 0))
    );
}

#[test]
fn worst_original_prompt_plan_capacity_and_invalid_rows_remain_bounded() {
    let c = Cache::new(16, 110).unwrap();
    assert!(c.get(u32::MAX).is_none());
    let rows: Vec<_> = (0..MAX_ROWS as u32).map(|r| r * 64).collect();
    let mut raw = zeros(rows.len() * 110).unwrap();
    let p = plan(&rows, 96, 320001536, &c, &mut raw).unwrap();
    assert_eq!(p.uses.len(), MAX_ROWS);
    assert_eq!(p.jobs.len(), MAX_ROWS);
    assert!(p.jobs.iter().all(|j| j.length == 4096 || j.length == 8192));
    assert!(p.jobs.windows(2).all(|w| w[0].offset < w[1].offset));
    let used = p.uses.capacity() * size_of::<Use>()
        + p.jobs.capacity() * size_of::<Job>()
        + rows.capacity() * 4
        + raw.capacity();
    assert!(2 * used <= PLAN_HOST + RAW_HOST);
    let too_many = vec![0; MAX_ROWS + 1];
    assert!(plan(&too_many, 96, 320001536, &c, &mut raw).is_err());
    assert!(plan(&[u32::MAX], 96, u64::MAX, &c, &mut vec![0; 110]).is_err());
    assert!(plan(&[1], u64::MAX, 320001536, &c, &mut vec![0; 110]).is_err());
}
