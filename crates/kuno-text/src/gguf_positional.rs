//! Bounded positional reads through a clone of the already-inspected descriptor.
//! No path reopen, shared seek cursor, mapping, model copy or payload retention.
use super::{GgufFile, TensorView};
use crate::{Error, Result};
use std::fs::File;

/// Pins the original model owner and its open descriptor for the entire reader.
/// The descriptor is cloned once; concurrent reads do not take the seek mutex.
/// Same-size/same-mtime malicious rewrites remain outside the original guard.
pub struct PositionalReader<'m> {
    model: &'m GgufFile,
    file: File,
}

impl GgufFile {
    /// Enable one shared read-only map of the already-inspected descriptor.
    /// Does not prefault, pin pages, copy the model or reopen its path.
    ///
    /// # Safety
    /// The caller must prevent in-place modification and truncation of this
    /// inode until this GgufFile and all its borrowed consumers are dropped.
    /// Atomic path replacement is harmless: the inspected descriptor stays
    /// pinned. Stat checks detect ordinary changes but cannot close the race
    /// between checking and using a file mapping.
    #[allow(unsafe_code)]
    pub unsafe fn enable_mapped_weights(&self) -> Result<()> {
        let file = self
            .file
            .lock()
            .map_err(|_| Error::InvalidFormat("model reader lock poisoned"))?;
        self.check_stat(&file)?;
        if self.mapping.get().is_none() {
            // SAFETY: the caller guarantees the inode is immutable for the full
            // model lifetime. Map uses the original fd, never a path lookup.
            let map = unsafe { memmap2::MmapOptions::new().map(&*file)? };
            if map.len() as u64 != self.file_len {
                return Err(Error::InvalidFormat("model mapping extent"));
            }
            self.check_stat(&file)?;
            let _ = self.mapping.set(map);
        }
        Ok(())
    }

    pub fn weights_mapped(&self) -> bool {
        self.mapping.get().is_some()
    }

    pub fn positional_reader(&self) -> Result<PositionalReader<'_>> {
        #[cfg(not(unix))]
        return Err(Error::InvalidFormat("positional reader requires Unix"));
        #[cfg(unix)]
        {
            let original = self
                .file
                .lock()
                .map_err(|_| Error::InvalidFormat("model reader lock poisoned"))?;
            self.check_stat(&original)?;
            let file = original.try_clone()?;
            self.check_stat(&file)?;
            Ok(PositionalReader { model: self, file })
        }
    }
}

impl<'m> PositionalReader<'m> {
    pub fn belongs_to(&self, model: &GgufFile) -> bool {
        std::ptr::eq(self.model, model)
    }

    pub fn owns(&self, tensor: TensorView<'_>) -> bool {
        tensor.belongs_to(self.model)
    }

    pub fn verify_file_unchanged(&self) -> Result<()> {
        self.model.check_stat(&self.file)
    }

    /// Borrow only the requested original packed span. All consumers must join
    /// before releasing the model; the returned slice cannot outlive its owner.
    /// None preserves the positional-read path when mapping was not enabled.
    pub fn mapped_range(
        &self,
        tensor: TensorView<'m>,
        offset: u64,
        length: usize,
    ) -> Result<Option<&'m [u8]>> {
        if !self.owns(tensor) {
            return Err(Error::InvalidFormat("positional tensor ownership"));
        }
        let length = u64::try_from(length).map_err(|_| Error::ArithmeticOverflow)?;
        if length > self.model.limits.max_tensor_read_bytes {
            return Err(Error::LimitExceeded("tensor read bytes"));
        }
        if offset
            .checked_add(length)
            .ok_or(Error::ArithmeticOverflow)?
            > tensor.info.byte_len
        {
            return Err(Error::OutOfBounds);
        }
        let absolute = self
            .model
            .data_offset
            .checked_add(tensor.info.offset)
            .and_then(|base| base.checked_add(offset))
            .ok_or(Error::ArithmeticOverflow)?;
        let end = absolute
            .checked_add(length)
            .ok_or(Error::ArithmeticOverflow)?;
        if end > self.model.file_len {
            return Err(Error::OutOfBounds);
        }
        self.verify_file_unchanged()?;
        self.model
            .mapping
            .get()
            .map(|map| {
                let first = usize::try_from(absolute).map_err(|_| Error::ArithmeticOverflow)?;
                let end = usize::try_from(end).map_err(|_| Error::ArithmeticOverflow)?;
                map.get(first..end).ok_or(Error::OutOfBounds)
            })
            .transpose()
    }

    /// Bounds and identity reject before mutation. Any I/O or late consistency
    /// failure clears the complete caller slice; no partial payload is accepted.
    pub fn read_range(
        &self,
        tensor: TensorView<'m>,
        byte_offset: u64,
        output: &mut [u8],
    ) -> Result<()> {
        if !self.owns(tensor) {
            return Err(Error::InvalidFormat("positional tensor ownership"));
        }
        let length = u64::try_from(output.len()).map_err(|_| Error::ArithmeticOverflow)?;
        if length > self.model.limits.max_tensor_read_bytes {
            return Err(Error::LimitExceeded("tensor read bytes"));
        }
        if byte_offset
            .checked_add(length)
            .ok_or(Error::ArithmeticOverflow)?
            > tensor.info.byte_len
        {
            return Err(Error::OutOfBounds);
        }
        let absolute = self
            .model
            .data_offset
            .checked_add(tensor.info.offset)
            .and_then(|base| base.checked_add(byte_offset))
            .ok_or(Error::ArithmeticOverflow)?;
        // Prove the complete absolute span before creating a read operation.
        absolute
            .checked_add(length)
            .filter(|&end| end <= self.model.file_len)
            .ok_or(Error::OutOfBounds)?;
        let result = (|| {
            self.verify_file_unchanged()?;
            #[cfg(unix)]
            {
                use std::os::unix::fs::FileExt;
                read_exact_at(output, absolute, |target, offset| {
                    self.file.read_at(target, offset)
                })?;
            }
            #[cfg(not(unix))]
            return Err(Error::InvalidFormat("positional reader requires Unix"));
            #[cfg(unix)]
            self.verify_file_unchanged()
        })();
        if result.is_err() {
            output.fill(0);
        }
        result
    }

    pub fn read_row(&self, tensor: TensorView<'m>, row: u64, output: &mut [u8]) -> Result<()> {
        if !self.owns(tensor) {
            return Err(Error::InvalidFormat("positional tensor ownership"));
        }
        if row >= tensor.info.row_count {
            return Err(Error::OutOfBounds);
        }
        if output.len() as u64 != tensor.info.row_bytes {
            return Err(Error::InvalidBufferLength);
        }
        let offset = row
            .checked_mul(tensor.info.row_bytes)
            .ok_or(Error::ArithmeticOverflow)?;
        self.read_range(tensor, offset, output)
    }
}

#[cfg(any(unix, test))]
fn read_exact_at(
    output: &mut [u8],
    first: u64,
    mut read: impl FnMut(&mut [u8], u64) -> std::io::Result<usize>,
) -> Result<()> {
    let length = u64::try_from(output.len()).map_err(|_| Error::ArithmeticOverflow)?;
    first.checked_add(length).ok_or(Error::ArithmeticOverflow)?;
    let mut done = 0usize;
    while done < output.len() {
        let offset = first
            .checked_add(done as u64)
            .ok_or(Error::ArithmeticOverflow)?;
        match read(&mut output[done..], offset) {
            Ok(0) => return Err(Error::Io(std::io::ErrorKind::UnexpectedEof)),
            Ok(count) if count <= output.len() - done => done += count,
            Ok(_) => return Err(Error::InvalidFormat("positional read count")),
            Err(error) if error.kind() == std::io::ErrorKind::Interrupted => continue,
            Err(error) => return Err(error.into()),
        }
    }
    Ok(())
}

#[cfg(all(test, unix))]
mod tests {
    use super::*;
    use crate::GgufLimits;
    use std::io::{Seek, SeekFrom, Write};
    use std::sync::atomic::{AtomicU64, Ordering};

    static NEXT: AtomicU64 = AtomicU64::new(0);
    struct Fixture {
        path: std::path::PathBuf,
        model: GgufFile,
        data_offset: u64,
    }
    impl Drop for Fixture {
        fn drop(&mut self) {
            let _ = std::fs::remove_file(&self.path);
        }
    }
    fn fixture(limit: u64) -> Fixture {
        let path = std::env::temp_dir().join(format!(
            "kuno-positional-{}-{}.gguf",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        let mut bytes = Vec::new();
        bytes.extend(b"GGUF");
        bytes.extend(3u32.to_le_bytes());
        bytes.extend(1u64.to_le_bytes());
        bytes.extend(0u64.to_le_bytes());
        bytes.extend(1u64.to_le_bytes());
        bytes.extend(b"w");
        bytes.extend(3u32.to_le_bytes());
        for dimension in [4u64, 2, 3] {
            bytes.extend(dimension.to_le_bytes());
        }
        bytes.extend(0u32.to_le_bytes());
        bytes.extend(0u64.to_le_bytes());
        bytes.resize(bytes.len().next_multiple_of(32), 0);
        let data_offset = bytes.len() as u64;
        for value in 0..24u32 {
            bytes.extend((value as f32).to_le_bytes());
        }
        std::fs::write(&path, bytes).unwrap();
        let model = GgufFile::open(
            &path,
            GgufLimits {
                max_tensor_read_bytes: limit,
                ..GgufLimits::default()
            },
        )
        .unwrap();
        Fixture {
            path,
            model,
            data_offset,
        }
    }

    #[test]
    fn actual_descriptor_parallel_reads_keep_original_seek_cursor() {
        let f = fixture(96);
        let reader = f.model.positional_reader().unwrap();
        let tensor = f.model.tensor("w").unwrap();
        let cursor = f.model.file.lock().unwrap().stream_position().unwrap();
        std::thread::scope(|scope| {
            let mut tasks = Vec::new();
            for row in [5u64, 0, 3, 1] {
                let reader = &reader;
                tasks.push(scope.spawn(move || {
                    let mut bytes = [0u8; 16];
                    reader.read_row(tensor, row, &mut bytes).unwrap();
                    for (column, word) in bytes.chunks_exact(4).enumerate() {
                        assert_eq!(
                            f32::from_le_bytes(word.try_into().unwrap()),
                            (row * 4 + column as u64) as f32
                        );
                    }
                }));
            }
            for task in tasks {
                task.join().unwrap();
            }
        });
        assert_eq!(
            f.model.file.lock().unwrap().stream_position().unwrap(),
            cursor
        );
        // Ordinary serialized access still sees its original tensor and bytes.
        let mut ordinary = [0u8; 16];
        tensor.read_row(2, &mut ordinary).unwrap();
        assert_eq!(f32::from_le_bytes(ordinary[..4].try_into().unwrap()), 8.);
    }

    #[test]
    #[allow(unsafe_code)]
    fn mapped_spans_share_original_inode_and_enforce_owner_bounds_and_limits() {
        let f = fixture(96);
        let other = fixture(96);
        let reader = f.model.positional_reader().unwrap();
        let tensor = f.model.tensor("w").unwrap();
        assert!(reader.mapped_range(tensor, 0, 16).unwrap().is_none());
        // SAFETY: fixture payload stays immutable until all consumers drop.
        unsafe {
            f.model.enable_mapped_weights().unwrap();
        }
        assert!(f.model.weights_mapped());
        let original = reader.mapped_range(tensor, 16, 16).unwrap().unwrap();
        let mut read = [0u8; 16];
        reader.read_range(tensor, 16, &mut read).unwrap();
        assert_eq!(original, read);
        assert!(
            reader
                .mapped_range(other.model.tensor("w").unwrap(), 0, 16)
                .is_err()
        );
        assert!(reader.mapped_range(tensor, 95, 2).is_err());
        assert!(reader.mapped_range(tensor, u64::MAX, 2).is_err());
        assert!(reader.mapped_range(tensor, 0, 97).is_err());
        std::fs::rename(&f.path, f.path.with_extension("pinned")).unwrap();
        std::fs::write(&f.path, b"replacement inode is unrelated").unwrap();
        assert_eq!(
            reader
                .mapped_range(tensor, 16, 16)
                .unwrap()
                .unwrap()
                .as_ptr(),
            original.as_ptr()
        );
        std::fs::remove_file(f.path.with_extension("pinned")).unwrap();
        std::thread::scope(|scope| {
            for _ in 0..4 {
                let reader = &reader;
                scope.spawn(move || {
                    assert_eq!(reader.mapped_range(tensor, 16, 16).unwrap().unwrap(), read)
                });
            }
        });
    }

    #[test]
    fn identity_extent_limit_and_overflow_reject_before_mutation() {
        let a = fixture(16);
        let b = fixture(96);
        let reader = a.model.positional_reader().unwrap();
        let mut marker = [0xa5u8; 20];
        assert!(
            reader
                .read_range(b.model.tensor("w").unwrap(), 0, &mut marker)
                .is_err()
        );
        assert!(marker.iter().all(|&x| x == 0xa5));
        let tensor = a.model.tensor("w").unwrap();
        assert!(matches!(
            reader.read_range(tensor, 0, &mut marker),
            Err(Error::LimitExceeded(_))
        ));
        assert!(matches!(
            reader.read_range(tensor, 95, &mut marker[..2]),
            Err(Error::OutOfBounds)
        ));
        assert!(matches!(
            reader.read_range(tensor, u64::MAX, &mut marker[..2]),
            Err(Error::ArithmeticOverflow)
        ));
        assert!(marker.iter().all(|&x| x == 0xa5));
        assert!(reader.belongs_to(&a.model));
        assert!(!reader.belongs_to(&b.model));
    }

    #[test]
    fn path_replacement_does_not_redirect_pinned_descriptor() {
        let f = fixture(96);
        let reader = f.model.positional_reader().unwrap();
        let replacement = f.path.with_extension("replacement");
        std::fs::write(&replacement, b"unrelated replacement").unwrap();
        std::fs::rename(&replacement, &f.path).unwrap();
        let mut bytes = [0u8; 16];
        reader
            .read_row(f.model.tensor("w").unwrap(), 5, &mut bytes)
            .unwrap();
        assert_eq!(f32::from_le_bytes(bytes[..4].try_into().unwrap()), 20.);
    }

    #[test]
    fn changed_and_truncated_descriptor_clear_entire_output() {
        let f = fixture(96);
        let reader = f.model.positional_reader().unwrap();
        let mut writable = std::fs::OpenOptions::new()
            .write(true)
            .open(&f.path)
            .unwrap();
        writable.seek(SeekFrom::Start(f.data_offset)).unwrap();
        writable.write_all(&[0u8; 4]).unwrap();
        // A changed length guarantees the original consistency guard sees it.
        writable.set_len(f.data_offset + 4).unwrap();
        let mut marker = [0xa5u8; 16];
        assert!(matches!(
            reader.read_row(f.model.tensor("w").unwrap(), 0, &mut marker),
            Err(Error::FileChanged)
        ));
        assert_eq!(marker, [0; 16]);
    }

    #[test]
    fn controlled_short_interrupt_error_and_overflow_loops() {
        let mut output = [0u8; 7];
        let mut calls = 0;
        read_exact_at(&mut output, 11, |target, offset| {
            calls += 1;
            if calls == 1 {
                return Err(std::io::ErrorKind::Interrupted.into());
            }
            let count = target.len().min(2);
            for (index, byte) in target[..count].iter_mut().enumerate() {
                *byte = (offset + index as u64) as u8;
            }
            Ok(count)
        })
        .unwrap();
        assert_eq!(output, [11, 12, 13, 14, 15, 16, 17]);
        assert_eq!(calls, 5);
        assert!(matches!(
            read_exact_at(&mut output, 0, |_, _| Ok(0)),
            Err(Error::Io(std::io::ErrorKind::UnexpectedEof))
        ));
        assert!(matches!(
            read_exact_at(&mut output, 0, |_, _| Err(
                std::io::ErrorKind::PermissionDenied.into()
            )),
            Err(Error::Io(std::io::ErrorKind::PermissionDenied))
        ));
        assert!(matches!(
            read_exact_at(&mut output, 0, |_, _| Ok(8)),
            Err(Error::InvalidFormat(_))
        ));
        assert!(matches!(
            read_exact_at(&mut output, u64::MAX, |_, _| unreachable!()),
            Err(Error::ArithmeticOverflow)
        ));
    }
}
