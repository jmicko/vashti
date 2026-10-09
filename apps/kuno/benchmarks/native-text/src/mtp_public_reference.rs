//! Compiled-public DAS target-conditioned original-Q8 MTP oracle.
//! Four teacher-forced rows and two provisional draft rows have distinct wire
//! modes. Full vectors, not draft tokens alone, are the numerical acceptance.
//! No private-request persistence, speculative acceptance or performance claim.
//!
//! Wire:320-byte header, then six records of eight little-endian u32 words
//! (mode,position,input,greedy,completed cursor,preceding position,V,H), followed
//! by V logits and H wide hidden values as little-endian F32. Teacher records
//! use input_token(p), zero HC at0 and target HC(p-1). Draft records are p1 with
//! target HC0 after p0 KV catch-up, then p2 with draft greedy1 and draft HC1.
//! Fixed header words pin version/counts/context/V/H/HC/D/main layers/MTP
//! layers/first MTP layer/tensor count/Q8 ID/token0/seed/mode mask/threads/record
//! words/Q heads/KV heads/head width/RoPE width/MTP context limit/experts/topk/
//! expert FF/HC low rank/KV width/main GGUF version/sidecar version/reserved.
//! Remaining bytes pin original source extents and revisions, sidecar SHA,
//! both DAS shard SHAs and the actual target-carry fixture SHA. Source digest
//! verification remains the protected runner's gate before this fixed writer.

use kuno_text::model::greedy;
use kuno_text::{Error, Result};
use std::io::{Read, Write};
use std::path::Path;
use zeroize::{Zeroize, Zeroizing};

pub const TEACHER_ROWS: usize = 4;
pub const DRAFT_ROWS: usize = 2;
pub const ROWS: usize = TEACHER_ROWS + DRAFT_ROWS;
pub const CONTEXT: usize = 4;
pub const VOCAB: usize = 248_320;
pub const HIDDEN: usize = 10_240;
pub const THREADS: usize = 16;
pub const SIDECAR_BYTES: u64 = 4_137_429_120;
pub const SIDECAR_REVISION: &str = "766911a6b7369840a91dbcd95f9f997acaab6cd6";
pub const SIDECAR_SHA256: &str = "58a47d6a216307e9b97172d5bfdc3436a8994b632f5e436b9bb6edf8951da8d1";
pub const TARGET_CARRY_SHA256: &str =
    "18788aee8ccb1defdbba624df3ae914c3cf574ab719c3d2a5f375e4048c67c20";
pub const ABS_TOLERANCE: f32 = 0.006;
pub const REL_TOLERANCE: f32 = 0.0003;
pub const HEADER_BYTES: usize = 320;
pub const ROW_BYTES: usize = 32 + 4 * (VOCAB + HIDDEN);
pub const FILE_BYTES: usize = HEADER_BYTES + ROWS * ROW_BYTES;
pub const OWNER_HOST_CAP: usize = 8 * 1024 * 1024;
pub const COUPLED_HOST_CAP: usize = 32 * 1024 * 1024;
const STREAM_BYTES: usize = 64 * 1024;
const MAGIC: &[u8; 8] = b"KMTP0001";
const HEADER_FIELDS: [u32; 32] = [
    1, 6, 4, 2, 4, 248320, 10240, 4, 2560, 48, 1, 48, 34, 8, 0, 424242, 7, 16, 8, 24, 2, 256, 64,
    2048, 512, 10, 640, 320, 512, 3, 3, 0,
];

fn err() -> Error {
    Error::InvalidFormat("fixed public MTP oracle rejected")
}
fn pin<const N: usize>(text: &str) -> Result<[u8; N]> {
    if text.len() != 2 * N {
        return Err(err());
    }
    let nibble = |b| match b {
        b'0'..=b'9' => Ok(b - b'0'),
        b'a'..=b'f' => Ok(b - b'a' + 10),
        _ => Err(err()),
    };
    let mut out = [0; N];
    for (i, pair) in text.as_bytes().chunks_exact(2).enumerate() {
        out[i] = nibble(pair[0])? * 16 + nibble(pair[1])?;
    }
    Ok(out)
}
fn header() -> Result<[u8; HEADER_BYTES]> {
    use crate::daslab_carry_reference as target;
    let mut out = [0; HEADER_BYTES];
    out[..8].copy_from_slice(MAGIC);
    for (i, value) in HEADER_FIELDS.into_iter().enumerate() {
        out[8 + i * 4..12 + i * 4].copy_from_slice(&value.to_le_bytes());
    }
    out[136..144].copy_from_slice(&SIDECAR_BYTES.to_le_bytes());
    out[144..152].copy_from_slice(&target::MODEL_BYTES.to_le_bytes());
    out[152..172].copy_from_slice(&pin::<20>(target::REVISION)?);
    out[172..192].copy_from_slice(&pin::<20>(SIDECAR_REVISION)?);
    out[192..224].copy_from_slice(&pin::<32>(SIDECAR_SHA256)?);
    out[224..256].copy_from_slice(&pin::<32>(target::SHARD_SHA256[0])?);
    out[256..288].copy_from_slice(&pin::<32>(target::SHARD_SHA256[1])?);
    // This digest also pins the source carry header's graph base and geometry.
    // Protected runner verifies actual payload hashes; this module does not
    // pretend a header digest alone verifies a file's content.
    out[288..320].copy_from_slice(&pin::<32>(TARGET_CARRY_SHA256)?);
    Ok(out)
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Conditioning {
    /// At0 use defined zero preceding HC; atp>0 use real target HC(p-1).
    TeacherForced,
    /// Pos1 consumes target input1 + real target HC0 after pos0 KV catch-up.
    ProvisionalTarget,
    /// Pos2 consumes provisional greedy1 + preceding MTP HC1.
    ProvisionalDraft,
}
impl Conditioning {
    fn wire(self) -> u32 {
        match self {
            Self::TeacherForced => 1,
            Self::ProvisionalTarget => 2,
            Self::ProvisionalDraft => 3,
        }
    }
    fn from_wire(value: u32) -> Result<Self> {
        match value {
            1 => Ok(Self::TeacherForced),
            2 => Ok(Self::ProvisionalTarget),
            3 => Ok(Self::ProvisionalDraft),
            _ => Err(err()),
        }
    }
}

/// Immutable full numeric row; metadata and both payload capacities wipe on
/// drop. Binding identities are deliberately process-local and never serialized.
pub struct Row {
    metadata: Zeroizing<[u32; 6]>,
    logits: Zeroizing<Vec<f32>>,
    hidden: Zeroizing<Vec<f32>>,
}
impl Row {
    pub fn new(
        mode: Conditioning,
        position: usize,
        input: u32,
        cursor: usize,
        logits: Zeroizing<Vec<f32>>,
        hidden: Zeroizing<Vec<f32>>,
    ) -> Result<Self> {
        if position >= CONTEXT
            || cursor != position + 1
            || input as usize >= VOCAB
            || logits.len() != VOCAB
            || hidden.len() != HIDDEN
            || logits.capacity() != VOCAB
            || hidden.capacity() != HIDDEN
            || logits.iter().chain(hidden.iter()).any(|v| !v.is_finite())
            || !logits.iter().any(|&v| v != 0.0)
            || !hidden.iter().any(|&v| v != 0.0)
        {
            return Err(err());
        }
        let token = greedy(&logits)?;
        let preceding = if position == 0 {
            u32::MAX
        } else {
            (position - 1) as u32
        };
        Ok(Self {
            metadata: Zeroizing::new([
                mode.wire(),
                position as u32,
                input,
                token,
                cursor as u32,
                preceding,
            ]),
            logits,
            hidden,
        })
    }
    pub fn conditioning(&self) -> Conditioning {
        Conditioning::from_wire(self.metadata[0]).expect("validated MTP row mode")
    }
    pub fn position(&self) -> usize {
        self.metadata[1] as usize
    }
    pub fn input_token(&self) -> u32 {
        self.metadata[2]
    }
    pub fn token(&self) -> u32 {
        self.metadata[3]
    }
    pub fn completed_cursor(&self) -> usize {
        self.metadata[4] as usize
    }
    pub fn logits(&self) -> &[f32] {
        &self.logits
    }
    pub fn hidden(&self) -> &[f32] {
        &self.hidden
    }
    fn owned_bytes(&self) -> usize {
        (self.logits.capacity() + self.hidden.capacity()) * 4
    }
}

pub struct Reference {
    rows: Vec<Row>,
}
pub struct Report {
    pub logits_max_abs_error: f64,
    pub hidden_max_abs_error: f64,
}
pub struct Agreements {
    pub teacher_forced: u32,
    pub provisional_first: u32,
    pub provisional_second: u32,
}
impl Reference {
    pub fn new(rows: Vec<Row>, target: &crate::daslab_carry_reference::Reference) -> Result<Self> {
        if rows.len() != ROWS || rows.capacity() != ROWS {
            return Err(err());
        }
        for (index, row) in rows.iter().enumerate() {
            let (mode, position, input) = if index < TEACHER_ROWS {
                (
                    Conditioning::TeacherForced,
                    index,
                    target.input_token(index)?,
                )
            } else if index == TEACHER_ROWS {
                (Conditioning::ProvisionalTarget, 1, target.input_token(1)?)
            } else {
                (
                    Conditioning::ProvisionalDraft,
                    2,
                    rows[TEACHER_ROWS].token(),
                )
            };
            if row.conditioning() != mode
                || row.position() != position
                || row.input_token() != input
                || row.completed_cursor() != position + 1
                || row.metadata[5]
                    != if position == 0 {
                        u32::MAX
                    } else {
                        (position - 1) as u32
                    }
            {
                return Err(err());
            }
        }
        let result = Self { rows };
        if result.owned_bytes() > OWNER_HOST_CAP
            || result
                .owned_bytes()
                .checked_add(target.owned_bytes()?)
                .and_then(|n| n.checked_add(STREAM_BYTES + HEADER_BYTES))
                .is_none_or(|n| n > COUPLED_HOST_CAP)
        {
            return Err(err());
        }
        // append_kv catch-up and step have identical KV: the first provisional
        // input/cache is the same as teacher-forced row1, so every output bit is.
        result.verify_bits(
            Conditioning::TeacherForced,
            1,
            result.provisional(0)?.logits(),
            result.provisional(0)?.hidden(),
        )?;
        Ok(result)
    }
    pub fn owned_bytes(&self) -> usize {
        std::mem::size_of::<Self>()
            + self.rows.capacity() * std::mem::size_of::<Row>()
            + self.rows.iter().map(Row::owned_bytes).sum::<usize>()
    }
    pub fn teacher_forced(&self, position: usize) -> Result<&Row> {
        if position >= TEACHER_ROWS {
            return Err(Error::OutOfBounds);
        }
        self.rows.get(position).ok_or_else(err)
    }
    pub fn provisional(&self, index: usize) -> Result<&Row> {
        if index >= DRAFT_ROWS {
            return Err(Error::OutOfBounds);
        }
        self.rows.get(TEACHER_ROWS + index).ok_or_else(err)
    }
    fn selected(&self, mode: Conditioning, index: usize) -> Result<&Row> {
        let row = if mode == Conditioning::TeacherForced {
            self.teacher_forced(index)?
        } else {
            self.provisional(index)?
        };
        if row.conditioning() != mode {
            return Err(err());
        }
        Ok(row)
    }
    #[allow(clippy::too_many_arguments)]
    pub fn verify(
        &self,
        mode: Conditioning,
        index: usize,
        input: u32,
        cursor: usize,
        logits: &[f32],
        hidden: &[f32],
    ) -> Result<Report> {
        let row = self.selected(mode, index)?;
        if input != row.input_token()
            || cursor != row.completed_cursor()
            || greedy(logits)? != row.token()
        {
            return Err(err());
        }
        Ok(Report {
            logits_max_abs_error: compare(logits, row.logits())?,
            hidden_max_abs_error: compare(hidden, row.hidden())?,
        })
    }
    pub fn verify_bits(
        &self,
        mode: Conditioning,
        index: usize,
        logits: &[f32],
        hidden: &[f32],
    ) -> Result<()> {
        let row = self.selected(mode, index)?;
        for (actual, expected) in [(logits, row.logits()), (hidden, row.hidden())] {
            if actual.len() != expected.len()
                || actual
                    .iter()
                    .zip(expected)
                    .any(|(a, b)| a.to_bits() != b.to_bits())
            {
                return Err(err());
            }
        }
        Ok(())
    }
    pub fn agreements(
        &self,
        target: &crate::daslab_carry_reference::Reference,
    ) -> Result<Agreements> {
        let mut teacher_forced = 0;
        for p in 0..TEACHER_ROWS {
            teacher_forced += u32::from(self.teacher_forced(p)?.token() == target.token(p)?);
        }
        Ok(Agreements {
            teacher_forced,
            provisional_first: u32::from(self.provisional(0)?.token() == target.token(1)?),
            provisional_second: u32::from(self.provisional(1)?.token() == target.token(2)?),
        })
    }
    pub fn load(
        path: impl AsRef<Path>,
        target: &crate::daslab_carry_reference::Reference,
    ) -> Result<Self> {
        let mut file = std::fs::File::open(path)?;
        if file.metadata()?.len() != FILE_BYTES as u64 {
            return Err(err());
        }
        Self::read(&mut file, target)
    }
    fn read(
        reader: &mut impl Read,
        target: &crate::daslab_carry_reference::Reference,
    ) -> Result<Self> {
        let mut actual = [0; HEADER_BYTES];
        reader.read_exact(&mut actual)?;
        if actual != header()? {
            return Err(err());
        }
        let mut rows = Vec::new();
        rows.try_reserve_exact(ROWS).map_err(|_| err())?;
        if rows.capacity() != ROWS {
            return Err(err());
        }
        for _ in 0..ROWS {
            let mut fields = Zeroizing::new([0u32; 8]);
            for value in fields.iter_mut() {
                let mut bytes = [0; 4];
                reader.read_exact(&mut bytes)?;
                *value = u32::from_le_bytes(bytes);
            }
            if fields[6] as usize != VOCAB || fields[7] as usize != HIDDEN {
                return Err(err());
            }
            let mode = Conditioning::from_wire(fields[0])?;
            let row = Row::new(
                mode,
                fields[1] as usize,
                fields[2],
                fields[4] as usize,
                values(reader, VOCAB)?,
                values(reader, HIDDEN)?,
            )?;
            if row.metadata.as_slice() != &fields[..6] {
                return Err(err());
            }
            rows.push(row);
        }
        if reader.read(&mut [0; 1])? != 0 {
            return Err(err());
        }
        Self::new(rows, target)
    }
    pub fn write_new(&self, path: impl AsRef<Path>) -> Result<()> {
        // No file exists before full source, CPU scheduling and replay gates.
        let mut file = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(path)?;
        self.write(&mut file)?;
        file.flush()?;
        file.sync_all()?;
        Ok(())
    }
    fn write(&self, writer: &mut impl Write) -> Result<()> {
        writer.write_all(&header()?)?;
        for row in &self.rows {
            for value in row
                .metadata
                .iter()
                .copied()
                .chain([VOCAB as u32, HIDDEN as u32])
            {
                writer.write_all(&value.to_le_bytes())?;
            }
            write_values(writer, row.logits())?;
            write_values(writer, row.hidden())?;
        }
        Ok(())
    }
}
fn compare(actual: &[f32], expected: &[f32]) -> Result<f64> {
    if actual.len() != expected.len() {
        return Err(err());
    }
    let mut maximum = 0f64;
    for (&a, &b) in actual.iter().zip(expected) {
        if !a.is_finite()
            || !b.is_finite()
            || (a - b).abs() > ABS_TOLERANCE + REL_TOLERANCE * b.abs()
        {
            return Err(err());
        }
        maximum = maximum.max((f64::from(a) - f64::from(b)).abs());
    }
    Ok(maximum)
}
fn scratch() -> Result<Zeroizing<Vec<u8>>> {
    let mut out = Zeroizing::new(Vec::new());
    out.try_reserve_exact(STREAM_BYTES).map_err(|_| err())?;
    if out.capacity() != STREAM_BYTES {
        return Err(err());
    }
    out.resize(STREAM_BYTES, 0);
    Ok(out)
}
fn values(reader: &mut impl Read, count: usize) -> Result<Zeroizing<Vec<f32>>> {
    if !matches!(count, VOCAB | HIDDEN) {
        return Err(err());
    }
    let mut out = Zeroizing::new(Vec::new());
    out.try_reserve_exact(count).map_err(|_| err())?;
    if out.capacity() != count {
        return Err(err());
    }
    let mut bytes = scratch()?;
    while out.len() < count {
        let n = (count - out.len()).min(STREAM_BYTES / 4);
        reader.read_exact(&mut bytes[..n * 4])?;
        for word in bytes[..n * 4].chunks_exact(4) {
            let v = f32::from_le_bytes(word.try_into().map_err(|_| err())?);
            if !v.is_finite() {
                return Err(err());
            }
            out.push(v);
        }
        bytes.as_mut_slice().zeroize();
    }
    Ok(out)
}
fn write_values(writer: &mut impl Write, values: &[f32]) -> Result<()> {
    let mut bytes = scratch()?;
    for chunk in values.chunks(STREAM_BYTES / 4) {
        for (word, value) in bytes.chunks_exact_mut(4).zip(chunk) {
            word.copy_from_slice(&value.to_le_bytes());
        }
        writer.write_all(&bytes[..chunk.len() * 4])?;
        bytes.as_mut_slice().zeroize();
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    fn floats(count: usize, value: f32) -> Zeroizing<Vec<f32>> {
        Zeroizing::new(vec![value; count])
    }
    fn target() -> crate::daslab_carry_reference::Reference {
        use crate::daslab_carry_reference as target;
        let mut rows = Vec::with_capacity(TEACHER_ROWS);
        for p in 0..TEACHER_ROWS {
            let mut logits = floats(VOCAB, 0.25);
            logits[p + 1] = 3.0;
            let hidden = floats(HIDDEN, p as f32 + 1.0);
            rows.push(target::Row::new(p, p as u32, &logits, &hidden).unwrap());
        }
        target::Reference::new(rows).unwrap()
    }
    fn rows() -> Vec<Row> {
        let mut rows = Vec::with_capacity(ROWS);
        for p in 0..TEACHER_ROWS {
            let mut logits = floats(VOCAB, 0.5);
            logits[if p == 1 { 6 } else { p + 1 }] = 5.0;
            rows.push(
                Row::new(
                    Conditioning::TeacherForced,
                    p,
                    p as u32,
                    p + 1,
                    logits,
                    floats(HIDDEN, p as f32 + 0.75),
                )
                .unwrap(),
            );
        }
        let first = &rows[1];
        let mut logits = floats(VOCAB, 0.0);
        logits.copy_from_slice(first.logits());
        let mut hidden = floats(HIDDEN, 0.0);
        hidden.copy_from_slice(first.hidden());
        rows.push(Row::new(Conditioning::ProvisionalTarget, 1, 1, 2, logits, hidden).unwrap());
        let mut logits = floats(VOCAB, 0.75);
        logits[99] = 5.0;
        rows.push(
            Row::new(
                Conditioning::ProvisionalDraft,
                2,
                6,
                3,
                logits,
                floats(HIDDEN, 2.25),
            )
            .unwrap(),
        );
        rows
    }
    fn wire(reference: &Reference) -> Zeroizing<Vec<u8>> {
        let mut bytes = Zeroizing::new(Vec::with_capacity(FILE_BYTES));
        reference.write(&mut *bytes).unwrap();
        assert_eq!(bytes.len(), FILE_BYTES);
        assert_eq!(bytes.capacity(), FILE_BYTES);
        bytes
    }
    fn word(bytes: &mut [u8], offset: usize, value: u32) {
        bytes[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
    }
    #[test]
    fn header_and_roundtrip_preserve_distinct_conditioning_and_full_vectors() {
        let target = target();
        let original = Reference::new(rows(), &target).unwrap();
        let bytes = wire(&original);
        let loaded = Reference::read(&mut Cursor::new(&bytes[..]), &target).unwrap();
        assert_eq!(FILE_BYTES, 6_205_952);
        assert_eq!(&bytes[..8], b"KMTP0001");
        assert_eq!(&bytes[288..320], &pin::<32>(TARGET_CARRY_SHA256).unwrap());
        assert!(original.owned_bytes() < OWNER_HOST_CAP);
        for p in 0..TEACHER_ROWS {
            let row = loaded.teacher_forced(p).unwrap();
            assert_eq!(row.position(), p);
            assert_eq!(row.input_token(), target.input_token(p).unwrap());
            assert_ne!(row.input_token(), target.token(p).unwrap());
            original
                .verify_bits(Conditioning::TeacherForced, p, row.logits(), row.hidden())
                .unwrap();
        }
        assert_eq!(loaded.provisional(1).unwrap().input_token(), 6);
        assert_ne!(
            loaded.provisional(1).unwrap().input_token(),
            target.input_token(2).unwrap()
        );
        let agreements = loaded.agreements(&target).unwrap();
        assert_eq!(agreements.teacher_forced, 3);
        assert_eq!(agreements.provisional_first, 0);
        assert_eq!(agreements.provisional_second, 0);
        assert!(loaded.teacher_forced(4).is_err());
        assert!(loaded.provisional(2).is_err());
    }
    #[test]
    fn source_geometry_and_record_metadata_corruption_fail_closed() {
        let target = target();
        let reference = Reference::new(rows(), &target).unwrap();
        let mut bytes = wire(&reference);
        for offset in [0, 8, 24, 56, 136, 144, 152, 172, 192, 224, 256, 288] {
            bytes[offset] ^= 1;
            assert!(Reference::read(&mut Cursor::new(&bytes[..]), &target).is_err());
            bytes[offset] ^= 1;
        }
        // Exact metadata must match the fixed target inputs and mode ordering.
        for (record, index, value) in [
            (0, 0, 2),
            (0, 1, 4),
            (0, 2, 1),
            (0, 3, 99),
            (0, 4, 2),
            (0, 5, 0),
            (0, 6, (VOCAB - 1) as u32),
            (0, 7, (HIDDEN - 1) as u32),
            (4, 0, 3),
            (5, 2, 2),
        ] {
            let offset = HEADER_BYTES + record * ROW_BYTES + index * 4;
            let previous: [u8; 4] = bytes[offset..offset + 4].try_into().unwrap();
            word(&mut bytes, offset, value);
            assert!(Reference::read(&mut Cursor::new(&bytes[..]), &target).is_err());
            bytes[offset..offset + 4].copy_from_slice(&previous);
        }
        for n in [0, HEADER_BYTES - 1, HEADER_BYTES + 31, FILE_BYTES - 1] {
            assert!(Reference::read(&mut Cursor::new(&bytes[..n]), &target).is_err());
        }
        // Stream exact EOF without making a second full numeric payload copy.
        let mut extra = Cursor::new(&bytes[..]).chain(Cursor::new([0u8]));
        assert!(Reference::read(&mut extra, &target).is_err());
    }
    #[test]
    fn full_tail_nonfinite_and_catchup_mismatch_are_rejected() {
        let target = target();
        let reference = Reference::new(rows(), &target).unwrap();
        let mut bytes = wire(&reference);
        for offset in [HEADER_BYTES + 32, FILE_BYTES - 4] {
            let original: [u8; 4] = bytes[offset..offset + 4].try_into().unwrap();
            for invalid in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
                word(&mut bytes, offset, invalid.to_bits());
                assert!(Reference::read(&mut Cursor::new(&bytes[..]), &target).is_err());
            }
            bytes[offset..offset + 4].copy_from_slice(&original);
        }
        let mut bad = rows();
        bad[TEACHER_ROWS].hidden[HIDDEN - 1] += 0.00001;
        assert!(Reference::new(bad, &target).is_err());
        assert!(
            Row::new(
                Conditioning::TeacherForced,
                0,
                0,
                1,
                floats(VOCAB, 0.0),
                floats(HIDDEN, 1.0)
            )
            .is_err()
        );
        assert!(
            Row::new(
                Conditioning::TeacherForced,
                0,
                0,
                1,
                floats(VOCAB, 1.0),
                floats(HIDDEN, 0.0)
            )
            .is_err()
        );
    }
    #[test]
    fn accepted_reference_checks_actual_input_cursor_greedy_and_complete_tails() {
        let target = target();
        let reference = Reference::new(rows(), &target).unwrap();
        let row = reference.teacher_forced(1).unwrap();
        assert!(
            reference
                .verify(
                    Conditioning::TeacherForced,
                    1,
                    1,
                    2,
                    row.logits(),
                    row.hidden()
                )
                .is_ok()
        );
        assert!(
            reference
                .verify(
                    Conditioning::TeacherForced,
                    1,
                    target.token(1).unwrap(),
                    2,
                    row.logits(),
                    row.hidden()
                )
                .is_err()
        );
        assert!(
            reference
                .verify(
                    Conditioning::TeacherForced,
                    1,
                    1,
                    3,
                    row.logits(),
                    row.hidden()
                )
                .is_err()
        );
        assert!(
            reference
                .verify(
                    Conditioning::ProvisionalDraft,
                    0,
                    1,
                    2,
                    row.logits(),
                    row.hidden()
                )
                .is_err()
        );
        let mut logits = floats(VOCAB, 0.0);
        logits.copy_from_slice(row.logits());
        logits[VOCAB - 1] += 0.02;
        assert!(
            reference
                .verify(Conditioning::TeacherForced, 1, 1, 2, &logits, row.hidden())
                .is_err()
        );
        logits.copy_from_slice(row.logits());
        logits[99] = 5.1;
        assert!(
            reference
                .verify(Conditioning::TeacherForced, 1, 1, 2, &logits, row.hidden())
                .is_err()
        );
        let mut hidden = floats(HIDDEN, 0.0);
        hidden.copy_from_slice(row.hidden());
        hidden[HIDDEN - 1] += 0.0001;
        assert!(
            reference
                .verify(Conditioning::TeacherForced, 1, 1, 2, row.logits(), &hidden)
                .is_ok()
        );
        assert!(
            reference
                .verify_bits(Conditioning::TeacherForced, 1, row.logits(), &hidden)
                .is_err()
        );
        hidden[HIDDEN - 1] += 0.02;
        assert!(
            reference
                .verify(Conditioning::TeacherForced, 1, 1, 2, row.logits(), &hidden)
                .is_err()
        );
        assert!(
            reference
                .verify(
                    Conditioning::TeacherForced,
                    1,
                    1,
                    2,
                    &row.logits()[..VOCAB - 1],
                    row.hidden()
                )
                .is_err()
        );
    }
    #[test]
    fn output_publication_never_overwrites_an_existing_file() {
        let target = target();
        let reference = Reference::new(rows(), &target).unwrap();
        let path =
            std::env::temp_dir().join(format!("kuno-mtp-fixed-reference-{}", std::process::id()));
        let _ = std::fs::remove_file(&path);
        reference.write_new(&path).unwrap();
        assert!(Reference::load(&path, &target).is_ok());
        assert!(reference.write_new(&path).is_err());
        std::fs::remove_file(path).unwrap();
    }
}
