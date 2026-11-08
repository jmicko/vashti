//! Fixed thirteen-position public DASLab logits and pre-output-HC residual.
//! This is a separate format: the historical four-row fixture is unchanged.
//! No prompt parser, process binding ID or private-request persistence path.
//! The protected runner independently verifies complete original artifact and
//! executable hashes; the fixed header records that public provenance.

use kuno_text::model::greedy;
use kuno_text::{Error, Result};
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::Path;
use zeroize::{Zeroize, Zeroizing};

pub const POSITIONS: usize = 13;
pub const CONTEXT: usize = 13;
pub const VOCAB: usize = 248_320;
pub const HIDDEN: usize = 10_240;
pub const MAX_CHUNK: usize = 3;
pub const SEED: u32 = 424_242;
pub const MODEL_BYTES: u64 = 83_617_662_656;
pub const REVISION: &str = "ed59f92082b1e93c0e96d60a8b11aab089b52f09";
pub const GRAPH_BASE: &str = "6a32b1156ca645dbbf425b4edbc3b29917b443c5";
/// Protected runner verifies this independent complete32-row CPU file before
/// launch; the module then verifies its entire numeric shape/chain and bitwise
/// parity with every newly computed target logit row0..12.
pub const LEGACY_SHA256: &str = "635385075cd6a46241c4451726b5a3f95d40bb98e2f6e96e89e2bfbfd5db45d6";
pub const LEGACY_FILE_BYTES: usize = 8 + 32 * (16 + VOCAB * 4);
pub const SHARD_BYTES: [u64; 2] = [54_817_524_224, 28_800_138_432];
pub const SHARD_SHA256: [&str; 2] = [
    "4c1eb2ceb4915e1192f4f386021897bde56a97f40a0bb78bb86465e0f7d2aca3",
    "316b46f3a2dbd68c900f43136ab9449f9dcc3725dfd8c794847c204bc161e113",
];
// New full-target carry gate, selected before device testing. It deliberately
// matches the established whole-MTP output bound, not the tighter projection
// bound. Do not relax it after a failed physical run.
pub const ABS_TOLERANCE: f32 = 0.006;
pub const REL_TOLERANCE: f32 = 0.0003;
const MAGIC: &[u8; 8] = b"KDHC0013";
const HEADER_FIELDS: [u32; 10] = [13, 13, 248_320, 10_240, 4, 2560, 48, 3, 0, SEED];
pub const HEADER_BYTES: usize = 160;
pub const ROW_BYTES: usize = 24 + 4 * (VOCAB + HIDDEN);
pub const FILE_BYTES: usize = HEADER_BYTES + POSITIONS * ROW_BYTES;
pub const OWNER_HOST_CAP: usize = 16 * 1024 * 1024;
pub const LOAD_HOST_CAP: usize = 32 * 1024 * 1024;
const STREAM_BYTES: usize = 64 * 1024;
const LEGACY_ROW_BYTES: usize = 16 + VOCAB * 4;

/// Public fixture metadata/numeric owners only; original graph workspaces and
/// model residency remain separately bounded by the protected CPU runner.
/// No full raw file is retained: all readers/writers stream through 64KiB.
pub fn writer_fixture_peak_bytes() -> Result<usize> {
    reference_owner_bytes()?
        .checked_add(legacy_owner_bytes()?)
        .and_then(|n| n.checked_add(4 * (VOCAB + HIDDEN)))
        .and_then(|n| n.checked_add(STREAM_BYTES + HEADER_BYTES))
        .ok_or(Error::ArithmeticOverflow)
        .and_then(|n| {
            if n > LOAD_HOST_CAP {
                Err(Error::LimitExceeded("public DASLab carry13 writer peak"))
            } else {
                Ok(n)
            }
        })
}
fn reference_owner_bytes() -> Result<usize> {
    let bytes = std::mem::size_of::<Reference>()
        .checked_add(POSITIONS * std::mem::size_of::<Row>())
        .and_then(|n| n.checked_add(POSITIONS * 4 * (VOCAB + HIDDEN)))
        .ok_or(Error::ArithmeticOverflow)?;
    if bytes > OWNER_HOST_CAP {
        return Err(Error::LimitExceeded("public DASLab carry13 complete owner"));
    }
    Ok(bytes)
}
fn legacy_owner_bytes() -> Result<usize> {
    std::mem::size_of::<LegacyLogits>()
        .checked_add(POSITIONS * std::mem::size_of::<Zeroizing<Vec<f32>>>())
        .and_then(|n| n.checked_add(POSITIONS * VOCAB * 4))
        .ok_or(Error::ArithmeticOverflow)
}

fn err() -> Error {
    Error::InvalidFormat("fixed public thirteen-position DASLab carry fixture rejected")
}

fn pin<const N: usize>(value: &str) -> Result<[u8; N]> {
    if value.len() != N * 2 {
        return Err(err());
    }
    let nibble = |v| match v {
        b'0'..=b'9' => Ok(v - b'0'),
        b'a'..=b'f' => Ok(v - b'a' + 10),
        _ => Err(err()),
    };
    let mut out = [0; N];
    for (i, pair) in value.as_bytes().chunks_exact(2).enumerate() {
        out[i] = nibble(pair[0])? * 16 + nibble(pair[1])?;
    }
    Ok(out)
}

fn header() -> Result<[u8; HEADER_BYTES]> {
    let mut out = [0; HEADER_BYTES];
    out[..8].copy_from_slice(MAGIC);
    for (i, field) in HEADER_FIELDS.into_iter().enumerate() {
        out[8 + i * 4..12 + i * 4].copy_from_slice(&field.to_le_bytes());
    }
    out[48..56].copy_from_slice(&MODEL_BYTES.to_le_bytes());
    out[56..76].copy_from_slice(&pin::<20>(REVISION)?);
    out[76..108].copy_from_slice(&pin::<32>(SHARD_SHA256[0])?);
    out[108..140].copy_from_slice(&pin::<32>(SHARD_SHA256[1])?);
    out[140..160].copy_from_slice(&pin::<20>(GRAPH_BASE)?);
    Ok(out)
}

fn read_u32(reader: &mut impl Read) -> Result<u32> {
    let mut word = [0; 4];
    reader.read_exact(&mut word)?;
    Ok(u32::from_le_bytes(word))
}

fn values(reader: &mut impl Read, count: usize) -> Result<Zeroizing<Vec<f32>>> {
    if !matches!(count, VOCAB | HIDDEN) {
        return Err(err());
    }
    let mut out = Zeroizing::new(Vec::new());
    out.try_reserve_exact(count)
        .map_err(|_| Error::LimitExceeded("public DASLab numeric owner"))?;
    if out.capacity() != count {
        return Err(err());
    }
    for _ in 0..count {
        let value = f32::from_bits(read_u32(reader)?);
        if !value.is_finite() {
            return Err(err());
        }
        out.push(value);
    }
    Ok(out)
}

fn copy_values(input: &[f32], count: usize) -> Result<Zeroizing<Vec<f32>>> {
    if input.len() != count || input.iter().any(|v| !v.is_finite()) {
        return Err(err());
    }
    let mut out = Zeroizing::new(Vec::new());
    out.try_reserve_exact(count)
        .map_err(|_| Error::LimitExceeded("public DASLab numeric owner"))?;
    if out.capacity() != count {
        return Err(err());
    }
    out.extend_from_slice(input);
    Ok(out)
}

/// One accepted public sequential row. Carry is four contiguous unnormalized
/// 2560-wide streams after layer47 FFN combine and before output_hc RMS/mix.
/// No mutable payload, Debug, Clone or process binding metadata is exposed.
pub struct Row {
    position: u32,
    input: u32,
    token: u32,
    completed: u32,
    logits: Zeroizing<Vec<f32>>,
    hidden: Zeroizing<Vec<f32>>,
}
impl Row {
    pub fn new(position: usize, input: u32, logits: &[f32], hidden: &[f32]) -> Result<Self> {
        if position >= POSITIONS || input as usize >= VOCAB {
            return Err(err());
        }
        let logits = copy_values(logits, VOCAB)?;
        let hidden = copy_values(hidden, HIDDEN)?;
        let token = greedy(&logits)?;
        Ok(Self {
            position: position as u32,
            input,
            token,
            completed: position as u32 + 1,
            logits,
            hidden,
        })
    }
    fn owned_bytes(&self) -> Result<usize> {
        self.logits
            .capacity()
            .checked_add(self.hidden.capacity())
            .and_then(|n| n.checked_mul(4))
            .ok_or(Error::ArithmeticOverflow)
    }
}
impl Drop for Row {
    fn drop(&mut self) {
        self.position.zeroize();
        self.input.zeroize();
        self.token.zeroize();
        self.completed.zeroize();
    }
}

/// Complete numeric-only accepted comparison. No IDs or sampled values logged.
pub struct Report {
    pub logits_max_abs_error: f64,
    pub hidden_max_abs_error: f64,
}

pub struct Reference {
    rows: Vec<Row>,
}
impl Reference {
    pub fn new(rows: Vec<Row>) -> Result<Self> {
        reference_owner_bytes()?;
        if rows.len() != POSITIONS || rows.capacity() != POSITIONS {
            return Err(err());
        }
        let result = Self { rows };
        for (position, row) in result.rows.iter().enumerate() {
            if row.position as usize != position
                || row.completed as usize != position + 1
                || row.input
                    != if position == 0 {
                        0
                    } else {
                        result.rows[position - 1].token
                    }
                || row.logits.len() != VOCAB
                || row.hidden.len() != HIDDEN
                || row
                    .logits
                    .iter()
                    .chain(row.hidden.iter())
                    .any(|v| !v.is_finite())
                || greedy(&row.logits)? != row.token
            {
                return Err(err());
            }
        }
        if result.owned_bytes()? > OWNER_HOST_CAP {
            return Err(Error::LimitExceeded("public DASLab carry complete owner"));
        }
        Ok(result)
    }

    pub fn owned_bytes(&self) -> Result<usize> {
        self.rows.iter().try_fold(
            std::mem::size_of::<Self>() + self.rows.capacity() * std::mem::size_of::<Row>(),
            |n, row| {
                n.checked_add(row.owned_bytes()?)
                    .ok_or(Error::ArithmeticOverflow)
            },
        )
    }

    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let file = std::fs::File::open(path)?;
        if file.metadata()?.len() != FILE_BYTES as u64 {
            return Err(err());
        }
        let mut reader = BufReader::with_capacity(STREAM_BYTES, file);
        Self::read(&mut reader)
    }

    fn read(reader: &mut impl Read) -> Result<Self> {
        // Precharge the complete fixed capacity before any numeric allocation.
        if reference_owner_bytes()?
            .checked_add(STREAM_BYTES + HEADER_BYTES)
            .is_none_or(|n| n > LOAD_HOST_CAP)
        {
            return Err(Error::LimitExceeded("public DASLab carry13 load peak"));
        }
        let mut actual = [0; HEADER_BYTES];
        reader.read_exact(&mut actual)?;
        if actual != header()? {
            return Err(err());
        }
        let mut rows = Vec::new();
        rows.try_reserve_exact(POSITIONS)
            .map_err(|_| Error::LimitExceeded("public DASLab rows"))?;
        if rows.capacity() != POSITIONS {
            return Err(err());
        }
        for position in 0..POSITIONS {
            let fields = [
                read_u32(reader)?,
                read_u32(reader)?,
                read_u32(reader)?,
                read_u32(reader)?,
                read_u32(reader)?,
                read_u32(reader)?,
            ];
            if fields[0] as usize != position
                || fields[3] as usize != VOCAB
                || fields[4] as usize != HIDDEN
                || fields[5] as usize != position + 1
                || fields[1] as usize >= VOCAB
                || fields[2] as usize >= VOCAB
            {
                return Err(err());
            }
            rows.push(Row {
                position: fields[0],
                input: fields[1],
                token: fields[2],
                completed: fields[5],
                logits: values(reader, VOCAB)?,
                hidden: values(reader, HIDDEN)?,
            });
        }
        if reader.read(&mut [0; 1])? != 0 {
            return Err(err());
        }
        let result = Self::new(rows)?;
        if result
            .owned_bytes()?
            .checked_add(STREAM_BYTES + HEADER_BYTES)
            .is_none_or(|n| n > LOAD_HOST_CAP)
        {
            return Err(Error::LimitExceeded("public DASLab carry load peak"));
        }
        Ok(result)
    }

    pub fn write_new(&self, path: impl AsRef<Path>) -> Result<()> {
        if self
            .owned_bytes()?
            .checked_add(STREAM_BYTES + HEADER_BYTES)
            .is_none_or(|n| n > LOAD_HOST_CAP)
        {
            return Err(Error::LimitExceeded("public DASLab carry13 write peak"));
        }
        let file = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(path)?;
        let mut writer = BufWriter::with_capacity(STREAM_BYTES, file);
        self.write(&mut writer)?;
        writer.flush()?;
        Ok(())
    }
    fn write(&self, writer: &mut impl Write) -> Result<()> {
        writer.write_all(&header()?)?;
        for row in &self.rows {
            for value in [
                row.position,
                row.input,
                row.token,
                VOCAB as u32,
                HIDDEN as u32,
                row.completed,
            ] {
                writer.write_all(&value.to_le_bytes())?;
            }
            for value in row.logits.iter().chain(row.hidden.iter()) {
                writer.write_all(&value.to_le_bytes())?;
            }
        }
        Ok(())
    }
    fn row(&self, position: usize) -> Result<&Row> {
        self.rows.get(position).ok_or(Error::OutOfBounds)
    }
    pub fn input_token(&self, position: usize) -> Result<u32> {
        Ok(self.row(position)?.input)
    }
    pub fn token(&self, position: usize) -> Result<u32> {
        Ok(self.row(position)?.token)
    }
    pub fn logits(&self, position: usize) -> Result<&[f32]> {
        Ok(&self.row(position)?.logits)
    }
    pub fn hidden(&self, position: usize) -> Result<&[f32]> {
        Ok(&self.row(position)?.hidden)
    }

    /// A chunk's final cursor is checked once, not against each sequential row's
    /// original local cursor. Every count1..3 chunk must end within thirteen rows.
    pub fn verify_completed_prefix(&self, start: usize, count: usize, cursor: usize) -> Result<()> {
        if !(1..=MAX_CHUNK).contains(&count)
            || start
                .checked_add(count)
                .is_none_or(|end| end > POSITIONS || cursor != end)
        {
            return Err(err());
        }
        Ok(())
    }

    pub fn verify(&self, position: usize, logits: &[f32], hidden: &[f32]) -> Result<Report> {
        let row = self.row(position)?;
        if logits.len() != VOCAB || hidden.len() != HIDDEN || greedy(logits)? != row.token {
            return Err(err());
        }
        Ok(Report {
            logits_max_abs_error: compare(logits, &row.logits)?,
            hidden_max_abs_error: compare(hidden, &row.hidden)?,
        })
    }

    /// Fresh original CPU replay is a separate bitwise provenance assertion,
    /// including signed zero; it does not replace the fixed GPU numeric gate.
    pub fn verify_cpu_bits(&self, position: usize, logits: &[f32], hidden: &[f32]) -> Result<()> {
        let row = self.row(position)?;
        if logits.len() != VOCAB
            || hidden.len() != HIDDEN
            || logits
                .iter()
                .zip(row.logits.iter())
                .chain(hidden.iter().zip(row.hidden.iter()))
                .any(|(a, r)| a.to_bits() != r.to_bits())
        {
            return Err(err());
        }
        Ok(())
    }
}

fn compare(actual: &[f32], reference: &[f32]) -> Result<f64> {
    if actual.len() != reference.len() {
        return Err(err());
    }
    let mut maximum = 0f64;
    for (&actual, &reference) in actual.iter().zip(reference) {
        let limit = ABS_TOLERANCE + REL_TOLERANCE * reference.abs();
        if !actual.is_finite() || !reference.is_finite() || (actual - reference).abs() > limit {
            return Err(err());
        }
        maximum = maximum.max((f64::from(actual) - f64::from(reference)).abs());
    }
    Ok(maximum)
}

/// Bitwise CPU provenance gate against ONLY the complete existing32-row original
/// fixed-public KQ3R0001 file. Only first13 rows are retained; the suffix is still
/// streamed and validated entirely, including its chain/finiteness/greedy IDs.
pub struct LegacyLogits {
    first: Vec<Zeroizing<Vec<f32>>>,
}
impl LegacyLogits {
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let file = std::fs::File::open(path)?;
        let bytes = file.metadata()?.len();
        let positions = match bytes {
            n if n == (8 + 32 * LEGACY_ROW_BYTES) as u64 => 32,
            _ => return Err(err()),
        };
        let mut reader = BufReader::with_capacity(STREAM_BYTES, file);
        Self::read(&mut reader, positions)
    }
    fn read(reader: &mut impl Read, positions: usize) -> Result<Self> {
        if positions != 32 {
            return Err(err());
        }
        // Thirteen retained rows plus ONE streamed suffix row are the peak.
        if legacy_owner_bytes()?
            .checked_add(VOCAB * 4 + STREAM_BYTES + 8)
            .is_none_or(|n| n > OWNER_HOST_CAP)
        {
            return Err(Error::LimitExceeded("public DASLab carry13 legacy peak"));
        }
        let mut magic = [0; 8];
        reader.read_exact(&mut magic)?;
        if &magic != b"KQ3R0001" {
            return Err(err());
        }
        let mut first = Vec::new();
        first
            .try_reserve_exact(POSITIONS)
            .map_err(|_| Error::LimitExceeded("public CPU logits prefix"))?;
        if first.capacity() != POSITIONS {
            return Err(err());
        }
        let mut input = 0;
        for position in 0..positions {
            let fields = [
                read_u32(reader)?,
                read_u32(reader)?,
                read_u32(reader)?,
                read_u32(reader)?,
            ];
            if fields[0] as usize != position || fields[1] != input || fields[3] as usize != VOCAB {
                return Err(err());
            }
            let logits = values(reader, VOCAB)?;
            if fields[2] != greedy(&logits)? {
                return Err(err());
            }
            input = fields[2];
            if position < POSITIONS {
                first.push(logits);
            }
        }
        if reader.read(&mut [0; 1])? != 0 {
            return Err(err());
        }
        Ok(Self { first })
    }
    pub fn verify_cpu_bits(&self, position: usize, actual: &[f32]) -> Result<()> {
        let expected = self.first.get(position).ok_or(Error::OutOfBounds)?;
        if actual.len() != VOCAB
            || actual
                .iter()
                .zip(expected.iter())
                .any(|(a, r)| a.to_bits() != r.to_bits())
        {
            return Err(err());
        }
        Ok(())
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    fn fixture() -> Reference {
        let mut rows = Vec::with_capacity(POSITIONS);
        let mut input = 0;
        for position in 0..POSITIONS {
            let mut logits = vec![0.; VOCAB];
            logits[position + 1] = 2.;
            logits[VOCAB - 1] = -0.;
            let mut hidden = vec![position as f32; HIDDEN];
            hidden[HIDDEN - 1] = -0.;
            rows.push(Row::new(position, input, &logits, &hidden).unwrap());
            input = position as u32 + 1;
        }
        Reference::new(rows).unwrap()
    }
    fn encoded() -> Vec<u8> {
        let mut bytes = Vec::with_capacity(FILE_BYTES);
        fixture().write(&mut bytes).unwrap();
        bytes
    }
    fn legacy() -> Vec<u8> {
        let mut bytes = Vec::with_capacity(LEGACY_FILE_BYTES);
        bytes.extend_from_slice(b"KQ3R0001");
        for position in 0..32 {
            for value in [
                position as u32,
                position as u32,
                position as u32 + 1,
                VOCAB as u32,
            ] {
                bytes.extend_from_slice(&value.to_le_bytes());
            }
            for index in 0..VOCAB {
                let value: f32 = if index == position + 1 {
                    2.
                } else if index == VOCAB - 1 {
                    -0.
                } else {
                    0.
                };
                bytes.extend_from_slice(&value.to_le_bytes());
            }
        }
        bytes
    }
    #[test]
    fn thirteen_complete_rows_preserve_all_bits_and_source_geometry() {
        assert_eq!(POSITIONS, 13);
        assert_eq!(CONTEXT, 13);
        assert_eq!(HEADER_BYTES, 160);
        assert_eq!(FILE_BYTES, 13_445_592);
        assert_eq!(LEGACY_FILE_BYTES, 31_785_480);
        assert_eq!(SHARD_BYTES.iter().sum::<u64>(), MODEL_BYTES);
        let original = fixture();
        let mut bytes = Vec::new();
        original.write(&mut bytes).unwrap();
        assert_eq!(bytes.len(), FILE_BYTES);
        let parsed = Reference::read(&mut Cursor::new(&bytes)).unwrap();
        assert_eq!(
            parsed.owned_bytes().unwrap(),
            reference_owner_bytes().unwrap()
        );
        assert!(parsed.owned_bytes().unwrap() < OWNER_HOST_CAP);
        assert!(writer_fixture_peak_bytes().unwrap() < LOAD_HOST_CAP);
        assert_eq!(LOAD_HOST_CAP, 32 * 1024 * 1024);
        for position in 0..POSITIONS {
            assert_eq!(parsed.input_token(position).unwrap(), position as u32);
            assert_eq!(parsed.token(position).unwrap(), position as u32 + 1);
            parsed
                .verify_cpu_bits(
                    position,
                    original.logits(position).unwrap(),
                    original.hidden(position).unwrap(),
                )
                .unwrap();
            let report = parsed
                .verify(
                    position,
                    original.logits(position).unwrap(),
                    original.hidden(position).unwrap(),
                )
                .unwrap();
            assert_eq!(report.logits_max_abs_error, 0.);
            assert_eq!(report.hidden_max_abs_error, 0.);
            assert_eq!(
                parsed.hidden(position).unwrap()[HIDDEN - 1].to_bits(),
                (-0f32).to_bits()
            );
        }
        assert!(parsed.input_token(POSITIONS).is_err());
        assert!(parsed.hidden(POSITIONS).is_err());
    }
    #[test]
    fn old_format_every_header_field_source_pin_and_row_chain_are_rejected() {
        let bytes = encoded();
        for at in [
            0, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 56, 76, 108, 140,
        ] {
            let mut bad = bytes.clone();
            bad[at] ^= 1;
            assert!(
                Reference::read(&mut Cursor::new(bad)).is_err(),
                "offset{at}"
            );
        }
        let mut old = bytes.clone();
        old[..8].copy_from_slice(b"KDHC0001");
        assert!(Reference::read(&mut Cursor::new(old)).is_err());
        let mut wrong_context = bytes.clone();
        wrong_context[12..16].copy_from_slice(&32u32.to_le_bytes());
        assert!(Reference::read(&mut Cursor::new(wrong_context)).is_err());
        // Both numeric roles remain completely finite-gated in the parser,
        // including the final carry word after all thirteen complete rows.
        for at in [HEADER_BYTES + 24 + (VOCAB - 1) * 4, FILE_BYTES - 4] {
            let mut bad = bytes.clone();
            bad[at..at + 4].copy_from_slice(&f32::NAN.to_le_bytes());
            assert!(Reference::read(&mut Cursor::new(bad)).is_err());
        }
        for at in [0, 4, 8, 12, 16, 20] {
            let mut bad = bytes.clone();
            let offset = HEADER_BYTES + 12 * ROW_BYTES + at;
            bad[offset] ^= 1;
            assert!(
                Reference::read(&mut Cursor::new(bad)).is_err(),
                "last_row_field{at}"
            );
        }
        for cut in [
            0,
            HEADER_BYTES - 1,
            HEADER_BYTES,
            HEADER_BYTES + 4 * ROW_BYTES,
            FILE_BYTES - 1,
        ] {
            assert!(Reference::read(&mut Cursor::new(&bytes[..cut])).is_err());
        }
        let mut extra = bytes;
        extra.push(0);
        assert!(Reference::read(&mut Cursor::new(extra)).is_err());
    }
    #[test]
    fn all_thirteen_full_vectors_and_exact_fresh_cpu_bits_are_required() {
        let reference = fixture();
        assert_eq!(ABS_TOLERANCE.to_bits(), 0.006f32.to_bits());
        assert_eq!(REL_TOLERANCE.to_bits(), 0.0003f32.to_bits());
        for position in 0..POSITIONS {
            for carry in [false, true] {
                let mut logits = reference.logits(position).unwrap().to_vec();
                let mut hidden = reference.hidden(position).unwrap().to_vec();
                for bad in [0.02, f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
                    if carry {
                        hidden[HIDDEN - 1] = bad;
                    } else {
                        logits[VOCAB - 1] = bad;
                    }
                    assert!(reference.verify(position, &logits, &hidden).is_err());
                }
            }
        }
        let logits = reference.logits(12).unwrap();
        let hidden = reference.hidden(12).unwrap();
        assert!(reference.verify(12, &logits[..VOCAB - 1], hidden).is_err());
        assert!(reference.verify(12, logits, &hidden[..HIDDEN - 1]).is_err());
        let mut long = hidden.to_vec();
        long.push(0.);
        assert!(reference.verify(12, logits, &long).is_err());
        let mut changed = hidden.to_vec();
        changed[HIDDEN - 1] = 0.;
        assert!(reference.verify(12, logits, &changed).is_ok());
        assert!(reference.verify_cpu_bits(12, logits, &changed).is_err());
        let mut changed = logits.to_vec();
        changed[VOCAB - 1] = 0.;
        assert!(reference.verify_cpu_bits(12, &changed, hidden).is_err());
        changed[14] = 3.;
        assert!(reference.verify(12, &changed, hidden).is_err());
        assert!(compare(&[f32::MAX], &[-f32::MAX]).is_err());
        assert!(compare(&[0.005], &[0.]).is_ok());
        assert!(compare(&[0.007], &[0.]).is_err());
        let mut too_few = fixture().rows;
        too_few.pop();
        assert!(Reference::new(too_few).is_err());
        assert!(Row::new(13, 0, logits, hidden).is_err());
        assert!(Row::new(0, VOCAB as u32, logits, hidden).is_err());
    }
    #[test]
    fn completed_prefix_cursor_is_independent_and_bounded_through_position_twelve() {
        let reference = fixture();
        for start in 0..POSITIONS {
            for count in 1..=MAX_CHUNK {
                assert_eq!(
                    reference
                        .verify_completed_prefix(start, count, start + count)
                        .is_ok(),
                    start + count <= POSITIONS
                );
                assert!(
                    reference
                        .verify_completed_prefix(start, count, start)
                        .is_err()
                );
            }
        }
        for (start, count, cursor) in [
            (0, 0, 0),
            (0, 4, 4),
            (12, 2, 14),
            (13, 1, 14),
            (usize::MAX, 1, 0),
        ] {
            assert!(
                reference
                    .verify_completed_prefix(start, count, cursor)
                    .is_err()
            );
        }
    }
    #[test]
    fn independent_legacy_requires_complete32_and_every_suffix_value() {
        let bytes = legacy();
        let reference = fixture();
        assert!(LegacyLogits::read(&mut Cursor::new(&bytes), 4).is_err());
        assert!(LegacyLogits::read(&mut Cursor::new(&bytes), 13).is_err());
        let parsed = LegacyLogits::read(&mut Cursor::new(&bytes), 32).unwrap();
        assert_eq!(parsed.first.len(), POSITIONS);
        for position in 0..POSITIONS {
            parsed
                .verify_cpu_bits(position, reference.logits(position).unwrap())
                .unwrap();
        }
        let mut wrong = reference.logits(12).unwrap().to_vec();
        wrong[VOCAB - 1] = 0.;
        assert!(parsed.verify_cpu_bits(12, &wrong).is_err());
        assert!(parsed.verify_cpu_bits(13, &wrong).is_err());
        for offset in [
            8 + 31 * LEGACY_ROW_BYTES,
            8 + 31 * LEGACY_ROW_BYTES + 4,
            8 + 31 * LEGACY_ROW_BYTES + 8,
            8 + 31 * LEGACY_ROW_BYTES + 12,
        ] {
            let mut bad = bytes.clone();
            bad[offset] ^= 1;
            assert!(LegacyLogits::read(&mut Cursor::new(bad), 32).is_err());
        }
        let mut nonfinite = bytes.clone();
        let end = nonfinite.len();
        nonfinite[end - 4..].copy_from_slice(&f32::NAN.to_le_bytes());
        assert!(LegacyLogits::read(&mut Cursor::new(nonfinite), 32).is_err());
        assert!(LegacyLogits::read(&mut Cursor::new(&bytes[..bytes.len() - 1]), 32).is_err());
        let mut extra = bytes;
        extra.push(0);
        assert!(LegacyLogits::read(&mut Cursor::new(extra), 32).is_err());
    }
    #[test]
    fn fresh_output_cannot_replace_existing_reference() {
        static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        struct Remove(std::path::PathBuf);
        impl Drop for Remove {
            fn drop(&mut self) {
                let _ = std::fs::remove_file(&self.0);
            }
        }
        let path = std::env::temp_dir().join(format!(
            "kuno-fixed-public-carry13-{}-{}.bin",
            std::process::id(),
            NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
        ));
        let remove = Remove(path);
        let reference = fixture();
        reference.write_new(&remove.0).unwrap();
        assert_eq!(
            std::fs::metadata(&remove.0).unwrap().len(),
            FILE_BYTES as u64
        );
        assert!(reference.write_new(&remove.0).is_err());
        let parsed = Reference::load(&remove.0).unwrap();
        parsed
            .verify_cpu_bits(
                12,
                reference.logits(12).unwrap(),
                reference.hidden(12).unwrap(),
            )
            .unwrap();
    }
}
