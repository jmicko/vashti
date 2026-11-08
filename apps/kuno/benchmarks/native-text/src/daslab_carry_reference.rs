//! Fixed public DASLab target logits and actual pre-output-HC residual.
//! No prompt parser, process binding ID or private-request persistence path.
//! The protected runner independently verifies complete original artifact and
//! executable hashes; the fixed header records that public provenance.

use kuno_text::model::greedy;
use kuno_text::{Error, Result};
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::Path;
use zeroize::{Zeroize, Zeroizing};

pub const POSITIONS: usize = 4;
pub const CONTEXT: usize = 4;
pub const VOCAB: usize = 248_320;
pub const HIDDEN: usize = 10_240;
pub const MAX_CHUNK: usize = 3;
pub const SEED: u32 = 424_242;
pub const MODEL_BYTES: u64 = 83_617_662_656;
pub const REVISION: &str = "ed59f92082b1e93c0e96d60a8b11aab089b52f09";
pub const GRAPH_BASE: &str = "6a32b1156ca645dbbf425b4edbc3b29917b443c5";
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
const MAGIC: &[u8; 8] = b"KDHC0001";
const HEADER_FIELDS: [u32; 10] = [4, 4, 248_320, 10_240, 4, 2560, 48, 3, 0, SEED];
pub const HEADER_BYTES: usize = 160;
pub const ROW_BYTES: usize = 24 + 4 * (VOCAB + HIDDEN);
pub const FILE_BYTES: usize = HEADER_BYTES + POSITIONS * ROW_BYTES;
pub const OWNER_HOST_CAP: usize = 8 * 1024 * 1024;
pub const LOAD_HOST_CAP: usize = 16 * 1024 * 1024;
const STREAM_BYTES: usize = 64 * 1024;
const LEGACY_ROW_BYTES: usize = 16 + VOCAB * 4;

fn err() -> Error {
    Error::InvalidFormat("fixed public DASLab target carry fixture rejected")
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
    /// original local cursor. Counts1..3 can end at4 for position3 continuation.
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

/// Bitwise CPU provenance gate against the complete existing4/32-row original
/// fixed-public KQ3R0001 file. Only first4 rows are retained; the suffix is still
/// streamed and validated entirely, including its chain/finiteness/greedy IDs.
pub struct LegacyLogits {
    first: Vec<Zeroizing<Vec<f32>>>,
}
impl LegacyLogits {
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let file = std::fs::File::open(path)?;
        let bytes = file.metadata()?.len();
        let positions = match bytes {
            n if n == (8 + 4 * LEGACY_ROW_BYTES) as u64 => 4,
            n if n == (8 + 32 * LEGACY_ROW_BYTES) as u64 => 32,
            _ => return Err(err()),
        };
        let mut reader = BufReader::with_capacity(STREAM_BYTES, file);
        Self::read(&mut reader, positions)
    }
    fn read(reader: &mut impl Read, positions: usize) -> Result<Self> {
        if !matches!(positions, 4 | 32) {
            return Err(err());
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
            let hidden = vec![position as f32; HIDDEN];
            rows.push(Row::new(position, input, &logits, &hidden).unwrap());
            input = position as u32 + 1;
        }
        Reference::new(rows).unwrap()
    }
    fn encoded() -> Vec<u8> {
        let mut bytes = Vec::new();
        fixture().write(&mut bytes).unwrap();
        bytes
    }
    #[test]
    fn complete_geometry_header_source_pin_and_round_trip_are_exact() {
        assert_eq!(HEADER_BYTES, 160);
        assert_eq!(FILE_BYTES, 4_137_216);
        assert_eq!(SHARD_BYTES.iter().sum::<u64>(), MODEL_BYTES);
        let bytes = encoded();
        assert_eq!(bytes.len(), FILE_BYTES);
        let parsed = Reference::read(&mut Cursor::new(bytes)).unwrap();
        assert!(parsed.owned_bytes().unwrap() < OWNER_HOST_CAP);
        for position in 0..POSITIONS {
            assert_eq!(parsed.input_token(position).unwrap(), position as u32);
            assert_eq!(parsed.token(position).unwrap(), position as u32 + 1);
            let report = parsed
                .verify(
                    position,
                    parsed.logits(position).unwrap(),
                    parsed.hidden(position).unwrap(),
                )
                .unwrap();
            assert_eq!(report.logits_max_abs_error, 0.);
            assert_eq!(report.hidden_max_abs_error, 0.);
        }
    }
    #[test]
    fn complete_source_header_row_order_chain_and_full_lengths_are_required() {
        let bytes = encoded();
        for at in [
            0,
            8,
            12,
            16,
            20,
            24,
            28,
            32,
            36,
            40,
            44,
            48,
            56,
            76,
            108,
            140,
            HEADER_BYTES,
            HEADER_BYTES + 4,
            HEADER_BYTES + 8,
            HEADER_BYTES + 12,
            HEADER_BYTES + 16,
            HEADER_BYTES + 20,
        ] {
            let mut bad = bytes.clone();
            bad[at] ^= 1;
            assert!(
                Reference::read(&mut Cursor::new(bad)).is_err(),
                "offset{at}"
            );
        }
        for position in 0..POSITIONS {
            let mut bad = bytes.clone();
            let last = HEADER_BYTES + (position + 1) * ROW_BYTES - 4;
            bad[last..last + 4].copy_from_slice(&f32::NAN.to_le_bytes());
            assert!(Reference::read(&mut Cursor::new(bad)).is_err());
        }
        for cut in [
            0,
            HEADER_BYTES - 1,
            HEADER_BYTES,
            HEADER_BYTES + ROW_BYTES,
            FILE_BYTES - 1,
        ] {
            assert!(Reference::read(&mut Cursor::new(&bytes[..cut])).is_err());
        }
        let mut extra = bytes;
        extra.push(0);
        assert!(Reference::read(&mut Cursor::new(extra)).is_err());
    }
    #[test]
    fn every_logit_and_carry_including_last_row_last_values_is_checked() {
        let r = fixture();
        assert_eq!(ABS_TOLERANCE.to_bits(), 0.006f32.to_bits());
        assert_eq!(REL_TOLERANCE.to_bits(), 0.0003f32.to_bits());
        for position in 0..POSITIONS {
            for hidden in [false, true] {
                let mut logits = r.logits(position).unwrap().to_vec();
                let mut carry = r.hidden(position).unwrap().to_vec();
                for bad in [
                    if hidden { position as f32 + 0.02 } else { 0.02 },
                    f32::NAN,
                    f32::INFINITY,
                    f32::NEG_INFINITY,
                ] {
                    if hidden {
                        carry[HIDDEN - 1] = bad;
                    } else {
                        logits[VOCAB - 1] = bad;
                    }
                    assert!(r.verify(position, &logits, &carry).is_err());
                }
            }
        }
        assert!(
            r.verify(0, &r.logits(0).unwrap()[..VOCAB - 1], r.hidden(0).unwrap())
                .is_err()
        );
        assert!(
            r.verify(0, r.logits(0).unwrap(), &r.hidden(0).unwrap()[..HIDDEN - 1])
                .is_err()
        );
        let mut extra = r.hidden(0).unwrap().to_vec();
        extra.push(0.);
        assert!(r.verify(0, r.logits(0).unwrap(), &extra).is_err());
        assert!(compare(&[f32::MAX], &[-f32::MAX]).is_err());
        assert!(compare(&[0.005], &[0.]).is_ok());
        assert!(compare(&[0.007], &[0.]).is_err());
        let mut wrong = r.logits(0).unwrap().to_vec();
        wrong[2] = 3.;
        assert!(r.verify(0, &wrong, r.hidden(0).unwrap()).is_err());
    }
    #[test]
    fn committed_prefix_checks_global_cursor_and_all_partitions_with_continuation() {
        let r = fixture();
        for partition in [&[1, 1, 1][..], &[1, 2], &[2, 1], &[3]] {
            let mut start = 0;
            for &count in partition {
                assert!(
                    r.verify_completed_prefix(start, count, start + count)
                        .is_ok()
                );
                start += count;
            }
            assert_eq!(start, 3);
            assert!(r.verify_completed_prefix(3, 1, 4).is_ok());
        }
        for args in [
            (0, 0, 0),
            (0, 4, 4),
            (0, 3, 1),
            (3, 2, 5),
            (4, 1, 5),
            (usize::MAX, 1, 0),
        ] {
            assert!(r.verify_completed_prefix(args.0, args.1, args.2).is_err());
        }
    }
    #[test]
    fn legacy_cpu_gate_is_bitwise_and_streams_the_complete_original_file() {
        let r = fixture();
        let mut bytes = b"KQ3R0001".to_vec();
        for position in 0..POSITIONS {
            for value in [
                position as u32,
                r.input_token(position).unwrap(),
                r.token(position).unwrap(),
                VOCAB as u32,
            ] {
                bytes.extend_from_slice(&value.to_le_bytes());
            }
            for value in r.logits(position).unwrap() {
                bytes.extend_from_slice(&value.to_le_bytes());
            }
        }
        let old = LegacyLogits::read(&mut Cursor::new(&bytes), 4).unwrap();
        assert!(old.verify_cpu_bits(3, r.logits(3).unwrap()).is_ok());
        let mut changed = r.logits(3).unwrap().to_vec();
        changed[VOCAB - 1] = -0.;
        assert!(old.verify_cpu_bits(3, &changed).is_err());
        assert!(LegacyLogits::read(&mut Cursor::new(&bytes[..bytes.len() - 1]), 4).is_err());
        let mut long = bytes.clone();
        for position in 4..32 {
            for value in [
                position as u32,
                if position == 4 { 4 } else { 0 },
                0,
                VOCAB as u32,
            ] {
                long.extend_from_slice(&value.to_le_bytes());
            }
            long.resize(long.len() + VOCAB * 4, 0);
        }
        assert!(LegacyLogits::read(&mut Cursor::new(&long), 32).is_ok());
        let last = long.len() - 4;
        long[last..].copy_from_slice(&f32::NAN.to_le_bytes());
        assert!(LegacyLogits::read(&mut Cursor::new(long), 32).is_err());
        bytes.push(0);
        assert!(LegacyLogits::read(&mut Cursor::new(bytes), 4).is_err());
    }
}
