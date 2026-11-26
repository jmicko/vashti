//! WAV input (PCM 8/16/24/32-bit or 32/64-bit float, any rate and channel count, mixed to
//! mono) and output (mono 16-bit PCM), and band-limited resampling.
use crate::Result;

const MAX_CHANNELS: u16 = 32;

/// Mono samples in [-1, 1] and their rate.
pub fn read(bytes: &[u8]) -> Result<(Vec<f32>, u32)> {
    let bad = "audio_format_unsupported";
    if bytes.len() < 12 || &bytes[..4] != b"RIFF" || &bytes[8..12] != b"WAVE" {
        return Err(bad);
    }
    let mut i = 12usize;
    let mut format: Option<(u16, u16, u32, u16)> = None;
    while i + 8 <= bytes.len() {
        let id = &bytes[i..i + 4];
        let len = u32::from_le_bytes(bytes[i + 4..i + 8].try_into().unwrap()) as usize;
        let start = i + 8;
        // A streamed file may declare more data than it holds: take what is there.
        let end = start.saturating_add(len).min(bytes.len());
        let body = &bytes[start..end];
        if id == b"fmt " {
            if body.len() < 16 {
                return Err(bad);
            }
            let mut tag = u16::from_le_bytes([body[0], body[1]]);
            let channels = u16::from_le_bytes([body[2], body[3]]);
            let rate = u32::from_le_bytes(body[4..8].try_into().unwrap());
            let bits = u16::from_le_bytes([body[14], body[15]]);
            // WAVE_FORMAT_EXTENSIBLE: the sub-format's first two bytes are the tag.
            if tag == 0xfffe && body.len() >= 26 {
                tag = u16::from_le_bytes([body[24], body[25]]);
            }
            format = Some((tag, channels, rate, bits));
        } else if id == b"data" {
            let (tag, channels, rate, bits) = format.ok_or(bad)?;
            if channels == 0 || channels > MAX_CHANNELS || !(1000..=384_000).contains(&rate) {
                return Err(bad);
            }
            let width = usize::from(bits / 8);
            let decode: fn(&[u8]) -> f32 = match (tag, bits) {
                (1, 8) => |b| (f32::from(b[0]) - 128.0) / 128.0,
                (1, 16) => |b| f32::from(i16::from_le_bytes([b[0], b[1]])) / 32768.0,
                (1, 24) => {
                    |b| (i32::from_le_bytes([0, b[0], b[1], b[2]]) >> 8) as f32 / 8_388_608.0
                }
                (1, 32) => |b| i32::from_le_bytes(b.try_into().unwrap()) as f32 / 2_147_483_648.0,
                (3, 32) => |b| f32::from_le_bytes(b.try_into().unwrap()),
                (3, 64) => |b| f64::from_le_bytes(b.try_into().unwrap()) as f32,
                _ => return Err(bad),
            };
            let frame = width * usize::from(channels);
            let samples = body
                .chunks_exact(frame)
                .map(|f| {
                    let sum: f32 = f.chunks_exact(width).map(decode).sum();
                    let v = sum / f32::from(channels);
                    if v.is_finite() {
                        v.clamp(-1.0, 1.0)
                    } else {
                        0.0
                    }
                })
                .collect();
            return Ok((samples, rate));
        }
        i = start.saturating_add(len).saturating_add(len & 1);
    }
    Err(bad)
}

/// Mono 16-bit PCM WAV.
pub fn write(samples: &[f32], rate: u32) -> Vec<u8> {
    let data = (samples.len() * 2) as u32;
    let mut b = Vec::with_capacity(44 + samples.len() * 2);
    b.extend(b"RIFF");
    b.extend((36 + data).to_le_bytes());
    b.extend(b"WAVEfmt ");
    b.extend(16u32.to_le_bytes());
    b.extend(1u16.to_le_bytes());
    b.extend(1u16.to_le_bytes());
    b.extend(rate.to_le_bytes());
    b.extend((rate * 2).to_le_bytes());
    b.extend(2u16.to_le_bytes());
    b.extend(16u16.to_le_bytes());
    b.extend(b"data");
    b.extend(data.to_le_bytes());
    for v in samples {
        let v = if v.is_finite() {
            v.clamp(-1.0, 1.0)
        } else {
            0.0
        };
        b.extend(((v * 32767.0).round() as i16).to_le_bytes());
    }
    b
}

/// Windowed-sinc (Blackman, 32 zero crossings) resampling, low-passed below the lower
/// Nyquist rate.
pub fn resample(s: &[f32], from: u32, to: u32) -> Vec<f32> {
    if from == to || s.is_empty() {
        return s.to_vec();
    }
    use rayon::prelude::*;
    let ratio = f64::from(to) / f64::from(from);
    let cutoff = ratio.min(1.0) * 0.95;
    let half = (32.0 / cutoff).ceil() as isize;
    let n = (s.len() as f64 * ratio).floor() as usize;
    (0..n)
        .into_par_iter()
        .map(|i| {
            let x = i as f64 / ratio;
            let centre = x.floor() as isize;
            let mut sum = 0f64;
            for j in centre - half + 1..=centre + half {
                if j < 0 || j as usize >= s.len() {
                    continue;
                }
                let d = x - j as f64;
                let t = d / half as f64;
                if t.abs() >= 1.0 {
                    continue;
                }
                let w = 0.42
                    + 0.5 * (std::f64::consts::PI * t).cos()
                    + 0.08 * (2.0 * std::f64::consts::PI * t).cos();
                let a = std::f64::consts::PI * d * cutoff;
                let sinc = if a.abs() < 1e-9 { 1.0 } else { a.sin() / a };
                sum += f64::from(s[j as usize]) * cutoff * sinc * w;
            }
            sum as f32
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tone(rate: u32, hz: f32, secs: f32) -> Vec<f32> {
        (0..(rate as f32 * secs) as usize)
            .map(|i| 0.5 * (2.0 * std::f32::consts::PI * hz * i as f32 / rate as f32).sin())
            .collect()
    }

    #[test]
    fn round_trips_pcm16() {
        let s = tone(24_000, 440.0, 0.1);
        let (back, rate) = read(&write(&s, 24_000)).unwrap();
        assert_eq!(rate, 24_000);
        assert_eq!(back.len(), s.len());
        assert!(s.iter().zip(&back).all(|(a, b)| (a - b).abs() < 1e-4));
    }

    #[test]
    fn reads_float_stereo_and_rejects_others() {
        let mut b = Vec::new();
        b.extend(b"RIFF\0\0\0\0WAVEfmt ");
        b.extend(16u32.to_le_bytes());
        b.extend(3u16.to_le_bytes());
        b.extend(2u16.to_le_bytes());
        b.extend(48_000u32.to_le_bytes());
        b.extend((48_000u32 * 8).to_le_bytes());
        b.extend(8u16.to_le_bytes());
        b.extend(32u16.to_le_bytes());
        b.extend(b"data");
        b.extend(16u32.to_le_bytes());
        for v in [0.5f32, -0.5, 1.0, 0.0] {
            b.extend(v.to_le_bytes());
        }
        let (s, rate) = read(&b).unwrap();
        assert_eq!((s, rate), (vec![0.0, 0.5], 48_000));
        assert!(read(b"RIFF....WAVE").is_err());
        assert!(read(b"not audio at all").is_err());
    }

    #[test]
    fn resampling_keeps_a_tone_and_removes_aliases() {
        let s = tone(48_000, 1000.0, 0.5);
        let r = resample(&s, 48_000, 16_000);
        assert_eq!(r.len(), 8000);
        let expect = tone(16_000, 1000.0, 0.5);
        let err = r[200..7800]
            .iter()
            .zip(&expect[200..7800])
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        assert!(err < 0.01, "{err}");
        // 12 kHz is above the new Nyquist rate: it must vanish, not fold to 4 kHz.
        let high = resample(&tone(48_000, 12_000.0, 0.5), 48_000, 16_000);
        let peak = high[200..7800].iter().fold(0f32, |m, v| m.max(v.abs()));
        assert!(peak < 0.01, "{peak}");
    }
}
