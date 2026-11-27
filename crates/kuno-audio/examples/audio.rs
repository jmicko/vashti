//! `audio transcribe <model dir> <wav> [language]` or `audio speak <model dir> <text> <out.wav>
//! [voice] [seed]`: the CPU engine on its own, timed (prints no audio content but the
//! transcript asked for).
use kuno_audio::Lfm3Audio;

fn read_wav(path: &str) -> (Vec<f32>, u32) {
    let b = std::fs::read(path).unwrap();
    assert_eq!(&b[..4], b"RIFF");
    let (mut i, mut rate, mut channels, mut bits) = (12usize, 0u32, 1u16, 16u16);
    loop {
        let id = &b[i..i + 4];
        let len = u32::from_le_bytes(b[i + 4..i + 8].try_into().unwrap()) as usize;
        let body = &b[i + 8..i + 8 + len];
        if id == b"fmt " {
            channels = u16::from_le_bytes(body[2..4].try_into().unwrap());
            rate = u32::from_le_bytes(body[4..8].try_into().unwrap());
            bits = u16::from_le_bytes(body[14..16].try_into().unwrap());
        } else if id == b"data" {
            assert_eq!(bits, 16);
            let s: Vec<f32> = body
                .chunks_exact(2 * channels as usize)
                .map(|f| {
                    f.chunks_exact(2)
                        .map(|c| f32::from(i16::from_le_bytes([c[0], c[1]])) / 32768.0)
                        .sum::<f32>()
                        / f32::from(channels)
                })
                .collect();
            return (s, rate);
        }
        i += 8 + len + (len & 1);
    }
}
/// Linear resampling to 16 kHz (the test clips; the service resamples properly).
fn to_16k(s: &[f32], rate: u32) -> Vec<f32> {
    if rate == 16_000 {
        return s.to_vec();
    }
    let n = (s.len() as u64 * 16_000 / u64::from(rate)) as usize;
    (0..n)
        .map(|i| {
            let x = i as f64 * f64::from(rate) / 16_000.0;
            let j = x as usize;
            let f = (x - j as f64) as f32;
            s[j] * (1.0 - f) + s.get(j + 1).copied().unwrap_or(0.0) * f
        })
        .collect()
}
fn write_wav(path: &str, s: &[f32], rate: u32) {
    let mut b = Vec::new();
    let data = (s.len() * 2) as u32;
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
    for v in s {
        b.extend(((v.clamp(-1.0, 1.0) * 32767.0) as i16).to_le_bytes());
    }
    std::fs::write(path, b).unwrap();
}

fn main() {
    if std::env::var_os("RAYON_NUM_THREADS").is_none() {
        kuno_audio::use_physical_cores();
    }
    let mut a: Vec<String> = std::env::args().skip(1).collect();
    // `--gpu PCI` first puts the backbone and depth decoder on that GPU.
    let gpu = if a.first().map(String::as_str) == Some("--gpu") {
        let pci = a.remove(1);
        a.remove(0);
        Some(pci)
    } else {
        None
    };
    let dir = std::path::Path::new(&a[1]);
    let t0 = std::time::Instant::now();
    let m = Lfm3Audio::open_on(
        &dir.join("LFM3-Audio-2B-Q8_0.gguf"),
        &dir.join("mmproj-LFM3-Audio-2B-F16.gguf"),
        gpu.as_deref(),
    )
    .unwrap();
    eprintln!("device {}", m.device());
    eprintln!("load {:.2} s", t0.elapsed().as_secs_f64());
    match a[0].as_str() {
        "encode" => {
            let (s, rate) = read_wav(&a[2]);
            let s = to_16k(&s, rate);
            let t0 = std::time::Instant::now();
            let rows = m.encode_audio(&s).unwrap();
            eprintln!(
                "encode {:.2} s for {:.2} s of audio ({} rows)",
                t0.elapsed().as_secs_f64(),
                s.len() as f64 / 16000.0,
                rows.len() / 2048
            );
        }
        "transcribe" => {
            let (s, rate) = read_wav(&a[2]);
            let s = to_16k(&s, rate);
            let t0 = std::time::Instant::now();
            let text = m.transcribe(&s, a.get(3).map(String::as_str), 512).unwrap();
            eprintln!(
                "transcribe {:.2} s for {:.2} s of audio",
                t0.elapsed().as_secs_f64(),
                s.len() as f64 / 16000.0
            );
            println!("{text}");
        }
        "speak" => {
            let t0 = std::time::Instant::now();
            let mut first = None;
            let voice = a.get(4).map_or("female", String::as_str);
            let seed = a.get(5).map_or(42, |s| s.parse().unwrap());
            let pcm = m
                .speak(&a[2], "American English", voice, seed, 3000, |_| {
                    first.get_or_insert(t0.elapsed().as_secs_f64());
                    true
                })
                .unwrap();
            let secs = pcm.len() as f64 / 24000.0;
            let took = t0.elapsed().as_secs_f64();
            eprintln!(
                "speak {took:.2} s for {secs:.2} s of audio ({:.2}x real time), first audio {:.2} s",
                secs / took,
                first.unwrap_or(0.0)
            );
            write_wav(&a[3], &pcm, 24000);
        }
        _ => panic!("transcribe or speak"),
    }
}
