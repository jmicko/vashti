//! In-memory MP4 output for generated clips: AV1 video (rav1e) and FLAC audio (flacenc),
//! both pure Rust. The container carries no metadata. `validate_mp4` is the gateway-side
//! structural check of what a worker returns; it allocates nothing proportional to content.
use crate::Result;

/// Box types the encoder writes. Anything else (udta, meta, free, ...) is rejected.
const CONTAINERS: [&[u8; 4]; 6] = [b"moov", b"trak", b"mdia", b"minf", b"dinf", b"stbl"];
const LEAVES: [&[u8; 4]; 13] = [
    b"mvhd", b"tkhd", b"mdhd", b"hdlr", b"vmhd", b"smhd", b"dref", b"stsd", b"stts", b"stss",
    b"stsc", b"stsz", b"stco",
];

/// Structural facts the gateway checks against the request.
#[derive(Debug, PartialEq, Eq)]
pub struct Mp4Summary {
    pub width: u32,
    pub height: u32,
    pub frames: u32,
    pub fps: u32,
    pub audio: bool,
}

fn be32(bytes: &[u8], at: usize) -> Result<u32> {
    bytes
        .get(at..at + 4)
        .map(|b| u32::from_be_bytes(b.try_into().unwrap()))
        .ok_or("video_output_invalid")
}

fn boxes(bytes: &[u8]) -> Result<Vec<([u8; 4], &[u8])>> {
    let mut at = 0;
    let mut out = Vec::new();
    while at < bytes.len() {
        let size = be32(bytes, at)? as usize;
        let end = at.checked_add(size).ok_or("video_output_invalid")?;
        if size < 8 || end > bytes.len() || out.len() > 64 {
            return Err("video_output_invalid");
        }
        out.push((
            bytes[at + 4..at + 8].try_into().unwrap(),
            &bytes[at + 8..end],
        ));
        at = end;
    }
    Ok(out)
}

fn child<'a>(children: &[([u8; 4], &'a [u8])], kind: &[u8; 4]) -> Result<&'a [u8]> {
    let mut found = children.iter().filter(|(k, _)| k == kind);
    match (found.next(), found.next()) {
        (Some((_, body)), None) => Ok(body),
        _ => Err("video_output_invalid"),
    }
}

/// Rejects every box type the encoder does not write, at any depth.
fn allowlisted(bytes: &[u8], depth: usize) -> Result<()> {
    for (kind, body) in boxes(bytes)? {
        if CONTAINERS.contains(&&kind) && depth < 6 {
            allowlisted(body, depth + 1)?;
        } else if !LEAVES.contains(&&kind) {
            return Err("video_output_invalid");
        }
    }
    Ok(())
}

struct Track<'a> {
    handler: [u8; 4],
    timescale: u32,
    stsd: &'a [u8],
    stts: &'a [u8],
    stsz: &'a [u8],
    stco: &'a [u8],
    tkhd: &'a [u8],
}

fn track(body: &[u8]) -> Result<Track<'_>> {
    let trak = boxes(body)?;
    let mdia = boxes(child(&trak, b"mdia")?)?;
    let mdhd = child(&mdia, b"mdhd")?;
    let hdlr = child(&mdia, b"hdlr")?;
    // An empty name: the handler box must carry no text.
    if hdlr.len() != 25 || hdlr[24] != 0 {
        return Err("video_output_invalid");
    }
    let stbl = boxes(child(&boxes(child(&mdia, b"minf")?)?, b"stbl")?)?;
    Ok(Track {
        handler: hdlr[8..12].try_into().unwrap(),
        timescale: be32(mdhd, 12)?,
        stsd: child(&stbl, b"stsd")?,
        stts: child(&stbl, b"stts")?,
        stsz: child(&stbl, b"stsz")?,
        stco: child(&stbl, b"stco")?,
        tkhd: child(&trak, b"tkhd")?,
    })
}

/// Sum of sample sizes, checking the single chunk lies inside `mdat`.
fn samples_in(track: &Track, mdat: std::ops::Range<usize>) -> Result<(u32, usize, usize)> {
    let count = be32(track.stsz, 8)?;
    if be32(track.stsz, 4)? != 0
        || track.stsz.len() != 12 + 4 * count as usize
        || be32(track.stco, 4)? != 1
        || track.stco.len() != 12
    {
        return Err("video_output_invalid");
    }
    let mut total = 0usize;
    for i in 0..count as usize {
        total = total
            .checked_add(be32(track.stsz, 12 + 4 * i)? as usize)
            .ok_or("video_output_invalid")?;
    }
    let offset = be32(track.stco, 8)? as usize;
    if offset < mdat.start || offset.checked_add(total).is_none_or(|end| end > mdat.end) {
        return Err("video_output_invalid");
    }
    Ok((count, offset, total))
}

/// Strict structural check of a worker-produced MP4 before it reaches the client.
pub fn validate_mp4(bytes: &[u8], max_bytes: usize) -> Result<Mp4Summary> {
    if bytes.len() > max_bytes {
        return Err("video_output_invalid");
    }
    let top = boxes(bytes)?;
    if top.len() != 3 || &top[0].0 != b"ftyp" || &top[1].0 != b"moov" || &top[2].0 != b"mdat" {
        return Err("video_output_invalid");
    }
    allowlisted(top[1].1, 0)?;
    let mdat_start = bytes.len() - top[2].1.len();
    let mdat = mdat_start..bytes.len();
    let moov = boxes(top[1].1)?;
    child(&moov, b"mvhd")?;
    let traks: Vec<_> = moov.iter().filter(|(k, _)| k == b"trak").collect();
    if !(1..=2).contains(&traks.len()) || moov.len() != traks.len() + 1 {
        return Err("video_output_invalid");
    }
    let video = track(traks[0].1)?;
    if video.handler != *b"vide" || video.stsd.get(12..16) != Some(b"av01") {
        return Err("video_output_invalid");
    }
    let (frames, video_offset, video_bytes) = samples_in(&video, mdat.clone())?;
    let delta = be32(video.stts, 12)?;
    if be32(video.stts, 4)? != 1
        || be32(video.stts, 8)? != frames
        || delta == 0
        || video.timescale % delta != 0
        || frames == 0
    {
        return Err("video_output_invalid");
    }
    let width = be32(video.tkhd, 76)? >> 16;
    let height = be32(video.tkhd, 80)? >> 16;
    let mut used = video_bytes;
    let audio = if let Some((_, body)) = traks.get(1) {
        let audio = track(body)?;
        if audio.handler != *b"soun" || audio.stsd.get(12..16) != Some(b"fLaC") {
            return Err("video_output_invalid");
        }
        let (_, audio_offset, audio_bytes) = samples_in(&audio, mdat.clone())?;
        // The two chunks are adjacent and do not overlap.
        if audio_offset != video_offset + video_bytes {
            return Err("video_output_invalid");
        }
        used += audio_bytes;
        true
    } else {
        false
    };
    if video_offset != mdat.start || used != mdat.len() {
        return Err("video_output_invalid");
    }
    Ok(Mp4Summary {
        width,
        height,
        frames,
        fps: video.timescale / delta,
        audio,
    })
}

/// Scale an image to cover `width x height` (aspect preserved) and centre-crop it to RGB.
/// Transparent pixels are composited over black. `code` names the input on failure.
pub fn cover_rgb(
    pixels: crate::references::Pixels,
    width: u32,
    height: u32,
    code: &'static str,
) -> Result<Vec<u8>> {
    use image::imageops::{FilterType, crop_imm, resize};
    let crate::references::Pixels {
        width: w,
        height: h,
        mut rgba,
    } = pixels;
    let scale = (f64::from(width) / f64::from(w)).max(f64::from(height) / f64::from(h));
    let sw = ((f64::from(w) * scale).round() as u32).max(width);
    let sh = ((f64::from(h) * scale).round() as u32).max(height);
    if sw > 8192 || sh > 8192 {
        crate::worker_protocol::clear_bytes(&mut rgba);
        return Err(code);
    }
    let source = image::RgbaImage::from_raw(w, h, std::mem::take(&mut rgba)).ok_or(code)?;
    let mut scaled = resize(&source, sw, sh, FilterType::Lanczos3);
    let mut source = source.into_raw();
    crate::worker_protocol::clear_bytes(&mut source);
    let cropped = crop_imm(&scaled, (sw - width) / 2, (sh - height) / 2, width, height).to_image();
    let mut out = Vec::with_capacity(width as usize * height as usize * 3);
    for p in cropped.pixels() {
        let a = u16::from(p[3]);
        out.extend(
            p.0[..3]
                .iter()
                .map(|c| ((u16::from(*c) * a + 127) / 255) as u8),
        );
    }
    let mut cropped = cropped.into_raw();
    crate::worker_protocol::clear_bytes(&mut cropped);
    scaled.iter_mut().for_each(|b| *b = 0);
    Ok(out)
}

/// RGB at the original size, transparent pixels composited over black. Reference images
/// keep their size: the engine scales them for the vision encoder and the latent itself.
pub fn rgb(pixels: crate::references::Pixels) -> Vec<u8> {
    let mut rgba = pixels.rgba;
    let mut out = Vec::with_capacity(rgba.len() / 4 * 3);
    for p in rgba.chunks_exact(4) {
        let a = u16::from(p[3]);
        out.extend(
            p[..3]
                .iter()
                .map(|c| ((u16::from(*c) * a + 127) / 255) as u8),
        );
    }
    crate::worker_protocol::clear_bytes(&mut rgba);
    out
}

#[cfg(test)]
mod cover_tests {
    use super::*;
    #[test]
    fn cover_fills_the_frame_and_crops_the_centre() {
        // 400x100 image: left half red, right half blue; cover into 64x64 keeps the centre.
        let mut rgba = Vec::new();
        for _ in 0..100 {
            for x in 0..400 {
                rgba.extend(if x < 200 {
                    [255, 0, 0, 255]
                } else {
                    [0, 0, 255, 255]
                });
            }
        }
        let pixels = crate::references::Pixels {
            width: 400,
            height: 100,
            rgba,
        };
        let rgb = cover_rgb(pixels, 64, 64, "video_first_frame_invalid").unwrap();
        assert_eq!(rgb.len(), 64 * 64 * 3);
        assert_eq!(&rgb[..3], &[255, 0, 0]);
        assert_eq!(&rgb[rgb.len() - 3..], &[0, 0, 255]);
    }
}

#[cfg(feature = "video-encode")]
pub use encode::*;

#[cfg(feature = "video-encode")]
mod encode {
    use super::*;
    use crate::worker_protocol::clear_bytes;
    const VIDEO_TIMESCALE_PER_FRAME: u32 = 512;
    const MOVIE_TIMESCALE: u32 = 1000;
    /// (av1C record, sample data, sample sizes, sync sample numbers)
    type EncodedVideo = (Vec<u8>, Vec<u8>, Vec<u32>, Vec<u32>);
    /// (STREAMINFO, sample data, sample sizes, total sample frames)
    type EncodedAudio = (Vec<u8>, Vec<u8>, Vec<u32>, u64);

    /// Interleaved float PCM as the native engine returns it.
    pub struct Audio {
        pub sample_rate: u32,
        pub channels: u32,
        pub samples: Vec<f32>,
    }
    impl Drop for Audio {
        fn drop(&mut self) {
            self.samples.iter_mut().for_each(|s| *s = 0.0);
        }
    }

    /// RGB8 frames, `width * height * 3` bytes each, concatenated.
    pub struct Clip {
        pub width: u32,
        pub height: u32,
        pub fps: u32,
        pub frames: u32,
        pub rgb: Vec<u8>,
        pub audio: Option<Audio>,
    }
    impl Drop for Clip {
        fn drop(&mut self) {
            clear_bytes(&mut self.rgb);
        }
    }

    struct Bytes(Vec<u8>);
    impl Bytes {
        fn u8(&mut self, v: u8) -> &mut Self {
            self.0.push(v);
            self
        }
        fn u16(&mut self, v: u16) -> &mut Self {
            self.0.extend_from_slice(&v.to_be_bytes());
            self
        }
        fn u32(&mut self, v: u32) -> &mut Self {
            self.0.extend_from_slice(&v.to_be_bytes());
            self
        }
        fn raw(&mut self, v: &[u8]) -> &mut Self {
            self.0.extend_from_slice(v);
            self
        }
        fn zeros(&mut self, n: usize) -> &mut Self {
            self.0.resize(self.0.len() + n, 0);
            self
        }
    }
    fn mp4_box(kind: &[u8; 4], body: &[u8]) -> Vec<u8> {
        let mut out = Vec::with_capacity(body.len() + 8);
        out.extend_from_slice(&(body.len() as u32 + 8).to_be_bytes());
        out.extend_from_slice(kind);
        out.extend_from_slice(body);
        out
    }
    fn full_box(kind: &[u8; 4], flags: u32, body: &[u8]) -> Vec<u8> {
        let mut b = Bytes(flags.to_be_bytes().to_vec());
        b.raw(body);
        mp4_box(kind, &b.0)
    }
    fn container(kind: &[u8; 4], children: &[Vec<u8>]) -> Vec<u8> {
        mp4_box(kind, &children.concat())
    }
    const MATRIX: [u32; 9] = [0x0001_0000, 0, 0, 0, 0x0001_0000, 0, 0, 0, 0x4000_0000];

    struct TrackData {
        id: u32,
        video: bool,
        timescale: u32,
        duration: u32,
        movie_duration: u32,
        width: u32,
        height: u32,
        sample_entry: Vec<u8>,
        /// (count, delta) runs.
        stts: Vec<(u32, u32)>,
        sync: Option<Vec<u32>>,
        sizes: Vec<u32>,
        offset: u32,
    }

    fn trak(t: &TrackData) -> Vec<u8> {
        let mut tkhd = Bytes(Vec::new());
        tkhd.u32(0)
            .u32(0)
            .u32(t.id)
            .u32(0)
            .u32(t.movie_duration)
            .zeros(8);
        tkhd.u16(0)
            .u16(0)
            .u16(if t.video { 0 } else { 0x0100 })
            .u16(0);
        MATRIX.iter().for_each(|m| {
            tkhd.u32(*m);
        });
        tkhd.u32(t.width << 16).u32(t.height << 16);
        let mut mdhd = Bytes(Vec::new());
        mdhd.u32(0)
            .u32(0)
            .u32(t.timescale)
            .u32(t.duration)
            .u16(0x55c4)
            .u16(0);
        let mut hdlr = Bytes(Vec::new());
        hdlr.u32(0)
            .raw(if t.video { b"vide" } else { b"soun" })
            .zeros(12)
            .u8(0);
        let media_header = if t.video {
            full_box(b"vmhd", 1, &[0; 8])
        } else {
            full_box(b"smhd", 0, &[0; 4])
        };
        let dref = full_box(
            b"dref",
            0,
            &[&1u32.to_be_bytes()[..], &full_box(b"url ", 1, &[])].concat(),
        );
        let dinf = container(b"dinf", &[dref]);
        let mut stsd = Bytes(Vec::new());
        stsd.u32(1).raw(&t.sample_entry);
        let mut stts = Bytes(Vec::new());
        stts.u32(t.stts.len() as u32);
        t.stts.iter().for_each(|(count, delta)| {
            stts.u32(*count).u32(*delta);
        });
        let mut stsc = Bytes(Vec::new());
        stsc.u32(1).u32(1).u32(t.sizes.len() as u32).u32(1);
        let mut stsz = Bytes(Vec::new());
        stsz.u32(0).u32(t.sizes.len() as u32);
        t.sizes.iter().for_each(|s| {
            stsz.u32(*s);
        });
        let mut stco = Bytes(Vec::new());
        stco.u32(1).u32(t.offset);
        let mut stbl = vec![full_box(b"stsd", 0, &stsd.0), full_box(b"stts", 0, &stts.0)];
        if let Some(sync) = &t.sync {
            let mut stss = Bytes(Vec::new());
            stss.u32(sync.len() as u32);
            sync.iter().for_each(|s| {
                stss.u32(*s);
            });
            stbl.push(full_box(b"stss", 0, &stss.0));
        }
        stbl.extend([
            full_box(b"stsc", 0, &stsc.0),
            full_box(b"stsz", 0, &stsz.0),
            full_box(b"stco", 0, &stco.0),
        ]);
        let minf = container(b"minf", &[media_header, dinf, container(b"stbl", &stbl)]);
        let mdia = container(
            b"mdia",
            &[
                full_box(b"mdhd", 0, &mdhd.0),
                full_box(b"hdlr", 0, &hdlr.0),
                minf,
            ],
        );
        container(b"trak", &[full_box(b"tkhd", 3, &tkhd.0), mdia])
    }

    fn moov(tracks: &[TrackData], movie_duration: u32) -> Vec<u8> {
        let mut mvhd = Bytes(Vec::new());
        mvhd.u32(0).u32(0).u32(MOVIE_TIMESCALE).u32(movie_duration);
        mvhd.u32(0x0001_0000).u16(0x0100).zeros(10);
        MATRIX.iter().for_each(|m| {
            mvhd.u32(*m);
        });
        mvhd.zeros(24).u32(tracks.len() as u32 + 1);
        let mut children = vec![full_box(b"mvhd", 0, &mvhd.0)];
        children.extend(tracks.iter().map(trak));
        container(b"moov", &children)
    }

    /// Drops temporal delimiters: AV1-in-ISOBMFF samples should not contain them.
    fn strip_temporal_delimiters(packet: &[u8]) -> Result<Vec<u8>> {
        let mut out = Vec::with_capacity(packet.len());
        let mut at = 0;
        while at < packet.len() {
            let header = packet[at];
            let kind = (header >> 3) & 0xf;
            let extension = header & 4 != 0;
            if header & 2 == 0 || header & 0x80 != 0 {
                return Err("video_encode_failed");
            }
            let mut cursor = at + 1 + usize::from(extension);
            let mut size = 0usize;
            for shift in (0..56).step_by(7) {
                let byte = *packet.get(cursor).ok_or("video_encode_failed")?;
                cursor += 1;
                size |= usize::from(byte & 0x7f) << shift;
                if byte & 0x80 == 0 {
                    break;
                }
            }
            let end = cursor.checked_add(size).filter(|e| *e <= packet.len());
            let end = end.ok_or("video_encode_failed")?;
            if kind != 2 {
                out.extend_from_slice(&packet[at..end]);
            }
            at = end;
        }
        Ok(out)
    }

    /// BT.709 limited range, chroma from the 2x2 RGB mean.
    fn to_yuv420(rgb: &[u8], width: usize, height: usize, planes: &mut [Vec<u8>; 3]) {
        let luma = |r: f32, g: f32, b: f32| 0.2126 * r + 0.7152 * g + 0.0722 * b;
        for y in 0..height {
            for x in 0..width {
                let p = &rgb[(y * width + x) * 3..][..3];
                let l = luma(p[0] as f32, p[1] as f32, p[2] as f32);
                planes[0][y * width + x] = (16.0 + l * (219.0 / 255.0)).round() as u8;
            }
        }
        let cw = width / 2;
        for y in 0..height / 2 {
            for x in 0..cw {
                let mut sum = [0f32; 3];
                for (dy, dx) in [(0, 0), (0, 1), (1, 0), (1, 1)] {
                    let p = &rgb[((2 * y + dy) * width + 2 * x + dx) * 3..][..3];
                    for c in 0..3 {
                        sum[c] += p[c] as f32 * 0.25;
                    }
                }
                let l = luma(sum[0], sum[1], sum[2]);
                let cb = 128.0 + (sum[2] - l) / 1.8556 * (224.0 / 255.0);
                let cr = 128.0 + (sum[0] - l) / 1.5748 * (224.0 / 255.0);
                planes[1][y * cw + x] = cb.round().clamp(16.0, 240.0) as u8;
                planes[2][y * cw + x] = cr.round().clamp(16.0, 240.0) as u8;
            }
        }
    }

    fn encode_video(clip: &Clip) -> Result<EncodedVideo> {
        use rav1e::prelude::*;
        let (w, h) = (clip.width as usize, clip.height as usize);
        let mut config = EncoderConfig::with_speed_preset(10);
        config.width = w;
        config.height = h;
        config.time_base = Rational::new(1, u64::from(clip.fps));
        config.chroma_sampling = ChromaSampling::Cs420;
        config.pixel_range = PixelRange::Limited;
        config.color_description = Some(ColorDescription {
            color_primaries: ColorPrimaries::BT709,
            transfer_characteristics: TransferCharacteristics::BT709,
            matrix_coefficients: MatrixCoefficients::BT709,
        });
        config.quantizer = 80;
        config.min_key_frame_interval = 12;
        // A keyframe every two seconds keeps seeking responsive.
        config.max_key_frame_interval = u64::from(clip.fps) * 2;
        config.tile_cols = 2;
        config.tile_rows = 2;
        let threads = std::thread::available_parallelism().map_or(4, |n| n.get());
        let mut context: Context<u8> = Config::new()
            .with_encoder_config(config)
            .with_threads(threads)
            .new_context()
            .map_err(|_| "video_encode_failed")?;
        let av1c = context.container_sequence_header();
        let mut data = Vec::new();
        let (mut sizes, mut sync) = (Vec::new(), Vec::new());
        let mut planes = [vec![0u8; w * h], vec![0u8; w * h / 4], vec![0u8; w * h / 4]];
        let mut receive = |context: &mut Context<u8>, data: &mut Vec<u8>| -> Result<bool> {
            loop {
                match context.receive_packet() {
                    Ok(packet) => {
                        let sample = strip_temporal_delimiters(&packet.data)?;
                        if packet.frame_type == FrameType::KEY {
                            sync.push(sizes.len() as u32 + 1);
                        }
                        sizes.push(sample.len() as u32);
                        data.extend_from_slice(&sample);
                    }
                    Err(EncoderStatus::Encoded) => {}
                    Err(EncoderStatus::NeedMoreData) => return Ok(false),
                    Err(EncoderStatus::LimitReached) => return Ok(true),
                    Err(_) => return Err("video_encode_failed"),
                }
            }
        };
        for frame in clip.rgb.chunks_exact(w * h * 3) {
            to_yuv420(frame, w, h, &mut planes);
            let mut input = context.new_frame();
            input.planes[0].copy_from_raw_u8(&planes[0], w, 1);
            input.planes[1].copy_from_raw_u8(&planes[1], w / 2, 1);
            input.planes[2].copy_from_raw_u8(&planes[2], w / 2, 1);
            context
                .send_frame(input)
                .map_err(|_| "video_encode_failed")?;
            receive(&mut context, &mut data)?;
        }
        context.flush();
        while !receive(&mut context, &mut data)? {}
        planes.iter_mut().for_each(clear_bytes);
        if sizes.len() != clip.frames as usize || sync.first() != Some(&1) {
            return Err("video_encode_failed");
        }
        Ok((av1c, data, sizes, sync))
    }

    /// 16-bit FLAC frames of 4096 samples (the last one shorter) and the STREAMINFO body.
    fn encode_audio(audio: &Audio) -> Result<EncodedAudio> {
        use flacenc::{bitsink::ByteSink, component::BitRepr, error::Verify};
        let channels = audio.channels as usize;
        if !(1..=2).contains(&channels)
            || !(8000..=65535).contains(&audio.sample_rate)
            || audio.samples.is_empty()
            || !audio.samples.len().is_multiple_of(channels)
        {
            return Err("video_audio_invalid");
        }
        let mut pcm: Vec<i32> = audio
            .samples
            .iter()
            .map(|s| (s.clamp(-1.0, 1.0) * 32767.0).round() as i32)
            .collect();
        let config = flacenc::config::Encoder::default()
            .into_verified()
            .map_err(|_| "video_encode_failed")?;
        let block = config.block_size;
        let source = flacenc::source::MemSource::from_samples(
            &pcm,
            channels,
            16,
            audio.sample_rate as usize,
        );
        let stream = flacenc::encode_with_fixed_block_size(&config, source, block)
            .map_err(|_| "video_encode_failed")?;
        pcm.iter_mut().for_each(|s| *s = 0);
        let mut info = ByteSink::new();
        stream
            .stream_info()
            .write(&mut info)
            .map_err(|_| "video_encode_failed")?;
        let (mut data, mut sizes) = (Vec::new(), Vec::new());
        for i in 0..stream.frame_count() {
            let mut sink = ByteSink::new();
            stream
                .frame(i)
                .ok_or("video_encode_failed")?
                .write(&mut sink)
                .map_err(|_| "video_encode_failed")?;
            sizes.push(sink.as_slice().len() as u32);
            data.extend_from_slice(sink.as_slice());
        }
        let total = (audio.samples.len() / channels) as u64;
        if info.as_slice().len() != 34 || sizes.is_empty() {
            return Err("video_encode_failed");
        }
        Ok((info.as_slice().to_vec(), data, sizes, total))
    }

    /// Encode a clip into a metadata-free MP4 with the sample tables before the media data.
    pub fn encode_mp4(clip: &Clip) -> Result<Vec<u8>> {
        let frame_bytes = clip.width as usize * clip.height as usize * 3;
        if clip.width == 0
            || clip.height == 0
            || !clip.width.is_multiple_of(2)
            || !clip.height.is_multiple_of(2)
            || !(1..=120).contains(&clip.fps)
            || clip.frames == 0
            || clip.rgb.len() != frame_bytes * clip.frames as usize
        {
            return Err("video_frames_invalid");
        }
        let (av1c, video_data, video_sizes, sync) = encode_video(clip)?;
        let delta = VIDEO_TIMESCALE_PER_FRAME;
        let video_duration = clip.frames * delta;
        let video_timescale = clip.fps * delta;
        let movie = |duration: u64, timescale: u32| {
            (duration * u64::from(MOVIE_TIMESCALE)).div_ceil(u64::from(timescale)) as u32
        };
        let mut entry = Bytes(Vec::new());
        entry.zeros(6).u16(1).u16(0).u16(0).zeros(12);
        entry.u16(clip.width as u16).u16(clip.height as u16);
        entry
            .u32(0x0048_0000)
            .u32(0x0048_0000)
            .u32(0)
            .u16(1)
            .zeros(32)
            .u16(0x18)
            .u16(0xffff);
        entry.raw(&mp4_box(b"av1C", &av1c));
        let mut tracks = vec![TrackData {
            id: 1,
            video: true,
            timescale: video_timescale,
            duration: video_duration,
            movie_duration: movie(video_duration.into(), video_timescale),
            width: clip.width,
            height: clip.height,
            sample_entry: mp4_box(b"av01", &entry.0),
            stts: vec![(clip.frames, delta)],
            sync: Some(sync),
            sizes: video_sizes,
            offset: 0,
        }];
        let mut audio_data = Vec::new();
        if let Some(audio) = &clip.audio {
            let (info, data, sizes, total) = encode_audio(audio)?;
            audio_data = data;
            let mut dfla = Bytes(Vec::new());
            dfla.u8(0x80).u8(0).u16(34).raw(&info); // last block, STREAMINFO, 34 bytes
            let mut entry = Bytes(Vec::new());
            entry
                .zeros(6)
                .u16(1)
                .zeros(8)
                .u16(audio.channels as u16)
                .u16(16)
                .u16(0)
                .u16(0);
            entry
                .u32(audio.sample_rate << 16)
                .raw(&full_box(b"dfLa", 0, &dfla.0));
            let block = 4096u32;
            let full = (sizes.len() - 1) as u32;
            let last = (total - u64::from(full) * u64::from(block)) as u32;
            let mut stts = vec![];
            if full > 0 {
                stts.push((full, block));
            }
            stts.push((1, last));
            tracks.push(TrackData {
                id: 2,
                video: false,
                timescale: audio.sample_rate,
                duration: total as u32,
                movie_duration: movie(total, audio.sample_rate),
                width: 0,
                height: 0,
                sample_entry: mp4_box(b"fLaC", &entry.0),
                stts,
                sync: None,
                sizes,
                offset: 0,
            });
        }
        let mut ftyp = Bytes(Vec::new());
        ftyp.raw(b"isom").u32(0x200).raw(b"isomiso6av01mp41");
        let ftyp = mp4_box(b"ftyp", &ftyp.0);
        let movie_duration = tracks.iter().map(|t| t.movie_duration).max().unwrap();
        // Offsets are fixed-width, so the moov size does not depend on their values.
        let header = ftyp.len() + moov(&tracks, movie_duration).len() + 8;
        tracks[0].offset = header as u32;
        if let Some(audio) = tracks.get_mut(1) {
            audio.offset = (header + video_data.len()) as u32;
        }
        let moov = moov(&tracks, movie_duration);
        let media = video_data.len() + audio_data.len();
        let mut out = Vec::with_capacity(header + media);
        out.extend_from_slice(&ftyp);
        out.extend_from_slice(&moov);
        out.extend_from_slice(&(media as u32 + 8).to_be_bytes());
        out.extend_from_slice(b"mdat");
        out.extend_from_slice(&video_data);
        out.extend_from_slice(&audio_data);
        clear_bytes(&mut audio_data);
        Ok(out)
    }

    #[cfg(test)]
    mod tests {
        use super::*;
        fn clip(frames: u32, audio: bool) -> Clip {
            let (w, h) = (256u32, 160u32);
            let mut rgb = Vec::new();
            for f in 0..frames {
                for y in 0..h {
                    for x in 0..w {
                        rgb.extend([(x + f * 3) as u8, (y * 2) as u8, ((x ^ y) + f) as u8]);
                    }
                }
            }
            let samples = (0..(frames as usize * 32000 / 24) * 2)
                .map(|i| ((i / 2) as f32 * 0.05).sin() * 0.5)
                .collect();
            Clip {
                width: w,
                height: h,
                fps: 24,
                frames,
                rgb,
                audio: audio.then_some(Audio {
                    sample_rate: 32000,
                    channels: 2,
                    samples,
                }),
            }
        }
        #[test]
        fn encodes_a_valid_metadata_free_mp4() {
            for audio in [true, false] {
                let mp4 = encode_mp4(&clip(22, audio)).unwrap();
                let summary = validate_mp4(&mp4, 64 << 20).unwrap();
                assert_eq!(
                    summary,
                    Mp4Summary {
                        width: 256,
                        height: 160,
                        frames: 22,
                        fps: 24,
                        audio
                    }
                );
            }
        }
        #[test]
        fn validation_rejects_metadata_truncation_and_trailing_bytes() {
            let mp4 = encode_mp4(&clip(22, true)).unwrap();
            for cut in [1, 8, mp4.len() / 2] {
                assert!(validate_mp4(&mp4[..mp4.len() - cut], 64 << 20).is_err());
            }
            let mut trailing = mp4.clone();
            trailing.extend_from_slice(&mp4_box(b"free", &[]));
            assert!(validate_mp4(&trailing, 64 << 20).is_err());
            assert!(validate_mp4(&mp4, mp4.len() - 1).is_err());
            // A udta box inside moov is metadata and must be rejected.
            let top = boxes(&mp4).unwrap();
            let mut moov_body = top[1].1.to_vec();
            moov_body.extend_from_slice(&mp4_box(b"udta", b"private"));
            let mut tampered = top[0].1.to_vec();
            tampered = mp4_box(b"ftyp", &tampered);
            tampered.extend_from_slice(&mp4_box(b"moov", &moov_body));
            tampered.extend_from_slice(&mp4_box(b"mdat", top[2].1));
            assert!(validate_mp4(&tampered, 64 << 20).is_err());
        }
        /// Writes a sample clip for checks with external players: KUNO_VIDEO_SAMPLE=path.
        #[test]
        fn write_sample_when_requested() {
            if let Some(path) = std::env::var_os("KUNO_VIDEO_SAMPLE") {
                std::fs::write(path, encode_mp4(&clip(124, true)).unwrap()).unwrap();
            }
        }
        #[test]
        fn temporal_delimiters_are_removed() {
            // TD (type 2, size 0) then a padding OBU (type 15) of 2 bytes.
            let packet = [0x12, 0x00, 0x7a, 0x02, 0xaa, 0xbb];
            assert_eq!(
                strip_temporal_delimiters(&packet).unwrap(),
                vec![0x7a, 0x02, 0xaa, 0xbb]
            );
            assert!(strip_temporal_delimiters(&[0x10]).is_err());
        }
    }
}
