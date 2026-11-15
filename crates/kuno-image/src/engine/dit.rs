//! MiniMax H3 diffusion transformer, CPU reference form.
//!
//! One call evaluates the network once (one sampler step) on a packed audio-video latent.
//! This form streams one layer's weights at a time and is the numeric reference for the GPU
//! kernels, not a production path: it is only practical for small latents.
//!
//! The 50 main blocks store their linear layers as INT8 with a per-row scale, in a rotated
//! basis ("ConvRot"): activations are rotated by a Hadamard transform in groups of 256,
//! rounded to INT8 with one scale per token, and multiplied as integers. The integer sums are
//! exact, so the arithmetic order of the quantizer below follows the C++ engine's.
use std::path::Path;

use rayon::prelude::*;

use super::{conv::threads, safetensors::Safetensors};

pub const VIDEO_CHANNELS: usize = 24;
pub const AUDIO_CHANNELS: usize = 32;
pub const TEXT_WIDTH: usize = 5120;
pub(super) const DIM: usize = 5376;
pub(super) const HEADS: usize = 56;
pub(super) const HEAD: usize = 128;
pub(super) const INNER: usize = HEADS * HEAD;
pub(super) const HIDDEN: usize = 14336;
/// Rotary frequencies per axis; the first `3 * FREQUENCIES` channel pairs of a head rotate.
pub(super) const FREQUENCIES: usize = 16;
pub(super) const PAIRS: usize = FREQUENCIES * 3;
pub(super) const TIME_WIDTH: usize = 8;
pub(super) const GROUP: usize = 256;
/// Video tokens are 1 x 2 x 2 latent positions.
pub(super) const PATCH: usize = 2;
pub(super) const PATCH_VALUES: usize = VIDEO_CHANNELS * PATCH * PATCH;
const EPS: f32 = 1e-5;
const FRAME_RESCALE: f32 = 5.0 / 3.0;
const VISUAL_CONDITION_TIME: f32 = 0.999;

/// Per-channel mean and deviation of the video decoder's latents; the DiT works on
/// normalized ones.
const LATENT_MEAN: [f32; VIDEO_CHANNELS] = [
    0.858_090_34,
    -0.960_659_15,
    1.066_164,
    -0.509_032_55,
    -0.272_758_2,
    -1.367_541_4,
    -0.255_325_5,
    -0.269_075_54,
    -0.537_684_1,
    -0.046_409_73,
    0.665_737,
    0.196_901_28,
    -0.546_060_8,
    -0.403_534_2,
    -0.236_830_25,
    0.259_284_53,
    -0.301_339_45,
    0.211_342,
    -1.120_684_9,
    0.358_193_34,
    -0.042_251_438,
    0.260_483,
    0.228_640_93,
    0.705_603_2,
];
const LATENT_STD: [f32; VIDEO_CHANNELS] = [
    1.222_377_4,
    1.276_726_4,
    1.683_177_5,
    1.754_945_5,
    1.563_621_6,
    2.194_143_5,
    0.965_313_8,
    1.056_988_6,
    0.841_948_9,
    0.772_995_3,
    1.895_593_8,
    0.946_841_84,
    0.799_680_95,
    0.449_889,
    0.719_74,
    0.693_629_3,
    2.961_095,
    2.769_42,
    3.049_618_5,
    2.108_805_4,
    3.276_226_3,
    3.162_735_7,
    2.281_681_3,
    2.612_784_4,
];
/// A DiT video latent `[24][...]` at the decoder's scale.
pub fn decoder_latent(latent: &[f32]) -> Vec<f32> {
    let plane = latent.len() / VIDEO_CHANNELS;
    latent
        .chunks_exact(plane)
        .zip(LATENT_MEAN.iter().zip(&LATENT_STD))
        .flat_map(|(channel, (mean, deviation))| channel.iter().map(move |v| v * deviation + mean))
        .collect()
}
/// A latent video `[24][frames][height][width]`.
pub struct Video<'a> {
    pub data: &'a [f32],
    pub width: usize,
    pub height: usize,
    pub frames: usize,
}
/// A latent stereo sound `[32][2][length]`.
pub struct Audio<'a> {
    pub data: &'a [f32],
    pub length: usize,
}
#[derive(Clone, Copy, PartialEq)]
pub enum ReferenceKind {
    Image,
    Video,
    Audio,
    VideoAudio,
}
#[derive(Clone, Copy)]
pub struct Reference {
    pub kind: ReferenceKind,
    pub video: Option<usize>,
    pub audio: Option<usize>,
}
/// One network evaluation.
pub struct Step<'a> {
    /// `[channels][frames][height][width]`: 24 video channels, then the audio latent
    /// `[32][2][audio_length]` flattened into further channels and zero padded.
    pub packed: &'a [f32],
    pub width: usize,
    pub height: usize,
    pub frames: usize,
    pub audio_length: usize,
    /// Sampler timestep, 0..1000.
    pub timestep: f32,
    /// Text encoder output `[tokens][5120]`.
    pub context: &'a [f32],
    /// Modality tag per text token; empty means all text.
    pub tags: &'a [i32],
    pub videos: &'a [Video<'a>],
    pub audios: &'a [Audio<'a>],
    /// For key-frame conditioning: 0 marks a first frame, anything else a last frame.
    pub keyframes: &'a [i32],
    pub references: &'a [Reference],
    pub video_shift: f32,
    pub audio_shift: f32,
}

/// A run of tokens sharing one modulation row.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) struct Span {
    pub start: usize,
    pub end: usize,
    pub row: usize,
}
#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) enum Source {
    Text,
    Video(usize),
    Audio(usize),
    TargetAudio,
    TargetVideo,
}
/// Token order, positions and modulation rows of one evaluation.
pub(super) struct Layout {
    /// `[tokens][time, height, width]`.
    pub positions: Vec<[f32; 3]>,
    /// Modulation spans covering every token; `row` is `timestep * 3 + modality`.
    pub spans: Vec<Span>,
    pub order: Vec<(usize, usize, Source)>,
    /// Target spans; `row` is the timestep index.
    pub video: Span,
    pub audio: Span,
    pub timesteps: Vec<f32>,
}
fn video_span(frame: usize) -> f32 {
    FRAME_RESCALE * if frame.is_multiple_of(5) { 1.0 } else { 4.0 }
}
fn axis(size: usize, root_area: f32) -> Vec<f32> {
    let count = size / PATCH;
    let ratio = size as f32 / root_area;
    (0..count)
        .map(|i| (i as f32 * (ratio / count as f32) + (1.0 - ratio) * 0.5) * 32.0)
        .collect()
}
fn timestep_row(values: &mut Vec<f32>, value: f32) -> usize {
    values.iter().position(|v| *v == value).unwrap_or_else(|| {
        values.push(value);
        values.len() - 1
    })
}
impl Layout {
    pub fn new(step: &Step, text: usize, video_time: f32, audio_time: f32) -> Self {
        let mut positions = Vec::new();
        let mut spans = Vec::new();
        let mut order = Vec::new();
        let mut timesteps = Vec::new();
        let root = ((step.height * step.width) as f32).sqrt();
        let (h_axis, w_axis) = (axis(step.height, root), axis(step.width, root));
        for i in 0..text {
            positions.push([i as f32, 0.0, 0.0]);
        }
        order.push((0, text, Source::Text));
        let video_row = timestep_row(&mut timesteps, video_time);
        let audio_row = timestep_row(&mut timesteps, audio_time);
        let condition_row = timestep_row(&mut timesteps, video_time.max(VISUAL_CONDITION_TIME));
        let audio_condition_row = timestep_row(&mut timesteps, audio_time.max(1.0));
        let tag = |i: usize| step.tags.get(i).copied().unwrap_or(1);
        let mut start = 0;
        for i in 1..=text {
            if i == text || (!step.tags.is_empty() && tag(i) != tag(start)) {
                spans.push(Span {
                    start,
                    end: i,
                    row: video_row * 3 + tag(start) as usize,
                });
                start = i;
            }
        }
        let mut row = text;
        let axes = |video: &Video| {
            let root = ((video.width * video.height) as f32).sqrt();
            (axis(video.height, root), axis(video.width, root))
        };
        let audio_positions =
            |positions: &mut Vec<[f32; 3]>, length: usize, cursor: f32, low: f32, high: f32| {
                for w in [low, high] {
                    for t in 0..length {
                        positions.push([cursor + t as f32, 0.0, w]);
                    }
                }
            };
        let mut cursor = text as f32;
        if step.references.is_empty() {
            let duration: f32 = (0..step.frames).map(video_span).sum();
            for (index, video) in step.videos.iter().enumerate() {
                let (h, w) = axes(video);
                let count = video.frames * h.len() * w.len();
                let first = step.keyframes.get(index).is_none_or(|k| *k == 0);
                let time = if first {
                    text as f32
                } else {
                    text as f32 + duration - FRAME_RESCALE
                };
                for _ in 0..video.frames {
                    for h in &h {
                        for w in &w {
                            positions.push([time, *h, *w]);
                        }
                    }
                }
                order.push((row, row + count, Source::Video(index)));
                spans.push(Span {
                    start: row,
                    end: row + count,
                    row: condition_row * 3,
                });
                row += count;
            }
        } else {
            for block in step.references {
                let video = block.video.map(|i| &step.videos[i]);
                let audio = block.audio.map(|i| (i, &step.audios[i]));
                let mut end = cursor;
                if matches!(block.kind, ReferenceKind::Audio | ReferenceKind::VideoAudio) {
                    let (index, audio) = audio.expect("audio reference without audio");
                    let (mut low, mut high) = (w_axis[0], w_axis[w_axis.len() - 1]);
                    if let Some(video) = video {
                        let (_, w) = axes(video);
                        (low, high) = (w[0], w[w.len() - 1]);
                    }
                    let count = audio.length * 2;
                    audio_positions(&mut positions, audio.length, cursor, low, high);
                    order.push((row, row + count, Source::Audio(index)));
                    spans.push(Span {
                        start: row,
                        end: row + count,
                        row: audio_condition_row * 3 + 2,
                    });
                    row += count;
                    end = end.max(cursor + audio.length as f32);
                }
                if block.kind != ReferenceKind::Audio {
                    let video = video.expect("visual reference without video");
                    let (h, w) = axes(video);
                    let count = video.frames * h.len() * w.len();
                    let mut time = cursor;
                    for t in 0..video.frames {
                        for h in &h {
                            for w in &w {
                                positions.push([time, *h, *w]);
                            }
                        }
                        time += video_span(t);
                    }
                    order.push((row, row + count, Source::Video(block.video.unwrap())));
                    spans.push(Span {
                        start: row,
                        end: row + count,
                        row: condition_row * 3,
                    });
                    row += count;
                    end = if block.kind == ReferenceKind::Image {
                        end.max(cursor + 1.0)
                    } else {
                        end.max(time)
                    };
                }
                cursor = end;
            }
        }
        let count = step.audio_length * 2;
        audio_positions(
            &mut positions,
            step.audio_length,
            cursor,
            w_axis[0],
            w_axis[w_axis.len() - 1],
        );
        let audio = Span {
            start: row,
            end: row + count,
            row: audio_row,
        };
        order.push((row, row + count, Source::TargetAudio));
        spans.push(Span {
            start: row,
            end: row + count,
            row: audio_row * 3 + 2,
        });
        row += count;
        for t in 0..step.frames {
            for h in &h_axis {
                for w in &w_axis {
                    positions.push([cursor, *h, *w]);
                }
            }
            cursor += video_span(t);
        }
        let count = step.frames * h_axis.len() * w_axis.len();
        let video = Span {
            start: row,
            end: row + count,
            row: video_row,
        };
        order.push((row, row + count, Source::TargetVideo));
        spans.push(Span {
            start: row,
            end: row + count,
            row: video_row * 3,
        });
        Self {
            positions,
            spans,
            order,
            video,
            audio,
            timesteps,
        }
    }
}

/// Sigma under one schedule shift re-expressed under another.
pub(super) fn shift_sigma(sigma: f32, from: f32, to: f32) -> f32 {
    let base = sigma / (from + sigma * (1.0 - from));
    to * base / (1.0 + (to - 1.0) * base)
}
/// `[frames][height][width]` channel-major latent to token rows `[tokens][channel][2][2]`.
pub(super) fn patchify(video: &Video) -> Vec<f32> {
    let (w2, h2) = (video.width / PATCH, video.height / PATCH);
    let plane = video.width * video.height;
    let mut out = Vec::with_capacity(video.frames * h2 * w2 * PATCH_VALUES);
    for t in 0..video.frames {
        for h in 0..h2 {
            for w in 0..w2 {
                for c in 0..VIDEO_CHANNELS {
                    let at = (c * video.frames + t) * plane;
                    for (dh, dw) in [(0, 0), (0, 1), (1, 0), (1, 1)] {
                        out.push(video.data[at + (h * 2 + dh) * video.width + w * 2 + dw]);
                    }
                }
            }
        }
    }
    out
}
/// Inverse of `patchify`, into `out` (`[24][frames][height][width]`), times `scale`.
pub(super) fn unpatchify(
    rows: &[f32],
    width: usize,
    height: usize,
    frames: usize,
    scale: f32,
    out: &mut [f32],
) {
    let (w2, h2) = (width / PATCH, height / PATCH);
    let plane = width * height;
    let mut values = rows.iter();
    for t in 0..frames {
        for h in 0..h2 {
            for w in 0..w2 {
                for c in 0..VIDEO_CHANNELS {
                    let at = (c * frames + t) * plane;
                    for (dh, dw) in [(0, 0), (0, 1), (1, 0), (1, 1)] {
                        out[at + (h * 2 + dh) * width + w * 2 + dw] =
                            scale * values.next().unwrap();
                    }
                }
            }
        }
    }
}
/// Stereo latent `[32][2][length]` to token rows `[2 * length][32]`, left channel first.
pub(super) fn audio_rows(data: &[f32], length: usize, scale: f32) -> Vec<f32> {
    let mut out = Vec::with_capacity(length * 2 * AUDIO_CHANNELS);
    for side in 0..2 {
        for t in 0..length {
            for c in 0..AUDIO_CHANNELS {
                out.push(scale * data[(c * 2 + side) * length + t]);
            }
        }
    }
    out
}

/// The regular Hadamard rotation of one 256-value group, in the C++ engine's operation order.
pub(super) fn rotate(values: &mut [f32; GROUP]) {
    let scale = 1.0 / (GROUP as f32).sqrt();
    for v in values.iter_mut() {
        *v *= scale;
    }
    let mut stride = 1;
    while stride < GROUP {
        for base in (0..GROUP).step_by(4 * stride) {
            for j in 0..stride {
                let (i0, i1, i2, i3) = (
                    base + j,
                    base + j + stride,
                    base + j + 2 * stride,
                    base + j + 3 * stride,
                );
                let (a, b, c, d) = (values[i0], values[i1], values[i2], values[i3]);
                values[i0] = a + b + c - d;
                values[i1] = a + b - c + d;
                values[i2] = a - b + c + d;
                values[i3] = -a + b + c + d;
            }
        }
        stride *= 4;
    }
}
/// Rotates and rounds every row of `x` to INT8; returns the bytes and one scale per row.
pub(super) fn quantize(x: &[f32], width: usize) -> (Vec<i8>, Vec<f32>) {
    assert!(width.is_multiple_of(GROUP) && x.len().is_multiple_of(width));
    let mut bytes = vec![0i8; x.len()];
    let scales: Vec<f32> = bytes
        .par_chunks_exact_mut(width)
        .zip(x.par_chunks_exact(width))
        .map(|(bytes, row)| {
            let mut rotated = vec![0f32; width];
            let mut top = 0f32;
            for (to, from) in rotated.chunks_exact_mut(GROUP).zip(row.chunks_exact(GROUP)) {
                let to: &mut [f32; GROUP] = to.try_into().unwrap();
                to.copy_from_slice(from);
                rotate(to);
                top = to.iter().fold(top, |m, v| m.max(v.abs()));
            }
            let scale = top / 127.0;
            if scale != 0.0 {
                let inverse = 1.0 / scale;
                for (byte, value) in bytes.iter_mut().zip(&rotated) {
                    *byte = (value * inverse).round_ties_even().clamp(-127.0, 127.0) as i8;
                }
            }
            scale
        })
        .collect();
    (bytes, scales)
}
fn dot(a: &[i8], b: &[i8]) -> i32 {
    let mut sum = 0i32;
    for (a, b) in a.chunks_exact(32).zip(b.chunks_exact(32)) {
        let mut part = 0i32;
        for i in 0..32 {
            part += i32::from(i16::from(a[i]) * i16::from(b[i]));
        }
        sum += part;
    }
    sum
}
/// An INT8 linear layer without bias: `weight` is `[outputs][inputs]` in the rotated basis.
pub(super) fn linear8(x: &[f32], inputs: usize, weight: &[i8], scale: &[f32]) -> Vec<f32> {
    let outputs = scale.len();
    assert!(weight.len() == outputs * inputs);
    let (bytes, scales) = quantize(x, inputs);
    let mut out = vec![0f32; x.len() / inputs * outputs];
    out.par_chunks_exact_mut(outputs)
        .zip(bytes.par_chunks_exact(inputs))
        .zip(&scales)
        .for_each(|((out, row), row_scale)| {
            for ((out, weight), scale) in out.iter_mut().zip(weight.chunks_exact(inputs)).zip(scale)
            {
                *out = dot(row, weight) as f32 * row_scale * scale;
            }
        });
    out
}
/// A float linear layer; `bias` may be empty.
fn linear(x: &[f32], inputs: usize, weight: &[f32], bias: &[f32]) -> Vec<f32> {
    let outputs = weight.len() / inputs;
    let rows = x.len() / inputs;
    assert!(x.len() == rows * inputs && weight.len() == outputs * inputs);
    let mut out = vec![0f32; rows * outputs];
    if !bias.is_empty() {
        for row in out.chunks_exact_mut(outputs) {
            row.copy_from_slice(bias);
        }
    }
    // SAFETY: the three matrices span exactly the slices checked above; the right-hand side
    // is the weight read transposed (row stride 1, column stride `inputs`).
    unsafe {
        gemm::gemm(
            rows,
            outputs,
            inputs,
            out.as_mut_ptr(),
            1,
            outputs as isize,
            true,
            x.as_ptr(),
            1,
            inputs as isize,
            weight.as_ptr(),
            inputs as isize,
            1,
            1.0,
            1.0,
            false,
            false,
            false,
            threads(),
        );
    }
    out
}
fn rms_norm(x: &[f32], weight: &[f32]) -> Vec<f32> {
    let mut out = x.to_vec();
    out.par_chunks_exact_mut(weight.len()).for_each(|row| {
        let mean = row.iter().map(|v| v * v).sum::<f32>() / row.len() as f32;
        let scale = 1.0 / (mean + EPS).sqrt();
        for (v, w) in row.iter_mut().zip(weight) {
            *v *= scale * w;
        }
    });
    out
}
/// `[tokens][3 * FREQUENCIES]` cosine and sine pairs: time, height, then width frequencies.
pub(super) fn rotary(positions: &[[f32; 3]], frequencies: &[f32]) -> Vec<[f32; 2]> {
    let mut out = Vec::with_capacity(positions.len() * PAIRS);
    for position in positions {
        for axis in position {
            for frequency in frequencies {
                let angle = frequency * axis;
                out.push([angle.cos(), angle.sin()]);
            }
        }
    }
    out
}
/// `qkv` is `[tokens][query 7168, key 7168, value 7168]`, each `[heads][128]`; returns
/// `[tokens][heads][128]`. Queries and keys get a weighted unit RMS and, when given, the
/// rotary positions on channel pairs `(i, i + 48)`.
fn attention(
    qkv: &[f32],
    query_weight: &[f32],
    key_weight: &[f32],
    rotary: Option<&[[f32; 2]]>,
) -> Vec<f32> {
    let tokens = qkv.len() / (INNER * 3);
    let heads: Vec<Vec<f32>> = (0..HEADS)
        .into_par_iter()
        .map(|head| {
            let side = |part: usize, weight: Option<&[f32]>| {
                let mut out = vec![0f32; tokens * HEAD];
                for (token, row) in out.chunks_exact_mut(HEAD).enumerate() {
                    row.copy_from_slice(
                        &qkv[token * INNER * 3 + part * INNER + head * HEAD..][..HEAD],
                    );
                    let Some(weight) = weight else { continue };
                    let mean = row.iter().map(|v| v * v).sum::<f32>() / HEAD as f32;
                    let scale = 1.0 / (mean + EPS).sqrt();
                    for (v, w) in row.iter_mut().zip(weight) {
                        *v *= scale * w;
                    }
                    if let Some(rotary) = rotary {
                        for (pair, [cos, sin]) in
                            rotary[token * PAIRS..][..PAIRS].iter().enumerate()
                        {
                            let (a, b) = (row[pair], row[pair + PAIRS]);
                            row[pair] = cos * a - sin * b;
                            row[pair + PAIRS] = sin * a + cos * b;
                        }
                    }
                }
                out
            };
            let (query, key, value) = (
                side(0, Some(query_weight)),
                side(1, Some(key_weight)),
                side(2, None),
            );
            let mut scores = vec![0f32; tokens * tokens];
            // SAFETY: query and key are `tokens x 128`, scores is `tokens x tokens`; the key
            // is read transposed.
            unsafe {
                gemm::gemm(
                    tokens,
                    tokens,
                    HEAD,
                    scores.as_mut_ptr(),
                    1,
                    tokens as isize,
                    false,
                    query.as_ptr(),
                    1,
                    HEAD as isize,
                    key.as_ptr(),
                    HEAD as isize,
                    1,
                    0.0,
                    1.0 / (HEAD as f32).sqrt(),
                    false,
                    false,
                    false,
                    gemm::Parallelism::None,
                );
            }
            for row in scores.chunks_exact_mut(tokens) {
                let top = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let mut sum = 0.0;
                for v in row.iter_mut() {
                    *v = (*v - top).exp();
                    sum += *v;
                }
                let scale = 1.0 / sum;
                row.iter_mut().for_each(|v| *v *= scale);
            }
            let mut out = vec![0f32; tokens * HEAD];
            // SAFETY: scores is `tokens x tokens`, value and out are `tokens x 128`.
            unsafe {
                gemm::gemm(
                    tokens,
                    HEAD,
                    tokens,
                    out.as_mut_ptr(),
                    1,
                    HEAD as isize,
                    false,
                    scores.as_ptr(),
                    1,
                    tokens as isize,
                    value.as_ptr(),
                    1,
                    HEAD as isize,
                    0.0,
                    1.0,
                    false,
                    false,
                    false,
                    gemm::Parallelism::None,
                );
            }
            out
        })
        .collect();
    let mut out = vec![0f32; tokens * INNER];
    for (head, values) in heads.iter().enumerate() {
        for (token, row) in values.chunks_exact(HEAD).enumerate() {
            out[token * INNER + head * HEAD..][..HEAD].copy_from_slice(row);
        }
    }
    out
}
/// `silu(first half) * second half` of each `[2 * HIDDEN]` row.
fn gate(x: &[f32]) -> Vec<f32> {
    let mut out = vec![0f32; x.len() / 2];
    out.par_chunks_exact_mut(HIDDEN)
        .zip(x.par_chunks_exact(HIDDEN * 2))
        .for_each(|(out, row)| {
            for i in 0..HIDDEN {
                let u = row[i];
                out[i] = u / (1.0 + (-u).exp()) * row[HIDDEN + i];
            }
        });
    out
}
/// `x + x * scale + shift` per span, with `parts` `[rows][parts][DIM]`.
fn modulate(
    x: &mut [f32],
    spans: &[Span],
    parts: &[f32],
    count: usize,
    shift: usize,
    scale: usize,
) {
    for span in spans {
        let row = &parts[span.row * count * DIM..][..count * DIM];
        let (shift, scale) = (&row[shift * DIM..][..DIM], &row[scale * DIM..][..DIM]);
        for token in x[span.start * DIM..span.end * DIM].chunks_exact_mut(DIM) {
            for i in 0..DIM {
                token[i] = token[i] + token[i] * scale[i] + shift[i];
            }
        }
    }
}
fn add_gated(x: &mut [f32], update: &[f32], spans: &[Span], parts: &[f32], gate: usize) {
    for span in spans {
        let gate = &parts[(span.row * 6 + gate) * DIM..][..DIM];
        for (token, update) in x[span.start * DIM..span.end * DIM]
            .chunks_exact_mut(DIM)
            .zip(update[span.start * DIM..span.end * DIM].chunks_exact(DIM))
        {
            for i in 0..DIM {
                token[i] += update[i] * gate[i];
            }
        }
    }
}

/// Rows the input projections turn into tokens.
pub(super) enum Rows {
    /// Refined text features, already `[tokens][DIM]`.
    Text(Vec<f32>),
    /// `[tokens][PATCH_VALUES]`.
    Video(Vec<f32>),
    /// `[tokens][AUDIO_CHANNELS]`.
    Audio(Vec<f32>),
}
/// One evaluation's inputs before the input projections.
pub(super) struct Inputs {
    /// Each source's rows in token order.
    pub rows: Vec<Rows>,
    pub layout: Layout,
    /// `[timesteps][TIME_WIDTH]`.
    pub times: Vec<f32>,
    /// `[tokens][PAIRS]` cosine and sine.
    pub rotary: Vec<[f32; 2]>,
}
/// A text refiner: text encoder features to the DiT's text rows.
pub(super) type Refine<'a> = dyn Fn(&[f32]) -> Result<Vec<f32>, &'static str> + 'a;
/// The state of one evaluation before the first block.
pub(super) struct Prepared {
    /// `[tokens][DIM]`.
    pub x: Vec<f32>,
    pub layout: Layout,
    /// `[timesteps][TIME_WIDTH]`.
    pub times: Vec<f32>,
    /// `[tokens][PAIRS]` cosine and sine.
    pub rotary: Vec<[f32; 2]>,
}
pub struct Dit {
    pub(super) file: Safetensors,
    pub(super) layers: usize,
    refiners: usize,
    grid: usize,
    /// The last text encoder output and its refined form.
    refined: std::sync::Mutex<Option<(Vec<f32>, Vec<f32>)>>,
}
impl Dit {
    pub fn open(path: &Path) -> Result<Self, &'static str> {
        let file = Safetensors::open(path)?;
        let (layers, refiners) = (file.count("blocks."), file.count("token_refiner.blocks."));
        // The time table's row count is not fixed by the architecture; probe the common size.
        let grid = [1025usize, 513, 2049, 257]
            .into_iter()
            .find(|grid| file.f32("adaln_t_table", &[*grid, TIME_WIDTH]).is_ok())
            .ok_or("model_file_invalid")?;
        if layers == 0 {
            return Err("model_file_invalid");
        }
        Ok(Self {
            file,
            layers,
            refiners,
            grid,
            refined: std::sync::Mutex::new(None),
        })
    }
    fn linear8(
        &self,
        name: &str,
        x: &[f32],
        inputs: usize,
        outputs: usize,
    ) -> Result<Vec<f32>, &'static str> {
        let weight = self
            .file
            .i8(&format!("{name}.weight"), &[outputs, inputs])?;
        let scale = self
            .file
            .f32(&format!("{name}.weight_scale"), &[outputs, 1])?;
        Ok(linear8(x, inputs, &weight, &scale))
    }
    /// `refine` with the last result kept: the text is the same for every step of a clip.
    /// The copy lives in RAM only, until the next clip's text or `forget`.
    fn refined(&self, context: &[f32], refine: &Refine) -> Result<Vec<f32>, &'static str> {
        let mut kept = self.refined.lock().map_err(|_| "model_state_lost")?;
        if let Some((text, refined)) = kept.as_ref()
            && text == context
        {
            return Ok(refined.clone());
        }
        let refined = refine(context)?;
        *kept = Some((context.to_vec(), refined.clone()));
        Ok(refined)
    }
    /// Drops the kept text features.
    pub fn forget(&self) {
        if let Ok(mut kept) = self.refined.lock() {
            *kept = None;
        }
    }
    /// Text encoder output to model width: projection and the unmodulated refiner blocks.
    pub fn refine(&self, context: &[f32]) -> Result<Vec<f32>, &'static str> {
        let f = &self.file;
        let mut x = self.condition(context)?;
        for layer in 0..self.refiners {
            let name = |part: &str| format!("token_refiner.blocks.{layer}.{part}");
            let h = rms_norm(&x, &f.floats(&name("norm1.weight"), &[DIM])?);
            let qkv = linear(
                &h,
                DIM,
                &f.floats(&name("attn.qkv_proj.weight"), &[INNER * 3, DIM])?,
                &[],
            );
            let mixed = attention(
                &qkv,
                &f.floats(&name("attn.q_norm.weight"), &[HEAD])?,
                &f.floats(&name("attn.k_norm.weight"), &[HEAD])?,
                None,
            );
            let update = linear(
                &mixed,
                INNER,
                &f.floats(&name("attn.out_proj.weight"), &[DIM, INNER])?,
                &[],
            );
            x.iter_mut().zip(&update).for_each(|(x, u)| *x += u);
            let h = rms_norm(&x, &f.floats(&name("norm2.weight"), &[DIM])?);
            let h = linear(
                &h,
                DIM,
                &f.floats(&name("mlp.fc1.weight"), &[HIDDEN * 2, DIM])?,
                &[],
            );
            let update = linear(
                &gate(&h),
                HIDDEN,
                &f.floats(&name("mlp.fc2.weight"), &[DIM, HIDDEN])?,
                &[],
            );
            x.iter_mut().zip(&update).for_each(|(x, u)| *x += u);
        }
        self.refined_norm(&x)
    }
    /// The refiner's input: text encoder features projected to `[tokens][DIM]`.
    pub(super) fn condition(&self, context: &[f32]) -> Result<Vec<f32>, &'static str> {
        let f = &self.file;
        Ok(linear(
            context,
            TEXT_WIDTH,
            &f.floats("condition_proj.weight", &[DIM, TEXT_WIDTH])?,
            &f.floats("condition_proj.bias", &[DIM])?,
        ))
    }
    /// The refiner's last step on its blocks' output.
    pub(super) fn refined_norm(&self, x: &[f32]) -> Result<Vec<f32>, &'static str> {
        Ok(rms_norm(
            x,
            &self
                .file
                .floats("token_refiner.final_norm.weight", &[DIM])?,
        ))
    }
    pub(super) fn refiners(&self) -> usize {
        self.refiners
    }
    /// Time features `[timesteps][8]`: the table read with linear interpolation.
    fn times(&self, timesteps: &[f32]) -> Result<Vec<f32>, &'static str> {
        let table = self.file.f32("adaln_t_table", &[self.grid, TIME_WIDTH])?;
        let mut out = Vec::with_capacity(timesteps.len() * TIME_WIDTH);
        for time in timesteps {
            let position = time.clamp(0.0, 1.0) * (self.grid - 1) as f32;
            let index = (position.floor() as usize).min(self.grid - 2);
            let fraction = position - index as f32;
            for i in 0..TIME_WIDTH {
                let (low, high) = (
                    table[index * TIME_WIDTH + i],
                    table[(index + 1) * TIME_WIDTH + i],
                );
                out.push(low + (high - low) * fraction);
            }
        }
        Ok(out)
    }
    /// Token stream, layout, time features and rotary turns at the first block's input.
    pub(super) fn prepare(&self, step: &Step) -> Result<Prepared, &'static str> {
        let f = &self.file;
        let Inputs {
            rows,
            layout,
            times,
            rotary,
        } = self.inputs(step)?;
        let tokens = layout.positions.len();
        let mut x = Vec::with_capacity(tokens * DIM);
        for rows in rows {
            x.extend(match rows {
                Rows::Text(text) => text,
                Rows::Video(patches) => linear(
                    &patches,
                    PATCH_VALUES,
                    &f.f32("video_patch_proj.weight", &[DIM, PATCH_VALUES])?,
                    &f.f32("video_patch_proj.bias", &[DIM])?,
                ),
                Rows::Audio(sound) => linear(
                    &sound,
                    AUDIO_CHANNELS,
                    &f.f32("audio_patch_proj.weight", &[DIM, AUDIO_CHANNELS])?,
                    &f.f32("audio_patch_proj.bias", &[DIM])?,
                ),
            });
        }
        assert!(x.len() == tokens * DIM);
        Ok(Prepared {
            x,
            layout,
            times,
            rotary,
        })
    }
    /// The input projections' vectors in one run: video weight `[DIM][PATCH_VALUES]`, audio
    /// weight `[DIM][AUDIO_CHANNELS]`, then both biases.
    pub(super) fn embed_vectors(&self) -> Result<Vec<f32>, &'static str> {
        let f = &self.file;
        let mut out = f.f32("video_patch_proj.weight", &[DIM, PATCH_VALUES])?;
        out.extend(f.f32("audio_patch_proj.weight", &[DIM, AUDIO_CHANNELS])?);
        out.extend(f.f32("video_patch_proj.bias", &[DIM])?);
        out.extend(f.f32("audio_patch_proj.bias", &[DIM])?);
        Ok(out)
    }
    /// Everything `prepare` needs short of the input projections.
    pub(super) fn inputs(&self, step: &Step) -> Result<Inputs, &'static str> {
        self.inputs_with(step, &|context| self.refine(context))
    }
    /// `inputs` with the text refiner given: `refine` runs when the text is new.
    pub(super) fn inputs_with(&self, step: &Step, refine: &Refine) -> Result<Inputs, &'static str> {
        let f = &self.file;
        let plane = step.width * step.height * step.frames;
        let (video_values, audio_values) = (
            plane * VIDEO_CHANNELS,
            step.audio_length * 2 * AUDIO_CHANNELS,
        );
        if !step.packed.len().is_multiple_of(plane)
            || step.packed.len() < video_values + audio_values
            || !step.context.len().is_multiple_of(TEXT_WIDTH)
            || step.audio_length == 0
        {
            return Err("video_input_invalid");
        }
        let sigma = (step.timestep / 1000.0).clamp(1e-6, 1.0);
        let audio_sigma = shift_sigma(sigma, step.video_shift, step.audio_shift);
        // The sampler carries the audio latent scaled so both follow the video schedule.
        let carrier = &step.packed[video_values..video_values + audio_values];
        let restore = audio_sigma / sigma;
        let text = step.context.len() / TEXT_WIDTH;
        let layout = Layout::new(step, text, 1.0 - sigma, 1.0 - audio_sigma);
        let mut rows = Vec::with_capacity(layout.order.len());
        for (_, _, source) in &layout.order {
            rows.push(match source {
                Source::Text => Rows::Text(self.refined(step.context, refine)?),
                Source::Video(index) => Rows::Video(patchify(&step.videos[*index])),
                Source::Audio(index) => Rows::Audio(audio_rows(
                    step.audios[*index].data,
                    step.audios[*index].length,
                    1.0,
                )),
                Source::TargetAudio => Rows::Audio(audio_rows(carrier, step.audio_length, restore)),
                Source::TargetVideo => Rows::Video(patchify(&Video {
                    data: &step.packed[..video_values],
                    width: step.width,
                    height: step.height,
                    frames: step.frames,
                })),
            });
        }
        let times = self.times(&layout.timesteps)?;
        let rotary = rotary(&layout.positions, &f.f32("rope.inv_freq", &[FREQUENCIES])?);
        Ok(Inputs {
            rows,
            layout,
            times,
            rotary,
        })
    }
    /// One block's modulation rows `[timesteps][3 modalities][shift, scale, gate, shift,
    /// scale, gate][DIM]`.
    pub(super) fn parts(&self, layer: usize, times: &[f32]) -> Result<Vec<f32>, &'static str> {
        let (weight, bias) = self.modulation(layer)?;
        Ok(linear(times, TIME_WIDTH, &weight, &bias))
    }
    /// The weight `[18 * DIM][TIME_WIDTH]` and bias of one block's modulation projection.
    pub(super) fn modulation(&self, layer: usize) -> Result<(Vec<f32>, Vec<f32>), &'static str> {
        let name = |part: &str| format!("blocks.{layer}.adaln_proj.linear.{part}");
        Ok((
            self.file.floats(&name("weight"), &[DIM * 18, TIME_WIDTH])?,
            self.file.floats(&name("bias"), &[DIM * 18])?,
        ))
    }
    /// Applies block `layer` to the token stream `x`.
    pub(super) fn block(
        &self,
        layer: usize,
        x: &mut [f32],
        parts: &[f32],
        spans: &[Span],
        rotary: &[[f32; 2]],
    ) -> Result<(), &'static str> {
        let f = &self.file;
        let name = |part: &str| format!("blocks.{layer}.{part}");
        let mut h = rms_norm(x, &f.floats(&name("norm1.weight"), &[DIM])?);
        modulate(&mut h, spans, parts, 6, 0, 1);
        let qkv = self.linear8(&name("attn.qkv_proj"), &h, DIM, INNER * 3)?;
        let mixed = attention(
            &qkv,
            &f.floats(&name("attn.q_norm.weight"), &[HEAD])?,
            &f.floats(&name("attn.k_norm.weight"), &[HEAD])?,
            Some(rotary),
        );
        let update = self.linear8(&name("attn.out_proj"), &mixed, INNER, DIM)?;
        add_gated(x, &update, spans, parts, 2);
        let mut h = rms_norm(x, &f.floats(&name("norm2.weight"), &[DIM])?);
        modulate(&mut h, spans, parts, 6, 3, 4);
        let h = self.linear8(&name("mlp.fc1"), &h, DIM, HIDDEN * 2)?;
        let update = self.linear8(&name("mlp.fc2"), &gate(&h), HIDDEN, DIM)?;
        add_gated(x, &update, spans, parts, 5);
        Ok(())
    }
    /// Evaluates the network; returns the packed velocity with `step.packed`'s layout.
    pub fn forward(&self, step: &Step) -> Result<Vec<f32>, &'static str> {
        let Prepared {
            mut x,
            layout,
            times,
            rotary,
        } = self.prepare(step)?;
        for layer in 0..self.layers {
            let parts = self.parts(layer, &times)?;
            self.block(layer, &mut x, &parts, &layout.spans, &rotary)?;
        }
        self.finish(step, &x, &layout, &times)
    }
    /// The output heads' modulation `[timesteps][shift, scale][DIM]`.
    pub(super) fn final_parts(&self, times: &[f32]) -> Result<Vec<f32>, &'static str> {
        let f = &self.file;
        Ok(linear(
            times,
            TIME_WIDTH,
            &f.floats(
                "final_layer.adaln_proj.linear.weight",
                &[DIM * 2, TIME_WIDTH],
            )?,
            &f.floats("final_layer.adaln_proj.linear.bias", &[DIM * 2])?,
        ))
    }
    /// The output heads' vectors in one run: norm weight, video weight `[PATCH_VALUES][DIM]`,
    /// audio weight `[AUDIO_CHANNELS][DIM]`, then both biases.
    pub(super) fn head_vectors(&self) -> Result<Vec<f32>, &'static str> {
        let f = &self.file;
        let mut out = f.floats("final_layer.norm.weight", &[DIM])?;
        for (name, outputs) in [("video_out", PATCH_VALUES), ("audio_out", AUDIO_CHANNELS)] {
            out.extend(f.f32(&format!("final_layer.{name}.weight"), &[outputs, DIM])?);
        }
        for (name, outputs) in [("video_out", PATCH_VALUES), ("audio_out", AUDIO_CHANNELS)] {
            out.extend(f.f32(&format!("final_layer.{name}.bias"), &[outputs])?);
        }
        Ok(out)
    }
    /// The output heads on the last block's token stream `x`: the packed velocity.
    pub(super) fn finish(
        &self,
        step: &Step,
        x: &[f32],
        layout: &Layout,
        times: &[f32],
    ) -> Result<Vec<f32>, &'static str> {
        let f = &self.file;
        let parts = self.final_parts(times)?;
        let norm = f.floats("final_layer.norm.weight", &[DIM])?;
        let finish = |span: Span, name: &str, outputs: usize| -> Result<Vec<f32>, &'static str> {
            let mut h = rms_norm(&x[span.start * DIM..span.end * DIM], &norm);
            let whole = Span {
                start: 0,
                end: span.end - span.start,
                row: span.row,
            };
            modulate(&mut h, &[whole], &parts, 2, 0, 1);
            Ok(linear(
                &h,
                DIM,
                &f.f32(&format!("final_layer.{name}.weight"), &[outputs, DIM])?,
                &f.f32(&format!("final_layer.{name}.bias"), &[outputs])?,
            ))
        };
        let video = finish(layout.video, "video_out", PATCH_VALUES)?;
        let audio = finish(layout.audio, "audio_out", AUDIO_CHANNELS)?;
        Ok(Self::pack(step, &video, &audio))
    }
    /// The packed velocity from the heads' outputs: video `[tokens][PATCH_VALUES]`, audio
    /// `[tokens][AUDIO_CHANNELS]`.
    pub(super) fn pack(step: &Step, video: &[f32], audio: &[f32]) -> Vec<f32> {
        let plane = step.width * step.height * step.frames;
        let (video_values, audio_values) = (
            plane * VIDEO_CHANNELS,
            step.audio_length * 2 * AUDIO_CHANNELS,
        );
        let sigma = (step.timestep / 1000.0).clamp(1e-6, 1.0);
        let audio_sigma = shift_sigma(sigma, step.video_shift, step.audio_shift);
        let audio_scale = step.video_shift / step.audio_shift;
        let carrier = &step.packed[video_values..video_values + audio_values];
        let restore = audio_sigma / sigma;
        let mut out = vec![0f32; step.packed.len()];
        unpatchify(video, step.width, step.height, step.frames, -1.0, &mut out);
        // The audio velocity, converted to the carried latent on the video schedule.
        let (keep, gain) = (1.0 - audio_scale, 1.0 + (audio_scale - 1.0) * audio_sigma);
        for side in 0..2 {
            for t in 0..step.audio_length {
                for c in 0..AUDIO_CHANNELS {
                    let at = (c * 2 + side) * step.audio_length + t;
                    let velocity = -audio[(side * step.audio_length + t) * AUDIO_CHANNELS + c];
                    out[video_values + at] = carrier[at] * restore * keep + velocity * gain;
                }
            }
        }
        out
    }
}

#[cfg(test)]
pub(super) mod tests {
    use super::*;

    #[test]
    fn rotation_is_its_own_inverse_and_keeps_length() {
        let mut values = [0f32; GROUP];
        for (i, v) in values.iter_mut().enumerate() {
            *v = ((i * 37 % 101) as f32 - 50.0) / 17.0;
        }
        let before = values;
        rotate(&mut values);
        let (a, b): (f32, f32) = (
            before.iter().map(|v| v * v).sum(),
            values.iter().map(|v| v * v).sum(),
        );
        assert!((a - b).abs() < 1e-3 * a);
        rotate(&mut values);
        for (a, b) in before.iter().zip(&values) {
            assert!((a - b).abs() < 1e-5, "{a} {b}");
        }
    }
    #[test]
    fn integer_linear_matches_a_float_product() {
        // A weight that is the rotation of a float matrix gives the float product back, up
        // to rounding of activations to 8 bits.
        let (inputs, outputs, rows) = (GROUP * 2, 5, 3);
        let value = |i: usize| (((i * 2654435761usize) >> 7) % 255) as f32 - 127.0;
        let weight: Vec<i8> = (0..outputs * inputs).map(|i| value(i) as i8).collect();
        let scale: Vec<f32> = (0..outputs).map(|i| 0.01 + i as f32 * 0.003).collect();
        let x: Vec<f32> = (0..rows * inputs).map(|i| value(i + 77) / 40.0).collect();
        let got = linear8(&x, inputs, &weight, &scale);
        for row in 0..rows {
            let mut rotated = x[row * inputs..][..inputs].to_vec();
            for group in rotated.chunks_exact_mut(GROUP) {
                rotate(group.try_into().unwrap());
            }
            for out in 0..outputs {
                let want: f32 = (0..inputs)
                    .map(|i| rotated[i] * f32::from(weight[out * inputs + i]))
                    .sum::<f32>()
                    * scale[out];
                let have = got[row * outputs + out];
                assert!(
                    (want - have).abs() < 0.02 * want.abs().max(1.0),
                    "{want} {have}"
                );
            }
        }
    }
    #[test]
    fn patches_round_trip_and_audio_rows_are_channel_last() {
        let (width, height, frames) = (4, 6, 2);
        let data: Vec<f32> = (0..VIDEO_CHANNELS * frames * height * width)
            .map(|i| i as f32)
            .collect();
        let rows = patchify(&Video {
            data: &data,
            width,
            height,
            frames,
        });
        // Token 1 is frame 0, row 0, column 1; channel 0 covers latent columns 2..4.
        assert_eq!(&rows[PATCH_VALUES..PATCH_VALUES + 4], &[2.0, 3.0, 6.0, 7.0]);
        let mut back = vec![0f32; data.len()];
        unpatchify(&rows, width, height, frames, 1.0, &mut back);
        assert_eq!(back, data);
        let audio: Vec<f32> = (0..AUDIO_CHANNELS * 2 * 3).map(|i| i as f32).collect();
        let rows = audio_rows(&audio, 3, 1.0);
        // Row 4 is the right channel at time 1; its channel 2 is index (2 * 2 + 1) * 3 + 1.
        assert_eq!(rows[4 * AUDIO_CHANNELS + 2], 16.0);
    }
    #[test]
    fn layout_orders_text_conditions_audio_then_video() {
        let condition = Video {
            data: &[],
            width: 4,
            height: 4,
            frames: 1,
        };
        let step = Step {
            packed: &[],
            width: 4,
            height: 4,
            frames: 2,
            audio_length: 3,
            timestep: 500.0,
            context: &[],
            tags: &[],
            videos: &[condition],
            audios: &[],
            keyframes: &[],
            references: &[],
            video_shift: 12.0,
            audio_shift: 3.0,
        };
        let layout = Layout::new(&step, 5, 0.5, 0.25);
        assert_eq!(layout.timesteps, vec![0.5, 0.25, 0.999, 1.0]);
        assert_eq!(layout.positions.len(), 5 + 4 + 6 + 8);
        assert_eq!(
            layout.spans,
            vec![
                Span {
                    start: 0,
                    end: 5,
                    row: 1
                },
                Span {
                    start: 5,
                    end: 9,
                    row: 6
                },
                Span {
                    start: 9,
                    end: 15,
                    row: 5
                },
                Span {
                    start: 15,
                    end: 23,
                    row: 0
                },
            ]
        );
        assert_eq!(
            layout.video,
            Span {
                start: 15,
                end: 23,
                row: 0
            }
        );
        assert_eq!(
            layout.audio,
            Span {
                start: 9,
                end: 15,
                row: 1
            }
        );
        // The first-frame condition sits at the clip's first time position, like the first
        // target frame; the second target frame is one short span later.
        assert_eq!(layout.positions[5][0], 5.0);
        assert_eq!(layout.positions[15][0], 5.0);
        assert_eq!(layout.positions[19][0], 5.0 + FRAME_RESCALE);
        // A square latent's axis is centred on 16.
        assert_eq!(layout.positions[15][1..], [0.0, 0.0]);
        assert_eq!(layout.positions[18][1..], [16.0, 16.0]);
    }

    /// Compares one evaluation with the C++ engine's, from tensors captured in a real run.
    #[test]
    #[ignore = "needs the model and a captured step"]
    fn step_matches_cpp_engine() {
        let (Ok(model), Ok(golden)) = (std::env::var("KUNO_DIT"), std::env::var("KUNO_DIT_GOLDEN"))
        else {
            panic!("set KUNO_DIT and KUNO_DIT_GOLDEN (a captured step's path prefix)");
        };
        let captured = Captured::read(&golden);
        let dit = Dit::open(Path::new(&model)).unwrap();
        let started = std::time::Instant::now();
        let got = dit.forward(&captured.step()).unwrap();
        let seconds = started.elapsed().as_secs_f64();
        let plane = captured.shape[0] * captured.shape[1] * captured.shape[2];
        let report = |name: &str, range: std::ops::Range<usize>| {
            let (mut error, mut power) = (0f64, 0f64);
            for i in range {
                error += f64::from(got[i] - captured.out[i]).powi(2);
                power += f64::from(captured.out[i]).powi(2);
            }
            let db = 10.0 * (power / error.max(1e-30)).log10();
            println!("{name}: {db:.1} dB from the C++ engine");
            db
        };
        let video = report("video", 0..plane * VIDEO_CHANNELS);
        let audio = report(
            "audio",
            plane * VIDEO_CHANNELS
                ..plane * VIDEO_CHANNELS + captured.audio_length * 2 * AUDIO_CHANNELS,
        );
        println!("reference evaluation took {seconds:.1} s");
        // The rounding to INT8 in every layer amplifies any difference in arithmetic: this
        // reference with attention operands rounded to half precision, as the C++ engine
        // has them, is itself only 30 dB (video) and 23 dB (audio) from the unrounded one.
        // A structural mistake lands far below these bounds.
        assert!(video > 24.0 && audio > 20.0);
    }
    pub struct Captured {
        pub shape: Vec<usize>,
        pub x: Vec<f32>,
        pub out: Vec<f32>,
        pub timestep: f32,
        pub context: Vec<f32>,
        pub videos: Vec<(Vec<f32>, Vec<usize>)>,
        pub audios: Vec<(Vec<f32>, usize)>,
        pub audio_length: usize,
        pub shifts: [f32; 2],
        pub tags: Vec<i32>,
        pub keyframes: Vec<i32>,
        pub references: Vec<Reference>,
    }
    impl Captured {
        /// `prefix` is `<dir>/stepN`; `<prefix>.txt` lists tensors and settings.
        pub fn read(prefix: &str) -> Self {
            let floats = |name: &str| -> Vec<f32> {
                std::fs::read(format!("{prefix}.{name}.f32"))
                    .unwrap()
                    .chunks_exact(4)
                    .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                    .collect()
            };
            let mut out = Self {
                shape: vec![],
                x: vec![],
                out: vec![],
                timestep: 0.0,
                context: vec![],
                videos: vec![],
                audios: vec![],
                audio_length: 0,
                shifts: [0.0; 2],
                tags: vec![],
                keyframes: vec![],
                references: vec![],
            };
            for line in std::fs::read_to_string(format!("{prefix}.txt"))
                .unwrap()
                .lines()
            {
                let mut words = line.split_whitespace();
                let name = words.next().unwrap();
                let rest: Vec<&str> = words.collect();
                let numbers = || {
                    rest.iter()
                        .map(|w| w.parse::<usize>().unwrap())
                        .collect::<Vec<_>>()
                };
                match name {
                    "x" => (out.x, out.shape) = (floats(name), numbers()),
                    "out" => out.out = floats(name),
                    "timestep" => out.timestep = floats(name)[0],
                    "context" => out.context = floats(name),
                    "audio_length" => out.audio_length = numbers()[0],
                    "shifts" => out.shifts = [rest[0].parse().unwrap(), rest[1].parse().unwrap()],
                    "tags" => out.tags = rest.iter().map(|w| w.parse().unwrap()).collect(),
                    "keyframes" => {
                        out.keyframes = rest.iter().map(|w| w.parse().unwrap()).collect()
                    }
                    "blocks" => {
                        for word in &rest {
                            let parts: Vec<i32> =
                                word.split(':').map(|p| p.parse().unwrap()).collect();
                            out.references.push(Reference {
                                kind: [
                                    ReferenceKind::Image,
                                    ReferenceKind::Video,
                                    ReferenceKind::Audio,
                                    ReferenceKind::VideoAudio,
                                ][parts[0] as usize],
                                video: (parts[1] >= 0).then_some(parts[1] as usize),
                                audio: (parts[2] >= 0).then_some(parts[2] as usize),
                            });
                        }
                    }
                    name if name.starts_with("cond") => out.videos.push((floats(name), numbers())),
                    name if name.starts_with("acond") => {
                        out.audios.push((floats(name), numbers()[0]))
                    }
                    _ => panic!("unknown line {line}"),
                }
            }
            out
        }
        pub fn step(&self) -> Step<'_> {
            // Leaked in tests only: the borrowed views need somewhere to live.
            let videos: &'static [Video] = Box::leak(
                self.videos
                    .iter()
                    .map(|(data, shape)| Video {
                        // SAFETY: test-only; `self` outlives every use of the step.
                        data: unsafe { std::slice::from_raw_parts(data.as_ptr(), data.len()) },
                        width: shape[0],
                        height: shape[1],
                        frames: shape[2],
                    })
                    .collect::<Vec<_>>()
                    .into_boxed_slice(),
            );
            let audios: &'static [Audio] = Box::leak(
                self.audios
                    .iter()
                    .map(|(data, length)| Audio {
                        // SAFETY: as above.
                        data: unsafe { std::slice::from_raw_parts(data.as_ptr(), data.len()) },
                        length: *length,
                    })
                    .collect::<Vec<_>>()
                    .into_boxed_slice(),
            );
            Step {
                packed: &self.x,
                width: self.shape[0],
                height: self.shape[1],
                frames: self.shape[2],
                audio_length: self.audio_length,
                timestep: self.timestep,
                context: &self.context,
                tags: &self.tags,
                videos,
                audios,
                keyframes: &self.keyframes,
                references: &self.references,
                video_shift: self.shifts[0],
                audio_shift: self.shifts[1],
            }
        }
    }
}
