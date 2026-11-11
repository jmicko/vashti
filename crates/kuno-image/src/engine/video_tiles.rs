//! Clip-level decode for the MiniMax H3 video decoder: the model only sees tiles of at most
//! 16x16 latent positions and 7 latent frames, so a clip is cut into overlapping tiles in
//! space and chunks in time, and the decoded pixels are cross-faded back together. The plan
//! and weights follow the stable-diffusion.cpp reference so both engines stitch alike.
use rayon::prelude::*;

use super::video_vae::{LATENT_CHANNELS, PATCH, PATCH_FRAMES};

const TILE: usize = 16;
const TARGET_OVERLAP: f32 = 0.25;
/// Latent frames per chunk, and how many of them the next chunk repeats.
const CHUNK: usize = 7;
const CHUNK_OVERLAP: usize = 2;
/// Of a full chunk's 28 decoded frames: 3 are warm-up, 17 are kept, 3 are dropped and the
/// last 5 are cross-faded into the next chunk.
const LEAD: usize = 3;
const KEEP_END: usize = 20;
const FADE: usize = 5;

/// Tiles along one axis: start and how many leading positions the last tile skips because it
/// was moved back to end at the edge; plus the overlap between neighbours.
struct Axis {
    size: usize,
    overlap: usize,
    tiles: Vec<(usize, usize)>,
}
fn axis(length: usize) -> Axis {
    let wanted = (TILE as f32 * TARGET_OVERLAP) as usize;
    let stride = TILE - wanted;
    let mut count = length.saturating_sub(wanted) / stride;
    let overshoot = ((count + 1) * stride + wanted) % length;
    if overshoot != stride && overshoot <= count * (TILE / 2 - wanted) {
        count += 1;
    }
    let factor = if count <= 2 {
        if length <= TILE {
            0.0
        } else {
            (2 * TILE - length) as f32 / TILE as f32
        }
    } else {
        (TILE * count - length) as f32 / (TILE * (count - 1)) as f32
    };
    let overlap = (TILE as f32 * factor) as usize;
    let (stride, size) = (TILE - overlap, TILE.min(length));
    let mut tiles = Vec::new();
    let mut start = 0;
    while start < length {
        if start + size >= length {
            tiles.push((length - size, start - (length - size)));
            break;
        }
        tiles.push((start, 0));
        start += stride;
    }
    Axis {
        size,
        overlap,
        tiles,
    }
}
fn smoother(x: f32) -> f32 {
    x * x * x * (x * (6.0 * x - 15.0) + 10.0)
}
/// Fade weights along one axis of a decoded tile, in pixels. `None` entries are skipped.
fn fade(axis: &Axis, start: usize, skip: usize, length: usize) -> Vec<Option<f32>> {
    let (size, overlap) = (axis.size * PATCH, (axis.overlap * PATCH) as f32);
    let (start, skip, length) = (start * PATCH, skip * PATCH, length * PATCH);
    (0..size)
        .map(|i| {
            if i < skip {
                return None;
            }
            let rise = if overlap > 0.0 && start > 0 {
                (i - skip) as f32 / overlap
            } else {
                1.0
            };
            let fall = if overlap > 0.0 && start < length - size {
                (size - i) as f32 / overlap
            } else {
                1.0
            };
            Some(smoother(rise.min(fall).min(1.0)))
        })
        .collect()
}
/// Decoded frames a clip of `frames` latent frames produces.
pub fn output_frames(frames: usize) -> usize {
    if frames <= 1 {
        1
    } else {
        (frames - 2) / 5 * 17 + 5
    }
}
/// Decodes `[24][frames][height][width]` latents (VAE scale) to RGB in 0..=1 as
/// `[3][output_frames][height * 16][width * 16]`. `tile` decodes one tile with the same
/// layouts (4 frames per latent frame).
pub fn decode_clip(
    latent: &[f32],
    width: usize,
    height: usize,
    frames: usize,
    mut tile: impl FnMut(&[f32], usize, usize, usize) -> Result<Vec<f32>, &'static str>,
) -> Result<Vec<f32>, &'static str> {
    let area = width * height;
    if area == 0 || frames == 0 || latent.len() != LATENT_CHANNELS * frames * area {
        return Err("video_latent_invalid");
    }
    let (columns, rows) = (axis(width), axis(height));
    let (out_w, out_h) = (width * PATCH, height * PATCH);
    let plane = out_w * out_h;
    // One time chunk: every spatial tile decoded and cross-faded, `[3][count * 4][plane]`.
    let mut chunk = |first: usize, count: usize| -> Result<Vec<f32>, &'static str> {
        let out_t = count * PATCH_FRAMES;
        let mut out = vec![0f32; 3 * out_t * plane];
        let blend = columns.overlap > 0 || rows.overlap > 0;
        for (y, skip_y) in &rows.tiles {
            for (x, skip_x) in &columns.tiles {
                let mut cut =
                    Vec::with_capacity(LATENT_CHANNELS * count * rows.size * columns.size);
                for channel in 0..LATENT_CHANNELS {
                    for t in first..first + count {
                        // Frames past the clip repeat its last frame.
                        let from = (channel * frames + t.min(frames - 1)) * area;
                        for row in y + 0..y + rows.size {
                            let at = from + row * width + x;
                            cut.extend_from_slice(&latent[at..at + columns.size]);
                        }
                    }
                }
                let pixels = tile(&cut, columns.size, rows.size, count)?;
                let (tile_w, tile_h) = (columns.size * PATCH, rows.size * PATCH);
                if pixels.len() != 3 * out_t * tile_w * tile_h {
                    return Err("video_decode_failed");
                }
                let (fade_x, fade_y) = (
                    fade(&columns, *x, *skip_x, width),
                    fade(&rows, *y, *skip_y, height),
                );
                out.par_chunks_exact_mut(plane)
                    .zip(pixels.par_chunks_exact(tile_w * tile_h))
                    .for_each(|(out, pixels)| {
                        for (iy, weight_y) in fade_y.iter().enumerate() {
                            let Some(weight_y) = weight_y else { continue };
                            let to = &mut out[(y * PATCH + iy) * out_w + x * PATCH..][..tile_w];
                            let from = &pixels[iy * tile_w..][..tile_w];
                            for ((to, from), weight_x) in to.iter_mut().zip(from).zip(&fade_x) {
                                let Some(weight_x) = weight_x else { continue };
                                if blend {
                                    *to += from * weight_y * weight_x;
                                } else {
                                    *to = *from;
                                }
                            }
                        }
                    });
            }
        }
        Ok(out)
    };
    let expected = output_frames(frames);
    let mut out = vec![0f32; 3 * expected * plane];
    if frames == 1 {
        // A single latent frame decodes to four frames; the last one is the image.
        let decoded = chunk(0, 1)?;
        for channel in 0..3 {
            let from = (channel * PATCH_FRAMES + PATCH_FRAMES - 1) * plane;
            out[channel * plane..][..plane].copy_from_slice(&decoded[from..from + plane]);
        }
        return Ok(out);
    }
    // Pad so the chunks tile the clip: starts every 5 latent frames, 7 long.
    let mut total = frames + LEAD;
    total += (5 - total % 5) % 5;
    if total / 5 < 2 {
        total += 5;
    }
    let total = total - LEAD;
    let mut written = 0;
    // Frames held back from the previous chunk for the cross-fade, `[3][held][plane]`.
    let mut held: Vec<f32> = Vec::new();
    let mut first = 0;
    while first < total - CHUNK_OVERLAP {
        let count = CHUNK.min(total - first);
        let last = first + count == total;
        let decoded = chunk(first, count)?;
        let decoded_frames = count * PATCH_FRAMES;
        let frame =
            |t: usize, channel: usize| &decoded[(channel * decoded_frames + t) * plane..][..plane];
        let keep_end = KEEP_END.min(decoded_frames);
        let keep_start = LEAD.min(keep_end);
        let held_frames = held.len() / (3 * plane);
        let extent = FADE.min(held_frames).min(keep_end - keep_start);
        let tail_start = KEEP_END + LEAD;
        let tail = if last && decoded_frames > tail_start {
            tail_start..decoded_frames
        } else {
            0..0
        };
        for (index, t) in (keep_start..keep_end).chain(tail).enumerate() {
            if written == expected {
                break;
            }
            for channel in 0..3 {
                let to = &mut out[(channel * expected + written) * plane..][..plane];
                to.copy_from_slice(frame(t, channel));
                if index < extent {
                    let new = index as f32 / extent as f32;
                    let old = 1.0 - new;
                    let before = &held
                        [(channel * held_frames + held_frames - extent + index) * plane..][..plane];
                    to.par_iter_mut()
                        .zip(before)
                        .for_each(|(to, before)| *to = before * old + *to * new);
                }
            }
            written += 1;
        }
        held.clear();
        if !last && decoded_frames > tail_start {
            let count = decoded_frames - tail_start;
            held.reserve(3 * count * plane);
            for channel in 0..3 {
                for t in tail_start..decoded_frames {
                    held.extend_from_slice(frame(t, channel));
                }
            }
        }
        first += CHUNK - CHUNK_OVERLAP;
    }
    if written != expected {
        return Err("video_decode_failed");
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn starts(length: usize) -> (Vec<(usize, usize)>, usize) {
        let axis = axis(length);
        (axis.tiles, axis.overlap)
    }
    #[test]
    fn tile_plan_matches_the_reference_engine() {
        // 832x544 pixels: the reference logs 4 by 3 tiles with overlaps 0.25 and 0.4375.
        assert_eq!(starts(52), (vec![(0, 0), (12, 0), (24, 0), (36, 0)], 4));
        assert_eq!(starts(34), (vec![(0, 0), (9, 0), (18, 0)], 7));
        assert_eq!(starts(16), (vec![(0, 0)], 0));
        assert_eq!(starts(9), (vec![(0, 0)], 0));
        // Two tiles share whatever is left over.
        assert_eq!(starts(20), (vec![(0, 0), (4, 0)], 12));
        for length in 1..200 {
            let axis = axis(length);
            let mut covered = 0;
            for (start, skip) in &axis.tiles {
                assert!(
                    start + skip <= covered && start + axis.size <= length,
                    "{length}"
                );
                covered = start + axis.size;
            }
            assert_eq!(covered, length);
        }
    }
    #[test]
    fn fades_sum_to_one_everywhere() {
        // A decoder that paints every pixel 1.0: any seam error shows up as a value off 1.
        for (width, height, frames) in [
            (52, 34, 17),
            (20, 16, 2),
            (9, 40, 1),
            (33, 17, 12),
            (16, 16, 7),
        ] {
            let latent = vec![0f32; LATENT_CHANNELS * frames * width * height];
            let out = decode_clip(&latent, width, height, frames, |_, w, h, t| {
                Ok(vec![1.0; 3 * t * 4 * w * 16 * h * 16])
            })
            .unwrap();
            assert_eq!(
                out.len(),
                3 * output_frames(frames) * width * 16 * height * 16
            );
            let worst = out.iter().fold(0f32, |m, v| m.max((v - 1.0).abs()));
            assert!(worst < 1e-5, "{width}x{height}x{frames}: {worst}");
        }
    }
    #[test]
    fn frames_come_from_the_right_latent_frames_and_positions() {
        // The fake decoder writes, for every pixel, the value of latent channel 0 at the
        // pixel's latent position plus its frame offset inside the latent frame / 8.
        let (width, height, frames) = (20, 18, 12);
        let area = width * height;
        let mut latent = vec![0f32; LATENT_CHANNELS * frames * area];
        for t in 0..frames {
            for i in 0..area {
                latent[t * area + i] = t as f32 + i as f32 / 1024.0;
            }
        }
        let out = decode_clip(&latent, width, height, frames, |cut, w, h, t| {
            let mut pixels = vec![0f32; 3 * t * 4 * w * 16 * h * 16];
            for (index, value) in pixels.iter_mut().enumerate() {
                let (x, y, frame) = (
                    index % (w * 16),
                    index / (w * 16) % (h * 16),
                    index / (w * h * 256) % (t * 4),
                );
                *value = cut[(frame / 4 * h + y / 16) * w + x / 16] + (frame % 4) as f32 / 8.0;
            }
            Ok(pixels)
        })
        .unwrap();
        let expected = output_frames(frames);
        assert_eq!(expected, 39);
        // Each chunk contributes 17 of every 20 decoded frames (3 warm-up frames are dropped),
        // and the final chunk also contributes its last 5.
        for f in 0..expected {
            for (x, y) in [(0, 0), (319, 287), (100, 200), (191, 15)] {
                let got = out[(f * height * 16 + y) * width * 16 + x];
                let decoded = f + 3 + 3 * (f / 17);
                let want = (decoded / 4) as f32
                    + (y / 16 * width + x / 16) as f32 / 1024.0
                    + (decoded % 4) as f32 / 8.0;
                assert!(
                    (got - want).abs() < 1e-4,
                    "frame {f} at {x},{y}: {got} vs {want}"
                );
            }
        }
    }
    #[test]
    fn invalid_latents_are_rejected() {
        let fail = |_: &[f32], _, _, _| Err("unused");
        assert_eq!(
            decode_clip(&[0.0; 5], 1, 1, 1, fail),
            Err("video_latent_invalid")
        );
        assert_eq!(decode_clip(&[], 0, 4, 1, fail), Err("video_latent_invalid"));
    }
}
