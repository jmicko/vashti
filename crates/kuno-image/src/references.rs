//! Ordered image inputs stay in memory; neither metadata nor paths are passed
//! to the native runtime. Decode bounds apply before allocating pixel storage.
use crate::Result;
use base64::Engine;
pub struct Pixels {
    pub width: u32,
    pub height: u32,
    pub rgba: Vec<u8>,
}
pub fn decode<S: AsRef<str>>(inputs: &[S]) -> Result<Vec<Pixels>> {
    if inputs.len() > 10 {
        return Err("image_reference_count_invalid");
    }
    let mut result = Vec::new();
    let mut total_pixels = 0u64;
    for input in inputs {
        let input = input.as_ref();
        if input.len() > 16 * 1024 * 1024 {
            return Err("image_reference_too_large");
        }
        let bytes = base64::engine::general_purpose::STANDARD
            .decode(input)
            .map_err(|_| "image_reference_base64_invalid")?;
        if !bytes.starts_with(b"\x89PNG\r\n\x1a\n") {
            let pixels = decode_photo(bytes, &mut total_pixels)?;
            result.push(pixels);
            continue;
        }
        let mut decoder = png::Decoder::new(std::io::Cursor::new(bytes));
        decoder.set_ignore_text_chunk(true);
        decoder.set_limits(png::Limits {
            bytes: 64 * 1024 * 1024,
        });
        decoder.set_transformations(png::Transformations::EXPAND | png::Transformations::STRIP_16);
        let mut reader = decoder
            .read_info()
            .map_err(|_| "image_reference_png_invalid")?;
        let info = reader.info();
        let (width, height) = (info.width, info.height);
        let pixels = u64::from(width) * u64::from(height);
        total_pixels += pixels;
        if width == 0
            || height == 0
            || width > 4096
            || height > 4096
            || total_pixels > 32 * 1024 * 1024
            || info.animation_control.is_some()
        {
            return Err("image_reference_dimensions_invalid");
        }
        let size = reader
            .output_buffer_size()
            .filter(|n| *n <= 64 * 1024 * 1024)
            .ok_or("image_reference_too_large")?;
        let mut raw = vec![0; size];
        let frame = reader
            .next_frame(&mut raw)
            .map_err(|_| "image_reference_png_invalid")?;
        reader.finish().map_err(|_| "image_reference_png_invalid")?;
        let mut rgba = Vec::with_capacity(pixels as usize * 4);
        match frame.color_type {
            png::ColorType::Rgba => rgba.extend_from_slice(&raw[..frame.buffer_size()]),
            png::ColorType::Rgb => {
                for p in raw[..frame.buffer_size()].chunks_exact(3) {
                    rgba.extend_from_slice(&[p[0], p[1], p[2], 255]);
                }
            }
            png::ColorType::Grayscale => {
                for &v in &raw[..frame.buffer_size()] {
                    rgba.extend_from_slice(&[v, v, v, 255]);
                }
            }
            png::ColorType::GrayscaleAlpha => {
                for p in raw[..frame.buffer_size()].chunks_exact(2) {
                    rgba.extend_from_slice(&[p[0], p[0], p[0], p[1]]);
                }
            }
            _ => return Err("image_reference_png_invalid"),
        }
        if rgba.len() != pixels as usize * 4 {
            return Err("image_reference_png_invalid");
        }
        result.push(Pixels {
            width,
            height,
            rgba,
        });
    }
    Ok(result)
}
fn decode_photo(bytes: Vec<u8>, total: &mut u64) -> Result<Pixels> {
    use image::ImageDecoder;
    let format = image::guess_format(&bytes).map_err(|_| "image_reference_format_unsupported")?;
    if !matches!(format, image::ImageFormat::Jpeg | image::ImageFormat::WebP) {
        return Err("image_reference_format_unsupported");
    }
    let cursor = std::io::Cursor::new(bytes);
    let mut decoder: Box<dyn ImageDecoder> = match format {
        image::ImageFormat::Jpeg => Box::new(
            image::codecs::jpeg::JpegDecoder::new(cursor)
                .map_err(|_| "image_reference_decode_failed")?,
        ),
        image::ImageFormat::WebP => {
            let decoder = image::codecs::webp::WebPDecoder::new(cursor)
                .map_err(|_| "image_reference_decode_failed")?;
            if decoder.has_animation() {
                return Err("image_reference_animation_unsupported");
            }
            Box::new(decoder)
        }
        _ => unreachable!(),
    };
    let mut limits = image::Limits::default();
    limits.max_image_width = Some(4096);
    limits.max_image_height = Some(4096);
    limits.max_alloc = Some(64 * 1024 * 1024);
    decoder
        .set_limits(limits)
        .map_err(|_| "image_reference_dimensions_invalid")?;
    let (width, height) = decoder.dimensions();
    *total += u64::from(width) * u64::from(height);
    if width == 0 || height == 0 || width > 4096 || height > 4096 || *total > 32 * 1024 * 1024 {
        return Err("image_reference_dimensions_invalid");
    }
    let orientation = decoder
        .orientation()
        .map_err(|_| "image_reference_decode_failed")?;
    let mut image =
        image::DynamicImage::from_decoder(decoder).map_err(|_| "image_reference_decode_failed")?;
    image.apply_orientation(orientation);
    Ok(Pixels {
        width: image.width(),
        height: image.height(),
        rgba: image.into_rgba8().into_raw(),
    })
}

/// Match the supported Comfy TextEncodeQwenImage21 resolution policy: each
/// reference is about one megapixel, aspect-preserved on a 32-pixel grid.
pub fn resize_for_qwen21(images: &mut [Pixels]) -> Result<()> {
    for pixels in images {
        let ratio = f64::from(pixels.width) / f64::from(pixels.height);
        let width = ((1024.0 * 1024.0 * ratio).sqrt() / 32.0)
            .round_ties_even()
            .max(1.0) as u32
            * 32;
        let height = ((1024.0 * 1024.0 / ratio).sqrt() / 32.0)
            .round_ties_even()
            .max(1.0) as u32
            * 32;
        if width > 4096 || height > 4096 {
            return Err("image_reference_aspect_ratio_unsupported");
        }
        if (width, height) == (pixels.width, pixels.height) {
            continue;
        }
        let input = image::RgbaImage::from_raw(
            pixels.width,
            pixels.height,
            std::mem::take(&mut pixels.rgba),
        )
        .ok_or("image_reference_png_invalid")?;
        pixels.rgba =
            image::imageops::resize(&input, width, height, image::imageops::FilterType::Lanczos3)
                .into_raw();
        pixels.width = width;
        pixels.height = height;
    }
    Ok(())
}
#[cfg(test)]
mod tests {
    use super::*;
    fn fixture(pixel: [u8; 4]) -> String {
        let mut bytes = Vec::new();
        {
            let mut e = png::Encoder::new(&mut bytes, 1, 1);
            e.set_color(png::ColorType::Rgba);
            e.set_depth(png::BitDepth::Eight);
            e.write_header().unwrap().write_image_data(&pixel).unwrap();
        }
        base64::engine::general_purpose::STANDARD.encode(bytes)
    }
    #[test]
    fn ordinary_photos_decode_and_resolve_reference_dimensions() {
        for format in [image::ImageFormat::Jpeg, image::ImageFormat::WebP] {
            let im = image::DynamicImage::ImageRgb8(image::RgbImage::from_pixel(
                900,
                591,
                image::Rgb([120, 20, 30]),
            ));
            let mut encoded = std::io::Cursor::new(Vec::new());
            im.write_to(&mut encoded, format).unwrap();
            let mut result =
                decode(&[base64::engine::general_purpose::STANDARD.encode(encoded.into_inner())])
                    .unwrap();
            assert_eq!((result[0].width, result[0].height), (900, 591));
            assert!(result[0].rgba.chunks_exact(4).all(|p| p[3] == 255));
            resize_for_qwen21(&mut result).unwrap();
            assert_eq!((result[0].width, result[0].height), (1248, 832));
        }
    }
    #[test]
    fn ordered_references_preserve_alpha_and_reject_invalid_inputs() {
        let a = fixture([255, 0, 0, 100]);
        let b = fixture([0, 0, 255, 0]);
        let result = decode(&[a.clone(), b]).unwrap();
        assert_eq!(result[0].rgba, [255, 0, 0, 100]);
        assert_eq!(result[1].rgba, [0, 0, 255, 0]);
        assert!(decode(&vec![a.clone(); 11]).is_err());
        assert_eq!(decode(&vec![a; 10]).unwrap().len(), 10);
        assert!(decode(&["/tmp/image.png"]).is_err());
        assert!(decode(&[base64::engine::general_purpose::STANDARD.encode(b"not png")]).is_err());
    }
}
