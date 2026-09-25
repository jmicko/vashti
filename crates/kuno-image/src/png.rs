use crate::{MAX_METADATA_BYTES, MAX_PNG_BYTES, Result};
use std::io::Read;
fn split(bytes: &[u8]) -> Result<(&[u8], &[u8])> {
    let n = bytes
        .iter()
        .position(|b| *b == 0)
        .ok_or("png_text_invalid")?;
    Ok((&bytes[..n], &bytes[n + 1..]))
}
fn decompress(bytes: &[u8]) -> Result<Vec<u8>> {
    let mut out = Vec::new();
    let mut decoder = flate2::read::ZlibDecoder::new(bytes).take((MAX_METADATA_BYTES + 1) as u64);
    decoder
        .read_to_end(&mut out)
        .map_err(|_| "png_text_compression_invalid")?;
    if out.len() > MAX_METADATA_BYTES {
        return Err("image_metadata_too_large");
    }
    Ok(out)
}
pub(crate) fn prompt(bytes: &[u8]) -> Result<Vec<u8>> {
    if bytes.len() > MAX_PNG_BYTES {
        return Err("image_file_too_large");
    }
    if !bytes.starts_with(b"\x89PNG\r\n\x1a\n") {
        return Err("image_not_png");
    }
    let mut at = 8usize;
    let mut count = 0;
    let mut found = None;
    let mut ended = false;
    let mut header = false;
    while at < bytes.len() {
        count += 1;
        if count > 16384 || bytes.len() - at < 12 {
            return Err("png_chunks_invalid");
        }
        let size = u32::from_be_bytes(bytes[at..at + 4].try_into().unwrap()) as usize;
        let end = at
            .checked_add(size)
            .and_then(|n| n.checked_add(12))
            .filter(|e| *e <= bytes.len())
            .ok_or("png_chunks_invalid")?;
        let kind = &bytes[at + 4..at + 8];
        let data = &bytes[at + 8..end - 4];
        let crc = u32::from_be_bytes(bytes[end - 4..end].try_into().unwrap());
        if crc32fast::hash(&bytes[at + 4..end - 4]) != crc {
            return Err("png_checksum_invalid");
        }
        if !header {
            if kind != b"IHDR" || size != 13 {
                return Err("png_header_invalid");
            }
            header = true;
        } else if kind == b"IHDR" {
            return Err("png_header_invalid");
        }
        if kind == b"IEND" {
            if size != 0 || end != bytes.len() {
                return Err("png_end_invalid");
            }
            ended = true;
            break;
        }
        if matches!(kind, b"tEXt" | b"zTXt" | b"iTXt") {
            let (key, rest) = split(data)?;
            if key == b"prompt" {
                if found.is_some() {
                    return Err("comfy_metadata_ambiguous");
                }
                if rest.len() > MAX_METADATA_BYTES {
                    return Err("image_metadata_too_large");
                }
                let text = match kind {
                    b"tEXt" => rest.to_vec(),
                    b"zTXt" => {
                        if rest.first() != Some(&0) {
                            return Err("png_text_compression_invalid");
                        }
                        decompress(&rest[1..])?
                    }
                    _ => {
                        if rest.len() < 2 || rest[0] > 1 || rest[1] != 0 {
                            return Err("png_text_compression_invalid");
                        }
                        let (_, rest2) = split(&rest[2..])?;
                        let (_, text) = split(rest2)?;
                        if rest[0] == 1 {
                            decompress(text)?
                        } else {
                            text.to_vec()
                        }
                    }
                };
                if text.len() > MAX_METADATA_BYTES {
                    return Err("image_metadata_too_large");
                }
                found = Some(text);
            }
        }
        at = end;
    }
    if !ended {
        return Err("png_end_missing");
    }
    found.ok_or("comfy_metadata_missing")
}
#[cfg(test)]
pub(crate) fn chunk(out: &mut Vec<u8>, kind: &[u8; 4], data: &[u8]) {
    out.extend_from_slice(&(data.len() as u32).to_be_bytes());
    let start = out.len();
    out.extend_from_slice(kind);
    out.extend_from_slice(data);
    let crc = crc32fast::hash(&out[start..]);
    out.extend_from_slice(&crc.to_be_bytes());
}
#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    fn png(kind: &[u8; 4], data: &[u8]) -> Vec<u8> {
        let mut p = b"\x89PNG\r\n\x1a\n".to_vec();
        chunk(&mut p, b"IHDR", &[0; 13]);
        chunk(&mut p, kind, data);
        chunk(&mut p, b"IEND", &[]);
        p
    }
    #[test]
    fn text_compressed_text_and_international_text_agree() {
        let body = b"{\"node\":42}";
        let mut z = flate2::write::ZlibEncoder::new(Vec::new(), flate2::Compression::default());
        z.write_all(body).unwrap();
        let z = z.finish().unwrap();
        for (kind, prefix, bytes) in [
            (b"tEXt", b"prompt\0".as_slice(), body.as_slice()),
            (b"zTXt", b"prompt\0\0".as_slice(), z.as_slice()),
            (b"iTXt", b"prompt\0\x01\0\0\0".as_slice(), z.as_slice()),
        ] {
            let mut d = prefix.to_vec();
            d.extend_from_slice(bytes);
            assert_eq!(prompt(&png(kind, &d)).unwrap(), body);
        }
    }
    #[test]
    fn bounds_crc_truncation_and_duplicates_are_checked() {
        let p = png(b"tEXt", b"prompt\0{}");
        for n in 0..p.len() {
            assert!(prompt(&p[..n]).is_err());
        }
        let mut corrupt = p.clone();
        corrupt[45] ^= 1;
        assert_eq!(prompt(&corrupt).unwrap_err(), "png_checksum_invalid");
        let mut duplicate = p[..p.len() - 12].to_vec();
        chunk(&mut duplicate, b"tEXt", b"prompt\0{}");
        chunk(&mut duplicate, b"IEND", &[]);
        assert_eq!(prompt(&duplicate).unwrap_err(), "comfy_metadata_ambiguous");
        let mut z = flate2::write::ZlibEncoder::new(Vec::new(), flate2::Compression::best());
        z.write_all(&vec![b'a'; MAX_METADATA_BYTES + 1]).unwrap();
        let mut d = b"prompt\0\0".to_vec();
        d.extend(z.finish().unwrap());
        assert_eq!(
            prompt(&png(b"zTXt", &d)).unwrap_err(),
            "image_metadata_too_large"
        );
    }
}
