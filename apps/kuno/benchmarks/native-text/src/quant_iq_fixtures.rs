//! Fixed public independently generated C oracle bytes for offline source tests
//! and the bounded IQ operator calibration. Never reads runtime/user content.

use kuno_text::TensorType;

pub(crate) fn records(dtype: u32) -> (&'static [u8], usize) {
    match dtype {
        super::IQ3_XXS => (
            include_bytes!("../../../../../crates/kuno-text/tests/fixtures/iq-b11232/iq3_xxs.bin"),
            4,
        ),
        super::IQ4_NL => (
            include_bytes!("../../../../../crates/kuno-text/tests/fixtures/iq-b11232/iq4_nl.bin"),
            2,
        ),
        super::IQ4_XS => (
            include_bytes!("../../../../../crates/kuno-text/tests/fixtures/iq-b11232/iq4_xs.bin"),
            8,
        ),
        _ => panic!("public IQ fixture dtype"),
    }
}

pub(crate) fn packed(dtype: u32, rows: usize, columns: usize, phase: usize) -> Vec<u8> {
    let kind = TensorType::from_gguf_id(dtype).unwrap();
    let width = kind.block_elements() as usize;
    let bytes = kind.block_bytes() as usize;
    assert!(columns.is_multiple_of(width));
    let (fixture, ordinary_blocks) = records(dtype);
    let stride = bytes + width * 4;
    let mut packed = Vec::new();
    for block in 0..rows * columns / width {
        let first = ((block + phase) % ordinary_blocks) * stride;
        packed.extend_from_slice(&fixture[first..first + bytes]);
    }
    packed
}

/// Independent pinned-C expected values; do not call the Rust quant decoder.
pub(crate) fn oracle(dtype: u32, rows: usize, columns: usize, phase: usize) -> Vec<f32> {
    let kind = TensorType::from_gguf_id(dtype).unwrap();
    let width = kind.block_elements() as usize;
    let bytes = kind.block_bytes() as usize;
    assert!(columns.is_multiple_of(width));
    let (fixture, ordinary_blocks) = records(dtype);
    let stride = bytes + width * 4;
    let mut expected = Vec::new();
    for block in 0..rows * columns / width {
        let first = ((block + phase) % ordinary_blocks) * stride + bytes;
        expected.extend(
            fixture[first..first + width * 4]
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes(b.try_into().unwrap())),
        );
    }
    expected
}
