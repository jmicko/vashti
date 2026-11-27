//! `decode_check <raw rows> <expected f32> <ggml type id> <row elements>`: kuno-text's decoder
//! against values dequantized elsewhere (gguf-py), as the worst absolute difference.
fn main() {
    let a: Vec<String> = std::env::args().skip(1).collect();
    let raw = std::fs::read(&a[0]).unwrap();
    let want: Vec<f32> = std::fs::read(&a[1])
        .unwrap()
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect();
    let kind = kuno_text::TensorType::from_gguf_id(a[2].parse().unwrap()).unwrap();
    let n: usize = a[3].parse().unwrap();
    let row_bytes = raw.len() / (want.len() / n);
    let mut worst = 0f32;
    for (r, packed) in raw.chunks_exact(row_bytes).enumerate() {
        let mut out = vec![0f32; n];
        kind.decode(packed, &mut out).unwrap();
        for (g, w) in out.iter().zip(&want[r * n..]) {
            worst = worst.max((g - w).abs());
        }
    }
    println!(
        "{kind:?}: worst difference {worst:e} over {} values",
        want.len()
    );
}
