//! Bounded model-geometry inspection; never prints templates, tokens or weights.

use std::collections::BTreeMap;

use kuno_text::{GgufFile, GgufLimits};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args_os().skip(1);
    let path = args.next().ok_or("usage: inspect MODEL.gguf")?;
    if args.next().is_some() {
        return Err("usage: inspect MODEL.gguf".into());
    }
    let model = GgufFile::open(path, GgufLimits::default())?;
    println!("GGUF version: {}", model.version());
    println!("File bytes: {}", model.file_len());
    println!("Tensor data offset: {}", model.data_offset());
    println!("Metadata entries: {}", model.metadata().len());
    println!("Tensors: {}", model.tensors().len());
    let mut types = BTreeMap::new();
    for tensor in model.tensors().values() {
        let count = types
            .entry(tensor.dtype().gguf_id())
            .or_insert((0_u64, 0_u64));
        count.0 += 1;
        count.1 += tensor.byte_len();
    }
    for (kind, (count, bytes)) in types {
        println!("Tensor type {kind}: count={count}, bytes={bytes}");
    }
    if let Some(ple) = model.tensor("per_layer_token_embd.weight") {
        let info = ple.info();
        println!(
            "PLE: dimensions={:?}, type={}, row_bytes={}, tensor_bytes={}",
            info.dimensions(),
            info.dtype().gguf_id(),
            info.row_bytes(),
            info.byte_len()
        );
    }
    model.verify_file_unchanged()?;
    Ok(())
}
