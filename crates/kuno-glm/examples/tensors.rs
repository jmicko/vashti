//! `tensors <shard dir> <prefix>`: names, types and shapes of the tensors starting with prefix.
fn main() {
    let mut args = std::env::args().skip(1);
    let dir = std::path::PathBuf::from(args.next().expect("shard directory"));
    let prefix = args.next().unwrap_or_default();
    let mut shards: Vec<_> = std::fs::read_dir(&dir)
        .unwrap()
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|e| e == "gguf"))
        .collect();
    shards.sort();
    let model = kuno_glm::Model::open(&shards).unwrap();
    for (name, info) in model.gguf.tensors() {
        if name.starts_with(&prefix) {
            println!(
                "{name} {:?} {} x {} ({} MB)",
                info.dtype(),
                info.row_count(),
                info.row_len(),
                info.byte_len() / 1_000_000
            );
        }
    }
}
