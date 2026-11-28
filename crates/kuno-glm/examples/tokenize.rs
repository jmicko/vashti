//! Compares the tokenizer with reference ids: `tokenize <shard dir> <text file> <ids json>`
//! (`{"tokens": [...]}`, e.g. from llama.cpp's server); prints the first difference.
use kuno_glm::Model;

fn main() {
    let mut args = std::env::args().skip(1);
    let dir = std::path::PathBuf::from(args.next().expect("shard directory"));
    let text = std::fs::read_to_string(args.next().expect("text file")).unwrap();
    let json = std::fs::read_to_string(args.next().expect("ids json")).unwrap();
    let want: Vec<u32> = json
        .split('[')
        .nth(1)
        .unwrap()
        .split(']')
        .next()
        .unwrap()
        .split(',')
        .map(|t| t.trim().parse().unwrap())
        .collect();
    let mut shards: Vec<_> = std::fs::read_dir(dir)
        .unwrap()
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|e| e == "gguf"))
        .collect();
    shards.sort();
    let model = Model::open(&shards).unwrap();
    let tokenizer = model.tokenizer().unwrap();
    let got = tokenizer.encode(&text, true).unwrap();
    match got.iter().zip(&want).position(|(a, b)| a != b) {
        Some(i) => println!(
            "differ at {i} of {}/{}: got {:?} want {:?}",
            got.len(),
            want.len(),
            &got[i..(i + 8).min(got.len())],
            &want[i..(i + 8).min(want.len())]
        ),
        None if got.len() == want.len() => println!("identical: {} tokens", got.len()),
        None => println!("prefix equal; lengths {} vs {}", got.len(), want.len()),
    }
    if got != want {
        println!(
            "reference decodes to {:?}",
            tokenizer.decode(&want).unwrap()
        );
    }
    let back = tokenizer.decode(&got).unwrap();
    println!(
        "round trip {}",
        if back == text { "exact" } else { "differs" }
    );
}
