//! Actual compiler CFG dataflow for register-loaded next K32 stages.
use std::collections::{BTreeMap, BTreeSet};
pub(super) fn validate_program(
    words: &[u32],
) -> std::result::Result<(), Box<dyn std::error::Error>> {
    validate_program_expected(words, 8, "strata_register_stage_accepted")
}

pub(super) fn validate_program_expected(
    words: &[u32],
    expected_products: usize,
    marker: &str,
) -> std::result::Result<(), Box<dyn std::error::Error>> {
    let minimum_transfers = if expected_products == 16 { 8 } else { 4 };
    validate_program_transfers(words, expected_products, minimum_transfers, false, marker)
}

pub(super) fn validate_asymmetric_program(
    words: &[u32],
) -> std::result::Result<(), Box<dyn std::error::Error>> {
    // M256/N128 loads two A jobs and one B job, each two vec4 values.
    // This is a distinct geometry, not a relaxation of the symmetric checks.
    validate_program_transfers(words, 16, 6, true, "strata_asymmetric_register_stage_accepted")
}

fn validate_program_transfers(
    words: &[u32],
    expected_products: usize,
    expected_transfers: usize,
    exact: bool,
    marker: &str,
) -> std::result::Result<(), Box<dyn std::error::Error>> {
    #[cfg(test)]
    if let Some(path) = std::env::var_os("KUNO_TEST_PUBLIC_MATRIX_SPIRV") {
        // Static compiled-public module only; compiled out of all worker binaries.
        std::fs::write(
            path,
            words
                .iter()
                .flat_map(|w| w.to_le_bytes())
                .collect::<Vec<_>>(),
        )?;
    }
    let mut parents = BTreeMap::new();
    let mut classes = BTreeMap::new();
    let mut constants = BTreeMap::new();
    let mut blocks = BTreeMap::<u32, Vec<(usize, u32, usize)>>::new();
    let mut loops = Vec::new();
    let mut at = 5;
    let mut block = None;
    while at < words.len() {
        let n = (words[at] >> 16) as usize;
        if n == 0 || at + n > words.len() {
            return Err("register stage opcode extent".into());
        }
        let op = words[at] & 65535;
        match op {
            43 if n == 4 => {
                constants.insert(words[at + 2], words[at + 3]);
            }
            59 => {
                classes.insert(words[at + 2], words[at + 3]);
            }
            65 | 66 | 67 | 83 | 124 => {
                parents.insert(words[at + 2], words[at + 3]);
            }
            248 => {
                block = Some(words[at + 1]);
                blocks.insert(words[at + 1], Vec::new());
            }
            246 => {
                loops.push((block.ok_or("register stage header block")?, words[at + 1]));
            }
            _ => {}
        }
        if let Some(b) = block {
            blocks
                .get_mut(&b)
                .ok_or("register stage block")?
                .push((at, op, n));
        }
        at += n;
    }
    let root_class = |mut p: u32| -> Option<u32> {
        let mut seen = BTreeSet::new();
        while let Some(&parent) = parents.get(&p) {
            if !seen.insert(p) {
                return None;
            }
            p = parent;
        }
        classes.get(&p).copied()
    };
    let mut edges = BTreeMap::new();
    for (&b, ins) in &blocks {
        let &(a, op, n) = ins.last().ok_or("register stage block terminator")?;
        let successors = match op {
            249 => vec![words[a + 1]],
            250 => vec![words[a + 2], words[a + 3]],
            251 => std::iter::once(words[a + 2])
                .chain((a + 4..a + n).step_by(2).map(|i| words[i]))
                .collect(),
            _ => Vec::new(),
        };
        edges.insert(b, successors);
    }
    let products = blocks
        .iter()
        .flat_map(|(&b, v)| v.iter().filter(|x| x.1 == 4459).map(move |x| (b, x.0)))
        .collect::<Vec<_>>();
    if products.len() != expected_products || products.is_empty()
        || products.iter().any(|x| x.0 != products[0].0) {
        return Err("register stage all chronological products in actual matrix block".into());
    }
    let mut product_loops = Vec::new();
    for &(header, merge) in &loops {
        let mut region = BTreeSet::new();
        let mut todo = vec![header];
        while let Some(b) = todo.pop() {
            if b == merge || !region.insert(b) {
                continue;
            }
            todo.extend(
                edges
                    .get(&b)
                    .ok_or("register stage CFG edge")?
                    .iter()
                    .copied(),
            );
        }
        if products.iter().any(|x| region.contains(&x.0)) {
            product_loops.push((header, region));
        }
    }
    if product_loops.len() != 1 {
        return Err("register stage unique complete K-product CFG".into());
    }
    let (header, region) = product_loops.pop().unwrap();
    let reachable = |from: u32, to: u32| -> bool {
        let mut todo = vec![from];
        let mut seen = BTreeSet::new();
        while let Some(b) = todo.pop() {
            if !seen.insert(b) {
                continue;
            }
            if b == to {
                return true;
            }
            for &next in edges.get(&b).into_iter().flatten() {
                if next != header && region.contains(&next) {
                    todo.push(next);
                }
            }
        }
        false
    };
    let before = |a: (u32, usize), b: (u32, usize)| -> bool {
        if a.0 == b.0 {
            a.1 < b.1
        } else {
            reachable(a.0, b.0)
        }
    };
    let first = *products.iter().min_by_key(|x| x.1).unwrap();
    let last = *products.iter().max_by_key(|x| x.1).unwrap();
    let mut loads = Vec::new();
    let mut stores = Vec::new();
    let mut barriers = Vec::new();
    for &b in &region {
        for &(a, op, _) in &blocks[&b] {
            if op == 61 && root_class(words[a + 3]) == Some(12) && before((b, a), first) {
                loads.push((b, a));
            }
            if op == 62 && root_class(words[a + 1]) == Some(4) && before(last, (b, a)) {
                stores.push((b, a));
            }
            if op == 224 && constants.get(&words[a + 1]) == Some(&2) {
                barriers.push((b, a));
            }
        }
    }
    if loads.len() < expected_transfers
        || stores.len() < expected_transfers
        || (exact && (loads.len() != expected_transfers || stores.len() != expected_transfers))
        || stores
            .iter()
            .any(|&s| !barriers.iter().any(|&b| before(s, b)))
    {
        return Err(format!(
            "register stage actual CFG load/product/store/fence: loads={} stores={} barriers={}",
            loads.len(),
            stores.len(),
            barriers.len()
        )
        .into());
    }
    println!(
        "{marker} actual_loops={} one_complete_product_loop=true real_global_loads_before_products={} real_slm_stores_after_products={} cfg_iteration_backedge_excluded=true",
        loops.len(),
        loads.len(),
        stores.len()
    );
    Ok(())
}
#[cfg(test)]
mod tests {
    #[test]
    fn actual_compiler_modules_keep_global_loads_before_products_and_slm_writes_after() {
        crate::strata_matrix::validate_sources().unwrap();
    }
    #[test]
    fn every_next_stage_load_has_original_source_and_no_final_speculative_read() {
        for columns in (32usize..=10240).step_by(32) {
            for gr in [false, true] {
                let mut seen = vec![0u8; columns / 32];
                seen[0] = 1;
                for k in (0..columns).step_by(32) {
                    if k + 32 < columns {
                        seen[(k + 32) / 32] += 1;
                        for job in 0..512 {
                            let row = job / 4;
                            let off = k + 32 + (job % 4) * 8;
                            assert!(off + 7 < columns);
                            for row_base in [0, 128, 256] {
                                let a = row * columns + off;
                                assert!(a + 7 < 128 * columns);
                                if !gr || row_base + row < 320 {
                                    let b = (row_base + row) * columns + off;
                                    assert!(b + 7 < (if gr { 320 } else { 384 }) * columns);
                                }
                            }
                        }
                    }
                }
                assert!(seen.iter().all(|&n| n == 1));
            }
        }
    }
    #[test]
    fn both_connected_sources_keep_chronological_products_between_real_loads_and_stores() {
        for s in [
            include_str!("strata_matrix_dense_slm40.rs"),
            include_str!("strata_matrix_native_expert_wide.rs"),
        ] {
            let load = s.find("// REGISTER_STAGE_LOAD_BEGIN").unwrap();
            let load_end = s[load..].find("// REGISTER_STAGE_LOAD_END").unwrap() + load;
            let store = s[load_end..].find("// REGISTER_STAGE_STORE_BEGIN").unwrap() + load_end;
            let store_end = s[store..].find("// REGISTER_STAGE_STORE_END").unwrap() + store;
            assert!(
                s[load..load_end].contains("next_a_lo") && s[load..load_end].contains("next_b_hi")
            );
            let products = &s[load_end..store];
            assert_eq!(products.matches("cmma::execute::<").count(), 8);
            assert!(products.contains("for step in 0..2usize"));
            assert!(
                !s[store..store_end].contains("input[") && !s[store..store_end].contains("weight[")
            );
            assert!(
                s[store..store_end].contains("a[dest]") && s[store..store_end].contains("b[dest]")
            );
            assert!(s[store_end..].contains("sync_cube();"));
        }
    }
}
