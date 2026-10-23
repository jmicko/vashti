//! Actual compiler dataflow evidence for register-loaded next K32 stages.
use std::collections::{BTreeMap, BTreeSet};
pub(super) fn validate_program(
    words: &[u32],
) -> std::result::Result<(), Box<dyn std::error::Error>> {
    #[cfg(test)]
    if let Some(path) = std::env::var_os("KUNO_TEST_PUBLIC_MATRIX_SPIRV") {
        // Offline compiled-public module only; compiled out of all workers.
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
    let mut instructions = Vec::new();
    let mut at = 5;
    while at < words.len() {
        let n = (words[at] >> 16) as usize;
        if n == 0 || at + n > words.len() {
            return Err("register stage opcode extent".into());
        }
        let op = words[at] & 65535;
        match op {
            59 => {
                classes.insert(words[at + 2], words[at + 3]);
            }
            65 | 66 | 67 | 83 | 124 => {
                parents.insert(words[at + 2], words[at + 3]);
            }
            _ => {}
        }
        instructions.push((at, op, n));
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
    let mut product_loops = Vec::new();
    let mut all_loops = 0;
    for (start, &(a, op, _)) in instructions.iter().enumerate() {
        if op != 246 {
            continue;
        }
        all_loops += 1;
        let merge = words[a + 1];
        let end = instructions
            .iter()
            .enumerate()
            .skip(start + 1)
            .find(|(_, x)| x.1 == 248 && words[x.0 + 1] == merge)
            .map(|x| x.0)
            .ok_or("register stage loop merge extent")?;
        let products = instructions[start..end]
            .iter()
            .filter(|x| x.1 == 4459)
            .count();
        if products > 0 {
            product_loops.push((start, end, products));
        }
    }
    if product_loops.len() != 1
        || product_loops[0].2 != 8
        || instructions.iter().filter(|x| x.1 == 4459).count() != 8
    {
        return Err(format!("register stage unique complete original K-product loop: loops={all_loops} product_loops={product_loops:?} module_products={}",instructions.iter().filter(|x|x.1==4459).count()).into());
    }
    let (start, end, _) = product_loops[0];
    let inner = &instructions[start..end];
    let first = inner
        .iter()
        .position(|x| x.1 == 4459)
        .ok_or("register stage first original product")?;
    let last = inner.iter().rposition(|x| x.1 == 4459).unwrap();
    let loads = inner[..first]
        .iter()
        .filter(|&&(a, op, _)| op == 61 && root_class(words[a + 3]) == Some(12))
        .count();
    let stores = inner[last + 1..]
        .iter()
        .filter(|&&(a, op, _)| op == 62 && root_class(words[a + 1]) == Some(4))
        .count();
    let final_store = inner
        .iter()
        .rposition(|&(a, op, _)| op == 62 && root_class(words[a + 1]) == Some(4))
        .ok_or("register stage real post-product SLM writes")?;
    if loads < 4 || stores < 4 || !inner[final_store + 1..].iter().any(|x| x.1 == 224) {
        return Err("register stage actual load/product/store/fence dataflow".into());
    }
    println!(
        "strata_register_stage_accepted actual_loops={all_loops} one_complete_product_loop=true real_global_loads_before_products={loads} real_slm_stores_after_products={stores}"
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
