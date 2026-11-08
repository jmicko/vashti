//! Coupled prompt QK byte extraction, paired cache reads and exact finite checks.
//! Same complete storage owner and 264-word rows; no allocation/view narrowing.
const TAG: &str = "// STRATA_QK_PACKED_CACHE: original eight signed bytes and half scale.\n";
const SIGNED: &str = "fn signed_code(bits:u32,index:u32)->i32 {return (bitcast<i32>((bits>>((index&3u)*8u))<<24u))>>24u;}";
const EXTRACT: &str = "fn signed_code(bits:u32,index:u32)->i32 {return extractBits(bitcast<i32>(bits),(index&3u)*8u,8u);}";
const CACHE: &str = "var<storage,read> cache:array<u32>;";
const PAIRS: &str = "var<storage,read> cache_pairs:array<vec2<u32>>;\nfn cache_word(i:u32)->u32 {return cache_pairs[i/2u][i&1u];}";
fn once(s: &mut String, old: &str, new: &str) {
    assert_eq!(s.matches(old).count(), 1, "packed QK producer drift: {old}");
    *s = s.replacen(old, new, 1);
}
pub(super) fn source(mut s: String) -> String {
    assert!(s.contains("STRATA_QK_COMPACT"));
    assert!(!s.contains("STRATA_QK_PACKED_CACHE"));
    once(&mut s, SIGNED, EXTRACT);
    once(&mut s, CACHE, PAIRS);
    for i in [
        "token*CACHE_WORDS+offset/4u",
        "token*CACHE_WORDS+256u+offset/128u",
    ] {
        once(&mut s, &format!("cache[{i}]"), &format!("cache_word({i})"));
    }
    for cell in 0..2 {
        let old = format!(
            "let kword0=cache[token{cell}*CACHE_WORDS+koff/4u];\nlet kword1=cache[token{cell}*CACHE_WORDS+koff/4u+1u];"
        );
        let new = format!(
            "let kwords=cache_pairs[(token{cell}*CACHE_WORDS+koff/4u)/2u];\nlet kword0=kwords.x;\nlet kword1=kwords.y;"
        );
        once(&mut s, &old, &new);
        once(
            &mut s,
            &format!("cache[token{cell}*CACHE_WORDS+256u+koff/128u]"),
            &format!("cache_word(token{cell}*CACHE_WORDS+256u+koff/128u)"),
        );
        // Every finite half scale times any signed byte is finite F32. Any
        // nonfinite half scale makes EVERY product nonfinite, including 0*Inf.
        // Check the shared scale once; retain all eight actual products, the
        // exact dot/FMA order, score/softmax/PV checks and original fault owner.
        let scale = "let kscale=select(kscales.x,kscales.y,((koff/64u)&1u)!=0u);";
        // Two occurrences, one in each original guarded cell: locate by kwords.
        let begin = s.find(&new).unwrap();
        let at = begin + s[begin..].find(scale).unwrap() + scale.len();
        s.insert_str(at, "\nif !finite(kscale) {atomicOr(&broken,1u);}");
        for j in 0..8 {
            once(
                &mut s,
                &format!("if !finite(k{cell}_{j}) {{atomicOr(&broken,1u);}}\n"),
                "",
            );
        }
    }
    assert!(!s.contains("cache["));
    s.insert_str(0, TAG);
    s
}

#[cfg(test)]
mod tests {
    use super::super::{
        Kernel, prompt_compact, prompt_fma, prompt_local_fault, prompt_paired, prompt_pv_gather,
        prompt_specialize, prompt_uniform,
    };
    use super::*;
    fn original(k: Kernel) -> String {
        let s = prompt_specialize::source(k.scalar_source());
        let s = prompt_pv_gather::source(prompt_paired::source(s));
        prompt_compact::source(prompt_uniform::source(prompt_fma::source(s)))
    }
    #[test]
    fn complete_inverse_keeps_all_other_arithmetic_owners_causal_checks_and_barriers() {
        for k in [Kernel::PromptAttention, Kernel::PromptAttentionFault] {
            let old = original(k);
            let new = source(old.clone());
            let mut restored = new.strip_prefix(TAG).unwrap().to_owned();
            once(&mut restored, EXTRACT, SIGNED);
            once(&mut restored, PAIRS, CACHE);
            for i in [
                "token*CACHE_WORDS+offset/4u",
                "token*CACHE_WORDS+256u+offset/128u",
            ] {
                once(
                    &mut restored,
                    &format!("cache_word({i})"),
                    &format!("cache[{i}]"),
                );
            }
            for cell in 0..2 {
                let n = format!(
                    "let kwords=cache_pairs[(token{cell}*CACHE_WORDS+koff/4u)/2u];\nlet kword0=kwords.x;\nlet kword1=kwords.y;"
                );
                let o = format!(
                    "let kword0=cache[token{cell}*CACHE_WORDS+koff/4u];\nlet kword1=cache[token{cell}*CACHE_WORDS+koff/4u+1u];"
                );
                once(&mut restored, &n, &o);
                once(
                    &mut restored,
                    &format!("cache_word(token{cell}*CACHE_WORDS+256u+koff/128u)"),
                    &format!("cache[token{cell}*CACHE_WORDS+256u+koff/128u]"),
                );
                let begin = restored.find(&o).unwrap();
                let at = begin
                    + restored[begin..]
                        .find("\nif !finite(kscale) {atomicOr(&broken,1u);}")
                        .unwrap();
                restored.replace_range(
                    at..at + "\nif !finite(kscale) {atomicOr(&broken,1u);}".len(),
                    "",
                );
                for j in 0..8 {
                    let product = format!(
                        "k{cell}_{j}=f32(signed_code(kword{},{}u))*kscale;\n",
                        j / 4,
                        j % 4
                    );
                    once(
                        &mut restored,
                        &product,
                        &format!("{product}if !finite(k{cell}_{j}) {{atomicOr(&broken,1u);}}\n"),
                    );
                }
            }
            assert_eq!(restored, old);
            assert!(std::panic::catch_unwind(|| source(new)).is_err());
        }
    }
    #[test]
    fn all_half_scales_and_signed_codes_have_exactly_the_original_fault_decision() {
        for bits in 0..=u16::MAX {
            let scale = half::f16::from_bits(bits).to_f32();
            for code in -128i32..128 {
                assert_eq!(
                    (scale * (code as f32)).is_finite(),
                    scale.is_finite(),
                    "scale={bits:x} code={code}"
                );
            }
        }
        // Actual original byte shift/extract sign extension for all positions,
        // bytes and surrounding poison; all finite half products fit exactly
        // in F32 (at most 19 significant bits), no new contraction introduced.
        for byte in 0..256u32 {
            for i in 0..4u32 {
                for other in [0, u32::MAX, 0x55aa1234] {
                    let mask = 255u32 << (i * 8);
                    let word = (other & !mask) | (byte << (i * 8));
                    let old = (((word >> (i * 8)) << 24) as i32) >> 24;
                    let extracted = (word >> (i * 8)) & 255;
                    let new = if extracted & 128 != 0 {
                        extracted as i32 - 256
                    } else {
                        extracted as i32
                    };
                    assert_eq!(old, new);
                    assert_eq!(new, byte as u8 as i8 as i32);
                }
            }
        }
    }
    #[test]
    fn full_row_owner_and_every_pair_access_are_original_in_bounds_words() {
        assert_eq!(crate::strata_qsa::CACHE_WORDS, 264);
        for token in [0u32, 1, 3, 31, 2047, 32767] {
            for h in 0..4u32 {
                for lane in 0..32u32 {
                    let off = h * 256 + lane * 8;
                    let word = token * 264 + off / 4;
                    assert_eq!(word % 2, 0);
                    assert_eq!(word / 2 * 2, word);
                    assert_eq!(word / 2 * 2 + 1, word + 1);
                    let scale = token * 264 + 256 + off / 128;
                    assert_eq!(scale / 2 * 2 + scale % 2, scale);
                    assert!(scale / 2 * 2 >= token * 264 && scale / 2 * 2 + 1 < (token + 1) * 264);
                }
            }
        }
    }
    #[test]
    fn actual_modules_have_vector_loads_signed_extraction_and_original_fmas_geometry_and_faults() {
        use wgpu::naga as n;
        for k in [Kernel::PromptAttention, Kernel::PromptAttentionFault] {
            let old = original(k);
            let new = source(old.clone());
            let finish = |s| {
                if k == Kernel::PromptAttentionFault {
                    prompt_local_fault::source(k, s)
                } else {
                    s
                }
            };
            let (before, ow) = prompt_specialize::tests::compiled(&finish(old));
            let (after, nw) = prompt_specialize::tests::compiled(&finish(new.clone()));
            assert_eq!(
                before.entry_points[0].workgroup_size,
                after.entry_points[0].workgroup_size
            );
            let globals = |m: &n::Module| {
                m.global_variables
                    .iter()
                    .map(|(_, g)| (g.binding.clone(), g.space))
                    .collect::<Vec<_>>()
            };
            assert_eq!(globals(&before), globals(&after));
            let mut vectors = std::collections::BTreeSet::new();
            let mut loads = 0;
            let mut extract = 0;
            let count = |w: &[u32], op: u32| {
                let mut a = 5;
                let mut n = 0;
                while a < w.len() {
                    let size = (w[a] >> 16) as usize;
                    assert!(size > 0);
                    n += usize::from(w[a] & 65535 == op);
                    a += size;
                }
                n
            };
            let mut a = 5;
            while a < nw.len() {
                let size = (nw[a] >> 16) as usize;
                let op = nw[a] & 65535;
                if op == 23 && nw[a + 3] == 2 {
                    vectors.insert(nw[a + 1]);
                }
                if op == 61 && vectors.contains(&nw[a + 1]) {
                    loads += 1;
                }
                if op == 202 {
                    extract += 1;
                } // OpBitFieldSExtract.
                a += size;
            }
            assert!(loads >= 2);
            assert!(extract > 0);
            for op in [224, 338, 345] {
                assert_eq!(count(&ow, op), count(&nw, op));
            }
            // Existing compact source proof still proves 180 original FMAs.
            assert_eq!(new.matches("dot=fma(").count(), 168);
            assert_eq!(new.matches("if !finite(kscale)").count(), 2);
            #[cfg(all(feature = "strata-qsa-packed-cache", not(feature="strata-qsa-reference-schedule")))]
            assert_eq!(k.source(), finish(new));
        }
    }
}
