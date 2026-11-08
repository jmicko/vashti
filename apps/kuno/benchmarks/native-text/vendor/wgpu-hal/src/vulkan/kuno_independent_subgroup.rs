//! Compiler-owned SIMD16 for the subgroup-independent prompt GDN entry.
//! This is not validation/device authority: original Naga/SPIR-V validation,
//! actual device limits, owning resources and queue fences remain mandatory.
pub const KUNO_GDN_SIMD16_ENTRY: &str = "kuno_prompt_gdn_recurrence16";

pub fn kuno_gdn_subgroup16(words: &[u32]) -> Option<u32> {
    if words.len() < 5 || words[0] != 0x07230203 || words[3] == 0 || words[4] != 0 {
        return None;
    }
    let mut entry = None;
    let mut local = None;
    let mut shader = false;
    let mut at = 5;
    while at < words.len() {
        let n = (words[at] >> 16) as usize;
        if n == 0 || at.checked_add(n).is_none_or(|end| end > words.len()) {
            return None;
        }
        let i = &words[at..at + n];
        match i[0] & 65535 {
            // No subgroup, cooperative, vendor, or other capability may select
            // this contract. Unknown capabilities retain original SIMD32.
            17 => {
                if n != 2 || i[1] != 1 || shader {
                    return None;
                }
                shader = true;
            }
            10 => return None, // Unknown extensions are not this compiler entry.
            11 => {
                if n < 3 || !literal(&i[2..], b"GLSL.std.450") {
                    return None;
                }
            }
            15 => {
                if n < 4
                    || i[1] != 5
                    || i[2] == 0
                    || i[2] >= words[3]
                    || !literal(&i[3..], KUNO_GDN_SIMD16_ENTRY.as_bytes())
                    || entry.replace(i[2]).is_some()
                {
                    return None;
                }
            }
            16 => {
                if n != 6 || i[2] != 17 || local.replace((i[1], i[3], i[4], i[5])).is_some() {
                    return None;
                }
            }
            331 => return None, // No specialized geometry or subgroup mode.
            // Only the original workgroup/local-invocation coordinates. No
            // subgroup IDs, widths, lanes, masks or membership dependencies.
            71 if n >= 3 && i[2] == 11 => {
                if n != 4 || !matches!(i[3], 26 | 29) {
                    return None;
                }
            }
            72 if n >= 4 && i[3] == 11 => return None,
            // Workgroup barriers retain precisely the original whole group.
            224 => {
                if n != 4 || constant(words, i[1]) != Some(2) || constant(words, i[2]) != Some(2) {
                    return None;
                }
            }
            225 => {
                if n != 3 || !matches!(constant(words, i[1]), Some(1 | 2)) {
                    return None;
                }
            }
            228 => {
                if n < 4 || !matches!(constant(words, i[2]), Some(1 | 2)) {
                    return None;
                }
            }
            227 | 229..=242 => {
                if n < 6 || !matches!(constant(words, i[4]), Some(1 | 2)) {
                    return None;
                }
            }
            // A valid subgroup instruction needs its missing capability, but
            // reject known group operations independently as well.
            318 | 319 | 333..=366 => return None,
            _ => {}
        }
        at += n;
    }
    (shader && local == Some((entry?, 128, 1, 1))).then_some(16)
}
fn literal(words: &[u32], expected: &[u8]) -> bool {
    let mut offset = 0;
    for word in words {
        for byte in word.to_le_bytes() {
            if byte == 0 {
                return offset == expected.len();
            }
            if expected.get(offset) != Some(&byte) {
                return false;
            }
            offset += 1;
        }
    }
    false
}
fn constant(words: &[u32], id: u32) -> Option<u32> {
    let mut value = None;
    let mut at = 5;
    while at < words.len() {
        let n = (words[at] >> 16) as usize;
        if n == 0 || at.checked_add(n).is_none_or(|end| end > words.len()) {
            return None;
        }
        let i = &words[at..at + n];
        if i[0] & 65535 == 43 && n == 4 && i[2] == id {
            if value.replace(i[3]).is_some() {
                return None;
            }
        }
        at += n;
    }
    value
}
