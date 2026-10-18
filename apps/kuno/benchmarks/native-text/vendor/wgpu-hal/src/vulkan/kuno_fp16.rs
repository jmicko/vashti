//! Explicit FP16 input preservation for the selected native cooperative path.
//! Pure SPIR-V transformation: runtime feature support is checked by Device.
use alloc::vec::Vec;

/// Preserve IEEE FP16 subnormal operands in the original cooperative module.
/// None means this is not an ordinary-FP16 cooperative compute shader.
pub fn kuno_cooperative_fp16_preserve(words: &[u32]) -> Result<Option<Vec<u32>>, &'static str> {
    if words.len() < 5 || words[0] != 0x07230203 {
        return Err("FP16 SPIRV header");
    }
    let mut instructions = Vec::new();
    let mut halves = Vec::new();
    let mut matrices = Vec::new();
    let mut entries = Vec::new();
    let mut capability = false;
    let mut cap_end = 5;
    let mut mode_end = 5;
    let mut at = 5;
    while at < words.len() {
        let count = (words[at] >> 16) as usize;
        let op = words[at] & 65535;
        if count == 0 || at.checked_add(count).is_none_or(|end| end > words.len()) {
            return Err("FP16 SPIRV instruction extent");
        }
        let i = &words[at..at + count];
        match op {
            17 if count == 2 => {
                cap_end = at + count;
                capability |= i[1] == 4464;
            }
            15 if count >= 4 => {
                entries.push((i[1], i[2]));
                mode_end = at + count;
            }
            16 | 331 => {
                mode_end = at + count;
            }
            // A BF16 encoding has an additional operand; do not change it.
            22 if count == 3 && i[2] == 16 => halves.push(i[1]),
            4456 if count >= 7 => matrices.push(i[2]),
            _ => {}
        }
        instructions.push((at, count, op));
        at += count;
    }
    if !matrices.iter().any(|id| halves.contains(id)) {
        return Ok(None);
    }
    if words[1] < 0x00010400 {
        return Err("FP16 controls require SPIRV1.4");
    }
    if entries.len() != 1 || entries[0].0 != 5 {
        return Err("FP16 cooperative compute entry");
    }
    let entry = entries[0].1;
    let mut preserved = false;
    for &(at, count, op) in &instructions {
        let i = &words[at..at + count];
        if op == 16 && count == 4 && i[1] == entry && i[3] == 16 {
            // SPV_KHR_float_controls: DenormPreserve / DenormFlushToZero.
            if i[2] == 4460 {
                return Err("conflicting FP16 denorm mode");
            }
            preserved |= i[2] == 4459;
        }
    }
    let mut output = Vec::with_capacity(words.len() + 6);
    output.extend_from_slice(&words[..5]);
    at = 5;
    loop {
        if at == cap_end && !capability {
            output.extend_from_slice(&[(2 << 16) | 17, 4464]);
        }
        if at == mode_end && !preserved {
            output.extend_from_slice(&[(4 << 16) | 16, entry, 4459, 16]);
        }
        if at == words.len() {
            break;
        }
        let count = (words[at] >> 16) as usize;
        output.extend_from_slice(&words[at..at + count]);
        at += count;
    }
    Ok(Some(output))
}
