//! Writer for the driver's native module container: a relocatable ELF for machine "Intel GT"
//! holding raw kernel code, a YAML description of each kernel and a compatibility note.

pub struct Kernel<'a> {
    pub name: &'a str,
    pub code: &'a [u8],
    /// Registers per thread: 128 or 256 (256 halves the threads per execution unit).
    pub grf: u32,
    /// Pointer arguments, delivered in `r1` at 8-byte steps.
    pub pointers: u32,
}

/// Target identification for Battlemage G21, as the driver writes it into its own modules:
/// (note type, value). 1 product family, 2 graphics core, 3 target metadata, 4 container
/// version, 6 product configuration, 7/8 driver-internal.
const COMPAT: [(u32, &[u8]); 7] = [
    (1, &[0xfa, 0x04, 0, 0]),
    (2, &[0x09, 0x0c, 0, 0]),
    (3, &[0, 0, 0x20, 0]),
    (4, b"1.73\0"),
    (6, &[0, 0x80, 0, 0x05]),
    (7, &[0x0a, 0, 0, 0]),
    (8, &[1, 0, 0, 0]),
];

fn pad(v: &mut Vec<u8>, to: usize) {
    while !v.len().is_multiple_of(to) {
        v.push(0);
    }
}

fn info(k: &Kernel) -> String {
    let mut s = format!(
        "---\nversion:         '1.73'\nkernels:\n  - name:            {}\n    execution_env:\n      grf_count:       {}\n      has_4gb_buffers: true\n      has_dpas:        true\n      inline_data_payload_size: 32\n      offset_to_skip_per_thread_data_load: 0\n      simd_size:       16\n      subgroup_independent_forward_progress: true\n      eu_thread_count: {}\n    payload_arguments:\n",
        k.name,
        k.grf,
        if k.grf == 256 { 4 } else { 8 }
    );
    assert!(
        k.pointers <= 4,
        "arguments must fit the 32-byte inline payload"
    );
    for i in 0..k.pointers {
        s += &format!(
            "      - arg_type:        arg_bypointer\n        offset:          {}\n        size:            8\n        arg_index:       {}\n        addrmode:        stateless\n        addrspace:       global\n        access_type:     readwrite\n",
            8 * i,
            i
        );
    }
    s + "...\n"
}

pub fn write(k: &Kernel) -> Vec<u8> {
    // The driver reads ahead of the last instruction, so the code is padded with a spare line.
    let mut code = k.code.to_vec();
    pad(&mut code, 64);
    code.extend([0u8; 64]);
    let mut note = Vec::new();
    for (ty, value) in COMPAT {
        for word in [8u32, value.len() as u32, ty] {
            note.extend(word.to_le_bytes());
        }
        note.extend(b"IntelGT\0");
        note.extend(value);
        pad(&mut note, 4);
    }
    let text = format!(".text.{}", k.name);
    let mut sections: Vec<(&str, u32, u64, Vec<u8>)> = vec![
        (&text, 1, 6, code),
        (".ze_info", 0xff00_0011, 0, info(k).into_bytes()),
        (".note.intelgt.compat", 7, 0, note),
    ];
    let mut strtab = vec![0u8];
    let mut names = Vec::new();
    for name in sections.iter().map(|s| s.0).chain([".strtab"]) {
        names.push(strtab.len() as u32);
        strtab.extend(name.as_bytes());
        strtab.push(0);
    }
    sections.push((".strtab", 3, 0, strtab));

    let mut out = vec![0u8; 64];
    let mut headers = vec![0u8; 64];
    for ((_, ty, flags, data), name) in sections.iter().zip(&names) {
        headers.extend(name.to_le_bytes());
        headers.extend(ty.to_le_bytes());
        for v in [*flags, 0, out.len() as u64, data.len() as u64] {
            headers.extend(v.to_le_bytes());
        }
        headers.extend([0u8; 24]);
        out.extend(data);
        pad(&mut out, 8);
    }
    let shoff = out.len() as u64;
    out[..9].copy_from_slice(&[0x7f, b'E', b'L', b'F', 2, 1, 1, 0, 1]);
    out[16..18].copy_from_slice(&1u16.to_le_bytes()); // relocatable
    out[18..20].copy_from_slice(&205u16.to_le_bytes()); // Intel GT
    out[20..24].copy_from_slice(&1u32.to_le_bytes());
    out[40..48].copy_from_slice(&shoff.to_le_bytes());
    out[52..54].copy_from_slice(&64u16.to_le_bytes()); // ELF header size
    out[58..60].copy_from_slice(&64u16.to_le_bytes()); // section header size
    out[60..62].copy_from_slice(&(sections.len() as u16 + 1).to_le_bytes());
    out[62..64].copy_from_slice(&(sections.len() as u16).to_le_bytes());
    out.extend(headers);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn container_is_a_well_formed_object_file() {
        let code = [0u8; 48];
        let out = write(&Kernel {
            name: "k",
            code: &code,
            grf: 256,
            pointers: 3,
        });
        assert_eq!(&out[..4], b"\x7fELF");
        let word = |at: usize| u64::from_le_bytes(out[at..at + 8].try_into().unwrap()) as usize;
        let (headers, count) = (word(40), u16::from_le_bytes([out[60], out[61]]) as usize);
        assert_eq!(count, 5);
        assert_eq!(out.len(), headers + count * 64);
        // Code section: padded to a cache line plus a spare one.
        let text = headers + 64;
        assert_eq!((word(text + 24), word(text + 32)), (64, 128));
        let info = headers + 128;
        let yaml =
            std::str::from_utf8(&out[word(info + 24)..word(info + 24) + word(info + 32)]).unwrap();
        assert!(yaml.contains("grf_count:       256") && yaml.contains("eu_thread_count: 4"));
        assert_eq!(yaml.matches("arg_bypointer").count(), 3);
    }
}
