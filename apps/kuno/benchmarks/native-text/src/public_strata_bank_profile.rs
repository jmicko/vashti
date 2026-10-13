//! Fixed PUBLIC setup ranking derived from pinned Strata, not native routes.
//! Copyright (c) 2026 Niko1221 and the Strata contributors; MIT notice and
//! original/derived artifacts are in fixtures/strata-bank-profile/LICENSE.
//! Only empty original 144-slot banks consume this table. No runtime path,
//! private histogram, weight conversion, allocation or GPU binding is added.
use kuno_text::{Error, Result};

pub(crate) const PUBLIC_STRATA_BANK_PROFILE_SHA256: &str =
    "8f59b4aa8873209dff11c11e37bcda9529a1335b724a1afeea37bf6388975baf";
pub(crate) const PUBLIC_STRATA_BANK_PROFILE_BYTES: usize = 48 * 144 * 2;
const PREFIX: &[u8; PUBLIC_STRATA_BANK_PROFILE_BYTES] =
    include_bytes!("../fixtures/strata-bank-profile/top144-u16le.bin");

pub(super) fn prefix(layer: usize) -> Result<[u16; crate::daslab_profile::BANK_SLOTS]> {
    if layer >= 48 {
        return Err(Error::InvalidFormat(
            "fixed public Strata bank profile layer",
        ));
    }
    let ids = if crate::strata_port::enabled() {
        let full = include_bytes!("../fixtures/strata-bank-profile/expert-profile.bin");
        let mut ids = [0u16; crate::daslab_profile::BANK_SLOTS];
        let mut count = 0;
        for rank in 0..24_576 {
            let at = 24 + rank * 4;
            let source_layer = u16::from_le_bytes([full[at], full[at + 1]]) as usize;
            if source_layer == layer && count < ids.len() {
                ids[count] = u16::from_le_bytes([full[at + 2], full[at + 3]]);
                count += 1;
            }
        }
        if count != ids.len() {
            return Err(Error::InvalidBufferLength);
        }
        ids
    } else {
        std::array::from_fn(|i| {
            let at = (layer * 144 + i) * 2;
            u16::from_le_bytes([PREFIX[at], PREFIX[at + 1]])
        })
    };
    validate_prefix(&ids)?;
    Ok(ids)
}

pub(super) fn validate_prefix(ids: &[u16]) -> Result<()> {
    if ids.len() != crate::daslab_profile::BANK_SLOTS
        || ids.iter().any(|&id| id >= 512)
        || ids.iter().enumerate().any(|(i, id)| ids[..i].contains(id))
    {
        return Err(Error::InvalidFormat(
            "fixed public Strata bank profile prefix",
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn profile_original_artifact_and_every_compact_rank_are_exact() {
        let full = include_bytes!("../fixtures/strata-bank-profile/expert-profile.bin");
        assert_eq!(full.len(), 196_632);
        assert_eq!(&full[..4], b"STRP");
        let word = |at| u32::from_le_bytes(full[at..at + 4].try_into().unwrap());
        assert_eq!(
            [word(4), word(8), word(12), word(16), word(20)],
            [1, 48, 512, 24_576, 24_576]
        );
        let mut ranked = [0usize; 48];
        let mut seen = [[false; 512]; 48];
        for rank in 0..24_576 {
            let at = 24 + rank * 4;
            let layer = u16::from_le_bytes([full[at], full[at + 1]]) as usize;
            let id = u16::from_le_bytes([full[at + 2], full[at + 3]]) as usize;
            assert!(layer < 48 && id < 512 && !seen[layer][id]);
            seen[layer][id] = true;
            let table_at = 24 + 24_576 * 4 + (layer * 512 + id) * 4;
            assert_eq!(word(table_at) as usize, rank);
            if ranked[layer] < 144 {
                assert_eq!(prefix(layer).unwrap()[ranked[layer]] as usize, id);
            }
            ranked[layer] += 1;
        }
        assert_eq!(ranked, [512; 48]);
        for (name, expected) in [
            ("expert-profile.bin", PUBLIC_STRATA_BANK_PROFILE_SHA256),
            (
                "top144-u16le.bin",
                "7bb7a111617d13f3daeed4a3b406e0f7a34b2a7fb07a1ee3845161a08ab57a5d",
            ),
        ] {
            // This standalone Vulkan laboratory is Linux-only; authenticate
            // the independent original fixture with the installed digest tool.
            let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("fixtures/strata-bank-profile")
                .join(name);
            let out = std::process::Command::new("sha256sum")
                .arg(path)
                .output()
                .unwrap();
            assert!(out.status.success());
            assert_eq!(&String::from_utf8(out.stdout).unwrap()[..64], expected);
        }
    }

    #[test]
    fn profile_invalid_layer_and_malformed_prefix_never_fall_back() {
        assert!(prefix(48).is_err());
        let mut ids = prefix(0).unwrap();
        ids[1] = ids[0];
        assert!(validate_prefix(&ids).is_err());
        ids[1] = 512;
        assert!(validate_prefix(&ids).is_err());
        assert!(validate_prefix(&ids[..143]).is_err());
    }
}
