use base64::{Engine, engine::general_purpose::STANDARD};
use ed25519_dalek::{Signature, VerifyingKey};
use serde::{Deserialize, Serialize};

pub const SCHEMA_VERSION: u32 = 1;
pub const SIGNING_CONTEXT: &str = "vashti-update-manifest-v1";
pub const RELEASE_PUBLIC_KEY_BASE64: &str = include_str!("../release-public-key.txt").trim_ascii();

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct SignedArtifact {
    pub version: String,
    pub target: String,
    pub filename: String,
    pub sha256: String,
    pub size_bytes: i64,
}

impl SignedArtifact {
    pub fn signing_message(&self) -> String {
        format!(
            "{SIGNING_CONTEXT}\nversion={}\ntarget={}\nfilename={}\nsha256={}\nsize_bytes={}\n",
            self.version, self.target, self.filename, self.sha256, self.size_bytes
        )
    }

    pub fn validate(&self) -> Result<(), ManifestError> {
        validate_field("version", &self.version)?;
        validate_field("target", &self.target)?;
        validate_field("filename", &self.filename)?;
        if self.size_bytes < 1 {
            return Err(ManifestError::InvalidSize);
        }
        if self.sha256.len() != 64 || !self.sha256.bytes().all(|value| value.is_ascii_hexdigit()) {
            return Err(ManifestError::InvalidSha256);
        }
        Ok(())
    }
}

#[derive(Debug, thiserror::Error)]
pub enum ManifestError {
    #[error("manifest field {0} contains an unsupported character")]
    InvalidField(&'static str),
    #[error("manifest artifact size must be positive")]
    InvalidSize,
    #[error("manifest SHA-256 must contain exactly 64 hexadecimal characters")]
    InvalidSha256,
    #[error("release public key is invalid")]
    InvalidPublicKey,
    #[error("release signature is invalid")]
    InvalidSignature,
}

pub fn verify_release_signature(
    artifact: &SignedArtifact,
    signature_base64: &str,
) -> Result<(), ManifestError> {
    artifact.validate()?;
    let key_bytes = STANDARD
        .decode(RELEASE_PUBLIC_KEY_BASE64)
        .map_err(|_| ManifestError::InvalidPublicKey)?;
    let key_bytes: [u8; 32] = key_bytes
        .try_into()
        .map_err(|_| ManifestError::InvalidPublicKey)?;
    let verifying_key =
        VerifyingKey::from_bytes(&key_bytes).map_err(|_| ManifestError::InvalidPublicKey)?;
    let signature_bytes = STANDARD
        .decode(signature_base64.trim())
        .map_err(|_| ManifestError::InvalidSignature)?;
    let signature =
        Signature::from_slice(&signature_bytes).map_err(|_| ManifestError::InvalidSignature)?;

    verifying_key
        .verify_strict(artifact.signing_message().as_bytes(), &signature)
        .map_err(|_| ManifestError::InvalidSignature)
}

fn validate_field(name: &'static str, value: &str) -> Result<(), ManifestError> {
    if value.is_empty()
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'.' | b'_' | b'-'))
    {
        return Err(ManifestError::InvalidField(name));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use ed25519_dalek::{Signer, SigningKey, Verifier};

    use super::*;

    fn artifact() -> SignedArtifact {
        SignedArtifact {
            version: "v1.2.3".to_string(),
            target: "linux-x86_64".to_string(),
            filename: "vashti-linux-x86_64.tar.gz".to_string(),
            sha256: "a".repeat(64),
            size_bytes: 42,
        }
    }

    #[test]
    fn canonical_message_is_stable() {
        assert_eq!(
            artifact().signing_message(),
            concat!(
                "vashti-update-manifest-v1\n",
                "version=v1.2.3\n",
                "target=linux-x86_64\n",
                "filename=vashti-linux-x86_64.tar.gz\n",
                "sha256=aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\n",
                "size_bytes=42\n"
            )
        );
    }

    #[test]
    fn altered_artifact_does_not_verify() {
        let signing_key = SigningKey::from_bytes(&[7; 32]);
        let original = artifact();
        let signature = signing_key.sign(original.signing_message().as_bytes());
        let mut altered = original;
        altered.size_bytes += 1;

        assert!(
            signing_key
                .verifying_key()
                .verify(altered.signing_message().as_bytes(), &signature)
                .is_err()
        );
    }

    #[test]
    fn production_public_key_verifies_release_fixture() {
        verify_release_signature(
            &artifact(),
            "hQJFKgrhovM2Sa9XtxaNgftzQosjzge6uUPDZBYf97/sIYabYPVqilGNc+L6K/dhB92A+dhhTPvCztd13NO9CQ==",
        )
        .expect("the embedded production public key must verify its release fixture");
    }
}
