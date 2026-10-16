//! Transient already-rendered UTF-8 input for the original protected session.
//! No command arguments, files, template rewriting, request receipts or logs.
//! Reading is not process protection: the enclosing caller must authenticate
//! model/host grants and disable swap/core dumps BEFORE providing private bytes.
use super::{Failure, MAX_OUTPUTS, Request, VOCAB, checked};
use kuno_text::{Qwen35Tokenizer, TokenizerLimits};
use std::io::{ErrorKind, Read};
use zeroize::Zeroizing;

pub(crate) const TEXT_BYTES: usize = 1024 * 1024;
// Separate from the live generation/suffix budget: text, BPE nodes/candidates,
// regex scratch and output IDs coexist during encoding, then text/scratch drop.
pub(crate) const HOST_BYTES: usize = 64 * 1024 * 1024;
pub(crate) fn tokenizer_limits(model_bytes: usize) -> TokenizerLimits {
    TokenizerLimits {
        max_vocabulary: VOCAB,
        max_input_bytes: TEXT_BYTES,
        max_output_tokens: MAX_OUTPUTS,
        max_piece_bytes: 64 * 1024,
        max_merge_steps: 4 * 1024 * 1024,
        max_decoded_bytes: TEXT_BYTES,
        max_decode_chunk_bytes: TEXT_BYTES,
        max_allocation_bytes: model_bytes,
        ..TokenizerLimits::default()
    }
}

/// Fixed controls have wiping sampler ownership and no display/serialization.
/// Text is read to EOF, bounded independently of the resulting token count.
pub(crate) struct Controls {
    pub outputs: usize,
    pub eos: u32,
    pub sampling: crate::strata_sampler::Params,
    pub speculation: crate::strata_speculation::Config,
    pub minimum: f32,
}
impl Controls {
    fn validate(&self) -> Result<(), Failure> {
        self.sampling.validate().map_err(|_| Failure::Rejected)?;
        self.speculation.validate().map_err(|_| Failure::Rejected)?;
        if !crate::strata_intel_core::enabled()
            || !(1..=MAX_OUTPUTS).contains(&self.outputs)
            || self.eos as usize >= VOCAB
            || !self.minimum.is_finite()
            || !(0.0..=1.0).contains(&self.minimum)
        {
            return Err(Failure::Rejected);
        }
        Ok(())
    }
}

fn read_rendered(reader: &mut dyn Read) -> Result<(Zeroizing<Vec<u8>>, usize), Failure> {
    // One fixed allocation, including a single overflow sentinel. No resize or
    // Vec growth can abandon old prompt bytes in an unguarded allocation.
    let mut bytes = Zeroizing::new(vec![0u8; TEXT_BYTES + 1]);
    let mut used = 0;
    loop {
        match reader.read(&mut bytes[used..]) {
            Ok(0) => break,
            Ok(count) if count <= bytes.len() - used => {
                used += count;
                if used > TEXT_BYTES {
                    return Err(Failure::Rejected);
                }
            }
            Ok(_) => return Err(Failure::Rejected),
            Err(error) if error.kind() == ErrorKind::Interrupted => continue,
            Err(_) => return Err(Failure::Rejected),
        }
    }
    if used == 0 || std::str::from_utf8(&bytes[..used]).is_err() {
        return Err(Failure::Rejected);
    }
    Ok((bytes, used))
}

pub(crate) fn request(
    reader: &mut dyn Read,
    tokenizer: &Qwen35Tokenizer,
    controls: Controls,
) -> Result<Request, Failure> {
    checked(|| {
        controls.validate()?;
        if tokenizer.vocabulary_len() != VOCAB || tokenizer.eos_token_id() != controls.eos {
            return Err(Failure::Rejected);
        }
        let (bytes, used) = read_rendered(reader)?;
        let text = std::str::from_utf8(&bytes[..used]).map_err(|_| Failure::Rejected)?;
        // The caller supplies the original model's tokenizer. A rendered prompt
        // intentionally parses its control tokens; no implicit BOS/template.
        let ids = Zeroizing::new(
            tokenizer
                .encode(text, true)
                .map_err(|_| Failure::Rejected)?,
        );
        Request::new(
            ids,
            controls.outputs,
            controls.eos,
            controls.sampling,
            controls.speculation,
            controls.minimum,
        )
        .map_err(|_| Failure::Rejected)
        // bytes and all tokenizer scratch drop before generation can allocate
        // its own bounded suffix/continuation state.
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{self, Cursor};
    struct Pieces {
        remaining: Vec<u8>,
        at: usize,
        interrupted: bool,
    }
    impl Read for Pieces {
        fn read(&mut self, out: &mut [u8]) -> io::Result<usize> {
            if !self.interrupted {
                self.interrupted = true;
                return Err(ErrorKind::Interrupted.into());
            }
            let n = (self.remaining.len() - self.at).min(out.len()).min(3);
            out[..n].copy_from_slice(&self.remaining[self.at..self.at + n]);
            self.at += n;
            Ok(n)
        }
    }
    #[test]
    fn bounded_rendered_input_preserves_utf8_controls_and_multiline_bytes_without_normalization() {
        let original = "<|im_start|>user\n  weasel 🦦\n<|im_end|>\n".as_bytes();
        let mut reader = Pieces {
            remaining: original.to_vec(),
            at: 0,
            interrupted: false,
        };
        let (bytes, n) = read_rendered(&mut reader).unwrap();
        assert_eq!(&bytes[..n], original);
        assert!(bytes[n..].iter().all(|&b| b == 0));
        assert_eq!(reader.at, original.len());
        assert_eq!(bytes.len(), TEXT_BYTES + 1);
    }
    #[test]
    fn overflow_reads_only_one_sentinel_and_io_utf8_errors_have_fixed_categories() {
        for length in [0, TEXT_BYTES, TEXT_BYTES + 1, TEXT_BYTES + 4096] {
            let mut reader = Cursor::new(vec![b'a'; length]);
            match read_rendered(&mut reader) {
                Ok((bytes, n)) => {
                    assert_eq!(length, TEXT_BYTES);
                    assert_eq!(n, length);
                    assert_eq!(bytes[n], 0);
                }
                Err(failure) => assert_eq!(failure, Failure::Rejected),
            }
            assert!(reader.position() <= (TEXT_BYTES + 1) as u64);
        }
        assert!(read_rendered(&mut Cursor::new(vec![0xff, 0xfe])).is_err());
        struct Broken;
        impl Read for Broken {
            fn read(&mut self, _: &mut [u8]) -> io::Result<usize> {
                Err(io::Error::other("synthetic private payload"))
            }
        }
        assert!(matches!(read_rendered(&mut Broken), Err(Failure::Rejected)));
    }
}
