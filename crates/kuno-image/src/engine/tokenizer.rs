//! Qwen2's byte-level BPE tokenizer, as MiniMax H3's text encoder (Qwen3-VL) reads prompts.
//!
//! Follows the C++ engine's `Qwen2Tokenizer` exactly: the vocabulary is the 256 byte tokens
//! (in GPT-2's byte-to-unicode order), one token per merge, then 31 special tokens; special
//! tokens are cut out first (earliest occurrence, the longest on a tie), the rest is split by
//! Qwen2's pattern and each piece merged by rank. Nothing is prepended or appended. The
//! merge list is `assets/qwen2-merges.txt` (see `assets/NOTICE`).
use std::collections::HashMap;

use fancy_regex::Regex;

const MERGES: &str = include_str!("../../assets/qwen2-merges.txt");
const PATTERN: &str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";
/// In id order after the merges.
pub const SPECIAL: [&str; 31] = [
    "<|endoftext|>",
    "<|im_start|>",
    "<|im_end|>",
    "<|object_ref_start|>",
    "<|object_ref_end|>",
    "<|box_start|>",
    "<|box_end|>",
    "<|quad_start|>",
    "<|quad_end|>",
    "<|vision_start|>",
    "<|vision_end|>",
    "<|vision_pad|>",
    "<|image_pad|>",
    "<|video_pad|>",
    "<tool_call>",
    "</tool_call>",
    "<|fim_prefix|>",
    "<|fim_middle|>",
    "<|fim_suffix|>",
    "<|fim_pad|>",
    "<|repo_name|>",
    "<|file_sep|>",
    "<tool_response>",
    "</tool_response>",
    "<think>",
    "</think>",
    "<|boi_token|>",
    "<|bor_token|>",
    "<|eor_token|>",
    "<|bot_token|>",
    "<|tms_token|>",
];

/// GPT-2's map from bytes to printable characters, in vocabulary order: (byte, char).
fn byte_chars() -> Vec<(u8, char)> {
    let mut out = Vec::with_capacity(256);
    let printable = |b: u32| (33..=126).contains(&b) || (161..=172).contains(&b) || (174..=255).contains(&b);
    for b in (0..256u32).filter(|&b| printable(b)) {
        out.push((b as u8, char::from_u32(b).unwrap()));
    }
    let mut n = 0;
    for b in (0..256u32).filter(|&b| !printable(b)) {
        out.push((b as u8, char::from_u32(256 + n).unwrap()));
        n += 1;
    }
    out
}

pub struct Tokenizer {
    /// Each byte's token.
    bytes: [u32; 256],
    /// (left, right) to (rank, merged token).
    merges: HashMap<(u32, u32), (u32, u32)>,
    /// The first special token's id.
    special: u32,
    pattern: Regex,
}
impl Tokenizer {
    pub fn new() -> Result<Self, &'static str> {
        let invalid = "tokenizer_invalid";
        let chars = byte_chars();
        let mut byte_of = HashMap::new();
        let mut bytes = [0u32; 256];
        for (id, &(b, c)) in chars.iter().enumerate() {
            byte_of.insert(c, b);
            bytes[b as usize] = id as u32;
        }
        let raw = |s: &str| -> Result<Vec<u8>, &'static str> {
            s.chars().map(|c| byte_of.get(&c).copied().ok_or(invalid)).collect()
        };
        // Token ids by raw bytes: the bytes, then one per merge.
        let mut ids: HashMap<Vec<u8>, u32> = HashMap::with_capacity(160_000);
        for (id, &(b, _)) in chars.iter().enumerate() {
            ids.insert(vec![b], id as u32);
        }
        let mut merges = HashMap::with_capacity(160_000);
        let mut next = 256u32;
        for (rank, line) in MERGES.split_terminator('\n').enumerate() {
            let (left, right) = line.split_once(' ').ok_or(invalid)?;
            let (left, right) = (raw(left)?, raw(right)?);
            let l = *ids.get(&left).ok_or(invalid)?;
            let r = *ids.get(&right).ok_or(invalid)?;
            let mut joined = left;
            joined.extend_from_slice(&right);
            // Concatenations are distinct (checked by a test), so ids are positional.
            let out = *ids.entry(joined).or_insert(next);
            next += 1;
            merges.entry((l, r)).or_insert((rank as u32, out));
        }
        Ok(Self {
            bytes,
            merges,
            special: next,
            pattern: Regex::new(PATTERN).map_err(|_| invalid)?,
        })
    }
    /// The id of a special token.
    pub fn special(&self, token: &str) -> Option<u32> {
        SPECIAL
            .iter()
            .position(|&s| s == token)
            .map(|i| self.special + i as u32)
    }
    pub fn encode(&self, text: &str) -> Result<Vec<u32>, &'static str> {
        let mut out = Vec::new();
        let mut rest = text;
        while !rest.is_empty() {
            // The earliest special token, the longest on a tie.
            let found = SPECIAL
                .iter()
                .enumerate()
                .filter_map(|(i, s)| rest.find(s).map(|at| (at, std::cmp::Reverse(s.len()), i)))
                .min();
            let (plain, special) = match found {
                Some((at, _, i)) => (&rest[..at], Some(i)),
                None => (rest, None),
            };
            for piece in self.pattern.find_iter(plain) {
                let piece = piece.map_err(|_| "tokenizer_invalid")?.as_str();
                if !piece.is_empty() {
                    self.bpe(piece.as_bytes(), &mut out);
                }
            }
            match special {
                Some(i) => {
                    out.push(self.special + i as u32);
                    rest = &rest[plain.len() + SPECIAL[i].len()..];
                }
                None => break,
            }
        }
        Ok(out)
    }
    /// Merges one piece by rank: the lowest-ranked adjacent pair everywhere it occurs, left
    /// to right, until no pair has a rank.
    fn bpe(&self, piece: &[u8], out: &mut Vec<u32>) {
        let mut word: Vec<u32> = piece.iter().map(|&b| self.bytes[b as usize]).collect();
        while word.len() > 1 {
            let Some((_, (first, second), merged)) = word
                .windows(2)
                .filter_map(|w| {
                    self.merges
                        .get(&(w[0], w[1]))
                        .map(|&(rank, merged)| (rank, (w[0], w[1]), merged))
                })
                .min()
            else {
                break;
            };
            let mut next = Vec::with_capacity(word.len());
            let mut i = 0;
            while i < word.len() {
                if i + 1 < word.len() && word[i] == first && word[i + 1] == second {
                    next.push(merged);
                    i += 2;
                } else {
                    next.push(word[i]);
                    i += 1;
                }
            }
            word = next;
        }
        out.extend_from_slice(&word);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vocabulary_matches_the_engine() {
        let t = Tokenizer::new().unwrap();
        assert_eq!(t.special("<|endoftext|>"), Some(151_643));
        assert_eq!(t.special("<|image_pad|>"), Some(151_655));
        assert_eq!(t.special("<|vision_start|>"), Some(151_652));
        assert_eq!(t.encode("").unwrap(), Vec::<u32>::new());
        // Every merge makes a distinct token, so ids are positional as in the engine.
        assert_eq!(t.special, 256 + MERGES.split_terminator('\n').count() as u32);
    }

    /// Ids from the C++ engine's tokenizer (its Oniguruma split) on 96 synthetic strings:
    /// specials, contractions, numbers, scripts, emoji, odd whitespace and random text.
    #[test]
    fn ids_match_the_engine() {
        let t = Tokenizer::new().unwrap();
        let cases: Vec<(String, Vec<u32>)> =
            serde_json::from_str(include_str!("../../tests/fixtures/qwen2-tokens.json")).unwrap();
        let mut wrong = 0;
        for (i, (text, want)) in cases.iter().enumerate() {
            let got = t.encode(text).unwrap();
            if &got != want {
                wrong += 1;
                eprintln!("case {i}: {} tokens, engine {}", got.len(), want.len());
            }
        }
        assert_eq!(wrong, 0, "{wrong} of {} cases differ", cases.len());
    }
}
