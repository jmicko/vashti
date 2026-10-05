//! Qwen35 GPT2 byte-level BPE, entirely in Rust.
//!
//! Regex and special-token semantics follow llama.cpp revision
//! `806eee9841de5f2c20f9d43914117f157d2baacc`, src/llama-vocab.cpp,
//! and Strata revision `6f32ec070f23ced9f50e704d854d775da52591ab`,
//! tools/strata_tokenizer.py. See NOTICE for upstream license provenance.
//! No chat template, normalization, implicit BOS or content cache is applied.

use std::cmp::Ordering;
use std::collections::{BTreeMap, BinaryHeap, HashMap, HashSet};

use aho_corasick::{AhoCorasick, AhoCorasickBuilder, AhoCorasickKind, MatchKind};
use fancy_regex::{Regex, RegexBuilder};
use zeroize::{Zeroize, Zeroizing};

use crate::{Error, GgufFile, MetadataValue, Result};

const PATTERN: &str = r"(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";

#[derive(Clone, Debug)]
pub struct TokenizerLimits {
    pub max_vocabulary: usize,
    pub max_merges: usize,
    pub max_token_bytes: usize,
    pub max_allocation_bytes: usize,
    pub max_special_tokens: usize,
    pub max_input_bytes: usize,
    pub max_piece_bytes: usize,
    pub max_output_tokens: usize,
    pub max_merge_steps: usize,
    pub regex_backtrack_limit: usize,
    pub max_decode_chunk_bytes: usize,
    pub max_decoded_bytes: usize,
}

impl Default for TokenizerLimits {
    fn default() -> Self {
        Self {
            max_vocabulary: 1_000_000,
            max_merges: 2_000_000,
            max_token_bytes: 16 * 1024,
            max_allocation_bytes: 192 * 1024 * 1024,
            max_special_tokens: 256,
            max_input_bytes: 1024 * 1024,
            max_piece_bytes: 256 * 1024,
            max_output_tokens: 262_144,
            max_merge_steps: 4 * 1024 * 1024,
            regex_backtrack_limit: 1_000_000,
            max_decode_chunk_bytes: 1024 * 1024,
            max_decoded_bytes: 4 * 1024 * 1024,
        }
    }
}

struct SpecialMatcher {
    automaton: AhoCorasick,
    token_ids: Vec<u32>,
}

#[derive(Clone, Copy)]
struct Merge {
    rank: u32,
    output: u32,
}

/// Model-only tokenizer data. Debug is intentionally not implemented: callers
/// must not accidentally dump a vocabulary, template or live encoding scratch.
pub struct Qwen35Tokenizer {
    pieces: Vec<Vec<u8>>,
    base_ids: [u32; 256],
    merges: HashMap<(u32, u32), Merge>,
    always_special: Option<SpecialMatcher>,
    all_special: Option<SpecialMatcher>,
    pretokenizer: Regex,
    eos: u32,
    bos: u32,
    limits: TokenizerLimits,
}

impl Qwen35Tokenizer {
    pub fn from_gguf(model: &GgufFile, limits: TokenizerLimits) -> Result<Self> {
        Self::from_metadata(model.metadata(), limits)
    }

    pub fn from_metadata(
        metadata: &BTreeMap<String, MetadataValue>,
        limits: TokenizerLimits,
    ) -> Result<Self> {
        if text(metadata, "tokenizer.ggml.model")? != "gpt2"
            || text(metadata, "tokenizer.ggml.pre")? != "qwen35"
        {
            return Err(Error::InvalidFormat(
                "unsupported tokenizer model or pretokenizer",
            ));
        }
        if metadata.get("tokenizer.ggml.add_bos_token") != Some(&MetadataValue::Bool(false))
            || !matches!(
                metadata.get("tokenizer.ggml.add_eos_token"),
                None | Some(MetadataValue::Bool(false))
            )
        {
            return Err(Error::InvalidFormat("implicit BOS/EOS is unsupported"));
        }
        let vocabulary = array(metadata, "tokenizer.ggml.tokens")?;
        let types = array(metadata, "tokenizer.ggml.token_type")?;
        let rules = array(metadata, "tokenizer.ggml.merges")?;
        if vocabulary.len() > limits.max_vocabulary || vocabulary.len() > u32::MAX as usize {
            return Err(Error::LimitExceeded("tokenizer vocabulary"));
        }
        if vocabulary.len() != types.len() || vocabulary.len() < 256 {
            return Err(Error::InvalidFormat(
                "tokenizer vocabulary/type count mismatch",
            ));
        }
        if rules.len() > limits.max_merges || rules.len() > u32::MAX as usize {
            return Err(Error::LimitExceeded("tokenizer merges"));
        }
        let mut allocated = vocabulary
            .len()
            .checked_mul(160)
            .and_then(|bytes| {
                rules
                    .len()
                    .checked_mul(64)
                    .and_then(|extra| bytes.checked_add(extra))
            })
            .ok_or(Error::ArithmeticOverflow)?;
        check_allocation(allocated, &limits)?;
        let eos = id(metadata, "tokenizer.ggml.eos_token_id", vocabulary.len())?;
        let bos = id(metadata, "tokenizer.ggml.bos_token_id", vocabulary.len())?;
        if !matches!(unsigned(&types[eos as usize]), Some(3 | 4))
            || !matches!(unsigned(&types[bos as usize]), Some(3 | 4))
        {
            return Err(Error::InvalidFormat("BOS/EOS id is not a special token"));
        }
        let reverse = reverse_byte_map();
        let mut pieces = Vec::new();
        pieces
            .try_reserve_exact(vocabulary.len())
            .map_err(|_| Error::LimitExceeded("tokenizer allocation"))?;
        let mut normal_ids = HashMap::new();
        normal_ids
            .try_reserve(vocabulary.len())
            .map_err(|_| Error::LimitExceeded("tokenizer allocation"))?;
        let mut all_strings = HashSet::new();
        all_strings
            .try_reserve(vocabulary.len())
            .map_err(|_| Error::LimitExceeded("tokenizer allocation"))?;
        let mut always = Vec::new();
        let mut all = Vec::new();
        for (index, (value, kind)) in vocabulary.iter().zip(types).enumerate() {
            let value = value
                .as_str()
                .ok_or(Error::InvalidFormat("vocabulary item is not a string"))?;
            let kind = unsigned(kind).ok_or(Error::InvalidFormat("token type is not unsigned"))?;
            if value.is_empty() || !all_strings.insert(value) {
                return Err(Error::InvalidFormat("empty or duplicate vocabulary item"));
            }
            if value.len() > limits.max_token_bytes {
                return Err(Error::LimitExceeded("tokenizer token bytes"));
            }
            allocated = allocated
                .checked_add(
                    value
                        .len()
                        .checked_mul(2)
                        .ok_or(Error::ArithmeticOverflow)?,
                )
                .ok_or(Error::ArithmeticOverflow)?;
            check_allocation(allocated, &limits)?;
            let token_id = index as u32;
            let bytes = match kind {
                1 => {
                    let bytes = unmapped_bytes(value, &reverse)?;
                    if normal_ids.insert(bytes.clone(), token_id).is_some() {
                        return Err(Error::InvalidFormat(
                            "duplicate byte-mapped vocabulary item",
                        ));
                    }
                    bytes
                }
                3 | 4 => {
                    // Budget both matcher representations conservatively before
                    // construction; force NFA below rather than a large DFA.
                    allocated = allocated
                        .checked_add(
                            value
                                .len()
                                .checked_mul(2048)
                                .ok_or(Error::ArithmeticOverflow)?,
                        )
                        .ok_or(Error::ArithmeticOverflow)?;
                    check_allocation(allocated, &limits)?;
                    all.push((value, token_id));
                    if all.len() > limits.max_special_tokens {
                        return Err(Error::LimitExceeded("tokenizer special tokens"));
                    }
                    if kind == 4 {
                        always.push((value, token_id));
                    }
                    value.as_bytes().to_vec()
                }
                5 => value.as_bytes().to_vec(),
                _ => return Err(Error::InvalidFormat("unsupported tokenizer token type")),
            };
            pieces.push(bytes);
        }
        let mut base_ids = [0_u32; 256];
        for (byte, id) in base_ids.iter_mut().enumerate() {
            *id = *normal_ids
                .get(&[byte as u8][..])
                .ok_or(Error::InvalidFormat("incomplete tokenizer byte alphabet"))?;
        }
        let mut merges = HashMap::new();
        merges
            .try_reserve(rules.len())
            .map_err(|_| Error::LimitExceeded("tokenizer allocation"))?;
        for (rank, value) in rules.iter().enumerate() {
            let value = value
                .as_str()
                .ok_or(Error::InvalidFormat("merge item is not a string"))?;
            if value.len()
                > limits
                    .max_token_bytes
                    .checked_mul(2)
                    .and_then(|n| n.checked_add(1))
                    .ok_or(Error::ArithmeticOverflow)?
            {
                return Err(Error::LimitExceeded("tokenizer merge bytes"));
            }
            let (left, right) = value
                .split_once(' ')
                .ok_or(Error::InvalidFormat("merge is not a pair"))?;
            if left.is_empty() || right.is_empty() || right.contains(' ') {
                return Err(Error::InvalidFormat("merge is not a pair"));
            }
            let mut left_bytes = unmapped_bytes(left, &reverse)?;
            let right_bytes = unmapped_bytes(right, &reverse)?;
            let left_id = *normal_ids
                .get(left_bytes.as_slice())
                .ok_or(Error::InvalidFormat("merge references an unknown token"))?;
            let right_id = *normal_ids
                .get(right_bytes.as_slice())
                .ok_or(Error::InvalidFormat("merge references an unknown token"))?;
            left_bytes.extend_from_slice(&right_bytes);
            let output = *normal_ids
                .get(left_bytes.as_slice())
                .ok_or(Error::InvalidFormat("merge result is not in vocabulary"))?;
            if merges
                .insert(
                    (left_id, right_id),
                    Merge {
                        rank: rank as u32,
                        output,
                    },
                )
                .is_some()
            {
                return Err(Error::InvalidFormat("duplicate ranked merge pair"));
            }
        }
        let always_special = special_matcher(&always)?;
        let all_special = special_matcher(&all)?;
        let pretokenizer = RegexBuilder::new(PATTERN)
            .backtrack_limit(limits.regex_backtrack_limit)
            .delegate_size_limit(8 * 1024 * 1024)
            .delegate_dfa_size_limit(2 * 1024 * 1024)
            .build()
            .map_err(|_| Error::InvalidFormat("tokenizer regex cannot be compiled"))?;
        Ok(Self {
            pieces,
            base_ids,
            merges,
            always_special,
            all_special,
            pretokenizer,
            eos,
            bos,
            limits,
        })
    }

    pub fn vocabulary_len(&self) -> usize {
        self.pieces.len()
    }
    pub fn eos_token_id(&self) -> u32 {
        self.eos
    }
    pub fn bos_token_id(&self) -> u32 {
        self.bos
    }

    /// CONTROL literals are matched only with parse_special=true. USER_DEFINED
    /// literals always match. No token is implicitly prepended or appended.
    pub fn encode(&self, input: &str, parse_special: bool) -> Result<Vec<u32>> {
        if input.len() > self.limits.max_input_bytes {
            return Err(Error::LimitExceeded("tokenizer input bytes"));
        }
        let mut output = Zeroizing::new(Vec::new());
        // Token count cannot exceed the input byte count. Reserve once so
        // sensitive token IDs never escape via an old reallocated buffer.
        output
            .try_reserve_exact(input.len().min(self.limits.max_output_tokens))
            .map_err(|_| Error::LimitExceeded("tokenizer output allocation"))?;
        let matcher = if parse_special {
            &self.all_special
        } else {
            &self.always_special
        };
        let mut steps = 0_usize;
        let mut start = 0;
        if let Some(matcher) = matcher {
            for matched in matcher.automaton.find_iter(input.as_bytes()) {
                self.encode_plain(&input[start..matched.start()], &mut output, &mut steps)?;
                append_token(
                    &mut output,
                    matcher.token_ids[matched.pattern().as_usize()],
                    &self.limits,
                )?;
                start = matched.end();
            }
        }
        self.encode_plain(&input[start..], &mut output, &mut steps)?;
        Ok(std::mem::take(&mut *output))
    }

    fn encode_plain(&self, input: &str, output: &mut Vec<u32>, steps: &mut usize) -> Result<()> {
        let mut position = 0;
        for found in self.pretokenizer.find_iter(input) {
            let found = found.map_err(|_| Error::LimitExceeded("tokenizer regex execution"))?;
            if found.start() != position || found.end() <= position {
                return Err(Error::InvalidFormat(
                    "tokenizer regex left an unmatched span",
                ));
            }
            self.encode_piece(found.as_str().as_bytes(), output, steps)?;
            position = found.end();
        }
        if position != input.len() {
            return Err(Error::InvalidFormat(
                "tokenizer regex left an unmatched span",
            ));
        }
        Ok(())
    }

    fn encode_piece(&self, bytes: &[u8], output: &mut Vec<u32>, steps: &mut usize) -> Result<()> {
        if bytes.len() > self.limits.max_piece_bytes {
            return Err(Error::LimitExceeded("tokenizer pretoken bytes"));
        }
        if bytes.is_empty() {
            return Ok(());
        }
        let mut scratch = BpeScratch {
            nodes: Zeroizing::new(Vec::new()),
            heap: BinaryHeap::new(),
        };
        scratch
            .nodes
            .try_reserve_exact(bytes.len())
            .map_err(|_| Error::LimitExceeded("tokenizer scratch"))?;
        // At most n-1 initial pairs and two new pairs per successful merge.
        // This also avoids leaving candidate token IDs in reallocated storage.
        scratch
            .heap
            .try_reserve(
                bytes
                    .len()
                    .checked_mul(3)
                    .ok_or(Error::ArithmeticOverflow)?,
            )
            .map_err(|_| Error::LimitExceeded("tokenizer scratch"))?;
        for (index, &byte) in bytes.iter().enumerate() {
            scratch.nodes.push(Node {
                token: self.base_ids[byte as usize],
                previous: index.checked_sub(1),
                next: (index + 1 < bytes.len()).then_some(index + 1),
                alive: true,
            });
        }
        for index in 0..bytes.len() - 1 {
            self.schedule(&mut scratch, index)?;
        }
        while let Some(pair) = scratch.heap.pop() {
            *steps = steps.checked_add(1).ok_or(Error::ArithmeticOverflow)?;
            if *steps > self.limits.max_merge_steps {
                return Err(Error::LimitExceeded("tokenizer merge steps"));
            }
            let left = &scratch.nodes[pair.left];
            let right = &scratch.nodes[pair.right];
            if !left.alive
                || !right.alive
                || left.next != Some(pair.right)
                || left.token != pair.left_token
                || right.token != pair.right_token
            {
                continue;
            }
            let merge = self.merges[&(left.token, right.token)];
            let after = right.next;
            scratch.nodes[pair.left].token = merge.output;
            scratch.nodes[pair.left].next = after;
            scratch.nodes[pair.right].alive = false;
            if let Some(after) = after {
                scratch.nodes[after].previous = Some(pair.left);
            }
            self.schedule(&mut scratch, pair.left)?;
            if let Some(before) = scratch.nodes[pair.left].previous {
                self.schedule(&mut scratch, before)?;
            }
        }
        for node in scratch.nodes.iter().filter(|node| node.alive) {
            append_token(output, node.token, &self.limits)?;
        }
        Ok(())
    }

    fn schedule(&self, scratch: &mut BpeScratch, left: usize) -> Result<()> {
        if let Some(right) = scratch.nodes[left].next {
            let left_token = scratch.nodes[left].token;
            let right_token = scratch.nodes[right].token;
            if let Some(merge) = self.merges.get(&(left_token, right_token)) {
                scratch
                    .heap
                    .try_reserve(1)
                    .map_err(|_| Error::LimitExceeded("tokenizer scratch"))?;
                scratch.heap.push(Candidate {
                    rank: merge.rank,
                    left,
                    right,
                    left_token,
                    right_token,
                });
            }
        }
        Ok(())
    }

    /// Raw token bytes. Combine them before Unicode decoding: an individual
    /// token need not end on a UTF8 boundary. Special IDs return their literals.
    pub fn token_bytes(&self, id: u32) -> Result<&[u8]> {
        self.pieces
            .get(id as usize)
            .map(Vec::as_slice)
            .ok_or(Error::OutOfBounds)
    }

    pub fn stream_decoder(&self) -> Utf8StreamDecoder {
        Utf8StreamDecoder::new(self.limits.max_decode_chunk_bytes)
    }

    pub fn decode(&self, ids: &[u32]) -> Result<String> {
        if ids.len() > self.limits.max_output_tokens {
            return Err(Error::LimitExceeded("tokenizer decode tokens"));
        }
        let mut decoder = self.stream_decoder();
        let mut output = Zeroizing::new(String::new());
        let raw_bytes = ids.iter().try_fold(0_usize, |total, &id| {
            total
                .checked_add(self.token_bytes(id)?.len())
                .ok_or(Error::ArithmeticOverflow)
        })?;
        let capacity = raw_bytes
            .checked_mul(3)
            .ok_or(Error::ArithmeticOverflow)?
            .min(self.limits.max_decoded_bytes);
        output
            .try_reserve_exact(capacity)
            .map_err(|_| Error::LimitExceeded("tokenizer decode allocation"))?;
        for &id in ids {
            let piece = Zeroizing::new(decoder.push_bytes(self.token_bytes(id)?)?);
            if output
                .len()
                .checked_add(piece.len())
                .ok_or(Error::ArithmeticOverflow)?
                > self.limits.max_decoded_bytes
            {
                return Err(Error::LimitExceeded("tokenizer decoded bytes"));
            }
            output.push_str(&piece);
        }
        let ending = decoder.finish();
        if output
            .len()
            .checked_add(ending.len())
            .ok_or(Error::ArithmeticOverflow)?
            > self.limits.max_decoded_bytes
        {
            return Err(Error::LimitExceeded("tokenizer decoded bytes"));
        }
        output.push_str(&ending);
        Ok(std::mem::take(&mut *output))
    }
}

fn text<'a>(metadata: &'a BTreeMap<String, MetadataValue>, key: &str) -> Result<&'a str> {
    metadata
        .get(key)
        .and_then(MetadataValue::as_str)
        .ok_or(Error::InvalidFormat(
            "missing or invalid tokenizer metadata",
        ))
}
fn array<'a>(
    metadata: &'a BTreeMap<String, MetadataValue>,
    key: &str,
) -> Result<&'a [MetadataValue]> {
    metadata
        .get(key)
        .and_then(MetadataValue::as_array)
        .ok_or(Error::InvalidFormat(
            "missing or invalid tokenizer metadata",
        ))
}
fn id(metadata: &BTreeMap<String, MetadataValue>, key: &str, count: usize) -> Result<u32> {
    let id = metadata
        .get(key)
        .and_then(unsigned)
        .ok_or(Error::InvalidFormat(
            "missing or invalid tokenizer token id",
        ))?;
    if id >= count as u64 {
        return Err(Error::InvalidFormat(
            "tokenizer token id exceeds vocabulary",
        ));
    }
    Ok(id as u32)
}
fn unsigned(value: &MetadataValue) -> Option<u64> {
    match value {
        MetadataValue::Unsigned(value) => Some(*value),
        MetadataValue::Signed(value) => u64::try_from(*value).ok(),
        _ => None,
    }
}
fn check_allocation(bytes: usize, limits: &TokenizerLimits) -> Result<()> {
    if bytes > limits.max_allocation_bytes {
        Err(Error::LimitExceeded("tokenizer allocation bytes"))
    } else {
        Ok(())
    }
}
fn special_matcher(patterns: &[(&str, u32)]) -> Result<Option<SpecialMatcher>> {
    if patterns.is_empty() {
        return Ok(None);
    }
    let automaton = AhoCorasickBuilder::new()
        .match_kind(MatchKind::LeftmostLongest)
        .kind(Some(AhoCorasickKind::NoncontiguousNFA))
        .build(patterns.iter().map(|(text, _)| text.as_bytes()))
        .map_err(|_| Error::LimitExceeded("tokenizer special matcher"))?;
    Ok(Some(SpecialMatcher {
        automaton,
        token_ids: patterns.iter().map(|(_, id)| *id).collect(),
    }))
}

fn reverse_byte_map() -> [Option<u8>; 512] {
    let mut map = [None; 512];
    let mut extra = 256;
    for byte in 0_u16..256 {
        let codepoint = if (33..=126).contains(&byte)
            || (161..=172).contains(&byte)
            || (174..=255).contains(&byte)
        {
            byte as usize
        } else {
            let codepoint = extra;
            extra += 1;
            codepoint
        };
        map[codepoint] = Some(byte as u8);
    }
    map
}
fn unmapped_bytes(text: &str, reverse: &[Option<u8>; 512]) -> Result<Vec<u8>> {
    text.chars()
        .map(|character| {
            reverse
                .get(character as usize)
                .copied()
                .flatten()
                .ok_or(Error::InvalidFormat("token is outside GPT2 byte alphabet"))
        })
        .collect()
}
fn append_token(output: &mut Vec<u32>, token: u32, limits: &TokenizerLimits) -> Result<()> {
    if output.len() >= limits.max_output_tokens {
        return Err(Error::LimitExceeded("tokenizer output tokens"));
    }
    output
        .try_reserve(1)
        .map_err(|_| Error::LimitExceeded("tokenizer output allocation"))?;
    output.push(token);
    Ok(())
}

struct Node {
    token: u32,
    previous: Option<usize>,
    next: Option<usize>,
    alive: bool,
}
impl Zeroize for Node {
    fn zeroize(&mut self) {
        self.token.zeroize();
        self.previous.zeroize();
        self.next.zeroize();
        self.alive.zeroize();
    }
}
#[derive(Eq, PartialEq)]
struct Candidate {
    rank: u32,
    left: usize,
    right: usize,
    left_token: u32,
    right_token: u32,
}
impl Ord for Candidate {
    fn cmp(&self, other: &Self) -> Ordering {
        (
            other.rank,
            other.left,
            other.right,
            other.left_token,
            other.right_token,
        )
            .cmp(&(
                self.rank,
                self.left,
                self.right,
                self.left_token,
                self.right_token,
            ))
    }
}
impl PartialOrd for Candidate {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Zeroize for Candidate {
    fn zeroize(&mut self) {
        self.rank.zeroize();
        self.left.zeroize();
        self.right.zeroize();
        self.left_token.zeroize();
        self.right_token.zeroize();
    }
}
impl Drop for Candidate {
    fn drop(&mut self) {
        self.zeroize();
    }
}
struct BpeScratch {
    nodes: Zeroizing<Vec<Node>>,
    heap: BinaryHeap<Candidate>,
}
impl Drop for BpeScratch {
    fn drop(&mut self) {
        std::mem::take(&mut self.heap).into_vec().zeroize();
    }
}

/// UTF8 replacement is delayed until enough token bytes arrive. Only an
/// incomplete tail (at most three bytes) is retained between calls. clear/Drop
/// erase the entire fixed-capacity tail; temporary combined chunks are zeroized.
pub struct Utf8StreamDecoder {
    pending: [u8; 4],
    pending_len: usize,
    max_chunk_bytes: usize,
}

impl Utf8StreamDecoder {
    pub fn new(max_chunk_bytes: usize) -> Self {
        Self {
            pending: [0; 4],
            pending_len: 0,
            max_chunk_bytes,
        }
    }

    pub fn push_bytes(&mut self, bytes: &[u8]) -> Result<String> {
        if bytes.len() > self.max_chunk_bytes {
            return Err(Error::LimitExceeded("tokenizer decode chunk bytes"));
        }
        let count = self
            .pending_len
            .checked_add(bytes.len())
            .ok_or(Error::ArithmeticOverflow)?;
        let mut combined = Zeroizing::new(Vec::new());
        combined
            .try_reserve_exact(count)
            .map_err(|_| Error::LimitExceeded("tokenizer decode allocation"))?;
        combined.extend_from_slice(&self.pending[..self.pending_len]);
        combined.extend_from_slice(bytes);
        self.clear();
        let mut output = String::new();
        // A fully invalid chunk expands to at most three UTF8 bytes per input
        // byte. Reserve before consuming the tail so allocation is bounded.
        output
            .try_reserve(count.checked_mul(3).ok_or(Error::ArithmeticOverflow)?)
            .map_err(|_| Error::LimitExceeded("tokenizer decode allocation"))?;
        let mut position = 0;
        while position < combined.len() {
            match std::str::from_utf8(&combined[position..]) {
                Ok(text) => {
                    output.push_str(text);
                    break;
                }
                Err(error) => {
                    let valid_end = position + error.valid_up_to();
                    output.push_str(std::str::from_utf8(&combined[position..valid_end]).unwrap());
                    position = valid_end;
                    if let Some(invalid) = error.error_len() {
                        output.push('\u{fffd}');
                        position += invalid;
                    } else {
                        let tail = &combined[position..];
                        if tail.len() > 3 {
                            return Err(Error::InvalidFormat("invalid tokenizer UTF8 tail"));
                        }
                        self.pending[..tail.len()].copy_from_slice(tail);
                        self.pending_len = tail.len();
                        break;
                    }
                }
            }
        }
        Ok(output)
    }

    pub fn finish(&mut self) -> String {
        let output = if self.pending_len == 0 {
            String::new()
        } else {
            "\u{fffd}".to_owned()
        };
        self.clear();
        output
    }

    pub fn clear(&mut self) {
        self.pending.zeroize();
        self.pending_len = 0;
    }
}
impl Drop for Utf8StreamDecoder {
    fn drop(&mut self) {
        self.clear();
    }
}
