//! Prompt emphasis as the C++ engine reads it (`parse_prompt_attention` in its util.cpp) and
//! its effect on the text encoder's output (`apply_token_weights`), kept byte for byte so a
//! prompt means the same as before: `(text)` weighs 1.1, `[text]` 1/1.1, `(text:1.5)` 1.5,
//! nesting multiplies, `\(` is a literal parenthesis and a standalone `BREAK` weighs -1.
//!
//! The engine runs a regular expression over what is left of the prompt, repeatedly, with
//! the alternatives (first that matches wins at a position): the escapes `\(`, `\)`, `\[`,
//! `\]`, `\\` and `\`; one of `(`, `[`, `)`, `]`; `\bBREAK\b`; a run of up to 1024 bytes
//! without `\()[]:B`; `:`; `\bB`. Every position matches one of them (a `B` always starts the
//! remaining text, where `\b` holds), so a direct scan gives the same pieces.

fn is_word(c: char) -> bool {
    c.is_ascii_alphanumeric() || c == '_'
}

/// The next token of the remaining text, which is never empty.
fn next_token(rest: &str) -> &str {
    let mut chars = rest.chars();
    let first = chars.next().unwrap_or_default();
    let second = chars.next();
    match first {
        '\\' => match second {
            Some('(' | ')' | '[' | ']' | '\\') => &rest[..2],
            _ => &rest[..1],
        },
        '(' | '[' | ')' | ']' | ':' => &rest[..1],
        'B' => {
            // `\bBREAK\b`: the text starts here, so only the end needs a boundary.
            let breaks =
                rest.starts_with("BREAK") && rest[5..].chars().next().is_none_or(|c| !is_word(c));
            if breaks { &rest[..5] } else { &rest[..1] }
        }
        _ => {
            // A run without the special characters, at most 1024 bytes (whole characters:
            // runs are merged again below, so where a long run splits does not matter).
            let mut end = 0;
            for (at, c) in rest.char_indices() {
                if matches!(c, '\\' | '(' | ')' | '[' | ']' | ':' | 'B') || at + c.len_utf8() > 1024
                {
                    break;
                }
                end = at + c.len_utf8();
            }
            &rest[..end.max(first.len_utf8())]
        }
    }
}

/// `number)` at the start of `s`: the number and the length including the parenthesis. The
/// number is an optional sign and digits or dots, read as a whole finite float.
fn weight_at(s: &str) -> Option<(f32, usize)> {
    let bytes = s.as_bytes();
    let mut end = usize::from(matches!(bytes.first(), Some(b'+' | b'-')));
    while end < bytes.len() && (bytes[end].is_ascii_digit() || bytes[end] == b'.') {
        end += 1;
    }
    if bytes.get(end) != Some(&b')') {
        return None;
    }
    let number = &s[..end];
    let digits = number.trim_start_matches(['+', '-']);
    // strtof stops at a second dot or reads nothing from "." or "": both are rejected.
    if digits.is_empty() || digits == "." || digits.matches('.').count() > 1 {
        return None;
    }
    let value: f32 = number.parse().ok()?;
    value.is_finite().then_some((value, end + 1))
}

/// The prompt's pieces with their weights, adjacent equal weights merged.
pub fn parse_attention(text: &str) -> Vec<(String, f32)> {
    let mut out: Vec<(String, f32)> = Vec::new();
    let (mut round, mut square) = (Vec::new(), Vec::new());
    let multiply = |out: &mut Vec<(String, f32)>, from: usize, by: f32| {
        for item in &mut out[from..] {
            item.1 *= by;
        }
    };
    let mut rest = text;
    while !rest.is_empty() {
        let token = next_token(rest);
        let suffix = &rest[token.len()..];
        if token == ":"
            && let Some((weight, length)) = weight_at(suffix)
        {
            if let Some(from) = round.pop() {
                multiply(&mut out, from, weight);
            }
            rest = &suffix[length..];
            continue;
        }
        match token {
            "(" => round.push(out.len()),
            "[" => square.push(out.len()),
            ")" if !round.is_empty() => {
                let from = round.pop().unwrap_or_default();
                multiply(&mut out, from, 1.1);
            }
            "]" if !square.is_empty() => {
                let from = square.pop().unwrap_or_default();
                multiply(&mut out, from, 1.0 / 1.1);
            }
            "\\(" => out.push(("(".into(), 1.0)),
            "BREAK" => out.push(("BREAK".into(), -1.0)),
            _ => out.push((token.into(), 1.0)),
        }
        rest = suffix;
    }
    for from in round {
        multiply(&mut out, from, 1.1);
    }
    for from in square {
        multiply(&mut out, from, 1.0 / 1.1);
    }
    if out.is_empty() {
        out.push((String::new(), 1.0));
    }
    let mut merged: Vec<(String, f32)> = Vec::with_capacity(out.len());
    for (piece, weight) in out {
        match merged.last_mut() {
            Some(last) if last.1 == weight => last.0 += &piece,
            _ => merged.push((piece, weight)),
        }
    }
    merged
}

/// Scales each token's row of `hidden` (`[tokens][width]`) by its weight and the whole by
/// the ratio of the old to the new mean, unless every weight is one.
pub fn apply_weights(hidden: &mut [f32], weights: &[f32]) {
    if weights.iter().all(|w| *w == 1.0) || weights.is_empty() {
        return;
    }
    let width = hidden.len() / weights.len();
    let mean = |values: &[f32]| {
        (values.iter().map(|v| f64::from(*v)).sum::<f64>() / values.len() as f64) as f32
    };
    let before = mean(hidden);
    for (row, weight) in hidden.chunks_exact_mut(width).zip(weights) {
        row.iter_mut().for_each(|v| *v *= weight);
    }
    let after = mean(hidden);
    if before.is_finite() && after.is_finite() && after != 0.0 {
        let ratio = before / after;
        hidden.iter_mut().for_each(|v| *v *= ratio);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Pieces and weights from the C++ engine's own function on the same prompts
    /// (`tests/fixtures/prompt-attention.json`).
    #[test]
    fn attention_matches_the_engine() {
        let cases: Vec<(String, Vec<(String, f32)>)> =
            serde_json::from_str(include_str!("../../tests/fixtures/prompt-attention.json"))
                .unwrap();
        assert!(cases.len() > 10);
        for (text, want) in cases {
            assert_eq!(parse_attention(&text), want, "{text:?}");
        }
    }

    #[test]
    fn weights_keep_the_mean() {
        let mut hidden = vec![1.0, 2.0, 3.0, 4.0];
        apply_weights(&mut hidden, &[1.0, 2.0]);
        // [1, 2, 6, 8] scaled by 2.5 / 4.25.
        let ratio = 2.5f32 / 4.25;
        assert_eq!(hidden, [ratio, 2.0 * ratio, 6.0 * ratio, 8.0 * ratio]);
        let mut same = vec![1.0, 2.0];
        apply_weights(&mut same, &[1.0]);
        assert_eq!(same, [1.0, 2.0]);
    }
}
