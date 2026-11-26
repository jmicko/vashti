//! Long inputs in pieces the model handles well and memory bounds: audio a minute at a
//! time, cut at the quietest 100 ms in the last ten seconds; text a few sentences at a time.

/// Ranges of `samples` (16 kHz) of at most `max` seconds, cut at pauses.
pub fn audio(samples: &[f32], max_seconds: usize) -> Vec<std::ops::Range<usize>> {
    let (rate, step) = (16_000usize, 1_600usize);
    let max = max_seconds * rate;
    let search = (max / 6).max(step);
    let mut out = Vec::new();
    let mut start = 0;
    while samples.len() - start > max {
        let (lo, hi) = (start + max - search, start + max);
        let mut best = (f32::MAX, hi);
        let mut at = lo;
        while at + step <= hi {
            let energy: f32 = samples[at..at + step].iter().map(|v| v * v).sum();
            if energy < best.0 {
                best = (energy, at + step / 2);
            }
            at += step;
        }
        out.push(start..best.1);
        start = best.1;
    }
    out.push(start..samples.len());
    out
}

/// `text` in pieces of at most about `max` characters, ending at sentence ends where
/// possible (then at spaces, then anywhere).
pub fn text(text: &str, max: usize) -> Vec<String> {
    let mut sentences = Vec::new();
    let mut current = String::new();
    for c in text.chars() {
        current.push(c);
        if matches!(c, '.' | '!' | '?' | '\n' | '。' | '！' | '？') {
            sentences.push(std::mem::take(&mut current));
        }
    }
    sentences.push(current);
    let mut out: Vec<String> = Vec::new();
    let mut piece = String::new();
    for s in sentences {
        if s.trim().is_empty() {
            piece.push_str(&s);
            continue;
        }
        if !piece.trim().is_empty() && piece.chars().count() + s.chars().count() > max {
            out.push(std::mem::take(&mut piece));
        }
        piece.push_str(&s);
        while piece.chars().count() > max {
            let chars: Vec<char> = piece.chars().collect();
            let cut = chars[..max]
                .iter()
                .rposition(|c| c.is_whitespace())
                .filter(|&i| i > 0)
                .unwrap_or(max);
            out.push(chars[..cut].iter().collect());
            piece = chars[cut..].iter().collect();
        }
    }
    out.push(piece);
    out.into_iter()
        .map(|s| s.trim().to_owned())
        .filter(|s| !s.is_empty())
        .collect()
}

#[cfg(test)]
mod tests {
    #[test]
    fn audio_cuts_at_the_pause() {
        let mut s = vec![0.5f32; 16_000 * 130];
        s[16_000 * 55..16_000 * 55 + 3200].fill(0.0);
        let r = super::audio(&s, 60);
        assert_eq!(r.len(), 3);
        assert!((16_000 * 55..16_000 * 55 + 3200).contains(&r[0].end));
        assert_eq!(r[2].end, s.len());
        assert!(r.iter().all(|r| r.len() <= 16_000 * 60));
        assert_eq!(super::audio(&s[..100], 60), vec![0..100]);
    }

    #[test]
    fn text_keeps_sentences_together() {
        let t = "One two. Three four five! Six?";
        assert_eq!(super::text(t, 100), vec![t]);
        assert_eq!(
            super::text(t, 20),
            vec!["One two.", "Three four five!", "Six?"]
        );
        let long = "word ".repeat(50);
        let p = super::text(&long, 30);
        assert!(p.iter().all(|p| p.chars().count() <= 30));
        assert_eq!(p.join(" ").split_whitespace().count(), 50);
    }
}
