//! Speech recognition and synthesis (LFM3-Audio-2B). Requests carry transient content; it must
//! never enter diagnostics or storage.
use serde::{Deserialize, Serialize};

/// The collection the audio endpoints need enabled and ready.
pub const AUDIO_COLLECTION: &str = "lfm3-audio-2b";
/// The largest request body (base64 audio).
pub const MAX_AUDIO_REQUEST_BYTES: usize = 48 * 1024 * 1024;
/// The longest text spoken in one request, in characters (long text is spoken a few
/// sentences at a time; long audio is transcribed a minute at a time, split at pauses).
pub const MAX_SPEECH_CHARACTERS: usize = 20_000;
pub const SPEECH_VOICES: [&str; 2] = ["female", "male"];
/// Language codes and the names the model's prompts use (for synthesis; recognition names
/// English plainly).
pub const SPEECH_LANGUAGES: [(&str, &str); 20] = [
    ("en", "American English"),
    ("en-us", "American English"),
    ("en-gb", "British English"),
    ("ar", "Arabic"),
    ("da", "Danish"),
    ("de", "German"),
    ("el", "Greek"),
    ("fi", "Finnish"),
    ("fr", "French"),
    ("he", "Hebrew"),
    ("it", "Italian"),
    ("ja", "Japanese"),
    ("ko", "Korean"),
    ("nl", "Dutch"),
    ("pl", "Polish"),
    ("pt", "Portuguese"),
    ("es", "Spanish"),
    ("sv", "Swedish"),
    ("tr", "Turkish"),
    ("zh", "Chinese"),
];

/// The synthesis prompt's name for a language code (case-insensitive), if known.
pub fn speech_language(code: &str) -> Option<&'static str> {
    SPEECH_LANGUAGES
        .iter()
        .find(|(c, _)| c.eq_ignore_ascii_case(code))
        .map(|(_, name)| *name)
}
/// The recognition prompt's name for a language code (English without a region).
pub fn transcription_language(code: &str) -> Option<&'static str> {
    if code.eq_ignore_ascii_case("en") || code.to_ascii_lowercase().starts_with("en-") {
        return Some("English");
    }
    speech_language(code)
}

/// `POST /kuno/v1/audio/transcriptions`: a WAV file (PCM 8/16/24/32-bit or float, any rate,
/// mono or multi-channel), base64; `language` a code from `SPEECH_LANGUAGES`, or absent for
/// the model to tell. The response is `TranscriptionResponse`.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TranscriptionRequest {
    pub audio: String,
    #[serde(default)]
    pub language: Option<String>,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TranscriptionResponse {
    pub text: String,
}

/// `POST /kuno/v1/audio/speech`: `input` spoken in `language` (default `en`) with `voice`
/// (default `female`); `seed` repeats a rendering. The response body is a 24 kHz mono 16-bit
/// WAV file (`audio/wav`).
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SpeechRequest {
    pub input: String,
    #[serde(default)]
    pub voice: Option<String>,
    #[serde(default)]
    pub language: Option<String>,
    #[serde(default)]
    pub seed: Option<u64>,
}

impl SpeechRequest {
    pub fn validate(&self) -> Result<(), &'static str> {
        if self.input.trim().is_empty() || self.input.chars().count() > MAX_SPEECH_CHARACTERS {
            return Err("speech_input_invalid");
        }
        if self
            .voice
            .as_deref()
            .is_some_and(|v| !SPEECH_VOICES.contains(&v))
        {
            return Err("speech_voice_invalid");
        }
        if self
            .language
            .as_deref()
            .is_some_and(|l| speech_language(l).is_none())
        {
            return Err("speech_language_invalid");
        }
        Ok(())
    }
}
impl TranscriptionRequest {
    pub fn validate(&self) -> Result<(), &'static str> {
        if self.audio.is_empty() {
            return Err("transcription_audio_invalid");
        }
        if self
            .language
            .as_deref()
            .is_some_and(|l| transcription_language(l).is_none())
        {
            return Err("transcription_language_invalid");
        }
        Ok(())
    }
}

/// The `audio` section of `/kuno/v1/capabilities`.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AudioCapabilities {
    pub api_version: u32,
    pub access: String,
    /// `ready`, `unavailable` (with `reason`), or `disabled`.
    pub state: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reason: Option<String>,
    pub transcription_path: String,
    pub speech_path: String,
    pub voices: Vec<String>,
    pub languages: Vec<String>,
    pub max_request_bytes: usize,
    pub max_speech_characters: usize,
    pub speech_sample_rate: u32,
    /// The device the model last loaded onto ("cpu" or the GPU's name); absent until the
    /// first request since Kuno started. A configured GPU without room falls back to "cpu".
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub device: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn requests_validate() {
        let ok = SpeechRequest {
            input: "Hello".into(),
            voice: Some("male".into()),
            language: Some("EN-gb".into()),
            seed: None,
        };
        assert!(ok.validate().is_ok());
        assert_eq!(speech_language("EN-gb"), Some("British English"));
        assert_eq!(transcription_language("en-gb"), Some("English"));
        let bad = SpeechRequest {
            voice: Some("robot".into()),
            ..ok.clone()
        };
        assert_eq!(bad.validate(), Err("speech_voice_invalid"));
        let empty = SpeechRequest {
            input: "  ".into(),
            ..ok
        };
        assert_eq!(empty.validate(), Err("speech_input_invalid"));
        let t: Result<TranscriptionRequest, _> =
            serde_json::from_str(r#"{"audio":"AA","extra":1}"#);
        assert!(t.is_err());
    }
}
