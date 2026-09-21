//! A bounded, non-validating JSON observer. The upstream remains the parser and
//! authority. Only top-level model identifiers and input presence are retained;
//! prompt strings, images, and nested values are scanned without copying them.

#[derive(Clone, Copy, Default, PartialEq)]
enum Field {
    Model,
    Input,
    #[default]
    Other,
}
#[derive(Clone, Copy, Default, PartialEq)]
enum Position {
    #[default]
    Start,
    Key,
    Colon,
    Value,
    After,
    Done,
    Invalid,
}

#[derive(Default)]
pub struct Observer {
    position: Position,
    depth: usize,
    string: bool,
    escape: bool,
    capture: bool,
    string_nonempty: bool,
    text: Vec<u8>,
    field: Field,
    model: Option<String>,
    model_seen: bool,
    duplicate_model: bool,
    input: bool,
    scalar: bool,
    container_input: bool,
}

impl Observer {
    pub fn feed(&mut self, bytes: &[u8]) {
        for &byte in bytes {
            self.byte(byte);
        }
    }

    pub fn model(&self) -> Option<&str> {
        if self.position == Position::Done && !self.duplicate_model && self.input {
            self.model.as_deref()
        } else {
            None
        }
    }

    fn byte(&mut self, b: u8) {
        if self.position == Position::Invalid {
            return;
        }
        if self.string {
            if self.capture {
                if self.text.len() >= 1536 {
                    self.capture = false;
                    self.text.clear();
                } else {
                    self.text.push(b);
                }
            }
            if self.escape {
                self.escape = false;
                self.string_nonempty = true;
                return;
            }
            if b == b'\\' {
                self.escape = true;
                self.string_nonempty = true;
                return;
            }
            if b != b'"' {
                self.string_nonempty = true;
                return;
            }
            self.string = false;
            if self.depth != 1 {
                return;
            }
            if self.position == Position::Key {
                self.field = match self.decoded().as_deref() {
                    Some("model") => Field::Model,
                    Some("prompt" | "messages" | "input" | "images" | "suffix") => Field::Input,
                    _ => Field::Other,
                };
                self.position = Position::Colon;
            } else if self.position == Position::Value {
                if self.field == Field::Model {
                    self.duplicate_model |= self.model_seen;
                    self.model_seen = true;
                    self.model = self.decoded().filter(|m| !m.is_empty() && m.len() <= 256);
                }
                if self.field == Field::Input {
                    self.input |= self.string_nonempty;
                }
                self.position = Position::After;
            }
            self.text.clear();
            return;
        }
        if b.is_ascii_whitespace() {
            return;
        }
        if self.container_input && self.depth == 2 && b != b']' && b != b'}' {
            self.input = true;
            self.container_input = false;
        }
        if self.position == Position::Done {
            self.position = Position::Invalid;
            return;
        }
        if b == b'"' {
            self.string = true;
            self.string_nonempty = false;
            self.capture = self.depth == 1
                && (self.position == Position::Key
                    || (self.position == Position::Value && self.field == Field::Model));
            self.text.clear();
            if self.capture {
                self.text.push(b);
            }
            return;
        }
        match b {
            b'{' | b'[' => {
                if self.depth == 0 {
                    if b != b'{' || self.position != Position::Start {
                        self.position = Position::Invalid;
                        return;
                    }
                    self.position = Position::Key;
                } else if self.depth == 1 {
                    if self.position != Position::Value {
                        self.position = Position::Invalid;
                        return;
                    }
                    if self.field == Field::Model {
                        self.duplicate_model = true;
                    }
                    self.container_input = self.field == Field::Input;
                }
                self.depth += 1;
                if self.depth > 128 {
                    self.position = Position::Invalid;
                }
            }
            b'}' | b']' => {
                if self.depth == 0 {
                    self.position = Position::Invalid;
                    return;
                }
                self.depth -= 1;
                if self.depth == 0 {
                    self.position = Position::Done;
                } else if self.depth == 1 {
                    self.position = Position::After;
                    self.container_input = false;
                }
            }
            b':' if self.depth == 1 && self.position == Position::Colon => {
                self.position = Position::Value
            }
            b',' if self.depth == 1 && (self.position == Position::After || self.scalar) => {
                self.position = Position::Key;
                self.scalar = false;
            }
            _ if self.depth == 1 && self.position == Position::Value => {
                self.scalar = true;
                if self.field == Field::Model {
                    self.duplicate_model = true;
                }
            }
            _ if self.depth > 1 => {}
            _ => self.position = Position::Invalid,
        }
    }

    fn decoded(&self) -> Option<String> {
        if self.capture {
            serde_json::from_slice(&self.text).ok()
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chunk_boundaries_escaped_keys_and_large_nested_payloads() {
        let body = format!(
            r#"{{"messages":[{{"content":"{}","model":"wrong"}}],"mo\u0064el":"qwen:latest","unknown":{{"x":[1,2]}}}}"#,
            "not retained ".repeat(100000)
        );
        for chunk_size in [1, 7, 4096] {
            let mut observer = Observer::default();
            for chunk in body.as_bytes().chunks(chunk_size) {
                observer.feed(chunk);
                assert!(observer.text.len() <= 1536);
            }
            assert_eq!(observer.model(), Some("qwen:latest"));
            assert!(observer.text.is_empty());
        }
    }

    #[test]
    fn preloads_duplicates_partial_requests_and_empty_inputs_do_not_count() {
        for body in [
            r#"{"model":"x","keep_alive":0}"#,
            r#"{"model":"x","prompt":"","messages":[],"input":null}"#,
            r#"{"model":"x","prompt":"hi","model":"y"}"#,
            r#"{"model":"x","prompt":"hi""#,
        ] {
            let mut observer = Observer::default();
            observer.feed(body.as_bytes());
            assert_eq!(observer.model(), None, "{body}");
        }
        for body in [
            r#"{"input":[1],"model":"x"}"#,
            r#"{"model":"x","prompt":"hello"}"#,
        ] {
            let mut observer = Observer::default();
            observer.feed(body.as_bytes());
            assert_eq!(observer.model(), Some("x"));
        }
    }
}
