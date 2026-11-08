//! Text-only rendering/parsing of the actual DASLab Qwen4exp embedded template.
//! Tool execution belongs to the caller. All generated/request buffers are transient.
use serde_json::{Value, json};
use zeroize::{Zeroize, Zeroizing};
type Result<T> = std::result::Result<T, &'static str>;
const LIMIT: usize = super::native_protocol::MAX_PROMPT_BYTES;
const OUTPUT_LIMIT: usize = 8 * 1024 * 1024;
const TOOL_GUIDE: &str = "# Tools\n\nYou have access to the following functions:\n\n<tools>";
// Exact text in tokenizer.chat_template of the enrolled, original model shards.
const TOOL_SUFFIX: &str = "\n</tools>\n\nIf you choose to call a function ONLY reply in the following format with NO suffix:\n\n<tool_call>\n<function=example_function_name>\n<parameter=example_parameter_1>\nvalue_1\n</parameter>\n<parameter=example_parameter_2>\nThis is the value for the second parameter\nthat can span\nmultiple lines\n</parameter>\n</function>\n</tool_call>\n\n<IMPORTANT>\nReminder:\n- Function calls MUST follow the specified format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags\n- Required parameters MUST be specified\n- You may provide optional reasoning for your function call in natural language BEFORE the function call, but NOT after\n- If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls\n</IMPORTANT>";
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Thinking {
    Off,
    Low,
    Medium,
    High,
}
impl Thinking {
    pub fn parse(v: Option<&Value>) -> Result<Self> {
        match v.filter(|v| !v.is_null()) {
            None | Some(Value::Bool(false)) => Ok(Self::Off),
            Some(Value::Bool(true)) => Ok(Self::High),
            Some(Value::String(v)) => match v.as_str() {
                "low" => Ok(Self::Low),
                "medium" => Ok(Self::Medium),
                "high" | "xhigh" => Ok(Self::High),
                _ => Err("native_thinking_invalid"),
            },
            _ => Err("native_thinking_invalid"),
        }
    }
    pub fn enabled(self) -> bool {
        self != Self::Off
    }
    fn instruction(self) -> &'static str {
        match self {
            Self::High => {
                "Reasoning effort is set to xhigh. Please think carefully through the task, validate key assumptions, consider plausible alternatives, and prioritize correctness, consistency, and clarity in the final answer."
            }
            Self::Low => {
                "Reasoning effort is set to low. Keep your thinking brief and focused, moving directly to the conclusion without unnecessary elaboration."
            }
            _ => "",
        }
    }
}
fn push(out: &mut String, s: &str) -> Result<()> {
    if out.len().checked_add(s.len()).is_none_or(|n| n > LIMIT) {
        return Err("native_prompt_too_large");
    }
    out.push_str(s);
    Ok(())
}
fn name(s: &str) -> bool {
    !s.is_empty()
        && s.len() <= 128
        && s.bytes()
            .all(|b| b.is_ascii_alphanumeric() || b"_.-".contains(&b))
}
fn tool_function(v: &Value) -> Result<&Value> {
    let f = v.get("function").unwrap_or(v);
    if v.get("type").is_some_and(|v| v != "function")
        || !f.is_object()
        || !f.get("name").and_then(Value::as_str).is_some_and(name)
    {
        return Err("native_tool_invalid");
    }
    Ok(f)
}
pub fn tools(v: Option<&Value>) -> Result<Vec<Value>> {
    let Some(v) = v.filter(|v| !v.is_null()) else {
        return Ok(Vec::new());
    };
    let a = v.as_array().ok_or("native_tools_invalid")?;
    if a.len() > 128 {
        return Err("native_tools_invalid");
    }
    for (i, t) in a.iter().enumerate() {
        let f = tool_function(t)?;
        if f.get("parameters").is_some_and(|p| !p.is_object())
            || a[..i]
                .iter()
                .any(|x| tool_function(x).is_ok_and(|x| x["name"] == f["name"]))
        {
            return Err("native_tool_invalid");
        }
    }
    Ok(a.clone())
}
fn text(v: &Value) -> Result<&str> {
    match v.get("content") {
        None | Some(Value::Null) => Ok(""),
        Some(Value::String(s)) => Ok(s.trim()),
        _ => Err("native_messages_invalid"),
    }
}
pub fn render(
    messages: &[Value],
    tools: &[Value],
    thinking: Thinking,
) -> Result<Zeroizing<Vec<u8>>> {
    for m in messages {
        let object = m.as_object().ok_or("native_messages_invalid")?;
        if object.keys().any(|k| {
            ![
                "role",
                "content",
                "thinking",
                "reasoning_content",
                "tool_calls",
                "tool_name",
                "tool_call_id",
                "images",
            ]
            .contains(&k.as_str())
        }) {
            return Err("native_message_feature_unsupported");
        }
        let role = m
            .get("role")
            .and_then(Value::as_str)
            .ok_or("native_messages_invalid")?;
        if m.get("images")
            .is_some_and(|v| !v.is_null() && !v.as_array().is_some_and(Vec::is_empty))
        {
            return Err("native_images_unsupported");
        }
        if role != "assistant"
            && ["thinking", "reasoning_content", "tool_calls"]
                .iter()
                .any(|k| m.get(k).is_some_and(|v| !v.is_null()))
        {
            return Err("native_message_feature_unsupported");
        }
        text(m)?;
    }
    let mut out = Zeroizing::new(String::with_capacity(LIMIT));
    let mut system = Zeroizing::new(String::with_capacity(LIMIT));
    let mut n = 0;
    for m in messages {
        let role = m
            .get("role")
            .and_then(Value::as_str)
            .ok_or("native_messages_invalid")?;
        if !["system", "developer"].contains(&role) {
            break;
        }
        let s = text(m)?;
        if !s.is_empty() {
            if !system.is_empty() {
                push(&mut system, "\n")?;
            }
            push(&mut system, s)?;
        }
        n += 1;
    }
    if !system.is_empty() || !tools.is_empty() || !thinking.instruction().is_empty() {
        push(&mut out, "<|im_start|>system\n")?;
        if !thinking.instruction().is_empty() {
            push(&mut out, thinking.instruction())?;
            if !system.is_empty() || !tools.is_empty() {
                push(&mut out, "\n\n")?;
            }
        }
        if !tools.is_empty() {
            push(&mut out, TOOL_GUIDE)?;
            for t in tools {
                push(&mut out, "\n")?;
                let s =
                    Zeroizing::new(serde_json::to_string(t).map_err(|_| "native_tool_invalid")?);
                push(&mut out, &s)?;
            }
            push(&mut out, TOOL_SUFFIX)?;
            if !system.is_empty() {
                push(&mut out, "\n\n")?;
            }
        }
        push(&mut out, &system)?;
        push(&mut out, "<|im_end|>\n")?;
    }
    for (i, m) in messages.iter().enumerate().skip(n) {
        let obj = m.as_object().ok_or("native_messages_invalid")?;
        if obj.keys().any(|k| {
            ![
                "role",
                "content",
                "thinking",
                "reasoning_content",
                "tool_calls",
                "tool_name",
                "tool_call_id",
                "images",
            ]
            .contains(&k.as_str())
        }) {
            return Err("native_message_feature_unsupported");
        }
        if m.get("images")
            .is_some_and(|v| !v.is_null() && !v.as_array().is_some_and(Vec::is_empty))
        {
            return Err("native_images_unsupported");
        }
        let role = m["role"].as_str().ok_or("native_messages_invalid")?;
        let content = text(m)?;
        match role {
            "user" => {
                push(&mut out, "<|im_start|>user\n")?;
                push(&mut out, content)?;
                push(&mut out, "<|im_end|>\n")?;
            }
            "assistant" => {
                push(&mut out, "<|im_start|>assistant\n<think>\n")?;
                let reasoning = m.get("thinking").or_else(|| m.get("reasoning_content"));
                if let Some(r) = reasoning.filter(|r| !r.is_null()) {
                    push(
                        &mut out,
                        r.as_str().ok_or("native_messages_invalid")?.trim(),
                    )?;
                }
                push(&mut out, "\n</think>\n\n")?;
                push(&mut out, content)?;
                if let Some(calls) = m.get("tool_calls").filter(|v| !v.is_null()) {
                    let calls = calls.as_array().ok_or("native_tool_invalid")?;
                    if calls.len() > 128 {
                        return Err("native_tool_invalid");
                    }
                    for (j, c) in calls.iter().enumerate() {
                        let f = tool_function(c)?;
                        if j == 0 && !content.is_empty() {
                            push(&mut out, "\n\n")?;
                        } else if j > 0 {
                            push(&mut out, "\n")?;
                        }
                        push(&mut out, "<tool_call>\n<function=")?;
                        push(&mut out, f["name"].as_str().unwrap())?;
                        push(&mut out, ">\n")?;
                        if let Some(args) = f.get("arguments").filter(|v| !v.is_null()) {
                            let args = args.as_object().ok_or("native_tool_arguments_invalid")?;
                            for (key, v) in args {
                                if !name(key) {
                                    return Err("native_tool_arguments_invalid");
                                }
                                push(&mut out, "<parameter=")?;
                                push(&mut out, key)?;
                                push(&mut out, ">\n")?;
                                if let Some(s) = v.as_str() {
                                    push(&mut out, s)?;
                                } else {
                                    let s = Zeroizing::new(
                                        serde_json::to_string(v)
                                            .map_err(|_| "native_tool_invalid")?,
                                    );
                                    push(&mut out, &s)?;
                                }
                                push(&mut out, "\n</parameter>\n")?;
                            }
                        }
                        push(&mut out, "</function>\n</tool_call>")?;
                    }
                }
                push(&mut out, "<|im_end|>\n")?;
            }
            "tool" => {
                if i == 0 || messages[i - 1]["role"] != "tool" {
                    push(&mut out, "<|im_start|>user")?;
                }
                push(&mut out, "\n<tool_response>\n")?;
                push(&mut out, content)?;
                push(&mut out, "\n</tool_response>")?;
                if i + 1 == messages.len() || messages[i + 1]["role"] != "tool" {
                    push(&mut out, "<|im_end|>\n")?;
                }
            }
            _ => return Err("native_message_role_unsupported"),
        }
    }
    push(&mut out, "<|im_start|>assistant\n<think>\n")?;
    if !thinking.enabled() {
        push(&mut out, "\n</think>\n\n")?;
    }
    Ok(Zeroizing::new(out.as_bytes().to_vec()))
}
pub struct Delta {
    pub thinking: Zeroizing<String>,
    pub content: Zeroizing<String>,
    pub calls: Vec<Value>,
}
impl Delta {
    fn empty() -> Self {
        Self {
            thinking: Zeroizing::new(String::new()),
            content: Zeroizing::new(String::new()),
            calls: Vec::new(),
        }
    }
}
pub struct Output {
    pending: Zeroizing<String>,
    thinking: bool,
    tool_block: bool,
    tools: Vec<Value>,
    literal: bool,
    count: usize,
    call_index: usize,
}
impl Drop for Output {
    fn drop(&mut self) {
        for t in &mut self.tools {
            super::native_text::scrub_json(t);
        }
    }
}
impl Output {
    pub fn new(thinking: bool, tools: Vec<Value>, literal: bool) -> Self {
        Self {
            pending: Zeroizing::new(String::with_capacity(OUTPUT_LIMIT)),
            thinking,
            tool_block: false,
            tools,
            literal,
            count: 0,
            call_index: 0,
        }
    }
    pub fn definitions(&self) -> &[Value] {
        &self.tools
    }
    pub fn feed(&mut self, s: &str, end: bool) -> Result<Delta> {
        self.count = self
            .count
            .checked_add(s.len())
            .ok_or("native_response_too_large")?;
        if self.count > OUTPUT_LIMIT {
            return Err("native_response_too_large");
        }
        self.pending.push_str(s);
        let mut d = Delta::empty();
        loop {
            if self.literal {
                d.content.push_str(&self.pending);
                self.pending.zeroize();
                break;
            }
            if self.tool_block {
                let Some(at) = self.pending.find("</tool_call>") else {
                    if end {
                        return Err("native_tool_call_incomplete");
                    }
                    break;
                };
                if d.calls.len() >= 128 {
                    return Err("native_tool_call_invalid");
                }
                let mut call = parse_call(&self.pending[..at], &self.tools)?;
                call["function"]["index"] = self.call_index.into();
                self.call_index += 1;
                if self.call_index > 128 {
                    return Err("native_tool_call_invalid");
                }
                d.calls.push(call);
                self.consume(at + "</tool_call>".len());
                self.tool_block = false;
                continue;
            }
            let markers: &[&str] = if self.thinking {
                &["</think>"]
            } else if self.tools.is_empty() {
                &[]
            } else {
                &["<tool_call>"]
            };
            let found = markers
                .iter()
                .filter_map(|m| self.pending.find(m).map(|n| (n, *m)))
                .min_by_key(|x| x.0);
            if let Some((at, marker)) = found {
                if self.thinking {
                    d.thinking.push_str(&self.pending[..at]);
                } else {
                    d.content.push_str(&self.pending[..at]);
                }
                self.consume(at + marker.len());
                match marker {
                    "</think>" => self.thinking = false,
                    "<think>" => self.thinking = true,
                    _ => self.tool_block = true,
                };
                continue;
            }
            let hold = if end {
                0
            } else {
                (1..=self.pending.len().min(12))
                    .filter(|&n| {
                        self.pending.is_char_boundary(self.pending.len() - n)
                            && markers
                                .iter()
                                .any(|m| m.starts_with(&self.pending[self.pending.len() - n..]))
                    })
                    .max()
                    .unwrap_or(0)
            };
            let n = self.pending.len() - hold;
            if self.thinking {
                d.thinking.push_str(&self.pending[..n]);
            } else {
                d.content.push_str(&self.pending[..n]);
            }
            self.consume(n);
            break;
        }
        Ok(d)
    }
    fn consume(&mut self, n: usize) {
        // In-place copy avoids freeing an abandoned unguarded private prefix.
        let remaining = self.pending.len() - n;
        unsafe {
            let v = self.pending.as_mut_vec();
            v.copy_within(n.., 0);
            v[remaining..].zeroize();
            v.truncate(remaining);
        }
    }
}
fn parse_call(block: &str, tools: &[Value]) -> Result<Value> {
    let mut s = block.trim();
    s = s
        .strip_prefix("<function=")
        .ok_or("native_tool_call_invalid")?;
    let (function, rest) = s.split_once('>').ok_or("native_tool_call_invalid")?;
    let definition = tools
        .iter()
        .filter_map(|t| tool_function(t).ok())
        .find(|f| f["name"] == function)
        .ok_or("native_tool_call_unknown")?;
    if !name(function) {
        return Err("native_tool_call_invalid");
    }
    let mut args = serde_json::Map::new();
    s = rest.trim();
    while let Some(rest) = s.strip_prefix("<parameter=") {
        let (key, rest) = rest.split_once('>').ok_or("native_tool_call_invalid")?;
        if !name(key) || args.contains_key(key) {
            return Err("native_tool_call_invalid");
        }
        let (value, rest) = rest
            .split_once("</parameter>")
            .ok_or("native_tool_call_invalid")?;
        let value = value.strip_prefix('\n').unwrap_or(value);
        let value = value.strip_suffix('\n').unwrap_or(value);
        let schema = &definition["parameters"]["properties"][key];
        let parsed = if schema["type"] == "string" {
            Value::String(value.to_owned())
        } else {
            serde_json::from_str(value).unwrap_or_else(|_| Value::String(value.to_owned()))
        };
        args.insert(key.to_owned(), parsed);
        s = rest.trim();
    }
    if s != "</function>" {
        return Err("native_tool_call_invalid");
    }
    Ok(json!({"function":{"name":function,"arguments":args}}))
}

impl Delta {
    pub fn annotate(&mut self, value: &mut Value, chat: bool) {
        let target = if chat { &mut value["message"] } else { value };
        if !self.thinking.is_empty() {
            target["thinking"] = self.thinking.as_str().into();
        }
        if !self.calls.is_empty() {
            target["tool_calls"] = Value::Array(std::mem::take(&mut self.calls));
        }
    }
}
impl Drop for Delta {
    fn drop(&mut self) {
        for c in &mut self.calls {
            super::native_text::scrub_json(c);
        }
    }
}
pub struct Collected {
    pub content: Zeroizing<String>,
    thinking: Zeroizing<String>,
    calls: Vec<Value>,
}
impl Default for Collected {
    fn default() -> Self {
        Self {
            content: Zeroizing::new(String::with_capacity(OUTPUT_LIMIT)),
            thinking: Zeroizing::new(String::with_capacity(OUTPUT_LIMIT)),
            calls: Vec::new(),
        }
    }
}
impl Drop for Collected {
    fn drop(&mut self) {
        for c in &mut self.calls {
            super::native_text::scrub_json(c);
        }
    }
}
impl Collected {
    pub fn push(&mut self, mut delta: Delta) {
        self.content.push_str(&delta.content);
        self.thinking.push_str(&delta.thinking);
        self.calls.append(&mut delta.calls);
    }
    pub fn annotate(&mut self, value: &mut Value, chat: bool) {
        let target = if chat { &mut value["message"] } else { value };
        if !self.thinking.is_empty() {
            target["thinking"] = self.thinking.as_str().into();
        }
        if !self.calls.is_empty() {
            target["tool_calls"] = Value::Array(std::mem::take(&mut self.calls));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn ts() -> Vec<Value> {
        vec![
            json!({"type":"function","function":{"name":"weather","parameters":{"type":"object","properties":{"city":{"type":"string"},"days":{"type":"integer"}}}}}),
        ]
    }
    #[test]
    fn fragmented_thinking_and_xml_tools_never_leak_markup_to_content() {
        let mut p = Output::new(true, ts(), false);
        let mut thinking = String::new();
        let mut content = String::new();
        let mut calls = Vec::new();
        let s = "Plan carefully.</think>\n\n<tool_call>\n<function=weather>\n<parameter=city>\nBoston\n</parameter>\n<parameter=days>\n3\n</parameter>\n</function>\n</tool_call>";
        for c in s.chars() {
            let mut d = p.feed(&c.to_string(), false).unwrap();
            thinking.push_str(&d.thinking);
            content.push_str(&d.content);
            calls.append(&mut d.calls);
        }
        let d = p.feed("", true).unwrap();
        content.push_str(&d.content);
        assert_eq!(thinking, "Plan carefully.");
        assert_eq!(content, "\n\n");
        assert_eq!(
            calls[0]["function"]["arguments"],
            json!({"city":"Boston","days":3})
        );
    }
    #[test]
    fn renderer_round_trips_reasoning_calls_and_multiple_tool_results() {
        let messages = vec![
            json!({"role":"user","content":"weather?"}),
            json!({"role":"assistant","content":"","thinking":"Need weather","tool_calls":[{"function":{"name":"weather","arguments":{"city":"Boston","days":3}}}]}),
            json!({"role":"tool","content":"sunny","tool_name":"weather"}),
            json!({"role":"tool","content":"warm","tool_name":"weather"}),
        ];
        let r = render(&messages, &ts(), Thinking::Low).unwrap();
        let s = std::str::from_utf8(&r).unwrap();
        assert!(s.contains("Need weather\n</think>"));
        assert!(s.contains("<parameter=days>\n3\n</parameter>"));
        assert!(s.contains("<tool_response>\nsunny\n</tool_response>\n<tool_response>\nwarm"));
        assert!(s.ends_with("assistant\n<think>\n"));
    }
    #[test]
    fn invalid_calls_fail_closed_and_literal_raw_is_untouched() {
        let mut p = Output::new(false, ts(), false);
        assert!(p.feed("<tool_call>oops", true).is_err());
        let mut p = Output::new(false, Vec::new(), true);
        assert_eq!(
            *p.feed("<think>literal 🦦", true).unwrap().content,
            "<think>literal 🦦"
        );
    }
    #[test]
    fn collected_and_streamed_calls_have_the_same_ollama_shape() {
        let mut output = Output::new(true, ts(), false);
        let mut collected = Collected::default();
        let mut delta=output.feed("plan</think><tool_call><function=weather><parameter=city>Boston</parameter><parameter=days>3</parameter></function></tool_call>",true).unwrap();
        let mut stream = json!({"message":{"role":"assistant","content":""}});
        delta.annotate(&mut stream, true);
        assert_eq!(stream["message"]["thinking"], "plan");
        assert_eq!(
            stream["message"]["tool_calls"][0]["function"]["arguments"]["days"],
            3
        );
        // Replay a separate parser to compare whole-response aggregation.
        let mut second = Output::new(true, ts(), false);
        collected.push(second.feed("plan</think><tool_call><function=weather><parameter=city>Boston</parameter><parameter=days>3</parameter></function></tool_call>",true).unwrap());
        let mut whole = json!({"message":{"role":"assistant","content":""}});
        collected.annotate(&mut whole, true);
        assert_eq!(whole, stream);
    }
}
