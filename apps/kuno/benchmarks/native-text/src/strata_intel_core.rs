//! Connected Intel core port. Shared original IQ tables are public weights;
//! they are loaded once, read directly, and never contain request content.
//! MIT Strata fb58e0dbc8399662c0e47c76578c6e878b14f6cf, iq_kernels.dp.cpp.
use cubecl::{prelude::ComputeClient, wgpu::WgpuRuntime};
use cubecl_runtime::server::{Handle, KernelArguments};
use kuno_text::{Error, Result};
use std::cell::RefCell;

pub(crate) const PUBLIC_TABLE_BYTES: u64 = 32768;
const TABLE_WORDS: usize = 8192;
thread_local! {static TABLE:RefCell<Option<Handle>>=const {RefCell::new(None)};}
pub(crate) const fn enabled() -> bool {
    cfg!(feature = "strata-intel-core")
}
fn reject() -> Error {
    Error::InvalidFormat("Strata Intel public table scope")
}
pub(crate) struct Tables {
    client: ComputeClient<WgpuRuntime>,
}
impl Tables {
    pub(crate) fn new(client: &ComputeClient<WgpuRuntime>) -> Result<Option<Self>> {
        if !enabled() {
            return Ok(None);
        }
        if TABLE.with(|t| t.borrow().is_some()) {
            return Err(reject());
        }
        let mut words = Vec::with_capacity(TABLE_WORDS);
        for dtype in [
            crate::quant_matvec::IQ3_XXS,
            crate::quant_matvec::IQ3_S,
            crate::quant_matvec::IQ2_S,
        ] {
            words.extend(crate::quant_matvec::iq_moe_integer::original_grid_words(
                dtype,
            ));
        }
        if words.len() != 2816 {
            return Err(reject());
        }
        words.resize(TABLE_WORDS, 0);
        let bytes = words.into_iter().flat_map(u32::to_le_bytes).collect();
        let handle = client.create_public_owned(bytes);
        let r = client
            .retained_resource_metadata(vec![handle.clone()])
            .map_err(crate::resident_util::fault)?;
        if handle.size_in_used() != PUBLIC_TABLE_BYTES
            || r.len() != 1
            || r[0].resource().offset != 0
            || r[0].resource().buffer.size() != PUBLIC_TABLE_BYTES
        {
            return Err(reject());
        }
        TABLE.with(|t| *t.borrow_mut() = Some(handle));
        Ok(Some(Self {
            client: client.clone(),
        }))
    }
}
impl Drop for Tables {
    fn drop(&mut self) {
        // Queue completion precedes dropping the last shared public model handle.
        let _ = self.client.flush();
        TABLE.with(|t| *t.borrow_mut() = None);
    }
}
pub(crate) fn bind(
    args: KernelArguments,
    stream: cubecl_common::stream_id::StreamId,
) -> Result<KernelArguments> {
    if !enabled() {
        return Ok(args);
    }
    TABLE.with(|t| {
        let t = t.borrow();
        let h = t.as_ref().ok_or_else(reject)?;
        if h.stream != stream || h.size_in_used() != PUBLIC_TABLE_BYTES {
            return Err(reject());
        }
        Ok(args.with_buffer(h.clone().binding()))
    })
}
pub(crate) fn shared_table(mut source: String, binding: usize, dtype: u32) -> String {
    if !enabled() {
        return source;
    }
    let n = crate::quant_matvec::iq_moe_integer::original_grid_words(dtype).len();
    if n > 0 {
        let decl = format!("var<workgroup> iq3_grid:array<u32,{n}>;");
        assert!(source.contains(&decl), "original shared grid declaration");
        source = source.replace(&decl, "");
        let start = source
            .find("switch local {")
            .expect("original public grid initializer");
        let end = crate::strata_port::function_end(&source, start);
        source.replace_range(start..end, "");
        let offset = match dtype {
            crate::quant_matvec::IQ3_XXS => 0,
            crate::quant_matvec::IQ3_S => 256,
            crate::quant_matvec::IQ2_S => 768,
            _ => unreachable!(),
        };
        source = source.replace("iq3_grid[", &format!("strata_grid[{offset}u+"));
    }
    // Declared for every GU format: actual owner shape is a dynamic guard, so
    // even formats without an IQ grid retain the same full descriptor layout.
    source.push_str(&format!(
        "\n@group(0) @binding({binding}) var<storage,read> strata_grid:array<u32>;\n"
    ));
    let guard = "    if arrayLength(&strata_grid)!=8192u {return false;}\n";
    let start = source.find("fn valid").expect("original owning guards");
    let brace = start + source[start..].find('{').unwrap() + 1;
    source.insert_str(brace, guard);
    source
}

/// Strata resolves one expert pointer before entering its packed dot loop.
/// WGSL cannot select a storage pointer, so specialize the five possible
/// source pages and select the complete projection once per workgroup.
/// This avoids a five-way source switch at every packed weight load.
pub(crate) fn specialize_projection(mut source: String, down: bool) -> String {
    if !enabled() {
        return source;
    }
    let mut helpers = Vec::new();
    let mut cursor = 0;
    while let Some(at) = source[cursor..].find("fn ") {
        let first = cursor + at;
        let name_first = first + 3;
        let name_end = name_first
            + source[name_first..]
                .find('(')
                .expect("shader function signature");
        let name = source[name_first..name_end].trim();
        let end = crate::strata_port::function_end(&source, first);
        if name.starts_with("gate_")
            || name.starts_with("up_")
            || name.starts_with("down_")
            || ["strata_gate", "strata_up", "strata_down"].contains(&name)
        {
            helpers.push((name.to_owned(), source[first..end].to_owned()));
        }
        cursor = end;
    }
    let first = source.find("fn main(").expect("projection entry");
    let end = crate::strata_port::function_end(&source, first);
    let brace = first + source[first..end].find('{').unwrap();
    let signature = &source[first + "fn main(".len()..brace];
    let args = signature.trim().strip_suffix(')').unwrap();
    let mut parameters = args.to_owned();
    while let Some(at) = parameters.find("@builtin(") {
        let stop = at + parameters[at..].find(')').unwrap() + 1;
        parameters.replace_range(at..stop, "");
    }
    let arguments = parameters
        .split(',')
        .map(|a| a.split_once(':').expect("projection parameter").0.trim())
        .collect::<Vec<_>>()
        .join(",");
    let body = source[brace + 1..end - 1].to_owned();
    let mut definitions = String::new();
    let roles: &[&str] = if down { &["down"] } else { &["gate", "up"] };
    for page in 0..5 {
        for (name, helper) in &helpers {
            if let Some(role) = name.strip_prefix("strata_") {
                if roles.contains(&role) {
                    definitions.push_str(&format!("fn {name}_page{page}(index:u32)->u32 {{return {role}{page}[index/4u][index%4u];}}\n"));
                }
            } else {
                definitions.push_str(&rename_projection_functions(helper, &helpers, page));
                definitions.push('\n');
            }
        }
        definitions.push_str(&format!(
            "fn strata_project_page{page}({parameters}) {{ {} }}\n",
            rename_projection_functions(&body, &helpers, page)
        ));
    }
    let mut dispatcher =
        String::from("\nif !valid_group(group.y) {return;}\nswitch strata_shard {\n");
    for page in 0..4 {
        dispatcher.push_str(&format!(
            "case {page}u: {{strata_project_page{page}({arguments});}}\n"
        ));
    }
    dispatcher.push_str(&format!(
        "default: {{strata_project_page4({arguments});}}\n}}\n"
    ));
    source.replace_range(brace + 1..end - 1, &dispatcher);
    source.push_str("\nvar<workgroup> strata_selected_page:u32;\n");
    source.push_str(&definitions);
    source
}
fn rename_projection_functions(
    source: &str,
    functions: &[(String, String)],
    page: usize,
) -> String {
    let mut result = String::with_capacity(source.len());
    let mut index = 0;
    let b = source.as_bytes();
    while index < b.len() {
        let start = index;
        if b[index].is_ascii_alphabetic() || b[index] == b'_' {
            index += 1;
            while index < b.len() && (b[index].is_ascii_alphanumeric() || b[index] == b'_') {
                index += 1;
            }
            let token = &source[start..index];
            result.push_str(token);
            if functions.iter().any(|(name, _)| name == token) {
                result.push_str(&format!("_page{page}"));
            }
        } else {
            let character = source[start..].chars().next().unwrap();
            index += character.len_utf8();
            result.push(character);
        }
    }
    result
}
