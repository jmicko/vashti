//! Scalarize the original fixed sixteen router candidates, like the pinned
//! SYCL route_kernel<16> unrolls. Arithmetic, rank order, masks and faults stay.
const LOOP: &str = "for(var i=0u;i<16u;i++) {";
fn close(s: &str, first: usize) -> usize {
    assert_eq!(s.as_bytes()[first], b'{');
    let mut depth = 0usize;
    for (n, &b) in s.as_bytes()[first..].iter().enumerate() {
        if b == b'{' {
            depth += 1;
        }
        if b == b'}' {
            depth -= 1;
            if depth == 0 {
                return first + n;
            }
        }
    }
    panic!("original router loop closing brace");
}
fn specialize_identifier(body: &str, value: usize) -> String {
    let mut result = String::new();
    let mut at = 0;
    while at < body.len() {
        let start = at;
        if body.as_bytes()[at].is_ascii_alphanumeric() || body.as_bytes()[at] == b'_' {
            while at < body.len()
                && (body.as_bytes()[at].is_ascii_alphanumeric() || body.as_bytes()[at] == b'_')
            {
                at += 1;
            }
            if &body[start..at] == "i" {
                result.push_str(&format!("{value}u"));
            } else {
                result.push_str(&body[start..at]);
            }
        } else {
            at += 1;
            result.push_str(&body[start..at]);
        }
    }
    result
}
pub(super) fn source(mut s: String) -> String {
    let declaration = "var candidates:array<f32,16>;";
    assert_eq!(
        s.matches(declaration).count(),
        1,
        "original router array drift"
    );
    assert_eq!(
        s.matches(LOOP).count(),
        5,
        "original fixed router loops drift"
    );
    for _ in 0..5 {
        let first = s.find(LOOP).unwrap();
        let open = first + LOOP.len() - 1;
        let end = close(&s, open);
        let body = &s[open + 1..end];
        let mut expanded = String::new();
        for i in 0..16 {
            expanded.push_str("{\n");
            expanded.push_str(&specialize_identifier(body, i));
            expanded.push_str("\n}\n");
        }
        s.replace_range(first..end + 1, &expanded);
    }
    let variables = (0..16)
        .map(|i| format!("var candidate{i}:f32;\n"))
        .collect::<String>();
    s = s.replacen(declaration, &variables, 1);
    for i in 0..16 {
        s = s.replace(&format!("candidates[{i}u]"), &format!("candidate{i}"));
    }
    assert!(
        !s.contains("candidates["),
        "no dynamically addressed candidate may remain"
    );
    s
}
#[cfg(test)]
mod tests {
    use super::*;
    use wgpu::naga as n;
    fn compiled(source: &str) -> (n::Module, Vec<u32>) {
        let m = n::front::wgsl::parse_str(source)
            .unwrap_or_else(|e| panic!("{}", e.emit_to_string(source)));
        let info = n::valid::Validator::new(
            n::valid::ValidationFlags::all(),
            n::valid::Capabilities::all(),
        )
        .validate(&m)
        .unwrap();
        let words = n::back::spv::write_vec(
            &m,
            &info,
            &n::back::spv::Options::default(),
            Some(&n::back::spv::PipelineOptions {
                shader_stage: n::ShaderStage::Compute,
                entry_point: "main".into(),
            }),
        )
        .unwrap();
        (m, words)
    }
    #[test]
    fn actual_scalarized_route_keeps_owners_geometry_and_eliminates_private_array() {
        let base = crate::strata_prompt::groups::shader_prefix()
            + include_str!("strata_prompt_route.wgsl");
        let new = source(base.clone());
        let (old, _) = compiled(&base);
        let (new, words) = compiled(&new);
        assert_eq!(
            old.entry_points[0].workgroup_size,
            new.entry_points[0].workgroup_size
        );
        let bindings = |m: &n::Module| {
            m.global_variables
                .iter()
                .map(|(_, g)| (g.binding.clone(), g.space))
                .collect::<Vec<_>>()
        };
        assert_eq!(bindings(&old), bindings(&new));
        let arrays = |m: &n::Module| {
            m.entry_points[0]
                .function
                .local_variables
                .iter()
                .filter(|(_, v)| matches!(m.types[v.ty].inner, n::TypeInner::Array { .. }))
                .count()
        };
        assert_eq!(arrays(&old), 1);
        assert_eq!(arrays(&new), 0);
        assert!(!words.is_empty());
    }
    #[test]
    fn expansion_is_exact_ordered_substitution_and_rejects_source_drift() {
        let base = crate::strata_prompt::groups::shader_prefix()
            + include_str!("strata_prompt_route.wgsl");
        let new = source(base.clone());
        // Every original non-loop interval, including all rank selection,
        // tie/remap/causal/fault publication, survives in its original order.
        let mut rest = base.as_str();
        let mut cursor = 0;
        loop {
            let Some(first) = rest.find(LOOP) else {
                assert!(new[cursor..].ends_with(rest));
                break;
            };
            let prefix = &rest[..first];
            // Only the first prefix changes its candidate declaration.
            let prefix = prefix.replace(
                "var candidates:array<f32,16>;",
                &(0..16)
                    .map(|i| format!("var candidate{i}:f32;\n"))
                    .collect::<String>(),
            );
            let relative = new[cursor..].find(&prefix).unwrap();
            assert_eq!(relative, 0);
            cursor += prefix.len();
            let open = first + LOOP.len() - 1;
            let end = close(rest, open);
            let body = &rest[open + 1..end];
            for i in 0..16 {
                let mut expected = format!("{{\n{}\n}}\n", specialize_identifier(body, i));
                for j in 0..16 {
                    expected =
                        expected.replace(&format!("candidates[{j}u]"), &format!("candidate{j}"));
                }
                assert!(new[cursor..].starts_with(&expected));
                cursor += expected.len();
            }
            rest = &rest[end + 1..];
        }
        assert!(
            std::panic::catch_unwind(|| source(base.replacen(
                LOOP,
                "for(var i=0u;i<15u;i++) {",
                1
            )))
            .is_err()
        );
    }
}
