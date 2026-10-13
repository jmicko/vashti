//! Experimental Rust/Vulkan port of Strata's Qwen3.8 native execution policy.
//! Source: MIT Strata fb58e0dbc8399662c0e47c76578c6e878b14f6cf.
//! See strata-port-NOTICE.txt. No C++/SYCL inference library is linked.
use cubecl::prelude::CubeDim;
use std::sync::atomic::{AtomicU64, Ordering};

pub(crate) const fn enabled() -> bool {
    cfg!(feature = "strata-rust-port")
}
pub(crate) fn fused_sources() -> bool {
    static SELECTED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    enabled()
        && *SELECTED.get_or_init(|| {
            std::env::var("KUNO_STRATA_FUSED_EXPERTS").as_deref() == Ok("five-tier-public")
        })
}
pub(crate) fn profile_target() -> bool {
    enabled() && std::env::var("KUNO_STRATA_PROFILE_TARGET").as_deref() == Ok("round-8")
}
static GU: [AtomicU64; 4] = [const { AtomicU64::new(0) }; 4];
static DOWN: [AtomicU64; 4] = [const { AtomicU64::new(0) }; 4];
pub(crate) fn credit(down: bool, count: usize) {
    if enabled() && (1..=4).contains(&count) {
        (if down { &DOWN } else { &GU })[count - 1].fetch_add(1, Ordering::Relaxed);
    }
}
static LOOKUP_WINDOWS: AtomicU64 = AtomicU64::new(0);
static LOOKUP_PROPOSED: AtomicU64 = AtomicU64::new(0);
/// Port of Strata's in-request suffix drafter. Suggestions are always checked
/// by the actual target; no token oracle or persisted history is consulted.
pub(crate) fn lookup(prompt: &[u32], emitted: &[u32], maximum: usize) -> Option<Lookup> {
    if !enabled() || maximum == 0 {
        return None;
    }
    let mut history = zeroize::Zeroizing::new(Vec::with_capacity(prompt.len() + emitted.len()));
    history.extend_from_slice(prompt);
    history.extend_from_slice(emitted);
    if history.len() < 4 {
        return None;
    }
    let current = history.len() - 1;
    let mut best = (0, 0);
    let mut ways = 0;
    // Strata keeps the four most recent trigram occurrences and extends each
    // backwards up to64 tokens, preferring the longest and then most recent.
    // The bounded public fixture is small enough to scan without a hash table.
    for end in (2..current).rev() {
        if history[end - 2..=end] != history[current - 2..=current] {
            continue;
        }
        ways += 1;
        let mut matched = 3;
        while matched < 64 && matched <= end && history[end - matched] == history[current - matched]
        {
            matched += 1;
        }
        if matched > best.1 {
            best = (end, matched);
        }
        if ways == 4 {
            break;
        }
    }
    if best.1 < 3 {
        return None;
    }
    let end = (best.0 + 1 + maximum).min(history.len());
    Some(Lookup {
        tokens: zeroize::Zeroizing::new(history[best.0 + 1..end].to_vec()),
        matched: best.1,
    })
}
pub(crate) struct Lookup {
    pub(crate) tokens: zeroize::Zeroizing<Vec<u32>>,
    pub(crate) matched: usize,
}
pub(crate) fn credit_lookup(proposed: usize) {
    LOOKUP_WINDOWS.fetch_add(1, Ordering::Relaxed);
    LOOKUP_PROPOSED.fetch_add(proposed as u64, Ordering::Relaxed);
}

/// MIT Strata DraftPolicy cost/acceptance controller, limited to the four-row
/// verifier available here. All observations are request-local and erased.
#[derive(Default)]
pub(crate) struct DraftPolicy {
    cost: [f64; 5],
    cost_n: [f64; 5],
    mtp_tok: [f64; 5],
    mtp_n: [f64; 5],
    ok: [f64; 4],
    bad: [f64; 4],
    last: [f64; 5],
    rounds: f64,
    mtp_ok: [f64; 3],
    mtp_bad: [f64; 3],
    chosen: [u64; 5],
}
impl DraftPolicy {
    fn bucket(matched: usize) -> usize {
        if matched < 6 {
            0
        } else if matched < 12 {
            1
        } else if matched < 24 {
            2
        } else {
            3
        }
    }
    fn cost_ms(&self, t: usize) -> f64 {
        const SHAPE: [f64; 5] = [0., 1., 1.35, 1.7, 2.05];
        if self.cost_n[t] > 0. {
            return self.cost[t];
        }
        let (mut numerator, mut denominator) = (0., 0.);
        for (u, _) in SHAPE.iter().enumerate().skip(1) {
            if self.cost_n[u] > 0. {
                let w = self.cost_n[u].min(20.);
                numerator += w * self.cost[u] * SHAPE[t] / SHAPE[u];
                denominator += w;
            }
        }
        if denominator > 0. {
            numerator / denominator
        } else {
            SHAPE[t]
        }
    }
    pub(crate) fn choose_mtp(&mut self, maximum: usize) -> usize {
        if !enabled() || !(1..=4).contains(&maximum) {
            return maximum;
        }
        let mut selected = maximum;
        // Keep the existing four-row path during warm-up, then estimate each
        // tested prefix from position-specific conditional acceptance. Rejected
        // suffixes never count as additional failed positions.
        if self.rounds >= 12. && maximum >= 3 {
            let mut expected = [0.; 5];
            expected[1] = 1.;
            let mut probability = 1.;
            for position in 0..maximum - 1 {
                probability *= (self.mtp_ok[position] + 4. * 0.86)
                    / (self.mtp_ok[position] + self.mtp_bad[position] + 4.);
                expected[position + 2] = expected[position + 1] + probability;
            }
            let baseline = expected[maximum] / self.cost_ms(maximum).max(0.001);
            let mut best = baseline;
            for count in 2..maximum {
                let rate = expected[count] / self.cost_ms(count).max(0.001);
                if rate > best * 1.03 {
                    best = rate;
                    selected = count;
                }
            }
        }
        self.chosen[selected] += 1;
        selected
    }
    pub(crate) fn receipt(&self) -> serde_json::Value {
        serde_json::json!({"adaptive_mtp_length":true,"chosen_window_counts":self.chosen,
            "mtp_conditional_accepted":self.mtp_ok,"mtp_conditional_rejected":self.mtp_bad,"history_persisted":false})
    }
    pub(crate) fn choose(&self, t: usize, proposal: &Lookup) -> Option<usize> {
        let bucket = Self::bucket(proposal.matched);
        let prior = [0.75, 0.88, 0.93, 0.96][bucket];
        let q = (self.ok[bucket] + 4. * prior) / (self.ok[bucket] + self.bad[bucket] + 4.);
        let mtp = if self.mtp_n[t] > 0. {
            self.mtp_tok[t]
        } else {
            1. + 0.7 * (t - 1) as f64
        };
        let base = mtp / self.cost_ms(t);
        let (mut expected, mut qi, mut best, mut size) = (1., 1., 0., 0);
        for k in 1..=proposal.tokens.len().min(t - 1) {
            qi *= q;
            expected += qi;
            let rate = expected / self.cost_ms(k + 1);
            if rate > best {
                best = rate;
                size = k + 1;
            }
        }
        (size > 0 && best > base * 1.03).then_some(size)
    }
    pub(crate) fn observe(
        &mut self,
        lookup: bool,
        t: usize,
        accepted: usize,
        matched: usize,
        ns: u64,
    ) {
        let ms = ns as f64 / 1e6;
        self.cost[t] = if self.cost_n[t] > 0. {
            0.9 * self.cost[t] + 0.1 * ms
        } else {
            ms
        };
        self.cost_n[t] += 1.;
        self.last[t] = self.rounds;
        self.rounds += 1.;
        if lookup {
            let b = Self::bucket(matched);
            self.ok[b] = 0.97 * self.ok[b] + (accepted - 1) as f64;
            self.bad[b] = 0.97 * self.bad[b] + if accepted < t { 1. } else { 0. };
        } else {
            for position in 0..t - 1 {
                if position <= accepted - 1 {
                    self.mtp_ok[position] *= 0.97;
                    self.mtp_bad[position] *= 0.97;
                    if position < accepted - 1 {
                        self.mtp_ok[position] += 1.;
                    } else {
                        self.mtp_bad[position] += 1.;
                    }
                }
            }
            self.mtp_tok[t] = if self.mtp_n[t] > 0. {
                0.95 * self.mtp_tok[t] + 0.05 * accepted as f64
            } else {
                accepted as f64
            };
            self.mtp_n[t] += 1.;
        }
    }
}
impl Drop for DraftPolicy {
    fn drop(&mut self) {
        use zeroize::Zeroize;
        self.cost.zeroize();
        self.cost_n.zeroize();
        self.mtp_tok.zeroize();
        self.mtp_n.zeroize();
        self.ok.zeroize();
        self.bad.zeroize();
        self.last.zeroize();
        self.rounds.zeroize();
        self.mtp_ok.zeroize();
        self.mtp_bad.zeroize();
        self.chosen.zeroize();
    }
}
pub(crate) fn public_preview(text: &str) {
    // Only the fixed public fixtures reach this lab; custom runner displays
    // this line live and omits it from the saved numeric diagnostic log.
    if enabled() {
        println!(
            "strata_public_preview={}",
            serde_json::to_string(text).unwrap()
        );
    }
}
pub(crate) fn receipt() -> serde_json::Value {
    serde_json::json!({"enabled": enabled(), "upstream_revision": "fb58e0dbc8399662c0e47c76578c6e878b14f6cf",
        "backend": "Rust/WGSL/Vulkan", "activation_planes": 1, "activation_scale": "fp16-q8_1",
        "projection_workgroup":256, "rows_per_workgroup":32, "group_stride":4,"suffix_lookup_windows":LOOKUP_WINDOWS.load(Ordering::Relaxed),"suffix_lookup_proposed":LOOKUP_PROPOSED.load(Ordering::Relaxed),
        "gu_dispatches_by_member_count":GU.each_ref().map(|v|v.load(Ordering::Relaxed)),
        "down_dispatches_by_member_count":DOWN.each_ref().map(|v|v.load(Ordering::Relaxed)),
        "fused_five_tier_experts":fused_sources(), "original_cpu13_precision_contract": false, "intermediate_numeric_scans":false,"published_outputs_checked_finite":true,"full_private_erasure":true,"installed_or_production": false})
}

fn replace(source: String, old: &str, new: &str) -> String {
    assert!(source.contains(old), "Strata port template drift: {old}");
    source.replace(old, new)
}
fn function_end(source: &str, first: usize) -> usize {
    let brace = first + source[first..].find('{').expect("function body");
    let mut depth = 0;
    for (i, b) in source.as_bytes().iter().enumerate().skip(brace) {
        match b {
            b'{' => depth += 1,
            b'}' => {
                depth -= 1;
                if depth == 0 {
                    return i + 1;
                }
            }
            _ => {}
        }
    }
    panic!("unclosed static shader function")
}
/// Native aligned vector source reads, preserving every original packed byte.
/// All frozen expert role roots are complete and multiples of sixteen bytes.
/// Scalar metadata uses one aligned vector; the IQ payload's load8 shares it.
pub(crate) fn wide_weights(mut source: String, down_projection: bool) -> String {
    if !enabled() {
        return source;
    }
    let names: &[&str] = if down_projection {
        &["down"]
    } else {
        &["gate", "up"]
    };
    if !source.contains(&format!("var<storage,read> {}:array<u32>", names[0])) {
        return source;
    }
    for name in names {
        source = source
            .replace(
                &format!("var<storage,read> {name}:array<u32>"),
                &format!("var<storage,read> {name}:array<vec4<u32>>"),
            )
            .replace(
                &format!("arrayLength(&{name})"),
                &format!("(arrayLength(&{name})*4u)"),
            );
        let needle = format!("{name}[");
        while let Some(first) = source.find(&needle) {
            let end = first
                + source[first..]
                    .find(']')
                    .expect("original scalar packed read");
            let expression = &source[first + needle.len()..end];
            let replacement = format!("{name}_scalar({expression})");
            source.replace_range(first..end + 1, &replacement);
        }
        if let Some(first) = source.find(&format!("fn {name}_load8(")) {
            let end = function_end(&source, first);
            source.replace_range(
                first..end,
                &format!(
                    r#"fn {name}_load8(at:u32)->vec2<u32> {{
                let low={name}[at/16u];var high=low;
                if at%16u>8u {{high={name}[at/16u+1u];}}
                var a=low.x;var b=low.y;var c=low.z;
                switch (at%16u)/4u {{
                    case 1u: {{a=low.y;b=low.z;c=low.w;}}
                    case 2u: {{a=low.z;b=low.w;c=high.x;}}
                    case 3u: {{a=low.w;b=high.x;c=high.y;}}
                    default: {{}}
                }}
                if at%4u==0u {{return vec2<u32>(a,b);}}
                return vec2<u32>((a>>16u)|(b<<16u),(b>>16u)|(c<<16u));
            }}"#
                ),
            );
        }
        source.push_str(&format!(
            "\nfn {name}_scalar(index:u32)->u32 {{return {name}[index/4u][index%4u];}}\n"
        ));
    }
    source
}

pub(crate) fn router_shader(mut source: String) -> String {
    if !enabled() {
        return source;
    }
    let first = source
        .find("    for(var row=0u;row<p.count;row++) {")
        .expect("router selection");
    let end = source[first..]
        .find("    for(var index=local;index<804u;")
        .unwrap()
        + first;
    source.replace_range(first..end, include_str!("strata-router-body.wgsl"));
    source = source
        .replace(
            "@compute @workgroup_size(64)",
            "@compute @workgroup_size(128)",
        )
        .replace("best:array<f32,64>", "best:array<f32,128>")
        .replace("best_id:array<u32,64>", "best_id:array<u32,128>")
        .replace(
            "var<workgroup> minimum_gap:f32;",
            "var<workgroup> minimum_gap:f32;\nvar<workgroup> row_gap:array<f32,4>;",
        )
        .replace("index+=64u", "index+=128u");
    source.replace("fn main(@builtin(local_invocation_index) local:u32) {",
        "fn main(@builtin(local_invocation_index) local:u32,@builtin(subgroup_id) sg:u32,@builtin(subgroup_invocation_id) lane:u32,@builtin(subgroup_size) width:u32) {")
        .replace("if !shape() {reject(1u);}","if !shape() || (width!=16u && width!=32u) {reject(1u);}")
}

/// Single Q8_1 plane. Codes use the unrounded F32 scale; the dot's scale is
/// rounded to F16, exactly the representation used by Strata's native IQ path.
fn quantizer(mut source: String) -> String {
    let first = source
        .find("fn prepare_q8(block:u32)")
        .expect("quantizer entry");
    let end = function_end(&source, first);
    let original = &source[first..end];
    let bound = original
        .split_once("block>=")
        .unwrap()
        .1
        .split_once(" {return;}")
        .unwrap()
        .0;
    let body = format!(
        r#"fn prepare_q8(block:u32) {{
    if block>={bound} {{return;}}
    var amax=0.0;var valid=true;
    for(var i=0u;i<32u;i++) {{let x=cached_input[block*32u+i];valid=valid && abs(x)<3.4028234663852886e38;amax=max(amax,abs(x));}}
    if !valid {{clear_q8(block);return;}}
    let d=amax/127.0;
    scale_first[block]=unpack2x16float(pack2x16float(vec2<f32>(d,0.0))).x;
    scale_second[block]=0.0;
    for(var word=0u;word<8u;word++) {{
        var packed=0u;
        for(var byte=0u;byte<4u;byte++) {{let q=q8_code(cached_input[block*32u+word*4u+byte],d);packed|=(bitcast<u32>(q)&255u)<<(byte*8u);}}
        packed_first[block*8u+word]=packed;packed_second[block*8u+word]=0u;
    }}
}}"#
    );
    source.replace_range(first..end, &body);
    source
}

fn rows(source: String, down: bool) -> String {
    let source = replace(
        source,
        "@compute @workgroup_size(64)",
        "@compute @workgroup_size(256)",
    );
    let source = replace(
        source,
        "fn main(@builtin(workgroup_id) group:vec3<u32>,@builtin(local_invocation_index) local:u32) {",
        "fn main(@builtin(workgroup_id) group:vec3<u32>,@builtin(local_invocation_index) local:u32,@builtin(subgroup_id) sg:u32,@builtin(subgroup_invocation_id) sub_lane:u32,@builtin(subgroup_size) width:u32,@builtin(num_subgroups) subgroups:u32) {",
    );
    let source = replace(
        source,
        "if group.x>=",
        "if (width!=16u && width!=32u) || subgroups!=256u/width || group.x>=",
    );
    let source = replace(
        source,
        "let row=group.x*8u+local/8u;let lane=local%8u;",
        "let row=group.x*32u+sg*(width/8u)+sub_lane/8u;let lane=sub_lane%8u;",
    );
    if down {
        replace(source, "group.x>=320u", "group.x>=80u")
    } else {
        replace(source, "group.x>=80u", "group.x>=20u")
    }
}

fn reduction(mut source: String, down: bool, members: usize) -> String {
    let vec = format!("vec{members}<f32>");
    let (first, last) = if down {
        let marker =
            format!("for(var j=0u;j<{members}u;j++) {{sums[local*{members}u+j]=values[j];}}");
        let at = source.find(&marker).expect("DOWN tree");
        let first = source[..at].rfind('\n').unwrap() + 1;
        let last = first + source[first..].find("    if lane==0u").unwrap();
        (first, last)
    } else {
        let marker = format!(
            "for(var j=0u;j<{members}u;j++) {{gate_sums[local*{members}u+j]=gates[j];up_sums[local*{members}u+j]=ups[j];}}"
        );
        let at = source.find(&marker).expect("GU tree");
        let first = source[..at].rfind('\n').unwrap() + 1;
        let last = at + source[at..].find("if lane==0u").unwrap();
        (first, last)
    };
    let tree = if down {
        format!(
            "    for(var stride=4u;stride>0u;stride/=2u) {{let other:{vec}=subgroupShuffleDown(values,stride);if lane<stride {{values+=other;}}}}\n"
        )
    } else {
        format!(
            "    for(var stride=4u;stride>0u;stride/=2u) {{let g:{vec}=subgroupShuffleDown(gates,stride);let u:{vec}=subgroupShuffleDown(ups,stride);if lane<stride {{gates+=g;ups+=u;}}}}\n    "
        )
    };
    source.replace_range(first..last, &tree);
    for (i, axis) in ["x", "y", "z", "w"].iter().take(members).enumerate() {
        for (old, new) in if down {
            vec![("sums", "values")]
        } else {
            vec![("gate_sums", "gates"), ("up_sums", "ups")]
        } {
            source = source.replace(
                &format!("{old}[local*{members}u+{i}u]"),
                &format!("{new}.{axis}"),
            );
        }
    }
    for name in if down {
        vec!["sums"]
    } else {
        vec!["gate_sums", "up_sums"]
    } {
        source = replace(
            source,
            &format!("var<workgroup> {name}:array<f32,{}>;\n", members * 64),
            "",
        );
    }
    if !down {
        source=source.replace("            // All lanes finish reading their reduction before the next group\n            // can overwrite any shared cell, including unused member cells.\n            workgroupBarrier();","");
    }
    source
}

fn gate_up(mut source: String, members: usize) -> String {
    source = rows(source, false);
    source = reduction(source, false, members);
    let words = members * 720;
    source = source.replacen(
        "@compute @workgroup_size(256)",
        &format!("var<workgroup> strata_input:array<u32,{words}>;\n@compute @workgroup_size(256)"),
        1,
    );
    let first = source
        .find("    workgroupBarrier();")
        .expect("immutable codebook barrier");
    let copy = format!(
        r#"    for(var word=local;word<{words}u;word+=256u) {{
        let token=word/720u;let at=word%720u;var value=0u;
        if token<p.h {{let frame=64u+token*1440u;if at<640u {{value=planes[frame+at];}}else {{value=planes[frame+1280u+at-640u];}}}}
        strata_input[word]=value;
    }}
"#
    );
    source.insert_str(first, &copy);
    for axis in ["x", "y", "z", "w"].iter().take(members) {
        let old = format!("frames.{axis}=64u+(");
        if let Some(first) = source.find(&old) {
            let end = first + source[first..].find(';').unwrap() + 1;
            let line = source[first..end].to_owned();
            source.replace_range(
                first..end,
                &line.replace("=64u+(", "=(").replace("*1440u", "*720u"),
            );
        }
        let dots = format!(
            "{{let first=planes[frames.{axis}+at];let second=planes[frames.{axis}+640u+at];gf.{axis}+=dot4I8Packed(gq,first);gs.{axis}+=dot4I8Packed(gq,second);uf.{axis}+=dot4I8Packed(uq,first);us.{axis}+=dot4I8Packed(uq,second);}}"
        );
        let primary = format!(
            "{{let first=strata_input[frames.{axis}+at];gf.{axis}+=dot4I8Packed(gq,first);uf.{axis}+=dot4I8Packed(uq,first);}}"
        );
        source = replace(source, &dots, &primary);
        let scales = format!(
            "{{let a=bitcast<f32>(planes[frames.{axis}+1280u+block]);let b=bitcast<f32>(planes[frames.{axis}+1360u+block]);let gp=a*f32(gf.{axis})+b*f32(gs.{axis});let up=a*f32(uf.{axis})+b*f32(us.{axis});gates.{axis}+=gd*gp;ups.{axis}+=ud*up;}}"
        );
        let primary = format!(
            "{{let a=bitcast<f32>(strata_input[frames.{axis}+640u+block]);gates.{axis}=fma(gd,a*f32(gf.{axis}),gates.{axis});ups.{axis}=fma(ud,a*f32(uf.{axis}),ups.{axis});}}"
        );
        source = replace(source, &scales, &primary);
    }
    source
}

fn down(mut source: String, members: usize) -> String {
    source = rows(source, true);
    source = reduction(source, true, members);
    let words = members * 180;
    source = source.replacen(
        "@compute @workgroup_size(256)",
        &format!("var<workgroup> strata_input:array<u32,{words}>;\n@compute @workgroup_size(256)"),
        1,
    );
    let copy = format!(
        r#"    for(var word=local;word<{words}u;word+=256u) {{
        let member=word/180u;let at=word%180u;var value=0u;
        if member<length {{let frame=frames[member];if at<160u {{value=planes[frame+at];}}else {{value=planes[frame+320u+at-160u];}}}}
        strata_input[word]=value;
    }}
    workgroupBarrier();
    frames=vec{members}<u32>({});
"#,
        (0..members)
            .map(|m| format!("{}u", m * 180))
            .collect::<Vec<_>>()
            .join(",")
    );
    let before = "    var values=";
    let at = source.find(before).unwrap();
    source.insert_str(at, &copy);
    for axis in ["x", "y", "z", "w"].iter().take(members) {
        let old = format!(
            "{{first.{axis}+=dot4I8Packed(q,planes[frames.{axis}+at]);second.{axis}+=dot4I8Packed(q,planes[frames.{axis}+160u+at]);}}"
        );
        let new = format!("{{first.{axis}+=dot4I8Packed(q,strata_input[frames.{axis}+at]);}}");
        source = replace(source, &old, &new);
        let old = format!(
            "{{let a=bitcast<f32>(planes[frames.{axis}+320u+block]);let b=bitcast<f32>(planes[frames.{axis}+340u+block]);let pair=a*f32(first.{axis})+b*f32(second.{axis});values.{axis}+=d*pair;}}"
        );
        let new = format!(
            "{{let a=bitcast<f32>(strata_input[frames.{axis}+160u+block]);values.{axis}=fma(d,a*f32(first.{axis}),values.{axis});}}"
        );
        source = replace(source, &old, &new);
    }
    source
}

pub(crate) fn shader(mut source: String, down_projection: bool, members: usize) -> String {
    if !enabled() {
        return source;
    }
    if source.contains("fn prepare_q8(block:u32)") {
        source = quantizer(source);
    }
    // Original record ownership/erasure is retained, with the actual new grid.
    source = source
        .replace("+2u]=80u;", "+2u]=20u;")
        .replace("+5u]=320u;", "+5u]=80u;")
        .replace("planes[base+2u]!=80u", "planes[base+2u]!=20u")
        .replace("planes[base+5u]!=320u", "planes[base+5u]!=80u");
    if source.contains("requires packed_4x8_integer_dot_product")
        && source.contains("let row=group.x*8u+local/8u;")
    {
        if down_projection {
            down(source, members)
        } else {
            gate_up(source, members)
        }
    } else {
        source
    }
}
pub(crate) fn dimension(original: CubeDim, project: bool) -> CubeDim {
    if enabled() && project {
        CubeDim::new_1d(256)
    } else {
        original
    }
}
pub(crate) fn validate_limits(limits: &wgpu::Limits) -> kuno_text::Result<()> {
    if enabled()
        && (limits.max_compute_invocations_per_workgroup < 256
            || limits.max_compute_workgroup_size_x < 256
            || limits.max_compute_workgroup_storage_size < 19712
            || fused_sources()
                && (limits.max_storage_buffers_per_shader_stage < 14
                    || limits.max_bindings_per_bind_group < 15))
    {
        return Err(kuno_text::Error::InvalidFormat(
            "Strata Rust workgroup limits",
        ));
    }
    Ok(())
}

/// Final full-set WGSL validation and SPIR-V lowering, after the one-shot port.
pub(crate) fn validate_shader_set() -> Result<(), Box<dyn std::error::Error>> {
    if !enabled() {
        return Err("Strata port feature required".into());
    }
    use wgpu::naga as n;
    for source in [
        router_shader(include_str!("resident_batch_router_gpu.wgsl").into()),
        router_shader(include_str!("resident_batch_router_gpu_four.wgsl").into()),
        crate::public_host_expert_mirror::strata_gpu_exchange::SHADER.into(),
        include_str!("strata-packet.wgsl").into(),
    ] {
        let module = n::front::wgsl::parse_str(&source).map_err(|e| e.emit_to_string(&source))?;
        let _ = n::valid::Validator::new(
            n::valid::ValidationFlags::all(),
            n::valid::Capabilities::all(),
        )
        .validate(&module)
        .map_err(|e| e.emit_to_string(&source))?;
        println!(
            "strata_new_controller_shader_accepted entries={}",
            module.entry_points.len()
        );
    }
    let mut shaders = crate::quant_matvec::frozen_gu_integer::strata_compiled_sources();
    shaders.extend(crate::quant_matvec::frozen_down_integer::strata_compiled_sources());
    shaders.extend(crate::q6_integer_matvec::batch::strata_compiled_sources());
    shaders.extend(crate::strata_expert_fused::compiled_sources());
    for (index, compiled) in shaders.iter().enumerate() {
        let module = n::front::wgsl::parse_str(&compiled.source)
            .map_err(|e| e.emit_to_string(&compiled.source))?;
        let info = n::valid::Validator::new(
            n::valid::ValidationFlags::all(),
            n::valid::Capabilities::all(),
        )
        .validate(&module)
        .map_err(|e| e.emit_to_string(&compiled.source))?;
        if compiled.cube_dim != CubeDim::new_1d(module.entry_points[0].workgroup_size[0]) {
            return Err("port metadata/WGSL dimension disagreement".into());
        }
        n::back::spv::write_vec(
            &module,
            &info,
            &n::back::spv::Options {
                lang_version: (1, 3),
                zero_initialize_workgroup_memory:
                    n::back::spv::ZeroInitializeWorkgroupMemoryMode::None,
                ..Default::default()
            },
            Some(&n::back::spv::PipelineOptions {
                shader_stage: n::ShaderStage::Compute,
                entry_point: "main".into(),
            }),
        )?;
        println!(
            "strata_shader_accepted index={index} workgroup={}",
            module.entry_points[0].workgroup_size[0]
        );
    }
    println!("strata_shader_set_complete shaders={}", shaders.len());
    Ok(())
}
