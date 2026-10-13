//! Unused count4 compact-shuffle DOWN candidate: copy original activation bits
//! once per workgroup before the unchanged two-plane dot/reduction body.
//! Source scheduling inspiration: pinned MIT Strata fb58e0d,
//! sycl/src/kernels/cuda/iq_kernels.dp.cpp native_down_multi_kernel. Existing
//! iq_moe_integer-NOTICE.txt retains upstream attribution. This is Rust-emitted
//! WGSL, not a linked SYCL/C++ implementation.

fn replace_once(source: String, before: &str, after: &str) -> String {
    assert_eq!(
        source.matches(before).count(),
        1,
        "fixed staged DOWN template drift"
    );
    source.replacen(before, after, 1)
}

pub(super) const COPY: &str = r#"    // Every local lane participates after the original uniform source guards.
    // Actual MAP members can be noncontiguous; frames retain those original IDs.
    for(var word=local;word<1440u;word+=64u) {
        let member=word/360u;var value=0u;
        if member<length {value=planes[frames[member]+word%360u];}
        staged_planes[word]=value;
    }
    workgroupBarrier();
    frames=vec4<u32>(0u,360u,720u,1080u);
"#;

pub(super) fn source(source: String) -> String {
    let source = replace_once(
        source,
        "@compute @workgroup_size(64)",
        "var<workgroup> staged_planes:array<u32,1440>;\n@compute @workgroup_size(64)",
    );
    let source = replace_once(
        source,
        "    var values=vec4<f32>(0.0);",
        &format!("{COPY}    var values=vec4<f32>(0.0);"),
    );
    // Only the generated projection accesses change storage class. Source
    // readiness/header guards and the cooperative copy still read full owners.
    let mut source = source;
    for axis in ["x", "y", "z", "w"] {
        let original = format!("planes[frames.{axis}+");
        assert_eq!(source.matches(&original).count(), 4);
        source = source.replace(&original, &format!("staged_planes[frames.{axis}+"));
    }
    source
}
