//! Fenced coarse timing for compiled public requests only. The private worker
//! never enables this switch; no request values or contents enter the receipt.
use cubecl::{prelude::*, wgpu::WgpuRuntime};
use std::{
    cell::{Cell, RefCell},
    collections::BTreeMap,
    sync::atomic::{AtomicBool, AtomicU64, Ordering},
    time::Instant,
};
static ENABLED: AtomicBool = AtomicBool::new(false);
static REQUEST: AtomicU64 = AtomicU64::new(0);
static FULL_HOST_TRACE: AtomicBool = AtomicBool::new(false);
thread_local! { static TOTALS: RefCell<BTreeMap<&'static str, (u64,u64,u64)>> = const { RefCell::new(BTreeMap::new()) }; }
thread_local! { static GR_SAMPLE: Cell<bool> = const { Cell::new(false) }; }

/// Six fenced substeps for ONE compiled-public HC attention invocation only.
/// Private/clean execution does not create a timer or query the environment.
/// This diagnostic adds fences, so its elapsed times are not clean selection.
pub(crate) struct GrSample(Option<bool>);
impl GrSample {
    pub(crate) fn new(start: usize, count: usize, layer: usize) -> Self {
        if !ENABLED.load(Ordering::Acquire) {
            return Self(None);
        }
        let selected = NativeSelection {
            authorized: std::env::var("KUNO_LAB_LANGUAGE_PUBLIC_ONLY").as_deref() == Ok("1"),
            request: REQUEST.load(Ordering::Relaxed),
            start,
            count,
            layer,
        }.selects();
        Self(selected.then(|| GR_SAMPLE.with(|v| v.replace(true))))
    }
}
impl Drop for GrSample {
    fn drop(&mut self) {
        if let Some(previous) = self.0 {
            GR_SAMPLE.with(|v| v.set(previous));
        }
    }
}
pub(crate) fn enable_compiled_public() {
    ENABLED.store(true, Ordering::Release);
}
/// Only the new fixed-public CLI calls this, after ordinary host-mode admission.
pub(crate) fn enable_full_host_trace() {FULL_HOST_TRACE.store(true,Ordering::Release);}
pub(crate) fn compiled_public_enabled() -> bool {
    ENABLED.load(Ordering::Acquire)
}
pub(crate) fn begin() {
    if ENABLED.load(Ordering::Acquire) {
        REQUEST.fetch_add(1, Ordering::Relaxed);
    }
    TOTALS.with(|v| v.borrow_mut().clear());
}

// PUBLIC_PREFILL_NATIVE_SAMPLE_BEGIN
#[cfg(not(feature="strata-small-quantized-matrix"))]
const NATIVE_CLASSES: [&str; 7] = [
    "dequant",
    "matrix",
    "gather",
    "activate",
    "ordered_copy",
    "ordered_fill",
    "other",
];
#[cfg(feature="strata-small-quantized-matrix")]
const NATIVE_CLASSES: [&str; 8] = [
    "dequant", "matrix", "gather", "activate", "ordered_copy", "ordered_fill",
    "other", "compact_copy",
];
const MATRIX_FAMILIES: [&str; 8] = [
    "wide_gu",
    "wide_down",
    "tail_gu",
    "tail_down",
    "macro",
    "narrow",
    "cohort",
    "other_matrix",
];
const NATIVE_TOTALS: usize = NATIVE_CLASSES.len() + MATRIX_FAMILIES.len();
fn matrix_family(name: &str) -> usize {
    #[cfg(feature="strata-large-cohorts")]
    if name.contains("::strata_matrix::cohorts::large::expert_large_cohort::"){return 6;}
    #[cfg(feature="strata-small-dual-cells")]
    if name.contains("::strata_matrix::cohorts::dual_regular::")||name.contains("::strata_matrix::cohorts::dual_small::"){return 6;}
    #[cfg(feature="strata-regular-cohorts")]
    if name.contains("::strata_matrix::cohorts::regular::expert_regular_cohort::"){return 6;}
    #[cfg(feature="strata-small-granular")]
    {if name.contains("::strata_matrix::native::granular_small::"){return 4;}if name.contains("::strata_matrix::cohorts::granular_small::"){return 6;}}
    #[cfg(feature="strata-local-prefetch")]
    {if name.contains("::strata_matrix::native::local_small::"){return 4;}if name.contains("::strata_matrix::cohorts::local_small::"){return 6;}}
    #[cfg(feature="strata-balanced-global")]
    {if name.contains("::strata_matrix::native::balanced_global::"){return 4;}if name.contains("::strata_matrix::cohorts::balanced_global::"){return 6;}}
    #[cfg(feature="strata-padded-global")]
    if name.contains("::strata_matrix::native::padded_global::") {return 4;}
    #[cfg(feature="strata-small-full-walk")]
    {if name.contains("::strata_matrix::native::small_full_walk::"){return 4;}
     if name.contains("::strata_matrix::cohorts::small_full_walk::"){return 6;}}

    #[cfg(feature="strata-global-small-products")]
    {
        if name.contains("::strata_matrix::native::global_small::"){return 4;}
        if name.contains("::strata_matrix::cohorts::global_small::"){return 6;}
    }
    #[cfg(feature="strata-local-prefetch")]
    {if name.contains("::native_expert_wide::local_wide32::expert_local_wide32_gu::ExpertLocalWide32Gu<"){return 0;}if name.contains("::native_expert_wide::local_wide32::expert_local_wide32_down::ExpertLocalWide32Down<"){return 1;}}
    #[cfg(feature="strata-balanced-wide32")]
    {if name.contains("::native_expert_wide::balanced_wide32::expert_balanced_wide32_gu::ExpertBalancedWide32Gu<"){return 0;}if name.contains("::native_expert_wide::balanced_wide32::expert_balanced_wide32_down::ExpertBalancedWide32Down<"){return 1;}}
    #[cfg(feature="strata-global-wide32")]
    {if name.contains("::native_expert_wide::global_wide32::expert_global_wide32_gu::ExpertGlobalWide32Gu<"){return 0;}if name.contains("::native_expert_wide::global_wide32::expert_global_wide32_down::ExpertGlobalWide32Down<"){return 1;}}
    #[cfg(feature="strata-global-register-n64")]
    {if name.contains("::native_expert_wide::global_n64::expert_global_n64_gu::ExpertGlobalN64Gu<"){return 0;}if name.contains("::native_expert_wide::global_n64::expert_global_n64_down::ExpertGlobalN64Down<"){return 1;}}
    #[cfg(feature="strata-global-transpose")]
    {if name.contains("::native_expert_wide::global_transpose::expert_global_transpose_gu::ExpertGlobalTransposeGu<"){return 0;}if name.contains("::native_expert_wide::global_transpose::expert_global_transpose_down::ExpertGlobalTransposeDown<"){return 1;}}
    #[cfg(feature="strata-global-register-n64")]
    {if name.contains("::native_expert_wide::global_n64::expert_global_n64_gu::ExpertGlobalN64Gu<"){return 0;}if name.contains("::native_expert_wide::global_n64::expert_global_n64_down::ExpertGlobalN64Down<"){return 1;}}

    #[cfg(feature="strata-global-image-experts")]
    {
        if name.contains("::native_expert_wide::global_register::expert_global_register_gu::ExpertGlobalRegisterGu<") {return 0;}
        if name.contains("::native_expert_wide::global_register::expert_global_register_down::ExpertGlobalRegisterDown<") {return 1;}
    }

    #[cfg(feature="strata-small-quantized-matrix")]
    if (name.contains("::strata_matrix::quant_cohort::expert_quant_cohort::ExpertQuantCohort<") || name.contains("::strata_matrix::quant_cohort::expert_quant_pipeline::ExpertQuantPipeline<")) {
        return 6;
    }
    if name.contains("::expert_wide_gu::ExpertWideGu")
        || name.contains("::expert_wide_gu_direct::ExpertWideGuDirect")
    {
        0
    } else if name.contains("::expert_wide_down::ExpertWideDown") {
        1
    } else if name.contains("::expert_tail_gu::ExpertTailGu") {
        2
    } else if name.contains("::expert_tail_down::ExpertTailDown") {
        3
    } else if (name.contains("::expert_macro::ExpertMacro") || name.contains("::expert_macro64::ExpertMacro64") || name.contains("::native::ordered::expert_macro64_deep::ExpertMacro64Deep")) {
        4
    } else if name.contains("::expert_narrow::ExpertNarrow") {
        5
    } else if (name.contains("::cohorts::expert_cohort::ExpertCohort") || name.contains("::cohorts::ordered::expert_cohort::ExpertCohort") || name.contains("::cohorts::small_direct::expert_cohort::ExpertCohort")) {
        6
    } else {
        7
    }
}
#[derive(Clone, Copy)]
struct NativeSelection {
    authorized: bool,
    request: u64,
    start: usize,
    count: usize,
    layer: usize,
}
impl NativeSelection {
    fn selects(self) -> bool {
        self.authorized
            && self.request == 3
            && self.start == 0
            && self.count == 4096
            && self.layer == 24
    }
}
fn native_class(name: &str) -> usize {
    #[cfg(feature="strata-small-dual-cells")]
    if name.contains("::strata_matrix::cohorts::dual_regular::")||name.contains("::strata_matrix::cohorts::dual_small::"){return 1;}
    #[cfg(feature="strata-large-cohorts")]
    {
        if name.contains("::strata_matrix::cohorts::large::expert_large_cohort::"){return 1;}
        match name {
            "StrataExpertLargeCohortGuDirect8Image"|"StrataExpertLargeCohortDownDirect8Image"=>return 0,
            "StrataExpertLargeCohortGather"=>return 2,
            #[cfg(feature="strata-layer-gather")]
            "StrataExpertLayerGather"=>return 2,
            "StrataExpertLargeCohortActivate"=>return 3,
            _=>{}
        }
    }
    #[cfg(feature="strata-regular-cohorts")]
    {
        if name.contains("::strata_matrix::cohorts::regular::expert_regular_cohort::"){return 1;}
        match name {
            "StrataExpertRegularCohortGuDirect8Image"|"StrataExpertRegularCohortDownDirect8Image"=>return 0,
            "StrataExpertRegularCohortGather"=>return 2,
            "StrataExpertRegularCohortActivate"=>return 3,
            _=>{}
        }
    }
    #[cfg(feature="strata-small-granular")]
    if name.contains("::strata_matrix::native::granular_small::")||name.contains("::strata_matrix::cohorts::granular_small::"){return 1;}
    #[cfg(feature="strata-local-prefetch")]
    if name.contains("::strata_matrix::native::local_small::")||name.contains("::strata_matrix::cohorts::local_small::"){return 1;}
    #[cfg(feature="strata-balanced-global")]
    if name.contains("::strata_matrix::native::balanced_global::")||name.contains("::strata_matrix::cohorts::balanced_global::"){return 1;}
    #[cfg(feature="strata-padded-global")]
    let name=match name {"StrataPromptExpertGatherPad32"=>"StrataPromptExpertGather","StrataPromptExpertF16SwigluPad32"=>"StrataPromptExpertF16Swiglu","StrataPromptExpertF16SwigluCheckGuPad32"=>"StrataPromptExpertF16SwigluCheckGu",_=>name};
    #[cfg(feature="strata-padded-global")]
    if name.contains("::strata_matrix::native::padded_global::") {return 1;}
    #[cfg(feature="strata-small-full-walk")]
    if name.contains("::strata_matrix::native::small_full_walk::") || name.contains("::strata_matrix::cohorts::small_full_walk::"){return 1;}

    #[cfg(feature="strata-global-small-products")]
    if name.contains("::strata_matrix::native::global_small::") || name.contains("::strata_matrix::cohorts::global_small::"){return 1;}
    #[cfg(feature="strata-local-prefetch")]
    if name.contains("::native_expert_wide::local_wide32::expert_local_wide32_gu::ExpertLocalWide32Gu<") || name.contains("::native_expert_wide::local_wide32::expert_local_wide32_down::ExpertLocalWide32Down<") || name.contains("::native_wide::local_wide32::dense_local_wide32_store::DenseLocalWide32Store<"){return 1;}
    #[cfg(feature="strata-balanced-wide32")]
    if name.contains("::native_expert_wide::balanced_wide32::expert_balanced_wide32_gu::ExpertBalancedWide32Gu<") || name.contains("::native_expert_wide::balanced_wide32::expert_balanced_wide32_down::ExpertBalancedWide32Down<") || name.contains("::native_wide::balanced_wide32::dense_balanced_wide32_store::DenseBalancedWide32Store<"){return 1;}
    #[cfg(feature="strata-global-wide32")]
    if name.contains("::native_expert_wide::global_wide32::expert_global_wide32_gu::ExpertGlobalWide32Gu<") || name.contains("::native_expert_wide::global_wide32::expert_global_wide32_down::ExpertGlobalWide32Down<") || name.contains("::native_wide::global_wide32::dense_global_wide32_store::DenseGlobalWide32Store<"){return 1;}
    #[cfg(feature="strata-global-register-n64")]
    if name.contains("::native_expert_wide::global_n64::expert_global_n64_gu::ExpertGlobalN64Gu<") || name.contains("::native_expert_wide::global_n64::expert_global_n64_down::ExpertGlobalN64Down<") || name.contains("::native_wide::global_n64::dense_global_n64_store::DenseGlobalN64Store<"){return 1;}
    #[cfg(feature="strata-global-transpose")]
    if name.contains("::native_expert_wide::global_transpose::expert_global_transpose_gu::ExpertGlobalTransposeGu<") || name.contains("::native_expert_wide::global_transpose::expert_global_transpose_down::ExpertGlobalTransposeDown<") || name.contains("::native_wide::global_transpose::dense_global_transpose_store::DenseGlobalTransposeStore<"){return 1;}
    #[cfg(feature="strata-global-register-n64")]
    if name.contains("::native_expert_wide::global_n64::expert_global_n64_gu::ExpertGlobalN64Gu<") || name.contains("::native_expert_wide::global_n64::expert_global_n64_down::ExpertGlobalN64Down<") || name.contains("::native_wide::global_n64::dense_global_n64_store::DenseGlobalN64Store<"){return 1;}

    #[cfg(feature="strata-global-image-experts")]
    if name.contains("::native_expert_wide::global_register::expert_global_register_gu::ExpertGlobalRegisterGu<")
        || name.contains("::native_expert_wide::global_register::expert_global_register_down::ExpertGlobalRegisterDown<") {return 1;}

    #[cfg(feature="strata-small-quantized-matrix")]
    {
        if matches!(name,"StrataExpertCohortGuCompact"|"StrataExpertCohortDownCompact") {
            return 7;
        }
        if (name.contains("::strata_matrix::quant_cohort::expert_quant_cohort::ExpertQuantCohort<") || name.contains("::strata_matrix::quant_cohort::expert_quant_pipeline::ExpertQuantPipeline<")) {
            return 1;
        }
    }
    match name {
        "StrataPromptExpertF16Image"
        | "StrataPromptExpertResidentGuImage"
        | "StrataPromptExpertResidentDownImage"
        | "StrataExpertCohortGuImage"
        | "StrataExpertCohortDownImage"
        | "StrataExpertCohortGuQuad32Image"
        | "StrataExpertCohortDownQuad32Image"
        | "StrataPromptExpertF16Direct8Image"
        | "StrataPromptExpertResidentGuDirect8Image"
        | "StrataPromptExpertResidentDownDirect8Image"
        | "StrataExpertCohortGuDirect8Image"
        | "StrataExpertCohortDownDirect8Image" => 0,
        "StrataPromptExpertGemm" => 1,
        "StrataPromptExpertGather" | "StrataExpertCohortGather" => 2,
        "StrataPromptExpertF16Swiglu"
        | "StrataPromptExpertF16SwigluCheckGu"
        | "StrataExpertCohortActivate" => 3,
        "OrderedCopy" => 4,
        "OrderedFill" => 5,
        _ if name.contains("::expert_wide_gu::ExpertWideGu")
            || name.contains("::expert_wide_gu_direct::ExpertWideGuDirect")
            || name.contains("::expert_wide_down::ExpertWideDown")
            || name.contains("::expert_tail_gu::ExpertTailGu")
            || name.contains("::expert_tail_down::ExpertTailDown")
            || name.contains("::cohorts::expert_cohort::ExpertCohort")
            || name.contains("::cohorts::ordered::expert_cohort::ExpertCohort")
            || name.contains("::cohorts::small_direct::expert_cohort::ExpertCohort")
            || name.contains("::expert::Expert")
            || (name.contains("::expert_macro::ExpertMacro") || name.contains("::expert_macro64::ExpertMacro64") || name.contains("::native::ordered::expert_macro64_deep::ExpertMacro64Deep"))
            || name.contains("::expert_narrow::ExpertNarrow")
            || name.contains("::native_wide::dense_asymmetric_store::DenseAsymmetricStore<")
            || name.contains("::native_wide::dense_global_register_store::DenseGlobalRegisterStore<")
            || name.contains("::library_dense::LibraryDense")
            || name.contains("::library_expert::LibraryExpert") =>
        {
            1
        }
        _ => 6,
    }
}
#[cfg(test)]
mod native_sample_tests {
    use super::*;
    #[cfg(feature = "strata-wide-shared-reuse")]
    #[test]
    fn genuine_shared_reuse_prefixes_keep_original_operator_and_matrix_family_classes() {
        for (name,family) in [
            ("x::strata_matrix::native_expert_wide::shared_reuse::expert_wide_gu::ExpertWideGu<T>",0),
            ("x::strata_matrix::native_expert_wide::shared_reuse::expert_wide_gu_direct::ExpertWideGuDirect<T>",0),
            ("x::strata_matrix::native_expert_wide::shared_reuse::expert_wide_down::ExpertWideDown<T>",1),
            ("x::strata_matrix::native_expert_wide::ordered::shared_reuse::expert_wide_down::ExpertWideDown<T>",1),
        ] {
            assert_eq!(native_class(name),1);
            assert_eq!(matrix_family(name),family);
        }
    }
    #[test]
    fn disabled_gr_sampling_preserves_scope_without_enabling_public_timing() {
        assert!(!ENABLED.load(Ordering::Acquire));
        let previous = GR_SAMPLE.with(|v| v.replace(false));
        let sample = GrSample::new(0, 4096, 24);
        assert!(sample.0.is_none());
        assert!(!GR_SAMPLE.with(Cell::get));
        drop(sample);
        assert!(!GR_SAMPLE.with(Cell::get));
        GR_SAMPLE.with(|v| v.set(previous));
    }
    #[test]
    fn actual_generated_wide_and_tail_names_are_matrix_not_other() {
        #[cfg(all(feature = "strata-intel-core", feature = "strata-native-expert-wide"))]
        for (family, name) in [0, 1, 2, 3, 0]
            .into_iter()
            .zip(crate::strata_matrix::wide_expert_profile_names())
        {
            assert_eq!(native_class(name), 1);
            assert_eq!(matrix_family(name), family);
        }
        for (name, family) in [
            ("x::expert_macro::ExpertMacro<T>", 4),
            ("x::expert_narrow::ExpertNarrow<T>", 5),
            ("x::cohorts::expert_cohort::ExpertCohort<T>", 6),
            ("StrataPromptExpertGemm", 7),
        ] {
            assert_eq!(native_class(name), 1);
            assert_eq!(matrix_family(name), family);
        }
        assert_eq!(native_class("unknown private or dynamic label"), 6);
    }
    #[test]
    fn cohort_real_program_names_retain_public_operator_classes() {
        for (name, class) in [
            ("StrataExpertCohortGather", 2),
            ("StrataExpertCohortGuImage", 0),
            ("StrataExpertCohortDownImage", 0),
            ("StrataExpertCohortActivate", 3),
            (
                "x::strata_matrix::cohorts::expert_cohort::ExpertCohort<WgpuRuntime>",
                1,
            ),
        ] {
            assert_eq!(native_class(name), class);
        }
    }
    #[cfg(feature = "strata-expert-ordered-output")]
    #[test]
    fn new_ordered_and_quad_entries_keep_their_actual_operator_classes() {
        for (name, family) in [
            ("x::native_expert_wide::ordered::expert_wide_down::ExpertWideDown<T>", 1),
            ("x::native_expert_wide::ordered::expert_tail_down::ExpertTailDown<T>", 3),
            ("x::native::ordered::expert_macro64_deep::ExpertMacro64Deep<T>", 4),
            ("x::cohorts::ordered::expert_cohort::ExpertCohort<T>", 6),
        ] {
            assert_eq!(native_class(name), 1);
            assert_eq!(matrix_family(name), family);
        }
        for name in ["StrataExpertCohortGuQuad32Image", "StrataExpertCohortDownQuad32Image"] {
            assert_eq!(native_class(name), 0);
        }
    }
    #[test]
    fn exactly_one_compiled_public_expert_interval_can_be_selected() {
        for authorized in [false, true] {
            for request in 0..5 {
                for start in [0, 4096] {
                    for count in [35, 149, 4095, 4096] {
                        for layer in 0..48 {
                            assert_eq!(
                                NativeSelection {
                                    authorized,
                                    request,
                                    start,
                                    count,
                                    layer
                                }
                                .selects(),
                                authorized
                                    && request == 3
                                    && start == 0
                                    && count == 4096
                                    && layer == 24
                            );
                        }
                    }
                }
            }
        }
    }
    #[test]
    fn reports_have_static_classes_including_real_generated_matrix_names() {
        for (name, class) in [
            ("StrataPromptExpertF16Image", 0),
            ("StrataPromptExpertResidentGuImage", 0),
            ("StrataPromptExpertResidentDownImage", 0),
            ("StrataPromptExpertGemm", 1),
            ("kuno_native_text::strata_matrix::native::expert::Expert", 1),
            (
                "kuno_native_text::strata_matrix::native::expert_macro::ExpertMacro",
                1,
            ),
            (
                "kuno_native_text::strata_matrix::native::expert_narrow::ExpertNarrow",
                1,
            ),
            (
                "kuno_native_text::strata_matrix::cubek::library_dense::LibraryDense",
                1,
            ),
            (
                "kuno_native_text::strata_matrix::cubek::routed_expert::library_expert::LibraryExpert",
                1,
            ),
            ("StrataPromptExpertGather", 2),
            ("StrataPromptExpertF16Swiglu", 3),
            ("StrataPromptExpertF16SwigluCheckGu", 3),
            ("OrderedCopy", 4),
            ("OrderedFill", 5),
            ("UnlabelledNativeOperator", 6),
        ] {
            assert_eq!(native_class(name), class);
            assert_eq!(
                NATIVE_CLASSES[class]
                    .as_bytes()
                    .iter()
                    .filter(|&&b| b == b':')
                    .count(),
                0
            );
        }
    }
}
// PUBLIC_PREFILL_NATIVE_SAMPLE_END

fn native_totals() -> [(u64, u64); NATIVE_TOTALS] {
    let mut classes = [(0u64, 0u64); NATIVE_TOTALS];
    for (name, calls, ns) in cubecl::wgpu::native_strata_operator_profile() {
        let class = native_class(&name);
        let total = &mut classes[class];
        total.0 = total.0.saturating_add(calls);
        total.1 = total.1.saturating_add(ns);
        if class == 1 {
            let family = &mut classes[NATIVE_CLASSES.len() + matrix_family(&name)];
            family.0 = family.0.saturating_add(calls);
            family.1 = family.1.saturating_add(ns);
        }
    }
    classes
}
/// One fixed-public expert interval only. Keep this guard alive through the
/// existing coarse Timer's real completion fence, not just enqueue_prompt().
/// Drop on error disables sampling without publishing incomplete timestamps.
pub(crate) struct NativeExpertLayer {
    before: Option<[(u64, u64); NATIVE_TOTALS]>,
    location: Option<(usize,usize,usize)>,
    #[cfg(feature = "strata-public-host-clock")]
    host_before: [[u64; 4]; 8],
    #[cfg(feature = "strata-public-host-clock")]
    batch_before: [[u64; 4]; 6],
    #[cfg(feature = "strata-copy-cohorts")]
    compact_before: [u64; 2],
    #[cfg(feature = "strata-expert-cohorts")]
    cohort_before: [u64; 25+cfg!(feature="strata-layer-gather") as usize],
}
impl NativeExpertLayer {
    pub(crate) fn new(start: usize, count: usize, layer: usize) -> Self {
        if !ENABLED.load(Ordering::Acquire) {
            return Self {
                before: None,
                location: None,
                #[cfg(feature = "strata-public-host-clock")]
                host_before: [[0; 4]; 8],
                #[cfg(feature = "strata-public-host-clock")]
                batch_before: [[0; 4]; 6],
                #[cfg(feature = "strata-copy-cohorts")]
                compact_before: [0; 2],
                #[cfg(feature = "strata-expert-cohorts")]
                cohort_before: [0; 25+cfg!(feature="strata-layer-gather") as usize],
            };
        }
        let selected = NativeSelection {
            authorized: std::env::var("KUNO_LAB_LANGUAGE_PUBLIC_ONLY").as_deref() == Ok("1")
                && (std::env::var("KUNO_STRATA_NATIVE_PROFILE").as_deref() == Ok("prefill-layer")
                    || (cfg!(feature = "strata-public-host-clock")
                        && std::env::var("KUNO_STRATA_NATIVE_PROFILE").as_deref()
                            == Ok("prefill-host-layer"))),
            request: REQUEST.load(Ordering::Relaxed),
            start,
            count,
            layer,
        }
        .selects() || full_host_selected(
            FULL_HOST_TRACE.load(Ordering::Acquire),
            std::env::var("KUNO_LAB_LANGUAGE_PUBLIC_ONLY").as_deref()==Ok("1"),
            std::env::var("KUNO_STRATA_NATIVE_PROFILE").as_deref()==Ok("prefill-host-layer"),
            REQUEST.load(Ordering::Relaxed),start,count,layer);
        if !selected {
            return Self {
                before: None,
                location: None,
                #[cfg(feature = "strata-public-host-clock")]
                host_before: [[0; 4]; 8],
                #[cfg(feature = "strata-public-host-clock")]
                batch_before: [[0; 4]; 6],
                #[cfg(feature = "strata-copy-cohorts")]
                compact_before: [0; 2],
                #[cfg(feature = "strata-expert-cohorts")]
                cohort_before: [0; 25+cfg!(feature="strata-layer-gather") as usize],
            };
        }
        let before = native_totals();
        cubecl::wgpu::set_native_strata_profile_active(true);
        Self {
            before: Some(before),
            location: Some((start,count,layer)),
            #[cfg(feature = "strata-public-host-clock")]
            host_before: cubecl::wgpu::native_strata_host_capture_profile(),
            #[cfg(feature = "strata-public-host-clock")]
            batch_before: cubecl::wgpu::native_strata_host_batch_profile(),
            #[cfg(feature = "strata-copy-cohorts")]
            compact_before: cubecl::wgpu::native_strata_segment_compact_profile(),
            #[cfg(feature = "strata-expert-cohorts")]
            cohort_before: cubecl::wgpu::native_strata_cohort_selection_stats(),
        }
    }
    pub(crate) fn finish(mut self) {
        if let Some(before) = self.before.take() {
            cubecl::wgpu::set_native_strata_profile_active(false);
            let after = native_totals();
            let classes: BTreeMap<_, _> = NATIVE_CLASSES
                .into_iter()
                .enumerate()
                .map(|(i, name)| {
                    (
                        name,
                        [
                            after[i].0.saturating_sub(before[i].0),
                            after[i].1.saturating_sub(before[i].1),
                        ],
                    )
                })
                .collect();
            let timestamp_calls = classes.values().map(|v| v[0]).sum::<u64>();
            // Counts/ns only, never source labels, experts, routes or tokens.
            // Zero timestamp_calls honestly means the existing graph threshold
            // did not sample this interval; no cap or fence is relaxed.
            #[allow(unused_mut)]
            let (start,count,layer)=self.location.take().expect("selected public location");
            let mut record = serde_json::json!({"sampled_layers":1,"request_index":3,"start":start,"count":count,"layer":layer,"timestamp_calls":timestamp_calls,"classes":classes});
            // Static family identifiers only, from the same completed fixed
            // public samples. No raw program name or request label is emitted.
            let matrix_families: BTreeMap<_, _> = MATRIX_FAMILIES
                .into_iter()
                .enumerate()
                .map(|(j, name)| {
                    let i = NATIVE_CLASSES.len() + j;
                    (
                        name,
                        [
                            after[i].0.saturating_sub(before[i].0),
                            after[i].1.saturating_sub(before[i].1),
                        ],
                    )
                })
                .collect();
            record["matrix_families"] = serde_json::json!(matrix_families);
            #[cfg(feature = "strata-public-host-clock")]
            {
                let after = cubecl::wgpu::native_strata_host_capture_profile();
                let host: BTreeMap<_, _> = cubecl::wgpu::NATIVE_CAPTURE_CPU_LABELS
                    .into_iter()
                    .enumerate()
                    .map(|(i, label)| {
                        (
                            label,
                            std::array::from_fn::<_, 4, _>(|j| {
                                after[i][j].saturating_sub(self.host_before[i][j])
                            }),
                        )
                    })
                    .collect();
                record["host_capture_columns"] = serde_json::json!([
                    "calls",
                    "wall_ns",
                    "thread_cpu_ns",
                    "cpu_clock_unavailable"
                ]);
                record["host_capture"] = serde_json::json!(host);
                let batch_after = cubecl::wgpu::native_strata_host_batch_profile();
                let batch: BTreeMap<_, _> = cubecl::wgpu::NATIVE_BATCH_CPU_LABELS
                    .into_iter()
                    .enumerate()
                    .map(|(i, label)| {
                        (
                            label,
                            std::array::from_fn::<_, 4, _>(|j| {
                                batch_after[i][j].saturating_sub(self.batch_before[i][j])
                            }),
                        )
                    })
                    .collect();
                record["host_batch"] = serde_json::json!(batch);
                record["host_only_normal_schedule"] = serde_json::json!(
                    std::env::var("KUNO_STRATA_NATIVE_PROFILE").as_deref()
                        == Ok("prefill-host-layer")
                );
            }
            #[cfg(feature = "strata-copy-cohorts")]
            {
                let after = cubecl::wgpu::native_strata_segment_compact_profile();
                record["segment_compaction_original_retained"] =
                    serde_json::json!(std::array::from_fn::<_, 2, _>(
                        |i| after[i].saturating_sub(self.compact_before[i])
                    ));
            }
            #[cfg(feature = "strata-expert-cohorts")]
            {
                let after = cubecl::wgpu::native_strata_cohort_selection_stats();
                let counters: BTreeMap<_, _> = cubecl::wgpu::NATIVE_COHORT_SELECTION_LABELS
                    .into_iter()
                    .enumerate()
                    .map(|(i, name)| (name, after[i].saturating_sub(self.cohort_before[i])))
                    .collect();
                record["cohort_selection"] = serde_json::json!(counters);
            }
            eprintln!("public_prefill_native_gpu_profile {}", record);
        }
    }
}
impl Drop for NativeExpertLayer {
    fn drop(&mut self) {
        if self.before.is_some() {
            cubecl::wgpu::set_native_strata_profile_active(false);
        }
    }
}
pub(crate) fn report() {
    if ENABLED.load(Ordering::Acquire) {
        TOTALS.with(|v| {
            let mut report = v.borrow().clone();
            let process = cubecl::wgpu::native_strata_graph_compute_barrier_stats();
            for (name, index) in [
                ("public_worker_total_idle_pool_retirements", 28),
                ("public_worker_total_idle_pool_adoptions", 29),
                ("public_worker_total_original_recording_retirements", 30),
            ] {
                report.insert(name, (process[index], 0, 0));
            }
            #[cfg(feature = "strata-qsa-xmx")]
            report.insert(
                "public_worker_total_qsa_xmx_launches",
                (crate::strata_qsa::prompt_xmx::launch_count(), 0, 0),
            );
            eprintln!(
                "public_prefill_profile {}",
                serde_json::to_string(&report).unwrap()
            )
        });
    }
}
/// Aggregate public-fixture resource counts only. Private inference never
/// enables this collector; source identities, offsets and route values are
/// deliberately absent. Counts use tuple slot0, with both time slots zero.
pub(crate) fn count(name: &'static str, amount: u64) {
    if ENABLED.load(Ordering::Acquire) {
        TOTALS.with(|totals| {
            let mut totals = totals.borrow_mut();
            let total = totals.entry(name).or_default();
            total.0 = total.0.saturating_add(amount);
        });
    }
}

/// Public-fixture aggregate geometry only, after the caller validates the map.
/// Never retain expert identities or enable this collector for private calls.
pub(crate) fn expert_member_geometry(map: &[u32], groups: usize) {
    if !ENABLED.load(Ordering::Acquire) {
        return;
    }
    let names = [
        ("expert_groups_rows_1_7", "expert_members_rows_1_7"),
        ("expert_groups_rows_8_15", "expert_members_rows_8_15"),
        ("expert_groups_rows_16_31", "expert_members_rows_16_31"),
        ("expert_groups_rows_32_63", "expert_members_rows_32_63"),
        ("expert_groups_rows_64_127", "expert_members_rows_64_127"),
        ("expert_groups_rows_128_255", "expert_members_rows_128_255"),
        (
            "expert_groups_rows_256_plus",
            "expert_members_rows_256_plus",
        ),
    ];
    let mut group_counts = [0u64; 7];
    let mut member_counts = [0u64; 7];
    let mut padded_members = 0u64;
    for &members in &map[516..516 + groups] {
        let bin = match members {
            0..=7 => 0,
            8..=15 => 1,
            16..=31 => 2,
            32..=63 => 3,
            64..=127 => 4,
            128..=255 => 5,
            _ => 6,
        };
        group_counts[bin] += 1;
        member_counts[bin] += u64::from(members);
        let tile = match members {
            0..=8 => 8,
            9..=31 => 16,
            _ => 32,
        };
        padded_members += u64::from(members.div_ceil(tile) * tile);
    }
    for (i, (group_name, member_name)) in names.into_iter().enumerate() {
        count(group_name, group_counts[i]);
        count(member_name, member_counts[i]);
    }
    // Actual fixed expert geometry: GU (2560,1280), DOWN (640,2560).
    // This is an arithmetic-work estimate, never a measured hardware FLOP rate.
    count(
        "expert_useful_matrix_flops",
        member_counts.iter().sum::<u64>() * 9_830_400,
    );
    count(
        "expert_tile_padded_matrix_flops",
        padded_members * 9_830_400,
    );
}
fn native_backend_counts() -> [u64; 38] {
    let (_, _, capture, wait) = cubecl::wgpu::native_strata_graph_execution_stats();
    let rolling = cubecl::wgpu::native_strata_graph_rolling_stats();
    let copies = cubecl::wgpu::native_strata_graph_copy_barrier_stats();
    let compute = cubecl::wgpu::native_strata_graph_compute_barrier_stats();
    [
        capture,
        wait,
        rolling[0],
        rolling[1],
        rolling[2],
        copies[0],
        copies[1],
        compute[0],
        compute[1],
        compute[2],
        compute[3],
        compute[4],
        compute[5],
        compute[6],
        compute[7],
        compute[8],
        compute[9],
        compute[10],
        compute[11],
        compute[12],
        compute[13],
        compute[14],
        compute[15],
        compute[16],
        compute[17],
        compute[18],
        compute[19],
        compute[20],
        compute[21],
        compute[22],
        compute[23],
        compute[24],
        compute[25],
        compute[26],
        compute[27],
        compute[28],
        compute[29],
        compute[30],
    ]
}
pub(crate) struct Timer(Option<(ComputeClient<WgpuRuntime>, Instant, [u64; 38])>);
impl Timer {
    pub(crate) fn for_gr_sample(client: &ComputeClient<WgpuRuntime>) -> kuno_text::Result<Self> {
        if !ENABLED.load(Ordering::Acquire) || !GR_SAMPLE.with(Cell::get) {
            return Ok(Self(None));
        }
        Self::new(client)
    }
    // PUBLIC_QSA_TIMER_ADMISSION_BEGIN
    /// No private client lookup, enqueue, flush or synchronization is added.
    /// Only the existing compiled-public collector may create this sub-timer.
    pub(crate) fn for_public_arena(
        arena: &crate::private_arena::PrivateArena,
    ) -> kuno_text::Result<Self> {
        if !ENABLED.load(Ordering::Acquire) {
            return Ok(Self(None));
        }
        Self::new(arena.client().map_err(crate::resident_util::fault)?)
    }
    // PUBLIC_QSA_TIMER_ADMISSION_END
    pub(crate) fn new(client: &ComputeClient<WgpuRuntime>) -> kuno_text::Result<Self> {
        if !ENABLED.load(Ordering::Acquire) {
            return Ok(Self(None));
        }
        cubecl::future::block_on(client.sync()).map_err(crate::resident_util::fault)?;
        Ok(Self(Some((
            client.clone(),
            Instant::now(),
            native_backend_counts(),
        ))))
    }
    pub(crate) fn mark(&mut self, name: &'static str) -> kuno_text::Result<()> {
        if let Some((client, start, native_before)) = &mut self.0 {
            let enqueue_ns = start.elapsed().as_nanos().min(u64::MAX as u128) as u64;
            cubecl::future::block_on(client.sync()).map_err(crate::resident_util::fault)?;
            let ns = start.elapsed().as_nanos().min(u64::MAX as u128) as u64;
            let native_after = native_backend_counts();
            TOTALS.with(|v| {
                let mut v = v.borrow_mut();
                let t = v.entry(name).or_default();
                t.0 += 1;
                t.1 = t.1.saturating_add(ns);
                t.2 = t.2.saturating_add(enqueue_ns);
                if name == "experts" {
                    // The enqueue slice ends BEFORE sync records native Vulkan
                    // descriptors/commands. Report that host elapsed time separately;
                    // Legacy *_cpu labels are Instant wall deltas, not thread CPU or GPU time. The public
                    // protected unit is the sole process/device workload.
                    for (index, name) in ["expert_native_capture_cpu", "expert_native_wait_cpu"]
                        .into_iter()
                        .enumerate()
                    {
                        let delta = native_after[index].saturating_sub(native_before[index]);
                        let total = v.entry(name).or_default();
                        total.0 += 1;
                        total.1 = total.1.saturating_add(delta);
                        total.2 = total.2.saturating_add(delta);
                    }
                    // Counts only after this interval's original real sync.
                    // These prove actual rolling issuance rather than selector
                    // intent; private calls never enable this collector.
                    for (index, name) in [
                        "expert_native_rolling_recordings",
                        "expert_native_rolling_windows",
                        "expert_native_rolling_segments",
                        "expert_native_copy_nodes_encoded",
                        "expert_native_copy_barriers_encoded",
                        "expert_native_compute_nodes_encoded",
                        "expert_native_compute_legacy_barriers",
                        "expert_native_compute_barriers_encoded",
                        "expert_native_compute_atomic_or_views",
                        "expert_native_compute_legacy_calls",
                        "expert_native_compute_calls_encoded",
                        "expert_native_compute_manifest_entries",
                        "expert_native_compute_execution_only_entries",
                        "expert_native_push_descriptor_dispatches",
                        "expert_native_fresh_descriptor_dispatches",
                        "expert_native_capture_cached_nodes",
                        "expert_native_capture_cached_views",
                        "expert_native_capture_root_revalidations",
                        "expert_native_canonical_source_lists",
                        "expert_native_readonly_full_roots",
                        "expert_native_factored_source_reads",
                        "expert_native_shared_read_nodes",
                        "expert_native_shared_read_bundles",
                        "expert_native_unscoped_full_capture_admissions",
                        "expert_native_protected_full_range_admissions",
                        "expert_native_push_template_dispatches",
                        "expert_native_reencoded_recordings",
                        "expert_native_reencoded_commands",
                        "expert_native_capacity_reencoded_recordings",
                        "expert_native_spare_commands_retained",
                        "expert_native_reencoding_workload_captures",
                        "expert_native_reencoding_fresh_captures",
                        "expert_native_reencoding_idle_candidates",
                        "expert_native_idle_pools_retired",
                        "expert_native_idle_pools_adopted",
                        "expert_native_original_recordings_retired",
                    ]
                    .into_iter()
                    .enumerate()
                    {
                        let delta =
                            native_after[index + 2].saturating_sub(native_before[index + 2]);
                        let total = v.entry(name).or_default();
                        total.0 = total.0.saturating_add(delta);
                    }
                }
            });
            *native_before = native_after;
            *start = Instant::now();
        }
        Ok(())
    }
}

/// Public-only boundary wall/thread-CPU pair. No fences or payload samples.
/// New boundary labels use [calls,wall_ns,thread_cpu_ns]; unavailable clocks
/// add an explicit anonymous counter rather than masquerading as zero CPU.
pub(crate) struct BoundaryTimer(Option<(Instant, Option<u64>)>);
fn thread_cpu_ns() -> Option<u64> {
    let mut ts = libc::timespec {
        tv_sec: 0,
        tv_nsec: 0,
    };
    // SAFETY: only this thread's clock is read, through an owned timespec.
    if unsafe { libc::clock_gettime(libc::CLOCK_THREAD_CPUTIME_ID, &mut ts) } != 0
        || ts.tv_sec < 0
        || !(0..1_000_000_000).contains(&ts.tv_nsec)
    {
        return None;
    }
    (ts.tv_sec as u64)
        .checked_mul(1_000_000_000)?
        .checked_add(ts.tv_nsec as u64)
}
impl BoundaryTimer {
    pub(crate) fn new() -> Self {
        Self(
            ENABLED
                .load(Ordering::Acquire)
                .then(|| (Instant::now(), thread_cpu_ns())),
        )
    }
    pub(crate) fn mark(&mut self, name: &'static str) {
        if let Some((start, cpu)) = &mut self.0 {
            let now = Instant::now();
            let after = thread_cpu_ns();
            let wall = now.duration_since(*start).as_nanos().min(u64::MAX as u128) as u64;
            let thread = cpu.and_then(|before| after.and_then(|a| a.checked_sub(before)));
            TOTALS.with(|v| {
                let mut v = v.borrow_mut();
                let t = v.entry(name).or_default();
                t.0 += 1;
                t.1 = t.1.saturating_add(wall);
                t.2 = t.2.saturating_add(thread.unwrap_or(0));
                if thread.is_none() {
                    v.entry("expert_boundary_cpu_clock_unavailable")
                        .or_default()
                        .0 += 1;
                }
            });
            *start = now;
            *cpu = after;
        }
    }
}

/// CPU-side scheduling slices; no additional device fence or changed overlap.
pub(crate) struct CpuTimer(Option<Instant>);
impl CpuTimer {
    pub(crate) fn new() -> Self {
        Self(ENABLED.load(Ordering::Acquire).then(Instant::now))
    }
    pub(crate) fn mark(&mut self, name: &'static str) {
        if let Some(start) = &mut self.0 {
            let ns = start.elapsed().as_nanos().min(u64::MAX as u128) as u64;
            TOTALS.with(|v| {
                let mut v = v.borrow_mut();
                let t = v.entry(name).or_default();
                t.0 += 1;
                t.1 = t.1.saturating_add(ns);
                t.2 = t.2.saturating_add(ns);
            });
            *start = Instant::now();
        }
    }
}

#[cfg(all(test,feature="strata-dequant-direct8"))]
mod direct8_profile_tests {
    use super::*;
    #[test]
    fn genuine_direct8_names_keep_same_dequant_profile_class() {
        for name in ["StrataPromptExpertF16Direct8Image","StrataPromptExpertResidentGuDirect8Image","StrataPromptExpertResidentDownDirect8Image","StrataExpertCohortGuDirect8Image","StrataExpertCohortDownDirect8Image"] {assert_eq!(native_class(name),0);}
        for name in ["StrataPromptExpertF16Direct8ImageFake","StrataExpertCohortGuDirect8Image::nested"] {assert_eq!(native_class(name),6);}
    }
}

#[cfg(all(test,feature="strata-small-direct-publish"))]
mod small_direct_profile_tests {
    use super::*;
    #[test]
    fn genuine_small_direct_producers_keep_matrix_family_and_phase_profiles(){
        for (name,family) in [
            ("probe::strata_matrix::native::small_direct::expert_macro::ExpertMacro<Runtime>",4),
            ("probe::strata_matrix::native::small_direct::expert_macro64::ExpertMacro64<Runtime>",4),
            ("probe::strata_matrix::native::small_direct::expert_narrow::ExpertNarrow<Runtime>",5),
            ("probe::strata_matrix::native::small_direct::expert_tail_gu::ExpertTailGu<Runtime>",2),
            ("probe::strata_matrix::native::small_direct::expert_tail_down::ExpertTailDown<Runtime>",3),
            ("probe::strata_matrix::cohorts::small_direct::expert_cohort::ExpertCohort<Runtime>",6),
        ]{assert_eq!(native_class(name),1);assert_eq!(matrix_family(name),family);}
    }
}

#[cfg(all(test,feature="strata-small-quantized-matrix"))]
mod small_quantized_profile_tests {
    use super::*;
    #[test]
    fn compact_copy_is_distinct_from_dequant_and_quantized_products_are_matrices() {
        for name in ["StrataExpertCohortGuCompact","StrataExpertCohortDownCompact"] {
            assert_eq!(native_class(name),7);
            assert_eq!(NATIVE_CLASSES[native_class(name)],"compact_copy");
        }
        let name="probe::strata_matrix::quant_cohort::expert_quant_cohort::ExpertQuantCohort<Runtime>";
        assert_eq!(native_class(name),1);
        assert_eq!(matrix_family(name),6);
        for name in ["StrataExpertCohortGuCompactFake","StrataExpertCohortDownCompact::nested",
            "probe::strata_matrix::quant_cohort::expert_quant_cohort::ExpertQuantCohortFake<Runtime>"] {
            assert_eq!(native_class(name),6);
        }
        assert_eq!(native_class("StrataExpertCohortGuDirect8Image"),0);
        assert_eq!(native_class("OrderedCopy"),4);
    }
}

#[cfg(all(test,feature="strata-global-image-experts"))]
mod global_image_expert_profile_tests {
 use super::*;
 #[test]
 fn actual_global_image_entries_are_matrices_in_original_gu_down_families(){
  let names=crate::strata_matrix::global_image_profile_names();
  for(i,n)in names.into_iter().enumerate(){assert_eq!(native_class(n),1);assert_eq!(matrix_family(n),i);}
 }
}

#[cfg(all(test,feature="strata-global-register-n64"))]
mod global_n64_profile_tests{
 use super::*;
 #[test]
 fn actual_n64_generated_producers_keep_original_matrix_and_expert_family_classes(){
  for(i,n)in crate::strata_matrix::global_n64_profile_names().into_iter().enumerate(){assert_eq!(native_class(n),1);if i<2{assert_eq!(matrix_family(n),i);}}
 }
}

#[cfg(all(test,feature="strata-global-transpose"))]
mod global_transpose_profile_tests{
 use super::*;
 #[test]
 fn actual_transpose_generated_producers_keep_original_matrix_and_expert_family_classes(){
  for(i,n)in crate::strata_matrix::global_transpose_profile_names().into_iter().enumerate(){assert_eq!(native_class(n),1);if i<2{assert_eq!(matrix_family(n),i);}}
 }
}

#[cfg(all(test,feature="strata-global-wide32"))]
mod global_wide32_profile_tests{
 use super::*;
 #[test]
 fn actual_wide32_generated_producers_keep_original_matrix_and_expert_family_classes(){
  for(i,n)in crate::strata_matrix::global_wide32_profile_names().into_iter().enumerate(){assert_eq!(native_class(n),1);if i<2{assert_eq!(matrix_family(n),i);}}
 }
}

#[cfg(all(test,feature="strata-global-small-products"))]
mod global_small_profile_tests{
 use super::*;
 #[test]
 fn actual_global_small_generated_names_keep_original_matrix_families(){
  for(i,name)in crate::strata_matrix::global_small::profile_names().into_iter().enumerate(){assert_eq!(native_class(name),1);assert_eq!(matrix_family(name),if i<4{4}else{6});}
 }
}

#[cfg(all(test,feature="strata-small-full-walk"))]mod small_full_walk_profile_tests{
 use super::*;
 #[test]fn actual_full_walk_generated_names_keep_original_macro_and_cohort_classes(){
  for(i,n)in crate::strata_matrix::small_full_walk::profile_names().into_iter().enumerate(){assert_eq!(native_class(n),1);assert_eq!(matrix_family(n),if i==4{6}else{4});}
 }
}

#[cfg(all(test,feature="strata-padded-global"))]
mod padded_global_profile_tests {
 use super::*;
 #[test] fn actual_padded_producers_and_macro_names_retain_original_semantic_classes(){
  for n in crate::strata_matrix::padded_global::profile_names().into_iter().take(4){assert_eq!(native_class(n),1);assert_eq!(matrix_family(n),4);}
  for(a,b)in [("StrataPromptExpertGatherPad32","StrataPromptExpertGather"),("StrataPromptExpertF16SwigluCheckGuPad32","StrataPromptExpertF16SwigluCheckGu")] {assert_eq!(native_class(a),native_class(b));}
 }
}

#[cfg(all(test,feature="strata-balanced-global"))]mod balanced_global_profile_tests {
 use super::*;
 #[test]fn actual_two_C_macro_and_packed_names_keep_original_semantic_classes(){
  for(i,n)in crate::strata_matrix::balanced_global::profile_names().into_iter().enumerate(){assert_eq!(native_class(n),1);assert_eq!(matrix_family(n),if i==4{6}else{4});}
 }
}

mod full_host_trace {use super::*;include!("strata_prefill_full_host_trace.rs");}
pub(crate) use full_host_trace::FullHostLayer;
use full_host_trace::full_host_selected;

#[cfg(all(test,feature="strata-balanced-wide32"))]
mod balanced_wide32_profile_tests{
 use super::*;
 #[test]
 fn actual_wide32_generated_producers_keep_original_matrix_and_expert_family_classes(){
  for(i,n)in crate::strata_matrix::balanced_wide32_profile_names().into_iter().enumerate(){assert_eq!(native_class(n),1);if i<2{assert_eq!(matrix_family(n),i);}}
 }
}
#[cfg(all(test,feature="strata-local-prefetch"))]
pub(crate) fn local_prefetch_test_classes(name:&str)->(usize,usize){(native_class(name),matrix_family(name))}

#[cfg(all(test,feature="strata-regular-cohorts"))]
mod regular_classes {
 #[test]
 fn genuine_regular_cohort_programs_keep_honest_public_matrix_and_operator_classes(){
  let name=core::any::type_name::<crate::strata_matrix::cohorts::regular::expert_regular_cohort::ExpertRegularCohort<cubecl::wgpu::WgpuRuntime>>();
  assert_eq!(super::matrix_family(name),6);assert_eq!(super::native_class(name),1);
  for (n,c)in [("StrataExpertRegularCohortGuDirect8Image",0),("StrataExpertRegularCohortDownDirect8Image",0),("StrataExpertRegularCohortGather",2),("StrataExpertRegularCohortActivate",3)]{assert_eq!(super::native_class(n),c);}
 }
}

#[cfg(all(test,feature="strata-small-dual-cells"))]
mod dual_profile_tests {
 use super::*;
 #[test]
 fn genuine_dual_cohorts_have_matrix_class_without_old_producer_credit() {
  for n in ["probe::strata_matrix::cohorts::dual_regular::expert_dual_regular_cohort::ExpertDualRegularCohort<R>","probe::strata_matrix::cohorts::dual_small::expert_dual_small_cohort::ExpertDualSmallCohort<R>"] {
   assert_eq!(native_class(n),1);assert_eq!(matrix_family(n),6);
  }
  for source in [include_str!("strata_matrix_dual_regular_cohort.rs"),include_str!("strata_matrix_dual_small_cohort.rs")] {
   assert!(!source.contains("\"expert_regular_cohort_"));assert!(!source.contains("\"expert_granular_small_cohort_"));assert!(!source.contains("\"expert_local_cohort_"));
  }
 }
}

#[cfg(all(test,feature="strata-large-cohorts"))]
#[test]
fn large_cohort_real_family_and_all_producers_are_profiled() {
    let name=core::any::type_name::<crate::strata_matrix::cohorts::large::expert_large_cohort::ExpertLargeCohort<cubecl::wgpu::WgpuRuntime>>();
    assert_eq!(native_class(name),1);assert_eq!(matrix_family(name),6);
    for (name,class) in [("StrataExpertLargeCohortGuDirect8Image",0),
        ("StrataExpertLargeCohortDownDirect8Image",0),("StrataExpertLargeCohortGather",2),
        ("StrataExpertLargeCohortActivate",3)] {assert_eq!(native_class(name),class);}
}
