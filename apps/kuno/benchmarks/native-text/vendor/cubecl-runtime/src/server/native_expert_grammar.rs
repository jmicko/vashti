//! Closed native expert grammar: six logical phases, authentic matrix pieces.
//! No resource lookup, device authority, completion or request cache lives here.
use super::native_expert_batch::{batch_fault, native_strata_expert_operator,
    native_strata_small_direct_generated, native_strata_small_direct_matrix, native_strata_global_small_tile};
use super::{Binding, CubeCount, KernelArguments, MetadataBindingInfo, ServerError};
use crate::memory_management::ManagedMemoryId;
use cubecl_common::stream_id::StreamId;
#[cfg(feature="kuno-expert-cohorts")]
#[path="native_expert_cohort_grammar.rs"]
mod cohort;

const MAX_MEMBERS: u32 = 4096;
const MIN_WIDE_MEMBERS: u32 = if cfg!(feature = "kuno-native-expert-medium") {
    128
} else {
    256
};
const MAP_BYTES: u64 = 65536 * 4;
const STATUS_BYTES: u64 = 131072 * 4;
const GU_BYTES: u64 = 4096 * 1280 * 4;
const PART_BYTES: u64 = 4096 * 10 * 2560 * 4;

#[derive(Clone, Copy, PartialEq, Eq)]
struct View {
    id: ManagedMemoryId,
    stream: StreamId,
    size: u64,
    first: u64,
    end: u64,
}
impl View {
    fn new(binding: &Binding) -> Result<Self, ServerError> {
        let first = binding.offset_start.unwrap_or(0);
        let end = binding.offset_end.unwrap_or(0);
        if first.checked_add(end).is_none_or(|n| n >= binding.size) {
            return Err(batch_fault("native expert grammar original view extent"));
        }
        Ok(Self {
            id: binding.memory.allocation_id(),
            stream: binding.stream,
            size: binding.size,
            first,
            end,
        })
    }
    fn used(self) -> u64 {
        self.size - self.first - self.end
    }
}

#[derive(Clone, Copy)]
struct Group {
    index: u32,
    members: u32,
    prompt: u32,
    input: View,
    map: View,
    fault: View,
}
#[derive(Clone, Copy, PartialEq, Eq)]
struct Params {
    columns: u32,
    rows: u32,
    count: u32,
    weight: u32,
    group: u32,
    base: u32,
    piece: u32,
}
#[derive(Clone, Copy)]
struct Pending {
    params: Params,
    views: [View; 6],
    ordered_down: bool,
}

/// Counts are returned only after a complete original batch is validated. The
/// backend separately counts actual enqueues and must match these inventories.
#[derive(Clone, Copy, Debug, Default)]
pub struct NativeStrataExpertGrammarCounts {
    pub cohort_groups:usize,
    pub cohort_launches:usize,
    pub cohort_batches:usize,
    pub quant_cohort_groups: usize,
    pub quant_cohort_launches: usize,
    pub quant_cohort_batches: usize,
    pub launches: usize,
    pub logical_phases: usize,
    pub matrix_pieces: usize,
    pub wide_prefixes: usize,
    pub wide_blocks: usize,
    pub tail_pieces: usize,
    pub direct_gu_prefixes: usize,
    pub gu_checks: usize,
}
#[derive(Default)]
pub struct NativeStrataExpertGrammar {
    phase: usize,
    groups: usize,
    #[cfg(feature="kuno-expert-cohorts")]
    cohort:Option<cohort::Cohort>,
    group: Option<Group>,
    pending: Option<Pending>,
    resident_images: Option<bool>,
    gu_views: Option<[View; 6]>,
    gu_deferred: bool,
    counts: NativeStrataExpertGrammarCounts,
}
fn word(info: &MetadataBindingInfo, index: usize) -> Result<u32, ServerError> {
    let bytes = info
        .data
        .get(index / 2)
        .ok_or_else(|| batch_fault("native expert grammar original INFO length"))?
        .to_ne_bytes();
    let offset = index % 2 * 4;
    Ok(u32::from_ne_bytes(
        bytes[offset..offset + 4].try_into().unwrap(),
    ))
}
fn grid(count: &CubeCount, x: u32, y: u32) -> bool {
    matches!(count, CubeCount::Static(a,b,c) if *a==x && *b==y && *c==1)
}
// Exact new identities; this maps roles, never resource or phase authority.
fn global_register_generated(name:&str,module:&str,ty:&str)->bool {
 if !cfg!(feature="kuno-global-image-experts") {return false;}
 let actual=match(module,ty) {
  ("expert_wide_gu_direct","ExpertWideGuDirect")=>"expert_global_register_gu::ExpertGlobalRegisterGu",
  ("expert_wide_down","ExpertWideDown")=>"expert_global_register_down::ExpertGlobalRegisterDown",
  _=>return false,
 };
 name.split_once("::strata_matrix::native_expert_wide::global_register::")
  .is_some_and(|(_,s)|s.strip_prefix(actual).is_some_and(super::native_expert_batch::native_strata_generated_type_suffix))
}
fn global_wide32_generated(name:&str,module:&str,ty:&str)->bool {
 if !cfg!(feature="kuno-global-wide32"){return false;}
 let actual=match(module,ty){
 ("expert_wide_gu_direct","ExpertWideGuDirect")=>"expert_global_wide32_gu::ExpertGlobalWide32Gu",
 ("expert_wide_down","ExpertWideDown")=>"expert_global_wide32_down::ExpertGlobalWide32Down",
 _=>return false};
 name.split_once("::strata_matrix::native_expert_wide::global_wide32::").is_some_and(|(_,s)|s.strip_prefix(actual).is_some_and(super::native_expert_batch::native_strata_generated_type_suffix))
}
fn global_wide32_prefix(name:&str)->bool {
 global_wide32_generated(name,"expert_wide_gu_direct","ExpertWideGuDirect") || global_wide32_generated(name,"expert_wide_down","ExpertWideDown")
}
fn global_transpose_generated(name:&str,module:&str,ty:&str)->bool {
 if !cfg!(feature="kuno-global-transpose"){return false;}
 let actual=match(module,ty){
 ("expert_wide_gu_direct","ExpertWideGuDirect")=>"expert_global_transpose_gu::ExpertGlobalTransposeGu",
 ("expert_wide_down","ExpertWideDown")=>"expert_global_transpose_down::ExpertGlobalTransposeDown",
 _=>return false};
 name.split_once("::strata_matrix::native_expert_wide::global_transpose::").is_some_and(|(_,s)|s.strip_prefix(actual).is_some_and(super::native_expert_batch::native_strata_generated_type_suffix))
}
fn global_transpose_prefix(name:&str)->bool {
 global_transpose_generated(name,"expert_wide_gu_direct","ExpertWideGuDirect") || global_transpose_generated(name,"expert_wide_down","ExpertWideDown")
}
fn global_n64_generated(name:&str,module:&str,ty:&str)->bool {
 if !cfg!(feature="kuno-global-register-n64"){return false;}
 let actual=match(module,ty){
 ("expert_wide_gu_direct","ExpertWideGuDirect")=>"expert_global_n64_gu::ExpertGlobalN64Gu",
 ("expert_wide_down","ExpertWideDown")=>"expert_global_n64_down::ExpertGlobalN64Down",
 _=>return false};
 name.split_once("::strata_matrix::native_expert_wide::global_n64::").is_some_and(|(_,s)|s.strip_prefix(actual).is_some_and(super::native_expert_batch::native_strata_generated_type_suffix))
}
fn global_n64_prefix(name:&str)->bool {
 global_n64_generated(name,"expert_wide_gu_direct","ExpertWideGuDirect") || global_n64_generated(name,"expert_wide_down","ExpertWideDown")
}
fn generated(name: &str, module: &str, ty: &str) -> bool {
    if name.contains("::global_small::") {return false;}
    if name.contains("::global_wide32::") {return global_wide32_generated(name,module,ty);}
    if name.contains("::global_transpose::") {return global_transpose_generated(name,module,ty);}
    if name.contains("::global_n64::") {return global_n64_generated(name,module,ty);}
    if name.contains("::global_register::") {
        return global_register_generated(name,module,ty);
    }
    // New entries have their own exact classifier. A forged old family nested
    // inside the new namespace must not escape into legacy suffix matching.
    if name.contains("::small_direct::") || name.contains("::quant_cohort::") { return false; }
    #[cfg(feature = "kuno-wide-shared-reuse")]
    if shared_reuse_generated(name, module, ty) {
        return true;
    }
    // Exact role-bearing generated suffix, rather than a generic wide/tail
    // substring. Runtime type arguments may follow the actual generated type.
    name.split_once("::native_expert_wide::")
        .is_some_and(|(_, suffix)| {
            #[cfg(feature = "kuno-expert-ordered-output")]
            let suffix = if let Some(ordered) = suffix.strip_prefix("ordered::") {
                if !matches!((module, ty),
                    ("expert_wide_down", "ExpertWideDown")
                    | ("expert_tail_down", "ExpertTailDown")) { return false; }
                ordered
            } else { suffix };
            // The new genuine K64 producers keep the complete original role,
            // INFO, owners, grid and phase grammar. Admit only this closed
            // additional namespace, never arbitrary nesting or deep tail names.
            #[cfg(feature = "kuno-deep-matrix")]
            let suffix = if let Some(deep) = suffix.strip_prefix("deep::") {
                if !matches!((module, ty),
                    ("expert_wide_gu", "ExpertWideGu")
                    | ("expert_wide_gu_direct", "ExpertWideGuDirect")
                    | ("expert_wide_down", "ExpertWideDown")) { return false; }
                deep
            } else { suffix };
            suffix
                .strip_prefix(module)
                .and_then(|s| s.strip_prefix("::"))
                .and_then(|s| s.strip_prefix(ty))
                .is_some_and(|s| s.is_empty() || s.starts_with('<'))
        })
}
#[cfg(feature = "kuno-wide-shared-reuse")]
fn shared_reuse_generated(name: &str, module: &str, ty: &str) -> bool {
    // These are separately generated K32 lifetime entries, not aliases for
    // arbitrary nesting, deep variants or changed tails. The original INFO,
    // role, prefix grid, owners, checker and exact tail checks still run.
    let (suffix, ordered) = if let Some((_, _ordered_suffix)) =
        name.split_once("::strata_matrix::native_expert_wide::ordered::shared_reuse::")
    {
        #[cfg(not(feature = "kuno-expert-ordered-output"))]
        return false;
        #[cfg(feature = "kuno-expert-ordered-output")]
        (_ordered_suffix, true)
    } else if let Some((_, suffix)) =
        name.split_once("::strata_matrix::native_expert_wide::shared_reuse::")
    {
        (suffix, false)
    } else {
        return false;
    };
    let allowed = if ordered {
        (module, ty) == ("expert_wide_down", "ExpertWideDown")
    } else {
        matches!((module, ty),
            ("expert_wide_gu", "ExpertWideGu")
            | ("expert_wide_gu_direct", "ExpertWideGuDirect")
            | ("expert_wide_down", "ExpertWideDown"))
    };
    allowed && suffix.strip_prefix(module)
        .and_then(|s| s.strip_prefix("::"))
        .and_then(|s| s.strip_prefix(ty))
        .is_some_and(|s| s.is_empty() || (s.starts_with('<') && s.ends_with('>')))
}
fn is_wide(name: &str, down: bool) -> bool {
    if down {
        generated(name, "expert_wide_down", "ExpertWideDown")
    } else {
        generated(name, "expert_wide_gu", "ExpertWideGu")
    }
}
fn is_direct_gu(name: &str) -> bool {
    generated(name, "expert_wide_gu_direct", "ExpertWideGuDirect")
}
fn is_tail(name: &str, down: bool) -> bool {
    if down && cfg!(feature = "kuno-expert-ordered-output")
        && native_strata_small_direct_generated(name, "expert_tail_down", "ExpertTailDown") {
        return true;
    }
    if !down && native_strata_small_direct_generated(name, "expert_tail_gu", "ExpertTailGu") {
        return true;
    }
    if down {
        generated(name, "expert_tail_down", "ExpertTailDown")
    } else {
        generated(name, "expert_tail_gu", "ExpertTailGu")
    }
}
fn ordered_down_layout(name: &str) -> bool {
    native_strata_global_small_tile(name,true).is_some() || name.contains("::ordered::") || global_wide32_generated(name,"expert_wide_down","ExpertWideDown") || global_transpose_generated(name,"expert_wide_down","ExpertWideDown") || global_n64_generated(name,"expert_wide_down","ExpertWideDown") || global_register_generated(name,"expert_wide_down","ExpertWideDown") || native_strata_small_direct_generated(
        name, "expert_tail_down", "ExpertTailDown")
}
fn matrix(
    args: &KernelArguments,
    wide: bool,
    vector: bool,
    down: bool,
    group: Group,
) -> Result<(Params, [View; 6]), ServerError> {
    let scalar_words = if wide { 8 } else { 6 };
    let total_words = scalar_words + 12;
    if args.buffers.len() != 6
        || !args.read_dependencies.is_empty()
        || !args.tensor_maps.is_empty()
        || !args.info.native_parameters.is_empty()
        || args.info.data.len() * 2 != total_words
        || args.info.dynamic_metadata_offset != total_words / 2
        || word(&args.info, scalar_words - 1)? != 0
    {
        return Err(batch_fault(
            "native expert grammar original matrix INFO schema",
        ));
    }
    let params = Params {
        columns: word(&args.info, 0)?,
        rows: word(&args.info, 1)?,
        count: word(&args.info, 2)?,
        weight: word(&args.info, 3)?,
        group: word(&args.info, 4)?,
        base: if wide { word(&args.info, 5)? } else { 0 },
        piece: if wide {
            word(&args.info, 6)?
        } else {
            word(&args.info, 2)?
        },
    };
    if params.columns != if down { 640 } else { 2560 }
        || params.rows != if down { 2560 } else { 1280 }
        || params.count != group.members
        || params.group != group.index
        || params.weight % 4 != 0
    {
        return Err(batch_fault(
            "native expert grammar original matrix geometry",
        ));
    }
    let views = [
        View::new(&args.buffers[0])?,
        View::new(&args.buffers[1])?,
        View::new(&args.buffers[2])?,
        View::new(&args.buffers[3])?,
        View::new(&args.buffers[4])?,
        View::new(&args.buffers[5])?,
    ];
    if views.iter().any(|v| v.stream != group.input.stream)
        || views[3] != group.map
        || views[5] != group.fault
        || views[3].used() != MAP_BYTES
        || views[4].used() != STATUS_BYTES
        || views[5].used() != 4
        || views[2].used() != if down { PART_BYTES } else { GU_BYTES }
        || [views[0], views[1]]
            .iter()
            .any(|v| v.first % 8 != 0 || v.used() % 8 != 0)
        || (u64::from(params.weight) + u64::from(params.rows) * u64::from(params.columns)) * 2
            > views[1].used()
    {
        return Err(batch_fault("native expert grammar original matrix owners"));
    }
    // The unchanged generated array metadata must describe these exact views.
    // Small original products use scalar f16, other families use vec4<f16>.
    for (i, view) in views.iter().enumerate() {
        let unit = if i < 2 { if vector { 8 } else { 2 } } else { 4 };
        let buffer_len = view.used() / unit;
        // Preserve actual KernelLauncher::process_array semantics. Existing
        // half-vector ArrayArgs pass their vector count as length; the original
        // launcher divides it by the vec4 type's vectorization once again.
        let logical_len = if vector && i < 2 {
            buffer_len / 4
        } else {
            buffer_len
        };
        if view.used() % unit != 0
            || u64::from(word(&args.info, scalar_words + i)?) != buffer_len
            || u64::from(word(&args.info, scalar_words + 6 + i)?) != logical_len
        {
            return Err(batch_fault(
                "native expert grammar original array INFO owner",
            ));
        }
    }
    Ok((params, views))
}

impl NativeStrataExpertGrammar {
    fn advance(&mut self) {
        self.counts.logical_phases += 1;
        self.phase += 1;
        if self.phase == 6 {
            self.phase = 0;
            self.groups += 1;
            self.group = None;
            self.resident_images = None;
            self.gu_views = None;
            self.gu_deferred = false;
        }
    }
    fn check_gu_activation(
        &self,
        count: &CubeCount,
        args: &KernelArguments,
    ) -> Result<(), ServerError> {
        let g = self
            .group
            .ok_or_else(|| batch_fault("GU checker group missing"))?;
        let gu = self
            .gu_views
            .ok_or_else(|| batch_fault("GU checker original product missing"))?;
        if args.buffers.len() != 5
            || args.info.data.len() != 8
            || args.info.dynamic_metadata_offset != 0
            || !args.info.native_parameters.is_empty()
            || !args.read_dependencies.is_empty()
            || !args.tensor_maps.is_empty()
            || !grid(count, (g.members * 5).div_ceil(4), 1)
            || word(&args.info, 0)? != g.prompt
            || word(&args.info, 1)? != 2560
            || word(&args.info, 2)? != 1280
            || word(&args.info, 4)? != g.index
            || word(&args.info, 5)? != 0
        {
            return Err(batch_fault("GU checker exact original schema/grid"));
        }
        let expected = [g.map, gu[2], g.input, gu[4], g.fault];
        for (actual, expected) in args.buffers.iter().zip(expected) {
            if View::new(actual)? != expected {
                return Err(batch_fault("GU checker exact original owners"));
            }
        }
        Ok(())
    }
    /// HOST copies never count as logical compute phases and may not split the
    /// coupled prefix/tail pair. Existing physical ownership checks still run.
    pub fn host_copies(&self) -> Result<(), ServerError> {
        if self.pending.is_some() {
            Err(batch_fault("native expert grammar split matrix copy"))
        } else {
            Ok(())
        }
    }
    pub fn launch(
        &mut self,
        name: &str,
        release: bool,
        count: &CubeCount,
        args: &KernelArguments,
    ) -> Result<(), ServerError> {
        let original_name=super::native_expert_batch::native_strata_expert_dequant_identity(name);
        if original_name!=name {
            let down=self.phase==2;
            if !matches!(self.phase,1|2)
                || word(&args.info,1)?!=if down {640}else{2560}
                || word(&args.info,2)?!=if down {2560}else{1280}
                || word(&args.info,5)?!=if down {2}else{0}
                || !grid(count,if down {6400}else{12800},1) {
                return Err(batch_fault("native expert direct8 authentic WG32 role/shape/grid"));
            }
        }
        let name=original_name;
        #[cfg(feature="kuno-expert-cohorts")]
        if self.cohort.is_some() || name=="StrataExpertCohortGather" {
            if self.phase!=0 || self.pending.is_some() || self.gu_deferred {return Err(batch_fault("native expert cohort interrupted original group"));}
            if self.cohort.is_none(){self.cohort=Some(cohort::Cohort::start(self.groups,args)?);}
            let cohort=self.cohort.as_mut().unwrap();
            let done=cohort.launch(name,release,count,args)?;
            self.counts.launches+=1;self.counts.cohort_launches+=1;
            if done {
                let groups=cohort.groups();
                if cohort.quantized() {
                    self.counts.quant_cohort_groups += groups;
                    self.counts.quant_cohort_batches += 1;
                    self.counts.quant_cohort_launches += cohort.launches();
                }
                self.groups+=groups;self.counts.cohort_groups+=groups;self.counts.cohort_batches+=1;
                self.counts.logical_phases+=groups*6;self.counts.matrix_pieces+=groups*2;self.counts.gu_checks+=groups;
                self.cohort=None;
            }
            return Ok(());
        }
        if self.groups >= 512 {
            return Err(batch_fault("native expert grammar group capacity"));
        }
        if self.phase == 3 || self.phase == 5 {
            let down = self.phase == 5;
            let group = self
                .group
                .ok_or_else(|| batch_fault("native expert grammar group missing"))?;
            if let Some(prefix) = self.pending {
                if release || !is_tail(name, down)
                    || ordered_down_layout(name) != prefix.ordered_down {
                    return Err(batch_fault("native expert grammar original tail order"));
                }
                let (tail, views) = matrix(args, true, true, down, group)?;
                let mut expected = prefix.params;
                expected.base = expected.piece;
                expected.piece = expected.count - expected.base;
                let bytes = u64::from(expected.base) * u64::from(expected.columns) * 2;
                let full = u64::from(expected.count) * u64::from(expected.columns) * 2;
                let mut input = prefix.views[0];
                input.first = input
                    .first
                    .checked_add(bytes)
                    .ok_or_else(|| batch_fault("native expert grammar tail offset overflow"))?;
                input.end = input
                    .end
                    .checked_add(prefix.views[0].used() - full)
                    .ok_or_else(|| batch_fault("native expert grammar tail extent overflow"))?;
                let tile = if expected.piece <= 8 {
                    8
                } else if expected.piece < 32 {
                    16
                } else {
                    32
                };
                if tail != expected
                    || views[0] != input
                    || views[1..] != prefix.views[1..]
                    || !grid(count, tail.rows / 64, tail.piece.div_ceil(tile))
                {
                    return Err(batch_fault(
                        "native expert grammar exact tail coverage/owners",
                    ));
                }
                self.pending = None;
                if !down && native_strata_small_direct_generated(name, "expert_tail_gu", "ExpertTailGu") {
                    // A checked old prefix may be followed by the genuine raw
                    // tail. Validate the whole GU owner before activation.
                    self.gu_deferred = true;
                }
                self.counts.tail_pieces += 1;
                self.counts.matrix_pieces += 1;
                self.advance();
            } else if native_strata_expert_operator(name, release, self.phase) {
                let (params, views) =
                    matrix(args, false, !name.contains("::expert::"), down, group)?;
                if views[0] != group.input
                    || u64::from(params.count) * u64::from(params.columns) * 2 > views[0].used()
                {
                    return Err(batch_fault(
                        "native expert grammar original full member owner",
                    ));
                }
                if native_strata_small_direct_matrix(name)
                    && ((name.contains("::expert_macro::") && params.count < 32)
                        || (name.contains("::expert_narrow::") && params.count >= 32)) {
                    return Err(batch_fault("native small direct member geometry"));
                }
                let global_small = native_strata_global_small_tile(name,down);
                let tile = if let Some(tile)=global_small {
                    if !if tile==32 {(32..64).contains(&params.count)} else {(64..256).contains(&params.count)} {
                        return Err(batch_fault("native global small original full member geometry"));
                    }
                    tile
                } else if name.contains("::expert_macro64::")
                    || name.contains("::native::ordered::expert_macro64_deep::") {
                    if !(64..256).contains(&params.count) {
                        return Err(batch_fault("native expert macro64 member geometry"));
                    }
                    64
                } else if name.contains("::expert_macro::") {
                    32
                } else if name.contains("::expert_narrow::") {
                    if params.count <= 8 { 8 } else { 16 }
                } else {
                    8
                };
                let n = if name.contains("::expert::") { 16 } else { 64 };
                if !grid(count, params.rows.div_ceil(n), params.count.div_ceil(tile)) {
                    return Err(batch_fault("native expert grammar original full dispatch"));
                }
                if !down {
                    self.gu_views = Some(views);
                    // Keep the original wide-prefix counter unchanged: this
                    // is a full small matrix, not a new wide prefix.
                    self.gu_deferred = native_strata_small_direct_matrix(name) || global_small.is_some();
                }
                self.counts.matrix_pieces += 1;
                self.advance();
            } else {
                if release || !(is_wide(name, down) || (!down && is_direct_gu(name))) {
                    return Err(batch_fault(
                        "native expert grammar original matrix role/order",
                    ));
                }
                let (params, views) = matrix(args, true, true, down, group)?;
                if params.count < MIN_WIDE_MEMBERS
                    || params.count > MAX_MEMBERS
                    || params.base != 0
                    || params.piece != params.count / 128 * 128
                    || views[0] != group.input
                    || u64::from(params.count) * u64::from(params.columns) * 2 > views[0].used()
                    || !grid(count, params.rows / if global_transpose_prefix(name)||global_n64_prefix(name){64}else{128}, params.piece / 128)
                {
                    return Err(batch_fault("native expert grammar exact wide prefix"));
                }
                if !down {
                    self.gu_views = Some(views);
                    self.gu_deferred = is_direct_gu(name);
                    self.counts.direct_gu_prefixes += usize::from(self.gu_deferred);
                }
                self.counts.wide_prefixes += 1;
                self.counts.wide_blocks += (params.piece / 128) as usize;
                self.counts.matrix_pieces += 1;
                if params.piece == params.count {
                    self.advance();
                } else {
                    self.pending = Some(Pending {
                        params, views, ordered_down: ordered_down_layout(name),
                    });
                }
            }
        } else {
            if self.phase == 4 && (self.gu_deferred || name == "StrataPromptExpertF16SwigluCheckGu")
            {
                if release || name != "StrataPromptExpertF16SwigluCheckGu" {
                    return Err(batch_fault("direct GU requires validating activation"));
                }
                self.check_gu_activation(count, args)?;
                self.counts.gu_checks += 1;
                self.gu_deferred = false;
            }
            if !native_strata_expert_operator(name, release, self.phase) {
                return Err(batch_fault(
                    "native expert original six-logical-phase order",
                ));
            }
            if self.phase == 1 {
                self.resident_images = Some(name == "StrataPromptExpertResidentGuImage");
            }
            if self.phase == 2
                && self.resident_images != Some(name == "StrataPromptExpertResidentDownImage")
            {
                return Err(batch_fault(
                    "native expert matched streamed/resident dequant pair",
                ));
            }
            if self.phase == 0 {
                if args.buffers.len() != 4
                    || args.info.data.len() != 8
                    || args.info.dynamic_metadata_offset != 0
                    || word(&args.info, 1)? != 2560
                    || word(&args.info, 2)? != 1280
                    || word(&args.info, 4)? != self.groups as u32
                {
                    return Err(batch_fault("native expert grammar original Gather schema"));
                }
                let CubeCount::Static(x, 1, 1) = count else {
                    return Err(batch_fault("native expert grammar original Gather grid"));
                };
                let members = x / 5;
                let prompt = word(&args.info, 0)?;
                if x % 5 != 0 || members == 0 || members > prompt || prompt > MAX_MEMBERS {
                    return Err(batch_fault("native expert grammar original member count"));
                }
                self.group = Some(Group {
                    index: self.groups as u32,
                    members,
                    prompt,
                    input: View::new(&args.buffers[2])?,
                    map: View::new(&args.buffers[0])?,
                    fault: View::new(&args.buffers[3])?,
                });
            }
            self.advance();
        }
        self.counts.launches += 1;
        Ok(())
    }
    pub fn finish(&self) -> Result<NativeStrataExpertGrammarCounts, ServerError> {
        #[cfg(feature="kuno-expert-cohorts")]
        if self.cohort.is_some(){return Err(batch_fault("native expert incomplete cohort"));}
        if self.phase != 0
            || self.pending.is_some()
            || self.gu_deferred
            || self.groups == 0
            || self.counts.logical_phases != self.groups * 6
            || self.counts.matrix_pieces != self.groups * 2 + self.counts.tail_pieces
            || self.counts.launches != (self.groups-self.counts.cohort_groups) * 4 + self.counts.matrix_pieces-self.counts.cohort_groups*2+self.counts.cohort_launches
        {
            return Err(batch_fault(
                "native expert incomplete original matrix phase",
            ));
        }
        Ok(self.counts)
    }
}

#[cfg(test)]
#[path = "native_expert_grammar_tests.rs"]
mod tests;
