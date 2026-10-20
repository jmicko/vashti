//! Closed native expert grammar: six logical phases, authentic matrix pieces.
//! No resource lookup, device authority, completion or request cache lives here.
use super::native_expert_batch::{batch_fault, native_strata_expert_operator};
use super::{Binding, CubeCount, KernelArguments, MetadataBindingInfo, ServerError};
use crate::memory_management::ManagedMemoryId;
use cubecl_common::stream_id::StreamId;

const MAX_MEMBERS: u32 = 4096;
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
}

/// Counts are returned only after a complete original batch is validated. The
/// backend separately counts actual enqueues and must match these inventories.
#[derive(Clone, Copy, Debug, Default)]
pub struct NativeStrataExpertGrammarCounts {
    pub launches: usize,
    pub logical_phases: usize,
    pub matrix_pieces: usize,
    pub wide_prefixes: usize,
    pub wide_blocks: usize,
    pub tail_pieces: usize,
}
#[derive(Default)]
pub struct NativeStrataExpertGrammar {
    phase: usize,
    groups: usize,
    group: Option<Group>,
    pending: Option<Pending>,
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
fn generated(name: &str, module: &str, ty: &str) -> bool {
    // Exact role-bearing generated suffix, rather than a generic wide/tail
    // substring. Runtime type arguments may follow the actual generated type.
    name.split_once("::native_expert_wide::")
        .is_some_and(|(_, suffix)| {
            suffix
                .strip_prefix(module)
                .and_then(|s| s.strip_prefix("::"))
                .and_then(|s| s.strip_prefix(ty))
                .is_some_and(|s| s.is_empty() || s.starts_with('<'))
        })
}
fn is_wide(name: &str, down: bool) -> bool {
    if down {
        generated(name, "expert_wide_down", "ExpertWideDown")
    } else {
        generated(name, "expert_wide_gu", "ExpertWideGu")
    }
}
fn is_tail(name: &str, down: bool) -> bool {
    if down {
        generated(name, "expert_tail_down", "ExpertTailDown")
    } else {
        generated(name, "expert_tail_gu", "ExpertTailGu")
    }
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
        }
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
        if self.groups >= 512 {
            return Err(batch_fault("native expert grammar group capacity"));
        }
        if self.phase == 3 || self.phase == 5 {
            let down = self.phase == 5;
            let group = self
                .group
                .ok_or_else(|| batch_fault("native expert grammar group missing"))?;
            if let Some(prefix) = self.pending {
                if release || !is_tail(name, down) {
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
                let tile = if name.contains("::expert_macro::") {
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
                self.counts.matrix_pieces += 1;
                self.advance();
            } else {
                if release || !is_wide(name, down) {
                    return Err(batch_fault(
                        "native expert grammar original matrix role/order",
                    ));
                }
                let (params, views) = matrix(args, true, true, down, group)?;
                if params.count < 256
                    || params.count > MAX_MEMBERS
                    || params.base != 0
                    || params.piece != params.count / 128 * 128
                    || views[0] != group.input
                    || u64::from(params.count) * u64::from(params.columns) * 2 > views[0].used()
                    || !grid(count, params.rows / 128, params.piece / 128)
                {
                    return Err(batch_fault("native expert grammar exact wide prefix"));
                }
                self.counts.wide_prefixes += 1;
                self.counts.wide_blocks += (params.piece / 128) as usize;
                self.counts.matrix_pieces += 1;
                if params.piece == params.count {
                    self.advance();
                } else {
                    self.pending = Some(Pending { params, views });
                }
            }
        } else {
            if !native_strata_expert_operator(name, release, self.phase) {
                return Err(batch_fault(
                    "native expert original six-logical-phase order",
                ));
            }
            if self.phase == 0 {
                if args.buffers.len() != 4
                    || args.info.data.len() != 8
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
        if self.phase != 0
            || self.pending.is_some()
            || self.groups == 0
            || self.counts.logical_phases != self.groups * 6
            || self.counts.matrix_pieces != self.groups * 2 + self.counts.tail_pieces
            || self.counts.launches != self.groups * 4 + self.counts.matrix_pieces
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
