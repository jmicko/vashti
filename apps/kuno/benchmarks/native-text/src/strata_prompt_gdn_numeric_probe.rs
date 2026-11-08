//! Fixed-public complete-owner parity for the real selected GDN recurrence
//! primitive against the unchanged scalar shader. No prompt scope, model
//! admission, accepted prefix, receipt, request content or timing is fabricated.
use super::*;
use crate::{gr_batch_probe, private_arena::BufferId};
use zeroize::Zeroizing;
type ProbeResult<T> = std::result::Result<T, Box<dyn std::error::Error>>;

// PROMPT_GDN_PARITY_FIXTURE_BEGIN
const OWNER_ROWS: usize = 65;
const HEAD_WIDTH: usize = 128;
const STATE_ROWS: usize = 128;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Control {
    Cold,
    Warm,
    NonfiniteState,
    NonfiniteGate,
}
impl Control {
    fn fault(self) -> u32 {
        u32::from(matches!(self, Self::NonfiniteState | Self::NonfiniteGate))
    }
    fn label(self) -> &'static str {
        match self {
            Self::Cold => "cold",
            Self::Warm => "warm_nonzero_state",
            Self::NonfiniteState => "nonfinite_state",
            Self::NonfiniteGate => "nonfinite_gate",
        }
    }
}

#[derive(Clone, Copy)]
struct Case {
    start: usize,
    count: usize,
    keys: usize,
    heads: usize,
    control: Control,
}
impl Case {
    fn valid(self) -> bool {
        (1..=OWNER_ROWS).contains(&self.count)
            && (1..=16).contains(&self.keys)
            && (1..=48).contains(&self.heads)
            && self.heads.is_multiple_of(self.keys)
            && self
                .start
                .checked_add(self.count)
                .is_some_and(|v| v <= 32768)
    }
    fn channels(self) -> usize {
        HEAD_WIDTH * (self.keys * 2 + self.heads)
    }
    fn state_words(self) -> usize {
        STATE_ROWS * self.heads * HEAD_WIDTH
    }
    fn output_width(self) -> usize {
        self.heads * HEAD_WIDTH
    }
    fn params(self) -> [u32; 8] {
        [
            self.count as u32,
            self.start as u32,
            self.channels() as u32,
            self.keys as u32,
            self.heads as u32,
            0,
            1e-6f32.to_bits(),
            0,
        ]
    }
}

fn fixture(case: Case) -> (Vec<u32>, Vec<u32>, Vec<u32>) {
    assert!(case.valid());
    // The future token rows are deliberately nonfinite. The real primitive's
    // count and next-token prefetch guard must keep every such row unread.
    let mut h = vec![f32::NAN.to_bits(); OWNER_ROWS * case.channels()];
    for t in 0..case.count {
        for channel in 0..case.channels() {
            h[t * case.channels() + channel] =
                ((((t * 7 + channel * 11 + channel / HEAD_WIDTH * 3) % 31) as f32 - 15.0) / 256.0)
                    .to_bits();
        }
    }
    let mut controls = vec![f32::NAN.to_bits(); OWNER_ROWS * case.heads * 2];
    for t in 0..case.count {
        for head in 0..case.heads {
            controls[t * case.heads * 2 + head] =
                (-0.125f32 * ((t + head) % 5 + 1) as f32).to_bits();
            controls[t * case.heads * 2 + case.heads + head] =
                (0.125f32 * ((t * 3 + head) % 6 + 1) as f32).to_bits();
        }
    }
    let mut state = vec![0; case.state_words()];
    if case.control != Control::Cold {
        for row in 0..STATE_ROWS {
            for head in 0..case.heads {
                for column in 0..HEAD_WIDTH {
                    state[(row * case.heads + head) * HEAD_WIDTH + column] =
                        ((((row * 13 + head * 5 + column * 3) % 17) as f32 - 8.0) / 2048.0)
                            .to_bits();
                }
            }
        }
    }
    match case.control {
        Control::NonfiniteState => {
            // Row65 and column37 cover the third row group and second CB32
            // tile rather than only the first scalar/column block.
            state[(65 * case.heads + case.heads - 1) * HEAD_WIDTH + 37] = f32::NAN.to_bits();
        }
        Control::NonfiniteGate => {
            controls[(case.count / 2) * case.heads * 2 + case.heads - 1] = f32::NAN.to_bits();
        }
        _ => {}
    }
    (h, controls, state)
}

fn owner_matches_input(words: &[u32], input: &[u32]) -> bool {
    words.len() >= input.len()
        && words[..input.len()] == *input
        && words[input.len()..].iter().all(|&word| word == 0)
}
fn finite_payload_and_zero_tail(words: &[u32], live: usize, logical: usize) -> bool {
    live <= logical
        && words.len() >= logical
        && words[..live]
            .iter()
            .all(|&word| f32::from_bits(word).is_finite())
        && words[live..].iter().all(|&word| word == 0)
}
fn fault_matches(words: &[u32], control: Control) -> bool {
    words.first().copied() == Some(control.fault()) && words[1..].iter().all(|&word| word == 0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fixture_covers_complete_rows_state_layout_and_unread_future() {
        for (start, count, keys, heads, control) in [
            (0, 1, 1, 1, Control::Cold),
            (17, 3, 1, 2, Control::Warm),
            (2047, 7, 2, 6, Control::Warm),
            (4095, 31, 2, 6, Control::Cold),
            (4096, 32, 2, 6, Control::Warm),
            (32735, 33, 2, 6, Control::Warm),
            (0, 3, 16, 48, Control::Warm),
            (64, 65, 2, 6, Control::Warm),
            (17, 3, 1, 2, Control::NonfiniteState),
            (2047, 7, 2, 6, Control::NonfiniteGate),
        ] {
            let case = Case {
                start,
                count,
                keys,
                heads,
                control,
            };
            assert!(case.valid());
            assert!(!control.label().is_empty());
            assert_eq!(case.output_width(), heads * HEAD_WIDTH);
            assert_eq!(
                case.params(),
                [
                    count as u32,
                    start as u32,
                    case.channels() as u32,
                    keys as u32,
                    heads as u32,
                    0,
                    1e-6f32.to_bits(),
                    0
                ]
            );
            let (h, controls, state) = fixture(case);
            assert_eq!(h.len(), OWNER_ROWS * case.channels());
            assert_eq!(state.len(), case.state_words());
            assert!(
                h[..count * case.channels()]
                    .iter()
                    .all(|&v| f32::from_bits(v).is_finite())
            );
            assert!(
                h[count * case.channels()..]
                    .iter()
                    .all(|&v| f32::from_bits(v).is_nan())
            );
            assert!(
                controls[count * heads * 2..]
                    .iter()
                    .all(|&v| f32::from_bits(v).is_nan())
            );
            assert_eq!(
                state
                    .iter()
                    .filter(|&&v| !f32::from_bits(v).is_finite())
                    .count(),
                usize::from(control == Control::NonfiniteState)
            );
            assert_eq!(
                controls[..count * heads * 2]
                    .iter()
                    .filter(|&&v| !f32::from_bits(v).is_finite())
                    .count(),
                usize::from(control == Control::NonfiniteGate)
            );
            if control == Control::Cold {
                assert!(state.iter().all(|&word| word == 0));
            }
        }
        for (start, count, keys, heads) in [
            (0, 0, 1, 1),
            (0, 66, 1, 1),
            (0, 1, 0, 1),
            (0, 1, 17, 48),
            (0, 1, 2, 3),
            (32767, 2, 1, 1),
            (usize::MAX, 1, 1, 1),
        ] {
            assert!(
                !Case {
                    start,
                    count,
                    keys,
                    heads,
                    control: Control::Cold
                }
                .valid()
            );
        }
    }

    #[test]
    fn checker_rejects_live_nan_inactive_writes_owner_tail_and_fault_padding() {
        let mut words = vec![0; 20];
        words[..3].fill(1f32.to_bits());
        assert!(finite_payload_and_zero_tail(&words, 3, 16));
        words[2] = f32::NAN.to_bits();
        assert!(!finite_payload_and_zero_tail(&words, 3, 16));
        words[2] = 1f32.to_bits();
        words[3] = 1;
        assert!(!finite_payload_and_zero_tail(&words, 3, 16));
        words[3] = 0;
        words[19] = 1;
        assert!(!finite_payload_and_zero_tail(&words, 3, 16));
        assert!(!finite_payload_and_zero_tail(&words, 17, 16));
        assert!(owner_matches_input(&[7, 8, 0, 0], &[7, 8]));
        assert!(!owner_matches_input(&[7, 9, 0, 0], &[7, 8]));
        assert!(!owner_matches_input(&[7, 8, 0, 1], &[7, 8]));
        assert!(!owner_matches_input(&[7], &[7, 8]));
        assert!(fault_matches(&[0, 0], Control::Cold));
        assert!(fault_matches(&[1, 0], Control::NonfiniteState));
        assert!(fault_matches(&[1, 0], Control::NonfiniteGate));
        assert!(!fault_matches(&[0, 0], Control::NonfiniteGate));
        assert!(!fault_matches(&[1, 1], Control::NonfiniteGate));
        assert!(!fault_matches(&[], Control::NonfiniteGate));
    }
}
// PROMPT_GDN_PARITY_FIXTURE_END

/// A distinct task identity prevents the original scalar source from ever
/// reusing the specialized production Op's compiled pipeline/cache entry.
struct ScalarRecurrence;
impl KernelMetadata for ScalarRecurrence {
    fn name(&self) -> &'static str {
        "StrataPublicOriginalScalarPromptGdnRecurrence"
    }
    fn id(&self) -> KernelId {
        KernelId::new::<Self>().cube_dim(CubeDim::new_1d(128))
    }
    fn address_type(&self) -> StorageType {
        StorageType::Scalar(ElemType::UInt(UIntKind::U32))
    }
}
impl CubeTask<AutoCompiler> for ScalarRecurrence {
    fn compile(
        &self,
        _: &mut AutoCompiler,
        _: &cubecl::WgpuCompilationOptions,
        _: ExecutionMode,
        _: StorageType,
    ) -> std::result::Result<CompiledKernel<AutoCompiler>, CompilationError> {
        Ok(CompiledKernel {
            entrypoint_name: "main".into(),
            debug_name: None,
            source: Op::RecurrenceFault.scalar_source(),
            repr: None,
            cube_dim: CubeDim::new_1d(128),
            debug_info: None,
        })
    }
}

fn primitive_launch(
    arena: &PrivateArena,
    original: bool,
    ids: [BufferId; 6],
    case: Case,
) -> ProbeResult<()> {
    if !crate::strata_intel_core::enabled()
        || !arena.public_owned_zero_audit_enabled()
        || crate::strata_prompt::active()
        || !case.valid()
    {
        return Err("public GDN primitive diagnostic admission".into());
    }
    let logical = [
        OWNER_ROWS * case.channels(),
        OWNER_ROWS * case.heads * 2,
        case.state_words(),
        case.state_words(),
        OWNER_ROWS * case.output_width(),
        1,
    ];
    let mut args = KernelArguments::new();
    let first_stream = arena.handle(ids[0])?.stream;
    for (index, (&id, words)) in ids.iter().zip(logical).enumerate() {
        if ids[..index].contains(&id) {
            return Err("public GDN duplicate original owner".into());
        }
        let layout = arena.layout(id)?;
        let owner = arena.handle(id)?;
        if layout.logical_words != words
            || owner.size_in_used() != layout.owned_bytes as u64
            || owner.stream != first_stream
            || !arena.contains_private_view(owner)?
        {
            return Err("public GDN complete original owner geometry".into());
        }
        for &previous in &ids[..index] {
            if crate::strata_prompt::storage::overlaps(arena.handle(previous)?, owner)? {
                return Err("public GDN overlapping original owners".into());
            }
        }
        args = args.with_buffer(owner.clone().binding());
    }
    args = args.with_info(MetadataBindingInfo::custom(
        case.params()
            .chunks_exact(2)
            .map(|v| v[0] as u64 | ((v[1] as u64) << 32))
            .collect(),
    ));
    let grid = CubeCount::Static((case.heads * 4) as u32, 1, 1);
    // SAFETY: this fixed-public diagnostic checks all original complete owners,
    // role lengths, client identity, grid and parameter bounds above, using the
    // original production CubeTask. No prompt authority/receipt is granted.
    // The production launch wrapper and its genuine PromptScope requirement
    // are untouched. This is a primitive control, not an admitted prompt run.
    unsafe {
        if original {
            arena
                .client()?
                .launch_unchecked(Box::new(ScalarRecurrence), grid, args);
        } else {
            arena
                .client()?
                .launch_unchecked(Box::new(Op::RecurrenceFault), grid, args);
        }
    }
    Ok(())
}

fn read_owner(arena: &PrivateArena, id: BufferId) -> ProbeResult<Zeroizing<Vec<u32>>> {
    let words = gr_batch_probe::read_native_public(arena, id)?;
    if words.len() != arena.layout(id)?.owned_words {
        return Err("public GDN complete owner read extent".into());
    }
    Ok(words)
}

fn check(arena: &mut PrivateArena, case: Case) -> ProbeResult<serde_json::Value> {
    let (h, controls, state) = fixture(case);
    let h = Zeroizing::new(h);
    let controls = Zeroizing::new(controls);
    let state = Zeroizing::new(state);
    let inputs = [
        arena.upload_u32(&h)?,
        arena.upload_u32(&controls)?,
        arena.upload_u32(&state)?,
    ];
    let reference = [
        arena.allocate_zeroed(case.state_words())?,
        arena.allocate_zeroed(OWNER_ROWS * case.output_width())?,
        arena.allocate_zeroed(1)?,
    ];
    let actual = [
        arena.allocate_zeroed(case.state_words())?,
        arena.allocate_zeroed(OWNER_ROWS * case.output_width())?,
        arena.allocate_zeroed(1)?,
    ];
    let owners = |out: [BufferId; 3]| [inputs[0], inputs[1], inputs[2], out[0], out[1], out[2]];
    eprintln!("public_gdn_compiler_scope reference");
    primitive_launch(arena, true, owners(reference), case)?;
    cubecl::future::block_on(arena.client()?.sync())?;
    eprintln!("public_gdn_compiler_scope optimized");
    primitive_launch(arena, false, owners(actual), case)?;
    cubecl::future::block_on(arena.client()?.sync())?;
    eprintln!("public_gdn_compiler_scope complete");
    let reference_state = read_owner(arena, reference[0])?;
    let reference_output = read_owner(arena, reference[1])?;
    let reference_fault = read_owner(arena, reference[2])?;
    let actual_state = read_owner(arena, actual[0])?;
    let actual_output = read_owner(arena, actual[1])?;
    let actual_fault = read_owner(arena, actual[2])?;
    if reference_state.as_slice() != actual_state.as_slice()
        || reference_output.as_slice() != actual_output.as_slice()
    {
        return Err("public GDN original scalar full state/output bit parity failed".into());
    }
    if !finite_payload_and_zero_tail(&actual_state, case.state_words(), case.state_words())
        || !finite_payload_and_zero_tail(
            &actual_output,
            case.count * case.output_width(),
            OWNER_ROWS * case.output_width(),
        )
    {
        return Err("public GDN nonfinite payload inactive row or owner padding changed".into());
    }
    if !fault_matches(&reference_fault, case.control)
        || !fault_matches(&actual_fault, case.control)
        || reference_fault.as_slice() != actual_fault.as_slice()
    {
        return Err("public GDN original scalar fault parity failed".into());
    }
    for (id, input) in inputs.into_iter().zip([&h, &controls, &state]) {
        if !owner_matches_input(&read_owner(arena, id)?, input) {
            return Err("public GDN original input or owning padding changed".into());
        }
    }
    Ok(
        serde_json::json!({"family":"prompt-gdn-recurrence-scalar-parity",
        "control":case.control.label(),"start":case.start,"count":case.count,
        "key_heads":case.keys,"value_heads":case.heads,"fault":case.control.fault(),
        "bit_exact_complete_output_and_next_state":true,"full_owner_and_tail_checked":true,
        "actual_recurrence_fault_primitive":true,"unchanged_scalar_shader_reference":true,
        "unique_reference_kernel_identity":true,"fabricated_prompt_scope":false,
        "admitted_prompt_run":false}),
    )
}

pub(crate) fn cases(arena: &mut PrivateArena) -> ProbeResult<Vec<serde_json::Value>> {
    let mut reports = Vec::new();
    for (start, count, keys, heads, control) in [
        (0, 1, 1, 1, Control::Cold),
        (17, 3, 1, 2, Control::Warm),
        (2047, 7, 2, 6, Control::Warm),
        (4095, 31, 2, 6, Control::Cold),
        (4096, 32, 2, 6, Control::Warm),
        (32735, 33, 2, 6, Control::Warm),
        (0, 3, 16, 48, Control::Warm),
        (64, 65, 2, 6, Control::Warm),
        (17, 3, 1, 2, Control::NonfiniteState),
        (2047, 7, 2, 6, Control::NonfiniteGate),
    ] {
        let attempt = check(
            arena,
            Case {
                start,
                count,
                keys,
                heads,
                control,
            },
        );
        arena.reset()?;
        reports.push(attempt?);
    }
    Ok(reports)
}
