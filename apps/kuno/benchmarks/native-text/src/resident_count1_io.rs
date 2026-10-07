//! The explicit hybrid count1 branch reuses the original router packet owner.
//! A completed owning map is destroyed before mandatory audited source release.
use super::*;
const MIXED_WORDS: usize = EMBEDDING;
const LOGIT_FIRST: usize = MIXED_WORDS;
const FAULT_FIRST: usize = MIXED_WORDS + EXPERTS;
const MIXED_OWNED_WORDS: usize = 131072 / 4;

/// Count1 changes only ignored logical rows. Prefix/stride gaps and allocator
/// padding are outside these intervals and remain independently inspected.
fn inactive_mixed_rows(
    count: usize,
    view: InputView,
    owned_words: usize,
) -> Result<Zeroizing<[[usize; 2]; MAX_BATCH - 1]>> {
    if count != 1 || BufferLayout::new(view.words).map_err(u::fault)?.owned_words != owned_words {
        return Err(error());
    }
    validate_view(MAX_BATCH, view.first, view.stride, MIXED_WORDS, view.words)?;
    let mut rows = Zeroizing::new([[0; 2]; MAX_BATCH - 1]);
    for token in 1..MAX_BATCH {
        let first = view
            .first
            .checked_add(
                token
                    .checked_mul(view.stride)
                    .ok_or(Error::ArithmeticOverflow)?,
            )
            .ok_or(Error::ArithmeticOverflow)?;
        rows[token - 1] = [
            first,
            first
                .checked_add(MIXED_WORDS)
                .ok_or(Error::ArithmeticOverflow)?,
        ];
    }
    Ok(rows)
}

pub(crate) struct Count1RouterInput {
    logits: RouterLogits,
    mixed: Box<Zeroizing<[f32; MIXED_WORDS]>>,
    origin: Option<Origin>,
}
struct Origin {
    binding: kuno_text::model::GraphBindingId,
    ids: Zeroizing<[BufferId; 2]>,
    view: Zeroizing<[usize; 3]>,
    start: Zeroizing<u64>,
}
impl Count1RouterInput {
    fn zero() -> Result<Self> {
        let out = Self {
            logits: RouterLogits::zero(1)?,
            mixed: Box::new(Zeroizing::new([0.; MIXED_WORDS])),
            origin: None,
        };
        if out
            .owned_bytes()
            .checked_add(ROUTER_BYTES)
            .is_none_or(|n| n > ROUTER_HOST_CAP)
        {
            return Err(error());
        }
        Ok(out)
    }
    pub(crate) fn logits(&self) -> &RouterLogits {
        &self.logits
    }
    pub(crate) fn mixed(&self) -> &[f32; MIXED_WORDS] {
        &self.mixed
    }
    pub(crate) fn owned_bytes(&self) -> usize {
        std::mem::size_of::<Self>()
            + std::mem::size_of::<RouterLogits>()
            + MAX_BATCH * EXPERTS * 4
            + MIXED_WORDS * 4
    }
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn validate_for(
        &self,
        graph: BoundGraph<'_, '_>,
        input: BufferId,
        fault: BufferId,
        layer: usize,
        start: u64,
        first: usize,
        stride: usize,
    ) -> Result<()> {
        let p = self.origin.as_ref().ok_or(error())?;
        if p.binding != graph.binding_id()
            || *p.ids != [input, fault]
            || *p.view != [layer, first, stride]
            || *p.start != start
        {
            return Err(error());
        }
        Ok(())
    }
    fn accept(&mut self, bytes: &[u8]) -> Result<()> {
        if bytes.len() != ROUTER_BYTES || word(bytes, FAULT_FIRST)? != 0 {
            return Err(error());
        }
        for i in 0..bytes.len() / 4 {
            let bits = word(bytes, i)?;
            if i < FAULT_FIRST {
                let value = f32::from_bits(bits);
                if !value.is_finite() {
                    return Err(error());
                }
                if i < LOGIT_FIRST {
                    self.mixed[i] = value;
                } else {
                    self.logits.rows[0][i - LOGIT_FIRST] = value;
                }
            } else if bits != 0 {
                return Err(error());
            }
        }
        Ok(())
    }
}

#[cube(launch)]
#[allow(clippy::collapsible_if)] // Nested guards prevent eager CubeCL indexing.
#[allow(clippy::neg_cmp_op_on_partial_ord)] // This f32 comparison also rejects NaN in CubeCL.
fn owning_count1_fold(
    input: &mut Array<u32>,
    router: &Array<u32>,
    fault: &mut Array<Atomic<u32>>,
    count: usize,
    input_first: usize,
    input_stride: usize,
    input_words: usize,
    router_first: usize,
) {
    let i = ABSOLUTE_POS;
    let mut valid = false;
    if count == 1
        && input_stride >= MIXED_WORDS
        && input_first <= input_words
        && input_words <= input.len()
    {
        if MIXED_WORDS <= input_words - input_first {
            let available = input_words - input_first - MIXED_WORDS;
            if MAX_BATCH - 1 <= available / input_stride {
                valid = true;
            }
        }
    }
    let mut bad = !valid;
    if i < input.len() {
        // Invalid counts/coordinates never normalize any source byte. The
        // explicit guard dominates the only private input store below.
        if valid {
            if i >= input_first && i < input_words {
                let local = i - input_first;
                let row = local / input_stride;
                let column = local % input_stride;
                if row < MAX_BATCH && column < MIXED_WORDS {
                    if row > 0 {
                        input[i] = 0;
                    } else {
                        let value = f32::reinterpret(input[i]);
                        if !(value.abs() <= f32::MAX) {
                            bad = true;
                        }
                    }
                } else {
                    if input[i] != 0 {
                        bad = true;
                    }
                }
            } else {
                if input[i] != 0 {
                    bad = true;
                }
            }
        }
    }
    if i < router.len() {
        if i < router_first || i >= router_first + EXPERTS {
            if router[i] != 0 {
                bad = true;
            }
        } else {
            let value = f32::reinterpret(router[i]);
            if !(value.abs() <= f32::MAX) {
                bad = true;
            }
        }
    }
    if i > 0 && i < fault.len() {
        if fault[i].load() != 0 {
            bad = true;
        }
    }
    if bad {
        fault[0].fetch_or(4);
    }
}
#[cube(launch)]
fn count1_packet(
    input: &Array<u32>,
    router: &Array<u32>,
    fault: &Array<u32>,
    out: &mut Array<u32>,
    count: usize,
    input_first: usize,
    router_first: usize,
) {
    let i = ABSOLUTE_POS;
    if i < out.len() {
        let mut value = 0u32;
        if count == 1 {
            if i < MIXED_WORDS {
                value = input[input_first + i];
            } else if i < FAULT_FIRST {
                value = router[router_first + i - LOGIT_FIRST];
            } else if i == FAULT_FIRST {
                value = fault[0];
            }
        } else {
            if i == FAULT_FIRST {
                value = 4;
            }
        }
        out[i] = value;
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn read_count1_router_input(
    graph: BoundGraph<'_, '_>,
    arena: &PrivateArena,
    admission: &ResidentAdmission,
    mixed: BufferId,
    router: BufferId,
    packet: BufferId,
    fault: BufferId,
    input_view: InputView,
    router_view: RouterView,
    count: usize,
    cancel: &AtomicBool,
    layer: usize,
    start: u64,
) -> Result<Count1RouterInput> {
    guarded(admission, || {
        active(arena, admission, cancel)?;
        shape(graph)?;
        if count != 1 || layer >= 48 || start >= 4096 {
            return Err(error());
        }
        let iv = Zeroizing::new(input_view);
        let rv = Zeroizing::new(router_view);
        validate_view(MAX_BATCH, rv.first, rv.stride, EXPERTS, rv.words)?;
        let _inactive = inactive_mixed_rows(count, *iv, MIXED_OWNED_WORDS)?;
        let [input, router_handle, packet_handle, fault_handle] = physical(
            arena,
            [mixed, router, packet, fault],
            [iv.words, rv.words, MAX_BATCH * EXPERTS + 1, 1],
            [131072, ROUTER_BYTES, ROUTER_BYTES, ROUTER_BYTES],
        )?;
        let client = arena.client().map_err(u::fault)?;
        owning_count1_fold::launch::<WgpuRuntime>(
            client,
            u::grid(131072 / 4)?,
            CubeDim::new_1d(64),
            u::raw_arg(input.clone(), 131072 / 4)?,
            u::raw_arg(router_handle.clone(), ROUTER_BYTES / 4)?,
            u::raw_arg(fault_handle.clone(), ROUTER_BYTES / 4)?,
            count,
            iv.first,
            iv.stride,
            iv.words,
            rv.first,
        );
        count1_packet::launch::<WgpuRuntime>(
            client,
            u::grid(ROUTER_BYTES / 4)?,
            CubeDim::new_1d(64),
            u::raw_arg(input, 131072 / 4)?,
            u::raw_arg(router_handle, ROUTER_BYTES / 4)?,
            u::raw_arg(fault_handle, ROUTER_BYTES / 4)?,
            u::raw_arg(packet_handle.clone(), ROUTER_BYTES / 4)?,
            count,
            iv.first,
            rv.first,
        );
        let retired = admission.retired_context();
        admission.readback_count1_bridge(cancel, || {
            let mut out = Count1RouterInput::zero()?;
            let mut out = completed_map(
                || read(arena, packet_handle, &retired),
                |bytes| {
                    out.accept(bytes)?;
                    Ok(out)
                },
            )?;
            active(arena, admission, cancel)?;
            out.origin = Some(Origin {
                binding: graph.binding_id(),
                ids: Zeroizing::new([mixed, fault]),
                view: Zeroizing::new([layer, iv.first, iv.stride]),
                start: Zeroizing::new(start),
            });
            Ok(out)
        })
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    fn fixture() -> Vec<u8> {
        let mut bytes = vec![0; ROUTER_BYTES];
        for i in 0..FAULT_FIRST {
            let value = if i % 19 == 0 {
                -0f32
            } else {
                (i % 37) as f32 / 41.
            };
            bytes[i * 4..i * 4 + 4].copy_from_slice(&value.to_bits().to_le_bytes());
        }
        bytes
    }
    #[test]
    fn reused_packet_preserves_signed_values_and_rejects_full_fault_tail() {
        let bytes = fixture();
        let mut out = Count1RouterInput::zero().unwrap();
        out.accept(&bytes).unwrap();
        assert_eq!(out.mixed()[0].to_bits(), (-0f32).to_bits());
        assert!(out.owned_bytes() + ROUTER_BYTES <= ROUTER_HOST_CAP);
        for at in [0, LOGIT_FIRST, FAULT_FIRST, FAULT_FIRST + 1, 4096, 8191] {
            let mut changed = bytes.clone();
            let value = if at < FAULT_FIRST {
                f32::NAN.to_bits()
            } else {
                1
            };
            changed[at * 4..at * 4 + 4].copy_from_slice(&value.to_le_bytes());
            assert!(Count1RouterInput::zero().unwrap().accept(&changed).is_err());
        }
    }

    fn mixed_fixture() -> (InputView, Vec<u32>) {
        let view = InputView {
            first: 3,
            stride: MIXED_WORDS + 5,
            words: 3 + MAX_BATCH * (MIXED_WORDS + 5) + 7,
        };
        let mut input = vec![0; MIXED_OWNED_WORDS];
        for token in 0..MAX_BATCH {
            for column in 0..MIXED_WORDS {
                input[view.first + token * view.stride + column] = if column == 0 {
                    (-0f32).to_bits()
                } else {
                    ((token + 1) as f32 + column as f32 / 4096.).to_bits()
                };
            }
        }
        (view, input)
    }

    // This uses production host-validated intervals. The offline test below
    // expands the actual mutating kernel; physical MAX8 -> count1 acceptance
    // must additionally exercise these stores on the original model owner.
    fn clear_public_fixture(count: usize, view: InputView, input: &mut [u32]) -> Result<()> {
        let rows = inactive_mixed_rows(count, view, input.len())?;
        for [first, end] in *rows {
            input[first..end].fill(0);
        }
        Ok(())
    }

    // Independent count1 invariant: one finite active row and exact zero at
    // every other owning location. In particular gaps are never logical rows.
    fn owning_input_accepts(view: InputView, input: &[u32]) -> bool {
        input.iter().enumerate().all(|(i, &bits)| {
            if (view.first..view.first + MIXED_WORDS).contains(&i) {
                f32::from_bits(bits).is_finite()
            } else {
                bits == 0
            }
        })
    }

    #[test]
    fn max8_mixed_rows_become_count1_without_changing_active_or_owning_gaps() {
        let (view, mut input) = mixed_fixture();
        let before = input.clone();
        assert!(!owning_input_accepts(view, &input));
        clear_public_fixture(1, view, &mut input).unwrap();
        assert!(owning_input_accepts(view, &input));
        assert_eq!(input[view.first], (-0f32).to_bits());
        for i in 0..input.len() {
            let inactive = (1..MAX_BATCH).any(|row| {
                (view.first + row * view.stride..view.first + row * view.stride + MIXED_WORDS)
                    .contains(&i)
            });
            assert_eq!(input[i], if inactive { 0 } else { before[i] });
        }
    }

    #[test]
    fn corrupt_mixed_active_gap_and_owning_tail_are_never_normalized() {
        let (view, input) = mixed_fixture();
        for at in [
            view.first - 1,
            view.first + MIXED_WORDS,
            view.first + view.stride + MIXED_WORDS,
            view.words - 1,
            MIXED_OWNED_WORDS - 1,
        ] {
            let mut changed = input.clone();
            changed[at] = 1;
            clear_public_fixture(1, view, &mut changed).unwrap();
            assert_eq!(changed[at], 1);
            assert!(!owning_input_accepts(view, &changed));
        }
        for bits in [f32::NAN.to_bits(), f32::INFINITY.to_bits()] {
            let mut changed = input.clone();
            changed[view.first] = bits;
            clear_public_fixture(1, view, &mut changed).unwrap();
            assert_eq!(changed[view.first], bits);
            assert!(!owning_input_accepts(view, &changed));
        }
    }

    #[test]
    fn malformed_mixed_count_and_view_reject_before_any_clear() {
        let (view, input) = mixed_fixture();
        for count in [0, 2, 8, 9] {
            let mut changed = input.clone();
            assert!(clear_public_fixture(count, view, &mut changed).is_err());
            assert_eq!(changed, input);
        }
        for bad in [
            InputView {
                stride: MIXED_WORDS - 1,
                ..view
            },
            InputView {
                first: usize::MAX,
                ..view
            },
            InputView {
                words: view.first + 7 * view.stride + MIXED_WORDS - 1,
                ..view
            },
            InputView {
                words: MIXED_OWNED_WORDS + 1,
                ..view
            },
        ] {
            let mut changed = input.clone();
            assert!(clear_public_fixture(1, bad, &mut changed).is_err());
            assert_eq!(changed, input);
        }
    }

    #[test]
    fn count1_normalization_shader_keeps_runtime_stride_count_and_full_owning_fault_scan() {
        use cubecl::compute::KernelBuilder;
        use cubecl::ir::{ElemType, FloatKind, StorageType, Type, UIntKind};
        use cubecl::server::ExecutionMode;
        use cubecl::wgpu::WgslCompiler;
        use cubecl::{Compiler, KernelSettings, WgpuCompilationOptions};
        for mode in [ExecutionMode::Checked, ExecutionMode::Unchecked] {
            let mut b = KernelBuilder::default();
            let uint = StorageType::Scalar(ElemType::UInt(UIntKind::U32));
            let float = StorageType::Scalar(ElemType::Float(FloatKind::F32));
            let atom = StorageType::Atomic(ElemType::UInt(UIntKind::U32));
            b.scope.register_type::<u32>(uint);
            b.scope.register_type::<usize>(uint);
            b.scope.register_type::<f32>(float);
            b.scope
                .register_type::<bool>(StorageType::Scalar(ElemType::Bool));
            let input = b.output_array(Type::new(uint)).into();
            let router = b.input_array(Type::new(uint)).into();
            let fault = b.output_array(Type::new(atom)).into();
            let count = b.scalar(uint).into();
            let input_first = b.scalar(uint).into();
            let input_stride = b.scalar(uint).into();
            let input_words = b.scalar(uint).into();
            let router_first = b.scalar(uint).into();
            owning_count1_fold::expand(
                &mut b.scope,
                input,
                router,
                fault,
                count,
                input_first,
                input_stride,
                input_words,
                router_first,
            );
            let source = WgslCompiler::default()
                .compile(
                    b.build(
                        KernelSettings::default()
                            .cube_dim(CubeDim::new_1d(64))
                            .kernel_name("count1_normalize_owned_input"),
                    ),
                    &WgpuCompilationOptions::default(),
                    mode,
                    uint,
                )
                .unwrap()
                .to_string();
            let module = wgpu::naga::front::wgsl::parse_str(&source).unwrap();
            wgpu::naga::valid::Validator::new(
                wgpu::naga::valid::ValidationFlags::all(),
                wgpu::naga::valid::Capabilities::all(),
            )
            .validate(&module)
            .unwrap();
            assert_eq!(module.entry_points.len(), 1);
            assert_eq!(module.entry_points[0].workgroup_size, [64, 1, 1]);
            for scalar in 0..5 {
                assert!(source.contains(&format!("scalars_u32[{scalar}]")));
            }
            assert!(source.contains("atomicOr"));
            assert!(source.contains("atomicLoad"));
            assert!(source.contains("bitcast<f32>"));
            assert!(!source.contains("workgroupBarrier"));
        }
    }
}
