//! The explicit hybrid count1 branch reuses the original router packet owner.
//! A completed owning map is destroyed before mandatory audited source release.
use super::*;
const MIXED_WORDS: usize = EMBEDDING;
const LOGIT_FIRST: usize = MIXED_WORDS;
const FAULT_FIRST: usize = MIXED_WORDS + EXPERTS;

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
fn owning_count1_fold(
    input: &Array<u32>,
    router: &Array<u32>,
    fault: &mut Array<Atomic<u32>>,
    count: usize,
    input_first: usize,
    router_first: usize,
) {
    let i = ABSOLUTE_POS;
    let mut bad = count != 1;
    if i < input.len() {
        if i < input_first || i >= input_first + MIXED_WORDS {
            if input[i] != 0 {
                bad = true;
            }
        }
    }
    if i < router.len() {
        if i < router_first || i >= router_first + EXPERTS {
            if router[i] != 0 {
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
        validate_view(MAX_BATCH, iv.first, iv.stride, MIXED_WORDS, iv.words)?;
        validate_view(MAX_BATCH, rv.first, rv.stride, EXPERTS, rv.words)?;
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
        admission.readback_audited(cancel, || {
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
}
