//! Exact GR read/combine using caller-owned resident buffers and whole-token faults.
use crate::{
    main_admission::{self, MainAdmission},
    private_arena::{BufferId, PrivateArena},
    resident_gdn::{BoundProjection, PublicGdnCache, ResidentProjector},
    resident_util as u, residual_gpu as r,
};
use cubecl::prelude::*;
use cubecl::server::Handle;
use cubecl::wgpu::WgpuRuntime;
use kuno_text::model::{AttentionKind, BoundGraph, GraphBindingId};
use kuno_text::{Error, Result};
use std::borrow::Cow;
use std::collections::BTreeMap;
use std::sync::atomic::AtomicBool;

pub struct PublicVectors {
    client: ComputeClient<WgpuRuntime>,
    binding: GraphBindingId,
    cache: BTreeMap<String, (usize, Handle)>,
    full_page_bytes: u64,
    prewarmed: bool,
}

const PUBLIC_SIN_NAME: &str = "kuno_public_qsa_rope_sin32_theta1e7_positions4096";
const PUBLIC_COS_NAME: &str = "kuno_public_qsa_rope_cos32_theta1e7_positions4096";
fn public_request_bytes(values: &[f32]) -> Result<u64> {
    if values.is_empty() || values.len() > 131072 || values.iter().any(|x| !x.is_finite()) {
        return Err(Error::InvalidBufferLength);
    }
    (values.len() as u64)
        .checked_mul(4)
        .ok_or(Error::ArithmeticOverflow)
}
fn public_page_bytes(values: &[f32]) -> Result<u64> {
    main_admission::page_bytes(public_request_bytes(values)?)
        .map_err(|_| Error::InvalidFormat("public-vector page geometry"))
}
fn padded_public_values(values: &[f32]) -> Result<Vec<f32>> {
    let words =
        usize::try_from(public_page_bytes(values)? / 4).map_err(|_| Error::ArithmeticOverflow)?;
    let mut padded = Vec::new();
    padded
        .try_reserve_exact(words)
        .map_err(|_| Error::LimitExceeded("public-vector padded scratch"))?;
    padded.resize(words, 0.0_f32);
    padded[..values.len()].copy_from_slice(values);
    Ok(padded)
}
fn public_logical_view(owned: Handle, requested: u64, page_bytes: u64) -> Result<Handle> {
    if requested == 0
        || requested > page_bytes
        || owned.size() != page_bytes
        || owned.size_in_used() != page_bytes
    {
        return Err(Error::InvalidFormat("public-vector owned page extent"));
    }
    let view = owned.offset_end(page_bytes - requested);
    if view.size_in_used() != requested {
        return Err(Error::InvalidFormat("public-vector logical view"));
    }
    Ok(view)
}
fn plan_page_bytes(plan: &[(&str, &[f32])]) -> Result<u64> {
    let mut names = std::collections::BTreeSet::new();
    plan.iter().try_fold(0_u64, |sum, &(name, values)| {
        if !names.insert(name) {
            return Err(Error::InvalidFormat("duplicate public-vector name"));
        }
        sum.checked_add(public_page_bytes(values)?)
            .ok_or(Error::ArithmeticOverflow)
    })
}
struct PublicVector<'model> {
    name: Cow<'model, str>,
    values: Cow<'model, [f32]>,
}
fn combined_controls(dt: &[f32], a: &[f32]) -> Result<Vec<f32>> {
    if dt.len() != a.len() || dt.is_empty() {
        return Err(Error::InvalidBufferLength);
    }
    let words = dt
        .len()
        .checked_add(a.len())
        .ok_or(Error::ArithmeticOverflow)?;
    let mut controls = Vec::new();
    controls
        .try_reserve_exact(words)
        .map_err(|_| Error::LimitExceeded("public GDN control scratch"))?;
    // Same exact converted values/order as PublicGdnParameters::prepare.
    controls.extend_from_slice(dt);
    controls.extend_from_slice(a);
    public_request_bytes(&controls)?;
    Ok(controls)
}
fn prewarm_plan<'model>(graph: BoundGraph<'model, '_>) -> Result<Vec<PublicVector<'model>>> {
    let mut plan = graph
        .static_vectors()
        .map(|(name, values)| PublicVector {
            name: Cow::Borrowed(name),
            values: Cow::Borrowed(values),
        })
        .collect::<Vec<_>>();
    let (sin, cos) = crate::qsa_gpu::rope_tables();
    for (name, values) in [(PUBLIC_SIN_NAME, sin), (PUBLIC_COS_NAME, cos)] {
        plan.push(PublicVector {
            name: Cow::Borrowed(name),
            values: Cow::Borrowed(values),
        });
    }
    // GDN controls are a derived PUBLIC cache entry, absent from static_vectors.
    for layer in 0..graph.geometry().layers {
        if graph.layer_kind(layer)? == AttentionKind::Gdn {
            let base = format!("blk.{layer}.");
            let dt = graph.static_vector(&format!("{base}ssm_dt.bias"))?;
            let a = graph.static_vector(&format!("{base}ssm_a"))?;
            plan.push(PublicVector {
                name: Cow::Owned(format!("{base}kuno_public_dt_and_a")),
                values: Cow::Owned(combined_controls(dt, a)?),
            });
        }
    }
    planned_pages(&plan)?;
    Ok(plan)
}
fn planned_pages(plan: &[PublicVector<'_>]) -> Result<u64> {
    let borrowed = plan
        .iter()
        .map(|v| (v.name.as_ref(), v.values.as_ref()))
        .collect::<Vec<_>>();
    plan_page_bytes(&borrowed)
}
fn admission_error(error: main_admission::AdmissionError) -> Error {
    match error {
        main_admission::AdmissionError::ArithmeticOverflow => Error::ArithmeticOverflow,
        _ => Error::LimitExceeded("public-vector main-pool admission"),
    }
}

impl PublicVectors {
    pub fn new(client: ComputeClient<WgpuRuntime>, binding: GraphBindingId) -> Self {
        Self {
            client,
            binding,
            cache: BTreeMap::new(),
            full_page_bytes: 0,
            prewarmed: false,
        }
    }
    /// Complete pages of all admitted static vectors, GDN combined controls
    /// and the two common RoPE tables. Pure host plan; no GPU allocation.
    pub fn prewarm_page_bytes(graph: BoundGraph<'_, '_>) -> Result<u64> {
        planned_pages(&prewarm_plan(graph)?)
    }
    /// Accounted enclosing pages, not payload bytes. Exact after prewarm for
    /// the required power-of-two exclusive allocator. Does not query a device.
    pub fn full_page_bytes(&self) -> u64 {
        self.full_page_bytes
    }
    /// Run during exclusive owner setup, before constructing request state.
    /// Existing entries are already included in admission's actual baseline;
    /// only missing vectors are precharged. Queries submit setup work, so no
    /// request graph should be pending. On ANY failure discard this context.
    /// Successful prewarm seals the inventory against late allocations.
    pub fn prewarm(
        &mut self,
        graph: BoundGraph<'_, '_>,
        admission: &mut MainAdmission,
    ) -> Result<u64> {
        if graph.binding_id() != self.binding {
            return Err(Error::InvalidFormat("public-vector graph binding"));
        }
        if admission.is_poisoned() {
            return Err(admission_error(main_admission::AdmissionError::Poisoned));
        }
        graph.verify_unchanged()?;
        let plan = prewarm_plan(graph)?;
        let expected = planned_pages(&plan)?;
        if self.cache.iter().any(|(name, (words, _))| {
            !plan
                .iter()
                .any(|v| name == v.name.as_ref() && *words == v.values.len())
        }) {
            return Err(Error::InvalidFormat("public-vector prewarm inventory"));
        }
        if self.prewarmed {
            if self.full_page_bytes != expected || self.cache.len() != plan.len() {
                return Err(Error::InvalidFormat("public-vector prewarm invariant"));
            }
            return Ok(expected);
        }
        let result = (|| {
            let before = self
                .client
                .memory_usage()
                .map_err(|_| Error::InvalidFormat("GPU memory telemetry"))?
                .bytes_reserved;
            admission.observe_before(before).map_err(admission_error)?;
            let requests = plan
                .iter()
                .filter(|v| !self.cache.contains_key(v.name.as_ref()))
                // Upload owns its whole bin, so charge its full padded extent.
                .map(|v| public_page_bytes(&v.values))
                .collect::<Result<Vec<_>>>()?;
            admission.precharge(&requests).map_err(admission_error)?;
            for v in &plan {
                self.upload_public_vector(self.binding, &v.name, &v.values)?;
            }
            let after = self
                .client
                .memory_usage()
                .map_err(|_| Error::InvalidFormat("GPU memory telemetry"))?
                .bytes_reserved;
            admission.observe_after(after).map_err(admission_error)?;
            if self.full_page_bytes != expected || self.cache.len() != plan.len() {
                return Err(Error::InvalidFormat("public-vector prewarm invariant"));
            }
            Ok(expected)
        })();
        match result {
            Ok(bytes) => {
                self.prewarmed = true;
                Ok(bytes)
            }
            Err(error) => {
                admission.poison();
                Err(error)
            }
        }
    }
    pub fn vector(&mut self, graph: BoundGraph<'_, '_>, name: &str) -> Result<Handle> {
        self.upload_public_vector(graph.binding_id(), name, graph.static_vector(name)?)
    }
}
impl PublicGdnCache for PublicVectors {
    fn upload_public_vector(
        &mut self,
        binding: GraphBindingId,
        name: &str,
        values: &[f32],
    ) -> Result<Handle> {
        if binding != self.binding {
            return Err(Error::InvalidBufferLength);
        }
        let page_bytes = public_page_bytes(values)?;
        if let Some((n, h)) = self.cache.get(name) {
            if *n != values.len() {
                return Err(Error::InvalidBufferLength);
            }
            return Ok(h.clone());
        }
        if self.prewarmed {
            return Err(Error::InvalidFormat("sealed public-vector cache miss"));
        }
        let total = self
            .full_page_bytes
            .checked_add(page_bytes)
            .ok_or(Error::ArithmeticOverflow)?;
        let requested = public_request_bytes(values)?;
        let padded = padded_public_values(values)?;
        let owned = self.client.create_from_slice(f32::as_bytes(&padded));
        // Keep the complete page owned but expose the original logical span.
        // This preserves existing shader array lengths and mixed-gamma choice.
        let handle = public_logical_view(owned, requested, page_bytes)?;
        self.cache
            .insert(name.to_owned(), (values.len(), handle.clone()));
        self.full_page_bytes = total;
        Ok(handle)
    }
}
pub struct ResidentProjection<'executor, 'file> {
    pub executor: &'executor mut crate::executor::GpuExecutor<'file>,
    pub fault: BufferId,
}
impl ResidentProjector for ResidentProjection<'_, '_> {
    fn enqueue(
        &mut self,
        projection: BoundProjection<'_>,
        arena: &PrivateArena,
        input: BufferId,
        output: BufferId,
        cancel: &AtomicBool,
    ) -> Result<()> {
        u::cancel(cancel)?;
        self.executor
            .check_client(arena.client().map_err(u::fault)?)?;
        if input == output
            || input == self.fault
            || output == self.fault
            || u::words(arena, input)? != projection.matrix.inputs()
            || u::words(arena, output)? != projection.matrix.outputs()
        {
            return Err(Error::InvalidBufferLength);
        }
        // SAFETY: arena active IDs are distinct exclusive spans on this client;
        // the full graph owns them through drain/erasure. Device results below
        // join the whole-token fault decision before any state commit.
        let fused = unsafe {
            self.executor.enqueue_matrix_with_finite_fault(
                projection.tensor,
                0,
                &u::handle(arena, input)?,
                projection.matrix.inputs(),
                &u::handle(arena, output)?,
                projection.matrix.outputs(),
                &u::handle(arena, self.fault)?,
                cancel,
            )?
        };
        if !fused {
            u::validate(arena, output, self.fault)?;
        }
        Ok(())
    }
}
pub fn project(
    graph: BoundGraph<'_, '_>,
    projector: &mut impl ResidentProjector,
    arena: &PrivateArena,
    name: &str,
    input: BufferId,
    output: BufferId,
    cancel: &AtomicBool,
) -> Result<()> {
    let tensor = graph.tensor(name)?;
    projector.enqueue(
        BoundProjection {
            tensor,
            matrix: kuno_text::cpu::Matrix::new(tensor)?,
        },
        arena,
        input,
        output,
        cancel,
    )
}

pub struct GrWorkspace {
    pub mixed: BufferId,
    pub injection: BufferId,
    norm: BufferId,
    down: BufferId,
    low: BufferId,
    up: BufferId,
    status_norm: BufferId,
    status_low: BufferId,
    status_mix: BufferId,
    status_combine: BufferId,
    layout: r::ResidualLayout,
}
impl GrWorkspace {
    pub fn allocate(graph: BoundGraph<'_, '_>, arena: &mut PrivateArena) -> Result<Self> {
        let g = graph.geometry();
        let layout = r::ResidualLayout::new(g.residual, g.low_rank).map_err(u::fault)?;
        let mut alloc = |n| arena.allocate_zeroed(n).map_err(u::fault);
        Ok(Self {
            mixed: alloc(g.residual.width)?,
            injection: alloc(g.residual.streams)?,
            norm: alloc(layout.residual_elements)?,
            down: alloc(g.low_rank)?,
            low: alloc(g.low_rank)?,
            up: alloc(layout.residual_elements)?,
            status_norm: alloc(g.residual.streams)?,
            status_low: alloc(g.low_rank)?,
            status_mix: alloc(g.residual.width)?,
            status_combine: alloc(layout.residual_elements)?,
            layout,
        })
    }
    #[allow(clippy::too_many_arguments)]
    pub fn read(
        &self,
        graph: BoundGraph<'_, '_>,
        arena: &PrivateArena,
        public: &mut PublicVectors,
        projector: &mut impl ResidentProjector,
        base: &str,
        residual: BufferId,
        inject: bool,
        fault: BufferId,
        cancel: &AtomicBool,
    ) -> Result<()> {
        let s = self.layout.shape;
        if u::words(arena, residual)? != self.layout.residual_elements {
            return Err(Error::InvalidBufferLength);
        }
        let name = format!("{base}_norm.weight");
        let gamma = public.vector(graph, &name)?;
        let client = arena.client().map_err(u::fault)?;
        r::grouped_rms_norm::launch::<WgpuRuntime>(
            client,
            CubeCount::Static(s.streams as u32, 1, 1),
            CubeDim::new_1d(64),
            u::arg(arena, residual)?,
            u::raw_arg(gamma, self.layout.residual_elements)?,
            u::arg(arena, self.norm)?,
            u::arg(arena, self.status_norm)?,
            graph.geometry().epsilon,
            s.streams,
            s.width,
        );
        u::status(arena, self.status_norm, fault)?;
        project(
            graph,
            projector,
            arena,
            &format!("{base}_down.weight"),
            self.norm,
            self.down,
            cancel,
        )?;
        r::activate_down::launch::<WgpuRuntime>(
            client,
            u::grid(self.layout.low_rank)?,
            CubeDim::new_1d(64),
            u::arg(arena, self.down)?,
            u::arg(arena, self.low)?,
            u::arg(arena, self.status_low)?,
            s.streams,
            self.layout.low_rank,
        );
        u::status(arena, self.status_low, fault)?;
        project(
            graph,
            projector,
            arena,
            &format!("{base}_up.weight"),
            self.low,
            self.up,
            cancel,
        )?;
        r::mix::launch::<WgpuRuntime>(
            client,
            u::grid(s.width)?,
            CubeDim::new_1d(64),
            u::arg(arena, self.norm)?,
            u::arg(arena, self.up)?,
            u::arg(arena, self.mixed)?,
            u::arg(arena, self.status_mix)?,
            s.streams,
            s.width,
        );
        u::status(arena, self.status_mix, fault)?;
        if inject {
            project(
                graph,
                projector,
                arena,
                &format!("{base}_inject.weight"),
                self.norm,
                self.injection,
                cancel,
            )?;
        }
        Ok(())
    }
    pub fn combine(
        &self,
        arena: &PrivateArena,
        residual: BufferId,
        block: BufferId,
        output: BufferId,
        fault: BufferId,
    ) -> Result<()> {
        let s = self.layout.shape;
        if [residual, block, self.injection, output, self.status_combine]
            .iter()
            .enumerate()
            .any(|(i, id)| {
                [residual, block, self.injection, output, self.status_combine][..i].contains(id)
            })
        {
            return Err(Error::InvalidBufferLength);
        }
        r::combine::launch::<WgpuRuntime>(
            arena.client().map_err(u::fault)?,
            u::grid(self.layout.residual_elements)?,
            CubeDim::new_1d(64),
            u::arg(arena, residual)?,
            u::arg(arena, block)?,
            u::arg(arena, self.injection)?,
            u::arg(arena, output)?,
            u::arg(arena, self.status_combine)?,
            s.streams,
            s.width,
        );
        u::status(arena, self.status_combine, fault)
    }
}

#[cfg(test)]
mod public_page_tests {
    use super::*;
    #[test]
    fn independent_inventory_counts_complete_pages_and_both_tables() {
        let gamma = [1.0_f32];
        let residual_gamma = vec![1.0; 10240];
        let convolution = vec![0.25; 40960];
        let table = vec![0.0; 131072];
        let plan = [
            ("tiny.gamma", gamma.as_slice()),
            ("residual.gamma", residual_gamma.as_slice()),
            ("conv", convolution.as_slice()),
            (PUBLIC_SIN_NAME, table.as_slice()),
            (PUBLIC_COS_NAME, table.as_slice()),
        ];
        // Independent fixture page sizes: 32K + 64K + 256K + 512K + 512K.
        assert_eq!(plan_page_bytes(&plan).unwrap(), 1_409_024);
        assert_eq!(public_request_bytes(&residual_gamma).unwrap(), 40_960);
    }
    #[test]
    fn duplicate_and_invalid_public_vectors_fail_before_upload() {
        let gamma = [1.0_f32];
        assert!(plan_page_bytes(&[("x", &gamma), ("x", &gamma)]).is_err());
        for values in [&[][..], &[f32::NAN][..], &[f32::INFINITY][..]] {
            assert!(public_page_bytes(values).is_err());
        }
        assert!(public_page_bytes(&vec![1.0; 131073]).is_err());
        assert_eq!(plan_page_bytes(&[]).unwrap(), 0);
    }
    #[test]
    fn inventory_precharge_rounds_every_public_name_and_does_not_refund() {
        let gamma = [1.0_f32];
        let requests = [public_request_bytes(&gamma).unwrap(); 3];
        let mut ledger = MainAdmission::new(3 * 32768, 0).unwrap();
        assert_eq!(ledger.precharge(&requests).unwrap(), 3 * 32768);
        ledger.observe_after(32768).unwrap();
        assert_eq!(ledger.upper_bytes(), 3 * 32768);
        ledger.completed_fence(32768).unwrap();
        assert_eq!(ledger.upper_bytes(), 32768);
    }
    #[test]
    fn combined_control_inventory_preserves_exact_dt_then_a_bits() {
        let dt = [0.0_f32, -0.0, 0.125];
        let a = [-1.0_f32, -0.5, -2.0];
        let controls = combined_controls(&dt, &a).unwrap();
        let expected = [
            0_u32, 0x80000000, 0x3e000000, 0xbf800000, 0xbf000000, 0xc0000000,
        ];
        assert_eq!(
            controls.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            expected
        );
        assert_eq!(public_page_bytes(&controls).unwrap(), 32768);
        assert!(combined_controls(&dt[..2], &a).is_err());
        assert!(combined_controls(&[], &[]).is_err());
        assert!(combined_controls(&[f32::NAN], &[-1.0]).is_err());
    }
    #[test]
    fn full_public_page_preserves_value_bits_and_original_logical_length() {
        let values = [-0.0_f32, 1.0, -0.125];
        let padded = padded_public_values(&values).unwrap();
        assert_eq!(padded.len(), 8192);
        assert_eq!(
            padded[..3].iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            [0x80000000, 0x3f800000, 0xbe000000]
        );
        assert!(padded[3..].iter().all(|v| v.to_bits() == 0));
        // Uninitialized host metadata handles only: no client/device operation.
        let stream = cubecl::stream_id::StreamId::current();
        let owned = Handle::new(stream, 32768);
        let view = public_logical_view(owned, 12, 32768).unwrap();
        assert_eq!(view.size(), 32768);
        assert_eq!(view.size_in_used(), 12);
        assert_eq!(view.offset_end, Some(32756));
        assert!(public_logical_view(Handle::new(stream, 32768), 32769, 32768).is_err());
        assert!(public_logical_view(Handle::new(stream, 32768), 12, 65536).is_err());
    }
}
