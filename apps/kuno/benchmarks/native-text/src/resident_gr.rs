//! Exact GR read/combine using caller-owned resident buffers and whole-token faults.
use crate::{
    private_arena::{BufferId, PrivateArena},
    resident_gdn::{BoundProjection, PublicGdnCache, ResidentProjector},
    resident_util as u, residual_gpu as r,
};
use cubecl::prelude::*;
use cubecl::server::Handle;
use cubecl::wgpu::WgpuRuntime;
use kuno_text::model::{BoundGraph, GraphBindingId};
use kuno_text::{Error, Result};
use std::collections::BTreeMap;
use std::sync::atomic::AtomicBool;

pub struct PublicVectors {
    client: ComputeClient<WgpuRuntime>,
    binding: GraphBindingId,
    cache: BTreeMap<String, (usize, Handle)>,
}
impl PublicVectors {
    pub fn new(client: ComputeClient<WgpuRuntime>, binding: GraphBindingId) -> Self {
        Self {
            client,
            binding,
            cache: BTreeMap::new(),
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
        if binding != self.binding
            || values.is_empty()
            || values.len() > 131072
            || values.iter().any(|x| !x.is_finite())
        {
            return Err(Error::InvalidBufferLength);
        }
        if let Some((n, h)) = self.cache.get(name) {
            if *n != values.len() {
                return Err(Error::InvalidBufferLength);
            }
            return Ok(h.clone());
        }
        let handle = self.client.create_from_slice(f32::as_bytes(values));
        self.cache
            .insert(name.to_owned(), (values.len(), handle.clone()));
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
            || u::words(arena, input)? != projection.matrix.inputs()
            || u::words(arena, output)? != projection.matrix.outputs()
        {
            return Err(Error::InvalidBufferLength);
        }
        // SAFETY: arena active IDs are distinct exclusive spans on this client;
        // the full graph owns them through drain/erasure. Device results below
        // join the whole-token fault decision before any state commit.
        unsafe {
            self.executor.enqueue_matrix(
                projection.tensor,
                0,
                &u::handle(arena, input)?,
                projection.matrix.inputs(),
                &u::handle(arena, output)?,
                projection.matrix.outputs(),
                cancel,
            )?;
        }
        u::validate(arena, output, self.fault)?;
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
