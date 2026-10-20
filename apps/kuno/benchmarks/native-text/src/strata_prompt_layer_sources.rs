//! Borrow one original Sources/layer during prompt enqueue. Only immutable
//! PUBLIC model handles are reused; routes/usage/requests are never retained.
use super::*;

pub(crate) struct PromptLayer<'sources, 'model, 'file> {
    sources: &'sources Sources,
    graph: BoundGraph<'model, 'file>,
    layer: usize,
    prototypes: [Handle; 3],
    // Construction and consumption stay on the original serialized worker.
    _same_worker: std::marker::PhantomData<std::rc::Rc<()>>,
}
// Handles/prototypes replace repeated temporary clones; this bounded bundle
// lives only through enqueue, before recording/submission's separate metadata.
const _: () = assert!(std::mem::size_of::<PromptLayer<'static, 'static, 'static>>() <= 1024);

impl Sources {
    pub(crate) fn prompt_layer<'sources, 'model, 'file>(
        &'sources self,
        graph: BoundGraph<'model, 'file>,
        layer: usize,
    ) -> Result<PromptLayer<'sources, 'model, 'file>> {
        self.validate_prompt_ready(graph)?;
        if layer >= LAYERS {
            return Err(reject());
        }
        // Use the ORIGINAL prototype resolver, including graph/file identity,
        // physical source plan, current loan shadow and registered HOST/cache
        // owner checks. An immutable borrow prevents all Sources mutation,
        // placement publication, table rebuild and loan changes until drop.
        let prototypes = [
            self.prototype(graph, layer, 0)?,
            self.prototype(graph, layer, 1)?,
            self.prototype(graph, layer, 2)?,
        ];
        let stream = self.tables.get(layer).ok_or_else(reject)?.stream;
        if prototypes.iter().any(|handle| handle.stream != stream) {
            return Err(reject());
        }
        Ok(PromptLayer {
            sources: self,
            graph,
            layer,
            prototypes,
            _same_worker: std::marker::PhantomData,
        })
    }
}

impl PromptLayer<'_, '_, '_> {
    pub(crate) fn prototype(&self, role: usize) -> Result<Handle> {
        // Retirement may be set independently through the shared context. Do
        // not cache a valid bit or bypass the actual loan/mutation phase gate.
        self.sources.validate_execution_attached()?;
        self.prototypes.get(role).cloned().ok_or_else(reject)
    }

    pub(crate) fn bind_image(&self, args: KernelArguments) -> Result<KernelArguments> {
        // Preserve the original current table and COMPLETE physical read set
        // on every launch; a cached prototype never substitutes dependency or
        // read-lease authority, nor narrows it to a selected expert.
        self.sources.bind_prompt_image(self.layer, args)
    }

    pub(crate) fn stage_group(
        &self,
        cell: usize,
        role_bytes: [usize; 3],
        destination: &Handle,
        first: usize,
    ) -> Result<()> {
        // Keep every original per-slot guard: live attachment/graph identity,
        // current effective HOST/cache/shadow role, registry/plan instance,
        // actual raw-ring owner/stream/bounds and checked backend copy enqueue.
        self.sources.stage_prompt_group(
            self.graph,
            self.layer,
            cell,
            role_bytes,
            destination,
            first,
        )
    }
}
