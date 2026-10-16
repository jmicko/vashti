//! Immutable public-model registrations; writes still require original rebase.
pub(crate) fn allowed(usage: wgpu::BufferUsages) -> bool {
    let storage = wgpu::BufferUsages::STORAGE;
    usage == storage
        || usage == storage | wgpu::BufferUsages::COPY_SRC
        || usage == storage | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn model_source_copy_usage_is_explicit_and_never_grants_maps_uniforms_or_extra_roles() {
        let s = wgpu::BufferUsages::STORAGE;
        let src = wgpu::BufferUsages::COPY_SRC;
        let dst = wgpu::BufferUsages::COPY_DST;
        for u in [s, s | src, s | src | dst] {
            assert!(allowed(u));
        }
        for u in [
            wgpu::BufferUsages::empty(),
            src,
            dst,
            src | dst,
            s | dst,
            s | src | dst | wgpu::BufferUsages::UNIFORM,
            s | src | wgpu::BufferUsages::MAP_READ,
            s | src | dst | wgpu::BufferUsages::MAP_WRITE,
        ] {
            assert!(!allowed(u));
        }
    }
}
