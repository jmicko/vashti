//! Exact HAL usage mappings for the actual owning command-pool family.
use ash::vk;
use wgpu_types as wgt;

fn queue_shader_stages(queue_flags: vk::QueueFlags) -> vk::PipelineStageFlags {
    let mut stages = vk::PipelineStageFlags::empty();
    if queue_flags.contains(vk::QueueFlags::GRAPHICS) {
        stages |= vk::PipelineStageFlags::VERTEX_SHADER | vk::PipelineStageFlags::FRAGMENT_SHADER;
    }
    if queue_flags.contains(vk::QueueFlags::COMPUTE) {
        stages |= vk::PipelineStageFlags::COMPUTE_SHADER;
    }
    stages
}

pub fn map_texture_usage_to_barrier(
    usage: wgt::TextureUses,
    queue_flags: vk::QueueFlags,
) -> (vk::PipelineStageFlags, vk::AccessFlags) {
    let mut stages = vk::PipelineStageFlags::empty();
    let mut access = vk::AccessFlags::empty();
    let shader_stages = queue_shader_stages(queue_flags);
    // These uses cannot be made legal by deleting their graphics stages.
    // The headless compute device does not admit render/present operations.
    assert!(
        queue_flags.contains(vk::QueueFlags::GRAPHICS)
            || !usage.intersects(
                wgt::TextureUses::COLOR_TARGET
                    | wgt::TextureUses::DEPTH_STENCIL_READ
                    | wgt::TextureUses::DEPTH_STENCIL_WRITE
                    | wgt::TextureUses::PRESENT
            ),
        "graphics texture usage on non-graphics queue"
    );
    assert!(
        !shader_stages.is_empty()
            || !usage.intersects(
                wgt::TextureUses::RESOURCE
                    | wgt::TextureUses::STORAGE_READ_ONLY
                    | wgt::TextureUses::STORAGE_WRITE_ONLY
                    | wgt::TextureUses::STORAGE_READ_WRITE
                    | wgt::TextureUses::STORAGE_ATOMIC
            ),
        "shader texture usage on non-shader queue"
    );

    if usage.contains(wgt::TextureUses::COPY_SRC) {
        stages |= vk::PipelineStageFlags::TRANSFER;
        access |= vk::AccessFlags::TRANSFER_READ;
    }
    if usage.contains(wgt::TextureUses::COPY_DST) {
        stages |= vk::PipelineStageFlags::TRANSFER;
        access |= vk::AccessFlags::TRANSFER_WRITE;
    }
    if usage.contains(wgt::TextureUses::RESOURCE) {
        stages |= shader_stages;
        access |= vk::AccessFlags::SHADER_READ;
    }
    if usage.contains(wgt::TextureUses::COLOR_TARGET) {
        stages |= vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT;
        access |= vk::AccessFlags::COLOR_ATTACHMENT_READ | vk::AccessFlags::COLOR_ATTACHMENT_WRITE;
    }
    if usage.intersects(wgt::TextureUses::DEPTH_STENCIL_READ) {
        stages |= vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS
            | vk::PipelineStageFlags::LATE_FRAGMENT_TESTS;
        access |= vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_READ;
    }
    if usage.intersects(wgt::TextureUses::DEPTH_STENCIL_WRITE) {
        stages |= vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS
            | vk::PipelineStageFlags::LATE_FRAGMENT_TESTS;
        access |= vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_READ
            | vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE;
    }
    if usage.intersects(
        wgt::TextureUses::STORAGE_READ_ONLY
            | wgt::TextureUses::STORAGE_READ_WRITE
            | wgt::TextureUses::STORAGE_ATOMIC,
    ) {
        stages |= shader_stages;
        access |= vk::AccessFlags::SHADER_READ;
    }
    if usage.intersects(
        wgt::TextureUses::STORAGE_WRITE_ONLY
            | wgt::TextureUses::STORAGE_READ_WRITE
            | wgt::TextureUses::STORAGE_ATOMIC,
    ) {
        stages |= shader_stages;
        access |= vk::AccessFlags::SHADER_WRITE;
    }

    if usage == wgt::TextureUses::UNINITIALIZED || usage == wgt::TextureUses::PRESENT {
        (
            vk::PipelineStageFlags::TOP_OF_PIPE,
            vk::AccessFlags::empty(),
        )
    } else {
        (stages, access)
    }
}

pub fn map_buffer_usage_to_barrier(
    usage: wgt::BufferUses,
    queue_flags: vk::QueueFlags,
) -> (vk::PipelineStageFlags, vk::AccessFlags) {
    let mut stages = vk::PipelineStageFlags::empty();
    let mut access = vk::AccessFlags::empty();
    let shader_stages = queue_shader_stages(queue_flags);
    assert!(
        queue_flags.contains(vk::QueueFlags::GRAPHICS)
            || !usage.intersects(wgt::BufferUses::INDEX | wgt::BufferUses::VERTEX),
        "graphics buffer usage on non-graphics queue"
    );
    assert!(
        !shader_stages.is_empty()
            || !usage.intersects(
                wgt::BufferUses::UNIFORM
                    | wgt::BufferUses::STORAGE_READ_ONLY
                    | wgt::BufferUses::STORAGE_READ_WRITE
                    | wgt::BufferUses::INDIRECT
            ),
        "shader/indirect buffer usage on non-shader queue"
    );

    if usage.contains(wgt::BufferUses::MAP_READ) {
        stages |= vk::PipelineStageFlags::HOST;
        access |= vk::AccessFlags::HOST_READ;
    }
    if usage.contains(wgt::BufferUses::MAP_WRITE) {
        stages |= vk::PipelineStageFlags::HOST;
        access |= vk::AccessFlags::HOST_WRITE;
    }
    if usage.contains(wgt::BufferUses::COPY_SRC) {
        stages |= vk::PipelineStageFlags::TRANSFER;
        access |= vk::AccessFlags::TRANSFER_READ;
    }
    if usage.contains(wgt::BufferUses::COPY_DST) {
        stages |= vk::PipelineStageFlags::TRANSFER;
        access |= vk::AccessFlags::TRANSFER_WRITE;
    }
    if usage.contains(wgt::BufferUses::UNIFORM) {
        stages |= shader_stages;
        access |= vk::AccessFlags::UNIFORM_READ;
    }
    if usage.intersects(wgt::BufferUses::STORAGE_READ_ONLY) {
        stages |= shader_stages;
        access |= vk::AccessFlags::SHADER_READ;
    }
    if usage.intersects(wgt::BufferUses::STORAGE_READ_WRITE) {
        stages |= shader_stages;
        access |= vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE;
    }
    if usage.contains(wgt::BufferUses::INDEX) {
        stages |= vk::PipelineStageFlags::VERTEX_INPUT;
        access |= vk::AccessFlags::INDEX_READ;
    }
    if usage.contains(wgt::BufferUses::VERTEX) {
        stages |= vk::PipelineStageFlags::VERTEX_INPUT;
        access |= vk::AccessFlags::VERTEX_ATTRIBUTE_READ;
    }
    if usage.contains(wgt::BufferUses::INDIRECT) {
        stages |= vk::PipelineStageFlags::DRAW_INDIRECT;
        access |= vk::AccessFlags::INDIRECT_COMMAND_READ;
    }
    if usage.intersects(
        wgt::BufferUses::BOTTOM_LEVEL_ACCELERATION_STRUCTURE_INPUT
            | wgt::BufferUses::TOP_LEVEL_ACCELERATION_STRUCTURE_INPUT
            | wgt::BufferUses::ACCELERATION_STRUCTURE_SCRATCH,
    ) {
        stages |= vk::PipelineStageFlags::ACCELERATION_STRUCTURE_BUILD_KHR;
        access |= vk::AccessFlags::ACCELERATION_STRUCTURE_READ_KHR
            | vk::AccessFlags::ACCELERATION_STRUCTURE_WRITE_KHR
            | vk::AccessFlags::SHADER_READ;
    }
    if usage.contains(wgt::BufferUses::ACCELERATION_STRUCTURE_QUERY) {
        stages |= vk::PipelineStageFlags::TRANSFER;
        access |= vk::AccessFlags::TRANSFER_WRITE;
    }

    (stages, access)
}
#[cfg(test)]
mod kuno_queue_barrier_tests {
    use super::*;

    #[test]
    fn shader_stages_follow_actual_queue_capabilities() {
        let c = vk::QueueFlags::COMPUTE;
        let g = vk::QueueFlags::GRAPHICS;
        let s = vk::PipelineStageFlags::COMPUTE_SHADER;
        let graphics =
            vk::PipelineStageFlags::VERTEX_SHADER | vk::PipelineStageFlags::FRAGMENT_SHADER;
        assert_eq!(queue_shader_stages(c), s);
        assert_eq!(queue_shader_stages(c | g), s | graphics);
        assert_eq!(queue_shader_stages(g), graphics);
        assert!(queue_shader_stages(vk::QueueFlags::TRANSFER).is_empty());
    }

    #[test]
    fn compute_buffer_barriers_preserve_accesses_host_transfer_and_indirect_stages() {
        use vk::{AccessFlags as A, PipelineStageFlags as S};
        use wgt::BufferUses as U;
        let c = vk::QueueFlags::COMPUTE | vk::QueueFlags::TRANSFER;
        for (usage, stages, access) in [
            (U::MAP_READ, S::HOST, A::HOST_READ),
            (U::MAP_WRITE, S::HOST, A::HOST_WRITE),
            (U::COPY_SRC, S::TRANSFER, A::TRANSFER_READ),
            (U::COPY_DST, S::TRANSFER, A::TRANSFER_WRITE),
            (U::UNIFORM, S::COMPUTE_SHADER, A::UNIFORM_READ),
            (U::STORAGE_READ_ONLY, S::COMPUTE_SHADER, A::SHADER_READ),
            (
                U::STORAGE_READ_WRITE,
                S::COMPUTE_SHADER,
                A::SHADER_READ | A::SHADER_WRITE,
            ),
            (U::INDIRECT, S::DRAW_INDIRECT, A::INDIRECT_COMMAND_READ),
            (
                U::ACCELERATION_STRUCTURE_QUERY,
                S::TRANSFER,
                A::TRANSFER_WRITE,
            ),
            (
                U::COPY_DST | U::STORAGE_READ_WRITE | U::MAP_READ | U::INDIRECT,
                S::TRANSFER | S::COMPUTE_SHADER | S::HOST | S::DRAW_INDIRECT,
                A::TRANSFER_WRITE
                    | A::SHADER_READ
                    | A::SHADER_WRITE
                    | A::HOST_READ
                    | A::INDIRECT_COMMAND_READ,
            ),
        ] {
            assert_eq!(map_buffer_usage_to_barrier(usage, c), (stages, access));
            // The original graphics+compute mapping has identical accesses;
            // only its shader stage population is broader.
            let (gs, ga) = map_buffer_usage_to_barrier(usage, c | vk::QueueFlags::GRAPHICS);
            assert_eq!(ga, access);
            let expected = if stages.contains(S::COMPUTE_SHADER) {
                stages | S::VERTEX_SHADER | S::FRAGMENT_SHADER
            } else {
                stages
            };
            assert_eq!(gs, expected);
        }
    }

    #[test]
    fn compute_texture_barriers_keep_exact_accesses_and_original_initialization() {
        use vk::{AccessFlags as A, PipelineStageFlags as S};
        use wgt::TextureUses as U;
        let c = vk::QueueFlags::COMPUTE | vk::QueueFlags::TRANSFER;
        for (usage, stages, access) in [
            (U::COPY_SRC, S::TRANSFER, A::TRANSFER_READ),
            (U::COPY_DST, S::TRANSFER, A::TRANSFER_WRITE),
            (U::RESOURCE, S::COMPUTE_SHADER, A::SHADER_READ),
            (U::STORAGE_READ_ONLY, S::COMPUTE_SHADER, A::SHADER_READ),
            (U::STORAGE_WRITE_ONLY, S::COMPUTE_SHADER, A::SHADER_WRITE),
            (
                U::STORAGE_READ_WRITE,
                S::COMPUTE_SHADER,
                A::SHADER_READ | A::SHADER_WRITE,
            ),
            (
                U::STORAGE_ATOMIC,
                S::COMPUTE_SHADER,
                A::SHADER_READ | A::SHADER_WRITE,
            ),
            (U::UNINITIALIZED, S::TOP_OF_PIPE, A::empty()),
        ] {
            assert_eq!(map_texture_usage_to_barrier(usage, c), (stages, access));
            let (gs, ga) = map_texture_usage_to_barrier(usage, c | vk::QueueFlags::GRAPHICS);
            assert_eq!(ga, access);
            let expected = if stages.contains(S::COMPUTE_SHADER) {
                stages | S::VERTEX_SHADER | S::FRAGMENT_SHADER
            } else {
                stages
            };
            assert_eq!(gs, expected);
        }
    }

    #[test]
    fn graphics_only_accesses_are_rejected_on_compute_not_silently_removed() {
        let c = vk::QueueFlags::COMPUTE;
        let g = c | vk::QueueFlags::GRAPHICS;
        for usage in [wgt::BufferUses::INDEX, wgt::BufferUses::VERTEX] {
            assert!(std::panic::catch_unwind(|| map_buffer_usage_to_barrier(usage, c)).is_err());
            let (s, a) = map_buffer_usage_to_barrier(usage, g);
            assert_eq!(s, vk::PipelineStageFlags::VERTEX_INPUT);
            assert!(!a.is_empty());
        }
        for usage in [
            wgt::TextureUses::COLOR_TARGET,
            wgt::TextureUses::DEPTH_STENCIL_READ,
            wgt::TextureUses::DEPTH_STENCIL_WRITE,
            wgt::TextureUses::PRESENT,
        ] {
            assert!(std::panic::catch_unwind(|| map_texture_usage_to_barrier(usage, c)).is_err());
            let (s, _) = map_texture_usage_to_barrier(usage, g);
            assert!(!s.is_empty());
        }
        assert!(
            std::panic::catch_unwind(|| map_buffer_usage_to_barrier(
                wgt::BufferUses::STORAGE_READ_ONLY,
                vk::QueueFlags::TRANSFER
            ))
            .is_err()
        );
        assert!(
            std::panic::catch_unwind(|| map_texture_usage_to_barrier(
                wgt::TextureUses::RESOURCE,
                vk::QueueFlags::TRANSFER
            ))
            .is_err()
        );
    }
}
