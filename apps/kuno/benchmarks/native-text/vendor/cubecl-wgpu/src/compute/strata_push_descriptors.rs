//! Command-buffer-owned COMPLETE descriptor writes. No resource lifetime or
//! completion authority; the original Recording still retains all real owners.
use ash::vk;
pub(super) const MAX_BINDINGS: usize = 32;
// Both fixed arrays, headers and alignment live concurrently with original
// capture scratch. Charged before any rolling prefix enters a device queue.
pub(super) const STACK_BYTES: usize = MAX_BINDINGS
    * (size_of::<vk::DescriptorBufferInfo>() + size_of::<vk::WriteDescriptorSet<'static>>())
    + MAX_BINDINGS * size_of::<vk::DescriptorUpdateTemplateEntry>()
    + 4096;
pub(super) fn templates_supported(instance_api: u32, device_api: u32, entry_points: bool) -> bool {
    instance_api >= vk::API_VERSION_1_1 && device_api >= vk::API_VERSION_1_1 && entry_points
}
pub(super) fn selected(enabled: bool, device_max: u32, count: usize) -> bool {
    enabled && (1..=MAX_BINDINGS).contains(&count) && count <= device_max as usize
}
/// A template contains only immutable layout metadata, never a buffer or a
/// pointer into a dispatch. The command consumes a full original infos slice.
pub(super) fn template_entries(
    types: &[vk::DescriptorType],
) -> Result<[vk::DescriptorUpdateTemplateEntry; MAX_BINDINGS], &'static str> {
    if !selected(true, MAX_BINDINGS as u32, types.len())
        || types.iter().any(|ty| {
            !matches!(
                *ty,
                vk::DescriptorType::STORAGE_BUFFER | vk::DescriptorType::UNIFORM_BUFFER
            )
        })
    {
        return Err("native push template complete binding inventory");
    }
    let mut entries = [vk::DescriptorUpdateTemplateEntry::default(); MAX_BINDINGS];
    for (i, ty) in types.iter().enumerate() {
        entries[i] = vk::DescriptorUpdateTemplateEntry::default()
            .dst_binding(i as u32)
            .dst_array_element(0)
            .descriptor_count(1)
            .descriptor_type(*ty)
            .offset(i * size_of::<vk::DescriptorBufferInfo>())
            .stride(size_of::<vk::DescriptorBufferInfo>());
    }
    Ok(entries)
}
pub(super) fn validate_infos(
    infos: &[vk::DescriptorBufferInfo],
    types: &[vk::DescriptorType],
) -> Result<(), &'static str> {
    if !selected(true, MAX_BINDINGS as u32, infos.len()) || types.len() != infos.len() {
        return Err("native push complete binding inventory");
    }
    for (info, ty) in infos.iter().zip(types) {
        if info.buffer == vk::Buffer::null()
            || info.range == 0
            || info.offset.checked_add(info.range).is_none()
            || !matches!(
                *ty,
                vk::DescriptorType::STORAGE_BUFFER | vk::DescriptorType::UNIFORM_BUFFER
            )
        {
            return Err("native push original binding view/type");
        }
    }
    Ok(())
}
pub(super) fn writes<'a>(
    infos: &'a [vk::DescriptorBufferInfo],
    types: &[vk::DescriptorType],
) -> Result<[vk::WriteDescriptorSet<'a>; MAX_BINDINGS], &'static str> {
    validate_infos(infos, types)?;
    let mut result = [vk::WriteDescriptorSet::default(); MAX_BINDINGS];
    for (index, (info, ty)) in infos.iter().zip(types).enumerate() {
        result[index] = vk::WriteDescriptorSet::default()
            .dst_binding(index as u32)
            .descriptor_type(*ty)
            .buffer_info(std::slice::from_ref(info));
    }
    Ok(result)
}
#[cfg(test)]
mod tests {
    use super::*;
    use ash::vk::Handle;
    #[test]
    fn push_template_requires_effective_core_version_and_every_actual_entry_point() {
        for instance in [
            vk::API_VERSION_1_0,
            vk::API_VERSION_1_1,
            vk::API_VERSION_1_3,
        ] {
            for device in [
                vk::API_VERSION_1_0,
                vk::API_VERSION_1_1,
                vk::API_VERSION_1_3,
            ] {
                for entries in [false, true] {
                    assert_eq!(
                        templates_supported(instance, device, entries),
                        instance >= vk::API_VERSION_1_1 && device >= vk::API_VERSION_1_1 && entries
                    );
                }
            }
        }
    }
    #[test]
    fn template_entries_interpret_the_exact_original_full_views_and_types() {
        for count in [1, 9, MAX_BINDINGS] {
            let infos: Vec<_> = (0..count)
                .map(|i| {
                    vk::DescriptorBufferInfo::default()
                        .buffer(vk::Buffer::from_raw((i / 2 + 1) as u64))
                        .offset((2048 * i) as u64)
                        .range((8192 + i) as u64)
                })
                .collect();
            let types: Vec<_> = (0..count)
                .map(|i| {
                    if i + 1 == count {
                        vk::DescriptorType::UNIFORM_BUFFER
                    } else {
                        vk::DescriptorType::STORAGE_BUFFER
                    }
                })
                .collect();
            validate_infos(&infos, &types).unwrap();
            let original = writes(&infos, &types).unwrap();
            let entries = template_entries(&types).unwrap();
            for (i, entry) in entries[..count].iter().enumerate() {
                let old = &original[i];
                assert_eq!(entry.dst_binding, old.dst_binding);
                assert_eq!(entry.dst_array_element, old.dst_array_element);
                assert_eq!(entry.descriptor_count, old.descriptor_count);
                assert_eq!(entry.descriptor_type, old.descriptor_type);
                assert_eq!(entry.stride, size_of::<vk::DescriptorBufferInfo>());
                assert_eq!(entry.offset, i * size_of::<vk::DescriptorBufferInfo>());
                let actual = unsafe {
                    &*infos
                        .as_ptr()
                        .cast::<u8>()
                        .add(entry.offset)
                        .cast::<vk::DescriptorBufferInfo>()
                };
                let expected = unsafe { &*old.p_buffer_info };
                assert_eq!(actual.buffer, expected.buffer);
                assert_eq!(actual.offset, expected.offset);
                assert_eq!(actual.range, expected.range);
            }
            assert!(entries[count..].iter().all(|e| e.descriptor_count == 0));
        }
    }
    #[test]
    fn template_inventory_and_original_complete_write_validation_reject_the_same_bad_inputs() {
        assert!(template_entries(&[]).is_err());
        assert!(template_entries(&[vk::DescriptorType::STORAGE_BUFFER; 33]).is_err());
        for ty in [
            vk::DescriptorType::STORAGE_BUFFER_DYNAMIC,
            vk::DescriptorType::SAMPLED_IMAGE,
            vk::DescriptorType::SAMPLER,
        ] {
            assert!(template_entries(&[ty]).is_err());
        }
        let good = vk::DescriptorBufferInfo::default()
            .buffer(vk::Buffer::from_raw(1))
            .offset(2048)
            .range(8192);
        for bad in [
            good.buffer(vk::Buffer::null()),
            good.range(0),
            good.offset(u64::MAX),
        ] {
            assert!(
                validate_infos(&[good, bad], &[vk::DescriptorType::STORAGE_BUFFER; 2]).is_err()
            );
            assert!(writes(&[good, bad], &[vk::DescriptorType::STORAGE_BUFFER; 2]).is_err());
        }
    }
    #[test]
    fn unsupported_or_oversized_layout_keeps_the_fresh_descriptor_path() {
        for enabled in [false, true] {
            for max in [0, 1, 8, 16, 32, 64] {
                for count in 0..=33 {
                    assert_eq!(
                        selected(enabled, max, count),
                        enabled && count > 0 && count <= 32 && count <= max as usize
                    );
                }
            }
        }
        assert!(STACK_BYTES < 16 * 1024);
    }
    #[test]
    fn actual_packer_retains_every_full_view_type_offset_and_binding_order() {
        for count in [1, 9, 32] {
            let infos: Vec<_> = (0..count)
                .map(|i| {
                    vk::DescriptorBufferInfo::default()
                        .buffer(vk::Buffer::from_raw((i / 2 + 1) as u64))
                        .offset((i * 2048) as u64)
                        .range((4096 + i) as u64)
                })
                .collect();
            let types: Vec<_> = (0..count)
                .map(|i| {
                    if i + 1 == count {
                        vk::DescriptorType::UNIFORM_BUFFER
                    } else {
                        vk::DescriptorType::STORAGE_BUFFER
                    }
                })
                .collect();
            let actual = writes(&infos, &types).unwrap();
            for (i, w) in actual[..count].iter().enumerate() {
                assert_eq!(w.dst_binding, i as u32);
                assert_eq!(w.dst_array_element, 0);
                assert_eq!(w.dst_set, vk::DescriptorSet::null());
                assert_eq!(w.descriptor_count, 1);
                assert_eq!(w.descriptor_type, types[i]);
                assert_eq!(w.p_buffer_info, std::ptr::from_ref(&infos[i]));
                let v = unsafe { &*w.p_buffer_info };
                assert_eq!(v.buffer, infos[i].buffer);
                assert_eq!(v.offset, infos[i].offset);
                assert_eq!(v.range, infos[i].range);
            }
            assert!(actual[count..].iter().all(|w| w.descriptor_count == 0));
        }
    }
    #[test]
    fn malformed_complete_views_cannot_encode_partial_push_state() {
        let valid = vk::DescriptorBufferInfo::default()
            .buffer(vk::Buffer::from_raw(5))
            .offset(2048)
            .range(8192);
        assert!(writes(&[], &[]).is_err());
        assert!(writes(&[valid], &[]).is_err());
        assert!(writes(&[valid; 33], &[vk::DescriptorType::STORAGE_BUFFER; 33]).is_err());
        assert!(writes(&[valid], &[vk::DescriptorType::STORAGE_BUFFER_DYNAMIC]).is_err());
        for invalid in [
            valid.buffer(vk::Buffer::null()),
            valid.range(0),
            valid.offset(u64::MAX),
        ] {
            assert!(writes(&[valid, invalid], &[vk::DescriptorType::STORAGE_BUFFER; 2]).is_err());
        }
    }
}
