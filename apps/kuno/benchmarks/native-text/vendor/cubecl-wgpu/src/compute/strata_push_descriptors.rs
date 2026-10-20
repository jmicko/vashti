//! Command-buffer-owned COMPLETE descriptor writes. No resource lifetime or
//! completion authority; the original Recording still retains all real owners.
use ash::vk;
pub(super) const MAX_BINDINGS: usize = 32;
// Both fixed arrays, headers and alignment live concurrently with original
// capture scratch. Charged before any rolling prefix enters a device queue.
pub(super) const STACK_BYTES: usize = MAX_BINDINGS
    * (size_of::<vk::DescriptorBufferInfo>() + size_of::<vk::WriteDescriptorSet<'static>>())
    + 4096;
pub(super) fn selected(enabled: bool, device_max: u32, count: usize) -> bool {
    enabled && (1..=MAX_BINDINGS).contains(&count) && count <= device_max as usize
}
pub(super) fn writes<'a>(
    infos: &'a [vk::DescriptorBufferInfo],
    types: &[vk::DescriptorType],
) -> Result<[vk::WriteDescriptorSet<'a>; MAX_BINDINGS], &'static str> {
    if !selected(true, MAX_BINDINGS as u32, infos.len()) || types.len() != infos.len() {
        return Err("native push complete binding inventory");
    }
    let mut result = [vk::WriteDescriptorSet::default(); MAX_BINDINGS];
    for (index, (info, ty)) in infos.iter().zip(types).enumerate() {
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
