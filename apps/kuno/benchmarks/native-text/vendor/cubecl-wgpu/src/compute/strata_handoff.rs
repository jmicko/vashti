//! Cross-compute-queue visibility for joined native branches and Wgpu uploads.
//! A host fence alone does not provide a Vulkan device memory dependency.
use ash::vk;

pub(super) struct Handoff {
    device: ash::Device,
    timeline: vk::Semaphore,
    fence: vk::Fence,
    epoch: u64,
    pending: bool,
}
impl Handoff {
    pub fn new(device: &ash::Device) -> Result<Self, &'static str> {
        let mut kind = vk::SemaphoreTypeCreateInfo::default()
            .semaphore_type(vk::SemaphoreType::TIMELINE)
            .initial_value(0);
        let timeline = unsafe {
            device.create_semaphore(
                &vk::SemaphoreCreateInfo::default().push_next(&mut kind),
                None,
            )
        }
        .map_err(|_| "native compute handoff semaphore")?;
        let fence = match unsafe { device.create_fence(&vk::FenceCreateInfo::default(), None) } {
            Ok(v) => v,
            Err(_) => {
                unsafe {
                    device.destroy_semaphore(timeline, None);
                }
                return Err("native compute handoff fence");
            }
        };
        Ok(Self {
            device: device.clone(),
            timeline,
            fence,
            epoch: 0,
            pending: false,
        })
    }
    pub fn complete(&mut self) -> Result<(), &'static str> {
        if self.pending {
            unsafe {
                self.device
                    .wait_for_fences(&[self.fence], true, 120_000_000_000)
            }
            .map_err(|_| "native compute handoff completion")?;
            self.pending = false;
        }
        Ok(())
    }
    /// Serialized owning stream only. Both queues belong to this exact device.
    /// Original work/owners remain pinned by Graph through its actual fence.
    pub fn transfer(&mut self, from: vk::Queue, to: vk::Queue) -> Result<(), &'static str> {
        if from == to {
            return Ok(());
        }
        self.complete()?;
        let value = next_epoch(&mut self.epoch)?;
        unsafe { self.device.reset_fences(&[self.fence]) }
            .map_err(|_| "native compute handoff reset")?;
        let semaphores = [self.timeline];
        let values = [value];
        let mut signal_info =
            vk::TimelineSemaphoreSubmitInfo::default().signal_semaphore_values(&values);
        let signal = vk::SubmitInfo::default()
            .signal_semaphores(&semaphores)
            .push_next(&mut signal_info);
        // A partial submission must retain the synchronization objects too.
        self.pending = true;
        unsafe { self.device.queue_submit(from, &[signal], vk::Fence::null()) }
            .map_err(|_| "native compute handoff signal")?;
        let stages = [vk::PipelineStageFlags::ALL_COMMANDS];
        let mut wait_info =
            vk::TimelineSemaphoreSubmitInfo::default().wait_semaphore_values(&values);
        let wait = vk::SubmitInfo::default()
            .wait_semaphores(&semaphores)
            .wait_dst_stage_mask(&stages)
            .push_next(&mut wait_info);
        unsafe { self.device.queue_submit(to, &[wait], self.fence) }
            .map_err(|_| "native compute handoff wait")?;
        Ok(())
    }
}
impl Drop for Handoff {
    fn drop(&mut self) {
        if self.complete().is_err() {
            // Do not destroy semaphore/fence handles with unknown pending use.
            // The protected process/device teardown remains the terminal owner.
            return;
        }
        unsafe {
            self.device.destroy_fence(self.fence, None);
            self.device.destroy_semaphore(self.timeline, None);
        }
    }
}
fn next_epoch(epoch: &mut u64) -> Result<u64, &'static str> {
    let next = epoch
        .checked_add(1)
        .ok_or("native compute handoff epoch exhausted")?;
    *epoch = next;
    Ok(next)
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn actual_handoff_epochs_never_repeat_or_wrap_after_partial_submission() {
        let mut epoch = 0;
        assert_eq!(next_epoch(&mut epoch), Ok(1));
        assert_eq!(next_epoch(&mut epoch), Ok(2));
        epoch = u64::MAX - 1;
        assert_eq!(next_epoch(&mut epoch), Ok(u64::MAX));
        assert!(next_epoch(&mut epoch).is_err());
        assert_eq!(epoch, u64::MAX);
    }
}
