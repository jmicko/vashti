//! Explicit protected GPU lifecycle controls for the real batched Wgpu guard.
use std::{sync::mpsc, time::Duration};

#[test]
#[ignore = "requires protected B70 cgroup, explicit public-only authorization"]
fn actual_batch_blocks_destruction_and_rejects_foreign_destroyed_and_over_budget_roots() {
    assert_eq!(
        std::env::var("KUNO_PUBLIC_BUFFER_GUARD_TEST").as_deref(),
        Ok("1")
    );
    cubecl::future::block_on(async {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::VULKAN,
            ..wgpu::InstanceDescriptor::new_without_display_handle()
        });
        let adapter = instance
            .enumerate_adapters(wgpu::Backends::VULKAN)
            .await
            .into_iter()
            .find(|a| {
                let i = a.get_info();
                i.vendor == 0x8086 && i.device == 0xe223
            })
            .expect("original B70 only");
        let (device, _) = adapter
            .request_device(&wgpu::DeviceDescriptor::default())
            .await
            .unwrap();
        let create = |device: &wgpu::Device| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: None,
                size: 256,
                usage: wgpu::BufferUsages::STORAGE,
                mapped_at_creation: false,
            })
        };
        let buffers = vec![create(&device), create(&device)];
        assert!(
            unsafe { wgpu::Buffer::kuno_as_hal_batch::<wgpu::hal::api::Vulkan>(&[], 65536) }
                .is_none()
        );
        assert!(
            unsafe { wgpu::Buffer::kuno_as_hal_batch::<wgpu::hal::api::Vulkan>(&buffers, 0) }
                .is_none()
        );
        let batch =
            unsafe { wgpu::Buffer::kuno_as_hal_batch::<wgpu::hal::api::Vulkan>(&buffers, 65536) }
                .unwrap();
        assert!(batch.get(0).is_some() && batch.get(1).is_some() && batch.get(2).is_none());
        assert!(batch.metadata_bytes().unwrap() <= 65536);
        // Begin an actual competing writer AFTER the guard has acquired ONE
        // device lock. This would reveal nested-reader/writer deadlocks.
        let first = buffers[0].clone();
        let (began_tx, began_rx) = mpsc::channel();
        let (done_tx, done_rx) = mpsc::channel();
        let writer = std::thread::spawn(move || {
            began_tx.send(()).unwrap();
            first.destroy();
            done_tx.send(()).unwrap();
        });
        began_rx.recv_timeout(Duration::from_secs(10)).unwrap();
        assert!(matches!(
            done_rx.recv_timeout(Duration::from_millis(100)),
            Err(mpsc::RecvTimeoutError::Timeout)
        ));
        assert!(batch.get(1).is_some());
        drop(batch);
        done_rx.recv_timeout(Duration::from_secs(10)).unwrap();
        writer.join().unwrap();
        assert!(
            unsafe { wgpu::Buffer::kuno_as_hal_batch::<wgpu::hal::api::Vulkan>(&buffers, 65536) }
                .is_none()
        );
        // A failed mixed-device lookup must release everything and leave the
        // valid resource immediately usable by a later original single lookup.
        let (other, _) = adapter
            .request_device(&wgpu::DeviceDescriptor::default())
            .await
            .unwrap();
        let mixed = [buffers[1].clone(), create(&other)];
        assert!(
            unsafe { wgpu::Buffer::kuno_as_hal_batch::<wgpu::hal::api::Vulkan>(&mixed, 65536) }
                .is_none()
        );
        assert!(unsafe { buffers[1].as_hal::<wgpu::hal::api::Vulkan>() }.is_some());
        drop(mixed);
        drop(buffers);
        device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
        other.poll(wgpu::PollType::wait_indefinitely()).unwrap();
    });
}
