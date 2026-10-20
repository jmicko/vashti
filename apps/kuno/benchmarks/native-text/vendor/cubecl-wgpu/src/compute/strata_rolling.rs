//! One original recording, incrementally submitted while later CLOSED command
//! segments are recorded. Complete owner/plan preflight precedes every prefix.
//! No new flight, prompt authority, host content, queue or completion credit.
use super::{submit::Batches, *};

// NATIVE_ROLLING_LIFECYCLE_BEGIN
const NODE_WINDOW: usize = 128;
const COMMAND_WINDOW: usize = 64;
const MAX_COMMANDS: usize = 32769;

trait Backend {
    fn queue(&self, index: usize) -> Result<usize, &'static str>;
    fn submit(&mut self, index: usize) -> Result<(), &'static str>;
    fn join(&mut self) -> Result<(), &'static str>;
    fn recover(&mut self, touched: [bool; 3]) -> bool;
}

/// Shared by the actual Vulkan path and CPU failure/unwind controls. A closed
/// prefix is submitted in GLOBAL plan order, never grouped ahead by queue.
/// Every segment's timeline waits therefore name already-submitted work.
struct Rolling<B: Backend> {
    backend: B,
    total: usize,
    closed: usize,
    next: usize,
    recorded_nodes: usize,
    submitted_nodes: usize,
    touched: [bool; 3],
    final_queued: bool,
    accepted: bool,
}
impl<B: Backend> Rolling<B> {
    fn new(backend: B, total: usize) -> Result<Self, &'static str> {
        if !(1..=MAX_COMMANDS).contains(&total) {
            return Err("native rolling command capacity");
        }
        Ok(Self {
            backend,
            total,
            closed: 0,
            next: 0,
            recorded_nodes: 0,
            submitted_nodes: 0,
            touched: [false; 3],
            final_queued: false,
            accepted: false,
        })
    }
    fn closed(&mut self, end: usize, nodes: usize, force: bool) -> Result<(), &'static str> {
        if self.final_queued
            || end < self.closed
            || end > self.total
            || nodes < self.recorded_nodes
            || nodes > 32768
        {
            return Err("native rolling closed prefix geometry");
        }
        self.closed = end;
        self.recorded_nodes = nodes;
        if !force && nodes - self.submitted_nodes < NODE_WINDOW && end - self.next < COMMAND_WINDOW
        {
            return Ok(());
        }
        while self.next < self.closed {
            let own = self.backend.queue(self.next)?;
            if own >= self.touched.len() {
                return Err("native rolling original queue index");
            }
            // Even an unsuccessful Vulkan call may have submitted some work.
            // Recovery fences cover every ATTEMPTED queue, not just successes.
            self.touched[own] = true;
            self.backend.submit(self.next)?;
            self.next += 1;
        }
        self.submitted_nodes = nodes;
        Ok(())
    }
    fn finish(&mut self) -> Result<(), &'static str> {
        if self.final_queued || self.closed != self.total || self.next != self.total {
            return Err("native rolling final join before complete recording");
        }
        self.touched[0] = true;
        self.backend.join()?;
        self.final_queued = true;
        Ok(())
    }
    fn accept(mut self) -> Result<(), &'static str> {
        if !self.final_queued {
            return Err("native rolling flight before actual final submission");
        }
        // The Graph has marked its one ORIGINAL matching flight submitted.
        // Its real final fence, not this metadata, owns ordinary completion.
        self.accepted = true;
        Ok(())
    }
}
impl<B: Backend> Drop for Rolling<B> {
    fn drop(&mut self) {
        if !self.accepted && self.touched.iter().any(|&used| used) {
            // Runs on Result error and panic before Graph/Recording owners
            // can be dropped. Failure grants no cleanup/reuse credit: Graph's
            // original partial unsubmitted flight remains quarantined.
            let _joined = self.backend.recover(self.touched);
        }
    }
}

// NATIVE_ROLLING_LIFECYCLE_END

struct Vulkan {
    device: ash::Device,
    queues: [vk::Queue; 3],
    copy: wgpu::hal::vulkan::KunoTransferQueue,
    batches: Batches,
    fence: vk::Fence,
    timelines: [vk::Semaphore; 3],
    base: u64,
    last: [u64; 3],
    recovery: [vk::Fence; 3],
    recovery_pending: bool,
}
impl Backend for Vulkan {
    fn queue(&self, index: usize) -> Result<usize, &'static str> {
        self.batches.record_queue(index)
    }
    fn submit(&mut self, index: usize) -> Result<(), &'static str> {
        let own = self.queue(index)?;
        self.batches
            .with_record(index, |submits| unsafe {
                if own == 1 {
                    self.copy.submit(submits, vk::Fence::null())
                } else {
                    self.device
                        .queue_submit(self.queues[own], submits, vk::Fence::null())
                }
            })?
            .map_err(|_| "native rolling original segment submit")
    }
    fn join(&mut self) -> Result<(), &'static str> {
        super::join_queues(
            &self.device,
            self.queues[0],
            self.timelines,
            self.base,
            self.last,
            self.fence,
        )
    }
    fn recover(&mut self, touched: [bool; 3]) -> bool {
        // No future unsubmitted signal can block this prefix: production
        // Batches preflight proves waits point strictly backward, and Rolling
        // submits every segment in that original global order.
        self.recovery_pending = true;
        let mut submitted = [false; 3];
        let mut okay = true;
        for own in 0..3 {
            if !touched[own] {
                continue;
            }
            let result = unsafe {
                let join = vk::SubmitInfo::default();
                if own == 1 {
                    self.copy.submit(&[join], self.recovery[own])
                } else {
                    self.device
                        .queue_submit(self.queues[own], &[join], self.recovery[own])
                }
            };
            submitted[own] = result.is_ok();
            okay &= result.is_ok();
        }
        // Attempt every actually queued recovery fence even if another queue
        // failed. Never wait on the ordinary final fence before it was queued.
        for (own, &queued) in submitted.iter().enumerate() {
            if queued {
                okay &= unsafe {
                    self.device
                        .wait_for_fences(&[self.recovery[own]], true, 120_000_000_000)
                }
                .is_ok();
            }
        }
        self.recovery_pending = !okay;
        okay
    }
}
impl Drop for Vulkan {
    fn drop(&mut self) {
        if self.recovery_pending {
            // Unknown device failure: the original Graph partial flight pins
            // all resources/pools until protected context/process recovery.
            // These possibly-live recovery fences must be quarantined too.
            return;
        }
        for fence in self.recovery {
            if fence != vk::Fence::null() {
                unsafe { self.device.destroy_fence(fence, None) };
            }
        }
    }
}

pub(crate) struct Submission(Rolling<Vulkan>);
const _: () = assert!(size_of::<Submission>() <= 4096);
impl Submission {
    #[allow(clippy::too_many_arguments)]
    pub(super) fn new(
        device: &ash::Device,
        queues: [vk::Queue; 3],
        copy: &wgpu::hal::vulkan::KunoTransferQueue,
        batches: Batches,
        fence: vk::Fence,
        timelines: [vk::Semaphore; 3],
        base: u64,
        last: [u64; 3],
        total: usize,
    ) -> Result<Self, &'static str> {
        let mut backend = Vulkan {
            device: device.clone(),
            queues,
            copy: copy.clone(),
            batches,
            fence,
            timelines,
            base,
            last,
            recovery: [vk::Fence::null(); 3],
            recovery_pending: false,
        };
        // Every allocation/preflight happens before the first prefix submit.
        for fence in &mut backend.recovery {
            *fence = unsafe { device.create_fence(&vk::FenceCreateInfo::default(), None) }
                .map_err(|_| "native rolling recovery fence allocation")?;
        }
        Ok(Self(Rolling::new(backend, total)?))
    }
    pub fn closed(&mut self, end: usize, nodes: usize, force: bool) -> Result<(), &'static str> {
        self.0.closed(end, nodes, force)
    }
    pub fn finish(&mut self) -> Result<(), &'static str> {
        self.0.finish()
    }
    pub fn accept(self) -> Result<(), &'static str> {
        self.0.accept()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{cell::RefCell, rc::Rc};
    #[derive(Clone, Debug, PartialEq, Eq)]
    enum Event {
        Submit(usize),
        Join,
        Recover([bool; 3]),
    }
    struct Fake {
        events: Rc<RefCell<Vec<Event>>>,
        fail: Option<usize>,
        join_fail: bool,
    }
    impl Backend for Fake {
        fn queue(&self, index: usize) -> Result<usize, &'static str> {
            Ok(index % 3)
        }
        fn submit(&mut self, index: usize) -> Result<(), &'static str> {
            self.events.borrow_mut().push(Event::Submit(index));
            if self.fail == Some(index) {
                Err("injected submission")
            } else {
                Ok(())
            }
        }
        fn join(&mut self) -> Result<(), &'static str> {
            self.events.borrow_mut().push(Event::Join);
            if self.join_fail {
                Err("injected join")
            } else {
                Ok(())
            }
        }
        fn recover(&mut self, touched: [bool; 3]) -> bool {
            self.events.borrow_mut().push(Event::Recover(touched));
            true
        }
    }
    fn make(
        total: usize,
        fail: Option<usize>,
        join_fail: bool,
    ) -> (Rolling<Fake>, Rc<RefCell<Vec<Event>>>) {
        let events = Rc::new(RefCell::new(Vec::new()));
        let backend = Fake {
            events: events.clone(),
            fail,
            join_fail,
        };
        (Rolling::new(backend, total).unwrap(), events)
    }
    #[test]
    fn actual_lifecycle_submits_only_closed_original_prefixes_in_global_order() {
        let (mut rolling, events) = make(140, None, false);
        rolling.closed(1, 0, false).unwrap();
        assert!(events.borrow().is_empty());
        rolling.closed(32, 127, false).unwrap();
        assert!(events.borrow().is_empty());
        rolling.closed(33, 128, false).unwrap();
        assert_eq!(
            *events.borrow(),
            (0..33).map(Event::Submit).collect::<Vec<_>>()
        );
        rolling.closed(97, 129, false).unwrap();
        assert_eq!(events.borrow().len(), 97);
        assert!(rolling.finish().is_err());
        rolling.closed(140, 130, true).unwrap();
        rolling.finish().unwrap();
        rolling.accept().unwrap();
        let mut expected: Vec<_> = (0..140).map(Event::Submit).collect();
        expected.push(Event::Join);
        assert_eq!(*events.borrow(), expected);
    }
    #[test]
    fn every_attempted_prefix_error_gets_recovery_before_return_without_fake_final_join() {
        for failed in 0..9 {
            let (mut rolling, events) = make(9, Some(failed), false);
            assert!(rolling.closed(9, 9, true).is_err());
            drop(rolling);
            let mut expected: Vec<_> = (0..=failed).map(Event::Submit).collect();
            let touched = std::array::from_fn(|q| (0..=failed).any(|i| i % 3 == q));
            expected.push(Event::Recover(touched));
            assert_eq!(*events.borrow(), expected);
        }
    }
    #[test]
    fn panic_after_prefix_and_failure_after_final_join_are_both_recovered() {
        let (mut rolling, events) = make(9, None, false);
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(move || {
            rolling.closed(6, 128, false).unwrap();
            panic!("injected recording panic");
        }));
        assert!(result.is_err());
        assert_eq!(events.borrow().last(), Some(&Event::Recover([true; 3])));
        let (mut rolling, events) = make(3, None, true);
        rolling.closed(3, 3, true).unwrap();
        assert!(rolling.finish().is_err());
        drop(rolling);
        assert_eq!(
            events.borrow().as_slice(),
            &[
                Event::Submit(0),
                Event::Submit(1),
                Event::Submit(2),
                Event::Join,
                Event::Recover([true; 3])
            ]
        );
        let (mut rolling, events) = make(3, None, false);
        rolling.closed(3, 3, true).unwrap();
        rolling.finish().unwrap();
        // An error before the Graph accepts its original flight must drain
        // even when the final fence submit was successful.
        drop(rolling);
        assert_eq!(events.borrow().last(), Some(&Event::Recover([true; 3])));
    }
    #[test]
    fn invalid_inventory_cannot_submit_or_manufacture_completion() {
        for total in [0, MAX_COMMANDS + 1] {
            let events = Rc::new(RefCell::new(Vec::new()));
            assert!(
                Rolling::new(
                    Fake {
                        events: events.clone(),
                        fail: None,
                        join_fail: false
                    },
                    total
                )
                .is_err()
            );
            assert!(events.borrow().is_empty());
        }
        let (mut rolling, events) = make(4, None, false);
        assert!(rolling.closed(5, 1, true).is_err());
        assert!(rolling.closed(1, 32769, true).is_err());
        assert!(rolling.finish().is_err());
        assert!(rolling.accept().is_err());
        assert!(events.borrow().is_empty());
    }
}
