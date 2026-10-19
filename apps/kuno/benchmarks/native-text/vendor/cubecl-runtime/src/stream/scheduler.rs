use crate::{
    config::streaming::StreamingLogLevel,
    logging::ServerLogger,
    stream::{StreamFactory, StreamPool},
};
use alloc::{format, sync::Arc, vec, vec::Vec};
use cubecl_common::stream_id::StreamId;

/// Defines a trait for a scheduler stream backend, specifying the types and behavior for task scheduling.
pub trait SchedulerStreamBackend {
    /// Type representing a task.
    type Task: core::fmt::Debug;
    /// Type representing a stream.
    type Stream: core::fmt::Debug;
    /// Type for the stream factory, which creates streams of type `Self::Stream`.
    type Factory: StreamFactory<Stream = Self::Stream>;

    /// Enqueues a task onto a given stream for execution.
    fn enqueue(task: Self::Task, stream: &mut Self::Stream);
    /// Preserve a complete already-owned single-stream task order. Backends may
    /// amortize synchronous bookkeeping; the default keeps individual enqueue.
    fn enqueue_many(tasks: impl Iterator<Item = Self::Task>, stream: &mut Self::Stream) {
        for task in tasks {
            Self::enqueue(task, stream);
        }
    }
    /// Flush the inner stream queue to ensure ordering between different streams.
    fn flush(stream: &mut Self::Stream);
    /// Returns a mutable reference to the stream factory.
    fn factory(&mut self) -> &mut Self::Factory;
}

/// Represents a multi-stream scheduler that manages task execution across multiple streams.
#[derive(Debug)]
pub struct SchedulerMultiStream<B: SchedulerStreamBackend> {
    /// Pool of streams managed by the scheduler.
    pool: StreamPool<SchedulerPoolMarker<B>>,
    /// Strategy for scheduling tasks (e.g., Interleave or Sequential).
    strategy: SchedulerStrategy,
    /// Maximum number of tasks allowed per stream before execution is triggered.
    max_tasks: usize,
    /// Server logger.
    pub logger: Arc<ServerLogger>,
}

/// Defines the scheduling strategy for task execution.
#[derive(Debug)]
pub enum SchedulerStrategy {
    /// Tasks from different streams are interleaved during execution.
    Interleave,
    /// Tasks from each stream are executed sequentially.
    Sequential,
}

/// Represents a single stream that holds tasks and a backend stream.
#[derive(Debug)]
pub struct Stream<B: SchedulerStreamBackend> {
    /// List of tasks queued for execution in this stream.
    tasks: Vec<B::Task>,
    /// The backend stream used for task execution.
    stream: B::Stream,
}

impl<B: SchedulerStreamBackend> Stream<B> {
    /// Flushes all tasks from the stream, returning them and clearing the internal task list.
    fn flush(&mut self) -> Vec<B::Task> {
        let mut returned = Vec::with_capacity(self.tasks.capacity());
        core::mem::swap(&mut returned, &mut self.tasks);
        returned
    }
}

#[derive(Debug)]
struct SchedulerPoolMarker<B: SchedulerStreamBackend> {
    backend: B,
}

impl<B: SchedulerStreamBackend> StreamFactory for SchedulerPoolMarker<B> {
    // The type of stream produced by this factory.
    type Stream = Stream<B>;

    // Creates a new stream with an empty task list and a backend stream.
    fn create(&mut self) -> Self::Stream {
        Stream {
            tasks: Vec::new(),
            // Uses the backend's factory to create a new stream.
            stream: self.backend.factory().create(),
        }
    }
}

/// Options for configuring a `SchedulerMultiStream`.
#[derive(Debug)]
pub struct SchedulerMultiStreamOptions {
    /// Maximum number of streams allowed in the pool.
    pub max_streams: u8,
    /// Maximum number of tasks per stream before execution is triggered.
    pub max_tasks: usize,
    /// The scheduling strategy to use.
    pub strategy: SchedulerStrategy,
}

impl<B: SchedulerStreamBackend> SchedulerMultiStream<B> {
    /// Creates a new `SchedulerMultiStream` with the given backend and options.
    pub fn new(
        logger: Arc<ServerLogger>,
        backend: B,
        options: SchedulerMultiStreamOptions,
    ) -> Self {
        Self {
            pool: StreamPool::new(SchedulerPoolMarker { backend }, options.max_streams, 0),
            max_tasks: options.max_tasks,
            strategy: options.strategy,
            logger,
        }
    }

    /// Returns a mutable reference to the backend stream for a given stream ID.
    pub fn stream(&mut self, stream_id: &StreamId) -> &mut B::Stream {
        let stream = self.pool.get_mut(stream_id);
        &mut stream.stream
    }

    /// Registers a task for execution on a specific stream, ensuring stream alignment.
    pub fn register(&mut self, stream_id: StreamId, task: B::Task, args_streams: &[StreamId]) {
        // Align streams to ensure dependencies are handled correctly.
        self.align_streams(stream_id, args_streams);

        // Get the stream for the given stream ID and add the task to its queue.
        let current = self.pool.get_mut(&stream_id);
        current.tasks.push(task);

        // If the task queue exceeds the maximum, execute the stream.
        if current.tasks.len() >= self.max_tasks {
            self.execute_streams(vec![stream_id]);
        }
    }

    /// Aligns streams by flushing tasks from streams that conflict with the given bindings.
    pub(crate) fn align_streams(&mut self, stream_id: StreamId, args_streams: &[StreamId]) {
        let mut to_flush = Vec::new();
        // Get the index of the target stream.
        let index = self.pool.stream_index(&stream_id);

        // Identify streams that need to be flushed due to conflicting bindings.
        for arg_stream in args_streams {
            let index_stream = self.pool.stream_index(arg_stream);
            if index != index_stream {
                to_flush.push(*arg_stream);

                self.logger.log_streaming(
                    |level| matches!(level, StreamingLogLevel::Full),
                    || format!("Binding on {} is shared on {}", arg_stream, stream_id),
                );
            }
        }

        // If no streams need flushing, return early.
        if to_flush.is_empty() {
            return;
        }

        self.logger.log_streaming(
            |level| !matches!(level, StreamingLogLevel::Disabled),
            || {
                format!(
                    "Flushing streams {to_flush:?} before registering more tasks on {stream_id}"
                )
            },
        );
        // Execute the streams that need to be flushed.
        self.execute_streams(to_flush);
    }

    /// Materialize one already aligned stream's tasks in their original order,
    /// without asking the backend to submit its command graph. This grants no
    /// completion. Only a backend that tracks subsequent same-stream commands
    /// and owns their dependency/final submission may use this queue boundary.
    pub fn enqueue_stream_tasks(&mut self, stream_id: StreamId) {
        let stream = self.pool.get_mut(&stream_id);
        let tasks = stream.flush();
        B::enqueue_many(tasks.into_iter(), &mut stream.stream);
    }

    /// Execute and submit the selected streams, preserving cross-stream ordering.
    pub fn execute_streams(&mut self, stream_ids: Vec<StreamId>) {
        let mut indices = Vec::with_capacity(stream_ids.len());

        // Collect unique stream indices to avoid redundant processing.
        for id in stream_ids {
            let index = self.pool.stream_index(&id);
            if !indices.contains(&index) {
                indices.push(index);
            }
        }

        // Create schedules for each stream to be executed.
        let mut schedules = Vec::new();
        for index in indices {
            let stream = unsafe { self.pool.get_mut_index(index) }; // Note: `unsafe` usage assumes valid index.
            let tasks = stream.flush();
            let num_tasks = tasks.len();

            schedules.push(Schedule {
                tasks: tasks.into_iter(),
                num_tasks,
                stream_index: index,
            });
        }

        // If no schedules were created, return early.
        if schedules.is_empty() {
            return;
        }

        // Execute schedules based on the configured strategy.
        match self.strategy {
            SchedulerStrategy::Interleave => self.execute_schedules_interleave(schedules),
            SchedulerStrategy::Sequential => self.execute_schedules_sequence(schedules),
        }
    }

    /// Executes schedules sequentially, processing each stream's tasks in order.
    fn execute_schedules_sequence(&mut self, schedules: Vec<Schedule<B>>) {
        for schedule in schedules {
            let stream = unsafe { self.pool.get_mut_index(schedule.stream_index) }; // Note: `unsafe` usage assumes valid index.
            B::enqueue_many(schedule.tasks, &mut stream.stream);

            // Makes sure the tasks are ordered on the compute queue.
            B::flush(&mut stream.stream);
        }
    }

    //// Executes schedules in an interleaved manner, alternating tasks from different streams.
    ///
    /// We chose the first stream as the one executing the tasks, ensuring proper ordering by
    /// flushing all other streams first and flushing the execution stream at the end.
    /// This way, we ensure that most tasks are actually interleaved on the real compute queue
    /// shared across all streams.
    fn execute_schedules_interleave(&mut self, mut schedules: Vec<Schedule<B>>) {
        if schedules.len() == 1 {
            let schedule = schedules.pop().unwrap();
            let stream = unsafe { self.pool.get_mut_index(schedule.stream_index) };
            B::enqueue_many(schedule.tasks, &mut stream.stream);
            B::flush(&mut stream.stream);
            return;
        }
        // Makes sure the tasks are ordered on the compute queue.
        for schedule in schedules.iter_mut().skip(1) {
            let stream = unsafe { self.pool.get_mut_index(schedule.stream_index) };
            B::flush(&mut stream.stream);
        }

        let execution_index = schedules.first().expect("At least one stream").stream_index;
        let stream = unsafe { self.pool.get_mut_index(execution_index) };

        // Find the maximum number of tasks across all schedules.
        let num_tasks_max = schedules
            .iter()
            .map(|s| s.num_tasks)
            .max()
            .expect("At least one schedule");

        // Iterate through tasks, interleaving them across streams.
        for _ in 0..num_tasks_max {
            for schedule in schedules.iter_mut() {
                // If there are tasks remaining in the schedule, enqueue the next one.
                if let Some(task) = schedule.tasks.next() {
                    B::enqueue(task, &mut stream.stream);
                }
            }
        }

        // Making sure all tasks are registered to the queue.
        B::flush(&mut stream.stream);
    }
}

// Represents a schedule for executing tasks on a specific stream.
struct Schedule<B: SchedulerStreamBackend> {
    // Iterator over the tasks to be executed.
    tasks: alloc::vec::IntoIter<B::Task>,
    // Number of tasks in the schedule.
    num_tasks: usize,
    // Index of the stream in the pool.
    stream_index: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;
    type Events = Arc<Mutex<Vec<(usize, usize, usize)>>>;
    #[derive(Debug)]
    struct BackendStream {
        id: usize,
        events: Events,
    }
    #[derive(Debug)]
    struct Factory {
        next: usize,
        events: Events,
    }
    impl StreamFactory for Factory {
        type Stream = BackendStream;
        fn create(&mut self) -> BackendStream {
            let id = self.next;
            self.next += 1;
            BackendStream {
                id,
                events: self.events.clone(),
            }
        }
    }
    struct Backend(Factory);
    impl SchedulerStreamBackend for Backend {
        type Task = usize;
        type Stream = BackendStream;
        type Factory = Factory;
        fn enqueue(task: usize, stream: &mut BackendStream) {
            stream.events.lock().unwrap().push((stream.id, 0, task));
        }
        fn enqueue_many(tasks: impl Iterator<Item = usize>, stream: &mut BackendStream) {
            stream.events.lock().unwrap().push((stream.id, 1, 0));
            for task in tasks {
                Self::enqueue(task, stream);
            }
        }
        fn flush(stream: &mut BackendStream) {
            stream.events.lock().unwrap().push((stream.id, 2, 0));
        }
        fn factory(&mut self) -> &mut Factory {
            &mut self.0
        }
    }
    fn scheduler(strategy: SchedulerStrategy) -> (SchedulerMultiStream<Backend>, Events) {
        let events = Arc::new(Mutex::new(Vec::new()));
        let backend = Backend(Factory {
            next: 0,
            events: events.clone(),
        });
        (
            SchedulerMultiStream::new(
                Arc::new(ServerLogger::default()),
                backend,
                SchedulerMultiStreamOptions {
                    max_streams: 4,
                    max_tasks: 256,
                    strategy,
                },
            ),
            events,
        )
    }
    #[test]
    fn single_stream_batches_preserve_original_order_and_flush_after_enqueue() {
        for strategy in [SchedulerStrategy::Interleave, SchedulerStrategy::Sequential] {
            let (mut scheduler, events) = scheduler(strategy);
            let id = StreamId { value: 0 };
            for task in 10..15 {
                scheduler.register(id, task, &[id]);
            }
            scheduler.execute_streams(vec![id, id]);
            assert_eq!(
                *events.lock().unwrap(),
                vec![
                    (0, 1, 0),
                    (0, 0, 10),
                    (0, 0, 11),
                    (0, 0, 12),
                    (0, 0, 13),
                    (0, 0, 14),
                    (0, 2, 0)
                ]
            );
        }
    }
    #[test]
    fn same_stream_materialization_keeps_order_without_submission_credit() {
        for strategy in [SchedulerStrategy::Interleave, SchedulerStrategy::Sequential] {
            let (mut scheduler, events) = scheduler(strategy);
            let id = StreamId { value: 0 };
            for task in 10..13 { scheduler.register(id,task,&[id]); }
            scheduler.enqueue_stream_tasks(id);
            for task in 13..15 { scheduler.register(id,task,&[id]); }
            scheduler.enqueue_stream_tasks(id);
            let observed=events.lock().unwrap().clone();
            assert_eq!(observed.iter().filter(|e|e.1==0).map(|e|e.2).collect::<Vec<_>>(), vec![10,11,12,13,14]);
            assert!(!observed.iter().any(|e|e.1==2));
            scheduler.execute_streams(vec![id]);
            assert_eq!(events.lock().unwrap().last(),Some(&(0,2,0)));
        }
    }

    #[test]
    fn multiple_interleaved_streams_keep_original_single_task_ordering() {
        let (mut scheduler, events) = scheduler(SchedulerStrategy::Interleave);
        let a = StreamId { value: 0 };
        let b = StreamId { value: 1 };
        for task in 10..12 {
            scheduler.register(a, task, &[a]);
        }
        for task in 20..22 {
            scheduler.register(b, task, &[b]);
        }
        scheduler.execute_streams(vec![a, b]);
        assert_eq!(
            *events.lock().unwrap(),
            vec![
                (1, 2, 0),
                (0, 0, 10),
                (0, 0, 20),
                (0, 0, 11),
                (0, 0, 21),
                (0, 2, 0)
            ]
        );
    }
}
