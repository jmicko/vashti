//! Conservative single-upstream admission, separate from future GPU placement.
use crate::journal::Journal;
use kuno_protocol::AdmissionStatus;
use std::path::Path;
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tokio::sync::{OwnedSemaphorePermit, Semaphore};

#[derive(Default)]
struct State {
    active: bool,
    waiting: usize,
    paused: bool,
    recovery_required: bool,
    completed: u64,
    rejected: u64,
}

pub struct Admission {
    state: Mutex<State>,
    semaphore: Arc<Semaphore>,
    max_waiting: usize,
    timeout: Duration,
    journal: Option<Journal>,
}

impl Admission {
    pub fn new(max_waiting: usize, timeout: Duration) -> Arc<Self> {
        Arc::new(Self {
            state: Mutex::new(State::default()),
            semaphore: Arc::new(Semaphore::new(1)),
            max_waiting,
            timeout,
            journal: None,
        })
    }

    pub fn persistent(
        max_waiting: usize,
        timeout: Duration,
        directory: &Path,
    ) -> Result<Arc<Self>, &'static str> {
        let journal =
            Journal::open(directory).map_err(|_| "admission_state_unavailable_or_in_use")?;
        let recovery_required = journal
            .uncertain()
            .map_err(|_| "admission_state_unreadable")?;
        Ok(Arc::new(Self {
            state: Mutex::new(State {
                recovery_required,
                ..State::default()
            }),
            semaphore: Arc::new(Semaphore::new(1)),
            max_waiting,
            timeout,
            journal: Some(journal),
        }))
    }

    pub fn status(&self) -> AdmissionStatus {
        let s = self.state.lock().unwrap();
        AdmissionStatus {
            active: s.active,
            waiting: s.waiting,
            paused: s.paused,
            recovery_required: s.recovery_required,
            completed: s.completed,
            rejected: s.rejected,
        }
    }

    /// Pause denies new work; active work is allowed to finish. Resume never
    /// clears an uncertain upstream. The caller must observe `active == false`
    /// before treating a pause as an exclusive maintenance reservation.
    pub fn pause(&self, paused: bool) {
        self.state.lock().unwrap().paused = paused;
    }

    pub async fn acquire(self: &Arc<Self>) -> Result<Lease, &'static str> {
        {
            let mut s = self.state.lock().unwrap();
            let error = if s.recovery_required {
                Some("upstream_recovery_required")
            } else if s.paused {
                Some("admission_paused")
            } else if s.waiting >= self.max_waiting && (s.active || s.waiting > 0) {
                Some("queue_full")
            } else {
                None
            };
            if let Some(error) = error {
                s.rejected += 1;
                return Err(error);
            }
            s.waiting += 1;
        }
        // RAII also removes clients cancelled while they are waiting.
        let waiting = Waiting(self.clone());
        let permit = match tokio::time::timeout(
            self.timeout,
            self.semaphore.clone().acquire_owned(),
        )
        .await
        {
            Ok(Ok(permit)) => permit,
            _ => {
                self.state.lock().unwrap().rejected += 1;
                return Err("queue_timeout");
            }
        };
        drop(waiting);
        let mut s = self.state.lock().unwrap();
        if s.paused || s.recovery_required {
            s.rejected += 1;
            return Err(if s.recovery_required {
                "upstream_recovery_required"
            } else {
                "admission_paused"
            });
        }
        s.active = true;
        Ok(Lease {
            admission: self.clone(),
            _permit: permit,
            dispatched: false,
            complete: false,
        })
    }
}

struct Waiting(Arc<Admission>);
impl Drop for Waiting {
    fn drop(&mut self) {
        self.0.state.lock().unwrap().waiting -= 1;
    }
}

pub struct Lease {
    admission: Arc<Admission>,
    _permit: OwnedSemaphorePermit,
    dispatched: bool,
    complete: bool,
}

impl Lease {
    pub fn dispatched(&mut self) -> Result<(), &'static str> {
        if let Some(journal) = &self.admission.journal
            && journal.begin().is_err()
        {
            self.admission.state.lock().unwrap().recovery_required = true;
            return Err("admission_state_write_failed");
        }
        self.dispatched = true;
        Ok(())
    }
    pub fn complete(&mut self) {
        self.complete = true;
    }
}

impl Drop for Lease {
    fn drop(&mut self) {
        let mut s = self.admission.state.lock().unwrap();
        s.active = false;
        if self.dispatched && !self.complete {
            s.recovery_required = true;
        }
        if self.dispatched
            && self.complete
            && let Some(journal) = &self.admission.journal
            && journal.finish().is_err()
        {
            s.recovery_required = true;
        }
        if self.complete {
            s.completed += 1;
        }
    }
}
