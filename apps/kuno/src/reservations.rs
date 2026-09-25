//! Atomic GPU/worker reservations. Waiting cancellation releases queue capacity;
//! dispatched cancellation quarantines its scope until explicit reconciliation.
//! The durable marker is content-free and conservatively blocks all scopes on
//! an uncertain process restart. Device-specific recovery requires a supervisor.
use crate::journal::Journal;
use std::{
    collections::{BTreeMap, BTreeSet, VecDeque},
    path::Path,
    sync::{Arc, Mutex},
    time::Duration,
};
use tokio::sync::Notify;

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Resource {
    Device(String),
    Worker(String),
}
pub type Scope = BTreeSet<Resource>;

pub fn plan_scope(plan: &crate::placement::Plan) -> Scope {
    plan.devices
        .iter()
        .cloned()
        .map(Resource::Device)
        .chain(plan.workers.iter().cloned().map(Resource::Worker))
        .collect()
}

#[derive(Default)]
struct State {
    next_id: u64,
    waiting: VecDeque<(u64, Scope)>,
    held: BTreeMap<u64, Scope>,
    quarantined: Scope,
    recovery_required: bool,
    dispatched: usize,
    marker: bool,
}

pub struct Reservations {
    state: Mutex<State>,
    changed: Notify,
    max_waiting: usize,
    timeout: Duration,
    journal: Option<Journal>,
}

impl Reservations {
    pub fn recovery_required(&self) -> bool {
        let state = self.state.lock().unwrap();
        state.recovery_required || !state.quarantined.is_empty()
    }

    pub fn new(max_waiting: usize, timeout: Duration) -> Arc<Self> {
        Arc::new(Self {
            state: Mutex::new(State::default()),
            changed: Notify::new(),
            max_waiting,
            timeout,
            journal: None,
        })
    }
    pub fn persistent(
        path: &Path,
        max_waiting: usize,
        timeout: Duration,
    ) -> Result<Arc<Self>, &'static str> {
        let journal = Journal::open(path).map_err(|_| "reservation_journal_unavailable")?;
        let uncertain = journal
            .uncertain()
            .map_err(|_| "reservation_journal_unreadable")?;
        Ok(Arc::new(Self {
            state: Mutex::new(State {
                recovery_required: uncertain,
                marker: uncertain,
                ..State::default()
            }),
            changed: Notify::new(),
            max_waiting,
            timeout,
            journal: Some(journal),
        }))
    }

    pub async fn acquire(self: &Arc<Self>, scope: Scope) -> Result<Reservation, &'static str> {
        if scope.is_empty()
            || scope.iter().any(|key| {
                let (Resource::Device(s) | Resource::Worker(s)) = key;
                s.is_empty() || s.len() > 256 || s.chars().any(char::is_control)
            })
        {
            return Err("reservation_scope_invalid");
        }
        let id = {
            let mut state = self.state.lock().unwrap();
            if state.recovery_required || !state.quarantined.is_disjoint(&scope) {
                return Err("resource_recovery_required");
            }
            let blocked = state.held.values().any(|keys| !keys.is_disjoint(&scope))
                || state
                    .waiting
                    .iter()
                    .any(|(_, keys)| !keys.is_disjoint(&scope));
            if blocked && state.waiting.len() >= self.max_waiting {
                return Err("resource_queue_full");
            }
            state.next_id = state
                .next_id
                .checked_add(1)
                .ok_or("reservation_identity_exhausted")?;
            let id = state.next_id;
            state.waiting.push_back((id, scope.clone()));
            id
        };
        let waiting = Waiting {
            owner: self.clone(),
            id,
        };
        let result = tokio::time::timeout(self.timeout, async {
            loop {
                let notified = self.changed.notified();
                tokio::pin!(notified);
                notified.as_mut().enable();
                {
                    let mut s = self.state.lock().unwrap();
                    if s.recovery_required || !s.quarantined.is_disjoint(&scope) {
                        return Err("resource_recovery_required");
                    }
                    // An earlier overlapping waiter has priority. Independent
                    // devices can run without waiting behind a busy GPU.
                    let earlier = s
                        .waiting
                        .iter()
                        .take_while(|(other, _)| *other != id)
                        .any(|(_, keys)| !keys.is_disjoint(&scope));
                    if !earlier && s.held.values().all(|keys| keys.is_disjoint(&scope)) {
                        s.held.insert(id, scope.clone());
                        return Ok(Reservation {
                            owner: self.clone(),
                            id,
                            scope,
                            dispatched: false,
                            complete: false,
                        });
                    }
                }
                notified.await;
            }
        })
        .await
        .unwrap_or(Err("resource_queue_timeout"));
        drop(waiting);
        result
    }

    pub fn quarantined(&self) -> Scope {
        self.state.lock().unwrap().quarantined.clone()
    }

    /// Only the owning supervisor may call this, after stopping/reaping its
    /// worker and establishing that no dispatched work can remain in scope.
    /// It cannot resolve a journal failure or unknown process-restart state.
    pub(crate) fn reconciled(&self, scope: &Scope) -> Result<(), &'static str> {
        let mut s = self.state.lock().unwrap();
        if s.recovery_required || s.held.values().any(|keys| !keys.is_disjoint(scope)) {
            return Err("resource_reconciliation_not_exclusive");
        }
        let remaining: Scope = s.quarantined.difference(scope).cloned().collect();
        if remaining.is_empty() && s.dispatched == 0 && s.marker {
            if let Some(journal) = &self.journal
                && journal.finish().is_err()
            {
                s.recovery_required = true;
                return Err("reservation_journal_write_failed");
            }
            s.marker = false;
        }
        s.quarantined = remaining;
        drop(s);
        self.changed.notify_waiters();
        Ok(())
    }
}

struct Waiting {
    owner: Arc<Reservations>,
    id: u64,
}
impl Drop for Waiting {
    fn drop(&mut self) {
        self.owner
            .state
            .lock()
            .unwrap()
            .waiting
            .retain(|(id, _)| *id != self.id);
        self.owner.changed.notify_waiters();
    }
}

pub struct Reservation {
    owner: Arc<Reservations>,
    id: u64,
    scope: Scope,
    dispatched: bool,
    complete: bool,
}

impl Reservation {
    /// Must precede any runtime action. Journal failure prevents dispatch.
    pub fn dispatched(&mut self) -> Result<(), &'static str> {
        if self.dispatched {
            return Ok(());
        }
        let mut state = self.owner.state.lock().unwrap();
        if state.recovery_required {
            return Err("resource_recovery_required");
        }
        if !state.marker {
            if let Some(journal) = &self.owner.journal
                && journal.begin().is_err()
            {
                state.recovery_required = true;
                self.owner.changed.notify_waiters();
                return Err("reservation_journal_write_failed");
            }
            state.marker = true;
        }
        state.dispatched += 1;
        self.dispatched = true;
        self.complete = false;
        Ok(())
    }
    /// Runtime completion/cancellation has been confirmed, not merely a client
    /// disconnect. Warm residency may remain; active computation must be over.
    pub fn complete(&mut self) {
        self.complete = true;
    }
}

impl Drop for Reservation {
    fn drop(&mut self) {
        let mut s = self.owner.state.lock().unwrap();
        s.held.remove(&self.id);
        if self.dispatched {
            s.dispatched -= 1;
            if !self.complete {
                s.quarantined.extend(self.scope.clone());
            }
            if s.dispatched == 0 && s.quarantined.is_empty() && !s.recovery_required {
                if let Some(journal) = &self.owner.journal
                    && journal.finish().is_err()
                {
                    s.recovery_required = true;
                } else {
                    s.marker = false;
                }
            }
        }
        drop(s);
        self.owner.changed.notify_waiters();
    }
}
