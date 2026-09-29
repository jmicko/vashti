//! Content-free operational events. Callers cannot pass messages, errors,
//! model identifiers or request data to this interface. Journald supplies the
//! wall/boot timestamps; durations use a monotonic clock.
use serde::Serialize;
use std::{io::Write, time::Instant};

#[derive(Clone, Copy, Serialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum Stage {
    WorkerSpawn,
    WorkerReady,
    WorkerLoad,
    WorkerLoadHeaders,
    WorkerLoadBody,
    WorkerUnload,
    WorkerUnloadHeaders,
    WorkerStop,
    WorkerSignal,
    WorkerGroupExit,
    WorkerReap,
    WorkerDropSignal,
    WorkerResponseBody,
    ManagedPrepare,
    ManagedProfile,
    ManagedPlacement,
    ManagedAbort,
    UpstreamHeaders,
    UpstreamBody,
    UpstreamDrain,
    AdmissionRecovery,
}

#[derive(Clone, Copy, Serialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum Outcome {
    Begin,
    Success,
    Failed,
    Timeout,
    Cancelled,
    Sigkill,
    Rejected,
    BodyError,
    TransportError,
    InvalidResponse,
    AlreadyStopped,
    Cached,
    IsolatedProbe,
    Supplied,
    Reuse,
    Preload,
    Required,
}

#[derive(Serialize)]
struct Record<'a> {
    schema: u8,
    stage: Stage,
    outcome: Outcome,
    elapsed_ms: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pid: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pci_address: Option<&'a str>,
}

fn pci_address(value: &str) -> bool {
    value.len() == 12
        && value.bytes().enumerate().all(|(i, b)| match i {
            4 | 7 => b == b':',
            10 => b == b'.',
            11 => matches!(b, b'0'..=b'7'),
            _ => b.is_ascii_hexdigit(),
        })
}

fn line(
    stage: Stage,
    outcome: Outcome,
    elapsed_ms: u64,
    pid: Option<u32>,
    pci: Option<&str>,
) -> String {
    // All variable strings pass this strict format check even when callers
    // already validated the worker configuration.
    let record = Record {
        schema: 1,
        stage,
        outcome,
        elapsed_ms,
        pid,
        pci_address: pci.filter(|s| pci_address(s)),
    };
    format!("kuno_lifecycle {}", serde_json::to_string(&record).unwrap())
}

pub(crate) struct Span {
    stage: Stage,
    started: Instant,
    pid: Option<u32>,
    pci: Option<String>,
    finished: bool,
}

impl Span {
    pub(crate) fn new(stage: Stage, pid: Option<u32>, pci: Option<&str>) -> Self {
        let span = Self {
            stage,
            started: Instant::now(),
            pid,
            pci: pci.filter(|s| pci_address(s)).map(str::to_owned),
            finished: false,
        };
        span.emit(Outcome::Begin);
        span
    }

    pub(crate) fn event(stage: Stage, outcome: Outcome, pid: Option<u32>, pci: Option<&str>) {
        // Diagnostics must not panic if the log sink disappears.
        let _ = writeln!(
            std::io::stderr().lock(),
            "{}",
            line(stage, outcome, 0, pid, pci)
        );
    }

    fn emit(&self, outcome: Outcome) {
        let elapsed = self.started.elapsed().as_millis().min(u64::MAX as u128) as u64;
        let _ = writeln!(
            std::io::stderr().lock(),
            "{}",
            line(self.stage, outcome, elapsed, self.pid, self.pci.as_deref())
        );
    }

    pub(crate) fn finish(mut self, outcome: Outcome) {
        self.finished = true;
        self.emit(outcome);
    }

    pub(crate) fn result<T, E>(self, result: &Result<T, E>) {
        self.finish(if result.is_ok() {
            Outcome::Success
        } else {
            Outcome::Failed
        });
    }
}

impl Drop for Span {
    fn drop(&mut self) {
        if !self.finished {
            self.emit(Outcome::Cancelled);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn records_have_only_fixed_fields_and_validated_device_identity() {
        for pci in [
            "DO-NOT-PERSIST",
            "0000:2d:00.0\nsecret",
            "0000:2d:00.8",
            "0000:2d:00.0",
        ] {
            let text = line(
                Stage::WorkerLoad,
                Outcome::Timeout,
                300_000,
                Some(42),
                Some(pci),
            );
            assert!(!text.contains("DO-NOT-PERSIST"));
            assert!(!text.contains("secret"));
            let record: serde_json::Value =
                serde_json::from_str(text.strip_prefix("kuno_lifecycle ").unwrap()).unwrap();
            assert_eq!(record["stage"], "worker_load");
            assert_eq!(record["outcome"], "timeout");
            assert_eq!(record["elapsed_ms"], 300_000);
            assert_eq!(record["pid"], 42);
            assert_eq!(
                record["pci_address"].as_str(),
                (pci == "0000:2d:00.0").then_some(pci)
            );
            assert!(record.as_object().unwrap().keys().all(|key| matches!(
                key.as_str(),
                "schema" | "stage" | "outcome" | "elapsed_ms" | "pid" | "pci_address"
            )));
        }
    }
}
