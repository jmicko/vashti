//! Opt-in public artifact admission and graph scheduling checks. No live user
//! inputs, GPU execution, or numerical parity/performance claim.
use kuno_text::model::MatrixExecutor;
use kuno_text::mtp::MtpGraph;
use kuno_text::{Error, GgufFile, GgufLimits, Result, TensorView};
use std::sync::atomic::AtomicBool;

struct SyntheticExecutor {
    names: Vec<String>,
    fail_after: Option<usize>,
}
impl MatrixExecutor for SyntheticExecutor {
    fn run(
        &mut self,
        tensor: TensorView<'_>,
        _expert: usize,
        input: &[f32],
        _cancel: &AtomicBool,
    ) -> Result<Vec<f32>> {
        assert_eq!(input.len() as u64, tensor.info().dimensions()[0]);
        self.names.push(tensor.info().name().into());
        if self.fail_after == Some(self.names.len()) {
            return Err(Error::InvalidFormat("synthetic MTP failure"));
        }
        Ok(vec![0.; tensor.info().dimensions()[1] as usize])
    }
}

#[test]
#[ignore = "requires public self-contained Q8 sidecar via KUNO_TEST_MTP"]
fn downloaded_head_binds_and_runs_bounded_synthetic_graph_and_recovery() {
    let path = std::env::var_os("KUNO_TEST_MTP").expect("KUNO_TEST_MTP is required");
    let file = GgufFile::open(path, GgufLimits::default()).unwrap();
    let graph = MtpGraph::bind(&file).unwrap();
    assert_eq!(graph.tensor_names().count(), 34);
    assert!(graph.tensor("blk.0.attn_q.weight").is_err());
    assert!(graph.sequence(0).is_err());
    assert!(graph.sequence(2049).is_err());
    let mut sequence = graph.sequence(4).unwrap();
    let mut executor = SyntheticExecutor {
        names: Vec::new(),
        fail_after: None,
    };
    let hidden = vec![0.; 10240];
    let cancel = AtomicBool::new(false);
    let out = graph
        .step(&mut executor, &mut sequence, 0, &hidden, &cancel)
        .unwrap();
    assert_eq!(
        (out.logits.len(), out.hidden.len(), sequence.position()),
        (248320, 10240, 1)
    );
    assert!(out.logits.iter().all(|v| v.to_bits() == 0));
    assert_eq!(
        executor
            .names
            .iter()
            .filter(|n| n.as_str() == "blk.48.nextn.eh_proj.weight")
            .count(),
        4
    );
    assert_eq!(
        executor
            .names
            .iter()
            .filter(|n| n.as_str() == "blk.48.ffn_gate_exps.weight")
            .count(),
        10
    );
    assert!(
        !executor
            .names
            .iter()
            .any(|n| n.contains("indexer") || n.contains("ssm") || n.contains("ple"))
    );
    let other = MtpGraph::bind(&file).unwrap();
    assert!(
        other
            .step(&mut executor, &mut sequence, 0, &hidden, &cancel)
            .is_err()
    );
    executor.fail_after = Some(executor.names.len() + 13); // after attention KV publication
    assert!(
        graph
            .step(&mut executor, &mut sequence, 0, &hidden, &cancel)
            .is_err()
    );
    assert_eq!(sequence.position(), 0);
    executor.fail_after = None;
    assert!(
        graph
            .step(&mut executor, &mut sequence, 0, &hidden, &cancel)
            .is_err()
    );
    sequence.reset();
    let recovered = graph
        .step(&mut executor, &mut sequence, 0, &hidden, &cancel)
        .unwrap();
    assert_eq!(&*out.logits, &*recovered.logits);
    assert_eq!(&*out.hidden, &*recovered.hidden);
    sequence.rewind(0).unwrap();
    let fresh = graph
        .step(&mut executor, &mut sequence, 0, &hidden, &cancel)
        .unwrap();
    assert_eq!(&*out.logits, &*fresh.logits);
}
