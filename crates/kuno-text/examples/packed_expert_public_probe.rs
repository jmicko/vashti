//! Fixed-public CPU numeric probe. Never accepts prompts, routes or private data.
use kuno_text::cpu_expert::{Backend, ExpertOperator, PackedExperts, PackedProjection, WorkerPlan};
use kuno_text::{Error, GgufFile, GgufLimits, Result, TensorType, TensorView};
use std::sync::atomic::AtomicBool;
use zeroize::Zeroizing;

// Independent original-row decoder + explicit scalar64 schedule. It never calls
// the CPU expert vector implementation or expands a complete expert to F32.
fn reference(
    tensor: TensorView<'_>,
    expert: usize,
    input: &[f32],
    rows: usize,
) -> Result<Zeroizing<Vec<f32>>> {
    let row_bytes =
        usize::try_from(tensor.info().row_bytes()).map_err(|_| Error::ArithmeticOverflow)?;
    let mut packed = vec![0; row_bytes]; // original public model bytes only
    let mut decoded = Zeroizing::new(vec![0.; input.len()]);
    let mut result = Zeroizing::new(vec![0.; rows]);
    let mut sums = Zeroizing::new([0.; 64]);
    for row in 0..rows {
        tensor.read_row((expert * rows + row) as u64, &mut packed)?;
        tensor.info().dtype().decode(&packed, &mut decoded)?;
        sums.fill(0.);
        for column in 0..input.len() {
            sums[column % 64] += decoded[column] * input[column];
        }
        for stride in [32, 16, 8, 4, 2, 1] {
            for lane in 0..stride {
                sums[lane] += sums[lane + stride];
            }
        }
        result[row] = sums[0];
    }
    if result.iter().any(|value| !value.is_finite()) {
        return Err(Error::InvalidFormat("public CPU reference nonfinite"));
    }
    Ok(result)
}

fn exact(left: &[f32], right: &[f32]) -> Result<()> {
    if left.len() != right.len()
        || left
            .iter()
            .zip(right)
            .any(|(a, b)| a.to_bits() != b.to_bits())
    {
        return Err(Error::InvalidFormat(
            "public original CPU64 numeric mismatch",
        ));
    }
    Ok(())
}

fn main() -> Result<()> {
    let path = std::env::args_os()
        .nth(1)
        .ok_or(Error::InvalidFormat("existing model required"))?;
    if !kuno_text::cpu_expert::avx2_available() {
        return Err(Error::InvalidFormat("public probe AVX2 required"));
    }
    let model = GgufFile::open(path, GgufLimits::default())?;
    let reader = model.positional_reader()?;
    let cancel = AtomicBool::new(false);
    let input = Zeroizing::new(
        (0..2560)
            .map(|i| ((i % 29) as f32 - 14.) / 512.)
            .collect::<Vec<_>>(),
    );
    for kind in [TensorType::Q5_0, TensorType::Q8_0] {
        let layer = (0..48)
            .find(|layer| {
                model
                    .tensor(&format!("blk.{layer}.ffn_down_exps.weight"))
                    .is_some_and(|tensor| tensor.info().dtype() == kind)
            })
            .ok_or(Error::InvalidFormat(
                "public probe original down format missing",
            ))?;
        let gate = model
            .tensor(&format!("blk.{layer}.ffn_gate_exps.weight"))
            .ok_or(Error::InvalidFormat("public original gate missing"))?;
        let up = model
            .tensor(&format!("blk.{layer}.ffn_up_exps.weight"))
            .ok_or(Error::InvalidFormat("public original up missing"))?;
        let down = model
            .tensor(&format!("blk.{layer}.ffn_down_exps.weight"))
            .ok_or(Error::InvalidFormat("public original down missing"))?;
        if gate.info().dimensions() != [2560, 640, 512]
            || up.info().dimensions() != [2560, 640, 512]
            || down.info().dimensions() != [640, 2560, 512]
            || gate.info().dtype() != TensorType::Q4K
            || up.info().dtype() != TensorType::Q4K
        {
            return Err(Error::InvalidFormat(
                "public probe original triple geometry",
            ));
        }
        let weights = PackedExperts::new(&reader, gate, up, down)?;
        let projections = [
            PackedProjection::new(&reader, gate)?,
            PackedProjection::new(&reader, up)?,
        ];
        for expert in [0usize, 7] {
            let raw_gate = reference(gate, expert, &input, 640)?;
            let raw_up = reference(up, expert, &input, 640)?;
            let activation = Zeroizing::new(
                raw_gate
                    .iter()
                    .zip(raw_up.iter())
                    .map(|(&g, &u)| kuno_text::ops::silu(g) * u)
                    .collect::<Vec<_>>(),
            );
            let partial = reference(down, expert, &activation, 2560)?;
            let portable = ExpertOperator::new(WorkerPlan::physical(1)?, Backend::Portable)?;
            let portable_partial = portable.execute(&weights, &[expert], &input, &cancel)?;
            exact(portable_partial.active(), &partial)?;
            for workers in [1, 4, 8, 16] {
                let operator = ExpertOperator::new(WorkerPlan::physical(workers)?, Backend::Avx2)?;
                let actual_gate = operator.project(&projections[0], expert, &input, &cancel)?;
                let actual_up = operator.project(&projections[1], expert, &input, &cancel)?;
                exact(actual_gate.values(), &raw_gate)?;
                exact(actual_up.values(), &raw_up)?;
                let actual = operator.execute(&weights, &[expert], &input, &cancel)?;
                exact(actual.active(), &partial)?;
                if actual.full()[2560..]
                    .iter()
                    .any(|value| value.to_bits() != 0)
                {
                    return Err(Error::InvalidFormat("public CPU inactive partial tail"));
                }
                println!(
                    "public_cpu_original_numeric down_dtype={} layer={} workers={} ranks=1 raw_rows=1280 partial_rows=2560 independent64_bit_match=true inactive_tail_zero=true",
                    kind.gguf_id(),
                    layer,
                    workers
                );
            }
        }
    }
    reader.verify_file_unchanged()?;
    println!("public_cpu_original_numeric cases=16 complete=true performance_claim=false");
    Ok(())
}
