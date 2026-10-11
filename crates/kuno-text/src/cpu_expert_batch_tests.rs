use batch::{ExpertView, Member, Rank};

const DAS_TRIPLES: [[TensorType; 3]; 7] = [
    [TensorType::Iq3Xxs, TensorType::Iq3Xxs, TensorType::Iq4Nl],
    [TensorType::Iq3Xxs, TensorType::Iq3Xxs, TensorType::Q2_0],
    [TensorType::Iq3S, TensorType::Iq3S, TensorType::Iq4Nl],
    [TensorType::Iq3S, TensorType::Iq3S, TensorType::Q2_0],
    [TensorType::Iq2S, TensorType::Iq2S, TensorType::Iq4Nl],
    [TensorType::Iq2S, TensorType::Iq2S, TensorType::Q2_0],
    [TensorType::Iq4Xs, TensorType::Iq4Xs, TensorType::Iq4Nl],
];

fn public_spans(
    reader: &PositionalReader<'_>,
    weights: &PackedExperts<'_, '_>,
) -> Vec<[Vec<u8>; 3]> {
    (0..weights.geometry.experts)
        .map(|expert| {
            std::array::from_fn(|part| {
                let mut packed = vec![0; weights.geometry.packed_bytes[part]];
                reader
                    .read_range(
                        weights.tensors[part],
                        (expert * weights.geometry.packed_bytes[part]) as u64,
                        &mut packed,
                    )
                    .unwrap();
                packed
            })
        })
        .collect()
}

fn public_views<'a>(
    weights: &'a PackedExperts<'_, '_>,
    packed: &'a [[Vec<u8>; 3]],
) -> Vec<ExpertView<'a>> {
    packed
        .iter()
        .enumerate()
        .map(|(expert, spans)| {
            weights
                .borrow_expert(expert, std::array::from_fn(|part| spans[part].as_slice()))
                .unwrap()
        })
        .collect()
}

fn pooled(backend: Backend, workers: usize) -> ExpertOperator {
    ExpertOperator::new_pooled(WorkerPlan::portable(workers).unwrap(), backend).unwrap()
}

#[test]
fn every_das_triple_count1_to3_matches_unchanged_original_and_pinned_c() {
    for kinds in DAS_TRIPLES {
        let f = fixture(kinds, 256, 256, 3);
        let reader = f.model.positional_reader().unwrap();
        let weights = set(&reader, &f);
        let packed = public_spans(&reader, &weights);
        let views = public_views(&weights, &packed);
        // Repeated sources across members appear at different original ranks.
        let selected = [[1, 0], [0, 2], [2, 1]];
        let original = [[0, 9], [1, 6], [3, 8]];
        let inputs: [Vec<f32>; 3] = std::array::from_fn(|member| {
            input(f.hidden)
                .into_iter()
                .map(|value| value * (1. + member as f32 / 4.))
                .collect()
        });
        let ranks: [Vec<Rank<'_>>; 3] = std::array::from_fn(|member| {
            original[member]
                .iter()
                .zip(selected[member])
                .map(|(&rank, expert)| Rank::new(rank, &views[expert]).unwrap())
                .collect()
        });
        let members: Vec<_> = (0..3)
            .map(|member| Member::new(&inputs[member], &ranks[member]).unwrap())
            .collect();
        let backends = if avx2_available() {
            vec![Backend::Portable, Backend::Avx2]
        } else {
            vec![Backend::Portable]
        };
        for backend in backends {
            let operator = pooled(backend, 3);
            let cancel = AtomicBool::new(false);
            for count in 1..=3 {
                let batch = operator
                    .execute_borrowed_batch_with_private_limit(
                        &weights,
                        &members[..count],
                        &cancel,
                        MAX_PRIVATE_HOST_BYTES,
                    )
                    .unwrap();
                assert_eq!(batch.members(), count);
                assert_eq!(batch.hidden(), f.hidden);
                assert_eq!(batch.full().len(), 3 * 10 * f.hidden);
                for member in 0..count {
                    let single = operator
                        .execute(&weights, &selected[member], &inputs[member], &cancel)
                        .unwrap();
                    let independent = expected_experts(&f, &selected[member], &inputs[member]);
                    same_bits(single.active(), &independent);
                    assert_eq!(batch.ranks(member).unwrap(), 2);
                    for (index, &rank) in original[member].iter().enumerate() {
                        same_bits(
                            batch.rank(member, rank).unwrap().unwrap(),
                            &single.active()[index * f.hidden..(index + 1) * f.hidden],
                        );
                    }
                    for rank in 0..10 {
                        if !original[member].contains(&rank) {
                            assert!(batch.rank(member, rank).unwrap().is_none());
                            assert!(
                                batch.member_full(member).unwrap()
                                    [rank * f.hidden..(rank + 1) * f.hidden]
                                    .iter()
                                    .all(|value| value.to_bits() == 0)
                            );
                        }
                    }
                }
                assert!(
                    batch.full()[count * 10 * f.hidden..]
                        .iter()
                        .all(|value| value.to_bits() == 0)
                );
                assert!(batch.rank(count, 0).is_err());
                assert!(batch.rank(0, 10).is_err());
            }
        }
    }
}

#[test]
fn empty_host_members_and_all_thirty_original_ranks_are_bounded() {
    let f = fixture([TensorType::Iq4Nl; 3], 64, 32, 10);
    let reader = f.model.positional_reader().unwrap();
    let weights = set(&reader, &f);
    let packed = public_spans(&reader, &weights);
    let views = public_views(&weights, &packed);
    let inputs = [input(64), input(64), input(64)];
    let operator = pooled(Backend::Portable, 3);
    let cancel = AtomicBool::new(false);
    let empty: Vec<_> = inputs
        .iter()
        .map(|input| Member::new(input, &[]).unwrap())
        .collect();
    let result = operator
        .execute_borrowed_batch_with_private_limit(
            &weights,
            &empty,
            &cancel,
            MAX_PRIVATE_HOST_BYTES,
        )
        .unwrap();
    assert!(result.full().iter().all(|value| value.to_bits() == 0));
    for member in 0..3 {
        assert_eq!(result.ranks(member).unwrap(), 0);
    }
    let ranks: [Vec<Rank<'_>>; 3] = std::array::from_fn(|member| {
        (0..10)
            .map(|rank| Rank::new(rank, &views[(rank + member) % 10]).unwrap())
            .collect()
    });
    let all: Vec<_> = (0..3)
        .map(|member| Member::new(&inputs[member], &ranks[member]).unwrap())
        .collect();
    let result = operator
        .execute_borrowed_batch_with_private_limit(&weights, &all, &cancel, MAX_PRIVATE_HOST_BYTES)
        .unwrap();
    for (member, input) in inputs.iter().enumerate() {
        let selected: Vec<_> = (0..10).map(|rank| (rank + member) % 10).collect();
        let single = operator
            .execute(&weights, &selected, input, &cancel)
            .unwrap();
        same_bits(result.member_full(member).unwrap(), single.active());
    }
    // An empty earlier member does not compress later original member IDs.
    let mixed = [
        Member::new(&inputs[0], &[]).unwrap(),
        Member::new(&inputs[1], &ranks[1][..1]).unwrap(),
        Member::new(&inputs[2], &[]).unwrap(),
    ];
    let result = operator
        .execute_borrowed_batch_with_private_limit(
            &weights,
            &mixed,
            &cancel,
            MAX_PRIVATE_HOST_BYTES,
        )
        .unwrap();
    assert!(
        result
            .member_full(0)
            .unwrap()
            .iter()
            .all(|value| value.to_bits() == 0)
    );
    assert!(
        result
            .member_full(2)
            .unwrap()
            .iter()
            .all(|value| value.to_bits() == 0)
    );
    assert!(result.rank(1, 0).unwrap().is_some());
}

#[test]
fn invalid_expert_spans_members_source_identity_and_inputs_reject() {
    let f = fixture([TensorType::Iq4Nl; 3], 64, 32, 2);
    let reader = f.model.positional_reader().unwrap();
    let weights = set(&reader, &f);
    let foreign_weights = set(&reader, &f);
    let packed = public_spans(&reader, &weights);
    let views = public_views(&weights, &packed);
    let spans = std::array::from_fn(|part| packed[0][part].as_slice());
    assert!(weights.borrow_expert(2, spans).is_err());
    for role in 0..3 {
        let mut short = spans;
        short[role] = &spans[role][..spans[role].len() - 1];
        assert!(weights.borrow_expert(0, short).is_err());
        let mut long = spans[role].to_vec();
        long.push(0);
        let mut too_long = spans;
        too_long[role] = &long;
        assert!(weights.borrow_expert(0, too_long).is_err());
    }
    assert!(Rank::new(10, &views[0]).is_err());
    let ranks = [Rank::new(0, &views[0]).unwrap()];
    let input = input(64);
    let operator = pooled(Backend::Portable, 2);
    let cancel = AtomicBool::new(false);
    let member = [Member::new(&input, &ranks).unwrap()];
    assert!(
        operator
            .execute_borrowed_batch_with_private_limit(
                &weights,
                &[],
                &cancel,
                MAX_PRIVATE_HOST_BYTES
            )
            .is_err()
    );
    let too_many: Vec<_> = (0..4)
        .map(|_| Member::new(&input, &ranks).unwrap())
        .collect();
    assert!(
        operator
            .execute_borrowed_batch_with_private_limit(
                &weights,
                &too_many,
                &cancel,
                MAX_PRIVATE_HOST_BYTES
            )
            .is_err()
    );
    assert!(
        operator
            .execute_borrowed_batch_with_private_limit(
                &foreign_weights,
                &member,
                &cancel,
                MAX_PRIVATE_HOST_BYTES
            )
            .is_err()
    );
    for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        let mut values = input.clone();
        values[63] = bad;
        let bad = [Member::new(&values, &ranks).unwrap()];
        assert!(
            operator
                .execute_borrowed_batch_with_private_limit(
                    &weights,
                    &bad,
                    &cancel,
                    MAX_PRIVATE_HOST_BYTES
                )
                .is_err()
        );
    }
    let short = [Member::new(&input[..63], &ranks).unwrap()];
    assert!(
        operator
            .execute_borrowed_batch_with_private_limit(
                &weights,
                &short,
                &cancel,
                MAX_PRIVATE_HOST_BYTES
            )
            .is_err()
    );
    let descending = [
        Rank::new(5, &views[0]).unwrap(),
        Rank::new(2, &views[1]).unwrap(),
    ];
    let repeated_rank = [
        Rank::new(2, &views[0]).unwrap(),
        Rank::new(2, &views[1]).unwrap(),
    ];
    let repeated_expert = [
        Rank::new(2, &views[0]).unwrap(),
        Rank::new(5, &views[0]).unwrap(),
    ];
    assert!(Member::new(&input, &descending).is_err());
    assert!(Member::new(&input, &repeated_rank).is_err());
    assert!(Member::new(&input, &repeated_expert).is_err());
    let eleven: Vec<_> = (0..11).map(|_| Rank::new(0, &views[0]).unwrap()).collect();
    assert!(Member::new(&input, &eleven).is_err());
    assert!(Member::new(&[], &[]).is_err());
    assert!(Member::new(&vec![0.; MAX_HIDDEN + 1], &[]).is_err());
}

#[test]
fn full_request_and_every_existing_worker_scratch_are_admitted_before_rows() {
    // F32 at this geometry exceeds the existing 8-MiB component bound; original
    // IQ4_NL is a valid packed view at the maximum H/FF scratch geometry.
    let f = fixture([TensorType::Iq4Nl; 3], 2560, 1024, 1);
    let reader = f.model.positional_reader().unwrap();
    let weights = set(&reader, &f);
    let operator = pooled(Backend::Portable, 16);
    let required = operator.borrowed_batch_private_bytes(&weights).unwrap();
    assert!(required <= MAX_PRIVATE_HOST_BYTES);
    assert!(required >= 3 * 2560 * 4 + 30 * 1024 * 4 + 30 * 2560 * 4 + 16 * 32768);
    let input = input(2560);
    let members = [Member::new(&input, &[]).unwrap()];
    let calls = AtomicUsize::new(0);
    let cancel = AtomicBool::new(false);
    assert!(
        operator
            .execute_borrowed_batch_with(&weights, &members, &cancel, required - 1, |_| {
                calls.fetch_add(1, Ordering::SeqCst);
            })
            .is_err()
    );
    assert_eq!(calls.load(Ordering::SeqCst), 0);
    assert!(
        operator
            .execute_borrowed_batch_with_private_limit(
                &weights,
                &members,
                &cancel,
                MAX_PRIVATE_HOST_BYTES + 1
            )
            .is_err()
    );
    let result = operator
        .execute_borrowed_batch_with_private_limit(&weights, &members, &cancel, required)
        .unwrap();
    assert_eq!(result.owned_bytes(), 30 * 2560 * 4);
    assert!(result.full().iter().all(|value| value.to_bits() == 0));
    // The scratch bound is the actual existing helper, including decoded-row
    // capacity, not a generic constant substituted for that check.
    assert!(Scratch::check_capacity(MAX_WORKER_PRIVATE_BYTES / 4).is_err());
}

#[test]
fn joined_reusable_pool_completes_all_members_before_return_and_after_failure() {
    let f = fixture([TensorType::Iq4Nl; 3], 64, 32, 3);
    let reader = f.model.positional_reader().unwrap();
    let weights = set(&reader, &f);
    let packed = public_spans(&reader, &weights);
    let views = public_views(&weights, &packed);
    let inputs = [input(64), input(64), input(64)];
    let ranks: [Vec<Rank<'_>>; 3] = std::array::from_fn(|member| {
        (0..3)
            .map(|rank| Rank::new(rank, &views[(rank + member) % 3]).unwrap())
            .collect()
    });
    let members: Vec<_> = (0..3)
        .map(|member| Member::new(&inputs[member], &ranks[member]).unwrap())
        .collect();
    let operator = pooled(Backend::Portable, 3);
    let cancel = AtomicBool::new(false);
    let calls = AtomicUsize::new(0);
    let result = operator
        .execute_borrowed_batch_with(&weights, &members, &cancel, MAX_PRIVATE_HOST_BYTES, |_| {
            calls.fetch_add(1, Ordering::SeqCst);
        })
        .unwrap();
    assert_eq!(calls.load(Ordering::SeqCst), 9 * (32 + 64));
    for member in 0..3 {
        assert_eq!(result.ranks(member).unwrap(), 3);
    }
    // Inject only a public test worker panic. The existing pool catches it,
    // joins consumers, wipes partial work, and remains usable for a fresh call.
    assert!(
        operator
            .execute_borrowed_batch_with(
                &weights,
                &members,
                &cancel,
                MAX_PRIVATE_HOST_BYTES,
                |_| {
                    panic!("fixed-public borrowed CPU worker failure");
                }
            )
            .is_err()
    );
    let after = operator
        .execute_borrowed_batch_with_private_limit(
            &weights,
            &members,
            &cancel,
            MAX_PRIVATE_HOST_BYTES,
        )
        .unwrap();
    same_bits(after.full(), result.full());
    cancel.store(true, Ordering::Relaxed);
    assert!(
        operator
            .execute_borrowed_batch_with_private_limit(
                &weights,
                &members,
                &cancel,
                MAX_PRIVATE_HOST_BYTES
            )
            .is_err()
    );
}

#[test]
fn original_pooled_policy_rejects_nonpooled_and_competing_arithmetic_options() {
    let f = fixture([TensorType::Iq4Nl; 3], 64, 32, 1);
    let reader = f.model.positional_reader().unwrap();
    let weights = set(&reader, &f);
    let input = input(64);
    let members = [Member::new(&input, &[]).unwrap()];
    let cancel = AtomicBool::new(false);
    let unpooled =
        ExpertOperator::new(WorkerPlan::portable(2).unwrap(), Backend::Portable).unwrap();
    assert!(unpooled.borrowed_batch_private_bytes(&weights).is_err());
    let operators = [
        pooled(Backend::Portable, 2).with_mapped_prefetch(),
        pooled(Backend::Portable, 2)
            .with_buffered_weights()
            .unwrap(),
        pooled(Backend::Portable, 2).with_phase_diagnostics(),
    ];
    for operator in operators {
        assert!(
            operator
                .execute_borrowed_batch_with_private_limit(
                    &weights,
                    &members,
                    &cancel,
                    MAX_PRIVATE_HOST_BYTES
                )
                .is_err()
        );
    }
    if avx2_available() {
        for operator in [
            pooled(Backend::Avx2, 2).with_q8_activations().unwrap(),
            pooled(Backend::Avx2, 2).with_q8_residual().unwrap(),
        ] {
            assert!(operator.borrowed_batch_private_bytes(&weights).is_err());
        }
    }
    if fma_available() {
        assert!(
            pooled(Backend::Avx2Fma, 2)
                .borrowed_batch_private_bytes(&weights)
                .is_err()
        );
    }
}
