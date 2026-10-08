//! Host-only policy diagnostics using the production completed-token schedule.
//! Include this file as a cfg(test) child module of adaptive_experts. All routes
//! are fixed public fixtures; no source fills, GPU completion or throughput is
//! established by these tests.
use super::*;

const ROUTES: usize = 10;
const WORKING: usize = 50;
const FIRST_NEW: usize = 200;
const GROUPS: usize = WORKING / ROUTES;
const PREFILL_TOKENS: usize = 75;
const REQUIRED_INSERTIONS: usize = LAYERS * WORKING;
const MIN_SHIFT_TOKENS: usize = REQUIRED_INSERTIONS.div_ceil(BUDGET);
const MAX_SHIFT_TOKENS: usize = 2000;
const STABLE_TOKENS: usize = 2 * LAYERS * GROUPS;

struct Fixture {
    cache: Cache,
    banks: Vec<BankSlots>,
    promotions: usize,
    private_peak: usize,
}

impl Fixture {
    fn new() -> Self {
        assert_eq!((LAYERS, CAPACITY, EXPERTS, BUDGET), (48, 144, 512, 4));
        assert_eq!(WORKING % ROUTES, 0);
        let mut banks = Vec::with_capacity(LAYERS);
        for _ in 0..LAYERS {
            let mut bank = BankSlots::new(CAPACITY, EXPERTS).unwrap();
            let selected = Zeroizing::new((0..CAPACITY).collect::<Vec<_>>());
            let plan = bank.plan(&selected).unwrap();
            bank.commit(plan).unwrap();
            assert!(
                (FIRST_NEW..FIRST_NEW + WORKING)
                    .all(|expert| bank.resident_slot(expert).unwrap().is_none())
            );
            banks.push(bank);
        }
        let mut cache = Cache::new().unwrap();
        let original_host: usize = banks
            .iter()
            .map(|bank| bank.original_metadata_bytes().unwrap())
            .sum();
        assert!(original_host <= 640 * 1024);
        // Actual ten-route observations establish finite old popularity before
        // the shift. The starting GPU bank contains every old fixture expert.
        for token in 0..PREFILL_TOKENS {
            for layer in 0..LAYERS {
                let ids: [usize; ROUTES] =
                    std::array::from_fn(|rank| (token * ROUTES + layer * 3 + rank) % CAPACITY);
                cache
                    .record(layer, token as u64 + 1, ids.into_iter())
                    .unwrap();
            }
        }
        cache.ready = true;
        assert!(cache.owned_bytes() <= HOST);
        Self {
            cache,
            banks,
            promotions: 0,
            private_peak: 0,
        }
    }

    fn routes(layer: usize, shift_token: usize) -> [usize; ROUTES] {
        // Each group contains ten distinct experts. Every one of the fifty
        // experts is selected once per five tokens, with a public layer offset.
        let group = (shift_token - 1 + layer) % GROUPS;
        std::array::from_fn(|rank| FIRST_NEW + group * ROUTES + rank)
    }

    fn token_boundary(&mut self, shift_token: usize) -> usize {
        let position = (PREFILL_TOKENS + shift_token) as u64;
        // Observe every accepted layer before choosing any promotion, matching
        // the all-family commit boundary. Actual GPU hits are observed too.
        for layer in 0..LAYERS {
            self.cache
                .record(
                    layer,
                    position,
                    Self::routes(layer, shift_token).into_iter(),
                )
                .unwrap();
        }

        let start = self.cache.first_layer();
        let mut next = (start + 1) % LAYERS;
        let mut pending = Vec::with_capacity(BUDGET);
        let mut touched = Zeroizing::new([false; LAYERS]);
        // The production reservation includes a conservative 4096-byte frame
        // allowance for the bounded pending vector, route scratch and helpers.
        let mut private_bytes = self.cache.owned_bytes() + 4096;
        for offset in 0..LAYERS {
            let layer = (start + offset) % LAYERS;
            let bank = &mut self.banks[layer];
            let Some((expert, victim)) = self.cache.candidate(layer, bank).unwrap() else {
                continue;
            };
            assert!(!touched[layer], "more than one promotion for one layer");
            touched[layer] = true;
            let selected = Zeroizing::new([expert]);
            let routes = Self::routes(layer, shift_token);
            assert!(routes.contains(&expert), "promotion is not a current miss");
            assert!(bank.resident_slot(expert).unwrap().is_none());
            if let Some(resident) = bank.resident_expert(victim).unwrap() {
                assert!(
                    !routes.contains(&resident),
                    "a current GPU selection was chosen as a victim"
                );
            }
            let plan = bank.plan_in_slots(&*selected, &[victim]).unwrap();
            bank.validate_pending(&plan, &*selected).unwrap();
            assert_eq!(plan.uploads.as_slice(), &[(expert, victim)]);
            private_bytes += plan.owned_metadata_bytes().unwrap();
            assert!(private_bytes <= HOST, "live adaptive plans exceed HOST");
            pending.push((layer, plan));
            next = (layer + 1) % LAYERS;
            if pending.len() == BUDGET {
                break;
            }
        }
        assert!(pending.len() <= BUDGET);
        self.private_peak = self.private_peak.max(private_bytes);
        let promoted = pending.len();
        // Plans coexist until the simulated completed grouped-fill boundary.
        // Host commits here assert scheduling only, without claiming GPU proof.
        for (layer, plan) in pending {
            self.banks[layer].commit(plan).unwrap();
        }
        self.promotions += promoted;
        self.cache.advance(next).unwrap();
        assert_eq!(self.cache.first_layer(), next);
        promoted
    }

    fn resident_new(&self) -> usize {
        self.banks
            .iter()
            .map(|bank| {
                (FIRST_NEW..FIRST_NEW + WORKING)
                    .filter(|&expert| bank.resident_slot(expert).unwrap().is_some())
                    .count()
            })
            .sum()
    }
}

impl Drop for Fixture {
    fn drop(&mut self) {
        self.cache.erase();
        // BankSlots owns its independently wiping state and plan metadata.
    }
}

#[test]
fn fifty_expert_shift_obeys_real_token_budget_and_then_stops_churning() {
    let mut fixture = Fixture::new();
    assert_eq!(MIN_SHIFT_TOKENS, 600);
    let mut settled_at = None;
    for token in 1..=MAX_SHIFT_TOKENS {
        fixture.token_boundary(token);
        let resident = fixture.resident_new();
        assert!(resident <= fixture.promotions);
        assert!(fixture.promotions <= token * BUDGET);
        if resident == REQUIRED_INSERTIONS {
            settled_at = Some(token);
            break;
        }
        assert!(
            token < MAX_SHIFT_TOKENS,
            "fifty-expert shift did not settle"
        );
    }
    let settled_at = settled_at.expect("fifty-expert shift did not settle");
    assert!(settled_at >= MIN_SHIFT_TOKENS);
    let promotions_at_settlement = fixture.promotions;
    // Membership alone would miss slot churn. Snapshot the exact fifty slot
    // assignments in a wiping owner, charged separately as test-only storage.
    let slots = Zeroizing::new(
        fixture
            .banks
            .iter()
            .flat_map(|bank| {
                (FIRST_NEW..FIRST_NEW + WORKING)
                    .map(move |expert| bank.resident_slot(expert).unwrap().unwrap())
            })
            .collect::<Vec<_>>(),
    );
    assert!(
        fixture.private_peak + slots.capacity() * std::mem::size_of::<usize>() <= HOST,
        "even the test-only slot snapshot must fit alongside adaptive plans"
    );
    for token in settled_at + 1..=settled_at + STABLE_TOKENS {
        assert_eq!(fixture.token_boundary(token), 0, "settled cache churned");
        assert_eq!(fixture.resident_new(), REQUIRED_INSERTIONS);
        for (layer, bank) in fixture.banks.iter().enumerate() {
            for (index, expert) in (FIRST_NEW..FIRST_NEW + WORKING).enumerate() {
                assert_eq!(
                    bank.resident_slot(expert).unwrap(),
                    Some(slots[layer * WORKING + index]),
                    "settled expert moved to another slot"
                );
            }
        }
    }
    assert_eq!(fixture.promotions, promotions_at_settlement);
    assert!(fixture.private_peak <= HOST);
}

#[test]
fn original_output_length_cannot_settle_all_fifty_expert_layer_shifts() {
    let mut fixture = Fixture::new();
    let output_tokens = 128;
    for token in 1..=output_tokens {
        fixture.token_boundary(token);
    }
    assert!(fixture.promotions <= output_tokens * BUDGET);
    assert!(fixture.resident_new() <= output_tokens * BUDGET);
    assert!(fixture.resident_new() < REQUIRED_INSERTIONS);
    assert!(fixture.private_peak <= HOST);
}
