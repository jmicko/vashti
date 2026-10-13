//! Request-local Strata decayed-frequency/recency tier. Every144 resident slot
//! can change; no reserved tiny swap partition and no routing history on disk.
use kuno_text::{Error, Result};
use zeroize::{Zeroize, Zeroizing};
pub(crate) const LAYERS: usize = 48;
pub(crate) const EXPERTS: usize = 512;
pub(crate) const RESIDENT: usize = crate::daslab_profile::BANK_SLOTS;
pub(crate) const METADATA_BYTES: usize = LAYERS * EXPERTS * (2 + 4 + 8) + 4096;
pub(crate) struct Cache {
    map: Zeroizing<Vec<u16>>,
    score: Zeroizing<Vec<f32>>,
    recent: Zeroizing<Vec<u64>>,
    clock: u64,
    pub(crate) swaps: u64,
    pub(crate) bytes: u64,
    pub(crate) adaptation_ns: u64,
}
impl Cache {
    pub(crate) fn new(map: Vec<u16>) -> Result<Self> {
        if map.len() != LAYERS * EXPERTS {
            return Err(Error::InvalidBufferLength);
        }
        for row in map.chunks_exact(EXPERTS) {
            let mut seen = [false; EXPERTS];
            for &slot in row {
                if slot as usize >= EXPERTS || seen[slot as usize] {
                    return Err(Error::InvalidFormat("Strata cache bijection"));
                }
                seen[slot as usize] = true;
            }
        }
        Ok(Self {
            map: Zeroizing::new(map),
            score: Zeroizing::new(vec![0.; LAYERS * EXPERTS]),
            recent: Zeroizing::new(vec![0; LAYERS * EXPERTS]),
            clock: 0,
            swaps: 0,
            bytes: 0,
            adaptation_ns: 0,
        })
    }
    pub(crate) fn mapping(&self, layer: usize) -> Result<Zeroizing<[u16; EXPERTS]>> {
        let mut out = Zeroizing::new([0; EXPERTS]);
        out.copy_from_slice(
            self.map
                .get(layer * EXPERTS..(layer + 1) * EXPERTS)
                .ok_or(Error::OutOfBounds)?,
        );
        Ok(out)
    }
    pub(crate) fn observe(&mut self, layer: usize, experts: &[u32]) -> Result<()> {
        if layer >= LAYERS || experts.len() > 40 || experts.iter().any(|&id| id >= EXPERTS as u32) {
            return Err(Error::OutOfBounds);
        }
        for &id in experts {
            let at = layer * EXPERTS + id as usize;
            self.score[at] += 1.;
            self.recent[at] = self.clock + 1;
        }
        Ok(())
    }
    pub(crate) fn plan(&mut self) -> Result<Zeroizing<Vec<[usize; 5]>>> {
        self.clock += 1;
        let mut swaps = Zeroizing::new(Vec::with_capacity(96));
        if self.clock % 4 != 0 {
            return Ok(swaps);
        }
        for layer in 0..LAYERS {
            let first = layer * EXPERTS;
            let mut selected = [false; EXPERTS];
            for _ in 0..2 {
                let newcomer = (0..EXPERTS)
                    .filter(|&i| self.map[first + i] >= RESIDENT as u16 && !selected[i])
                    .max_by(|&a, &b| {
                        self.score[first + a]
                            .total_cmp(&self.score[first + b])
                            .then_with(|| b.cmp(&a))
                    });
                let victim = (0..EXPERTS)
                    .filter(|&i| {
                        self.map[first + i] < RESIDENT as u16
                            && !selected[i]
                            && self.recent[first + i] + 2 < self.clock
                    })
                    .min_by(|&a, &b| {
                        self.score[first + a]
                            .total_cmp(&self.score[first + b])
                            .then_with(|| self.recent[first + a].cmp(&self.recent[first + b]))
                            .then_with(|| a.cmp(&b))
                    });
                if let (Some(new), Some(old)) = (newcomer, victim) {
                    if self.score[first + new] <= self.score[first + old] + 1. {
                        break;
                    }
                    swaps.push([
                        layer,
                        old,
                        new,
                        self.map[first + old] as usize,
                        self.map[first + new] as usize,
                    ]);
                    selected[new] = true;
                    selected[old] = true;
                }
            }
        }
        for score in self.score.iter_mut() {
            *score *= 0.7;
        }
        Ok(swaps)
    }
    pub(crate) fn commit(&mut self, swap: [usize; 5], bytes: u64) -> Result<()> {
        let [layer, old, new, resident, host] = swap;
        if layer >= LAYERS
            || old >= EXPERTS
            || new >= EXPERTS
            || resident >= RESIDENT
            || !(RESIDENT..EXPERTS).contains(&host)
        {
            return Err(Error::OutOfBounds);
        }
        let first = layer * EXPERTS;
        if self.map[first + old] as usize != resident || self.map[first + new] as usize != host {
            return Err(Error::InvalidFormat("Strata cache stale swap"));
        }
        self.map.swap(first + old, first + new);
        self.swaps += 1;
        self.bytes += bytes;
        Ok(())
    }
}
impl Drop for Cache {
    fn drop(&mut self) {
        self.clock.zeroize();
        self.swaps.zeroize();
        self.bytes.zeroize();
        self.adaptation_ns.zeroize();
    }
}
