//! Reusable capture-only host containers. They grant no access or completion;
//! callers refill them from every original resource with the original checks.
use ash::vk;
use std::{collections::HashMap, mem::size_of};

// Original per-node bound: 32 descriptors, 4096 physical reads, one indirect
// resource. Copy/fill require fewer. No range or owner is merged by this bound.
const ACCESS_CAP: usize = 32 + 4096 + 1;
const BYTES_CAP: usize = 1024 * 1024;

pub(super) struct Scratch<A> {
    pub ranges: Vec<A>,
    pub accesses: HashMap<vk::Buffer, vk::AccessFlags>,
    pub barriers: Vec<vk::BufferMemoryBarrier<'static>>,
}
impl<A> Scratch<A> {
    pub(super) fn new(count: usize) -> Result<Self, &'static str> {
        if count == 0 || count > ACCESS_CAP {
            return Err("native capture access capacity");
        }
        let mut result = Self {
            ranges: Vec::new(),
            accesses: HashMap::new(),
            barriers: Vec::new(),
        };
        result
            .ranges
            .try_reserve_exact(count)
            .map_err(|_| "native capture range allocation")?;
        result
            .accesses
            .try_reserve(count)
            .map_err(|_| "native capture access allocation")?;
        result
            .barriers
            .try_reserve_exact(count)
            .map_err(|_| "native capture barrier allocation")?;
        result.metadata_bytes()?;
        Ok(result)
    }
    pub(super) fn metadata_bytes(&self) -> Result<usize, &'static str> {
        // HashMap capacity is usable entries, not raw buckets. Twice its
        // capacity bounds the bucket count including control bytes and aligns
        // every entry conservatively. Include headers, stack and alignment.
        let bytes = self
            .ranges
            .capacity()
            .checked_mul(size_of::<A>())
            .and_then(|n| {
                n.checked_add(
                    self.barriers
                        .capacity()
                        .checked_mul(size_of::<vk::BufferMemoryBarrier<'static>>())?,
                )
            })
            .and_then(|n| {
                n.checked_add(
                    self.accesses
                        .capacity()
                        .checked_mul(2)?
                        .checked_mul(size_of::<(vk::Buffer, vk::AccessFlags)>() + 1)?,
                )
            })
            .and_then(|n| n.checked_add(4096))
            .ok_or("native capture scratch metadata overflow")?;
        if bytes > BYTES_CAP {
            return Err("native capture scratch metadata capacity");
        }
        Ok(bytes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ash::vk::Handle;

    #[test]
    fn real_capacity_is_bounded_and_reused_without_losing_alias_accesses() {
        assert!(Scratch::<[u64; 4]>::new(0).is_err());
        assert!(Scratch::<[u64; 4]>::new(ACCESS_CAP + 1).is_err());
        let mut scratch = Scratch::<[u64; 4]>::new(ACCESS_CAP).unwrap();
        let budget = scratch.metadata_bytes().unwrap();
        assert!(budget < BYTES_CAP);
        let capacities = (
            scratch.ranges.capacity(),
            scratch.accesses.capacity(),
            scratch.barriers.capacity(),
        );
        for round in 0..4 {
            scratch.ranges.clear();
            scratch.accesses.clear();
            scratch.barriers.clear();
            for i in 0..ACCESS_CAP {
                let buffer = vk::Buffer::from_raw((i / 2 + 1) as u64);
                let access = if i % 2 == 0 {
                    vk::AccessFlags::SHADER_READ
                } else {
                    vk::AccessFlags::SHADER_WRITE
                };
                scratch
                    .ranges
                    .push([buffer.as_raw(), i as u64, round, access.as_raw() as u64]);
                *scratch.accesses.entry(buffer).or_default() |= access;
            }
            for (&buffer, &access) in &scratch.accesses {
                scratch.barriers.push(
                    vk::BufferMemoryBarrier::default()
                        .buffer(buffer)
                        .dst_access_mask(access),
                );
            }
            assert_eq!(scratch.ranges.len(), ACCESS_CAP);
            assert_eq!(scratch.ranges[1][1], 1);
            assert_eq!(
                scratch.accesses[&vk::Buffer::from_raw(1)],
                vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE
            );
            assert_eq!(
                capacities,
                (
                    scratch.ranges.capacity(),
                    scratch.accesses.capacity(),
                    scratch.barriers.capacity()
                )
            );
            assert_eq!(budget, scratch.metadata_bytes().unwrap());
        }
    }
}
