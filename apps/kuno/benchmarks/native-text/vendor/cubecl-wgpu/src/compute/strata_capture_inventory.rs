//! Capture-local index into the ORIGINAL immutable replay signature. No second
//! view image, resource substitution, access narrowing or completion authority.
use std::{
    cell::Cell,
    mem::size_of,
    sync::atomic::{Ordering, compiler_fence},
};
const MAX_NODES: usize = 32768;
const MAX_ROOTS: usize = 65536;
const MAX_BYTES: usize = 1024 * 1024;
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct Root {
    pub raw: u64,
    pub bytes: u64,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct View {
    pub raw: u64,
    pub first: u64,
    pub bytes: u64,
}
struct Plain<T: Copy>(Vec<T>);
impl<T: Copy> Plain<T> {
    fn new(n: usize) -> Result<Self, &'static str> {
        let mut data = Vec::new();
        data.try_reserve_exact(n)
            .map_err(|_| "native capture index allocation")?;
        Ok(Self(data))
    }
    fn bytes(&self) -> Result<usize, &'static str> {
        self.0
            .capacity()
            .checked_mul(size_of::<T>())
            .ok_or("native capture index size overflow")
    }
    fn erase(&mut self) {
        // T is Copy: all logical values are retired before zeroing the entire
        // exclusive allocation, including uninitialized spare capacity. Only
        // writes occur; aligned words and exact byte edges stay in that span.
        self.0.clear();
        let length = self.0.capacity() * size_of::<T>();
        let p = self.0.as_mut_ptr().cast::<u8>();
        let prefix = p.align_offset(align_of::<u64>()).min(length);
        for i in 0..prefix {
            unsafe { p.add(i).write_volatile(0) }
        }
        for i in 0..(length - prefix) / 8 {
            unsafe { p.add(prefix).cast::<u64>().add(i).write_volatile(0) }
        }
        for i in prefix + (length - prefix) / 8 * 8..length {
            unsafe { p.add(i).write_volatile(0) }
        }
        compiler_fence(Ordering::SeqCst);
    }
}
impl<T: Copy> Drop for Plain<T> {
    fn drop(&mut self) {
        self.erase()
    }
}
pub(super) struct Snapshot {
    offsets: Plain<usize>,
    roots: Plain<Root>,
    address: usize,
    words: usize,
    bytes: usize,
    lookups: Cell<u64>,
    revalidations: Cell<u64>,
}
fn number(n: u64) -> Result<usize, &'static str> {
    usize::try_from(n).map_err(|_| "native capture signature count overflow")
}
fn header(s: &[u64], at: usize) -> Result<(usize, usize, usize, usize), &'static str> {
    let h = s
        .get(
            at..at
                .checked_add(4)
                .ok_or("native capture signature overflow")?,
        )
        .ok_or("native capture truncated node header")?;
    if h[1] > 1 || h[2] > 1 {
        return Err("native capture node flag");
    }
    let desc = number(h[3])?;
    if !(1..=32).contains(&desc) {
        return Err("native capture descriptor count");
    }
    let physical = at
        .checked_add(4)
        .and_then(|n| n.checked_add(desc.checked_mul(3)?))
        .ok_or("native capture signature overflow")?;
    let reads = number(
        *s.get(physical)
            .ok_or("native capture physical count missing")?,
    )?;
    if reads > 4096 {
        return Err("native capture physical count");
    }
    let dispatch = physical
        .checked_add(1)
        .and_then(|n| n.checked_add(reads.checked_mul(3)?))
        .ok_or("native capture signature overflow")?;
    let end = dispatch
        .checked_add(4)
        .ok_or("native capture signature overflow")?;
    let d = s
        .get(dispatch..end)
        .ok_or("native capture truncated dispatch")?;
    match d[0] {
        0 | 1 if h[0] != 0 => {}
        2 if h[0] == 0 && desc == 2 && reads == 0 => {}
        3 if h[0] == 0 && desc == 1 && reads == 0 => {}
        _ => return Err("native capture actual dispatch shape"),
    }
    Ok((physical, dispatch, end, desc))
}
fn view(s: &[u64], at: usize) -> Result<View, &'static str> {
    let v = s
        .get(at..at.checked_add(3).ok_or("native capture view overflow")?)
        .ok_or("native capture view missing")?;
    Ok(View {
        raw: v[0],
        first: v[1],
        bytes: v[2],
    })
}
impl Snapshot {
    pub fn build(
        s: &[u64],
        nodes: usize,
        roots: usize,
        mut root: impl FnMut(usize) -> Result<Root, &'static str>,
        budget: usize,
    ) -> Result<Self, &'static str> {
        if nodes == 0 || nodes > MAX_NODES || roots == 0 || roots > MAX_ROOTS {
            return Err("native capture index inventory capacity");
        }
        let wanted = 4096usize
            .checked_add(
                nodes
                    .checked_mul(size_of::<usize>())
                    .ok_or("native capture index overflow")?,
            )
            .and_then(|n| n.checked_add(roots.checked_mul(2 * size_of::<Root>())?))
            .ok_or("native capture index overflow")?;
        if wanted > budget.min(MAX_BYTES) {
            return Err("native capture index metadata capacity");
        }
        let offsets = Plain::new(nodes)?;
        let mut original = Plain::new(roots)?;
        let mut sorted = Plain::new(roots)?;
        let peak = 4096usize
            .checked_add(offsets.bytes()?)
            .and_then(|n| n.checked_add(original.bytes().ok()?))
            .and_then(|n| n.checked_add(sorted.bytes().ok()?))
            .ok_or("native capture index overflow")?;
        if peak > budget.min(MAX_BYTES) {
            return Err("native capture index actual metadata capacity");
        }
        for i in 0..roots {
            let r = root(i)?;
            if r.raw == 0 || r.bytes == 0 {
                return Err("native capture actual root");
            };
            original.0.push(r);
            sorted.0.push(r)
        }
        sorted.0.sort_unstable_by_key(|r| r.raw);
        let valid = |v: View| -> Result<(), &'static str> {
            let i = sorted
                .0
                .binary_search_by_key(&v.raw, |r| r.raw)
                .map_err(|_| "native capture foreign root")?;
            if v.bytes == 0
                || v.first
                    .checked_add(v.bytes)
                    .is_none_or(|end| end > sorted.0[i].bytes)
            {
                return Err("native capture original full view extent");
            };
            Ok(())
        };
        let mut result = Self {
            offsets,
            roots: original,
            address: s.as_ptr() as usize,
            words: s.len(),
            bytes: 4096 + nodes * size_of::<usize>() + roots * size_of::<Root>(),
            lookups: Cell::new(0),
            revalidations: Cell::new(0),
        };
        result.bytes = 4096 + result.offsets.bytes()? + result.roots.bytes()?;
        let mut at = 0;
        while at < s.len() {
            if result.offsets.0.len() == nodes {
                return Err("native capture extra original node");
            }
            let (physical, dispatch, end, desc) = header(s, at)?;
            for i in 0..desc {
                valid(view(s, at + 4 + 3 * i)?)?
            }
            let reads = number(s[physical])?;
            for i in 0..reads {
                valid(view(s, physical + 1 + 3 * i)?)?
            }
            if s[dispatch] == 1 {
                valid(view(s, dispatch + 1)?)?
            }
            result.offsets.0.push(at);
            at = end;
        }
        if result.offsets.0.len() != nodes {
            return Err("native capture incomplete original nodes");
        }
        Ok(result)
    }
    pub fn metadata_bytes(&self) -> usize {
        self.bytes
    }
    pub fn roots(&self) -> usize {
        self.roots.0.len()
    }
    pub fn check_root(&self, i: usize, r: Root) -> Result<(), &'static str> {
        if self.roots.0.get(i) != Some(&r) {
            return Err("native capture actual guarded root changed");
        };
        Ok(())
    }
    pub fn note_revalidated(&self) {
        self.revalidations
            .set(self.revalidations.get() + self.roots() as u64)
    }
    pub fn check_signature(&self, s: &[u64]) -> Result<(), &'static str> {
        if self.address != s.as_ptr() as usize || self.words != s.len() {
            return Err("native capture original signature owner changed");
        };
        Ok(())
    }
    pub fn stats(&self) -> [u64; 3] {
        [
            self.offsets.0.len() as u64,
            self.lookups.get(),
            self.revalidations.get(),
        ]
    }
    pub fn node<'a>(&'a self, s: &'a [u64], index: usize) -> Result<Node<'a>, &'static str> {
        self.check_signature(s)?;
        let at = *self
            .offsets
            .0
            .get(index)
            .ok_or("native capture original node index")?;
        let (physical, dispatch, _, desc) = header(s, at)?;
        Ok(Node {
            source: s,
            owner: self,
            at,
            physical,
            dispatch,
            desc,
        })
    }
}
impl Drop for Snapshot {
    fn drop(&mut self) {
        self.address = 0;
        self.words = 0;
        self.bytes = 0;
        self.lookups.set(0);
        self.revalidations.set(0);
        compiler_fence(Ordering::SeqCst)
    }
}
pub(super) struct Node<'a> {
    source: &'a [u64],
    owner: &'a Snapshot,
    at: usize,
    physical: usize,
    dispatch: usize,
    desc: usize,
}
impl Node<'_> {
    pub fn descriptors(&self) -> usize {
        self.desc
    }
    pub fn reads(&self) -> usize {
        self.source[self.physical] as usize
    }
    pub fn tag(&self) -> u64 {
        self.source[self.dispatch]
    }
    fn get(&self, at: usize) -> Result<View, &'static str> {
        let v = view(self.source, at)?;
        self.owner.lookups.set(self.owner.lookups.get() + 1);
        Ok(v)
    }
    pub fn descriptor(&self, i: usize) -> Result<View, &'static str> {
        if i >= self.desc {
            return Err("native capture descriptor index");
        };
        self.get(self.at + 4 + 3 * i)
    }
    pub fn physical(&self, i: usize) -> Result<View, &'static str> {
        if i >= self.reads() {
            return Err("native capture physical index");
        };
        self.get(self.physical + 1 + 3 * i)
    }
    pub fn indirect(&self) -> Result<View, &'static str> {
        if self.tag() != 1 {
            return Err("native capture indirect identity");
        };
        self.get(self.dispatch + 1)
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    fn signature() -> Vec<u64> {
        vec![
            91, 0, 0, 2, 1, 16, 48, 1, 128, 32, 2, 2, 0, 64, 1, 0, 256, 1, 3, 8, 12, 0, 0, 0, 1, 1,
            256, 64, 0, 3, 0, 0, 0,
        ]
    }
    fn build(s: &[u64], n: usize, b: usize) -> Result<Snapshot, &'static str> {
        Snapshot::build(
            s,
            n,
            3,
            |i| {
                Ok(Root {
                    raw: i as u64 + 1,
                    bytes: 4096,
                })
            },
            b,
        )
    }
    #[test]
    fn original_signature_views_preserve_aliases_bda_and_indirect_without_copying() {
        let s = signature();
        let c = build(&s, 2, 1 << 20).unwrap();
        let n = c.node(&s, 0).unwrap();
        assert_eq!((n.descriptors(), n.reads(), n.tag()), (2, 2, 1));
        assert_eq!(
            n.descriptor(0).unwrap(),
            View {
                raw: 1,
                first: 16,
                bytes: 48
            }
        );
        assert_eq!(
            n.descriptor(1).unwrap(),
            View {
                raw: 1,
                first: 128,
                bytes: 32
            }
        );
        assert_eq!(
            n.physical(0).unwrap(),
            View {
                raw: 2,
                first: 0,
                bytes: 64
            }
        );
        assert_eq!(
            n.physical(1).unwrap(),
            View {
                raw: 1,
                first: 0,
                bytes: 256
            }
        );
        assert_eq!(
            n.indirect().unwrap(),
            View {
                raw: 3,
                first: 8,
                bytes: 12
            }
        );
        assert_eq!(
            c.node(&s, 1).unwrap().descriptor(0).unwrap(),
            View {
                raw: 1,
                first: 256,
                bytes: 64
            }
        );
        assert_eq!(c.stats(), [2, 6, 0]);
        assert!(n.descriptor(2).is_err());
        assert!(n.physical(2).is_err());
        assert!(c.node(&s, 1).unwrap().indirect().is_err());
        assert!(c.node(&s, 2).is_err());
    }
    #[test]
    fn equal_bytes_are_not_the_original_owner_and_every_root_is_revalidated() {
        let s = signature();
        let c = build(&s, 2, 1 << 20).unwrap();
        let copy = s.clone();
        assert!(c.check_signature(&copy).is_err());
        assert!(c.check_signature(&s[..s.len() - 1]).is_err());
        for i in 0..3 {
            assert!(
                c.check_root(
                    i,
                    Root {
                        raw: i as u64 + 1,
                        bytes: 4096
                    }
                )
                .is_ok()
            );
            assert!(
                c.check_root(
                    i,
                    Root {
                        raw: i as u64 + 9,
                        bytes: 4096
                    }
                )
                .is_err()
            );
            assert!(
                c.check_root(
                    i,
                    Root {
                        raw: i as u64 + 1,
                        bytes: 8192
                    }
                )
                .is_err()
            )
        }
        assert!(
            c.check_root(
                3,
                Root {
                    raw: 4,
                    bytes: 4096
                }
            )
            .is_err()
        );
        c.note_revalidated();
        assert_eq!(c.stats()[2], 3);
    }
    #[test]
    fn malformed_incomplete_foreign_and_overflowing_inventory_declines_the_optimization() {
        let s = signature();
        assert!(build(&s, 1, 1 << 20).is_err());
        assert!(build(&s, 3, 1 << 20).is_err());
        assert!(build(&s, 2, 32).is_err());
        for cut in 0..s.len() {
            assert!(build(&s[..cut], 2, 1 << 20).is_err())
        }
        for (i, v) in [
            (3, 33),
            (4, 99),
            (5, u64::MAX),
            (6, 0),
            (10, 4097),
            (18, 7),
            (20, 0),
        ] {
            let mut bad = s.clone();
            bad[i] = v;
            assert!(build(&bad, 2, 1 << 20).is_err(), "{i}")
        }
    }
    #[test]
    fn actual_plain_metadata_wipe_covers_full_unused_capacity() {
        let mut data = Plain::<u64>::new(128).unwrap();
        unsafe {
            std::ptr::write_bytes(
                data.0.as_mut_ptr().cast::<u8>(),
                0xa5,
                data.bytes().unwrap(),
            )
        };
        data.0.extend([7, 8, 9]);
        data.erase();
        assert!(data.0.is_empty());
        let bytes = unsafe {
            std::slice::from_raw_parts(data.0.as_ptr().cast::<u8>(), data.bytes().unwrap())
        };
        assert!(bytes.iter().all(|b| *b == 0));
    }
}
