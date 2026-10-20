//! Exact expanded replay semantics with capture-local complete source lists.
//! Only genuine immutable source-owner lists can be interned by the producer.
#[cfg(test)]
use super::capture_inventory::Root;
use super::capture_inventory::{Plain, View};
use ash::vk;
use std::{mem::size_of, ops::Range};
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Span {
    first: usize,
    len: usize,
}
impl Span {
    fn range(self) -> Range<usize> {
        self.first..self.first + self.len
    }
}
#[derive(Clone, Copy)]
pub(super) struct Entry {
    pub view: View,
    pub flags: vk::AccessFlags,
    pub atomic_or: bool,
}
#[derive(Clone, Copy, Debug)]
enum Reads {
    Inline(Span),
    Shared(usize),
}
#[derive(Clone, Copy)]
pub(super) struct Record {
    pipeline: u64,
    shared: bool,
    release: bool,
    desc: Span,
    reads: Reads,
    dispatch: [u64; 4],
}
pub(super) struct Compact {
    nodes: Plain<Record>,
    views: Plain<Entry>,
    bundles: Plain<Span>,
    expected: usize,
    expected_views: usize,
    expected_bundles: usize,
    words: usize,
}
pub(super) enum Signature {
    Flat(Vec<u64>),
    Compact(Compact),
}
pub(super) struct Builder {
    value: Signature,
    expected_nodes: usize,
    expected_words: usize,
    seen: usize,
}
fn append_flat(target: &mut Vec<u64>, words: &[u64]) -> Result<(), &'static str> {
    if target
        .len()
        .checked_add(words.len())
        .is_none_or(|n| n > target.capacity())
    {
        return Err("native signature precharged flat capacity");
    }
    target.extend_from_slice(words);
    Ok(())
}
fn valid(v: View) -> Result<(), &'static str> {
    if v.raw == 0 || v.bytes == 0 || v.first.checked_add(v.bytes).is_none() {
        Err("native signature original view")
    } else {
        Ok(())
    }
}
impl Builder {
    pub fn new(
        factor: bool,
        nodes: usize,
        views: usize,
        bundles: usize,
        words: usize,
        budget: usize,
    ) -> Result<Self, &'static str> {
        if !(1..=32768).contains(&nodes) || bundles > 32768 {
            return Err("native signature original inventory");
        }
        let compact = 4096usize
            .checked_add(
                nodes
                    .checked_mul(size_of::<Record>())
                    .ok_or("native signature size overflow")?,
            )
            .and_then(|n| n.checked_add(views.checked_mul(size_of::<Entry>())?))
            .and_then(|n| n.checked_add(bundles.checked_mul(size_of::<Span>())?));
        let value = if factor && compact.is_some_and(|n| n <= budget) {
            let c = Compact {
                nodes: Plain::new(nodes)?,
                views: Plain::new(views)?,
                bundles: Plain::new(bundles)?,
                expected: nodes,
                expected_views: views,
                expected_bundles: bundles,
                words: 0,
            };
            let s = Signature::Compact(c);
            if s.metadata_bytes()? > budget {
                return Err("native signature actual compact capacity");
            };
            s
        } else {
            if words
                .checked_mul(8)
                .and_then(|n| n.checked_add(4096))
                .is_none_or(|n| n > budget)
            {
                return Err("native signature flat capacity");
            }
            let mut v = Vec::new();
            v.try_reserve_exact(words)
                .map_err(|_| "native signature flat allocation")?;
            let s = Signature::Flat(v);
            if s.metadata_bytes()? > budget {
                return Err("native signature actual flat capacity");
            };
            s
        };
        Ok(Self {
            value,
            expected_nodes: nodes,
            expected_words: words,
            seen: 0,
        })
    }
    pub fn metadata_bytes(&self) -> Result<usize, &'static str> {
        self.value.metadata_bytes()
    }
    pub fn factored(&self) -> bool {
        matches!(self.value, Signature::Compact(_))
    }
    pub fn bundle(
        &mut self,
        views: impl Iterator<Item = Result<View, &'static str>>,
    ) -> Result<(), &'static str> {
        let Signature::Compact(c) = &mut self.value else {
            return Err("native signature flat cannot intern");
        };
        if c.bundles.0.len() == c.expected_bundles {
            return Err("native signature bundle capacity");
        }
        let first = c.views.0.len();
        for v in views {
            let v = v?;
            valid(v)?;
            if c.views.0.len() == c.expected_views {
                return Err("native signature view capacity");
            };
            c.views.0.push(Entry {
                view: v,
                flags: vk::AccessFlags::SHADER_READ,
                atomic_or: false,
            })
        }
        if c.views.0.len() == first || c.views.0.len() - first > 4096 {
            return Err("native signature source bundle extent");
        }
        c.bundles.0.push(Span {
            first,
            len: c.views.0.len() - first,
        });
        Ok(())
    }
    pub fn node(
        &mut self,
        pipeline: u64,
        shared: bool,
        release: bool,
        ndesc: usize,
        nreads: usize,
        bundle: Option<usize>,
        dispatch: [u64; 4],
        desc: impl Iterator<Item = Result<Entry, &'static str>>,
        reads: impl Iterator<Item = Result<View, &'static str>>,
    ) -> Result<(), &'static str> {
        if self.seen == self.expected_nodes || !(1..=32).contains(&ndesc) || nreads > 4096 {
            return Err("native signature node count");
        }
        match dispatch[0] {
            0 | 1 if pipeline != 0 => {}
            2 if pipeline == 0 && ndesc == 2 && nreads == 0 => {}
            3 if pipeline == 0 && ndesc == 1 && nreads == 0 => {}
            _ => return Err("native signature original dispatch"),
        }
        if dispatch[0] == 1 {
            valid(View {
                raw: dispatch[1],
                first: dispatch[2],
                bytes: dispatch[3],
            })?
        }
        match &mut self.value {
            Signature::Flat(v) => {
                if bundle.is_some() {
                    return Err("native signature flat shared reference");
                }
                append_flat(v, &[pipeline, shared as u64, release as u64, ndesc as u64])?;
                let mut seen = 0;
                for e in desc {
                    let e = e?;
                    valid(e.view)?;
                    append_flat(v, &[e.view.raw, e.view.first, e.view.bytes])?;
                    seen += 1;
                }
                if seen != ndesc {
                    return Err("native signature descriptor inventory");
                }
                append_flat(v, &[nreads as u64])?;
                let mut seen = 0;
                for e in reads {
                    let e = e?;
                    valid(e)?;
                    append_flat(v, &[e.raw, e.first, e.bytes])?;
                    seen += 1;
                }
                if seen != nreads {
                    return Err("native signature physical inventory");
                };
                append_flat(v, &dispatch)?;
            }
            Signature::Compact(c) => {
                let start = c.views.0.len();
                let mut seen = 0;
                for e in desc {
                    let e = e?;
                    valid(e.view)?;
                    if c.views.0.len() == c.expected_views {
                        return Err("native signature view capacity");
                    };
                    c.views.0.push(e);
                    seen += 1;
                }
                if seen != ndesc {
                    return Err("native signature descriptor inventory");
                }
                let desc = Span {
                    first: start,
                    len: seen,
                };
                let reads = if let Some(i) = bundle {
                    if c.bundles.0.get(i).is_none_or(|r| r.len != nreads) {
                        return Err("native signature full source reference");
                    };
                    Reads::Shared(i)
                } else {
                    let first = c.views.0.len();
                    let mut seen = 0;
                    for v in reads {
                        let v = v?;
                        valid(v)?;
                        if c.views.0.len() == c.expected_views {
                            return Err("native signature view capacity");
                        };
                        c.views.0.push(Entry {
                            view: v,
                            flags: vk::AccessFlags::SHADER_READ,
                            atomic_or: false,
                        });
                        seen += 1;
                    }
                    if seen != nreads {
                        return Err("native signature physical inventory");
                    };
                    Reads::Inline(Span { first, len: seen })
                };
                c.nodes.0.push(Record {
                    pipeline,
                    shared,
                    release,
                    desc,
                    reads,
                    dispatch,
                });
                c.words = c
                    .words
                    .checked_add(9 + 3 * (ndesc + nreads))
                    .ok_or("native signature word overflow")?;
            }
        }
        self.seen += 1;
        Ok(())
    }
    pub fn finish(self) -> Result<Signature, &'static str> {
        if self.seen != self.expected_nodes {
            return Err("native signature incomplete original nodes");
        }
        match &self.value {
            Signature::Flat(v) if v.len() != self.expected_words => {
                return Err("native signature expanded word inventory");
            }
            Signature::Compact(c)
                if c.nodes.0.len() != c.expected
                    || c.bundles.0.len() != c.expected_bundles
                    || c.views.0.len() != c.expected_views
                    || c.words != self.expected_words =>
            {
                return Err("native signature complete source inventory");
            }
            _ => {}
        }
        Ok(self.value)
    }
}
impl Compact {
    fn reads(&self, n: &Record) -> Span {
        match n.reads {
            Reads::Inline(r) => r,
            Reads::Shared(i) => self.bundles.0[i],
        }
    }
    pub fn identity(&self) -> [usize; 6] {
        [
            self.nodes.0.as_ptr() as usize,
            self.nodes.0.len(),
            self.views.0.as_ptr() as usize,
            self.views.0.len(),
            self.bundles.0.as_ptr() as usize,
            self.bundles.0.len(),
        ]
    }
    pub fn source_lists(&self) -> usize {
        self.bundles.0.len()
    }
    pub fn nodes(&self) -> usize {
        self.nodes.0.len()
    }
    pub fn views(&self) -> impl Iterator<Item = View> + '_ {
        self.views.0.iter().map(|v| v.view).chain(
            self.nodes
                .0
                .iter()
                .filter(|n| n.dispatch[0] == 1)
                .map(|n| View {
                    raw: n.dispatch[1],
                    first: n.dispatch[2],
                    bytes: n.dispatch[3],
                }),
        )
    }
    pub fn node<'a>(&'a self, index: usize) -> Result<Node<'a>, &'static str> {
        let data = self
            .nodes
            .0
            .get(index)
            .ok_or("native signature original node index")?;
        Ok(Node::Compact {
            owner: self,
            data,
            scope: None,
            seal: None,
        })
    }
    fn word(&self, index: usize, at: usize) -> Option<u64> {
        let n = self.nodes.0.get(index)?;
        let reads = self.reads(n);
        if at < 4 {
            return Some(
                [
                    n.pipeline,
                    n.shared as u64,
                    n.release as u64,
                    n.desc.len as u64,
                ][at],
            );
        };
        let mut p = at - 4;
        if p < 3 * n.desc.len {
            let v = self.views.0[n.desc.first + p / 3].view;
            return Some([v.raw, v.first, v.bytes][p % 3]);
        };
        p -= 3 * n.desc.len;
        if p == 0 {
            return Some(reads.len as u64);
        };
        p -= 1;
        if p < 3 * reads.len {
            let v = self.views.0[reads.first + p / 3].view;
            return Some([v.raw, v.first, v.bytes][p % 3]);
        };
        p -= 3 * reads.len;
        n.dispatch.get(p).copied()
    }
}
impl Signature {
    pub fn metadata_bytes(&self) -> Result<usize, &'static str> {
        match self {
            Self::Flat(v) => 4096usize
                .checked_add(
                    v.capacity()
                        .checked_mul(8)
                        .ok_or("native signature metadata overflow")?,
                )
                .ok_or("native signature metadata overflow"),
            Self::Compact(c) => 4096usize
                .checked_add(c.nodes.bytes()?)
                .and_then(|n| n.checked_add(c.views.bytes().ok()?))
                .and_then(|n| n.checked_add(c.bundles.bytes().ok()?))
                .ok_or("native signature metadata overflow"),
        }
    }
    pub fn flat(&self) -> Option<&[u64]> {
        match self {
            Self::Flat(v) => Some(v),
            _ => None,
        }
    }
    pub fn compact(&self) -> Option<&Compact> {
        match self {
            Self::Compact(c) => Some(c),
            _ => None,
        }
    }
    pub fn words(&self) -> Words<'_> {
        Words {
            owner: self,
            node: 0,
            at: 0,
        }
    }
}
pub(super) struct Words<'a> {
    owner: &'a Signature,
    node: usize,
    at: usize,
}
impl Iterator for Words<'_> {
    type Item = u64;
    fn next(&mut self) -> Option<u64> {
        match self.owner {
            Signature::Flat(v) => {
                let r = v.get(self.at).copied();
                self.at += usize::from(r.is_some());
                r
            }
            Signature::Compact(c) => loop {
                if self.node == c.nodes() {
                    return None;
                };
                if let Some(v) = c.word(self.node, self.at) {
                    self.at += 1;
                    return Some(v);
                };
                self.node += 1;
                self.at = 0
            },
        }
    }
}
impl PartialEq for Signature {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Flat(a), Self::Flat(b)) => a == b,
            (Self::Compact(a), Self::Compact(b)) => {
                if a.nodes() != b.nodes() {
                    return false;
                }
                let bundles_equal = a.bundles.0.len() == b.bundles.0.len()
                    && a.bundles
                        .0
                        .iter()
                        .zip(&b.bundles.0)
                        .all(|(x, y)| same(&a.views.0[x.range()], &b.views.0[y.range()]));
                a.nodes.0.iter().zip(&b.nodes.0).all(|(x,y)|x.pipeline==y.pipeline&&x.shared==y.shared&&x.release==y.release&&x.dispatch==y.dispatch&&same(&a.views.0[x.desc.range()],&b.views.0[y.desc.range()])&&{
   if bundles_equal&&matches!((x.reads,y.reads),(Reads::Shared(i),Reads::Shared(j)) if i==j){true}else{same(&a.views.0[a.reads(x).range()],&b.views.0[b.reads(y).range()])}
  })
            }
            _ => self.words().eq(other.words()),
        }
    }
}
fn same(a: &[Entry], b: &[Entry]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(a, b)| a.view == b.view)
}
/// A complete no-writer certificate; it never narrows a full source view.
pub(super) struct Readonly {
    roots: Plain<View>,
    active: Plain<usize>,
    ranges: Plain<Span>,
    identity: [usize; 6],
    bytes: usize,
}
impl Readonly {
    pub fn build(
        c: &Compact,
        mut root_size: impl FnMut(u64) -> Option<u64>,
        budget: usize,
    ) -> Result<Self, &'static str> {
        let supported = vk::AccessFlags::SHADER_READ
            | vk::AccessFlags::SHADER_WRITE
            | vk::AccessFlags::UNIFORM_READ
            | vk::AccessFlags::INDIRECT_COMMAND_READ
            | vk::AccessFlags::TRANSFER_READ
            | vk::AccessFlags::TRANSFER_WRITE;
        if c.views.0.iter().any(|v| !supported.contains(v.flags)) {
            return Err("native readonly unknown original access class");
        }
        let count = c.views.0.len();
        let bundle_views = c
            .bundles
            .0
            .iter()
            .try_fold(0usize, |n, r| n.checked_add(r.len))
            .ok_or("native readonly count overflow")?;
        let wanted = 4096usize
            .checked_add(
                count
                    .checked_mul(8)
                    .ok_or("native readonly size overflow")?,
            )
            .and_then(|n| {
                n.checked_add(
                    bundle_views.checked_mul(size_of::<View>() + size_of::<usize>() + 3 * 32)?,
                )
            })
            .and_then(|n| n.checked_add(c.bundles.0.len().checked_mul(size_of::<Span>())?))
            .ok_or("native readonly size overflow")?;
        if wanted > budget {
            return Err("native readonly metadata capacity");
        }
        let mut writes = Plain::<u64>::new(count)?;
        for v in &c.views.0 {
            if v.flags
                .intersects(vk::AccessFlags::SHADER_WRITE | vk::AccessFlags::TRANSFER_WRITE)
            {
                writes.0.push(v.view.raw)
            }
        }
        writes.0.sort_unstable();
        writes.0.dedup();
        let mut roots = Plain::new(bundle_views)?;
        for range in &c.bundles.0 {
            for v in &c.views.0[range.range()] {
                if v.view.first == 0
                    && root_size(v.view.raw) == Some(v.view.bytes)
                    && writes.0.binary_search(&v.view.raw).is_err()
                {
                    roots.0.push(v.view)
                }
            }
        }
        roots.0.sort_unstable_by_key(|r| r.raw);
        roots.0.dedup_by_key(|r| r.raw);
        let mut active = Plain::new(bundle_views)?;
        let mut ranges = Plain::new(c.bundles.0.len())?;
        for r in &c.bundles.0 {
            let first = active.0.len();
            for (i, v) in c.views.0[r.range()].iter().enumerate() {
                if roots
                    .0
                    .binary_search_by_key(&v.view.raw, |v| v.raw)
                    .is_err()
                {
                    active.0.push(i)
                }
            }
            ranges.0.push(Span {
                first,
                len: active.0.len() - first,
            })
        }
        let bytes =
            4096usize + roots.bytes()? + active.bytes()? + ranges.bytes()? + roots.0.len() * 3 * 32;
        if bytes + writes.bytes()? > budget {
            return Err("native readonly actual metadata capacity");
        }
        Ok(Self {
            roots,
            active,
            ranges,
            identity: c.identity(),
            bytes,
        })
    }
    pub fn factored_count(&self, c: &Compact) -> Result<u64, &'static str> {
        if self.identity != c.identity() {
            return Err("native readonly signature owner changed");
        }
        c.nodes.0.iter().try_fold(0u64, |total, n| {
            let n = match n.reads {
                Reads::Shared(i) => c.bundles.0[i].len - self.ranges.0[i].len,
                _ => 0,
            };
            total
                .checked_add(n as u64)
                .ok_or("native readonly count overflow")
        })
    }
    pub fn metadata_bytes(&self) -> usize {
        self.bytes
    }
    pub fn roots(&self) -> &[View] {
        &self.roots.0
    }
    pub fn contains(&self, raw: u64) -> bool {
        self.roots.0.binary_search_by_key(&raw, |r| r.raw).is_ok()
    }
    fn active(&self, c: &Compact, source: usize) -> Result<&[usize], &'static str> {
        if self.identity != c.identity() {
            return Err("native readonly signature owner changed");
        };
        let r = *self
            .ranges
            .0
            .get(source)
            .ok_or("native readonly source reference")?;
        Ok(&self.active.0[r.range()])
    }
}
pub(super) enum Node<'a> {
    Flat(super::capture_inventory::Node<'a>),
    Compact {
        owner: &'a Compact,
        data: &'a Record,
        scope: Option<&'a super::capture_inventory::Snapshot>,
        seal: Option<&'a Readonly>,
    },
}
impl Node<'_> {
    pub fn attach<'a>(
        self,
        scope: &'a super::capture_inventory::Snapshot,
        seal: Option<&'a Readonly>,
    ) -> Node<'a>
    where
        Self: 'a,
    {
        match self {
            Self::Compact { owner, data, .. } => Node::Compact {
                owner,
                data,
                scope: Some(scope),
                seal,
            },
            Self::Flat(v) => Node::Flat(v),
        }
    }
    pub fn descriptors(&self) -> usize {
        match self {
            Self::Flat(v) => v.descriptors(),
            Self::Compact { data, .. } => data.desc.len,
        }
    }
    pub fn reads(&self) -> usize {
        match self {
            Self::Flat(v) => v.reads(),
            Self::Compact { owner, data, .. } => owner.reads(data).len,
        }
    }
    pub fn tag(&self) -> u64 {
        match self {
            Self::Flat(v) => v.tag(),
            Self::Compact { data, .. } => data.dispatch[0],
        }
    }
    fn note(&self, v: View) -> View {
        if let Self::Compact { scope: Some(s), .. } = self {
            s.note_view()
        };
        v
    }
    pub fn descriptor(&self, i: usize) -> Result<View, &'static str> {
        match self {
            Self::Flat(v) => v.descriptor(i),
            Self::Compact { owner, data, .. } => {
                if i >= data.desc.len {
                    return Err("native signature descriptor index");
                };
                Ok(self.note(owner.views.0[data.desc.first + i].view))
            }
        }
    }
    pub fn physical(&self, i: usize) -> Result<View, &'static str> {
        match self {
            Self::Flat(v) => v.physical(i),
            Self::Compact { owner, data, .. } => {
                let r = owner.reads(data);
                if i >= r.len {
                    return Err("native signature physical index");
                };
                Ok(self.note(owner.views.0[r.first + i].view))
            }
        }
    }
    pub fn indirect(&self) -> Result<View, &'static str> {
        match self {
            Self::Flat(v) => v.indirect(),
            Self::Compact { data, .. } => {
                if data.dispatch[0] != 1 {
                    return Err("native signature indirect identity");
                };
                Ok(self.note(View {
                    raw: data.dispatch[1],
                    first: data.dispatch[2],
                    bytes: data.dispatch[3],
                }))
            }
        }
    }
    pub fn active_indices(&self) -> Result<Active<'_>, &'static str> {
        match self {
            Self::Compact {
                owner,
                data:
                    Record {
                        reads: Reads::Shared(i),
                        ..
                    },
                seal: Some(seal),
                ..
            } => Ok(Active::Selected(seal.active(owner, *i)?.iter())),
            _ => Ok(Active::All(0..self.reads())),
        }
    }
}
pub(super) enum Active<'a> {
    All(Range<usize>),
    Selected(std::slice::Iter<'a, usize>),
}
impl Iterator for Active<'_> {
    type Item = usize;
    fn next(&mut self) -> Option<usize> {
        match self {
            Self::All(i) => i.next(),
            Self::Selected(i) => i.next().copied(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::strata_schedule as schedule;
    fn v(raw: u64, first: u64, bytes: u64) -> View {
        View { raw, first, bytes }
    }
    #[derive(Clone)]
    struct Input {
        pipeline: u64,
        shared: bool,
        release: bool,
        desc: Vec<Entry>,
        reads: Vec<View>,
        dispatch: [u64; 4],
        bundle: Option<usize>,
    }
    fn inputs() -> Vec<Input> {
        let reads = vec![v(10, 0, 1024), v(20, 0, 1024), v(10, 0, 1024)];
        vec![
            Input {
                pipeline: 101,
                shared: false,
                release: false,
                desc: vec![Entry {
                    view: v(30, 64, 64),
                    flags: vk::AccessFlags::SHADER_WRITE,
                    atomic_or: false,
                }],
                reads: reads.clone(),
                dispatch: [0, 2, 1, 1],
                bundle: Some(0),
            },
            Input {
                pipeline: 0,
                shared: false,
                release: false,
                desc: vec![
                    Entry {
                        view: v(40, 0, 32),
                        flags: vk::AccessFlags::TRANSFER_READ,
                        atomic_or: false,
                    },
                    Entry {
                        view: v(20, 8, 32),
                        flags: vk::AccessFlags::TRANSFER_WRITE,
                        atomic_or: false,
                    },
                ],
                reads: vec![],
                dispatch: [2, 0, 0, 0],
                bundle: None,
            },
            Input {
                pipeline: 102,
                shared: true,
                release: true,
                desc: vec![Entry {
                    view: v(50, 0, 128),
                    flags: vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE,
                    atomic_or: false,
                }],
                reads,
                dispatch: [1, 60, 0, 12],
                bundle: Some(0),
            },
        ]
    }
    fn build(
        factor: bool,
        input: &[Input],
        sources: &[Vec<View>],
    ) -> Result<Signature, &'static str> {
        let own = input
            .iter()
            .map(|n| n.desc.len() + if n.bundle.is_none() { n.reads.len() } else { 0 })
            .sum::<usize>();
        let full = input
            .iter()
            .map(|n| 9 + 3 * (n.desc.len() + n.reads.len()))
            .sum();
        let mut b = Builder::new(
            factor,
            input.len(),
            own + sources.iter().map(Vec::len).sum::<usize>(),
            sources.len(),
            full,
            1 << 24,
        )?;
        if b.factored() {
            for s in sources {
                b.bundle(s.iter().copied().map(Ok))?
            }
        }
        for n in input {
            b.node(
                n.pipeline,
                n.shared,
                n.release,
                n.desc.len(),
                n.reads.len(),
                if b.factored() { n.bundle } else { None },
                n.dispatch,
                n.desc.iter().copied().map(Ok),
                n.reads.iter().copied().map(Ok),
            )?
        }
        b.finish()
    }
    fn legacy(input: &[Input]) -> Vec<u64> {
        let mut result = Vec::new();
        for n in input {
            result.extend([
                n.pipeline,
                n.shared as u64,
                n.release as u64,
                n.desc.len() as u64,
            ]);
            for e in &n.desc {
                result.extend([e.view.raw, e.view.first, e.view.bytes])
            }
            result.push(n.reads.len() as u64);
            for e in &n.reads {
                result.extend([e.raw, e.first, e.bytes])
            }
            result.extend(n.dispatch)
        }
        result
    }
    #[test]
    fn actual_factored_producer_expands_to_every_original_field_and_exact_replay_equality() {
        let input = inputs();
        let sources = vec![input[0].reads.clone()];
        let flat = build(false, &input, &sources).unwrap();
        let shared = build(true, &input, &sources).unwrap();
        let old = legacy(&input);
        assert_eq!(shared.words().collect::<Vec<_>>(), old);
        assert_eq!(flat.words().collect::<Vec<_>>(), old);
        assert!(flat == shared && shared == flat);
        let same = build(true, &input, &sources).unwrap();
        assert!(shared == same);
        for i in 0..old.len() {
            let mut bad = old.clone();
            bad[i] = bad[i].wrapping_add(1);
            assert!(shared != Signature::Flat(bad), "original field {i}")
        }
        let mut duplicate = input.clone();
        duplicate[2].bundle = Some(1);
        let other = build(true, &duplicate, &[sources[0].clone(), sources[0].clone()]).unwrap();
        assert!(shared == other && other == shared);
        assert_eq!(other.compact().unwrap().bundles.0.len(), 2);
        assert!(shared.metadata_bytes().unwrap() < 1 << 24);
    }
    #[test]
    fn strict_whole_recording_writes_and_full_physical_extent_control_the_certificate() {
        let mut input = inputs();
        let source = vec![input[0].reads.clone()];
        let sig = build(true, &input, &source).unwrap();
        let c = sig.compact().unwrap();
        let r = Readonly::build(c, |_| Some(1024), 1 << 24).unwrap();
        assert_eq!(r.roots(), &[v(10, 0, 1024)]);
        assert_eq!(r.active(c, 0).unwrap(), &[1]);
        assert_eq!(c.node(0).unwrap().reads(), 3);
        input[2].desc[0].view = v(10, 900, 64);
        let sig = build(true, &input, &source).unwrap();
        let r = Readonly::build(sig.compact().unwrap(), |_| Some(1024), 1 << 24).unwrap();
        assert!(r.roots().is_empty());
        let sig = build(true, &inputs(), &source).unwrap();
        let c = sig.compact().unwrap();
        assert!(
            Readonly::build(c, |_| Some(2048), 1 << 24)
                .unwrap()
                .roots()
                .is_empty()
        );
        assert!(
            Readonly::build(c, |_| None, 1 << 24)
                .unwrap()
                .roots()
                .is_empty()
        );
        assert!(Readonly::build(c, |_| Some(1024), 32).is_err());
        input[2].desc[0].flags = vk::AccessFlags::HOST_WRITE;
        assert!(
            Readonly::build(
                build(true, &input, &source).unwrap().compact().unwrap(),
                |_| Some(1024),
                1 << 24
            )
            .is_err()
        );
    }
    #[test]
    fn canonical_root_evidence_and_source_certificates_bind_actual_owners_and_full_views() {
        let input = inputs();
        let sources = vec![input[0].reads.clone()];
        let a = build(true, &input, &sources).unwrap();
        let b = build(true, &input, &sources).unwrap();
        let c = a.compact().unwrap();
        let evidence = super::super::capture_inventory::Snapshot::build_canonical(
            c.identity(),
            c.nodes(),
            c.views(),
            6,
            |i| {
                Ok(Root {
                    raw: (i as u64 + 1) * 10,
                    bytes: 1024,
                })
            },
            1 << 20,
        )
        .unwrap();
        assert!(evidence.check_canonical(c.identity()).is_ok());
        assert!(
            evidence
                .check_canonical(b.compact().unwrap().identity())
                .is_err()
        );
        let seal = Readonly::build(c, |r| evidence.root_size(r), 1 << 24).unwrap();
        assert!(seal.active(b.compact().unwrap(), 0).is_err());
        assert_eq!(seal.active(c, 0).unwrap(), &[1]);
        assert!(
            super::super::capture_inventory::Snapshot::build_canonical(
                c.identity(),
                c.nodes(),
                c.views(),
                5,
                |i| Ok(Root {
                    raw: (i as u64 + 1) * 10,
                    bytes: 1024
                }),
                1 << 20
            )
            .is_err()
        );
    }
    #[test]
    fn incomplete_unknown_reference_count_and_metadata_cannot_publish_a_signature() {
        let input = inputs();
        let mut b = Builder::new(true, 3, 7, 1, 57, 1 << 24).unwrap();
        b.bundle(input[0].reads.iter().copied().map(Ok)).unwrap();
        assert!(b.finish().is_err());
        let mut b = Builder::new(true, 3, 7, 1, 57, 1 << 24).unwrap();
        b.bundle(input[0].reads.iter().copied().map(Ok)).unwrap();
        let n = &input[0];
        assert!(
            b.node(
                n.pipeline,
                n.shared,
                n.release,
                1,
                3,
                Some(7),
                n.dispatch,
                n.desc.iter().copied().map(Ok),
                n.reads.iter().copied().map(Ok)
            )
            .is_err()
        );
        assert!(Builder::new(true, 3, 7, 1, 57, 32).is_err());
        let mut input = inputs();
        input[0].reads.pop();
        assert!(build(true, &input, &[inputs()[0].reads.clone()]).is_err());
    }
    fn accesses(n: &Input) -> Vec<schedule::Access> {
        let mut v = n
            .desc
            .iter()
            .map(|e| schedule::Access {
                buffer: e.view.raw,
                first: e.view.first,
                bytes: e.view.bytes,
                write: e
                    .flags
                    .intersects(vk::AccessFlags::SHADER_WRITE | vk::AccessFlags::TRANSFER_WRITE),
            })
            .collect::<Vec<_>>();
        v.extend(n.reads.iter().map(|v| schedule::Access {
            buffer: v.raw,
            first: v.first,
            bytes: v.bytes,
            write: false,
        }));
        if n.dispatch[0] == 1 {
            v.push(schedule::Access {
                buffer: n.dispatch[1],
                first: n.dispatch[2],
                bytes: n.dispatch[3],
                write: false,
            })
        };
        v
    }
    fn plan(
        input: &[Input],
        seal: Option<&Readonly>,
        lane: usize,
        serialize: bool,
    ) -> schedule::Plan {
        let mut p = schedule::Planner::for_lane(input.len(), serialize, lane).unwrap();
        if let Some(r) = seal {
            p.retain_readonly(r.roots().iter().map(|v| schedule::Access {
                buffer: v.raw,
                first: v.first,
                bytes: v.bytes,
                write: false,
            }))
            .unwrap()
        }
        for n in input {
            let all = accesses(n);
            let active = all
                .into_iter()
                .filter(|v| !seal.is_some_and(|s| s.contains(v.buffer)))
                .collect::<Vec<_>>();
            p.push_accesses(n.dispatch[0] >= 2, n.shared, n.release, &active)
                .unwrap()
        }
        p.finish().unwrap()
    }
    fn compare(a: &schedule::Plan, b: &schedule::Plan) {
        assert_eq!(a.node_segments, b.node_segments);
        assert_eq!((a.last_copy, a.last_shared), (b.last_copy, b.last_shared));
        assert_eq!(a.segments.len(), b.segments.len());
        for (a, b) in a.segments.iter().zip(&b.segments) {
            assert_eq!(
                (a.copy, a.shared, a.waits, a.wait, a.signal),
                (b.copy, b.shared, b.waits, b.wait, b.signal)
            )
        }
        assert_eq!(
            a.accesses()
                .iter()
                .map(|a| (a.buffer, a.first, a.bytes, a.write))
                .collect::<Vec<_>>(),
            b.accesses()
                .iter()
                .map(|a| (a.buffer, a.first, a.bytes, a.write))
                .collect::<Vec<_>>()
        );
    }
    #[test]
    fn complete_factored_plans_match_original_segments_waits_releases_and_external_footprints() {
        for seed in 0..8u64 {
            let mut input = Vec::new();
            let sources = vec![vec![v(10, 0, 1024), v(20, 0, 1024), v(10, 0, 1024)]];
            for i in 0..512usize {
                let copy = i % 11 == 0;
                let raw = if i == 511 && seed % 2 == 0 {
                    10
                } else if i % 7 == 0 {
                    20
                } else {
                    30 + 10 * ((i + seed as usize) % 4) as u64
                };
                let write = i % 3 != 0;
                let desc = if copy {
                    vec![
                        Entry {
                            view: v(80, 0, 32),
                            flags: vk::AccessFlags::TRANSFER_READ,
                            atomic_or: false,
                        },
                        Entry {
                            view: v(raw, ((i % 8) * 64) as u64, 32),
                            flags: vk::AccessFlags::TRANSFER_WRITE,
                            atomic_or: false,
                        },
                    ]
                } else {
                    vec![Entry {
                        view: v(raw, ((i % 8) * 64) as u64, 32),
                        flags: if write {
                            vk::AccessFlags::SHADER_WRITE
                        } else {
                            vk::AccessFlags::SHADER_READ
                        },
                        atomic_or: false,
                    }]
                };
                input.push(Input {
                    pipeline: if copy { 0 } else { 100 + (i % 3) as u64 },
                    shared: i % 5 == 0,
                    release: i % 13 == 0,
                    desc,
                    reads: if copy { vec![] } else { sources[0].clone() },
                    bundle: (!copy).then_some(0),
                    dispatch: if copy { [2, 0, 0, 0] } else { [0, 1, 1, 1] },
                })
            }
            let sig = build(true, &input, &sources).unwrap();
            assert_eq!(sig.words().collect::<Vec<_>>(), legacy(&input));
            let seal = Readonly::build(sig.compact().unwrap(), |_| Some(1024), 1 << 24).unwrap();
            for lane in 0..2 {
                for serialize in [false, true] {
                    let old = plan(&input, None, lane, serialize);
                    let new = plan(&input, Some(&seal), lane, serialize);
                    compare(&old, &new);
                    for root in [10, 20, 30, 80, 99] {
                        let a = [schedule::Access {
                            buffer: root,
                            first: 900,
                            bytes: 8,
                            write: true,
                        }];
                        assert_eq!(old.conflicts(&a).unwrap(), new.conflicts(&a).unwrap())
                    }
                }
            }
        }
    }
}
#[cfg(test)]
mod compute_equivalence_tests {
    use super::*;
    use crate::strata_compute_barrier_controls as compute;
    use crate::strata_schedule as schedule;
    #[test]
    fn original_compute_dependencies_are_identical_when_only_certified_full_reads_are_factored() {
        let views = [
            View {
                raw: 10,
                first: 0,
                bytes: 1024,
            },
            View {
                raw: 20,
                first: 0,
                bytes: 1024,
            },
        ];
        let mut builder = Builder::new(true, 3, 5, 1, 54, 1 << 20).unwrap();
        builder.bundle(views.iter().copied().map(Ok)).unwrap();
        for (i, raw) in [30, 20, 40].into_iter().enumerate() {
            builder
                .node(
                    100,
                    false,
                    false,
                    1,
                    2,
                    Some(0),
                    [0, 1, 1, 1],
                    std::iter::once(Ok(Entry {
                        view: View {
                            raw,
                            first: 64,
                            bytes: 32,
                        },
                        flags: vk::AccessFlags::SHADER_WRITE,
                        atomic_or: false,
                    })),
                    std::iter::empty(),
                )
                .unwrap();
            assert!(i < 3);
        }
        let signature = builder.finish().unwrap();
        let c = signature.compact().unwrap();
        let seal = Readonly::build(c, |_| Some(1024), 1 << 20).unwrap();
        assert!(seal.contains(10) && !seal.contains(20));
        let mut old = compute::Recorder::broad(3, 1 << 20).unwrap();
        let mut new = compute::Recorder::broad(3, 1 << 20).unwrap();
        let mut old_plan = schedule::Planner::new(3, false).unwrap();
        let mut new_plan = schedule::Planner::new(3, false).unwrap();
        new_plan
            .retain_readonly(seal.roots().iter().map(|v| schedule::Access {
                buffer: v.raw,
                first: v.first,
                bytes: v.bytes,
                write: false,
            }))
            .unwrap();
        for i in 0..3 {
            let n = &c.nodes.0[i];
            let mut all = c.views.0[n.desc.range()]
                .iter()
                .chain(c.views.0[c.reads(n).range()].iter())
                .map(|e| compute::Access {
                    buffer: e.view.raw,
                    first: e.view.first,
                    bytes: e.view.bytes,
                    flags: e.flags,
                    atomic_or: e.atomic_or,
                })
                .collect::<Vec<_>>();
            let mut active = all
                .iter()
                .copied()
                .filter(|a| !seal.contains(a.buffer))
                .collect::<Vec<_>>();
            old_plan
                .push_accesses(
                    false,
                    false,
                    false,
                    &all.iter()
                        .map(|a| schedule::Access {
                            buffer: a.buffer,
                            first: a.first,
                            bytes: a.bytes,
                            write: a.flags.intersects(vk::AccessFlags::SHADER_WRITE),
                        })
                        .collect::<Vec<_>>(),
                )
                .unwrap();
            new_plan
                .push_accesses(
                    false,
                    false,
                    false,
                    &active
                        .iter()
                        .map(|a| schedule::Access {
                            buffer: a.buffer,
                            first: a.first,
                            bytes: a.bytes,
                            write: a.flags.intersects(vk::AccessFlags::SHADER_WRITE),
                        })
                        .collect::<Vec<_>>(),
                )
                .unwrap();
            old.push(0, Some(&mut all)).unwrap();
            new.push(0, Some(&mut active)).unwrap();
        }
        let old = old.finish().unwrap();
        let new = new.finish().unwrap();
        assert_eq!(old.call_counts(), new.call_counts());
        assert_eq!(old.candidates(), new.candidates());
        for i in 0..3 {
            assert_eq!(
                old.node(i)
                    .unwrap()
                    .iter()
                    .map(|d| (d.buffer, d.before, d.after))
                    .collect::<Vec<_>>(),
                new.node(i)
                    .unwrap()
                    .iter()
                    .map(|d| (d.buffer, d.before, d.after))
                    .collect::<Vec<_>>()
            )
        }
        let old = old_plan.finish().unwrap();
        let new = new_plan.finish().unwrap();
        assert_eq!(
            old.accesses()
                .iter()
                .map(|a| (a.buffer, a.first, a.bytes, a.write))
                .collect::<Vec<_>>(),
            new.accesses()
                .iter()
                .map(|a| (a.buffer, a.first, a.bytes, a.write))
                .collect::<Vec<_>>()
        );
    }
}
