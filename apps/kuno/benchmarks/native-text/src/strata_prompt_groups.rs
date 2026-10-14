//! MIT Strata Intel prompt grouping: logical expert order, stable token/rank
//! rows, one bounded host grouping per layer. All maps remain request-private.
use super::*;
use zeroize::Zeroizing;
pub(crate) const HOST_BYTES: usize = 256 * 1024;
pub(crate) const RUNTIME_BYTES: u64 = 8 * 1024 * 1024;
pub(crate) const CAPACITY: usize = 256;
pub(crate) const LOGICAL: usize = 4100;
const MEMBERS: usize = 1540;
const LENGTHS: usize = 516;
const OFFSETS: usize = 1028;
pub(crate) fn group(raw: &[u32], count: usize) -> Result<Zeroizing<Vec<u32>>> {
    if !(1..=CAPACITY).contains(&count)
        || raw.len() != LOGICAL + count * 10
        || raw[..4] != [count as u32, 0, count as u32, 0]
    {
        return Err(Error::InvalidBufferLength);
    }
    let mut lengths = Zeroizing::new([0usize; 512]);
    let mut cells = Zeroizing::new([512u32; 512]);
    let mut inverse = Zeroizing::new([512u32; 512]);
    for route in 0..count * 10 {
        let logical = raw[LOGICAL + route];
        let cell = raw[MEMBERS + route];
        if logical >= 512 || cell >= 512 {
            return Err(Error::InvalidBufferLength);
        }
        for earlier in (route / 10) * 10..route {
            if raw[LOGICAL + earlier] == logical {
                return Err(Error::InvalidBufferLength);
            }
        }
        let l = logical as usize;
        let c = cell as usize;
        if (cells[l] != 512 && cells[l] != cell) || (inverse[c] != 512 && inverse[c] != logical) {
            return Err(Error::InvalidBufferLength);
        }
        cells[l] = cell;
        inverse[c] = logical;
        lengths[l] += 1;
    }
    let mut map = Zeroizing::new(vec![0u32; 8192]);
    map[..3].copy_from_slice(&[count as u32, 0, count as u32]);
    let mut offsets = Zeroizing::new([0usize; 512]);
    let mut n = 0;
    let mut groups = 0;
    for logical in 0..512 {
        if lengths[logical] == 0 {
            continue;
        }
        map[4 + groups] = cells[logical];
        map[LENGTHS + groups] = lengths[logical] as u32;
        map[OFFSETS + groups] = n as u32;
        offsets[logical] = n;
        n += lengths[logical];
        groups += 1;
    }
    map[3] = groups as u32;
    for route in 0..count * 10 {
        let logical = raw[LOGICAL + route] as usize;
        map[MEMBERS + offsets[logical]] = route as u32;
        offsets[logical] += 1;
    }
    validate(&map, count)?;
    Ok(map)
}
pub(crate) fn validate(map: &[u32], count: usize) -> Result<usize> {
    if !(1..=CAPACITY).contains(&count)
        || map.len() != 8192
        || map[..3] != [count as u32, 0, count as u32]
    {
        return Err(Error::InvalidBufferLength);
    }
    let groups = map[3] as usize;
    if groups == 0 || groups > 512 || groups > count * 10 {
        return Err(Error::InvalidBufferLength);
    }
    let mut seen = Zeroizing::new([false; CAPACITY * 10]);
    let mut seen_cells = Zeroizing::new([false; 512]);
    let mut cursor = 0;
    for g in 0..groups {
        let cell = map[4 + g] as usize;
        let length = map[LENGTHS + g] as usize;
        if cell >= 512
            || seen_cells[cell]
            || length == 0
            || length > count
            || map[OFFSETS + g] as usize != cursor
            || cursor + length > count * 10
        {
            return Err(Error::InvalidBufferLength);
        }
        seen_cells[cell] = true;
        let mut previous = None;
        for i in 0..length {
            let route = map[MEMBERS + cursor + i] as usize;
            if route >= count * 10 || seen[route] || previous.is_some_and(|p| p >= route / 10) {
                return Err(Error::InvalidBufferLength);
            }
            seen[route] = true;
            previous = Some(route / 10);
        }
        cursor += length;
    }
    if cursor != count * 10 || seen[..count * 10].iter().any(|&v| !v) {
        return Err(Error::InvalidBufferLength);
    }
    Ok(groups)
}
/// Borrow the unused upper half of the already-admitted raw-stage root. Raw
/// transfers remain below64 MiB; these distinct views hold one group's GU and
/// all actual rank outputs. No larger permanent MAX8 allocation is introduced.
pub(crate) fn scratch() -> Result<[Handle; 2]> {
    let root = stages()?[2].clone();
    if root.size_in_used() != 128 * 1024 * 1024
        || root.offset_start.unwrap_or(0) != 0
        || root.offset_end.unwrap_or(0) != 0
    {
        return Err(Error::InvalidBufferLength);
    }
    let view = |start: u64, bytes: u64| {
        root.clone()
            .offset_start(start)
            .offset_end(128 * 1024 * 1024 - start - bytes)
    };
    Ok([
        view(64 * 1024 * 1024, 2 * 1024 * 1024),
        view(66 * 1024 * 1024, 32 * 1024 * 1024),
    ])
}
