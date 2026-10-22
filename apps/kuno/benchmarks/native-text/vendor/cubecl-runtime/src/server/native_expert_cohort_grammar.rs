//! Distinct small-expert cohort grammar. Every INFO record, actual owning view,
//! per-role phase and physical grid is authenticated before any enqueue.
use super::*;
const WORKING: u64 = 80 * 1024 * 1024;
const X: u64 = 20 * 1024 * 1024;
const H: u64 = 5 * 1024 * 1024;
const RAW: u64 = 32 * 1024 * 1024;
pub(super) struct Cohort {
    words: [u32; 48],
    phase: usize,
    families: [bool; 2],
    map: View,
    fault: View,
    input: View,
    x: View,
    images: Option<View>,
    raw: Option<View>,
    grid: Option<View>,
    addresses: Option<View>,
    gu: Option<View>,
    h: Option<View>,
    status: Option<View>,
}
struct Words([u32; 48]);
impl core::ops::Deref for Words {
    type Target = [u32; 48];
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
impl Drop for Words {
    fn drop(&mut self) {
        for word in &mut self.0 {
            unsafe {
                core::ptr::write_volatile(word, 0);
            }
        }
        core::sync::atomic::compiler_fence(core::sync::atomic::Ordering::SeqCst);
    }
}
fn values(args: &KernelArguments) -> Result<Words, ServerError> {
    let mut result = [0u32; 48];
    for (i, x) in result.iter_mut().enumerate() {
        *x = word(&args.info, i)?;
    }
    Ok(Words(result))
}
fn same(a: &[u32; 48], b: &[u32; 48]) -> bool {
    [0, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15]
        .into_iter()
        .chain(16..48)
        .all(|i| a[i] == b[i])
}
impl Cohort {
    pub(super) fn start(first: usize, args: &KernelArguments) -> Result<Self, ServerError> {
        let w = values(args)?;
        if args.buffers.len() != 4
            || args.info.data.len() != 24
            || args.info.dynamic_metadata_offset != 0
            || !args.info.native_parameters.is_empty()
            || !args.read_dependencies.is_empty()
            || !args.tensor_maps.is_empty()
            || w[0] == 0
            || w[0] > 4096
            || w[4] as usize != first
            || w[9] == 0
            || w[9] > 8
            || first + w[9] as usize > 512
            || w[6] == 0
            || w[7] != w[6]
            || w[6] > 1048576
            || w[8] <= 2 * w[6]
            || u64::from(w[8]) * 4 * 8 > RAW
            || w[12] > 1
            || w[13..16] != [0; 3]
        {
            return Err(batch_fault("native expert cohort header/INFO capacity"));
        }
        let mut prefix = 0;
        let mut max = 0;
        let mut slots = [false; 8];
        let mut families = [false; 2];
        for i in 0..8 {
            let record = &w[16 + i * 4..20 + i * 4];
            if i >= w[9] as usize {
                if record != [0; 4] {
                    return Err(batch_fault("native expert cohort spare INFO"));
                }
                continue;
            }
            let [group, members, row, slot] = <[u32; 4]>::try_from(record).unwrap();
            if group != w[4] + i as u32
                || members == 0
                || members >= 32
                || members > w[0]
                || row != prefix
                || slot > 8
                || (slot < 8 && slots[slot as usize])
            {
                return Err(batch_fault(
                    "native expert cohort original group/slot/row partition",
                ));
            }
            if slot < 8 {
                slots[slot as usize] = true;
            }
            let tile = if members <= 8 { 8 } else { 16 };
            prefix += members.div_ceil(tile) * tile;
            max = max.max(members);
            families[usize::from(tile == 16)] = true;
        }
        if prefix > 4096 || w[10] != max {
            return Err(batch_fault("native expert cohort complete row capacity"));
        }
        let map = View::new(&args.buffers[0])?;
        let input = View::new(&args.buffers[1])?;
        let x = View::new(&args.buffers[2])?;
        let fault = View::new(&args.buffers[3])?;
        if map.used() != MAP_BYTES
            || fault.used() != 4
            || x.used() != X
            || x.first % 256 != 0
            || input.used() < u64::from(w[0]) * 2560 * if w[12] == 1 { 2 } else { 4 }
            || [input, x, fault].iter().any(|v| v.stream != map.stream)
        {
            return Err(batch_fault("native expert cohort original Gather owners"));
        }
        Ok(Self {
            words: *w,
            phase: 0,
            families,
            map,
            fault,
            input,
            x,
            images: None,
            raw: None,
            grid: None,
            addresses: None,
            gu: None,
            h: None,
            status: None,
        })
    }
    fn expected(&self) -> Option<(&'static str, bool, u32)> {
        let mut stage = 0;
        for entry in [
            ("StrataExpertCohortGather", false, 0),
            ("StrataExpertCohortGuImage", false, 0),
            ("StrataExpertCohortDownImage", true, 0),
        ] {
            if stage == self.phase {
                return Some(entry);
            }
            stage += 1;
        }
        for down in [false, true] {
            for (i, tile) in [8, 16].into_iter().enumerate() {
                if self.families[i] {
                    if stage == self.phase {
                        return Some((if down { "matrix-down" } else { "matrix-gu" }, false, tile));
                    }
                    stage += 1;
                }
            }
            if !down {
                if stage == self.phase {
                    return Some(("StrataExpertCohortActivate", false, 0));
                }
                stage += 1;
            }
        }
        None
    }
    pub(super) fn launch(
        &mut self,
        name: &str,
        release: bool,
        count: &CubeCount,
        args: &KernelArguments,
    ) -> Result<bool, ServerError> {
        let (expected, expected_release, tile) = self
            .expected()
            .ok_or_else(|| batch_fault("native expert cohort surplus phase"))?;
        let matrix = expected.starts_with("matrix");
        let down = expected == "matrix-down" || expected == "StrataExpertCohortDownImage";
        let is_matrix = name
            .split_once("::cohorts::expert_cohort::")
            .is_some_and(|(_, v)| {
                v.strip_prefix("ExpertCohort")
                    .is_some_and(|s| s.is_empty() || s.starts_with('<'))
            });
        if release != expected_release || if matrix { !is_matrix } else { name != expected } {
            return Err(batch_fault("native expert cohort authentic phase/order"));
        }
        let w = values(args)?;
        if !same(&self.words, &w)
            || !args.info.native_parameters.is_empty()
            || !args.tensor_maps.is_empty()
            || (!matrix && (args.info.data.len() != 24 || args.info.dynamic_metadata_offset != 0))
        {
            return Err(batch_fault("native expert cohort immutable original INFO"));
        }
        let z = w[9];
        if matrix {
            if args.buffers.len() != 6
                || !args.read_dependencies.is_empty()
                || args.info.data.len() != 30
                || args.info.dynamic_metadata_offset != 30
                || w[1] != if down { 640 } else { 2560 }
                || w[2] != if down { 2560 } else { 1280 }
                || w[3] != 0
                || w[5] != if down { 2 } else { 0 }
                || w[11] != tile
                || !matches!(count,CubeCount::Static(x,y,c)if *x==w[2]/64 && *y==if tile==8{1}else{2} && *c==z)
            {
                return Err(batch_fault("native expert cohort matrix schema/grid"));
            }
            let views = [
                View::new(&args.buffers[0])?,
                View::new(&args.buffers[1])?,
                View::new(&args.buffers[2])?,
                View::new(&args.buffers[3])?,
                View::new(&args.buffers[4])?,
                View::new(&args.buffers[5])?,
            ];
            if views[0]
                != if down {
                    self.h.ok_or_else(|| {
                        batch_fault("native expert cohort hidden producer missing")
                    })?
                } else {
                    self.x
                }
                || Some(views[1]) != self.images
                || views[3] != self.map
                || views[5] != self.fault
                || views[2].used() != if down { PART_BYTES } else { GU_BYTES }
                || views[4].used() != STATUS_BYTES
                || views.iter().any(|v| v.stream != self.map.stream)
            {
                return Err(batch_fault("native expert cohort matrix actual owners"));
            }
            for (i, v) in views.iter().enumerate() {
                let unit = if i < 2 { 8 } else { 4 };
                let len = v.used() / unit;
                if v.first % unit != 0
                    || v.used() % unit != 0
                    || u64::from(word(&args.info, 48 + i)?) != len
                    || u64::from(word(&args.info, 54 + i)?) != len
                {
                    return Err(batch_fault("native expert cohort matrix array INFO"));
                }
            }
            if down {
                if Some(views[4]) != self.status {
                    return Err(batch_fault("native expert cohort status producer changed"));
                }
            } else {
                if self.gu.is_some_and(|v| v != views[2])
                    || self.status.is_some_and(|v| v != views[4])
                {
                    return Err(batch_fault("native expert cohort GU family owners changed"));
                }
                self.gu = Some(views[2]);
                self.status = Some(views[4]);
            }
        } else if self.phase == 0 {
            if args.buffers.len() != 4
                || w[1] != 2560
                || w[2] != 1280
                || w[3] != 0
                || w[5] != 0
                || w[11] != 0
                || !matches!(count,CubeCount::Static(x,1,c)if *x==w[10]*5 && *c==z)
                || [self.map, self.input, self.x, self.fault]
                    .into_iter()
                    .zip(&args.buffers)
                    .any(|(a, b)| View::new(b).map_or(true, |v| v != a))
            {
                return Err(batch_fault("native expert cohort Gather grid/bindings"));
            }
        } else if expected == "StrataExpertCohortActivate" {
            if args.buffers.len() != 5
                || !args.read_dependencies.is_empty()
                || w[1] != 2560
                || w[2] != 1280
                || w[3] != 0
                || w[5] != 0
                || w[11] != 0
                || !matches!(count,CubeCount::Static(x,1,c)if *x==(w[10]*320).div_ceil(256) && *c==z)
            {
                return Err(batch_fault("native expert cohort activation shape"));
            }
            let views = [
                View::new(&args.buffers[0])?,
                View::new(&args.buffers[1])?,
                View::new(&args.buffers[2])?,
                View::new(&args.buffers[3])?,
                View::new(&args.buffers[4])?,
            ];
            if views[0] != self.map
                || Some(views[1]) != self.gu
                || views[2].used() != H
                || Some(views[3]) != self.status
                || views[4] != self.fault
                || views[2].id != self.x.id
                || views[2].first != self.x.first + X
                || views[2].stream != self.x.stream
            {
                return Err(batch_fault(
                    "native expert cohort disjoint authentic hidden owner",
                ));
            }
            self.h = Some(views[2]);
        } else {
            let offset = usize::from(!down);
            let expected_len = 7 + offset;
            if args.buffers.len() != expected_len
                || args.read_dependencies.is_empty()
                || w[11] != 0
                || w[3] == 0
                || w[1] != if down { 640 } else { 2560 }
                || w[2] != if down { 2560 } else { 1280 }
                || w[5] != if down { 2 } else { 0 }
                || !matches!(count,CubeCount::Static(x,1,c)if *x==(w[1]*w[2]/32).div_ceil(256) && *c==z)
            {
                return Err(batch_fault(
                    "native expert cohort exact dequant role/schema",
                ));
            }
            let map = View::new(&args.buffers[1 + offset])?;
            let raw = View::new(&args.buffers[2 + offset])?;
            let image = View::new(&args.buffers[3 + offset])?;
            let fault = View::new(&args.buffers[4 + offset])?;
            let grid = View::new(&args.buffers[5 + offset])?;
            let addresses = View::new(&args.buffers[6 + offset])?;
            if map != self.map
                || fault != self.fault
                || raw.used() != RAW
                || raw.first != 0
                || raw.end != 0
                || image.used() != WORKING
                || grid.used() != 32768
                || addresses.used() != 32768
                || [raw, image, grid, addresses]
                    .iter()
                    .any(|v| v.stream != self.map.stream)
                || self.raw.is_some_and(|v| v != raw)
                || self.images.is_some_and(|v| v != image)
                || self.grid.is_some_and(|v| v != grid)
                || self.addresses.is_some_and(|v| v != addresses)
            {
                return Err(batch_fault(
                    "native expert cohort actual raw/image/public owners",
                ));
            }
            self.raw = Some(raw);
            self.images = Some(image);
            self.grid = Some(grid);
            self.addresses = Some(addresses);
        }
        self.phase += 1;
        Ok(self.expected().is_none())
    }
    pub(super) fn groups(&self) -> usize {
        self.words[9] as usize
    }
}
impl Drop for Cohort {
    fn drop(&mut self) {
        for word in &mut self.words {
            unsafe {
                core::ptr::write_volatile(word, 0);
            }
        }
        core::sync::atomic::compiler_fence(core::sync::atomic::Ordering::SeqCst);
    }
}
