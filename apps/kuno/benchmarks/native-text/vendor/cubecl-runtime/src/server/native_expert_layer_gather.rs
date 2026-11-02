//! Genuine immutable whole-layer input authority, derived only from its producer.
use super::*;
use alloc::vec::Vec;
pub(super) const X_BYTES: u64 = PADDED_ROWS * 2560 * 2;
const X_FIRST: u64 =
    WORKING_BYTES + GU_BYTES + PART_BYTES + 4 * (131072 + 65536 + 65536 + 131072) + 6 * 256;
/// Int8 activation region after the layer X ([Xq | Hq], see strata_prompt_groups).
pub(super) const XQ_BYTES: u64 = ROWS * 720 * 4;
pub(super) const HQ_BYTES: u64 = ROWS * 180 * 4;
pub(super) struct Layer {
    pub(super) owners: [Binding; 4],
    pub(super) h: View,
    /// Set by the layer's single ClaudeQuantX: (Xq, Hq) views.
    pub(super) q: Option<(View, View)>,
    pub(super) prompt: u32,
    pub(super) half: u32,
    pub(super) groups: usize,
    packed_large: bool,
}
impl Layer {
    pub(super) fn start(
        release: bool,
        count: &CubeCount,
        args: &KernelArguments,
    ) -> Result<Self, ServerError> {
        Self::start_inner(release, count, args, false)
    }
    #[cfg(feature = "kuno-large-packed-tiles")]
    pub(super) fn start_packed(
        release: bool,
        count: &CubeCount,
        args: &KernelArguments,
    ) -> Result<Self, ServerError> {
        Self::start_inner(release, count, args, true)
    }
    fn start_inner(
        release: bool,
        count: &CubeCount,
        args: &KernelArguments,
        packed_large: bool,
    ) -> Result<Self, ServerError> {
        if release
            || args.buffers.len() != 5
            || args.info.data.len() != 4
            || args.info.dynamic_metadata_offset != 0
            || !args.info.native_parameters.is_empty()
            || !args.read_dependencies.is_empty()
            || !args.tensor_maps.is_empty()
        {
            return Err(batch_fault("layer gather exact producer schema"));
        }
        let n = word(&args.info, 0)?;
        let groups = word(&args.info, 1)?;
        let half = word(&args.info, 2)?;
        let padded = word(&args.info, 3)?;
        if !(1..=MAX_MEMBERS).contains(&n)
            || !(1..=512).contains(&groups)
            || groups > n * 10
            || half > 1
            || padded < n * 10
            || u64::from(padded) > PADDED_ROWS
            || padded % 32 != 0
            || word(&args.info, 4)? != u32::from(packed_large)
            || (5..8).any(|i| word(&args.info, i).map_or(true, |v| v != 0))
            || !grid(count, 5, padded)
        {
            return Err(batch_fault("layer gather exact producer geometry"));
        }
        let views = args
            .buffers
            .iter()
            .map(View::new)
            .collect::<Result<Vec<_>, _>>()?;
        let [map, input, x, h, fault] =
            <[View; 5]>::try_from(views).map_err(|_| batch_fault("layer gather owner count"))?;
        if map.used() != MAP_BYTES
            || fault.used() != 4
            || x.used() != X_BYTES
            || x.first != X_FIRST
            // The phase root: layer X plus the int8 activation region.
            || x.size != KUNO_PHASE_ROOT_BYTES
            || map.id != x.id
            || map.size != x.size
            || map.first + map.used() > x.first
            || h.used() != H_ROWS_BYTES
            || h.first != X_ROWS_BYTES
            || h.size != KUNO_PERSISTENT_ROOT_BYTES
            || h.id == x.id
            || input.used() < u64::from(n) * 2560 * if half == 1 { 2 } else { 4 }
            || [input, x, h, fault].iter().any(|v| v.stream != map.stream)
        {
            return Err(batch_fault(
                "layer gather authentic complete physical views",
            ));
        }
        if input.id == x.id
            || fault.id == x.id
            || fault.id == h.id
            || input.id == h.id
                && (input.first < h.first + h.used() && h.first < input.first + input.used())
        {
            return Err(batch_fault("layer gather live view alias"));
        }
        Ok(Self {
            owners: [
                args.buffers[0].clone(),
                args.buffers[1].clone(),
                args.buffers[2].clone(),
                args.buffers[4].clone(),
            ],
            h,
            q: None,
            prompt: n,
            half,
            groups: groups as usize,
            packed_large,
        })
    }
    /// The layer's one int8 activation quantization: whole-prompt input to Xq.
    pub(super) fn accept_quant(&mut self, release: bool, count: &CubeCount, args: &KernelArguments) -> Result<(), ServerError> {
        if self.q.is_some()
            || release
            || args.buffers.len() != 3
            || args.info.data.len() != 4
            || args.info.dynamic_metadata_offset != 0
            || !args.info.native_parameters.is_empty()
            || !args.read_dependencies.is_empty()
            || !args.tensor_maps.is_empty()
            || word(&args.info, 0)? != self.prompt
            || word(&args.info, 1)? != self.half
            || (2..8).any(|i| word(&args.info, i).map_or(true, |v| v != 0))
            || !grid(count, (self.prompt * 80).div_ceil(256), 1)
        {
            return Err(batch_fault("layer int8 quantization schema"));
        }
        let input = View::new(&args.buffers[0])?;
        let q = View::new(&args.buffers[1])?;
        let fault = View::new(&args.buffers[2])?;
        let x = View::new(&self.owners[2])?;
        if input != View::new(&self.owners[1])?
            || fault != View::new(&self.owners[3])?
            || q.id != x.id
            || q.size != x.size
            || q.first != x.first + x.used() + 256
            || q.used() != XQ_BYTES + HQ_BYTES
            || q.stream != x.stream
        {
            return Err(batch_fault("layer int8 quantization owners"));
        }
        let mut xq = q;
        xq.end = q.end + HQ_BYTES;
        let mut hq = q;
        hq.first = q.first + XQ_BYTES;
        self.q = Some((xq, hq));
        Ok(())
    }
    pub(super) fn unit_args(&self, args: &KernelArguments) -> Result<KernelArguments, ServerError> {
        if args.info.dynamic_metadata_offset != 0 || !args.info.native_parameters.is_empty() || word(&args.info, 0)? != self.prompt
            || word(&args.info, 13)? != 1
            || word(&args.info, 12)? != self.half
            // Large-only packing is sealed by the genuine layer producer.
            // Small/regular cohorts retain their original row-major segments.
            || word(&args.info, 14)? != u32::from(self.packed_large && word(&args.info, 17)? >= 256)
            || word(&args.info, 15)? != 0
        {
            return Err(batch_fault("layer cohort original publication mismatch"));
        }
        Ok(KernelArguments {
            buffers: self.owners.to_vec(),
            info: MetadataBindingInfo::custom(args.info.data.clone()),
            ..Default::default()
        })
    }
}
