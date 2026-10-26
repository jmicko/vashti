//! Genuine immutable whole-layer input authority, derived only from its producer.
use super::*;
use alloc::vec::Vec;
pub(super) const X_BYTES: u64 = 56832 * 2560 * 2;
const X_FIRST: u64 =
    80 * 1024 * 1024 + GU_BYTES + PART_BYTES + 4 * (131072 + 65536 + 65536 + 131072) + 6 * 256;
pub(super) struct Layer {
    pub(super) owners: [Binding; 4],
    pub(super) h: View,
    pub(super) prompt: u32,
    pub(super) half: u32,
    pub(super) groups: usize,
}
impl Layer {
    pub(super) fn start(
        release: bool,
        count: &CubeCount,
        args: &KernelArguments,
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
        if !(1..=4096).contains(&n)
            || !(1..=512).contains(&groups)
            || groups > n * 10
            || half > 1
            || padded < n * 10
            || padded > 56832
            || padded % 32 != 0
            || (4..8).any(|i| word(&args.info, i).map_or(true, |v| v != 0))
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
            || x.size != 780 * 1024 * 1024
            || map.id != x.id
            || map.size != x.size
            || map.first + map.used() > x.first
            || h.used() != 5 * 1024 * 1024
            || h.first != 20 * 1024 * 1024
            || h.size != 756 * 1024 * 1024
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
            prompt: n,
            half,
            groups: groups as usize,
        })
    }
    pub(super) fn unit_args(&self, args: &KernelArguments) -> Result<KernelArguments, ServerError> {
        if args.info.dynamic_metadata_offset != 0 || !args.info.native_parameters.is_empty() || word(&args.info, 0)? != self.prompt
            || word(&args.info, 13)? != 1
            || word(&args.info, 12)? != self.half
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
