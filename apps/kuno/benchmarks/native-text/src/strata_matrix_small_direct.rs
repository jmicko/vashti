// Additive small-expert direct publication. Products retain their original
// chronological K16 arithmetic; GU finite closure and ordered DOWN inverse
// consumption are mandatory parts of this producer contract.
use super::*;

#[cube]
#[allow(clippy::too_many_arguments)]
pub(crate) fn publish(
    acc: &cmma::Matrix<f32>,
    scratch: &mut SharedMemory<f32>,
    output: &mut Array<f32>,
    map: &Array<u32>,
    status: &mut Array<u32>,
    fault: &mut Array<Atomic<u32>>,
    member0: usize,
    row0: usize,
    count: usize,
    gu_prefix: usize,
    group: usize,
    #[comptime] down: bool,
) {
    let sg = usize::cast_from(PLANE_POS);
    let lane = usize::cast_from(UNIT_POS_PLANE);
    let prefix = usize::cast_from(map[1028 + group]);
    let rows = comptime!(if down { 2560usize } else { 1280usize });
    let complete = member0 + 8 <= count && row0 + 16 <= rows;
    let mut route_good = complete;
    if complete && lane < 8 {
        let route = usize::cast_from(map[1540 + prefix + member0 + lane]);
        route_good = route < usize::cast_from(map[0]) * 10;
    }
    if plane_all(route_good) {
        let mut first = (gu_prefix + member0) * 1280 + row0;
        let mut stride = 1280u32;
        if comptime!(down) {
            first = (prefix + member0) * 2560 + row0;
            stride = 2560u32;
        }
        cmma::store(
            &mut output.slice_mut(first, output.len()),
            acc,
            stride,
            cmma::MatrixLayout::RowMajor,
        );
        if comptime!(down) && row0 == 0 && lane < 8 {
            let route = usize::cast_from(map[1540 + prefix + member0 + lane]);
            status[comptime!(crate::strata_prompt::groups::ROUTES) + route] = 2;
        }
    } else {
        cmma::store(
            &mut scratch.slice_mut(sg * 128, sg * 128 + 128),
            acc,
            16,
            cmma::MatrixLayout::RowMajor,
        );
        // This subgroup alone owns its128-F32 slot. The feature requires the
        // authentic subgroup AcquireRelease|WorkgroupMemory backend lowering.
        sync_plane();
        #[unroll]
        for round in 0..8usize {
            let i = lane + round * 16;
            let member = member0 + i / 16;
            let row = row0 + i % 16;
            if member < count && row < rows {
                let route = usize::cast_from(map[1540 + prefix + member]);
                if route < usize::cast_from(map[0]) * 10 {
                    let value = scratch[sg * 128 + i];
                    let finite = value.abs() <= f32::MAX;
                    let mut first = (gu_prefix + member) * 1280;
                    if comptime!(down) {
                        first = (prefix + member) * 2560;
                    }
                    if finite {
                        output[first + row] = value;
                    } else {
                        output[first + row] = 0.0;
                        fault[0].fetch_or(128);
                    }
                    if comptime!(down) && row == 0 {
                        let mut good = 0u32;
                        if finite && status[route] == 1 {
                            good = 1;
                        }
                        status[comptime!(crate::strata_prompt::groups::ROUTES) + route] = good;
                    }
                } else {
                    fault[0].fetch_or(128);
                }
            }
        }
        sync_plane();
    }
}

#[path = "strata_matrix_small_direct_products.rs"]
mod products;
include!("strata_matrix_small_direct_entries.rs");

#[path = "strata_matrix_small_direct_validation.rs"]
mod validation;
pub(crate) use validation::validate as validate_sources;
#[path = "strata_matrix_small_direct_probe.rs"]
mod probe;
pub(crate) use probe::cases as public_numeric;
#[cfg(test)]
#[path = "strata_matrix_small_direct_tests.rs"]
mod tests;
