//! Genuine ordered DOWN identity for the shared K32 lifetime-reuse arithmetic.
use super::*;
#[cube(launch_unchecked)]
pub fn expert_wide_down(
    input: &Array<Vector<f16, Const<4>>>,
    weight: &Array<Vector<f16, Const<4>>>,
    output: &mut Array<f32>,
    map: &Array<u32>,
    status: &mut Array<u32>,
    fault: &mut Array<Atomic<u32>>,
    columns: u32,
    rows: u32,
    count: u32,
    weight_first: u32,
    group: u32,
    member_base: u32,
    piece_members: u32,
) {
    crate::strata_matrix::native_expert_wide::shared_reuse::wide(
        input,
        weight,
        output,
        map,
        status,
        fault,
        columns,
        rows,
        count,
        weight_first,
        group,
        member_base,
        piece_members,
        true,
        false,
        true,
    );
}

pub(crate) fn definition() -> KernelDefinition {
    let mut b = KernelBuilder::default();
    cubecl::ir::AddressType::U32.register(&mut b.scope);
    let input = b.input_array(Vector::<f16, Const<4>>::as_type(&b.scope));
    let weight = b.input_array(Vector::<f16, Const<4>>::as_type(&b.scope));
    let output = b.output_array(f32::as_type(&b.scope));
    let map = b.input_array(u32::as_type(&b.scope));
    let status = b.output_array(u32::as_type(&b.scope));
    let fault = b.output_array(
        cubecl::ir::StorageType::Atomic(ElemType::UInt(cubecl::ir::UIntKind::U32)).into(),
    );
    let a: [cubecl::ir::ManagedVariable; 7] =
        std::array::from_fn(|_| b.scalar(u32::as_type(&b.scope).storage_type()));
    expert_wide_down::expand(
        &mut b.scope,
        input.into(),
        weight.into(),
        output.into(),
        map.into(),
        status.into(),
        fault.into(),
        a[0].clone().into(),
        a[1].clone().into(),
        a[2].clone().into(),
        a[3].clone().into(),
        a[4].clone().into(),
        a[5].clone().into(),
        a[6].clone().into(),
    );
    let errors = b.scope.pop_errors();
    assert!(errors.is_empty(), "{errors:?}");
    b.build(KernelSettings::default().cube_dim(CubeDim::new_1d(THREADS)))
}
