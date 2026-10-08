//! Architecture-neutral packed arithmetic shared by the two fixed-public labs.
pub fn shader(dtype: u32, constants: &str, body: &str) -> String {
    format!(
        "const DTYPE:u32={dtype}u;\n{constants}\n{}\n{body}",
        include_str!("common.wgsl")
    )
}
pub fn matrix_shader(dtype: u32, constants: &str, vector: bool) -> String {
    if vector {
        shader(dtype, constants, include_str!("matrix_vector.wgsl"))
            .replace("arena:array<f32>", "arena4:array<vec4<f32>>")
    } else {
        shader(dtype, constants, include_str!("matrix.wgsl"))
    }
}
#[allow(dead_code)] // Shared module is compiled separately by each laboratory.
pub const ORNITH_ROUTER: &str =
    "const EXPERTS:u32=256u;const RANKS:u32=8u;const ROUTE_FIRST:u32=0u;const PAIRED:bool=false;";
#[allow(dead_code)]
pub const NEXT_ROUTER: &str =
    "const EXPERTS:u32=512u;const RANKS:u32=10u;const ROUTE_FIRST:u32=0u;const PAIRED:bool=true;";
