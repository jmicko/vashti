//! Bounded packed W[R,K] times token-major X[B,K], producing Y[B,R].
//!
//! This is a single multi-RHS operation: each lane decodes a weight ONCE per
//! column, then uses it across one/four/eight independent token accumulators.
//! Every token retains the existing 64-lane column order and reduction tree.
//! Original seven-format decoding is inherited from the parent; no full-weight
//! conversion, activation quantization, foreign engine or extra dependency.
//!
//! Launch with scalar arrays, CubeDim=(64,1,1), and geometry.grid(). Validate
//! all full-buffer spans first. Geometry and tile policy alone may key cached
//! pipelines; count, selected weight span and activation positions stay runtime.
//! The owner must prove distinct packed/input/output bindings, same model/device
//! and arena epochs, accept finite output/device completion, and erase private
//! activation/scalar/staging spans on all exits. This primitive is not an arena,
//! transaction, privacy certificate or device-performance acceptance.

use super::{BF16, F16, F32, Q4_K, Q5_0, Q6_K, Q8_0, packed_value};
use cubecl::prelude::*;
use zeroize::Zeroize;

pub const LANES: usize = 64;
pub const MAX_BATCH: usize = 8;
pub const MAX_GRID_X: usize = 65535;

/// Immutable public matrix/tile policy. It contains no private dispatch values.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct MatrixGeometry {
    dtype: u32,
    rows: usize,
    columns: usize,
    tile_tokens: usize,
    member_capacity: usize,
    row_bytes: usize,
    total_bytes: usize,
}

impl MatrixGeometry {
    pub fn new(
        dtype: u32,
        rows: usize,
        columns: usize,
        tile_tokens: usize,
    ) -> Result<Self, &'static str> {
        if !matches!(tile_tokens, 1 | 4 | 8) || rows > MAX_GRID_X {
            return Err("unsupported public batch tile or row grid");
        }
        let (elements, bytes) = super::block_layout(dtype).ok_or("unsupported GGUF dtype")?;
        if rows == 0 || columns == 0 || !columns.is_multiple_of(elements) {
            return Err("rows/columns must contain nonzero whole blocks");
        }
        let row_bytes = (columns / elements)
            .checked_mul(bytes)
            .ok_or("row byte size overflow")?;
        let total_bytes = rows.checked_mul(row_bytes).ok_or("matrix size overflow")?;
        super::validate_layout(dtype, rows, columns, total_bytes, columns, rows)?;
        Ok(Self {
            dtype,
            rows,
            columns,
            tile_tokens,
            member_capacity: MAX_BATCH,
            row_bytes,
            total_bytes,
        })
    }
    pub fn dtype(self) -> u32 {
        self.dtype
    }
    pub fn rows(self) -> usize {
        self.rows
    }
    pub fn columns(self) -> usize {
        self.columns
    }
    pub fn tile_tokens(self) -> usize {
        self.tile_tokens
    }
    /// Explicit public count1 dispatch; storage owners remain complete MAX8
    /// bindings. This is never selected from a runtime/private token count.
    pub fn with_single_member(mut self) -> Result<Self, &'static str> {
        if self.tile_tokens != 1 {
            return Err("single-member projection requires public tile1");
        }
        self.member_capacity = 1;
        Ok(self)
    }
    pub fn member_capacity(self) -> usize {
        self.member_capacity
    }
    pub fn row_bytes(self) -> usize {
        self.row_bytes
    }
    pub fn total_bytes(self) -> usize {
        self.total_bytes
    }
    /// Fixed public capacity grid, including inactive tail tiles. Actual count
    /// cannot change pipeline geometry; inactive tokens perform no I/O.
    pub fn grid(self) -> CubeCount {
        CubeCount::Static(
            self.rows as u32,
            (self.member_capacity() / self.tile_tokens) as u32,
            1,
        )
    }
    /// Check byte/element addressing BEFORE constructing unchecked ArrayArgs.
    /// Buffer lengths describe complete bindings, not temporary sliced handles.
    pub fn validate_spans(
        self,
        spans: &BatchSpans,
        packed_words: usize,
        input_words: usize,
        output_words: usize,
    ) -> Result<(), &'static str> {
        if !(1..=self.member_capacity()).contains(&spans.count)
            || spans.input_stride < self.columns
            || spans.output_stride < self.rows
        {
            return Err("invalid runtime batch count or overlapping token rows");
        }
        if self.dtype == F32 && !spans.packed_first.is_multiple_of(4) {
            return Err("F32 packed base must be word aligned");
        }
        let packed_end = spans
            .packed_first
            .checked_add(self.total_bytes)
            .ok_or("packed span overflow")?;
        let input_end = span_end(
            spans.input_first,
            spans.input_stride,
            spans.count,
            self.columns,
        )?;
        let output_end = span_end(
            spans.output_first,
            spans.output_stride,
            spans.count,
            self.rows,
        )?;
        let scalar_values = [
            spans.count,
            spans.packed_first,
            spans.input_first,
            spans.input_stride,
            spans.output_first,
            spans.output_stride,
            packed_end,
            input_end,
            output_end,
            packed_words,
            input_words,
            output_words,
        ];
        if scalar_values.iter().any(|&v| v > u32::MAX as usize)
            || packed_end.div_ceil(4) > packed_words
            || input_end > input_words
            || output_end > output_words
        {
            return Err("full-buffer span exceeds binding or shader indexing");
        }
        Ok(())
    }
}

fn span_end(
    first: usize,
    stride: usize,
    count: usize,
    width: usize,
) -> Result<usize, &'static str> {
    (count - 1)
        .checked_mul(stride)
        .and_then(|last| first.checked_add(last))
        .and_then(|last| last.checked_add(width))
        .ok_or("activation span overflow")
}

/// Request-private runtime addressing, never a pipeline/cache key or log item.
/// Integer positions are wiped on Drop; no Clone/Copy/Debug/Hash/serialization.
pub struct BatchSpans {
    count: usize,
    packed_first: usize,
    input_first: usize,
    input_stride: usize,
    output_first: usize,
    output_stride: usize,
}
impl BatchSpans {
    pub fn new(
        count: usize,
        packed_first: usize,
        input_first: usize,
        input_stride: usize,
        output_first: usize,
        output_stride: usize,
    ) -> Self {
        Self {
            count,
            packed_first,
            input_first,
            input_stride,
            output_first,
            output_stride,
        }
    }
    pub fn count(&self) -> usize {
        self.count
    }
    pub fn packed_first(&self) -> usize {
        self.packed_first
    }
    pub fn input_first(&self) -> usize {
        self.input_first
    }
    pub fn input_stride(&self) -> usize {
        self.input_stride
    }
    pub fn output_first(&self) -> usize {
        self.output_first
    }
    pub fn output_stride(&self) -> usize {
        self.output_stride
    }
    fn wipe(&mut self) {
        self.count.zeroize();
        self.packed_first.zeroize();
        self.input_first.zeroize();
        self.input_stride.zeroize();
        self.output_first.zeroize();
        self.output_stride.zeroize();
    }
}
impl Drop for BatchSpans {
    fn drop(&mut self) {
        self.wipe();
    }
}

/// Preconditions: validated MatrixGeometry/BatchSpans/full array lengths and
/// nonaliasing same-owner bindings; grid=geometry.grid(), cube=(64,1,1).
/// All barriers are unconditional, including inactive rows/tokens.
#[cube(launch, launch_unchecked)]
pub fn packed_matmul(
    packed: &Array<u32>,
    input: &Array<f32>,
    output: &mut Array<f32>,
    count: usize,
    packed_first: usize,
    input_first: usize,
    input_stride: usize,
    output_first: usize,
    output_stride: usize,
    #[comptime] rows: usize,
    #[comptime] columns: usize,
    #[comptime] dtype: u32,
    #[comptime] tile_tokens: usize,
) {
    let row = CUBE_POS_X as usize;
    let lane = UNIT_POS_X as usize;
    let first_token = CUBE_POS_Y as usize * tile_tokens;
    let mut partials = Array::<f32>::new(tile_tokens);
    #[unroll]
    for t in 0..tile_tokens {
        partials[t] = 0.0_f32;
    }
    if row < rows && first_token < count {
        let mut row_bytes = 0_usize;
        if comptime!(dtype == F32) {
            row_bytes = columns * 4;
        } else if comptime!(dtype == F16 || dtype == BF16) {
            row_bytes = columns * 2;
        } else if comptime!(dtype == Q5_0) {
            row_bytes = (columns / 32) * 22;
        } else if comptime!(dtype == Q8_0) {
            row_bytes = (columns / 32) * 34;
        } else if comptime!(dtype == Q4_K) {
            row_bytes = (columns / 256) * 144;
        } else if comptime!(dtype == Q6_K) {
            row_bytes = (columns / 256) * 210;
        }
        let row_offset = packed_first + row * row_bytes;
        for tile in 0..columns / LANES {
            let column = tile * LANES + lane;
            let weight = packed_value(packed, row_offset, column, dtype);
            #[unroll]
            for t in 0..tile_tokens {
                let token = first_token + t;
                if token < count {
                    partials[t] += weight * input[input_first + token * input_stride + column];
                }
            }
        }
        if lane < columns % LANES {
            let column = (columns / LANES) * LANES + lane;
            let weight = packed_value(packed, row_offset, column, dtype);
            #[unroll]
            for t in 0..tile_tokens {
                let token = first_token + t;
                if token < count {
                    partials[t] += weight * input[input_first + token * input_stride + column];
                }
            }
        }
    }
    let mut sums = SharedMemory::<f32>::new(LANES * tile_tokens);
    #[unroll]
    for t in 0..tile_tokens {
        sums[t * LANES + lane] = partials[t];
    }
    sync_cube();
    let mut stride = 32_usize;
    while stride > 0 {
        if lane < stride {
            #[unroll]
            for t in 0..tile_tokens {
                sums[t * LANES + lane] += sums[t * LANES + lane + stride];
            }
        }
        sync_cube();
        stride /= 2;
    }
    if lane == 0 && row < rows {
        #[unroll]
        for t in 0..tile_tokens {
            let token = first_token + t;
            if token < count {
                output[output_first + token * output_stride + row] = sums[t * LANES];
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use kuno_text::TensorType;

    fn fixture(dtype: u32, rows: usize, columns: usize) -> Vec<u8> {
        let (elements, bytes) = super::super::block_layout(dtype).unwrap();
        let mut result = vec![0_u8; rows * columns / elements * bytes];
        for (index, block) in result.chunks_exact_mut(bytes).enumerate() {
            match dtype {
                F32 => {
                    let value = (index as i32 % 29 - 14) as f32 / 32.0;
                    block.copy_from_slice(&value.to_le_bytes());
                }
                F16 => {
                    let values = [0x3000_u16, 0xb400, 0x3800, 0xb800, 0x3400, 0x0000];
                    block.copy_from_slice(&values[index % values.len()].to_le_bytes());
                }
                BF16 => {
                    let values = [0x3e00_u16, 0xbe80, 0x3f00, 0xbf00, 0x3e80, 0x0000];
                    block.copy_from_slice(&values[index % values.len()].to_le_bytes());
                }
                Q5_0 | Q8_0 => {
                    block[..2].copy_from_slice(&0x3000_u16.to_le_bytes());
                    for (i, byte) in block[2..].iter_mut().enumerate() {
                        *byte = (i * 43 + index * 17 + 11) as u8;
                    }
                }
                Q4_K => {
                    block[..2].copy_from_slice(&0x3000_u16.to_le_bytes());
                    block[2..4].copy_from_slice(&0x2c00_u16.to_le_bytes());
                    for (i, byte) in block[4..].iter_mut().enumerate() {
                        *byte = (i * 31 + index * 13 + 23) as u8;
                    }
                }
                Q6_K => {
                    for (i, byte) in block[..208].iter_mut().enumerate() {
                        *byte = (i * 37 + index * 29 + 7) as u8;
                    }
                    block[208..].copy_from_slice(&0x3000_u16.to_le_bytes());
                }
                _ => unreachable!(),
            }
        }
        result
    }

    // CPU schedule emulator using INDEPENDENT core decoding. The expected dot
    // below uses independent f64 accumulation, not this schedule or the shader.
    fn emulate(
        geometry: MatrixGeometry,
        spans: &BatchSpans,
        decoded: &[f32],
        input: &[f32],
        output: &mut [f32],
    ) -> Vec<usize> {
        let mut visits = vec![0; decoded.len()];
        for tile_first in (0..geometry.member_capacity).step_by(geometry.tile_tokens) {
            if tile_first >= spans.count {
                continue;
            }
            for row in 0..geometry.rows {
                let mut partials = [[0_f32; LANES]; MAX_BATCH];
                for lane in 0..LANES {
                    for column in (lane..geometry.columns).step_by(LANES) {
                        let index = row * geometry.columns + column;
                        let weight = decoded[index];
                        visits[index] += 1;
                        for (local, token_partials) in
                            partials[..geometry.tile_tokens].iter_mut().enumerate()
                        {
                            let token = tile_first + local;
                            if token < spans.count {
                                token_partials[lane] += weight
                                    * input
                                        [spans.input_first + token * spans.input_stride + column];
                            }
                        }
                    }
                }
                let mut stride = LANES / 2;
                while stride > 0 {
                    for token_partials in &mut partials[..geometry.tile_tokens] {
                        for lane in 0..stride {
                            token_partials[lane] += token_partials[lane + stride];
                        }
                    }
                    stride /= 2;
                }
                for (local, token_partials) in partials[..geometry.tile_tokens].iter().enumerate() {
                    let token = tile_first + local;
                    if token < spans.count {
                        output[spans.output_first + token * spans.output_stride + row] =
                            token_partials[0];
                    }
                }
            }
        }
        visits
    }

    #[test]
    fn all_formats_batches_and_tiles_match_independent_numeric_expectations() {
        for dtype in [F32, F16, BF16, Q5_0, Q8_0, Q4_K, Q6_K] {
            let (block, _) = super::super::block_layout(dtype).unwrap();
            let columns = if block == 1 { 137 } else { block * 3 };
            let rows = 3;
            let packed = fixture(dtype, rows, columns);
            let mut decoded = vec![0_f32; rows * columns];
            TensorType::from_gguf_id(dtype)
                .unwrap()
                .decode(&packed, &mut decoded)
                .unwrap();
            for tile_tokens in [1, 4, 8] {
                let geometry = MatrixGeometry::new(dtype, rows, columns, tile_tokens).unwrap();
                let packed_first = if dtype == F32 { 12 } else { 3 };
                let packed_words = (packed_first + packed.len()).div_ceil(4);
                for count in 1..=MAX_BATCH {
                    let spans = BatchSpans::new(count, packed_first, 9, columns + 11, 7, rows + 5);
                    let mut input = vec![12345_f32; 9 + MAX_BATCH * spans.input_stride];
                    for token in 0..count {
                        for column in 0..columns {
                            input[9 + token * spans.input_stride + column] =
                                ((column * 3 + token * 7) % 9) as f32 / 8.0 - 0.5;
                        }
                    }
                    let mut output = vec![-12345_f32; 7 + MAX_BATCH * spans.output_stride];
                    geometry
                        .validate_spans(&spans, packed_words, input.len(), output.len())
                        .unwrap();
                    let visits = emulate(geometry, &spans, &decoded, &input, &mut output);
                    assert!(visits.iter().all(|&n| n == count.div_ceil(tile_tokens)));
                    for (index, &actual) in output.iter().enumerate() {
                        let token = index.saturating_sub(spans.output_first) / spans.output_stride;
                        let row = index.saturating_sub(spans.output_first) % spans.output_stride;
                        if index < spans.output_first || token >= count || row >= rows {
                            assert_eq!(actual, -12345.0, "output padding changed");
                            continue;
                        }
                        let weights = &decoded[row * columns..(row + 1) * columns];
                        let x = &input[spans.input_first + token * spans.input_stride..][..columns];
                        let expected = weights
                            .iter()
                            .zip(x)
                            .map(|(&w, &v)| f64::from(w) * f64::from(v))
                            .sum::<f64>();
                        let absolute = weights
                            .iter()
                            .zip(x)
                            .map(|(&w, &v)| (f64::from(w) * f64::from(v)).abs())
                            .sum::<f64>();
                        assert!(
                            (f64::from(actual) - expected).abs()
                                <= 1e-6 + absolute * f64::from(f32::EPSILON) * 16.0
                        );
                        assert_eq!(
                            actual.to_bits(),
                            super::super::cooperative::reference_tree_dot(weights, x)
                                .unwrap()
                                .to_bits()
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn geometry_and_runtime_bounds_reject_before_any_upload_or_launch() {
        for (dtype, rows, columns, tile) in [
            (99, 1, 1, 4),
            (F32, 0, 1, 4),
            (F32, 1, 0, 4),
            (F32, 1, 1, 0),
            (F32, 1, 1, 2),
            (F32, 1, 1, 3),
            (F32, 1, 1, 16),
            (F32, 65536, 1, 4),
            (F32, 1, usize::MAX, 4),
            (Q4_K, 1, 255, 4),
            (Q4_K, 65535, (usize::MAX / 256) * 256, 4),
            (F32, 65535, u32::MAX as usize / 4, 4),
        ] {
            assert!(MatrixGeometry::new(dtype, rows, columns, tile).is_err());
        }
        let geometry = MatrixGeometry::new(F32, 3, 65, 4).unwrap();
        let good = BatchSpans::new(8, 12, 9, 76, 7, 8);
        let packed_words = (12 + geometry.total_bytes()).div_ceil(4);
        let input_words = span_end(9, 76, 8, 65).unwrap();
        let output_words = span_end(7, 8, 8, 3).unwrap();
        geometry
            .validate_spans(&good, packed_words, input_words, output_words)
            .unwrap();
        assert!(
            geometry
                .validate_spans(&good, packed_words - 1, input_words, output_words)
                .is_err()
        );
        assert!(
            geometry
                .validate_spans(&good, packed_words, input_words - 1, output_words)
                .is_err()
        );
        assert!(
            geometry
                .validate_spans(&good, packed_words, input_words, output_words - 1)
                .is_err()
        );
        assert!(
            geometry
                .validate_spans(&good, usize::MAX, input_words, output_words)
                .is_err()
        );
        for spans in [
            BatchSpans::new(0, 12, 9, 76, 7, 8),
            BatchSpans::new(9, 12, 9, 76, 7, 8),
            BatchSpans::new(8, 13, 9, 76, 7, 8),
            BatchSpans::new(8, 12, 9, 64, 7, 8),
            BatchSpans::new(8, 12, 9, 76, 7, 2),
            BatchSpans::new(8, usize::MAX, 9, 76, 7, 8),
            BatchSpans::new(8, 12, usize::MAX, 76, 7, 8),
            BatchSpans::new(8, 12, 9, usize::MAX, 7, 8),
            BatchSpans::new(8, 12, 9, 76, usize::MAX, 8),
            BatchSpans::new(8, 12, 9, 76, 7, usize::MAX),
            BatchSpans::new(8, 12, u32::MAX as usize, 76, 7, 8),
        ] {
            assert!(
                geometry
                    .validate_spans(&spans, packed_words, input_words, output_words)
                    .is_err()
            );
        }
    }

    #[test]
    fn runtime_span_owner_wipes_all_private_addressing_fields() {
        let mut spans = BatchSpans::new(7, 11, 13, 17, 19, 23);
        spans.wipe();
        assert_eq!(spans.count(), 0);
        assert_eq!(spans.packed_first(), 0);
        assert_eq!(spans.input_first(), 0);
        assert_eq!(spans.input_stride(), 0);
        assert_eq!(spans.output_first(), 0);
        assert_eq!(spans.output_stride(), 0);
    }

    fn shader(geometry: MatrixGeometry, mode: cubecl::server::ExecutionMode) -> String {
        use cubecl::compute::KernelBuilder;
        use cubecl::ir::{ElemType, FloatKind, IntKind, StorageType, Type, UIntKind};
        use cubecl::wgpu::WgslCompiler;
        use cubecl::{Compiler, KernelSettings, WgpuCompilationOptions};
        let uint = StorageType::Scalar(ElemType::UInt(UIntKind::U32));
        let float = StorageType::Scalar(ElemType::Float(FloatKind::F32));
        let mut b = KernelBuilder::default();
        b.scope.register_type::<u32>(uint);
        b.scope.register_type::<usize>(uint);
        b.scope.register_type::<f32>(float);
        b.scope
            .register_type::<i32>(StorageType::Scalar(ElemType::Int(IntKind::I32)));
        b.scope
            .register_type::<bool>(StorageType::Scalar(ElemType::Bool));
        let packed = b.input_array(Type::new(uint)).into();
        let input = b.input_array(Type::new(float)).into();
        let output = b.output_array(Type::new(float)).into();
        let count = b.scalar(uint).into();
        let packed_first = b.scalar(uint).into();
        let input_first = b.scalar(uint).into();
        let input_stride = b.scalar(uint).into();
        let output_first = b.scalar(uint).into();
        let output_stride = b.scalar(uint).into();
        packed_matmul::expand(
            &mut b.scope,
            packed,
            input,
            output,
            count,
            packed_first,
            input_first,
            input_stride,
            output_first,
            output_stride,
            geometry.rows,
            geometry.columns,
            geometry.dtype,
            geometry.tile_tokens,
        );
        WgslCompiler::default()
            .compile(
                b.build(
                    KernelSettings::default()
                        .cube_dim(CubeDim::new_1d(64))
                        .kernel_name("packed_multi_rhs"),
                ),
                &WgpuCompilationOptions::default(),
                mode,
                uint,
            )
            .unwrap()
            .to_string()
    }

    #[test]
    fn offline_wgsl_all_formats_both_modes_reuse_weights_and_keep_private_scalars_runtime() {
        use cubecl::server::ExecutionMode;
        for dtype in [F32, F16, BF16, Q5_0, Q8_0, Q4_K, Q6_K] {
            for mode in [ExecutionMode::Checked, ExecutionMode::Unchecked] {
                let one = shader(MatrixGeometry::new(dtype, 3, 256, 1).unwrap(), mode);
                let a = shader(MatrixGeometry::new(dtype, 3, 256, 4).unwrap(), mode);
                let b = shader(MatrixGeometry::new(dtype, 3, 256, 8).unwrap(), mode);
                for source in [&one, &a, &b] {
                    assert!(source.contains("@compute") && source.contains("workgroupBarrier"));
                    let module = wgpu::naga::front::wgsl::parse_str(source).unwrap();
                    wgpu::naga::valid::Validator::new(
                        wgpu::naga::valid::ValidationFlags::all(),
                        wgpu::naga::valid::Capabilities::empty(),
                    )
                    .validate(&module)
                    .unwrap();
                    for scalar in 0..6 {
                        assert!(
                            source.contains(&format!("scalars_u32[{scalar}]")),
                            "private scalar missing from runtime: {source}"
                        );
                    }
                }
                let a_loads = a.matches("buffer_0_global[").count();
                let b_loads = b.matches("buffer_0_global[").count();
                assert!(a_loads > 0);
                assert_eq!(one.matches("buffer_0_global[").count(), a_loads);
                assert_eq!(a_loads, b_loads, "weight decode multiplied with token tile");
                assert_eq!(
                    a.matches("buffer_1_global[").count(),
                    4 * one.matches("buffer_1_global[").count()
                );
                assert_eq!(
                    b.matches("buffer_1_global[").count(),
                    2 * a.matches("buffer_1_global[").count()
                );
                assert_eq!(one.matches("buffer_2_global[").count(), 1);
                assert_eq!(a.matches("buffer_2_global[").count(), 4);
                assert_eq!(b.matches("buffer_2_global[").count(), 8);
            }
        }
    }

    #[test]
    fn only_public_geometry_enters_specialization_or_pipeline_key() {
        use std::hash::{DefaultHasher, Hash, Hasher};
        let geometry = MatrixGeometry::new(Q4_K, 3, 256, 4).unwrap();
        let mut expected = DefaultHasher::new();
        geometry.hash(&mut expected);
        let expected = expected.finish();
        for count in 1..=MAX_BATCH {
            let spans = BatchSpans::new(count, count * 3, count * 7, 267, count * 11, 8);
            let mut actual = DefaultHasher::new();
            geometry.hash(&mut actual);
            assert_eq!(actual.finish(), expected);
            assert_eq!(spans.count(), count);
        }
        let source = include_str!("quant_matmul.rs");
        let signature = source
            .split("pub fn packed_matmul(")
            .nth(1)
            .unwrap()
            .split(") {")
            .next()
            .unwrap();
        let runtime = signature.split("#[comptime]").next().unwrap();
        for field in [
            "count",
            "packed_first",
            "input_first",
            "input_stride",
            "output_first",
            "output_stride",
        ] {
            assert!(runtime.contains(&format!("{field}: usize")));
        }
        assert_eq!(signature.matches("#[comptime]").count(), 4);
    }

    #[test]
    fn each_public_tile_keeps_max8_grid_independent_of_active_count() {
        for tile in [1, 4, 8] {
            let geometry = MatrixGeometry::new(F32, 3, 65, tile).unwrap();
            for count in [1, 3, 5, 8] {
                let spans = BatchSpans::new(count, 12, 9, 76, 7, 8);
                geometry.validate_spans(&spans, 198, 617, 71).unwrap();
                assert!(
                    matches!(geometry.grid(), CubeCount::Static(3, y, 1) if y == (MAX_BATCH / tile) as u32)
                );
            }
        }
    }

    #[test]
    fn explicit_single_member_policy_keeps_full_bindings_and_rejects_other_counts() {
        for dtype in [F32, F16, BF16, Q5_0, Q8_0, Q4_K, Q6_K] {
            let original = MatrixGeometry::new(dtype, 3, 256, 1).unwrap();
            let single = original.with_single_member().unwrap();
            assert_eq!(original.member_capacity(), MAX_BATCH);
            assert_eq!(single.member_capacity(), 1);
            assert_eq!(single.total_bytes(), original.total_bytes());
            assert!(matches!(single.grid(), CubeCount::Static(3, 1, 1)));
            assert!(matches!(original.grid(), CubeCount::Static(3, 8, 1)));
            for count in [0, 1, 2, 3, 5, 8, usize::MAX] {
                let spans = BatchSpans::new(count, 0, 9, 267, 7, 8);
                assert_eq!(
                    single
                        .validate_spans(&spans, original.total_bytes().div_ceil(4), 2145, 71)
                        .is_ok(),
                    count == 1
                );
            }
            for tile in [4, 8] {
                assert!(
                    MatrixGeometry::new(dtype, 3, 256, tile)
                        .unwrap()
                        .with_single_member()
                        .is_err()
                );
            }
        }
    }

    #[test]
    fn single_member_after_max8_prefill_preserves_every_inactive_word_and_original_math() {
        for dtype in [F32, F16, BF16, Q5_0, Q8_0, Q4_K, Q6_K] {
            let columns = 256;
            let rows = 3;
            let packed = fixture(dtype, rows, columns);
            let mut decoded = vec![0.; rows * columns];
            TensorType::from_gguf_id(dtype)
                .unwrap()
                .decode(&packed, &mut decoded)
                .unwrap();
            let original = MatrixGeometry::new(dtype, rows, columns, 1).unwrap();
            let single = original.with_single_member().unwrap();
            let mut input = vec![31415.; 9 + MAX_BATCH * 267];
            for token in 0..MAX_BATCH {
                for column in 0..columns {
                    input[9 + token * 267 + column] =
                        ((column * 3 + token * 5) % 17) as f32 / 17. - 8. / 17.;
                }
            }
            let mut prefill = vec![-12345.; 7 + MAX_BATCH * 8];
            emulate(
                MatrixGeometry::new(dtype, rows, columns, 8).unwrap(),
                &BatchSpans::new(8, 0, 9, 267, 7, 8),
                &decoded,
                &input,
                &mut prefill,
            );
            let before = prefill.clone();
            for column in 0..columns {
                input[9 + column] = -input[9 + column] + 0.0625;
            }
            let spans = BatchSpans::new(1, 0, 9, 267, 7, 8);
            let mut old = before.clone();
            let mut actual = before.clone();
            emulate(original, &spans, &decoded, &input, &mut old);
            emulate(single, &spans, &decoded, &input, &mut actual);
            assert_eq!(
                actual.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                old.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
            );
            for (i, value) in actual.iter().enumerate() {
                if !(7..10).contains(&i) {
                    assert_eq!(value.to_bits(), before[i].to_bits());
                }
            }
        }
    }
}
