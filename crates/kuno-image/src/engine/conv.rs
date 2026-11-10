//! 1D convolutions over channel-major `[channels][time]` f32 buffers, on CPU threads.
//! Each kernel tap is one matrix product on an unpadded window, so nothing is copied.
use gemm::Parallelism;

pub(super) fn threads() -> Parallelism {
    Parallelism::Rayon(0)
}
/// dst[rows x n] (row stride `dst_rs`, column stride `dst_cs`) += lhs[rows x k] * rhs[k x n].
///
/// SAFETY: the caller guarantees all three strided windows lie inside their buffers.
#[allow(clippy::too_many_arguments)]
unsafe fn accumulate(
    rows: usize,
    n: usize,
    k: usize,
    dst: *mut f32,
    dst_cs: usize,
    dst_rs: usize,
    lhs: *const f32,
    rhs: *const f32,
    rhs_rs: usize,
) {
    if n == 0 {
        return;
    }
    unsafe {
        gemm::gemm(
            rows,
            n,
            k,
            dst,
            dst_cs as isize,
            dst_rs as isize,
            true,
            lhs,
            1,
            k as isize,
            rhs,
            1,
            rhs_rs as isize,
            1.0,
            1.0,
            false,
            false,
            false,
            threads(),
        )
    }
}
fn with_bias(bias: Option<&[f32]>, channels: usize, len: usize) -> Vec<f32> {
    let mut out = vec![0.0; channels * len];
    if let Some(bias) = bias {
        for (row, b) in out.chunks_exact_mut(len.max(1)).zip(bias) {
            row.fill(*b);
        }
    }
    out
}
/// Stride-1 convolution with "same" zero padding (PyTorch `Conv1d`, odd kernels).
pub struct Conv1d {
    cin: usize,
    cout: usize,
    kernel: usize,
    dilation: usize,
    /// Per tap, a contiguous `[cout][cin]` matrix.
    taps: Vec<f32>,
    bias: Option<Vec<f32>>,
}
impl Conv1d {
    /// `weight` is PyTorch's `[cout][cin][kernel]`.
    pub fn new(
        weight: &[f32],
        bias: Option<Vec<f32>>,
        cin: usize,
        cout: usize,
        kernel: usize,
        dilation: usize,
    ) -> Result<Self, &'static str> {
        if weight.len() != cout * cin * kernel
            || kernel.is_multiple_of(2)
            || dilation == 0
            || bias.as_ref().is_some_and(|b| b.len() != cout)
        {
            return Err("model_tensor_invalid");
        }
        let mut taps = vec![0.0; weight.len()];
        for j in 0..kernel {
            for (dst, src) in taps[j * cout * cin..][..cout * cin]
                .iter_mut()
                .zip(weight[j..].iter().step_by(kernel))
            {
                *dst = *src;
            }
        }
        Ok(Self {
            cin,
            cout,
            kernel,
            dilation,
            taps,
            bias,
        })
    }
    pub fn channels(&self) -> (usize, usize) {
        (self.cin, self.cout)
    }
    pub fn forward(&self, x: &[f32], len: usize) -> Vec<f32> {
        assert_eq!(x.len(), self.cin * len);
        let mut out = with_bias(self.bias.as_deref(), self.cout, len);
        let pad = self.dilation * (self.kernel - 1) / 2;
        for j in 0..self.kernel {
            // out[t] += w_j * x[t + shift], for every t where the input exists.
            let shift = (j * self.dilation) as isize - pad as isize;
            let first = (-shift).max(0) as usize;
            let end = (len as isize - shift.max(0)).max(0) as usize;
            if first >= end {
                continue;
            }
            // SAFETY: columns first..end of `out` and first+shift..end+shift of `x` are in range.
            unsafe {
                accumulate(
                    self.cout,
                    end - first,
                    self.cin,
                    out.as_mut_ptr().add(first),
                    1,
                    len,
                    self.taps[j * self.cout * self.cin..].as_ptr(),
                    x.as_ptr().add((first as isize + shift) as usize),
                    len,
                )
            }
        }
        out
    }
}
/// Upsampling transposed convolution whose output is exactly `stride` times longer
/// (PyTorch `ConvTranspose1d` with `padding = (kernel - stride) / 2`).
pub struct ConvTranspose1d {
    cin: usize,
    cout: usize,
    kernel: usize,
    stride: usize,
    taps: Vec<f32>,
    bias: Vec<f32>,
}
impl ConvTranspose1d {
    /// `weight` is PyTorch's `[cin][cout][kernel]`.
    pub fn new(
        weight: &[f32],
        bias: Vec<f32>,
        cin: usize,
        cout: usize,
        kernel: usize,
        stride: usize,
    ) -> Result<Self, &'static str> {
        if weight.len() != cin * cout * kernel
            || stride == 0
            || kernel < stride
            || !(kernel - stride).is_multiple_of(2)
            || bias.len() != cout
        {
            return Err("model_tensor_invalid");
        }
        let mut taps = vec![0.0; weight.len()];
        for j in 0..kernel {
            for o in 0..cout {
                for i in 0..cin {
                    taps[(j * cout + o) * cin + i] = weight[(i * cout + o) * kernel + j];
                }
            }
        }
        Ok(Self {
            cin,
            cout,
            kernel,
            stride,
            taps,
            bias,
        })
    }
    pub fn stride(&self) -> usize {
        self.stride
    }
    pub fn forward(&self, x: &[f32], len: usize) -> Vec<f32> {
        assert_eq!(x.len(), self.cin * len);
        let (stride, out_len) = (self.stride, len * self.stride);
        let mut out = with_bias(Some(&self.bias), self.cout, out_len);
        let pad = (self.kernel - stride) / 2;
        for j in 0..self.kernel {
            // out[t * stride + j - pad] += w_j * x[t], for every t whose output exists.
            if j >= out_len + pad {
                continue;
            }
            let first = (pad.saturating_sub(j)).div_ceil(stride);
            let end = ((out_len + pad - j - 1) / stride + 1).min(len);
            if first >= end {
                continue;
            }
            // SAFETY: outputs first*stride+j-pad ..= (end-1)*stride+j-pad are within 0..out_len.
            unsafe {
                accumulate(
                    self.cout,
                    end - first,
                    self.cin,
                    out.as_mut_ptr().add(first * stride + j - pad),
                    stride,
                    out_len,
                    self.taps[j * self.cout * self.cin..].as_ptr(),
                    x.as_ptr().add(first),
                    len,
                )
            }
        }
        out
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    pub fn noise(count: usize, seed: u64) -> Vec<f32> {
        let mut s = seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1;
        (0..count)
            .map(|_| {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                (s >> 40) as f32 / (1u64 << 23) as f32 - 1.0
            })
            .collect()
    }
    pub fn close(a: &[f32], b: &[f32], tolerance: f32) {
        assert_eq!(a.len(), b.len());
        for (i, (a, b)) in a.iter().zip(b).enumerate() {
            assert!(
                (a - b).abs() <= tolerance * (1.0 + b.abs()),
                "index {i}: {a} vs {b}"
            );
        }
    }
    #[test]
    fn conv1d_matches_direct_sum_with_dilation_and_short_inputs() {
        for (cin, cout, kernel, dilation, len) in [
            (3, 5, 7, 1, 40),
            (4, 2, 3, 5, 9),
            (2, 3, 11, 5, 4),
            (5, 1, 1, 1, 6),
            (3, 3, 7, 3, 1),
        ] {
            let weight = noise(cout * cin * kernel, 1);
            let bias = noise(cout, 2);
            let x = noise(cin * len, 3);
            let got = Conv1d::new(&weight, Some(bias.clone()), cin, cout, kernel, dilation)
                .unwrap()
                .forward(&x, len);
            let pad = (dilation * (kernel - 1) / 2) as isize;
            let mut want = vec![0.0; cout * len];
            for o in 0..cout {
                for t in 0..len {
                    let mut sum = bias[o] as f64;
                    for i in 0..cin {
                        for j in 0..kernel {
                            let s = t as isize + (j * dilation) as isize - pad;
                            if s >= 0 && (s as usize) < len {
                                sum += weight[(o * cin + i) * kernel + j] as f64
                                    * x[i * len + s as usize] as f64;
                            }
                        }
                    }
                    want[o * len + t] = sum as f32;
                }
            }
            close(&got, &want, 2e-5);
        }
    }
    #[test]
    fn conv_transpose_matches_scatter_definition() {
        for (cin, cout, kernel, stride, len) in [
            (4, 3, 9, 5, 7),
            (2, 5, 4, 2, 11),
            (3, 2, 9, 5, 1),
            (2, 2, 4, 2, 1),
        ] {
            let weight = noise(cin * cout * kernel, 4);
            let bias = noise(cout, 5);
            let x = noise(cin * len, 6);
            let got = ConvTranspose1d::new(&weight, bias.clone(), cin, cout, kernel, stride)
                .unwrap()
                .forward(&x, len);
            let pad = (kernel - stride) / 2;
            let full = (len - 1) * stride + kernel;
            let mut want = vec![0.0; cout * len * stride];
            for o in 0..cout {
                let mut row = vec![0f64; full];
                for i in 0..cin {
                    for t in 0..len {
                        for j in 0..kernel {
                            row[t * stride + j] +=
                                weight[(i * cout + o) * kernel + j] as f64 * x[i * len + t] as f64;
                        }
                    }
                }
                for n in 0..len * stride {
                    want[o * len * stride + n] = (row[n + pad] + bias[o] as f64) as f32;
                }
            }
            close(&got, &want, 2e-5);
        }
    }
    #[test]
    fn wrong_shapes_are_rejected() {
        assert!(Conv1d::new(&[0.0; 12], None, 2, 2, 4, 1).is_err());
        assert!(Conv1d::new(&[0.0; 12], Some(vec![0.0]), 2, 2, 3, 1).is_err());
        assert!(ConvTranspose1d::new(&[0.0; 12], vec![0.0; 2], 2, 2, 3, 2).is_err());
    }
}
