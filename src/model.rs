use anyhow::{Context, Result, bail};
use ndarray::{Array1, Array2, Array3, azip, s};
use ndarray_npy::NpzReader;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::{
    fs::File,
    io::{Cursor, Read, Seek},
    path::Path,
};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelMeta {
    pub idim: usize,
    pub r: usize,
    pub m: usize,
    pub h: usize,
    pub p: usize,
    pub n1: usize,
    pub s1: usize,
    pub n2: usize,
    pub s2: usize,
    pub odim: usize,
    pub dropout: f32,
}

#[derive(Debug, Clone)]
struct Linear {
    weight_t: Array2<f32>, // [in, out]
    bias: Option<Array1<f32>>,
}

impl Linear {
    fn apply(&self, x: &Array2<f32>) -> Array2<f32> {
        // y[N,out] = x[N,in] * weight_t[in,out], then add bias.
        //
        // Hand the matrices to the `gemm` crate instead of ndarray's default
        // `matrixmultiply`-backed dot. gemm does runtime CPU-feature detection
        // (AVX-512 / AVX2 / AVX / SSE on x86, NEON on ARM, scalar otherwise) and
        // picks the best micro-kernel, with no compile-time ISA requirement, so
        // the binary still runs on old CPUs (they simply take the scalar path).
        //
        // Both ndarray and gemm express strides as element counts. We read the
        // real strides from each view so this works regardless of how the array
        // is laid out (note weight_t is column-major after reversed_axes()).
        let (n, k) = (x.nrows(), x.ncols());
        let out_dim = self.weight_t.ncols();
        let mut y = Array2::<f32>::zeros((n, out_dim));

        let x_strides = x.strides();
        let w_strides = self.weight_t.strides();
        let lhs_cs = x_strides[1];
        let lhs_rs = x_strides[0];
        let rhs_cs = w_strides[1];
        let rhs_rs = w_strides[0];

        // dst is a freshly-allocated row-major Array2.
        let ys = y.as_slice_mut().expect("freshly allocated array is contiguous");
        let lhs_ptr = x.as_ptr();
        let rhs_ptr = self.weight_t.as_ptr();

        unsafe {
            gemm::gemm(
                n,                // m = rows of x / y
                out_dim,          // n = output columns
                k,                // k = contraction dim (input features)
                ys.as_mut_ptr(),  // dst [m, n] row-major
                1isize,           // dst_cs (col stride = 1 element)
                out_dim as isize, // dst_rs (row stride = n elements)
                false,            // read_dst (alpha == 0)
                lhs_ptr,          // lhs [m, k]
                lhs_cs,
                lhs_rs,
                rhs_ptr,          // rhs [k, n]
                rhs_cs,
                rhs_rs,
                0.0f32,           // alpha = dst scale (dst not read, so 0)
                1.0f32,           // beta = product scale (gemm computes alpha*dst + beta*lhs*rhs)
                false,            // conj
                false,
                false,
                gemm::Parallelism::Rayon(0),
            );
        }

        if let Some(bias) = &self.bias {
            for mut row in y.rows_mut() {
                for i in 0..bias.len() {
                    row[i] += bias[i];
                }
            }
        }
        y
    }
}

#[derive(Debug, Clone)]
struct FsmnLayer {
    lookback_weight: Array2<f32>,  // [P, N1]
    lookahead_weight: Array2<f32>, // [P, N2]
    n1: usize,
    s1: usize,
    n2: usize,
    s2: usize,
}

impl FsmnLayer {
    fn apply(&self, x: &Array2<f32>) -> Array2<f32> {
        let t_max = x.nrows();
        let p = x.ncols();
        let mut out = vec![0.0f32; t_max * p];
        out.par_chunks_mut(p).enumerate().for_each(|(t, row)| {
            for c in 0..p {
                let mut lb = 0.0f32;
                for k in 0..self.n1 {
                    let idx = t as isize - ((self.n1 - 1 - k) * self.s1) as isize;
                    if idx >= 0 {
                        lb += self.lookback_weight[[c, k]] * x[[idx as usize, c]];
                    }
                }
                let mut la = 0.0f32;
                for k in 0..self.n2 {
                    let idx = t + (k + 1) * self.s2;
                    if idx < t_max {
                        la += self.lookahead_weight[[c, k]] * x[[idx, c]];
                    }
                }
                row[c] = x[[t, c]] + lb + la;
            }
        });
        Array2::from_shape_vec((t_max, p), out).expect("shape should always match")
    }
}

#[derive(Debug, Clone)]
struct DfsmnBlock {
    fc1: Linear,
    fc2: Linear,
    fsmn: FsmnLayer,
}

#[derive(Debug, Clone)]
pub struct DetectModel {
    pub meta: ModelMeta,
    fc1: Linear,
    fc2: Linear,
    fsmn1: FsmnLayer,
    blocks: Vec<DfsmnBlock>,
    dnn_layers: Vec<Linear>,
    out: Linear,
}

impl DetectModel {
    pub fn from_dir(model_dir: &Path) -> Result<Self> {
        let meta_path = model_dir.join("model_meta.json");
        let meta_text = std::fs::read_to_string(&meta_path)
            .with_context(|| format!("failed to read {}", meta_path.display()))?;
        let meta: ModelMeta =
            serde_json::from_str(&meta_text).context("bad model_meta.json format")?;

        let npz_path = model_dir.join("weights.npz");
        let f = File::open(&npz_path)
            .with_context(|| format!("failed to open {}", npz_path.display()))?;
        let mut npz = NpzReader::new(f).context("failed to parse weights.npz")?;
        Self::from_meta_and_npz(meta, &mut npz)
    }

    pub fn from_embedded(meta_json: &[u8], weights_npz: &[u8]) -> Result<Self> {
        let meta_text =
            std::str::from_utf8(meta_json).context("embedded model_meta is not utf8")?;
        let meta: ModelMeta =
            serde_json::from_str(meta_text).context("bad embedded model_meta.json format")?;
        let cursor = Cursor::new(weights_npz);
        let mut npz = NpzReader::new(cursor).context("failed to parse embedded weights.npz")?;
        Self::from_meta_and_npz(meta, &mut npz)
    }

    fn from_meta_and_npz<R: Read + Seek>(meta: ModelMeta, npz: &mut NpzReader<R>) -> Result<Self> {
        let fc1 = Linear {
            weight_t: read_arr2(npz, "dfsmn.fc1.0.weight")?.reversed_axes(),
            bias: Some(read_arr1(npz, "dfsmn.fc1.0.bias")?),
        };
        let fc2 = Linear {
            weight_t: read_arr2(npz, "dfsmn.fc2.0.weight")?.reversed_axes(),
            bias: Some(read_arr1(npz, "dfsmn.fc2.0.bias")?),
        };

        let fsmn1 = FsmnLayer {
            lookback_weight: read_depthwise(npz, "dfsmn.fsmn1.lookback_filter.weight")?,
            lookahead_weight: read_depthwise(npz, "dfsmn.fsmn1.lookahead_filter.weight")?,
            n1: meta.n1,
            s1: meta.s1,
            n2: meta.n2,
            s2: meta.s2,
        };

        let mut blocks = Vec::new();
        for i in 0..(meta.r.saturating_sub(1)) {
            let b = DfsmnBlock {
                fc1: Linear {
                    weight_t: read_arr2(npz, &format!("dfsmn.fsmns.{i}.fc1.0.weight"))?
                        .reversed_axes(),
                    bias: Some(read_arr1(npz, &format!("dfsmn.fsmns.{i}.fc1.0.bias"))?),
                },
                fc2: Linear {
                    weight_t: read_arr2(npz, &format!("dfsmn.fsmns.{i}.fc2.weight"))?
                        .reversed_axes(),
                    bias: None,
                },
                fsmn: FsmnLayer {
                    lookback_weight: read_depthwise(
                        npz,
                        &format!("dfsmn.fsmns.{i}.fsmn.lookback_filter.weight"),
                    )?,
                    lookahead_weight: read_depthwise(
                        npz,
                        &format!("dfsmn.fsmns.{i}.fsmn.lookahead_filter.weight"),
                    )?,
                    n1: meta.n1,
                    s1: meta.s1,
                    n2: meta.n2,
                    s2: meta.s2,
                },
            };
            blocks.push(b);
        }

        let mut dnn_layers = Vec::new();
        let mut dnn_idx = 0usize;
        loop {
            let w_key = format!("dfsmn.dnns.{dnn_idx}.weight");
            let b_key = format!("dfsmn.dnns.{dnn_idx}.bias");
            let Ok(weight) =
                npz.by_name::<ndarray::OwnedRepr<f32>, ndarray::Ix2>(&format!("{w_key}.npy"))
            else {
                break;
            };
            let bias = npz
                .by_name::<ndarray::OwnedRepr<f32>, ndarray::Ix1>(&format!("{b_key}.npy"))
                .with_context(|| format!("missing {b_key}"))?;
            dnn_layers.push(Linear {
                weight_t: weight.reversed_axes(),
                bias: Some(bias),
            });
            dnn_idx += 3;
        }
        if dnn_layers.is_empty() {
            bail!("no DNN layers found in weights.npz");
        }

        let out = Linear {
            weight_t: read_arr2(npz, "out.weight")?.reversed_axes(),
            bias: Some(read_arr1(npz, "out.bias")?),
        };

        Ok(Self {
            meta,
            fc1,
            fc2,
            fsmn1,
            blocks,
            dnn_layers,
            out,
        })
    }

    pub fn forward(&self, feat: &Array2<f32>) -> Array1<f32> {
        // fc1 + relu
        let h = relu_inplace(self.fc1.apply(feat));
        // fc2 + relu
        let p = relu_inplace(self.fc2.apply(&h));
        let mut memory = self.fsmn1.apply(&p);
        for block in &self.blocks {
            let h = relu_inplace(block.fc1.apply(&memory));
            let p = block.fc2.apply(&h);
            let f = block.fsmn.apply(&p);
            // residual: memory += f  (in place to avoid a fresh allocation)
            azip!((m in &mut memory, fv in &f) *m += *fv);
        }
        let mut out = memory;
        for d in &self.dnn_layers {
            out = relu_inplace(d.apply(&out));
        }
        let logits = self.out.apply(&out);
        logits.column(0).to_owned().mapv(sigmoid)
    }
}

fn read_arr1<R: Read + Seek>(npz: &mut NpzReader<R>, key: &str) -> Result<Array1<f32>> {
    npz.by_name(&format!("{key}.npy"))
        .with_context(|| format!("missing key {key}"))
}

fn read_arr2<R: Read + Seek>(npz: &mut NpzReader<R>, key: &str) -> Result<Array2<f32>> {
    npz.by_name(&format!("{key}.npy"))
        .with_context(|| format!("missing key {key}"))
}

fn read_depthwise<R: Read + Seek>(npz: &mut NpzReader<R>, key: &str) -> Result<Array2<f32>> {
    let arr: Array3<f32> = npz
        .by_name(&format!("{key}.npy"))
        .with_context(|| format!("missing key {key}"))?;
    let p = arr.shape()[0];
    let k = arr.shape()[2];
    let mut out = Array2::<f32>::zeros((p, k));
    for c in 0..p {
        out.slice_mut(s![c, ..]).assign(&arr.slice(s![c, 0, ..]));
    }
    Ok(out)
}

fn relu_inplace(mut x: Array2<f32>) -> Array2<f32> {
    // Apply ReLU in place on the already-owned buffer, avoiding the extra copy
    // that mapv would make. Numerically identical to the previous mapv version.
    x.mapv_inplace(|v| if v > 0.0 { v } else { 0.0 });
    x
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn lookahead_uses_future_frames_only() {
        let layer = FsmnLayer {
            lookback_weight: array![[0.0f32]],
            lookahead_weight: array![[1.0f32, 1.0f32]],
            n1: 1,
            s1: 1,
            n2: 2,
            s2: 1,
        };
        let x = array![[1.0f32], [2.0], [3.0], [4.0]];
        let y = layer.apply(&x);
        // y[t] = x[t] + x[t+1] + x[t+2]
        assert!((y[[0, 0]] - 6.0).abs() < 1e-6);
        assert!((y[[1, 0]] - 9.0).abs() < 1e-6);
        assert!((y[[2, 0]] - 7.0).abs() < 1e-6);
        assert!((y[[3, 0]] - 4.0).abs() < 1e-6);
    }

    #[test]
    fn linear_apply_matches_ndarray_dot() {
        // Build a [in=5, out=3] weight_t (note: stored transposed, matching how
        // from_meta_and_npz builds it via reversed_axes()).
        let weight_t = array![
            [0.1f32, 0.2, 0.3],
            [0.4, 0.5, 0.6],
            [0.7, 0.8, 0.9],
            [1.0, 1.1, 1.2],
            [1.3, 1.4, 1.5],
        ];
        let bias = array![0.01f32, 0.02, 0.03];
        let lin = Linear {
            weight_t,
            bias: Some(bias),
        };
        let x = array![
            [1.0f32, 2.0, 3.0, 4.0, 5.0],
            [0.5, 0.5, 0.5, 0.5, 0.5],
            [2.0, 2.0, 2.0, 2.0, 2.0],
        ];
        let y_gemm = lin.apply(&x);
        // Reference via ndarray's own dot + bias.
        let mut y_ref = x.dot(&lin.weight_t);
        if let Some(b) = &lin.bias {
            for mut row in y_ref.rows_mut() {
                for i in 0..b.len() {
                    row[i] += b[i];
                }
            }
        }
        let max_diff = y_gemm
            .iter()
            .zip(y_ref.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_diff < 1e-5,
            "gemm path diverges from ndarray dot: max_diff = {max_diff}"
        );
    }
}
