use anyhow::{Context, Result, bail};
use ndarray::Array2;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CmvnFile {
    pub dim: usize,
    pub means: Vec<f32>,
    pub inverse_std_variances: Vec<f32>,
}

#[derive(Debug, Clone)]
pub struct Cmvn {
    means: Vec<f32>,
    inv_std: Vec<f32>,
}

impl Cmvn {
    pub fn from_json_str(text: &str) -> Result<Self> {
        let file: CmvnFile = serde_json::from_str(text).context("bad cmvn json content")?;
        if file.means.len() != file.dim || file.inverse_std_variances.len() != file.dim {
            bail!("cmvn dimension mismatch");
        }
        Ok(Self {
            means: file.means,
            inv_std: file.inverse_std_variances,
        })
    }

    pub fn apply(&self, feat: &mut Array2<f32>) -> Result<()> {
        if feat.ncols() != self.means.len() {
            bail!(
                "cmvn dim mismatch: feat_dim={} cmvn_dim={}",
                feat.ncols(),
                self.means.len()
            );
        }
        for mut row in feat.rows_mut() {
            for d in 0..self.means.len() {
                row[d] = (row[d] - self.means[d]) * self.inv_std[d];
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn cmvn_applies_expected_formula() {
        let cmvn = Cmvn {
            means: vec![1.0, 2.0],
            inv_std: vec![0.5, 2.0],
        };
        let mut feat = array![[3.0f32, 4.0f32]];
        cmvn.apply(&mut feat).unwrap();
        assert!((feat[[0, 0]] - 1.0).abs() < 1e-6);
        assert!((feat[[0, 1]] - 4.0).abs() < 1e-6);
    }
}
