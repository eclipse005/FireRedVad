use anyhow::Result;
use ndarray::Array2;
use realfft::RealFftPlanner;
use std::f32::consts::PI;

pub const SAMPLE_RATE: usize = 16_000;
pub const FRAME_LENGTH_MS: usize = 25;
pub const FRAME_SHIFT_MS: usize = 10;
pub const FRAME_LENGTH_SAMPLES: usize = SAMPLE_RATE * FRAME_LENGTH_MS / 1000;
pub const FRAME_SHIFT_SAMPLES: usize = SAMPLE_RATE * FRAME_SHIFT_MS / 1000;

pub struct FbankExtractor {
    num_mel_bins: usize,
    frame_length: usize,
    frame_shift: usize,
    nfft: usize,
    window: Vec<f32>,
    mel_bands: Vec<MelBand>,
}

struct MelBand {
    indices: Vec<usize>,
    weights: Vec<f32>,
}

impl FbankExtractor {
    pub fn new(num_mel_bins: usize) -> Self {
        let frame_length = FRAME_LENGTH_SAMPLES;
        let frame_shift = FRAME_SHIFT_SAMPLES;
        let nfft = frame_length.next_power_of_two();
        let window = povey_window(frame_length);
        let mel_filter = build_mel_filterbank(num_mel_bins, nfft, SAMPLE_RATE);
        let mel_bands = dense_to_sparse_bands(&mel_filter);
        Self {
            num_mel_bins,
            frame_length,
            frame_shift,
            nfft,
            window,
            mel_bands,
        }
    }

    pub fn extract(&self, wav: &[f32]) -> Result<Array2<f32>> {
        let num_frames = if wav.len() < self.frame_length {
            0
        } else {
            1 + (wav.len() - self.frame_length) / self.frame_shift
        };
        let mut out = Array2::<f32>::zeros((num_frames, self.num_mel_bins));
        if num_frames == 0 {
            return Ok(out);
        }

        let mut planner = RealFftPlanner::<f32>::new();
        let r2c = planner.plan_fft_forward(self.nfft);
        let mut inbuf = vec![0.0f32; self.nfft];
        let mut spectrum = r2c.make_output_vec();
        let mut power = vec![0.0f32; self.nfft / 2 + 1];

        for t in 0..num_frames {
            let start = t * self.frame_shift;
            let frame = &wav[start..start + self.frame_length];
            apply_preemphasis_and_window(frame, &self.window, &mut inbuf[..self.frame_length]);
            inbuf[self.frame_length..].fill(0.0);
            r2c.process(&mut inbuf, &mut spectrum)
                .map_err(|e| anyhow::anyhow!("fft failed: {e}"))?;

            for (i, c) in spectrum.iter().enumerate() {
                power[i] = c.re * c.re + c.im * c.im;
            }

            for m in 0..self.num_mel_bins {
                let mut e = 0.0f32;
                let band = &self.mel_bands[m];
                for i in 0..band.indices.len() {
                    e += band.weights[i] * power[band.indices[i]];
                }
                out[[t, m]] = e.max(1e-10).ln();
            }
        }
        Ok(out)
    }
}

fn apply_preemphasis_and_window(frame: &[f32], win: &[f32], out: &mut [f32]) {
    let pre = 0.97f32;
    if frame.is_empty() {
        return;
    }
    let mean = frame.iter().sum::<f32>() / frame.len() as f32;
    out[0] = (frame[0] - mean) * win[0];
    for i in 1..frame.len() {
        let cur = frame[i] - mean;
        let prev = frame[i - 1] - mean;
        out[i] = (cur - pre * prev) * win[i];
    }
}

fn povey_window(n: usize) -> Vec<f32> {
    let mut w = vec![0.0f32; n];
    for (i, wi) in w.iter_mut().enumerate() {
        let hamming = 0.5 - 0.5 * (2.0 * PI * i as f32 / (n as f32 - 1.0)).cos();
        *wi = hamming.powf(0.85);
    }
    w
}

fn hz_to_mel(hz: f32) -> f32 {
    1127.0 * (1.0 + hz / 700.0).ln()
}

fn build_mel_filterbank(num_mel_bins: usize, nfft: usize, sr: usize) -> Array2<f32> {
    let n_freq = nfft / 2 + 1;
    let nyquist = sr as f32 / 2.0;
    let mel_low = hz_to_mel(20.0);
    let mel_high = hz_to_mel(nyquist);

    let mut mel_points = vec![0.0f32; num_mel_bins + 2];
    for (i, p) in mel_points.iter_mut().enumerate() {
        *p = mel_low + (mel_high - mel_low) * i as f32 / (num_mel_bins + 1) as f32;
    }

    let mut fb = Array2::<f32>::zeros((num_mel_bins, n_freq));
    for m in 0..num_mel_bins {
        let left_mel = mel_points[m];
        let center_mel = mel_points[m + 1];
        let right_mel = mel_points[m + 2];
        for k in 0..n_freq {
            let hz = k as f32 * sr as f32 / nfft as f32;
            let mel = hz_to_mel(hz);
            let w = if mel > left_mel && mel <= center_mel {
                (mel - left_mel) / (center_mel - left_mel)
            } else if mel > center_mel && mel < right_mel {
                (right_mel - mel) / (right_mel - center_mel)
            } else {
                0.0
            };
            if w > 0.0 {
                fb[[m, k]] = w;
            }
        }
    }
    fb
}

fn dense_to_sparse_bands(fb: &Array2<f32>) -> Vec<MelBand> {
    let mut bands = Vec::with_capacity(fb.nrows());
    for m in 0..fb.nrows() {
        let mut indices = Vec::new();
        let mut weights = Vec::new();
        for k in 0..fb.ncols() {
            let w = fb[[m, k]];
            if w != 0.0 {
                indices.push(k);
                weights.push(w);
            }
        }
        bands.push(MelBand { indices, weights });
    }
    bands
}
