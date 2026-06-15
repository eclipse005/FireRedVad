//! High-level VAD facade.
//!
//! [`Vad`] bundles the embedded pretrained model, CMVN stats, and fbank
//! extractor into one reusable object, exposing a one-shot `detect_wav` /
//! `detect_pcm` API so callers never have to wire the pipeline together
//! themselves. The CLI is now just a thin wrapper around this type.

use crate::audio::read_wav_and_resample_to_16k;
use crate::cmvn::Cmvn;
use crate::fbank::FbankExtractor;
use crate::model::DetectModel;
use crate::postprocess::{VadConfig, VadOutput, decision_to_segment, process_probs};
use anyhow::Result;
use ndarray::{Array1, Array2, s};
use rayon::prelude::*;
use std::path::Path;

// The pretrained model is compiled into the binary via include_bytes! /
// include_str!, so a single shipped artifact is self-contained.
const EMBED_MODEL_META: &[u8] = include_bytes!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/model/model_meta.json"
));
const EMBED_CMVN_JSON: &str = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/model/cmvn.json"));
const EMBED_WEIGHTS_NPZ: &[u8] =
    include_bytes!(concat!(env!("CARGO_MANIFEST_DIR"), "/model/weights.npz"));

/// Default max frames per parallel inference chunk. Long audio is split into
/// non-overlapping chunks of this size and forwarded in parallel across cores.
pub const DEFAULT_CHUNK_MAX_FRAME: usize = 30_000;

/// One-shot voice-activity detector backed by the embedded pretrained model.
///
/// Build it once with [`Vad::new`] and reuse it across many calls to
/// [`detect_wav`](Self::detect_wav) / [`detect_pcm`](Self::detect_pcm).
pub struct Vad {
    model: DetectModel,
    cmvn: Cmvn,
    fbank: FbankExtractor,
}

impl Vad {
    /// Create a detector loaded with the built-in pretrained weights.
    ///
    /// All model assets (weights, CMVN, metadata) are compiled into the
    /// binary, so this needs no external files.
    pub fn new() -> Result<Self> {
        let model = DetectModel::from_embedded(EMBED_MODEL_META, EMBED_WEIGHTS_NPZ)?;
        let cmvn = Cmvn::from_json_str(EMBED_CMVN_JSON)?;
        // The fbank output dimension must match the model's input dimension.
        let fbank = FbankExtractor::new(model.meta.idim);
        Ok(Self { model, cmvn, fbank })
    }

    /// Run VAD on a WAV file.
    ///
    /// The file is decoded, down-mixed to mono, and resampled to 16 kHz
    /// internally; PCM16 and float32 samples at any sample rate are accepted.
    /// `cfg` controls all post-processing thresholds and smoothing.
    pub fn detect_wav<P: AsRef<Path>>(&self, path: P, cfg: &VadConfig) -> Result<VadOutput> {
        self.detect_wav_chunked(path, cfg, DEFAULT_CHUNK_MAX_FRAME)
    }

    /// Like [`detect_wav`](Self::detect_wav) but lets the caller override the
    /// max frames per parallel inference chunk.
    pub fn detect_wav_chunked<P: AsRef<Path>>(
        &self,
        path: P,
        cfg: &VadConfig,
        chunk_max_frame: usize,
    ) -> Result<VadOutput> {
        let audio = read_wav_and_resample_to_16k(path.as_ref())?;
        let sample_rate_in = audio.sample_rate_in;
        self.detect_pcm_chunked(&audio.samples_16k_mono, cfg, chunk_max_frame)
            .map(|mut out| {
                out.wav_path = path.as_ref().to_string_lossy().into_owned();
                out.sample_rate_in = sample_rate_in;
                out
            })
    }

    /// Run VAD on raw 16 kHz mono PCM samples.
    ///
    /// Intended for advanced callers that feed their own audio front-end
    /// (e.g. streaming, in-memory buffers). `samples` must be mono at
    /// 16 kHz. `cfg` controls all post-processing thresholds and smoothing.
    pub fn detect_pcm(&self, samples_16k_mono: &[f32], cfg: &VadConfig) -> Result<VadOutput> {
        self.detect_pcm_chunked(samples_16k_mono, cfg, DEFAULT_CHUNK_MAX_FRAME)
    }

    /// Like [`detect_pcm`](Self::detect_pcm) but lets the caller override the
    /// max frames per parallel inference chunk.
    pub fn detect_pcm_chunked(
        &self,
        samples_16k_mono: &[f32],
        cfg: &VadConfig,
        chunk_max_frame: usize,
    ) -> Result<VadOutput> {
        let chunk = chunk_max_frame.max(1);
        let dur = if samples_16k_mono.is_empty() {
            0.0
        } else {
            samples_16k_mono.len() as f32 / 16_000.0
        };

        let mut feat = self.fbank.extract(samples_16k_mono)?;
        self.cmvn.apply(&mut feat)?;

        let probs = self.forward_chunked(&feat, chunk)?;
        let decisions = process_probs(probs.as_slice().unwrap_or(&[]), cfg);
        let segments = decision_to_segment(&decisions, Some(dur));

        Ok(VadOutput {
            dur: ((dur * 1000.0).round()) / 1000.0,
            timestamps: segments,
            wav_path: String::new(),
            sample_rate_in: 16_000,
            sample_rate_model: 16_000,
        })
    }

    /// Forward features in non-overlapping parallel chunks.
    ///
    /// Chunks are independent (each runs a full forward pass over its own
    /// rows), so they are safe to run in parallel. Each chunk produces its own
    /// owned probability vector; we collect them in order and concatenate, so
    /// the final result is identical to a single serial forward pass.
    fn forward_chunked(&self, feat: &Array2<f32>, chunk_max_frame: usize) -> Result<Array1<f32>> {
        if feat.nrows() <= chunk_max_frame {
            return Ok(self.model.forward(feat));
        }
        let total = feat.nrows();
        let start_offsets: Vec<usize> = (0..total).step_by(chunk_max_frame).collect();
        let per_chunk: Vec<Array1<f32>> = start_offsets
            .into_par_iter()
            .map(|start| -> Array1<f32> {
                let end = (start + chunk_max_frame).min(total);
                let chunk = feat.slice(s![start..end, ..]).to_owned();
                self.model.forward(&chunk)
            })
            .collect();
        let cap = per_chunk.iter().map(|c| c.len()).sum();
        let mut out = Vec::<f32>::with_capacity(cap);
        for c in per_chunk {
            out.extend(c.iter().copied());
        }
        Ok(Array1::from_vec(out))
    }
}
