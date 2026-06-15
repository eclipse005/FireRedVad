use crate::audio::read_wav_and_resample_to_16k;
use crate::cmvn::Cmvn;
use crate::fbank::FbankExtractor;
use crate::model::DetectModel;
use crate::postprocess::{VadConfig, VadOutput, decision_to_segment, process_probs};
use anyhow::Result;
use clap::Parser;
use ndarray::{Array1, Array2, s};
use rayon::prelude::*;
use std::path::PathBuf;

const EMBED_MODEL_META: &[u8] = include_bytes!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/model/model_meta.json"
));
const EMBED_CMVN_JSON: &str =
    include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/model/cmvn.json"));
const EMBED_WEIGHTS_NPZ: &[u8] =
    include_bytes!(concat!(env!("CARGO_MANIFEST_DIR"), "/model/weights.npz"));

#[derive(Debug, Parser)]
#[command(author, version, about = "FireRedVAD non-streaming Rust CLI")]
struct Args {
    #[arg(value_name = "WAV")]
    wav: PathBuf,
    #[arg(long, default_value_t = 0.4)]
    speech_threshold: f32,
    #[arg(long, default_value_t = 5)]
    smooth_window_size: usize,
    #[arg(long, default_value_t = 20)]
    min_speech_frame: usize,
    #[arg(long, default_value_t = 2000)]
    max_speech_frame: usize,
    #[arg(long, default_value_t = 20)]
    min_silence_frame: usize,
    #[arg(long, default_value_t = 0)]
    merge_silence_frame: usize,
    #[arg(long, default_value_t = 0)]
    extend_speech_frame: usize,
    #[arg(long, default_value_t = 30000)]
    chunk_max_frame: usize,
}

pub fn run() -> Result<()> {
    let args = Args::parse();
    let cfg = VadConfig {
        smooth_window_size: args.smooth_window_size.max(1),
        speech_threshold: args.speech_threshold,
        min_speech_frame: args.min_speech_frame.max(1),
        max_speech_frame: args.max_speech_frame.max(1),
        min_silence_frame: args.min_silence_frame,
        merge_silence_frame: args.merge_silence_frame,
        extend_speech_frame: args.extend_speech_frame,
    };

    let model = DetectModel::from_embedded(EMBED_MODEL_META, EMBED_WEIGHTS_NPZ)?;
    let cmvn = Cmvn::from_json_str(EMBED_CMVN_JSON)?;
    let audio = read_wav_and_resample_to_16k(&args.wav)?;
    let dur = if audio.samples_16k_mono.is_empty() {
        0.0
    } else {
        audio.samples_16k_mono.len() as f32 / 16_000.0
    };

    let fbank = FbankExtractor::new(model.meta.idim);
    let mut feat = fbank.extract(&audio.samples_16k_mono)?;
    cmvn.apply(&mut feat)?;

    let probs = forward_chunked(&model, &feat, args.chunk_max_frame)?;
    let decisions = process_probs(probs.as_slice().unwrap_or(&[]), &cfg);
    let segments = decision_to_segment(&decisions, Some(dur));
    let output = VadOutput {
        dur: ((dur * 1000.0).round()) / 1000.0,
        timestamps: segments,
        wav_path: args.wav.to_string_lossy().to_string(),
        sample_rate_in: audio.sample_rate_in,
        sample_rate_model: 16_000,
    };
    println!("{}", serde_json::to_string(&output)?);
    Ok(())
}

fn forward_chunked(
    model: &DetectModel,
    feat: &Array2<f32>,
    chunk_max_frame: usize,
) -> Result<Array1<f32>> {
    if feat.nrows() <= chunk_max_frame {
        return Ok(model.forward(feat));
    }
    // Chunks are non-overlapping and independent (each runs a full forward pass
    // over its own rows), so they are safe to run in parallel. Each chunk
    // produces its own owned probability vector; we collect them in order and
    // concatenate, so the final result is identical to the original serial loop.
    let total = feat.nrows();
    let start_offsets: Vec<usize> = (0..total).step_by(chunk_max_frame).collect();
    let per_chunk: Vec<Array1<f32>> = start_offsets
        .into_par_iter()
        .map(|start| -> Array1<f32> {
            let end = (start + chunk_max_frame).min(total);
            let chunk = feat.slice(s![start..end, ..]).to_owned();
            model.forward(&chunk)
        })
        .collect();
    let cap = per_chunk.iter().map(|c| c.len()).sum();
    let mut out = Vec::<f32>::with_capacity(cap);
    for c in per_chunk {
        out.extend(c.iter().copied());
    }
    Ok(Array1::from_vec(out))
}
