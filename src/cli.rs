use crate::postprocess::{FUNASR_OFFLINE_SCHEDULE, VadConfig};
use crate::vad::Vad;
use anyhow::Result;
use clap::Parser;
use std::path::PathBuf;

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
    /// Enable dynamic silence threshold (FunASR offline schedule: longer
    /// speech segments get tighter silence cuts). Overrides min_silence_frame.
    #[arg(long, default_value_t = false)]
    dynamic_vad: bool,
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
        silence_schedule: if args.dynamic_vad {
            FUNASR_OFFLINE_SCHEDULE.to_vec()
        } else {
            Vec::new()
        },
    };

    let vad = Vad::new()?;
    let output = vad.detect_wav_chunked(&args.wav, &cfg, args.chunk_max_frame)?;
    println!("{}", serde_json::to_string(&output)?);
    Ok(())
}
