use anyhow::{Context, Result, bail};
use hound::WavReader;
use rubato::{
    Resampler, SincFixedIn, SincInterpolationParameters, SincInterpolationType, WindowFunction,
};
use std::path::Path;

pub struct AudioData {
    pub samples_16k_mono: Vec<f32>,
    pub sample_rate_in: u32,
}

pub fn read_wav_and_resample_to_16k(path: &Path) -> Result<AudioData> {
    let mut reader =
        WavReader::open(path).with_context(|| format!("failed to open wav: {}", path.display()))?;
    let spec = reader.spec();
    let sr_in = spec.sample_rate;
    let channels = usize::from(spec.channels);
    if channels == 0 {
        bail!("invalid wav with zero channels");
    }

    let samples_f32 = match (spec.sample_format, spec.bits_per_sample) {
        (hound::SampleFormat::Int, 16) => {
            let mut out = Vec::new();
            for s in reader.samples::<i16>() {
                let v = s.with_context(|| format!("bad PCM16 sample in {}", path.display()))?;
                out.push(v as f32);
            }
            out
        }
        (hound::SampleFormat::Float, 32) => {
            let mut out = Vec::new();
            for s in reader.samples::<f32>() {
                let v = s.with_context(|| format!("bad f32 sample in {}", path.display()))?;
                out.push(v * 32768.0);
            }
            out
        }
        _ => bail!(
            "unsupported wav format: {:?} {} bits",
            spec.sample_format,
            spec.bits_per_sample
        ),
    };

    let mono = if channels == 1 {
        samples_f32
    } else {
        let frames = samples_f32.len() / channels;
        let mut out = Vec::with_capacity(frames);
        for i in 0..frames {
            let mut acc = 0.0f32;
            for c in 0..channels {
                acc += samples_f32[i * channels + c];
            }
            out.push(acc / channels as f32);
        }
        out
    };

    let samples_16k_mono = if sr_in == 16_000 {
        mono
    } else {
        resample_to_16k(&mono, sr_in)?
    };

    Ok(AudioData {
        samples_16k_mono,
        sample_rate_in: sr_in,
    })
}

fn resample_to_16k(input: &[f32], sr_in: u32) -> Result<Vec<f32>> {
    if input.is_empty() {
        return Ok(Vec::new());
    }

    let ratio = 16_000.0f64 / sr_in as f64;
    let params = SincInterpolationParameters {
        sinc_len: 256,
        f_cutoff: 0.95,
        interpolation: SincInterpolationType::Linear,
        oversampling_factor: 160,
        window: WindowFunction::BlackmanHarris2,
    };
    let mut resampler = SincFixedIn::<f32>::new(ratio, 2.0, params, 1024, 1)
        .context("failed to create resampler")?;

    let mut output = Vec::<f32>::new();
    let mut pos = 0usize;
    let chunk = resampler.input_frames_max();
    while pos < input.len() {
        let end = (pos + chunk).min(input.len());
        let mut inbuf = vec![input[pos..end].to_vec()];
        if end - pos < chunk {
            inbuf[0].resize(chunk, 0.0);
        }
        let out = resampler
            .process(&inbuf, None)
            .context("resample process failed")?;
        output.extend_from_slice(&out[0]);
        pos = end;
    }
    Ok(output)
}
