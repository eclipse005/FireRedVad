use serde::Serialize;
use std::collections::VecDeque;

pub const FRAME_LENGTH_S: f32 = 0.025;
pub const FRAME_SHIFT_S: f32 = 0.010;

#[derive(Debug, Clone)]
pub struct VadConfig {
    pub smooth_window_size: usize,
    pub speech_threshold: f32,
    pub min_speech_frame: usize,
    pub max_speech_frame: usize,
    pub min_silence_frame: usize,
    pub merge_silence_frame: usize,
    pub extend_speech_frame: usize,
}

impl Default for VadConfig {
    fn default() -> Self {
        Self {
            smooth_window_size: 5,
            speech_threshold: 0.4,
            min_speech_frame: 20,
            max_speech_frame: 2000,
            min_silence_frame: 20,
            merge_silence_frame: 0,
            extend_speech_frame: 0,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum VadState {
    Silence,
    PossibleSpeech,
    Speech,
    PossibleSilence,
}

pub fn process_probs(raw_probs: &[f32], cfg: &VadConfig) -> Vec<i32> {
    if raw_probs.is_empty() {
        return Vec::new();
    }
    let smoothed = smooth_prob(raw_probs, cfg.smooth_window_size);
    let binary = apply_threshold(&smoothed, cfg.speech_threshold);
    let decisions = smooth_preds_with_state_machine(&binary, cfg);
    let fixed = fix_smooth_window_start(&decisions, cfg.smooth_window_size);
    let merged = merge_short_silence_segments(&fixed, cfg.merge_silence_frame);
    let extended = extend_speech_segments(&merged, cfg.extend_speech_frame);
    split_long_speech_segments(&extended, raw_probs, cfg.max_speech_frame)
}

fn smooth_prob(probs: &[f32], smooth_window_size: usize) -> Vec<f64> {
    if smooth_window_size <= 1 {
        return probs.iter().map(|&x| x as f64).collect();
    }
    let mut out = vec![0.0f64; probs.len()];
    let mut window = VecDeque::<f64>::new();
    let mut sum = 0.0f64;
    for (i, &p) in probs.iter().enumerate() {
        let v = p as f64;
        window.push_back(v);
        sum += v;
        if window.len() > smooth_window_size {
            sum -= window.pop_front().unwrap_or(0.0);
        }
        out[i] = sum / window.len() as f64;
    }
    out
}

fn apply_threshold(probs: &[f64], threshold: f32) -> Vec<i32> {
    let th = threshold as f64;
    probs.iter().map(|&p| (p >= th) as i32).collect()
}

fn smooth_preds_with_state_machine(binary_preds: &[i32], cfg: &VadConfig) -> Vec<i32> {
    if cfg.min_speech_frame == 0 && cfg.min_silence_frame == 0 {
        return binary_preds.to_vec();
    }
    let mut decisions = vec![0i32; binary_preds.len()];
    let mut state = VadState::Silence;
    let mut speech_start = -1isize;
    let mut silence_start = -1isize;

    for (t, &is_speech) in binary_preds.iter().enumerate() {
        match state {
            VadState::Silence => {
                if is_speech == 1 {
                    state = VadState::PossibleSpeech;
                    speech_start = t as isize;
                }
            }
            VadState::PossibleSpeech => {
                if is_speech == 1 {
                    if t as isize - speech_start >= cfg.min_speech_frame as isize {
                        state = VadState::Speech;
                        for v in decisions
                            .iter_mut()
                            .take(t)
                            .skip(speech_start.max(0) as usize)
                        {
                            *v = 1;
                        }
                    }
                } else {
                    state = VadState::Silence;
                    speech_start = -1;
                }
            }
            VadState::Speech => {
                if is_speech == 0 {
                    state = VadState::PossibleSilence;
                    silence_start = t as isize;
                }
            }
            VadState::PossibleSilence => {
                if is_speech == 0 {
                    if t as isize - silence_start >= cfg.min_silence_frame as isize {
                        state = VadState::Silence;
                        speech_start = -1;
                    }
                } else {
                    state = VadState::Speech;
                    silence_start = -1;
                }
            }
        }

        decisions[t] = if state == VadState::Speech || state == VadState::PossibleSilence {
            1
        } else {
            0
        };
    }
    decisions
}

fn fix_smooth_window_start(decisions: &[i32], smooth_window_size: usize) -> Vec<i32> {
    let mut out = decisions.to_vec();
    for t in 0..decisions.len() {
        if t > 0 && decisions[t - 1] == 0 && decisions[t] == 1 {
            let start = t.saturating_sub(smooth_window_size);
            for v in out.iter_mut().take(t).skip(start) {
                *v = 1;
            }
        }
    }
    out
}

fn merge_short_silence_segments(decisions: &[i32], merge_silence_frame: usize) -> Vec<i32> {
    if merge_silence_frame == 0 {
        return decisions.to_vec();
    }
    let mut out = decisions.to_vec();
    let mut silence_start: Option<usize> = None;
    for t in 0..decisions.len() {
        if t > 0 && decisions[t - 1] == 1 && decisions[t] == 0 && silence_start.is_none() {
            silence_start = Some(t);
        } else if t > 0 && decisions[t - 1] == 0 && decisions[t] == 1 {
            if let Some(s) = silence_start {
                let silence = t - s;
                if silence < merge_silence_frame {
                    for v in out.iter_mut().take(t).skip(s) {
                        *v = 1;
                    }
                }
            }
            silence_start = None;
        }
    }
    out
}

fn extend_speech_segments(decisions: &[i32], extend_speech_frame: usize) -> Vec<i32> {
    if extend_speech_frame == 0 {
        return decisions.to_vec();
    }
    let mut out = vec![0i32; decisions.len()];
    for (t, &v) in decisions.iter().enumerate() {
        if v == 1 {
            let start = t.saturating_sub(extend_speech_frame);
            let end = (t + extend_speech_frame + 1).min(decisions.len());
            for x in out.iter_mut().take(end).skip(start) {
                *x = 1;
            }
        }
    }
    out
}

fn split_long_speech_segments(
    decisions: &[i32],
    probs: &[f32],
    max_speech_frame: usize,
) -> Vec<i32> {
    let mut out = decisions.to_vec();
    for (start_s, end_s) in decision_to_segment(decisions, None) {
        let start = (start_s / FRAME_SHIFT_S) as usize;
        let end = (end_s / FRAME_SHIFT_S) as usize;
        // 防止浮点数精度问题导致数组越界
        let start = start.min(probs.len());
        let end = end.min(probs.len());
        if end > start && end - start > max_speech_frame {
            let points = find_split_points(&probs[start..end], max_speech_frame);
            for p in points {
                let idx = start + p;
                if idx < out.len() {
                    out[idx] = 0;
                }
            }
        }
    }
    out
}

fn find_split_points(probs: &[f32], max_speech_frame: usize) -> Vec<usize> {
    let mut split_points = Vec::new();
    let length = probs.len();
    let mut start = 0usize;
    while start < length {
        if length - start <= max_speech_frame {
            break;
        }
        let ws = start + max_speech_frame / 2;
        let we = (start + max_speech_frame).min(length);
        let mut min_idx = ws;
        let mut min_v = f32::INFINITY;
        for (i, &v) in probs.iter().enumerate().take(we).skip(ws) {
            if v < min_v {
                min_v = v;
                min_idx = i;
            }
        }
        split_points.push(min_idx);
        start = min_idx + 1;
    }
    split_points
}

pub fn decision_to_segment(decisions: &[i32], wav_dur: Option<f32>) -> Vec<(f32, f32)> {
    let mut segments = Vec::new();
    let mut speech_start: Option<usize> = None;
    for (t, &decision) in decisions.iter().enumerate() {
        if decision == 1 && speech_start.is_none() {
            speech_start = Some(t);
        } else if decision == 0 && speech_start.is_some() {
            let s = speech_start.take().unwrap_or(0);
            segments.push((s as f32 * FRAME_SHIFT_S, t as f32 * FRAME_SHIFT_S));
        }
    }
    if let Some(s) = speech_start {
        let mut end_time = decisions.len() as f32 * FRAME_SHIFT_S + FRAME_LENGTH_S;
        if let Some(dur) = wav_dur {
            end_time = end_time.min(dur);
        }
        segments.push((s as f32 * FRAME_SHIFT_S, end_time));
    }
    segments
        .into_iter()
        .map(|(s, e)| {
            (
                ((s * 1000.0).round() / 1000.0),
                ((e * 1000.0).round() / 1000.0),
            )
        })
        .collect()
}

#[derive(Serialize)]
pub struct VadOutput {
    pub dur: f32,
    pub timestamps: Vec<(f32, f32)>,
    pub wav_path: String,
    pub sample_rate_in: u32,
    pub sample_rate_model: u32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn state_machine_promotes_stable_speech() {
        let cfg = VadConfig {
            min_speech_frame: 3,
            min_silence_frame: 2,
            ..VadConfig::default()
        };
        let b = vec![0, 1, 1, 1, 1, 0, 0, 0];
        let d = smooth_preds_with_state_machine(&b, &cfg);
        assert_eq!(&d[..6], &[0, 1, 1, 1, 1, 1]);
    }

    #[test]
    fn split_long_segments_clamps_end_index_to_probs_len() {
        // Regression case: decision_to_segment() may produce end index > probs.len()
        // for trailing speech because it appends FRAME_LENGTH_S before converting to frame index.
        let probs = vec![0.9f32; 11_304];
        let decisions = vec![1i32; probs.len()];
        let out = split_long_speech_segments(&decisions, &probs, 2_000);
        assert_eq!(out.len(), decisions.len());
    }
}
