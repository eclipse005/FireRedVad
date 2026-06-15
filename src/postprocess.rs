use serde::Serialize;
use std::collections::VecDeque;

pub const FRAME_LENGTH_S: f32 = 0.025;
pub const FRAME_SHIFT_S: f32 = 0.010;

/// Frame shift in milliseconds (10 ms). Used to convert frame counts ↔ ms.
pub const FRAME_SHIFT_MS: u32 = 10;

#[derive(Debug, Clone)]
pub struct VadConfig {
    pub smooth_window_size: usize,
    pub speech_threshold: f32,
    pub min_speech_frame: usize,
    pub max_speech_frame: usize,
    pub min_silence_frame: usize,
    pub merge_silence_frame: usize,
    pub extend_speech_frame: usize,
    /// Dynamic silence-threshold schedule. Each entry is
    /// `(accumulated_speech_upper_ms, silence_threshold_ms)`: once the current
    /// speech segment has run for that long, silence is cut at the given
    /// threshold. Longer segments thus get tighter silence cuts.
    ///
    /// Empty (default) → a fixed `min_silence_frame` is used (legacy behavior,
    /// fully backward-compatible). Set it to [`FUNASR_OFFLINE_SCHEDULE`] via
    /// `--dynamic-vad` to enable dynamic thresholds.
    pub silence_schedule: Vec<(u32, u32)>,
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
            silence_schedule: Vec::new(),
        }
    }
}

/// FunASR FSMN-VAD offline dynamic silence-threshold schedule.
///
/// Each row: `(accumulated_speech_upper_ms, silence_threshold_ms)`. The longer
/// a speech segment has run, the shorter a silence gap is needed to cut it.
/// Sourced from FunASR's `fsmn_vad_streaming/dynamic_vad.py` `DEFAULT_SILENCE_SCHEDULE`.
pub const FUNASR_OFFLINE_SCHEDULE: &[(u32, u32)] = &[
    (5_000, 2_000),
    (10_000, 1_500),
    (15_000, 1_000),
    (30_000, 800),
    (45_000, 400),
    (u32::MAX, 100),
];

/// Resolve the silence-cut threshold (in frames) for a speech segment that has
/// already accumulated `accumulated_frames`.
///
/// With a non-empty `schedule`, look up the first row whose accumulated upper
/// bound covers the current duration and convert its ms threshold to frames.
/// With an empty schedule (dynamic-VAD disabled) just return `fallback`, so the
/// behavior is identical to the legacy fixed-threshold path.
fn silence_threshold_for(
    accumulated_frames: usize,
    schedule: &[(u32, u32)],
    fallback: usize,
) -> usize {
    if schedule.is_empty() {
        return fallback;
    }
    let ms = (accumulated_frames as u32).saturating_mul(FRAME_SHIFT_MS);
    schedule
        .iter()
        .find(|(upper_ms, _)| *upper_ms >= ms)
        .map(|(_, thresh_ms)| (*thresh_ms / FRAME_SHIFT_MS) as usize)
        .unwrap_or(fallback)
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
    if cfg.min_speech_frame == 0 && cfg.min_silence_frame == 0 && cfg.silence_schedule.is_empty() {
        return binary_preds.to_vec();
    }
    let mut decisions = vec![0i32; binary_preds.len()];
    let mut state = VadState::Silence;
    let mut speech_start = -1isize;
    let mut silence_start = -1isize;
    // First frame of the currently-confirmed speech segment (set when
    // PossibleSpeech promotes to Speech). Drives the dynamic silence threshold:
    // the longer the segment has run, the tighter the silence cut.
    let mut speech_seg_start: usize = 0;

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
                        speech_seg_start = speech_start.max(0) as usize;
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
                    // Dynamic threshold: based on how long the current speech
                    // segment has accumulated. Falls back to the fixed
                    // min_silence_frame when no schedule is configured.
                    let accumulated = t.saturating_sub(speech_seg_start);
                    let thresh = silence_threshold_for(
                        accumulated,
                        &cfg.silence_schedule,
                        cfg.min_silence_frame,
                    );
                    if t as isize - silence_start >= thresh as isize {
                        state = VadState::Silence;
                        speech_start = -1;
                        speech_seg_start = 0;
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

    #[test]
    fn silence_threshold_lookup_matches_funasr_offline_table() {
        let sched = FUNASR_OFFLINE_SCHEDULE;
        // (accumulated_frames, expected_threshold_frames) — each band of the
        // FunASR schedule, checked at its low end, the boundary, and just over.
        // ≤5s  → 2000ms = 200 frames
        assert_eq!(silence_threshold_for(0, sched, 999), 200); // 0ms
        assert_eq!(silence_threshold_for(500, sched, 999), 200); // 5000ms (band edge)
        // ≤10s → 1500ms = 150 frames
        assert_eq!(silence_threshold_for(501, sched, 999), 150); // 5010ms
        assert_eq!(silence_threshold_for(1000, sched, 999), 150); // 10000ms (band edge)
        // ≤15s → 1000ms = 100 frames
        assert_eq!(silence_threshold_for(1001, sched, 999), 100); // 10010ms
        assert_eq!(silence_threshold_for(1500, sched, 999), 100); // 15000ms (band edge)
        // ≤30s → 800ms = 80 frames
        assert_eq!(silence_threshold_for(1501, sched, 999), 80); // 15010ms
        assert_eq!(silence_threshold_for(3000, sched, 999), 80); // 30000ms (band edge)
        // ≤45s → 400ms = 40 frames
        assert_eq!(silence_threshold_for(3001, sched, 999), 40); // 30010ms
        assert_eq!(silence_threshold_for(4500, sched, 999), 40); // 45000ms (band edge)
        // >45s → 100ms = 10 frames
        assert_eq!(silence_threshold_for(4501, sched, 999), 10); // 45010ms
        assert_eq!(silence_threshold_for(7000, sched, 999), 10); // 70000ms
    }

    #[test]
    fn silence_threshold_empty_schedule_falls_back_to_fixed() {
        // Dynamic-VAD disabled (empty schedule) must return the fixed fallback,
        // so the legacy behavior is byte-for-byte preserved.
        let empty: &[(u32, u32)] = &[];
        assert_eq!(silence_threshold_for(0, empty, 20), 20);
        assert_eq!(silence_threshold_for(6000, empty, 20), 20);
    }

    #[test]
    fn dynamic_vad_cuts_long_segment_on_short_silence() {
        // A long speech segment (well past 60s of frames) followed by a short
        // silence gap that is shorter than the legacy threshold but long enough
        // for the tightened dynamic threshold (>60s → 100ms = 10 frames), then a
        // resumption of speech long enough to form a second segment.
        //
        // Legacy fixed min_silence_frame = 200 frames → the 50-frame gap does
        // NOT cut, so the whole run stays as one segment.
        // Dynamic schedule (>60s → 10 frames) → the 50-frame gap DOES cut,
        // yielding two segments.
        let speech_frames = 7_000; // 70s of speech (7000 frames * 10ms)
        let gap = 50; // 500ms silence
        let tail_speech = 50; // resume speech, long enough to confirm a 2nd segment
        let mut binary = vec![1i32; speech_frames];
        binary.extend(std::iter::repeat_n(0, gap));
        binary.extend(std::iter::repeat_n(1, tail_speech));

        // Legacy: no schedule, fixed 200-frame silence threshold.
        let cfg_legacy = VadConfig {
            min_speech_frame: 1,
            min_silence_frame: 200,
            ..VadConfig::default()
        };
        let d_legacy = smooth_preds_with_state_machine(&binary, &cfg_legacy);
        let cuts_legacy = decision_to_segment(&d_legacy, None).len();

        // Dynamic: FunASR offline schedule; at 70s the threshold is 10 frames.
        let cfg_dyn = VadConfig {
            min_speech_frame: 1,
            min_silence_frame: 200,
            silence_schedule: FUNASR_OFFLINE_SCHEDULE.to_vec(),
            ..VadConfig::default()
        };
        let d_dyn = smooth_preds_with_state_machine(&binary, &cfg_dyn);
        let cuts_dyn = decision_to_segment(&d_dyn, None).len();

        // Legacy keeps it as one segment (gap too short); dynamic splits it.
        assert_eq!(cuts_legacy, 1, "legacy should not cut on a 50-frame gap");
        assert_eq!(cuts_dyn, 2, "dynamic should cut at >60s threshold (10 frames)");
    }
}
