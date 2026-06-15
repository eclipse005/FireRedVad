#!/usr/bin/env python
import argparse
import json
from pathlib import Path

from fireredasr2s.fireredvad.vad import FireRedVad, FireRedVadConfig
from fireredasr2s.fireredvad.core.audio_feat import AudioFeat


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--wav", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--speech-threshold", type=float, default=0.4)
    parser.add_argument("--smooth-window-size", type=int, default=5)
    parser.add_argument("--min-speech-frame", type=int, default=20)
    parser.add_argument("--max-speech-frame", type=int, default=2000)
    parser.add_argument("--min-silence-frame", type=int, default=20)
    parser.add_argument("--merge-silence-frame", type=int, default=0)
    parser.add_argument("--extend-speech-frame", type=int, default=0)
    parser.add_argument("--chunk-max-frame", type=int, default=30000)
    parser.add_argument("--light", action="store_true")
    args = parser.parse_args()

    cfg = FireRedVadConfig(
        use_gpu=False,
        smooth_window_size=args.smooth_window_size,
        speech_threshold=args.speech_threshold,
        min_speech_frame=args.min_speech_frame,
        max_speech_frame=args.max_speech_frame,
        min_silence_frame=args.min_silence_frame,
        merge_silence_frame=args.merge_silence_frame,
        extend_speech_frame=args.extend_speech_frame,
        chunk_max_frame=args.chunk_max_frame,
    )
    vad = FireRedVad.from_pretrained(str(args.model_dir), cfg)
    result, probs = vad.detect(str(args.wav), do_postprocess=True)

    af = AudioFeat(str(args.model_dir / "cmvn.ark"))
    feat_cmvn, dur = af.extract(str(args.wav))
    if args.light:
        feat_raw_flat = []
        feat_cmvn_flat = []
    else:
        af_raw = AudioFeat("")
        feat_raw, _ = af_raw.extract(str(args.wav))
        feat_raw_flat = [float(x) for x in feat_raw.reshape(-1).tolist()]
        feat_cmvn_flat = [float(x) for x in feat_cmvn.reshape(-1).tolist()]
    raw_probs = probs.tolist()
    smoothed_probs = vad.vad_postprocessor._smooth_prob(raw_probs).tolist()
    binary_preds = vad.vad_postprocessor._apply_threshold(smoothed_probs)
    state_decisions = vad.vad_postprocessor._smooth_preds_with_state_machine(binary_preds)
    fixed_decisions = vad.vad_postprocessor._fix_smooth_window_start(state_decisions)
    merged_decisions = vad.vad_postprocessor._merge_short_silence_segments(fixed_decisions)
    extended_decisions = vad.vad_postprocessor._extend_speech_segments(merged_decisions)
    decisions = vad.vad_postprocessor._split_long_speech_segments(extended_decisions, raw_probs)
    timestamps = vad.vad_postprocessor.decision_to_segment(decisions, dur)

    payload = {
        "wav_path": str(args.wav),
        "sample_rate_model": 16000,
        "dur": float(result["dur"]),
        "feat_shape": [int(feat_cmvn.shape[0]), int(feat_cmvn.shape[1])],
        "feat_raw": feat_raw_flat,
        "feat_cmvn": feat_cmvn_flat,
        "probs": [float(x) for x in raw_probs],
        "smoothed_probs": [float(x) for x in smoothed_probs],
        "binary_preds": [int(x) for x in binary_preds],
        "state_decisions": [int(x) for x in state_decisions],
        "fixed_decisions": [int(x) for x in fixed_decisions],
        "merged_decisions": [int(x) for x in merged_decisions],
        "extended_decisions": [int(x) for x in extended_decisions],
        "final_decisions": [int(x) for x in decisions],
        "timestamps": [[float(s), float(e)] for s, e in timestamps],
    }
    args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
