#!/usr/bin/env python
import argparse
import json
from pathlib import Path

import kaldiio
import numpy as np
import torch

from fireredasr2s.fireredvad.vad import FireRedVad, FireRedVadConfig
from fireredasr2s.fireredvad.core.audio_feat import AudioFeat, CMVN


def read_meta(args_obj):
    return {
        "idim": int(args_obj.idim),
        "r": int(args_obj.R),
        "m": int(args_obj.M),
        "h": int(args_obj.H),
        "p": int(args_obj.P),
        "n1": int(args_obj.N1),
        "s1": int(args_obj.S1),
        "n2": int(args_obj.N2),
        "s2": int(args_obj.S2),
        "odim": int(args_obj.odim),
        "dropout": float(args_obj.dropout),
    }


def export_cmvn(cmvn_path: Path, out_path: Path):
    cmvn = CMVN(str(cmvn_path))
    payload = {
        "dim": int(cmvn.dim),
        "means": [float(x) for x in cmvn.means.tolist()],
        "inverse_std_variances": [float(x) for x in cmvn.inverse_std_variances.tolist()],
    }
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def export_weights(state_dict, out_npz: Path):
    arrays = {}
    for k, v in state_dict.items():
        arrays[k] = v.detach().cpu().numpy().astype(np.float32)
    np.savez(out_npz, **arrays)


def export_golden(model_dir: Path, wav_path: Path, out_json: Path):
    cfg = FireRedVadConfig(use_gpu=False)
    vad = FireRedVad.from_pretrained(str(model_dir), cfg)
    result, probs = vad.detect(str(wav_path), do_postprocess=True)

    af = AudioFeat(str(model_dir / "cmvn.ark"))
    feat, dur = af.extract(str(wav_path))
    payload = {
        "wav_path": str(wav_path),
        "dur": float(result["dur"]),
        "timestamps": [[float(s), float(e)] for s, e in result["timestamps"]],
        "feat_shape": [int(x) for x in feat.shape],
        "feat_row0_head": [float(x) for x in feat[0, :8].tolist()] if feat.shape[0] > 0 else [],
        "probs_len": int(probs.numel()),
        "probs_head": [float(x) for x in probs[:20].tolist()],
    }
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--wav", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    pth = args.model_dir / "model.pth.tar"
    cmvn = args.model_dir / "cmvn.ark"
    pkg = torch.load(str(pth), map_location="cpu", weights_only=False)
    state_dict = pkg["model_state_dict"]
    meta = read_meta(pkg["args"])

    (args.out_dir / "model_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    export_cmvn(cmvn, args.out_dir / "cmvn.json")
    export_weights(state_dict, args.out_dir / "weights.npz")
    export_golden(args.model_dir, args.wav, args.out_dir / "golden_hello_zh.json")

    print(f"Exported to {args.out_dir}")


if __name__ == "__main__":
    main()
