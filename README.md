# FireRedVAD

A self-contained **native Rust CLI for non-streaming Voice Activity Detection
(VAD)** on Windows. The model weights, CMVN stats, and metadata are embedded
directly into the binary at build time via `include_bytes!`, so the resulting
`fireredvad.exe` needs no model files, runtime, or external dependencies — just
download and run.

This is an independent Rust reimplementation of the VAD module from
[FireRedTeam/FireRedASR2S](https://github.com/FireRedTeam/FireRedASR2S),
producing detection results consistent with the upstream Python implementation.

## Benchmark

The underlying model is unchanged from upstream, so detection quality matches
the published numbers (source: [FireRedTeam/FireRedASR2S](https://github.com/FireRedTeam/FireRedASR2S)):

| Metric \ Model | FireRedVAD | Silero-VAD | TEN-VAD | FunASR-VAD | WebRTC-VAD |
|:--|--:|--:|--:|--:|--:|
| AUC-ROC ↑        | **99.60** | 97.99 | 97.81 | -     | -     |
| F1 score ↑       | **97.57** | 95.95 | 95.19 | 90.91 | 52.30 |
| False Alarm Rate ↓ | **2.69** | 9.41  | 15.47 | 44.03 | 2.83  |
| Miss Rate ↓      | 3.62      | 3.95  | 2.95  | 0.42  | 64.15 |

## Features

- **Self-contained binary** — model weights are compiled in; one executable, no
  model files to ship or load.
- **CPU-optimized inference** — matrix multiplications go through the [`gemm`](https://crates.io/crates/gemm)
  crate, which does runtime CPU-feature detection (AVX-512 / AVX2 / AVX / SSE on
  x86, NEON on ARM) and selects the best micro-kernel. No target-cpu is pinned,
  so the binary runs on any baseline machine.
- **Parallel chunked inference** — long audio is split into non-overlapping
  chunks processed in parallel with [`rayon`](https://crates.io/crates/rayon).
- **Format-tolerant input** — multi-channel WAV is auto-mixed to mono; any
  sample rate is resampled to 16 kHz.

## Quick Start

### Option A — Prebuilt binary

Download the latest release from the [Releases page](../../releases) and run:

```powershell
.\fireredvad.exe D:\path\to\audio.wav
```

### Option B — Build from source

```powershell
cargo build --release
# binary at target\release\fireredvad.exe
```

## Usage

```powershell
.\fireredvad.exe D:\path\to\audio.wav
```

Write the result to a file:

```powershell
.\fireredvad.exe D:\path\to\audio.wav > vad_result.json
```

Example output:

```json
{
  "dur": 2.32,
  "timestamps": [[0.44, 1.82]],
  "wav_path": "D:\\path\\to\\audio.wav",
  "sample_rate_in": 48000,
  "sample_rate_model": 16000
}
```

`timestamps` is a list of `[start_sec, end_sec]` pairs marking speech segments.

### CLI Parameters

| Parameter | Default | Description |
|:--|:--|:--|
| `<wav_path>` | *(required)* | Input WAV path |
| `--speech-threshold`    | `0.4`   | Speech probability threshold |
| `--smooth-window-size`  | `5`     | Sliding-window smoothing size (frames) |
| `--min-speech-frame`    | `20`    | Min consecutive speech frames to confirm speech |
| `--max-speech-frame`    | `2000`  | Max speech segment length before forced split |
| `--min-silence-frame`   | `20`    | Min consecutive silence frames to confirm silence |
| `--merge-silence-frame` | `0`     | Merge silence gaps shorter than this |
| `--extend-speech-frame` | `0`     | Extend each speech frame bidirectionally by this many frames |
| `--chunk-max-frame`     | `30000` | Max frames per parallel inference chunk |

## How It Works

```
WAV ─► mix to mono ─► resample to 16 kHz ─► fbank (80-dim) ─► CMVN
   ─► DFSMN encoder ─► sigmoid speech probs ─► state-machine postprocess ─► segments
```

1. **Audio frontend** (`src/audio.rs`) — reads PCM16 / float32 WAV, downmixes to
   mono, and resamples to 16 kHz.
2. **Feature extraction** (`src/fbank.rs`) — 80-dim log-Mel filterbank with a
   Povey window.
3. **Normalization** (`src/cmvn.rs`) — applies CMVN mean/variance normalization.
4. **Acoustic model** (`src/model.rs`) — a Deep FSMN (DFSMN) encoder; matrix
   products are dispatched to `gemm` for runtime CPU-specific acceleration.
5. **Post-processing** (`src/postprocess.rs`) — smoothing, thresholding, a state
   machine for min speech/silence durations, merging, and segment splitting.

## Model Weights

The prebuilt weights in `model/` (`weights.npz`, `cmvn.json`, `model_meta.json`)
originate from the [FireRedTeam/FireRedASR2S](https://github.com/FireRedTeam/FireRedASR2S)
VAD checkpoint and are redistributed under the same Apache-2.0 license.

## Acknowledgements

- [FireRedTeam/FireRedASR2S](https://github.com/FireRedTeam/FireRedASR2S) — the
  original VAD model and benchmark this project reimplements in Rust.

## License

[Apache-2.0](LICENSE). The bundled model weights are redistributed from
FireRedASR2S under the same license.
