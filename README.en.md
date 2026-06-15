English | [简体中文](README.md)

# FireRedVAD

> **A native-Rust port of the industrial-grade VAD** — weights, CMVN stats, and
> metadata are compiled in via `include_bytes!`. The single-file
> `fireredvad.exe` runs with zero runtime dependencies. It can also be embedded
> into your own pipeline as a Rust crate via the high-level `Vad` API.

---

## Headline Number

| Platform | Audio | Median wall-clock | **RTFx** |
|:--|:--|--:|--:|
| Intel Core Ultra 7 265K (20 cores) | 7,835 s · 16 kHz · mono · PCM16 | 3.96 s | **≈ 1,978×** |
| same setup · best | 7,835 s · 16 kHz · mono · PCM16 | 3.90 s | **≈ 2,010×** |

RTFx is `audio_duration / inference_time`. On the same audio, this Rust build
is typically 10×–100× faster than the upstream Python (PyTorch) script, and
removes the "GPU required" barrier that PyTorch inference imposes. Methodology:
5 back-to-back runs, median reported; warm-up discarded. See
`tmp/bench_rtfx.ps1` for the reproducible script.

---

## What This Is

- **Not** a wrapper around the upstream Python package. It is a ground-up Rust
  reimplementation of the same VAD pipeline (decode → fbank → CMVN → DFSMN
  encoder → sigmoid → state-machine postprocess), consuming the
  [FireRedTeam/FireRedVAD](https://huggingface.co/FireRedTeam/FireRedVAD)
  weights directly. Detection quality therefore matches the published numbers
  exactly (table below).
- The inference backend uses [`gemm`](https://crates.io/crates/gemm), which does
  runtime CPU-feature detection (AVX-512 / AVX2 / AVX / SSE on x86, NEON on ARM)
  and picks the fastest micro-kernel. `Cargo.toml` deliberately does not pin
  `target-cpu`, so one binary runs optimally on anything from low-end machines
  to servers.
- Long audio is split into 30,000-frame chunks and forwarded in parallel with
  [`rayon`](https://crates.io/crates/rayon), scaling across all available cores.

---

## Detection Quality

Same origin as upstream ([FireRedTeam/FireRedVAD](https://huggingface.co/FireRedTeam/FireRedVAD) /
[FireRedTeam/FireRedASR2S](https://github.com/FireRedTeam/FireRedASR2S)),
test set FLEURS-VAD-102:

| Metric \ Model | FireRedVAD | Silero-VAD | TEN-VAD | FunASR-VAD | WebRTC-VAD |
|:--|--:|--:|--:|--:|--:|
| AUC-ROC ↑        | **99.60** | 97.99 | 97.81 | -     | -     |
| F1 score ↑       | **97.57** | 95.95 | 95.19 | 90.91 | 52.30 |
| False Alarm Rate ↓ | **2.69** | 9.41  | 15.47 | 44.03 | 2.83  |
| Miss Rate ↓      | 3.62      | 3.95  | 2.95  | 0.42  | 64.15 |

> FunASR-VAD's low Miss Rate comes at the cost of a 44% false-alarm rate — it
> over-predicts speech. Silero / TEN / FunASR / WebRTC numbers are cited from
> the upstream technical reports.

---

## Features

- **Single-file binary** — model weights are compiled in; distribution is just
  one `.exe`, no model files, env vars, or runtimes needed.
- **CPU-optimized inference** — `gemm` matrix multiplies probe AVX-512/AVX2/AVX/SSE
  (x86) or NEON (ARM) at runtime and pick the best micro-kernel. `Cargo.toml`
  intentionally does not pin `target-cpu`, so the same binary runs well on both
  old and new CPUs.
- **Parallel chunked inference** — long audio is split into 30,000-frame chunks
  and processed in parallel, scaling linearly to all available cores.
- **Format-tolerant** — multi-channel WAV is auto-down-mixed to mono; any sample
  rate is auto-resampled to 16 kHz; PCM16 and float32 are both accepted.
- **Dynamic VAD** — optional dynamic silence threshold (`--dynamic-vad`): the
  longer a speech segment runs, the tighter its silence cut, preserving long
  segments and reducing boundary-cut losses (from FunASR's offline strategy).

---

## Quick Start

### A. Prebuilt binary

Download the latest build from the [Releases](../../releases) page:

```powershell
.\fireredvad.exe D:\path\to\audio.wav
```

### B. Build from source

```powershell
cargo build --release
# artifact at target\release\fireredvad.exe
```

---

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

`timestamps` is a list of `[start_sec, end_sec]` pairs marking all speech segments.

---

## Use as a Rust Library

Besides the CLI binary, `fireredvad` is published as a Rust crate. The
high-level `Vad` API wraps model loading, the audio front-end, inference, and
post-processing — VAD in three lines from your own program.

```toml
# Cargo.toml
[dependencies]
fireredvad = "0.1"
```

```rust
use fireredvad::{Vad, VadConfig};

fn main() -> anyhow::Result<()> {
    let vad = Vad::new()?;                              // built-in pretrained model, zero config
    let out = vad.detect_wav("audio.wav", &VadConfig::default())?;
    for (start, end) in &out.timestamps {
        println!("{start:.3}s -> {end:.3}s");
    }
    Ok(())
}
```

The fields of `VadConfig` map one-to-one to the **CLI Parameters** table below —
the same threshold / smoothing / min-frames semantics are shared between the
library and the CLI. For custom audio sources (streaming, in-memory buffers),
use `detect_pcm` with raw 16 kHz mono samples instead.

`VadConfig.silence_schedule` is also settable programmatically: pass
`FUNASR_OFFLINE_SCHEDULE.to_vec()` (or your own `(upper_ms, threshold_ms)` table)
to enable dynamic silence cuts from library code — the same thing the
`--dynamic-vad` CLI flag does.

---

## CLI Parameters

| Parameter | Default | Description |
|:--|:--|:--|
| `<wav_path>` | *(required)* | Input WAV path |
| `--speech-threshold`    | `0.4`   | Speech probability threshold |
| `--smooth-window-size`  | `5`     | Smoothing window (frames) |
| `--min-speech-frame`    | `20`    | Min consecutive speech frames |
| `--max-speech-frame`    | `2000`  | Max segment length (force-split if exceeded) |
| `--min-silence-frame`   | `20`    | Min consecutive silence frames |
| `--merge-silence-frame` | `0`     | Merge silence gaps shorter than this |
| `--extend-speech-frame` | `0`     | Bidirectional extension (frames) |
| `--chunk-max-frame`     | `30000` | Max frames per parallel chunk |
| `--dynamic-vad`         | `off`   | Enable dynamic silence threshold (FunASR offline strategy) |

---

## Dynamic VAD

By default `--min-silence-frame` (default 20 frames = 200 ms) is a **fixed**
silence-cut threshold: regardless of how long the current utterance has run, a
silence gap must last that long to trigger a cut. On long audio this leaves
over-long segments (breathing/thinking pauses never reach the cut threshold).

With `--dynamic-vad`, the silence-cut threshold **tightens as the current
speech segment runs longer** — the longer the segment, the shorter a pause that
suffices to cut. This mirrors the offline dynamic-threshold strategy of
[FunASR](https://github.com/modelscope/FunASR)'s FSMN-VAD, useful for ASR data
slicing where long segments should be broken up sensibly.

```powershell
.\fireredvad.exe D:\path\to\audio.wav --dynamic-vad
```

**Built-in schedule** (FunASR offline):

| Segment accumulated | Silence threshold |
|:--|--:|
| ≤ 5 s          | 2000 ms |
| 5–10 s         | 2000 ms |
| 10–15 s        | 1000 ms |
| 15–20 s        | 1000 ms |
| 20–30 s        |  800 ms |
| 30–45 s        |  600 ms |
| 45–60 s        |  300 ms |
| > 60 s         |  100 ms |

> `--dynamic-vad` is mutually exclusive with `--min-silence-frame` (the fixed
> value is ignored once dynamic is on). `--max-speech-frame` still applies as a
> hard ceiling: a segment with no silence gap at all is still force-cut by it.

---

## How It Works

```
WAV ─► down-mix to mono ─► resample to 16 kHz ─► 80-dim fbank ─► CMVN
   ─► DFSMN encoder ─► sigmoid speech probability ─► state-machine postprocess ─► segments
```

1. **Audio front-end** (`src/audio.rs`) — read PCM16 / float32 WAV, down-mix to
   mono, resample to 16 kHz.
2. **Feature extraction** (`src/fbank.rs`) — 80-dim log-Mel filterbank + Povey window.
3. **Normalization** (`src/cmvn.rs`) — CMVN mean/variance normalization.
4. **Acoustic model** (`src/model.rs`) — Deep FSMN encoder; matrix multiplies go
   through `gemm`, dispatched by CPU features at runtime.
5. **Post-processing** (`src/postprocess.rs`) — smoothing, thresholding,
   min-speech/silence state machine, gap merging, segment splitting.

---

## Model Weights

The pretrained weights under `model/` (`weights.npz` / `cmvn.json` /
`model_meta.json`) originate from the
[FireRedTeam/FireRedASR2S](https://github.com/FireRedTeam/FireRedASR2S) VAD
checkpoint, redistributed under Apache-2.0.

---

## Acknowledgements

- [FireRedTeam/FireRedVAD](https://huggingface.co/FireRedTeam/FireRedVAD) — the
  upstream VAD model and original benchmark.
- [FireRedTeam/FireRedASR2S](https://github.com/FireRedTeam/FireRedASR2S) — the
  VAD module source within the upstream ASR + VAD + LID + Punc system.

---

## License

[Apache-2.0](LICENSE). The bundled model weights are redistributed from
FireRedASR2S under the same license.
