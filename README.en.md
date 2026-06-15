English | [简体中文](README.md)

# FireRedVAD

A native-Rust port of the industrial-grade VAD. Weights are compiled into the binary — single file, zero runtime dependencies. Can also be embedded as a Rust crate.

## Performance

On an Intel Core Ultra 7 265K (20 cores), 7835s of audio runs in **3.96s** — **RTFx ≈ 1978×**. 10–100× faster than the upstream Python (PyTorch), no GPU needed.

> Methodology: median of 5 back-to-back runs. See `tmp/bench_rtfx.ps1`.

## Accuracy

Same origin as upstream ([FireRedTeam/FireRedVAD](https://huggingface.co/FireRedTeam/FireRedVAD)), test set FLEURS-VAD-102:

| Metric | FireRedVAD | Silero | TEN | FunASR | WebRTC |
|:--|--:|--:|--:|--:|--:|
| AUC-ROC ↑ | **99.60** | 97.99 | 97.81 | - | - |
| F1 ↑ | **97.57** | 95.95 | 95.19 | 90.91 | 52.30 |
| False Alarm ↓ | **2.69** | 9.41 | 15.47 | 44.03 | 2.83 |

## Features

- **Single-file binary** — weights compiled in, no model files or runtimes
- **CPU-optimized** — `gemm` probes AVX-512/AVX2/AVX/SSE/NEON at runtime, picks the fastest kernel
- **Parallel inference** — `rayon` chunked parallelism, scales across cores
- **Format-tolerant** — auto down-mix, auto resample to 16kHz, PCM16/float32
- **Dynamic VAD** — optional dynamic silence threshold, keeps utterances whole

## Quick Start

**Prebuilt binary** (from [Releases](../../releases)):
```powershell
.\fireredvad.exe D:\path\to\audio.wav
```

**Build from source**:
```powershell
cargo build --release
```

## Usage

```powershell
.\fireredvad.exe audio.wav > result.json
```

Output:
```json
{
  "dur": 2.32,
  "timestamps": [[0.44, 1.82]],
  "wav_path": "audio.wav",
  "sample_rate_in": 48000,
  "sample_rate_model": 16000
}
```

`timestamps` is a list of `[start_sec, end_sec]` speech segments.

## Use as a Rust Library

```toml
[dependencies]
fireredvad = "0.1"
```

```rust
use fireredvad::{Vad, VadConfig};

fn main() -> anyhow::Result<()> {
    let vad = Vad::new()?;
    let out = vad.detect_wav("audio.wav", &VadConfig::default())?;
    for (start, end) in &out.timestamps {
        println!("{start:.3}s -> {end:.3}s");
    }
    Ok(())
}
```

`VadConfig` fields map one-to-one to the CLI parameters below. For custom audio sources, use `detect_pcm` with raw 16kHz mono samples.

## CLI Parameters

| Parameter | Default | Description |
|:--|:--|:--|
| `<wav>` | *(required)* | Input WAV path |
| `--speech-threshold` | `0.4` | Speech probability threshold |
| `--smooth-window-size` | `5` | Smoothing window (frames) |
| `--min-speech-frame` | `20` | Min consecutive speech frames |
| `--max-speech-frame` | `2000` | Max segment length, force-split if exceeded |
| `--min-silence-frame` | `20` | Min consecutive silence frames |
| `--merge-silence-frame` | `0` | Merge silence gaps shorter than this |
| `--extend-speech-frame` | `0` | Bidirectional extension (frames) |
| `--chunk-max-frame` | `30000` | Max frames per parallel chunk |
| `--dynamic-vad` | `off` | Enable dynamic silence threshold |

## Dynamic VAD

By default `--min-silence-frame` is a **fixed** 200ms: no matter how long the utterance, a 200ms pause cuts it. This shreds complete sentences (every breath or thinking pause triggers a cut).

`--dynamic-vad` tightens the threshold as the **current segment accumulates** — the longer the segment, the shorter a pause that suffices to cut:

| Segment so far | Silence must exceed | to cut |
|:--|--:|--:|
| ≤ 5 s | 2000 ms | 2s |
| ≤ 10 s | 1500 ms | 1.5s |
| ≤ 15 s | 1000 ms | 1s |
| ≤ 30 s | 800 ms | |
| ≤ 45 s | 400 ms | |
| > 45 s | 100 ms | cut on any pause |

```powershell
.\fireredvad.exe audio.wav --dynamic-vad
```

> - Mutually exclusive with `--min-silence-frame` (fixed value ignored when on)
> - `--max-speech-frame` still applies as a hard ceiling
> - Schedule sourced from [FunASR](https://github.com/modelscope/FunASR) `fsmn_vad_streaming/dynamic_vad.py`

## How It Works

```
WAV → down-mix mono → resample 16kHz → 80-dim fbank → CMVN → DFSMN encoder → sigmoid → state-machine postprocess → segments
```

| Module | Role |
|:--|:--|
| `audio.rs` | decode PCM16/float32 WAV, down-mix, resample |
| `fbank.rs` | 80-dim log-Mel filterbank + Povey window |
| `cmvn.rs` | CMVN mean/variance normalization |
| `model.rs` | Deep FSMN encoder (`gemm` matmul) |
| `postprocess.rs` | smoothing, thresholding, state machine, splitting |

## Weights

The `model/` weights originate from [FireRedTeam/FireRedASR2S](https://github.com/FireRedTeam/FireRedASR2S), redistributed under Apache-2.0.

## License

[Apache-2.0](LICENSE)
