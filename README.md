# FireRedVAD

> **原生 Rust 实现的工业级 VAD** — 模型权重、CMVN 统计量与 metadata 全部以
> `include_bytes!` 编译进二进制；下载 `fireredvad.exe` 直接跑，零运行时依赖。
> 也可作为 Rust crate（`Vad` 高层 API）嵌入到你自己的音频管线中。
>
> **A native-Rust port of the industrial-grade VAD** — weights, CMVN stats, and
> metadata are compiled in via `include_bytes!`. The single-file
> `fireredvad.exe` runs with zero runtime dependencies. It can also be embedded
> into your own pipeline as a Rust crate via the high-level `Vad` API.

---

## 性能亮点 · Headline Number

| 平台 / Platform | 音频 / Audio | 中位耗时 / Median wall-clock | **RTFx** |
|:--|:--|--:|--:|
| Intel Core Ultra 7 265K（20 核） | 7,835 s · 16 kHz · mono · PCM16 | 3.96 s | **≈ 1,978×** |
| 同上 · 最快一次 / same setup · best | 7,835 s · 16 kHz · mono · PCM16 | 3.90 s | **≈ 2,010×** |

RTFx = 音频时长 / 推理耗时。即 **每一秒 CPU 时间可处理近两千秒音频**——属于
"算力远富余"的级别，可以把 VAD 整段串进更大的实时管线而无后顾之忧。同一段音频，
本仓库的 Rust 实现相比原版 Python 推理脚本（PyTorch + 解释器开销）通常快
**一到两个数量级**；与 PyTorch GPU 推理相比，则去掉了"必须配卡"的硬性门槛。

测试方法：5 次连续推理取中位（去除冷启动），详细脚本见 `tmp/bench_rtfx.ps1`。
RTFx is `audio_duration / inference_time`. On the same audio, this Rust build
is typically 10×–100× faster than the upstream Python (PyTorch) script, and
removes the "GPU required" barrier that PyTorch inference imposes. Methodology:
5 back-to-back runs, median reported; warm-up discarded. See
`tmp/bench_rtfx.ps1` for the reproducible script.

---

## 项目定位 · What This Is

- **不是**上游 Python 仓库的镜像或绑定。本仓库基于
  [FireRedTeam/FireRedVAD](https://huggingface.co/FireRedTeam/FireRedVAD)
  的官方权重，**在 Rust 中从头重写**：前端解码 → fbank → CMVN → DFSMN 编码器
  → sigmoid → 状态机后处理，全部用 Rust 实现。
- 模型结构与权重数值与上游一致，因此 **检测准确度与上游官方结果完全一致**
  （见下表）。
- 推理后端用 [`gemm`](https://crates.io/crates/gemm)，运行期做 CPU 特性探测
  （x86 上 AVX-512 / AVX2 / AVX / SSE，ARM 上 NEON），自动挑最快的微内核。
  编译期不锁 `target-cpu`，同一份二进制在低端机到服务器上都能跑出对应平台
  的最优速度。
- 长音频用 [`rayon`](https://crates.io/crates/rayon) 切成 30,000 帧为一块并行
  推理，单文件多核自动扩展。

This is **not** a wrapper around the upstream Python package. It is a
ground-up Rust reimplementation of the same VAD pipeline (decode → fbank →
CMVN → DFSMN encoder → sigmoid → state-machine postprocess), consuming the
upstream weights directly. Detection quality therefore matches the published
numbers exactly (table below).

---

## 检测准确度 · Detection Quality

与上游同源（[FireRedTeam/FireRedVAD](https://huggingface.co/FireRedTeam/FireRedVAD) /
[FireRedTeam/FireRedASR2S](https://github.com/FireRedTeam/FireRedASR2S)），
测试集 FLEURS-VAD-102：

| Metric \ Model | FireRedVAD | Silero-VAD | TEN-VAD | FunASR-VAD | WebRTC-VAD |
|:--|--:|--:|--:|--:|--:|
| AUC-ROC ↑        | **99.60** | 97.99 | 97.81 | -     | -     |
| F1 score ↑       | **97.57** | 95.95 | 95.19 | 90.91 | 52.30 |
| False Alarm Rate ↓ | **2.69** | 9.41  | 15.47 | 44.03 | 2.83  |
| Miss Rate ↓      | 3.62      | 3.95  | 2.95  | 0.42  | 64.15 |

> FunASR-VAD 的低 Miss Rate 是以 44% 的高误报率换来的——过预测语音段。
> Silero / TEN / FunASR / WebRTC 的数字引用自上游技术报告。

---

## 特性 · Features

- **单文件可执行** — 模型权重编译进二进制；分发只需一个 `.exe`，无需
  任何模型文件、环境变量或运行时。
- **CPU 优化推理** — `gemm` 矩阵乘运行期探测 AVX-512/AVX2/AVX/SSE（x86）
  或 NEON（ARM），自动选用最优微内核。`Cargo.toml` 故意不锁 `target-cpu`，
  同一份二进制在新老机器上都能跑。
- **并行分块推理** — 长音频用 `rayon` 切成 30,000 帧为一块并行处理，线性
  扩展到全部可用核。
- **格式宽容** — 多通道 WAV 自动混音为单声道；任意采样率自动重采样到
  16 kHz；PCM16 / float32 都吃。

---

## 快速开始 · Quick Start

### A. 下载预编译二进制 / Prebuilt binary

到 [Releases](../../releases) 页面下载最新版：

```powershell
.\fireredvad.exe D:\path\to\audio.wav
```

### B. 从源码构建 / Build from source

```powershell
cargo build --release
# 产物位于 target\release\fireredvad.exe
```

---

## 用法 · Usage

```powershell
.\fireredvad.exe D:\path\to\audio.wav
```

把结果写到文件：

```powershell
.\fireredvad.exe D:\path\to\audio.wav > vad_result.json
```

输出示例：

```json
{
  "dur": 2.32,
  "timestamps": [[0.44, 1.82]],
  "wav_path": "D:\\path\\to\\audio.wav",
  "sample_rate_in": 48000,
  "sample_rate_model": 16000
}
```

`timestamps` 是若干 `[start_sec, end_sec]` 二元组，标记所有语音段。

---

## 作为 Rust 库使用 · Use as a Rust Library

除了命令行可执行程序，`fireredvad` 也发布为 Rust crate。高层 `Vad` API 封装了
模型加载、音频前端、推理与后处理，三行代码即可在你的程序里完成 VAD。

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
    let vad = Vad::new()?;                              // 内置预训练模型，零配置
    let out = vad.detect_wav("audio.wav", &VadConfig::default())?;
    for (start, end) in &out.timestamps {
        println!("{start:.3}s -> {end:.3}s");
    }
    Ok(())
}
```

`VadConfig` 的字段与下方「命令行参数」表一一对应 —— 同一套阈值/平滑/最短帧数
语义，库和 CLI 共用。需要自定义音频源（如流式、内存缓冲）时，改用
`detect_pcm` 接收原始 16 kHz 单声道样本即可。

The fields of `VadConfig` map one-to-one to the **CLI Parameters** table below
— the same threshold / smoothing / min-frames semantics are shared between the
library and the CLI. For custom audio sources (streaming, in-memory buffers),
use `detect_pcm` with raw 16 kHz mono samples instead.

`VadConfig.silence_schedule` is also settable programmatically: pass
`FUNASR_OFFLINE_SCHEDULE.to_vec()` (or your own `(upper_ms, threshold_ms)`
table) to enable dynamic silence cuts from library code — the same thing the
`--dynamic-vad` CLI flag does.

---

## 命令行参数 · CLI Parameters

| 参数 / Parameter | 默认值 / Default | 说明 / Description |
|:--|:--|:--|
| `<wav_path>` | *(必填 / required)* | 输入 WAV 路径 / Input WAV path |
| `--speech-threshold`    | `0.4`   | 语音概率阈值 / Speech probability threshold |
| `--smooth-window-size`  | `5`     | 滑动窗口平滑（帧）/ Smoothing window (frames) |
| `--min-speech-frame`    | `20`    | 确认为语音所需的最少连续帧 / Min consecutive speech frames |
| `--max-speech-frame`    | `2000`  | 单段语音的最大长度（超过则强制切分）/ Max segment length |
| `--min-silence-frame`   | `20`    | 确认为静音所需的最少连续帧 / Min consecutive silence frames |
| `--merge-silence-frame` | `0`     | 短于该帧数的静音缝直接合并 / Merge short silence gaps |
| `--extend-speech-frame` | `0`     | 语音段前后各延展的帧数 / Bidirectional extension (frames) |
| `--chunk-max-frame`     | `30000` | 并行分块推理的最大帧数 / Max frames per parallel chunk |
| `--dynamic-vad`         | `off`   | 启用动态静音阈值(FunASR offline 策略，段越长静音切分越紧）/ Enable dynamic silence threshold |

---

## 动态 VAD · Dynamic VAD

默认情况下，`--min-silence-frame`（默认 20 帧 = 200 ms）是**固定**的静音切分
阈值：无论当前语音段说了多久，都要静音这么久才切段。这在长音频里会留下过长的
整段（说话人的换气停顿、思考停顿不足以触发切分）。

开启 `--dynamic-vad` 后，静音切分阈值随**当前语音段的累积时长动态收紧**——段越长，
一个越短的停顿就足以切段。这来自 [FunASR](https://github.com/modelscope/FunASR)
FSMN-VAD 的 offline 动态阈值策略，适合 ASR 数据切片等"希望长段被合理切碎"的场景。

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

**内置阈值表 · Built-in schedule**（FunASR offline）：

| 当前段累积时长 / Segment accumulated | 静音切分阈值 / Silence threshold |
|:--|--:|
| ≤ 5 s          | 2000 ms |
| 5–10 s         | 2000 ms |
| 10–15 s        | 1000 ms |
| 15–20 s        | 1000 ms |
| 20–30 s        |  800 ms |
| 30–45 s        |  600 ms |
| 45–60 s        |  300 ms |
| > 60 s         |  100 ms |

> `--dynamic-vad` 与 `--min-silence-frame` 互斥：开启动态后，固定阈值被忽略。
> `--max-speech-frame`（硬切上限）依然生效作为兜底——当一整段没有任何静音停顿时，
> 仍按它强制切分。两者互补。
>
> `--dynamic-vad` is mutually exclusive with `--min-silence-frame` (the fixed
> value is ignored once dynamic is on). `--max-speech-frame` still applies as a
> hard ceiling: a segment with no silence gap at all is still force-cut by it.

---

## 工作流程 · How It Works

```
WAV ─► 混音到 mono ─► 重采样到 16 kHz ─► 80 维 fbank ─► CMVN
   ─► DFSMN 编码器 ─► sigmoid 语音概率 ─► 状态机后处理 ─► 语音段
```

1. **音频前端**（`src/audio.rs`）— 读 PCM16 / float32 WAV，混音到 mono，
   重采样到 16 kHz。
2. **特征提取**（`src/fbank.rs`）— 80 维 log-Mel 滤波器组 + Povey 窗。
3. **归一化**（`src/cmvn.rs`）— CMVN 均值/方差归一化。
4. **声学模型**（`src/model.rs`）— Deep FSMN 编码器；矩阵乘走 `gemm`，
   运行期按 CPU 特性派发。
5. **后处理**（`src/postprocess.rs`）— 平滑、阈值、最小语音/静音时长
   状态机、缝合并、段切分。

---

## 模型权重 · Model Weights

`model/` 下的预训练权重（`weights.npz` / `cmvn.json` / `model_meta.json`）
源自 [FireRedTeam/FireRedASR2S](https://github.com/FireRedTeam/FireRedASR2S)
VAD checkpoint，沿用 Apache-2.0 协议再分发。

---

## 致谢 · Acknowledgements

- [FireRedTeam/FireRedVAD](https://huggingface.co/FireRedTeam/FireRedVAD) —
  上游 VAD 模型与原始 benchmark。
- [FireRedTeam/FireRedASR2S](https://github.com/FireRedTeam/FireRedASR2S) —
  上游 ASR + VAD + LID + Punc 一体化系统的 VAD 模块来源。

---

## 许可证 · License

[Apache-2.0](LICENSE)。打包的模型权重按相同协议从 FireRedASR2S 再分发。
