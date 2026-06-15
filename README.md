[English](README.en.md) | 简体中文

# FireRedVAD

原生 Rust 实现的工业级 VAD。模型权重编译进二进制,单文件可执行,零运行时依赖。也可作为 Rust crate 嵌入。

## 性能

Intel Core Ultra 7 265K(20 核)上,7835 秒音频推理耗时 **3.96 秒**,**RTFx ≈ 1978×**。相比原版 Python(PyTorch)快 10–100 倍,且无需 GPU。

> 测试方法:5 次连续推理取中位,脚本见 `tmp/bench_rtfx.ps1`。

## 准确度

与上游同源([FireRedTeam/FireRedVAD](https://huggingface.co/FireRedTeam/FireRedVAD)),测试集 FLEURS-VAD-102:

| Metric | FireRedVAD | Silero | TEN | FunASR | WebRTC |
|:--|--:|--:|--:|--:|--:|
| AUC-ROC ↑ | **99.60** | 97.99 | 97.81 | - | - |
| F1 ↑ | **97.57** | 95.95 | 95.19 | 90.91 | 52.30 |
| 误报率 ↓ | **2.69** | 9.41 | 15.47 | 44.03 | 2.83 |

## 特性

- **单文件可执行** — 权重编译进二进制,无需任何模型文件或运行时
- **CPU 加速** — `gemm` 运行期探测 AVX-512/AVX2/AVX/SSE/NEON,自动选最快微内核
- **并行推理** — `rayon` 切块并行,多核线性扩展
- **格式宽容** — 多通道自动混音,任意采样率重采样到 16kHz,PCM16/float32 都吃
- **动态 VAD** — 可选的动态静音阈值,避免把完整句子切碎

## 快速开始

**下载二进制**(从 [Releases](../../releases)):
```powershell
.\fireredvad.exe D:\path\to\audio.wav
```

**从源码构建**:
```powershell
cargo build --release
```

## 用法

```powershell
.\fireredvad.exe audio.wav > result.json
```

输出:
```json
{
  "dur": 2.32,
  "timestamps": [[0.44, 1.82]],
  "wav_path": "audio.wav",
  "sample_rate_in": 48000,
  "sample_rate_model": 16000
}
```

`timestamps` 是 `[start_sec, end_sec]` 语音段列表。

## 作为 Rust 库使用

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

`VadConfig` 字段与下方 CLI 参数一一对应。自定义音频源可用 `detect_pcm` 接收原始 16kHz 单声道样本。

## CLI 参数

| 参数 | 默认值 | 说明 |
|:--|:--|:--|
| `<wav>` | *(必填)* | 输入 WAV 路径 |
| `--speech-threshold` | `0.4` | 语音概率阈值 |
| `--smooth-window-size` | `5` | 滑动窗口平滑(帧) |
| `--min-speech-frame` | `20` | 确认语音的最少连续帧 |
| `--max-speech-frame` | `2000` | 单段最大长度,超过强制切分 |
| `--min-silence-frame` | `20` | 确认静音的最少连续帧 |
| `--merge-silence-frame` | `0` | 短于该帧数的静音缝合并 |
| `--extend-speech-frame` | `0` | 语音段前后延展帧数 |
| `--chunk-max-frame` | `30000` | 并行分块的最大帧数 |
| `--dynamic-vad` | `off` | 启用动态静音阈值 |

## 动态 VAD

默认 `--min-silence-frame` 是**固定**的 200ms:不管说了多久,停 200ms 就切。这会把完整句子切碎(换气、思考停顿都会触发切分)。

`--dynamic-vad` 让切分阈值随**当前段的累积时长**收紧——段越长,越短的停顿就足以切分:

| 当前段已说 | 需静音超过 | 才切 |
|:--|--:|--:|
| ≤ 5 s | 2000 ms | 2 秒 |
| ≤ 10 s | 1500 ms | 1.5 秒 |
| ≤ 15 s | 1000 ms | 1 秒 |
| ≤ 30 s | 800 ms | |
| ≤ 45 s | 400 ms | |
| > 45 s | 100 ms | 一停就切 |

```powershell
.\fireredvad.exe audio.wav --dynamic-vad
```

> - 与 `--min-silence-frame` 互斥,开启后固定阈值被忽略
> - `--max-speech-frame` 仍作为兜底硬切
> - 表源自 [FunASR](https://github.com/modelscope/FunASR) `fsmn_vad_streaming/dynamic_vad.py`

## 工作流程

```
WAV → 混音 mono → 重采样 16kHz → 80维 fbank → CMVN → DFSMN 编码器 → sigmoid → 状态机后处理 → 语音段
```

| 模块 | 作用 |
|:--|:--|
| `audio.rs` | 解码 PCM16/float32 WAV,混音,重采样 |
| `fbank.rs` | 80维 log-Mel 滤波器组 + Povey 窗 |
| `cmvn.rs` | CMVN 均值/方差归一化 |
| `model.rs` | Deep FSMN 编码器(`gemm` 矩阵乘) |
| `postprocess.rs` | 平滑、阈值、状态机、切段 |

## 权重来源

`model/` 下权重源自 [FireRedTeam/FireRedASR2S](https://github.com/FireRedTeam/FireRedASR2S),Apache-2.0 协议再分发。

## 许可证

[Apache-2.0](LICENSE)
