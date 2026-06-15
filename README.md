# FireRedVAD (Windows Binary)

Native Rust CLI for non-streaming Voice Activity Detection (VAD).
The model is embedded inside `fireredvad.exe`, so no external model files are required.

Upstream project: [FireRedTeam/FireRedASR2S](https://github.com/FireRedTeam/FireRedASR2S)

## FireRedVAD Benchmark (from FireRedASR2S)

| Metric\Model | FireRedVAD | Silero-VAD | TEN-VAD | FunASR-VAD | WebRTC-VAD |
|:--|--:|--:|--:|--:|--:|
| AUC-ROC ↑ | **99.60** | 97.99 | 97.81 | - | - |
| F1 score ↑ | **97.57** | 95.95 | 95.19 | 90.91 | 52.30 |
| False Alarm Rate ↓ | **2.69** | 9.41 | 15.47 | 44.03 | 2.83 |
| Miss Rate ↓ | 3.62 | 3.95 | 2.95 | 0.42 | 64.15 |

Source: [FireRedTeam/FireRedASR2S](https://github.com/FireRedTeam/FireRedASR2S).

## Usage

Run with a WAV file:

```powershell
.\fireredvad.exe D:\path\to\audio.wav
```

Save output JSON:

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

## CLI Parameters

- `<wav_path>` required, input WAV path
- `--speech-threshold` default `0.4`
- `--smooth-window-size` default `5`
- `--min-speech-frame` default `20`
- `--max-speech-frame` default `2000`
- `--min-silence-frame` default `20`
- `--merge-silence-frame` default `0`
- `--extend-speech-frame` default `0`
- `--chunk-max-frame` default `30000`

## Input Notes

- WAV input
- Multi-channel is supported (auto mixed to mono)
- Any sample rate is supported (auto resampled to 16k)

## License

Apache-2.0.
