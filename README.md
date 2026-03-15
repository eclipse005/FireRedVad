# FireRedVAD (Windows Binary)

Native Rust CLI for FireRedVAD on Windows with embedded model and Python-parity timestamps.

## Download

Download the latest `fireredvad.exe` from **Releases**.

## Usage

Open `cmd` or PowerShell in the folder containing `fireredvad.exe`, then run:

```powershell
.\fireredvad.exe --wav D:\path\to\audio.wav
```

Save output to a file:

```powershell
.\fireredvad.exe --wav D:\path\to\audio.wav > vad_result.json
```

## Input Requirements

- WAV input
- Multi-channel supported (automatically mixed to mono)
- Any sample rate supported (automatically resampled to 16k)

## Output Format

The CLI prints one JSON object to stdout, for example:

```json
{
  "dur": 2.32,
  "timestamps": [[0.44, 1.82]],
  "wav_path": "D:\\path\\to\\audio.wav",
  "sample_rate_in": 48000,
  "sample_rate_model": 16000
}
```

## CLI Options

- `--wav <path>` (required)
- `--speech-threshold` (default: `0.4`)
- `--smooth-window-size` (default: `5`)
- `--min-speech-frame` (default: `20`)
- `--max-speech-frame` (default: `2000`)
- `--min-silence-frame` (default: `20`)
- `--merge-silence-frame` (default: `0`)
- `--extend-speech-frame` (default: `0`)
- `--chunk-max-frame` (default: `30000`)

## Notes

- This repository currently distributes **Windows binary only**.
- If you double-click the executable without arguments, it will show usage guidance.
- For large files, processing time depends on CPU performance and disk speed.

## License

Apache-2.0.
This project is based on FireRedTeam/FireRedASR2S and includes binary distribution under the Apache-2.0 license.
