# ASR operator

Transcribes a local video or audio file with an open-source Whisper model and
writes the complete transcript plus timestamped segments to JSON. The default
model is `mlx-community/whisper-small-mlx`, optimized for Apple Silicon through
MLX.

## Requirements

- Apple Silicon Mac
- Python 3.10+
- Python dependencies from `requirements.txt`
- `ffmpeg` and `ffprobe` available on `PATH`

```bash
python3 -m pip install -r operators/asr/requirements.txt
```

## Usage

The JSON result is written beside the input video by default:

```bash
python3 operators/asr/transcribe_video.py \
  "/path/to/video.mp4"
```

Choose an explicit output file or language when needed:

```bash
python3 operators/asr/transcribe_video.py \
  "/path/to/video.mp4" \
  --output "/path/to/result.json" \
  --language zh
```

Use `--word-timestamps` to include word-level timestamps. Models are cached in
`~/.cache/demoai-data/asr` by default; override that location with
`--model-cache`.

Videos without an audio stream are valid inputs. They produce a successful JSON
result with `audio_present: false`, empty text, and no transcript segments.
