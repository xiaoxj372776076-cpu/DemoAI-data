#!/usr/bin/env python3
"""Transcribe a local media file with Whisper and write structured JSON."""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any, Dict, Optional, Sequence


DEFAULT_MODEL = "mlx-community/whisper-small-mlx"
DEFAULT_MODEL_CACHE = Path.home() / ".cache" / "demoai-data" / "asr"


def input_media(value: str) -> Path:
    """Validate and return an existing local media file."""
    path = Path(value).expanduser().resolve()
    if not path.is_file():
        raise argparse.ArgumentTypeError(f"input media file does not exist: {path}")
    return path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Transcribe a local video or audio file to JSON with Whisper."
    )
    parser.add_argument("input", type=input_media, help="local video or audio file")
    parser.add_argument(
        "--output",
        type=Path,
        help="JSON output file (default: INPUT.asr.json beside the media file)",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"MLX Whisper model path or Hugging Face repository (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--model-cache",
        type=Path,
        default=DEFAULT_MODEL_CACHE,
        help="model cache directory (default: ~/.cache/demoai-data/asr)",
    )
    parser.add_argument(
        "--language",
        help="spoken language code, such as zh or en (default: auto-detect)",
    )
    parser.add_argument(
        "--word-timestamps",
        action="store_true",
        help="include word-level timestamps in each segment",
    )
    return parser


def default_output_path(input_path: Path) -> Path:
    """Return the default JSON path beside an input media file."""
    return input_path.with_name(f"{input_path.stem}.asr.json")


def media_duration(input_path: Path) -> Optional[float]:
    """Read media duration with ffprobe when it is available."""
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        return None

    completed = subprocess.run(
        [
            ffprobe,
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(input_path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        return None
    try:
        return round(float(completed.stdout.strip()), 3)
    except ValueError:
        return None


def media_has_audio(input_path: Path) -> Optional[bool]:
    """Return whether the media has an audio stream when ffprobe is available."""
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        return None

    completed = subprocess.run(
        [
            ffprobe,
            "-v",
            "error",
            "-select_streams",
            "a:0",
            "-show_entries",
            "stream=index",
            "-of",
            "csv=p=0",
            str(input_path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        return None
    return bool(completed.stdout.strip())


def json_value(value: Any) -> Any:
    """Convert model output values into JSON-compatible Python values."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(item) for item in value]
    if hasattr(value, "item"):
        return json_value(value.item())
    raise TypeError(f"unsupported result value: {type(value).__name__}")


def transcribe(
    input_path: Path,
    model: str,
    model_cache: Path,
    language: Optional[str] = None,
    word_timestamps: bool = False,
) -> Dict[str, Any]:
    """Run MLX Whisper and return a JSON-compatible result document."""
    duration = media_duration(input_path)
    audio_present = media_has_audio(input_path)
    if audio_present is False:
        return {
            "source": str(input_path),
            "model": model,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "duration_seconds": duration,
            "audio_present": False,
            "language": None,
            "text": "",
            "segments": [],
        }

    cache_path = model_cache.expanduser().resolve()
    hub_cache = cache_path / "hub"
    xet_cache = cache_path / "xet"
    hub_cache.mkdir(parents=True, exist_ok=True)
    xet_cache.mkdir(parents=True, exist_ok=True)

    os.environ["HF_HOME"] = str(cache_path)
    os.environ["HF_HUB_CACHE"] = str(hub_cache)
    os.environ["HF_XET_CACHE"] = str(xet_cache)

    import mlx_whisper

    result = mlx_whisper.transcribe(
        str(input_path),
        path_or_hf_repo=model,
        language=language,
        task="transcribe",
        word_timestamps=word_timestamps,
        verbose=False,
    )
    normalized = json_value(result)
    return {
        "source": str(input_path),
        "model": model,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "duration_seconds": duration,
        "audio_present": audio_present,
        "language": normalized.get("language"),
        "text": normalized.get("text", "").strip(),
        "segments": normalized.get("segments", []),
    }


def write_result(result: Dict[str, Any], output_path: Path) -> Path:
    """Atomically write an ASR result document as UTF-8 JSON."""
    destination = output_path.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(f"{destination.suffix}.tmp")
    temporary.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(destination)
    return destination


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    output_path = (
        args.output.expanduser().resolve()
        if args.output
        else default_output_path(args.input)
    )

    try:
        result = transcribe(
            input_path=args.input,
            model=args.model,
            model_cache=args.model_cache,
            language=args.language,
            word_timestamps=args.word_timestamps,
        )
        written_path = write_result(result, output_path)
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        print(f"ASR failed: {error}", file=sys.stderr)
        return 1

    print(f"ASR completed: {written_path}")
    print(f"Language: {result['language']}")
    print(f"Segments: {len(result['segments'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
