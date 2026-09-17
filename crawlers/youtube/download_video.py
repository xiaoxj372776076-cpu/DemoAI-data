#!/usr/bin/env python3
"""Download a YouTube video by ID with a 1080p60 quality cap."""

import argparse
import re
import shutil
import sys
from pathlib import Path
from typing import Optional, Sequence

import yt_dlp


VIDEO_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{11}$")
FORMAT_SELECTOR = (
    "bestvideo[height<=1080][fps<=60][protocol=https]+"
    "bestaudio[protocol=https]/"
    "best[height<=1080][fps<=60][protocol=https]/"
    "bestvideo[height<=1080][fps<=60]+bestaudio/"
    "best[height<=1080][fps<=60]/"
    "bestvideo[height<=1080]+bestaudio/"
    "best[height<=1080]"
)
SUPPORTED_BROWSERS = ("chrome", "safari", "firefox", "edge", "brave")


def video_id(value: str) -> str:
    """Validate and return a YouTube video ID."""
    if not VIDEO_ID_PATTERN.fullmatch(value):
        raise argparse.ArgumentTypeError(
            "video ID must be exactly 11 characters using letters, numbers, '_' or '-'"
        )
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download a YouTube video at the best quality up to 1080p60."
    )
    parser.add_argument("video_id", type=video_id, help="11-character YouTube video ID")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path.home() / "Downloads",
        help="destination directory (default: ~/Downloads)",
    )
    parser.add_argument(
        "--cookies-from-browser",
        choices=SUPPORTED_BROWSERS,
        help="reuse a local browser session when YouTube requires sign-in",
    )
    return parser


def download(
    video_id_value: str,
    output_dir: Path,
    cookies_from_browser: Optional[str] = None,
) -> Path:
    """Download one video and return its final local path."""
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg is required but was not found on PATH")

    destination = output_dir.expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    url = f"https://www.youtube.com/watch?v={video_id_value}"

    options = {
        "format": FORMAT_SELECTOR,
        "format_sort": ["res:1080", "fps:60"],
        "merge_output_format": "mp4",
        "outtmpl": str(destination / "%(title).180B [%(id)s].%(ext)s"),
        "noplaylist": True,
        "overwrites": False,
        "restrictfilenames": True,
    }
    node_path = shutil.which("node")
    if node_path:
        options["js_runtimes"] = {"node": {"path": node_path}}
    if cookies_from_browser:
        options["cookiesfrombrowser"] = (cookies_from_browser,)

    with yt_dlp.YoutubeDL(options) as downloader:
        info = downloader.extract_info(url, download=True)
        prepared = Path(downloader.prepare_filename(info))
        candidates = [prepared.with_suffix(".mp4"), prepared]
        requested_downloads = info.get("requested_downloads") or []
        candidates.extend(
            Path(item["filepath"])
            for item in requested_downloads
            if item.get("filepath")
        )

        for candidate in candidates:
            if candidate.exists():
                return candidate

        downloaded_files = sorted(
            (
                path
                for path in destination.iterdir()
                if video_id_value in path.name
                and path.suffix.lower() in {".mp4", ".mkv", ".webm"}
            ),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        if downloaded_files:
            return downloaded_files[0]

        raise RuntimeError("download finished, but the output file could not be found")


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        output_path = download(
            args.video_id,
            args.output_dir,
            cookies_from_browser=args.cookies_from_browser,
        )
    except (RuntimeError, yt_dlp.utils.DownloadError) as error:
        print(f"Download failed: {error}", file=sys.stderr)
        return 1

    print(f"Download completed: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
