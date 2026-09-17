#!/usr/bin/env python3
"""Download a Hugging Face dataset repository to a local directory."""

import argparse
import os
import re
import sys
from pathlib import Path
from typing import Optional, Sequence, Tuple
from urllib.parse import unquote, urlparse

from huggingface_hub import snapshot_download
from huggingface_hub.errors import (
    HFValidationError,
    HfHubHTTPError,
    LocalEntryNotFoundError,
)


REPO_SEGMENT = r"[A-Za-z0-9](?:[A-Za-z0-9._-]{0,94}[A-Za-z0-9])?"
DATASET_ID_PATTERN = re.compile(rf"^{REPO_SEGMENT}(?:/{REPO_SEGMENT})?$")
HUGGING_FACE_HOSTS = {"huggingface.co", "www.huggingface.co"}


def dataset_id(value: str) -> str:
    """Normalize a Hugging Face dataset URL or dataset ID."""
    raw_value = value.strip()
    parsed = urlparse(raw_value)

    if parsed.scheme or parsed.netloc:
        if parsed.scheme != "https" or parsed.hostname not in HUGGING_FACE_HOSTS:
            raise argparse.ArgumentTypeError(
                "dataset URL must use https://huggingface.co/datasets/..."
            )

        path_parts = [unquote(part) for part in parsed.path.split("/") if part]
        if len(path_parts) < 2 or path_parts[0] != "datasets":
            raise argparse.ArgumentTypeError(
                "dataset URL must use https://huggingface.co/datasets/..."
            )

        repo_parts = path_parts[1:3]
        raw_value = "/".join(repo_parts)
    else:
        raw_value = raw_value.strip("/")

    if not DATASET_ID_PATTERN.fullmatch(raw_value):
        raise argparse.ArgumentTypeError(
            "dataset must be a name or owner/name using letters, numbers, '.', '_' or '-'"
        )
    return raw_value


def positive_integer(value: str) -> int:
    workers = int(value)
    if workers < 1:
        raise argparse.ArgumentTypeError("workers must be at least 1")
    return workers


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download a Hugging Face dataset repository snapshot."
    )
    parser.add_argument(
        "dataset",
        type=dataset_id,
        help="dataset ID (owner/name) or Hugging Face dataset URL",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path.home() / "Downloads" / "huggingface-datasets",
        help="parent destination directory (default: ~/Downloads/huggingface-datasets)",
    )
    parser.add_argument(
        "--revision",
        help="optional branch, tag, or commit to download",
    )
    parser.add_argument(
        "--workers",
        type=positive_integer,
        default=8,
        help="parallel download workers (default: 8)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="download files again instead of reusing matching local files",
    )
    return parser


def local_dataset_dir(dataset: str, output_dir: Path) -> Path:
    """Return the stable local destination for a dataset ID."""
    return output_dir.expanduser().resolve() / dataset.replace("/", "--")


def download(
    dataset: str,
    output_dir: Path,
    revision: Optional[str] = None,
    workers: int = 8,
    force: bool = False,
) -> Path:
    """Download one dataset snapshot and return its local directory."""
    destination = local_dataset_dir(dataset, output_dir)
    destination.parent.mkdir(parents=True, exist_ok=True)
    cache_root = destination.parent / ".cache" / "huggingface"
    hub_cache = cache_root / "hub"
    xet_cache = cache_root / "xet"
    hub_cache.mkdir(parents=True, exist_ok=True)
    xet_cache.mkdir(parents=True, exist_ok=True)
    os.environ["HF_XET_CACHE"] = str(xet_cache)

    snapshot_path = snapshot_download(
        repo_id=dataset,
        repo_type="dataset",
        revision=revision,
        cache_dir=hub_cache,
        local_dir=destination,
        max_workers=workers,
        force_download=force,
    )
    downloaded_path = Path(snapshot_path).resolve()
    if not downloaded_path.is_dir():
        raise RuntimeError("download finished, but the dataset directory was not found")
    return downloaded_path


def dataset_summary(dataset_dir: Path) -> Tuple[int, int]:
    """Return the number and total size of downloaded dataset files."""
    files = [
        path
        for path in dataset_dir.rglob("*")
        if path.is_file() and ".cache" not in path.relative_to(dataset_dir).parts
    ]
    return len(files), sum(path.stat().st_size for path in files)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        output_path = download(
            args.dataset,
            args.output_dir,
            revision=args.revision,
            workers=args.workers,
            force=args.force,
        )
        file_count, total_bytes = dataset_summary(output_path)
        if file_count == 0:
            raise RuntimeError("download finished, but no dataset files were found")
    except (
        HFValidationError,
        HfHubHTTPError,
        LocalEntryNotFoundError,
        OSError,
        RuntimeError,
    ) as error:
        print(f"Download failed: {error}", file=sys.stderr)
        return 1

    print(f"Download completed: {output_path}")
    print(f"Dataset files: {file_count}")
    print(f"Dataset size: {total_bytes} bytes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
