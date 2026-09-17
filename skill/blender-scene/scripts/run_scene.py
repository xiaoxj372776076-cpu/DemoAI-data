#!/usr/bin/env python3
"""Run a generated Blender scene script and verify its artifact bundle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import struct
import subprocess
import sys
from typing import Any, Dict, Optional, Sequence


DEFAULT_MACOS_BLENDER = Path("/Applications/Blender.app/Contents/MacOS/Blender")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a bpy scene generator and validate its saved artifacts."
    )
    parser.add_argument("--script", type=Path, required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--blender", type=Path)
    return parser


def resolve_blender(explicit: Optional[Path] = None) -> Path:
    if explicit is not None:
        candidate = explicit.expanduser().resolve()
        if not candidate.is_file():
            raise FileNotFoundError(f"Blender executable does not exist: {candidate}")
        return candidate

    on_path = shutil.which("blender")
    if on_path:
        return Path(on_path).resolve()
    if DEFAULT_MACOS_BLENDER.is_file():
        return DEFAULT_MACOS_BLENDER
    raise FileNotFoundError("Blender was not found on PATH or in /Applications")


def png_dimensions(path: Path) -> tuple[int, int]:
    header = path.read_bytes()[:24]
    if len(header) < 24 or header[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError(f"not a valid PNG file: {path}")
    width, height = struct.unpack(">II", header[16:24])
    if width < 1 or height < 1:
        raise ValueError(f"invalid PNG dimensions: {path}")
    return width, height


def validate_artifacts(output_dir: Path) -> Dict[str, Any]:
    prompt_path = output_dir / "prompt.txt"
    code_path = output_dir / "scene.py"
    blend_files = sorted(output_dir.glob("*.blend"))
    render_files = sorted(output_dir.glob("*.png"))

    missing = []
    if not prompt_path.is_file() or not prompt_path.read_text(encoding="utf-8").strip():
        missing.append("non-empty prompt.txt")
    if not code_path.is_file() or code_path.stat().st_size == 0:
        missing.append("scene.py")
    if not blend_files:
        missing.append("a .blend scene")
    if not render_files:
        missing.append("a PNG render")
    if missing:
        raise RuntimeError("incomplete Blender output: " + ", ".join(missing))

    renders = [
        {
            "path": str(path),
            "width": png_dimensions(path)[0],
            "height": png_dimensions(path)[1],
        }
        for path in render_files
    ]
    return {
        "output_dir": str(output_dir),
        "prompt": str(prompt_path),
        "code": str(code_path),
        "blend_files": [str(path) for path in blend_files],
        "renders": renders,
    }


def run_scene(
    blender: Path,
    script: Path,
    prompt: str,
    output_dir: Path,
) -> Dict[str, Any]:
    source = script.expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"scene script does not exist: {source}")
    destination = output_dir.expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "prompt.txt").write_text(prompt.strip() + "\n", encoding="utf-8")

    archived_script = destination / "scene.py"
    if source != archived_script:
        shutil.copy2(source, archived_script)

    command = [
        str(blender),
        "--background",
        "--python",
        str(archived_script),
        "--",
        "--prompt",
        prompt,
        "--output-dir",
        str(destination),
    ]
    completed = subprocess.run(command, check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"Blender exited with code {completed.returncode}")
    return validate_artifacts(destination)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = run_scene(
            blender=resolve_blender(args.blender),
            script=args.script,
            prompt=args.prompt,
            output_dir=args.output_dir,
        )
    except (FileNotFoundError, OSError, RuntimeError, ValueError) as error:
        print(f"Blender scene generation failed: {error}", file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
