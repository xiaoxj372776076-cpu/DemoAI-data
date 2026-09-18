#!/usr/bin/env python3
"""Parse a whole PDF into one HTML document with the dots.ocr model.

This is the multi-page sibling of the ``doc_ocr`` operator. It takes a PDF,
rasterises the requested pages, runs the same dots.ocr vision-language model
over each page image, and stitches the per-page answers back together in the
original page order:

  * one semantic HTML document (page order preserved, page markers inserted),
  * one structured JSON document with a per-page section **and** a flattened
    list of blocks that carries the page number of every block,
  * optionally a long PNG rendering of the merged result (``--render``),
    per-page HTML renderings (``--render-per-page``) and per-page layout
    overlays (``--overlay``).

The model is loaded once and reused for every page, and every page must be
parsed successfully for the run to be considered successful: a single broken
page makes the operator exit non-zero and marks that page in the artifacts.
"""

import argparse
import importlib.util
import json
import re
import sys
import tempfile
from datetime import datetime, timezone
from html import escape
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple


CORE_PATH = Path(__file__).resolve().parents[1] / "doc_ocr" / "parse_document.py"


def load_core():
    """Import the doc_ocr operator so this operator can reuse its pipeline."""
    spec = importlib.util.spec_from_file_location("demoai_doc_ocr_core", CORE_PATH)
    if spec is None or spec.loader is None:  # pragma: no cover - defensive
        raise RuntimeError(f"cannot load the doc_ocr operator from {CORE_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault(spec.name, module)
    spec.loader.exec_module(module)
    return module


CORE = load_core()

DEFAULT_MODEL = CORE.DEFAULT_MODEL
DEFAULT_MODEL_ENV = CORE.DEFAULT_MODEL_ENV

PDF_HTML_STYLE = CORE.HTML_STYLE + """
.page { border-top: 1px dashed #d0d5dd; margin-top: 26px; padding-top: 10px; }
.page:first-of-type { border-top: none; margin-top: 0; padding-top: 0; }
.page-marker { color: #3370ff; font-size: 12px; font-weight: bold;
               letter-spacing: .08em; margin-bottom: 8px; }
.page-error { background: #fff1f0; border-left: 3px solid #f54a45; color: #a31515;
              padding: 8px 12px; margin: 10px 0; white-space: pre-wrap; }
"""

PAGE_SPEC_PATTERN = re.compile(r"^(\d+)(?:-(\d+))?$")


class PageParseError(RuntimeError):
    """Raised when a single page cannot be turned into layout blocks."""


def input_pdf(value: str) -> Path:
    """Validate that the input is an existing PDF file."""
    path = Path(value).expanduser().resolve()
    if not path.is_file():
        raise argparse.ArgumentTypeError(f"input file does not exist: {path}")
    if path.suffix.lower() != ".pdf":
        raise argparse.ArgumentTypeError(
            f"this operator only accepts PDF input, got: {path.suffix or 'file'}"
        )
    return path


def positive_integer(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be at least 1")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Parse a whole PDF into a single HTML document with dots.ocr."
    )
    parser.add_argument("input", type=input_pdf, help="PDF file to parse")
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="directory for the HTML/JSON artifacts (default: beside the input)",
    )
    parser.add_argument(
        "--model",
        default=None,
        help=(
            "MLX dots.ocr model path or Hugging Face repo "
            f"(default: ${DEFAULT_MODEL_ENV} or {DEFAULT_MODEL})"
        ),
    )
    parser.add_argument(
        "--prompt-mode",
        choices=sorted(CORE.PROMPTS),
        default="layout_all",
        help="prompt preset sent to the model (default: layout_all)",
    )
    parser.add_argument(
        "--max-tokens",
        type=positive_integer,
        default=8192,
        help="maximum generated tokens per page (default: 8192)",
    )
    parser.add_argument(
        "--pages",
        default="all",
        help=(
            'pages to parse: "all" (default) or a comma separated list of '
            'numbers and ranges, e.g. "1-3,5,8-10"'
        ),
    )
    parser.add_argument(
        "--dpi",
        type=positive_integer,
        default=150,
        help="rasterisation DPI for each page (default: 150)",
    )
    parser.add_argument(
        "--max-image-side",
        type=int,
        default=0,
        help=(
            "downscale each page image so its longest edge is at most N pixels "
            "(0 disables downscaling; lower values are much faster)"
        ),
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="rasterise the merged HTML into <stem>.doc.png",
    )
    parser.add_argument(
        "--render-per-page",
        action="store_true",
        help="also rasterise every page into <stem>-pages/page-0001.doc.png",
    )
    parser.add_argument(
        "--overlay",
        action="store_true",
        help="draw the detected boxes onto <stem>-pages/page-0001.layout.png",
    )
    parser.add_argument(
        "--keep-page-images",
        action="store_true",
        help="keep the rasterised page images in <stem>-pages/",
    )
    parser.add_argument(
        "--stop-on-first-error",
        action="store_true",
        help="abort the run as soon as a page fails instead of parsing the rest",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="only print the written artifact paths",
    )
    return parser


def pdf_page_count(pdf_path: Path) -> int:
    """Return the number of pages of a PDF."""
    try:
        import pymupdf
    except ImportError as error:  # pragma: no cover - dependency hint
        raise RuntimeError(
            "PyMuPDF is required to read PDF input (pip install pymupdf)"
        ) from error

    with pymupdf.open(str(pdf_path)) as document:
        return int(document.page_count)


def parse_page_spec(spec: str, page_count: int) -> List[int]:
    """Turn a CLI page specification into a sorted list of 1-based pages."""
    text = (spec or "").strip().lower()
    if text in {"", "all", "*"}:
        return list(range(1, page_count + 1))

    pages: set = set()
    for chunk in text.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        match = PAGE_SPEC_PATTERN.match(chunk)
        if not match:
            raise ValueError(f"invalid page specification: {chunk!r}")
        start = int(match.group(1))
        end = int(match.group(2)) if match.group(2) else start
        if start > end:
            raise ValueError(f"invalid page range (start after end): {chunk!r}")
        if start < 1 or end > page_count:
            raise ValueError(
                f"page range {chunk!r} is outside 1..{page_count}"
            )
        pages.update(range(start, end + 1))
    if not pages:
        raise ValueError(f"no page selected by specification: {spec!r}")
    return sorted(pages)


def downscale_image(image_path: Path, max_side: int) -> Path:
    """Shrink an image in place so its longest edge fits ``max_side``."""
    try:
        from PIL import Image
    except ImportError as error:  # pragma: no cover - dependency hint
        raise RuntimeError("Pillow is required to resize pages (pip install pillow)") from error

    if max_side <= 0:
        return image_path
    with Image.open(str(image_path)) as source:
        width, height = source.size
        if max(width, height) <= max_side:
            return image_path
        scale = max_side / float(max(width, height))
        resized = source.convert("RGB").resize(
            (max(1, int(round(width * scale))), max(1, int(round(height * scale)))),
            Image.LANCZOS,
        )
        resized.save(str(image_path))
    return image_path


def rasterise_pages(
    pdf_path: Path,
    pages: Sequence[int],
    dpi: int,
    target_dir: Path,
    max_image_side: int = 0,
) -> List[Path]:
    """Render every requested PDF page into ``target_dir`` and return the PNGs."""
    try:
        import pymupdf
    except ImportError as error:  # pragma: no cover - dependency hint
        raise RuntimeError(
            "PyMuPDF is required to read PDF input (pip install pymupdf)"
        ) from error

    target_dir.mkdir(parents=True, exist_ok=True)
    rendered: List[Path] = []
    with pymupdf.open(str(pdf_path)) as document:
        for page_number in pages:
            if page_number > document.page_count:
                raise RuntimeError(
                    f"{pdf_path.name} has {document.page_count} pages, "
                    f"page {page_number} was requested"
                )
            page = document[page_number - 1]
            pixmap = page.get_pixmap(dpi=dpi)
            target = target_dir / f"page-{page_number:04d}.png"
            pixmap.save(str(target))
            downscale_image(target, max_image_side)
            rendered.append(target)
    return rendered


def clear_model_cache() -> None:
    """Release MLX buffers between pages; never fatal."""
    try:  # pragma: no cover - depends on the MLX runtime
        import mlx.core as mx

        mx.clear_cache()
    except Exception:  # noqa: BLE001 - best effort
        pass


def parse_single_page(
    handle: Any,
    image: Path,
    page_number: int,
    prompt_mode: str,
    max_tokens: int,
) -> Tuple[List[Dict[str, Any]], str]:
    """Run the model over one page image and return its layout blocks."""
    answer = handle.generate(image, prompt_mode=prompt_mode, max_tokens=max_tokens)
    items = CORE.extract_json_objects(answer)
    if not items:
        raise PageParseError(
            "the model returned no layout JSON "
            f"(answer starts with: {answer[:120].strip()!r})"
        )
    return CORE.build_blocks(items), answer


def annotate_blocks(
    blocks: Sequence[Dict[str, Any]],
    page_number: int,
    offset: int,
) -> List[Dict[str, Any]]:
    """Give every block a global index and remember where it came from."""
    annotated: List[Dict[str, Any]] = []
    for position, block in enumerate(blocks, start=1):
        item = dict(block)
        item["index"] = offset + position
        item["page"] = page_number
        item["page_index"] = position
        annotated.append(item)
    return annotated


def pages_to_html(
    page_results: Sequence[Dict[str, Any]],
    source: Path,
    model: str,
) -> str:
    """Assemble every page into one HTML document, keeping the PDF order."""
    sections: List[str] = []
    for result in page_results:
        blocks = result["blocks"]
        marker = (
            f'<section class="page" data-page="{result["page"]}">\n'
            f'<div class="page-marker">PAGE {result["page"]}'
            f' · {len(blocks)} block{"" if len(blocks) == 1 else "s"}</div>\n'
        )
        if result.get("status") != "ok":
            sections.append(
                marker
                + f'<div class="page-error">{escape(result["error"])}</div>\n</section>'
            )
            continue
        body = "\n".join(CORE.block_to_html(block) for block in blocks)
        sections.append(f"{marker}{body}\n</section>")

    return (
        "<!DOCTYPE html>\n"
        '<html lang="en">\n<head>\n'
        '<meta charset="utf-8">\n'
        "<title>" + escape(source.name) + " · dots.ocr</title>\n"
        "<style>" + PDF_HTML_STYLE + "</style>\n"
        "</head>\n<body>\n"
        + "\n".join(sections)
        + "\n"
        f'<hr>\n<p class="page-footer">parsed from <code>{escape(str(source))}</code> '
        f"({len(page_results)} pages) with <code>{escape(model)}</code></p>\n"
        "</body>\n</html>\n"
    )


def write_json(path: Path, payload: Dict[str, Any]) -> Path:
    """Atomically write a UTF-8 JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(
        json.dumps(CORE.json_value(payload), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)
    return path


def default_model_factory(model_name: Optional[str]) -> Callable[[], Any]:
    def factory() -> Any:
        return CORE.DocumentModel.load(model_name)

    return factory


def parse_pdf(
    input_file: Path,
    output_dir: Path,
    model_name: Optional[str] = None,
    prompt_mode: str = "layout_all",
    max_tokens: int = 8192,
    pages: str = "all",
    dpi: int = 150,
    max_image_side: int = 0,
    render: bool = False,
    render_per_page: bool = False,
    overlay: bool = False,
    keep_page_images: bool = False,
    stop_on_first_error: bool = False,
    quiet: bool = False,
    model_factory: Optional[Callable[[], Any]] = None,
) -> Dict[str, Any]:
    """Parse every requested page of a PDF into one merged document."""
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    total_pages = pdf_page_count(input_file)
    selected = parse_page_spec(pages, total_pages)

    stem = input_file.stem
    artifacts_dir = output_dir / f"{stem}-pages"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    workspace = Path(tempfile.mkdtemp(prefix="pdf-ocr-")) if not keep_page_images else artifacts_dir

    factory = model_factory or default_model_factory(model_name)
    handle = None
    html_path = output_dir / f"{stem}.doc.html"
    json_path = output_dir / f"{stem}.doc.json"
    page_results: List[Dict[str, Any]] = []
    all_blocks: List[Dict[str, Any]] = []

    page_images = rasterise_pages(input_file, selected, dpi, workspace, max_image_side)
    handle = factory()
    model_path = str(getattr(handle, "model_path", model_name or DEFAULT_MODEL))

    offset = 0
    for position, (page_number, image) in enumerate(zip(selected, page_images), start=1):
        entry: Dict[str, Any] = {
            "page": page_number,
            "image": str(image),
            "status": "ok",
            "error": None,
            "block_count": 0,
            "blocks": [],
        }
        try:
            blocks, _answer = parse_single_page(
                handle, image, page_number, prompt_mode, max_tokens
            )
        except Exception as error:  # noqa: BLE001 - one bad page must not kill the run
            entry["status"] = "error"
            entry["error"] = f"{type(error).__name__}: {error}"
        else:
            blocks_with_scope = annotate_blocks(blocks, page_number, offset)
            offset += len(blocks_with_scope)
            entry["blocks"] = blocks_with_scope
            entry["block_count"] = len(blocks_with_scope)
            all_blocks.extend(blocks_with_scope)

        page_results.append(entry)
        if not quiet:
            state = "ok " if entry["status"] == "ok" else "ERR"
            print(f"[{position}/{len(selected)}] page {page_number}: {state} "
                  f"{entry['block_count']} blocks")
        if entry["status"] != "ok" and stop_on_first_error:
            break

        clear_model_cache()

    failed_pages = [entry["page"] for entry in page_results if entry["status"] != "ok"]
    html = pages_to_html(page_results, input_file, model_path)
    CORE.write_text(html_path, html)

    category_counts: Dict[str, int] = {}
    for block in all_blocks:
        category_counts[block["category"]] = category_counts.get(block["category"], 0) + 1

    result: Dict[str, Any] = {
        "source": str(input_file),
        "model": model_path,
        "prompt_mode": prompt_mode,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "pdf_page_count": total_pages,
        "requested_pages": pages,
        "selected_pages": selected,
        "parsed_page_count": len(page_results),
        "status": "ok" if not failed_pages else "partial",
        "failed_pages": failed_pages,
        "block_count": len(all_blocks),
        "category_counts": category_counts,
        "categories": list(CORE.LAYOUT_CATEGORIES),
        "html_path": str(html_path),
        "page_results": CORE.json_value(page_results),
        "blocks": CORE.json_value(all_blocks),
        "success": not failed_pages,
    }

    if render:
        png_path = output_dir / f"{stem}.doc.png"
        CORE.render_html_to_png(html, png_path, user_css=PDF_HTML_STYLE)
        result["rendered_html_image"] = str(png_path)

    if render_per_page or overlay:
        for entry in page_results:
            if entry["status"] != "ok":
                continue
            if render_per_page:
                page_html = CORE.blocks_to_html(
                    entry["blocks"], Path(entry["image"]), model_path
                )
                target = artifacts_dir / f"page-{entry['page']:04d}.doc.png"
                CORE.render_html_to_png(page_html, target)
            if overlay:
                source_image = Path(entry["image"])
                if source_image.is_file():
                    CORE.draw_layout_overlay(
                        source_image,
                        entry["blocks"],
                        artifacts_dir / f"page-{entry['page']:04d}.layout.png",
                    )

    write_json(json_path, result)
    result["json_path"] = str(json_path)
    result["page_image_dir"] = str(artifacts_dir) if keep_page_images else None

    if not keep_page_images:
        CORE.cleanup_directory(workspace)
    return result


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir
        else args.input.parent
    )

    try:
        result = parse_pdf(
            input_file=args.input,
            output_dir=output_dir,
            model_name=args.model,
            prompt_mode=args.prompt_mode,
            max_tokens=args.max_tokens,
            pages=args.pages,
            dpi=args.dpi,
            max_image_side=args.max_image_side,
            render=args.render,
            render_per_page=args.render_per_page,
            overlay=args.overlay,
            keep_page_images=args.keep_page_images,
            stop_on_first_error=args.stop_on_first_error,
            quiet=args.quiet,
        )
    except (OSError, RuntimeError, ValueError) as error:
        print(f"PDF OCR failed: {error}", file=sys.stderr)
        return 1

    if args.quiet:
        print(result["json_path"])
        return 0 if result["success"] else 1

    print(
        f"Parsed {result['parsed_page_count']}/{len(result['selected_pages'])} pages, "
        f"{result['block_count']} blocks -> {result['json_path']}"
    )
    for category, count in sorted(result["category_counts"].items()):
        print(f"  {category}: {count}")
    for key in ("rendered_html_image", "html_path", "page_image_dir"):
        if result.get(key):
            print(f"  {key}: {result[key]}")
    if result["failed_pages"]:
        print(
            f"FAILED pages: {', '.join(str(page) for page in result['failed_pages'])}",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
