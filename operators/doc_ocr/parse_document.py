#!/usr/bin/env python3
"""Parse a document image into HTML with the dots.ocr vision-language model.

The operator runs RedNote's dots.ocr (an MLX build of rednote-hilab/dots.ocr),
asks the model for the full layout of the page, and turns the model answer into

  * a semantic HTML document (tables become real <table> markup, formulas keep
    their LaTeX source, headers/footers are marked up separately), and
  * a structured JSON document with one entry per layout block (bbox + category
    + text) so downstream code can filter or geolocate any piece of the page.

Optionally the generated HTML is rasterised into a PNG (``--render``) and the
layout boxes are drawn on top of the source image (``--overlay``).
"""

import argparse
import json
import os
import re
import shutil
import sys
import tempfile
from datetime import datetime, timezone
from html import escape
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


DEFAULT_MODEL = "mlx-community/dots.ocr-6bit"
DEFAULT_MODEL_ENV = "DEMOAI_DOC_OCR_MODEL"

LAYOUT_CATEGORIES = (
    "Caption",
    "Footnote",
    "Formula",
    "List-item",
    "Page-footer",
    "Page-header",
    "Picture",
    "Section-header",
    "Table",
    "Text",
    "Title",
)

# dots.ocr is fine-tuned for document layout, and the compact instruction below
# is what the MLX quantisation answers most reliably: a bare JSON list with one
# object per layout element. The verbose official `prompt_layout_all_en` tends to
# be echoed back instead of answered by the quantised builds, so we keep it short
# and state the same rendering rules (HTML tables, LaTeX formulas, Markdown text).
PROMPT_LAYOUT_ALL = (
    "Output the layout of this page as a JSON list. Each item: "
    '{"bbox": [x1, y1, x2, y2], "category": one of '
    "Caption, Footnote, Formula, List-item, Page-footer, Page-header, Picture, "
    "Section-header, Table, Text, Title, "
    '"text": the content of the element}. Rules: format Table text as HTML, '
    "Formula text as LaTeX, everything else as Markdown; keep the original "
    "language without translation; sort the elements by human reading order. "
    "Answer with the JSON list only:"
)

PROMPT_OCR = """Extract the text content from this image, preserving the reading order.
Output plain Markdown text only, without any additional explanation.
"""

PROMPT_LAYOUT_ONLY = """Please output the layout information from the PDF image, including each layout element's bbox and its category.

1. Bbox format: [x1, y1, x2, y2]
2. Layout Categories: The possible categories are ['Caption', 'Footnote', 'Formula', 'List-item', 'Page-footer', 'Page-header', 'Picture', 'Section-header', 'Table', 'Text', 'Title'].
3. The output should be sorted according to human reading order.

Output Format:
```json
[
    {
        "bbox": [x1, y1, x2, y2],
        "category": "Category"
    },
    ...
]
```
"""

PROMPTS = {
    "layout_all": PROMPT_LAYOUT_ALL,
    "ocr": PROMPT_OCR,
    "layout_only": PROMPT_LAYOUT_ONLY,
}

HTML_STYLE = """
body { font-family: "Helvetica Neue", Helvetica, Arial, sans-serif; color: #1f2329;
       line-height: 1.6; margin: 32px; }
h1 { font-size: 22px; margin: 0 0 12px; }
h2 { font-size: 18px; margin: 22px 0 8px; }
p { margin: 0 0 10px; }
table { border-collapse: collapse; table-layout: fixed; width: 100%; margin: 12px 0; font-size: 11px; }
th, td { border: 1px solid #c9ced6; padding: 2px 5px; text-align: left; overflow-wrap: anywhere; }
th { background: #f2f4f7; }
.formula { background: #f7f8fa; border-left: 3px solid #3370ff; padding: 8px 12px;
           margin: 10px 0; font-family: Menlo, monospace; white-space: pre-wrap; }
.picture, .page-header, .page-footer, .caption, .footnote { color: #646a73; }
.page-header, .page-footer { font-size: 12px; border-bottom: 1px solid #e3e6ea; }
ul { margin: 0 0 10px; padding-left: 22px; }
.block { margin-bottom: 8px; }
.block-anchor { color: #b7bcc4; font-size: 11px; }
"""


def input_path(value: str) -> Path:
    """Validate and return an existing image or PDF file."""
    path = Path(value).expanduser().resolve()
    if not path.is_file():
        raise argparse.ArgumentTypeError(f"input file does not exist: {path}")
    suffix = path.suffix.lower()
    if suffix not in {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff", ".pdf"}:
        raise argparse.ArgumentTypeError(f"unsupported input format: {suffix}")
    return path


def positive_integer(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be at least 1")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Parse a document image into HTML with dots.ocr."
    )
    parser.add_argument("input", type=input_path, help="document image or PDF file")
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
        choices=sorted(PROMPTS),
        default="layout_all",
        help="prompt preset sent to the model (default: layout_all)",
    )
    parser.add_argument(
        "--max-tokens",
        type=positive_integer,
        default=8192,
        help="maximum generated tokens (default: 8192)",
    )
    parser.add_argument(
        "--pdf-page",
        type=positive_integer,
        default=1,
        help="1-based page to rasterise when the input is a PDF (default: 1)",
    )
    parser.add_argument(
        "--dpi",
        type=positive_integer,
        default=200,
        help="rasterisation DPI for PDF pages (default: 200)",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="rasterise the generated HTML into <stem>.doc.png",
    )
    parser.add_argument(
        "--overlay",
        action="store_true",
        help="draw the detected layout boxes onto <stem>.layout.png",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="only print the written artifact paths",
    )
    return parser


def resolve_model(requested: Optional[str]) -> str:
    """Return the model to load, honouring the environment override."""
    if requested:
        return requested
    from_environment = os.environ.get(DEFAULT_MODEL_ENV)
    if from_environment:
        return from_environment
    return DEFAULT_MODEL


def default_output_dir(input_file: Path) -> Path:
    return input_file.parent


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
    return str(value)


def render_pdf_page(pdf_path: Path, page_number: int, dpi: int, target: Path) -> Path:
    """Rasterise one page of a PDF into a PNG file."""
    try:
        import pymupdf
    except ImportError as error:  # pragma: no cover - dependency hint
        raise RuntimeError(
            "PyMuPDF is required to read PDF input (pip install pymupdf)"
        ) from error

    with pymupdf.open(str(pdf_path)) as document:
        if page_number > document.page_count:
            raise RuntimeError(
                f"{pdf_path.name} has {document.page_count} pages, "
                f"page {page_number} was requested"
            )
        page = document[page_number - 1]
        pixmap = page.get_pixmap(dpi=dpi)
        target.parent.mkdir(parents=True, exist_ok=True)
        pixmap.save(str(target))
    return target


def prepare_image(input_file: Path, page_number: int, dpi: int, output_dir: Path) -> Tuple[Path, Optional[Path]]:
    """Return the image to feed the model plus the rasterised page (if any)."""
    if input_file.suffix.lower() != ".pdf":
        return input_file, None
    rendered = output_dir / f"{input_file.stem}-page{page_number}.png"
    return render_pdf_page(input_file, page_number, dpi, rendered), rendered


def extract_json_objects(text: str) -> List[Dict[str, Any]]:
    """Return every JSON object list found in the model answer."""
    candidates: List[str] = []

    fenced = re.findall(r"```(?:json)?\s*(.*?)```", text, flags=re.DOTALL)
    candidates.extend(fenced)
    candidates.append(text)

    for candidate in candidates:
        stripped = candidate.strip()
        if not stripped.startswith(("[", "{")):
            start = stripped.find("[")
            if start == -1:
                continue
            stripped = stripped[start:]
        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError:
            # Truncated generation: try to salvage a complete prefix of the list.
            parsed = salvage_partial_list(stripped)
            if parsed is None:
                continue
        if isinstance(parsed, dict) and isinstance(parsed.get("layout"), list):
            parsed = parsed["layout"]
        if isinstance(parsed, list):
            return [item for item in parsed if isinstance(item, dict)]
    return []


def salvage_partial_list(raw: str) -> Optional[List[Dict[str, Any]]]:
    """Recover complete objects from a JSON list that was cut off mid-stream."""
    decoder = json.JSONDecoder()
    if not raw.startswith("["):
        return None
    index = 1
    items: List[Dict[str, Any]] = []
    while index < len(raw):
        while index < len(raw) and raw[index] in " \n\r\t,":
            index += 1
        if index >= len(raw) or raw[index] != "{":
            break
        try:
            item, offset = decoder.raw_decode(raw, index)
        except json.JSONDecodeError:
            break
        if isinstance(item, dict):
            items.append(item)
        index = offset
    return items or None


def normalise_bbox(value: Any) -> Optional[List[float]]:
    """Coerce a model bbox into a list of four floats."""
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    try:
        return [round(float(number), 2) for number in value]
    except (TypeError, ValueError):
        return None


def build_blocks(raw_items: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Normalise raw model items into layout blocks with stable metadata."""
    blocks: List[Dict[str, Any]] = []
    for index, item in enumerate(raw_items, start=1):
        bbox = normalise_bbox(item.get("bbox"))
        category = str(item.get("category") or "Text").strip() or "Text"
        text = item.get("text")
        blocks.append(
            {
                "index": index,
                "category": category,
                "bbox": bbox,
                "text_format": "html" if category == "Table" else ("latex" if category == "Formula" else "markdown"),
                "text": "" if text is None else str(text),
            }
        )
    return blocks


def markdown_to_html(text: str) -> str:
    """Render the light Markdown subset dots.ocr emits for prose blocks."""
    lines = [line.rstrip() for line in text.splitlines()]
    html_lines: List[str] = []
    in_list = False
    paragraph: List[str] = []

    def flush_paragraph() -> None:
        nonlocal paragraph
        if paragraph:
            html_lines.append("<p>" + escape(" ".join(paragraph)).strip() + "</p>")
            paragraph = []

    for line in lines:
        stripped = line.strip()
        if not stripped:
            flush_paragraph()
            if in_list:
                html_lines.append("</ul>")
                in_list = False
            continue
        if stripped.startswith("- ") or stripped.startswith("* "):
            flush_paragraph()
            if not in_list:
                html_lines.append("<ul>")
                in_list = True
            html_lines.append("<li>" + escape(stripped[2:].strip()) + "</li>")
            continue
        if in_list:
            html_lines.append("</ul>")
            in_list = False
        paragraph.append(stripped)
    flush_paragraph()
    if in_list:
        html_lines.append("</ul>")
    return "\n".join(html_lines) or "<p></p>"


def block_to_html(block: Dict[str, Any]) -> str:
    """Render one layout block as an HTML fragment."""
    category = block["category"]
    text = block["text"]
    anchor = f'<span class="block-anchor">#{block["index"]} {escape(category)}</span>'

    if category == "Table":
        body = text if "<table" in text.lower() else f"<p>{escape(text)}</p>"
        return f'<div class="block block-table">{anchor}\n{body}</div>'
    if category == "Formula":
        return f'<div class="block formula">{anchor}<code>{escape(text)}</code></div>'
    if category == "Title":
        return f'<div class="block">{anchor}<h1>{escape(text)}</h1></div>'
    if category == "Section-header":
        return f'<div class="block">{anchor}<h2>{escape(text)}</h2></div>'
    if category == "Picture":
        return f'<div class="block picture">{anchor}<em>[figure]</em></div>'
    if category in {"Caption", "Footnote"}:
        return f'<div class="block {category.lower()}">{anchor}<p>{escape(text)}</p></div>'
    if category in {"Page-header", "Page-footer"}:
        return f'<div class="block {category.lower()}">{anchor}<p>{escape(text)}</p></div>'
    return f'<div class="block">{anchor}\n{markdown_to_html(text)}</div>'


def blocks_to_html(blocks: Sequence[Dict[str, Any]], source: Path, model: str) -> str:
    """Assemble the parsed blocks into a standalone HTML document."""
    body = "\n".join(block_to_html(block) for block in blocks)
    return (
        "<!DOCTYPE html>\n"
        '<html lang="en">\n<head>\n'
        '<meta charset="utf-8">\n'
        "<title>" + escape(source.name) + " · dots.ocr</title>\n"
        "<style>" + HTML_STYLE + "</style>\n"
        "</head>\n<body>\n"
        f"{body}\n"
        f'<hr>\n<p class="page-footer">parsed from <code>{escape(str(source))}</code> '
        f"with <code>{escape(model)}</code></p>\n"
        "</body>\n</html>\n"
    )


def write_text(path: Path, content: str) -> Path:
    """Atomically write a UTF-8 text file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)
    return path


def render_html_to_png(html: str, png_path: Path, width: int = 1100) -> Path:
    """Rasterise an HTML document into a single PNG using PyMuPDF's story API."""
    try:
        import pymupdf
        from PIL import Image
    except ImportError as error:  # pragma: no cover - dependency hint
        raise RuntimeError(
            "PyMuPDF and Pillow are required for --render (pip install pymupdf pillow)"
        ) from error

    png_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as workspace:
        pdf_path = Path(workspace) / "story.pdf"
        story = pymupdf.Story(html=html, user_css=HTML_STYLE)
        writer = pymupdf.DocumentWriter(str(pdf_path))
        # A page slightly wider than A4 keeps wide result tables from overflowing.
        mediabox = pymupdf.Rect(0, 0, 720, 960)
        more = True
        while more:
            device = writer.begin_page(mediabox)
            more, _ = story.place(mediabox)
            story.draw(device)
            writer.end_page()
        writer.close()

        with pymupdf.open(str(pdf_path)) as document:
            images = []
            for page in document:
                pixmap = page.get_pixmap(dpi=110)
                images.append(Image.frombytes("RGB", (pixmap.width, pixmap.height), pixmap.samples))
        if not images:
            raise RuntimeError("PyMuPDF produced no page for the parsed HTML")
        total_height = sum(image.height for image in images)
        canvas = Image.new("RGB", (max(image.width for image in images), total_height), "white")
        offset = 0
        for image in images:
            canvas.paste(image, (0, offset))
            offset += image.height
        canvas.save(str(png_path))
    return png_path


def draw_layout_overlay(image_path: Path, blocks: Sequence[Dict[str, Any]], target: Path) -> Path:
    """Draw the detected boxes on top of the source image."""
    try:
        from PIL import Image, ImageDraw
    except ImportError as error:  # pragma: no cover - dependency hint
        raise RuntimeError("Pillow is required for --overlay (pip install pillow)") from error

    target.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(str(image_path)) as source:
        canvas = source.convert("RGB")
    drawer = ImageDraw.Draw(canvas)
    for block in blocks:
        bbox = block.get("bbox")
        if not bbox:
            continue
        x1, y1, x2, y2 = bbox
        drawer.rectangle([x1, y1, x2, y2], outline=(51, 112, 255), width=3)
        drawer.text((x1 + 4, max(y1 - 16, 0)), f"{block['index']} {block['category']}", fill=(220, 53, 69))
    canvas.save(str(target))
    return target


def cleanup_directory(path: Path) -> None:
    """Best-effort cleanup: the host may block bulk deletes, never fail for it."""
    try:
        shutil.rmtree(path)
    except Exception:  # noqa: BLE001 - cleanup must never break the operator
        pass


def run_model(
    image: Path,
    model_name: str,
    prompt_mode: str,
    max_tokens: int,
) -> Tuple[str, str]:
    """Load the MLX model and return (raw answer, resolved model name)."""
    try:
        from mlx_vlm import generate, load
        from mlx_vlm.prompt_utils import apply_chat_template
    except ImportError as error:  # pragma: no cover - dependency hint
        raise RuntimeError(
            "mlx-vlm is required for document OCR (pip install mlx-vlm)"
        ) from error

    model_path = resolve_model(model_name)
    model, processor = load(model_path)
    prompt = apply_chat_template(
        processor, model.config, PROMPTS[prompt_mode], num_images=1
    )
    output = generate(
        model,
        processor,
        prompt,
        image=str(image),
        max_tokens=max_tokens,
        temperature=0.0,
        verbose=False,
    )
    return getattr(output, "text", str(output)), model_path


def parse_document(
    input_file: Path,
    output_dir: Path,
    model_name: Optional[str],
    prompt_mode: str,
    max_tokens: int,
    pdf_page: int,
    dpi: int,
    render: bool,
    overlay: bool,
) -> Dict[str, Any]:
    """Parse one document page and write HTML/JSON artifacts."""
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    image, rendered_page = prepare_image(input_file, pdf_page, dpi, output_dir)
    raw_answer, resolved_model = run_model(image, model_name, prompt_mode, max_tokens)
    blocks = build_blocks(extract_json_objects(raw_answer))

    stem = input_file.stem
    html_path = output_dir / f"{stem}.doc.html"
    json_path = output_dir / f"{stem}.doc.json"

    html = blocks_to_html(blocks, input_file, resolved_model)
    write_text(html_path, html)

    result: Dict[str, Any] = {
        "source": str(input_file),
        "image": str(image),
        "rendered_page_image": str(rendered_page) if rendered_page else None,
        "model": resolved_model,
        "prompt_mode": prompt_mode,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "categories": list(LAYOUT_CATEGORIES),
        "block_count": len(blocks),
        "category_counts": json_value(category_counts(blocks)),
        "html_path": str(html_path),
        "blocks": json_value(blocks),
        "raw_answer": raw_answer if prompt_mode == "ocr" else None,
    }

    if render:
        png_path = output_dir / f"{stem}.doc.png"
        render_html_to_png(html, png_path)
        result["rendered_html_image"] = str(png_path)
    if overlay:
        overlay_path = output_dir / f"{stem}.layout.png"
        draw_layout_overlay(image, blocks, overlay_path)
        result["layout_overlay_image"] = str(overlay_path)

    write_text(json_path, json.dumps(json_value(result), ensure_ascii=False, indent=2) + "\n")
    result["json_path"] = str(json_path)
    return result


def category_counts(blocks: Sequence[Dict[str, Any]]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for block in blocks:
        counts[block["category"]] = counts.get(block["category"], 0) + 1
    return counts


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir
        else default_output_dir(args.input)
    )

    try:
        result = parse_document(
            input_file=args.input,
            output_dir=output_dir,
            model_name=args.model,
            prompt_mode=args.prompt_mode,
            max_tokens=args.max_tokens,
            pdf_page=args.pdf_page,
            dpi=args.dpi,
            render=args.render,
            overlay=args.overlay,
        )
    except (OSError, RuntimeError, ValueError) as error:
        print(f"Document OCR failed: {error}", file=sys.stderr)
        return 1

    if args.quiet:
        print(result["json_path"])
        return 0

    print(f"Parsed {result['block_count']} layout blocks -> {result['json_path']}")
    print(f"  HTML: {result['html_path']}")
    for category, count in sorted((result["category_counts"] or {}).items()):
        print(f"  {category}: {count}")
    if result.get("rendered_html_image"):
        print(f"  rendered HTML: {result['rendered_html_image']}")
    if result.get("layout_overlay_image"):
        print(f"  layout overlay: {result['layout_overlay_image']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
