# pdf_ocr operator

Parse a **whole PDF** into one HTML document with
[dots.ocr](https://github.com/rednote-hilab/dots.ocr) running on MLX.

This is the multi-page sibling of the [`doc_ocr`](../doc_ocr/README.md) operator.
`doc_ocr` looks at a single page image; `pdf_ocr` takes a PDF and drives the very
same pipeline over every page:

```
PDF ──▶ rasterise pages ──▶ dots.ocr per page ──▶ normalise blocks
     ──▶ merge in PDF order ──▶ HTML + JSON (+ rendered PNG, + layout overlays)
```

Model loading is the expensive part (tens of seconds and a few GB), so the
handle is built once and reused for every page.

## Install

```bash
python -m venv .venv312 && .venv312/bin/pip install -r operators/pdf_ocr/requirements.txt
```

Requirements: `mlx`, `mlx-vlm`, `pillow`, `pymupdf` (Apple Silicon via MLX).

Model weights (default `mlx-community/dots.ocr-6bit`):

```bash
export HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1
huggingface-cli download mlx-community/dots.ocr-6bit --local-dir models/dots.ocr-6bit
export DEMOAI_DOC_OCR_MODEL=$PWD/models/dots.ocr-6bit
```

## Usage

```bash
python operators/pdf_ocr/parse_pdf.py paper.pdf --render --overlay
```

Parses every page and writes artifacts next to the PDF (or `--output-dir`):

| File | Content |
|---|---|
| `<stem>.doc.html` | semantic HTML for the whole document, `<section data-page="N">` per page, in PDF order |
| `<stem>.doc.json` | structured result: per-page status + blocks and one flattened `blocks` list |
| `<stem>.doc.png` | the merged HTML rasterised into one tall PNG (page order preserved) |
| `<stem>-pages/page-000N.png` | rasterised page images (only with `--keep-page-images`) |
| `<stem>-pages/page-000N.doc.png` | per-page HTML rendering (`--render-per-page`) |
| `<stem>-pages/page-000N.layout.png` | detected boxes drawn on the page (`--overlay`) |

### Useful flags

| Flag | Meaning |
|---|---|
| `--pages all` (default) | pages to parse; also `"1-3,5,8-10"` or `"2"` |
| `--dpi 150` | rasterisation DPI |
| `--max-image-side N` | downscale each page so its longest edge ≤ N px. **The single biggest speed dial**: a dense page costs ~3 min at full 200 DPI and well under a minute at 1450 px |
| `--model` | MLX model path or HF repo (else `$DEMOAI_DOC_OCR_MODEL`) |
| `--prompt-mode layout_all` | `layout_all` (default, blocks + text), `ocr` (plain text), `layout_only` (boxes only) |
| `--max-tokens 8192` | generation budget **per page** |
| `--render` / `--render-per-page` / `--overlay` | image artifacts described above |
| `--keep-page-images` | keep the rasterised page PNGs instead of deleting them |
| `--stop-on-first-error` | abort at the first broken page instead of parsing the rest |

## Success semantics

**Every requested page must parse successfully for the run to succeed.**

* A page succeeds when the model answer contains layout JSON it can be normalised
  into blocks (an empty `[]` is fine — that means a blank page).
* A page fails when the model or the parser raises, or when the answer holds no
  layout JSON at all.
* Failures never kill the run mid-way: every other page is still parsed, the
  broken page is recorded (`status: "error"` + reason) and rendered visibly in
  the HTML as an error box. The process **exits 1** and the JSON carries
  `status: "partial"` plus `failed_pages` — so a 200-page document with one bad
  page still gives you 199 pages instead of nothing.
* `--stop-on-first-error` turns this into fail-fast for long documents.

## Output shape

```json
{
  "source": "/path/paper.pdf",
  "model": "/path/dots.ocr-6bit",
  "pdf_page_count": 9,
  "selected_pages": [1, 2, 3],
  "status": "ok",
  "failed_pages": [],
  "success": true,
  "block_count": 78,
  "category_counts": {"Text": 60, "Section-header": 8},
  "page_results": [
    {"page": 1, "status": "ok", "error": null, "block_count": 9,
     "blocks": [{"index": 1, "page": 1, "page_index": 1,
                 "category": "Title", "bbox": [..], "text_format": "markdown",
                 "text": ".."}]}
  ],
  "blocks": [ ... same blocks, flattened across all pages in reading order ... ]
}
```

Each block keeps both its **global** `index` (running across the whole document)
and its `page` / `page_index` (its place inside that one page), so downstream
code can either stream the document linearly or address content page-wise.

## Reference run

Apple Silicon M1, 16 GB, `dots.ocr-6bit`, AlexNet paper (9 pages, dense two-column
layouts with figures and result tables) at `--dpi 150 --max-image-side 1450`:
every page parsed, ~1 minute per page, peak RSS ~5 GB. Dropping to 1200 px is
roughly twice as fast; full 200 DPI pages are roughly three times slower.
