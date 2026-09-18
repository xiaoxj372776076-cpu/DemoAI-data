import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).parents[1] / "operators" / "pdf_ocr" / "parse_pdf.py"
SPEC = importlib.util.spec_from_file_location("parse_pdf", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


try:  # pragma: no cover - exercised through the tests below
    import pymupdf
except ImportError:  # pragma: no cover - dependency hint
    pymupdf = None


def answer_for_page(page_number: int) -> str:
    return json.dumps(
        [
            {
                "bbox": [10, 10, 200, 40],
                "category": "Title",
                "text": f"Heading of page {page_number}",
            },
            {
                "bbox": [10, 50, 200, 90],
                "category": "Text",
                "text": f"Body copy on page {page_number}.",
            },
        ]
    )


class FakeModel:
    """Stands in for the MLX dots.ocr handle: one call per page image."""

    def __init__(self, fails_on=()) -> None:
        self.model_path = "fake/dots.ocr"
        self.fails_on = set(fails_on)
        self.calls = []

    def generate(self, image, prompt_mode="layout_all", max_tokens=8192):
        page_number = int(Path(image).stem.split("-")[1])
        self.calls.append(page_number)
        if page_number in self.fails_on:
            raise RuntimeError(f"model blew up on page {page_number}")
        return answer_for_page(page_number)


def make_pdf(path: Path, pages: int = 3) -> Path:
    if pymupdf is None:  # pragma: no cover - dependency hint
        raise unittest.SkipTest("pymupdf is required for PDF fixtures")
    document = pymupdf.open()
    for number in range(1, pages + 1):
        page = document.new_page()
        page.insert_text((72, 100), f"Document page {number}", fontsize=18)
    if hasattr(document, "save"):
        document.save(str(path))
    document.close()
    return path


class PdfOcrOperatorTests(unittest.TestCase):
    def test_page_spec_defaults_to_every_page(self):
        self.assertEqual(MODULE.parse_page_spec("all", 4), [1, 2, 3, 4])

    def test_page_spec_supports_ranges_and_deduplicates(self):
        self.assertEqual(MODULE.parse_page_spec("1,3", 5), [1, 3])
        self.assertEqual(MODULE.parse_page_spec("2-4", 5), [2, 3, 4])
        self.assertEqual(MODULE.parse_page_spec("1-2,2,5", 5), [1, 2, 5])

    def test_page_spec_rejects_garbage_and_out_of_range_pages(self):
        with self.assertRaises(ValueError):
            MODULE.parse_page_spec("abc", 5)
        with self.assertRaises(ValueError):
            MODULE.parse_page_spec("4-2", 5)
        with self.assertRaises(ValueError):
            MODULE.parse_page_spec("9", 5)

    def test_annotate_blocks_numbers_across_pages(self):
        first = MODULE.CORE.build_blocks(
            [{"bbox": [0, 0, 1, 1], "category": "Text", "text": "a"}]
        )
        second = MODULE.CORE.build_blocks(
            [{"bbox": [0, 0, 1, 1], "category": "Text", "text": "b"}]
        )

        first_blocks = MODULE.annotate_blocks(first, 1, offset=0)
        second_blocks = MODULE.annotate_blocks(second, 2, offset=len(first_blocks))

        self.assertEqual([block["index"] for block in first_blocks], [1])
        self.assertEqual([block["index"] for block in second_blocks], [2])
        self.assertEqual([block["page"] for block in second_blocks], [2])
        self.assertEqual([block["page_index"] for block in second_blocks], [1])

    def test_pages_to_html_keeps_pdf_order_and_marks_pages(self):
        pages = [
            {"page": 1, "status": "ok", "error": None, "blocks": [], "block_count": 0},
            {"page": 2, "status": "ok", "error": None, "blocks": [], "block_count": 0},
        ]

        html = MODULE.pages_to_html(pages, Path("doc.pdf"), "fake/model")

        self.assertLess(html.index("PAGE 1"), html.index("PAGE 2"))
        self.assertIn('data-page="1"', html)
        self.assertIn('data-page="2"', html)

    def test_pages_to_html_reports_broken_pages_inline(self):
        pages = [
            {
                "page": 3,
                "status": "error",
                "error": "RuntimeError: model blew up on page 3",
                "blocks": [],
                "block_count": 0,
            }
        ]

        html = MODULE.pages_to_html(pages, Path("doc.pdf"), "fake/model")

        self.assertIn("page-error", html)
        self.assertIn("blew up on page 3", html)

    @unittest.skipIf(pymupdf is None, "pymupdf is required to build PDF fixtures")
    def test_rasterises_every_requested_page(self):
        with tempfile.TemporaryDirectory() as workspace:
            pdf = make_pdf(Path(workspace) / "sample.pdf", pages=3)
            target = Path(workspace) / "images"

            images = MODULE.rasterise_pages(pdf, [1, 3], 72, target, max_image_side=0)

            self.assertEqual([path.name for path in images], ["page-0001.png", "page-0003.png"])
            self.assertTrue(all(path.is_file() for path in images))

    @unittest.skipIf(pymupdf is None, "pymupdf is required to build PDF fixtures")
    def test_pdf_page_count_matches_fixture(self):
        with tempfile.TemporaryDirectory() as workspace:
            pdf = make_pdf(Path(workspace) / "sample.pdf", pages=2)
            self.assertEqual(MODULE.pdf_page_count(pdf), 2)

    @unittest.skipIf(pymupdf is None, "pymupdf is required to build PDF fixtures")
    def test_whole_pdf_is_parsed_in_order(self):
        with tempfile.TemporaryDirectory() as workspace:
            pdf = make_pdf(Path(workspace) / "sample.pdf", pages=3)
            output = Path(workspace) / "out"
            model = FakeModel()

            result = MODULE.parse_pdf(
                pdf, output, model_factory=lambda: model, quiet=True
            )

            self.assertTrue(result["success"])
            self.assertEqual(result["status"], "ok")
            self.assertEqual(result["failed_pages"], [])
            self.assertEqual(result["selected_pages"], [1, 2, 3])
            self.assertEqual(result["block_count"], 6)
            self.assertEqual([block["page"] for block in result["blocks"]], [1, 1, 2, 2, 3, 3])
            self.assertEqual([block["index"] for block in result["blocks"]], [1, 2, 3, 4, 5, 6])
            self.assertEqual(result["blocks"][2]["text"], "Heading of page 2")

            html = Path(result["html_path"]).read_text(encoding="utf-8")
            self.assertLess(html.index("Heading of page 1"), html.index("Heading of page 3"))

            payload = json.loads(Path(result["json_path"]).read_text(encoding="utf-8"))
            self.assertEqual(payload["pdf_page_count"], 3)
            self.assertEqual([entry["page"] for entry in payload["page_results"]], [1, 2, 3])
            self.assertEqual(payload["page_results"][1]["block_count"], 2)

    @unittest.skipIf(pymupdf is None, "pymupdf is required to build PDF fixtures")
    def test_one_bad_page_fails_the_whole_document(self):
        with tempfile.TemporaryDirectory() as workspace:
            pdf = make_pdf(Path(workspace) / "sample.pdf", pages=3)
            output = Path(workspace) / "out"
            model = FakeModel(fails_on={2})

            result = MODULE.parse_pdf(
                pdf, output, model_factory=lambda: model, quiet=True
            )

            self.assertFalse(result["success"])
            self.assertEqual(result["status"], "partial")
            self.assertEqual(result["failed_pages"], [2])
            self.assertIn("blew up on page 2", result["page_results"][1]["error"])
            # The healthy pages around the failure are still merged, in order.
            self.assertEqual([block["page"] for block in result["blocks"]], [1, 1, 3, 3])
            self.assertIn("page-error", Path(result["html_path"]).read_text(encoding="utf-8"))

    @unittest.skipIf(pymupdf is None, "pymupdf is required to build PDF fixtures")
    def test_stop_on_first_error_halts_the_run(self):
        with tempfile.TemporaryDirectory() as workspace:
            pdf = make_pdf(Path(workspace) / "sample.pdf", pages=4)
            output = Path(workspace) / "out"
            model = FakeModel(fails_on={2})

            result = MODULE.parse_pdf(
                pdf,
                output,
                model_factory=lambda: model,
                stop_on_first_error=True,
                quiet=True,
            )

            self.assertEqual(model.calls, [1, 2])
            self.assertEqual(result["parsed_page_count"], 2)

    @unittest.skipIf(pymupdf is None, "pymupdf is required to build PDF fixtures")
    def test_only_selected_pages_are_parsed(self):
        with tempfile.TemporaryDirectory() as workspace:
            pdf = make_pdf(Path(workspace) / "sample.pdf", pages=5)
            output = Path(workspace) / "out"
            model = FakeModel()

            result = MODULE.parse_pdf(
                pdf, output, pages="2,4-5", model_factory=lambda: model, quiet=True
            )

            self.assertEqual(result["selected_pages"], [2, 4, 5])
            self.assertEqual(model.calls, [2, 4, 5])
            self.assertEqual([entry["page"] for entry in result["page_results"]], [2, 4, 5])


if __name__ == "__main__":
    unittest.main()
