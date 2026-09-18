import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest


MODULE_PATH = Path(__file__).parents[1] / "operators" / "doc_ocr" / "parse_document.py"
SPEC = importlib.util.spec_from_file_location("parse_document", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class DocOcrOperatorTests(unittest.TestCase):
    def test_extracts_fenced_json(self):
        answer = (
            "Here you go:\n```json\n"
            '[{"bbox": [1, 2, 3, 4], "category": "Title", "text": "Hello"}]\n'
            "```\n"
        )

        blocks = MODULE.build_blocks(MODULE.extract_json_objects(answer))

        self.assertEqual(len(blocks), 1)
        self.assertEqual(blocks[0]["category"], "Title")
        self.assertEqual(blocks[0]["text"], "Hello")
        self.assertEqual(blocks[0]["bbox"], [1.0, 2.0, 3.0, 4.0])
        self.assertEqual(blocks[0]["text_format"], "markdown")

    def test_salvages_truncated_json_list(self):
        answer = (
            '[{"bbox": [0, 0, 10, 10], "category": "Text", "text": "first"}, '
            '{"bbox": [0, 12, 10, 22], "category": "Text", "text": "seco'
        )

        blocks = MODULE.build_blocks(MODULE.extract_json_objects(answer))

        self.assertEqual(len(blocks), 1)
        self.assertEqual(blocks[0]["text"], "first")

    def test_ignores_answer_without_json(self):
        self.assertEqual(MODULE.extract_json_objects("I could not parse this page."), [])

    def test_table_block_keeps_html(self):
        table_html = "<table><tr><th>a</th><td>1</td></tr></table>"
        block = {
            "index": 3,
            "category": "Table",
            "bbox": [0, 0, 10, 10],
            "text_format": "html",
            "text": table_html,
        }

        fragment = MODULE.block_to_html(block)

        self.assertIn(table_html, fragment)
        self.assertIn("block-table", fragment)

    def test_formula_block_escapes_latex(self):
        block = {
            "index": 2,
            "category": "Formula",
            "bbox": [0, 0, 10, 10],
            "text_format": "latex",
            "text": "E = mc^2 <script>alert(1)</script>",
        }

        fragment = MODULE.block_to_html(block)

        self.assertIn("<code>", fragment)
        self.assertNotIn("<script>", fragment)

    def test_markdown_subset_becomes_paragraphs_and_lists(self):
        html = MODULE.markdown_to_html("Title line\n- first item\n- second item\n")

        self.assertIn("<p>Title line</p>", html)
        self.assertIn("<li>first item</li>", html)
        self.assertIn("<li>second item</li>", html)
        self.assertTrue(html.endswith("</ul>"))

    def test_writes_html_and_json_atomically(self):
        blocks = MODULE.build_blocks(
            [
                {"bbox": [0, 0, 20, 10], "category": "Title", "text": "Report"},
                {"bbox": [0, 12, 20, 40], "category": "Text", "text": "Body text."},
            ]
        )

        with tempfile.TemporaryDirectory() as workspace:
            html_path = Path(workspace) / "page.doc.html"
            json_path = Path(workspace) / "page.doc.json"
            html = MODULE.blocks_to_html(blocks, Path("page.png"), "test-model")
            MODULE.write_text(html_path, html)
            MODULE.write_text(json_path, json.dumps({"blocks": blocks}))

            self.assertIn("<h1>Report</h1>", html_path.read_text(encoding="utf-8"))
            self.assertIn("Body text.", html_path.read_text(encoding="utf-8"))
            self.assertEqual(json.loads(json_path.read_text(encoding="utf-8"))["blocks"][0]["category"], "Title")
            self.assertEqual(
                sorted(path.name for path in Path(workspace).iterdir()),
                ["page.doc.html", "page.doc.json"],
            )

    def test_category_counts(self):
        blocks = MODULE.build_blocks(
            [
                {"bbox": [0, 0, 1, 1], "category": "Table", "text": ""},
                {"bbox": [0, 2, 1, 3], "category": "Table", "text": ""},
                {"bbox": [0, 4, 1, 5], "category": "Text", "text": "x"},
            ]
        )

        self.assertEqual(MODULE.category_counts(blocks), {"Table": 2, "Text": 1})

    def test_resolve_model_prefers_explicit_argument(self):
        self.assertEqual(MODULE.resolve_model("some/path"), "some/path")


if __name__ == "__main__":
    unittest.main()
