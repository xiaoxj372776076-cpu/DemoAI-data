import argparse
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


MODULE_PATH = (
    Path(__file__).parents[1]
    / "operators"
    / "image_text_consistency"
    / "score_image_text.py"
)
SPEC = importlib.util.spec_from_file_location("score_image_text", MODULE_PATH)
score_image_text = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(score_image_text)


class InputValidationTest(unittest.TestCase):
    def test_rejects_missing_image(self):
        with self.assertRaises(argparse.ArgumentTypeError):
            score_image_text.input_image("/missing/image.jpg")

    def test_rejects_unsupported_format(self):
        with self.assertRaises(argparse.ArgumentTypeError):
            score_image_text.input_image("/tmp/image.gif")

    def test_accepts_supported_format(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = Path(temp_dir) / "photo.jpg"
            image_path.touch()
            self.assertEqual(
                score_image_text.input_image(str(image_path)), image_path.resolve()
            )

    def test_default_output_is_next_to_input(self):
        input_path = Path("/tmp/photo.jpeg")
        self.assertEqual(
            score_image_text.default_output_path(input_path),
            Path("/tmp/photo.consistency.json"),
        )

    def test_batch_size_must_be_positive(self):
        with self.assertRaises(argparse.ArgumentTypeError):
            score_image_text.positive_integer("0")

    def test_parser_requires_at_least_one_text(self):
        with self.assertRaises(SystemExit):
            score_image_text.build_parser().parse_args(["/tmp/photo.jpg"])

    def test_parser_accepts_multiple_texts(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = Path(temp_dir) / "photo.jpg"
            image_path.touch()
            args = score_image_text.build_parser().parse_args(
                [str(image_path), "一个孩子", "一只老虎"]
            )
        self.assertEqual(args.texts, ["一个孩子", "一只老虎"])


class ScoreComputationTest(unittest.TestCase):
    def test_scores_are_softmax_over_cosine_similarities(self):
        import torch

        image_features = torch.tensor([[1.0, 0.0, 0.0]])
        text_features = torch.tensor(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.7071068, 0.7071068, 0.0],
            ]
        )

        scores = score_image_text.score_texts(
            ["aligned", "orthogonal", "diagonal"],
            image_features,
            text_features,
        )

        self.assertEqual([item["text"] for item in scores], [
            "aligned",
            "orthogonal",
            "diagonal",
        ])
        self.assertAlmostEqual(scores[0]["cosine_similarity"], 1.0, places=5)
        self.assertAlmostEqual(scores[1]["cosine_similarity"], 0.0, places=5)
        self.assertAlmostEqual(
            scores[2]["cosine_similarity"], 0.7071068, places=5
        )
        total = sum(item["consistency_score"] for item in scores)
        self.assertAlmostEqual(total, 1.0, places=5)
        self.assertEqual(
            max(scores, key=lambda item: item["consistency_score"])["text"],
            "aligned",
        )


class WriteResultTest(unittest.TestCase):
    def test_write_result_preserves_unicode(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "result.json"
            result = {"scores": [{"text": "这是一个儿童", "consistency_score": 0.9}]}

            written_path = score_image_text.write_result(result, output_path)

            self.assertEqual(written_path, output_path.resolve())
            self.assertEqual(
                json.loads(output_path.read_text(encoding="utf-8")), result
            )
            self.assertFalse((Path(temp_dir) / "result.json.tmp").exists())


if __name__ == "__main__":
    unittest.main()
