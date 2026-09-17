import argparse
import importlib.util
from pathlib import Path
import tempfile
import unittest


MODULE_PATH = (
    Path(__file__).parents[1]
    / "crawlers"
    / "huggingface"
    / "download_dataset.py"
)
SPEC = importlib.util.spec_from_file_location("download_dataset", MODULE_PATH)
download_dataset = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(download_dataset)


class DatasetIdTest(unittest.TestCase):
    def test_accepts_namespaced_dataset_id(self):
        self.assertEqual(
            download_dataset.dataset_id("stanfordnlp/imdb"),
            "stanfordnlp/imdb",
        )

    def test_accepts_single_dataset_name(self):
        self.assertEqual(download_dataset.dataset_id("imdb"), "imdb")

    def test_extracts_id_from_dataset_url(self):
        self.assertEqual(
            download_dataset.dataset_id(
                "https://huggingface.co/datasets/stanfordnlp/imdb"
            ),
            "stanfordnlp/imdb",
        )

    def test_extracts_id_from_dataset_tree_url(self):
        self.assertEqual(
            download_dataset.dataset_id(
                "https://huggingface.co/datasets/stanfordnlp/imdb/tree/main"
            ),
            "stanfordnlp/imdb",
        )

    def test_rejects_non_huggingface_url(self):
        with self.assertRaises(argparse.ArgumentTypeError):
            download_dataset.dataset_id("https://example.com/datasets/test/data")

    def test_rejects_model_url(self):
        with self.assertRaises(argparse.ArgumentTypeError):
            download_dataset.dataset_id("https://huggingface.co/openai/model")

    def test_local_directory_replaces_namespace_separator(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            result = download_dataset.local_dataset_dir(
                "stanfordnlp/imdb", Path(temp_dir)
            )
            self.assertEqual(result.name, "stanfordnlp--imdb")

    def test_workers_must_be_positive(self):
        with self.assertRaises(argparse.ArgumentTypeError):
            download_dataset.positive_integer("0")


if __name__ == "__main__":
    unittest.main()
