import argparse
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


MODULE_PATH = (
    Path(__file__).parents[1] / "operators" / "asr" / "transcribe_video.py"
)
SPEC = importlib.util.spec_from_file_location("transcribe_video", MODULE_PATH)
transcribe_video = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(transcribe_video)


class AsrOperatorTest(unittest.TestCase):
    def test_rejects_missing_input_file(self):
        with self.assertRaises(argparse.ArgumentTypeError):
            transcribe_video.input_media("/missing/video.mp4")

    def test_default_output_is_next_to_input(self):
        input_path = Path("/tmp/example.video.mp4")
        self.assertEqual(
            transcribe_video.default_output_path(input_path),
            Path("/tmp/example.video.asr.json"),
        )

    def test_json_value_normalizes_nested_tuples(self):
        self.assertEqual(
            transcribe_video.json_value({"timestamps": (0.0, 1.25)}),
            {"timestamps": [0.0, 1.25]},
        )

    def test_write_result_preserves_unicode(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "result.json"
            result = {"text": "你好，Whisper", "segments": []}
            written_path = transcribe_video.write_result(result, output_path)

            self.assertEqual(written_path, output_path.resolve())
            self.assertEqual(
                json.loads(output_path.read_text(encoding="utf-8")),
                result,
            )
            self.assertFalse((Path(temp_dir) / "result.json.tmp").exists())


if __name__ == "__main__":
    unittest.main()
