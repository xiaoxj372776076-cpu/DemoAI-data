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

    def test_transcribe_returns_empty_result_for_video_without_audio(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            input_path = Path(temp_dir) / "silent.mp4"
            input_path.touch()
            original_duration = transcribe_video.media_duration
            original_has_audio = transcribe_video.media_has_audio
            transcribe_video.media_duration = lambda _: 64.73
            transcribe_video.media_has_audio = lambda _: False
            try:
                result = transcribe_video.transcribe(
                    input_path,
                    transcribe_video.DEFAULT_MODEL,
                    Path(temp_dir) / "cache",
                )
            finally:
                transcribe_video.media_duration = original_duration
                transcribe_video.media_has_audio = original_has_audio

            self.assertFalse(result["audio_present"])
            self.assertEqual(result["duration_seconds"], 64.73)
            self.assertEqual(result["text"], "")
            self.assertEqual(result["segments"], [])

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
