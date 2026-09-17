import argparse
import importlib.util
from pathlib import Path
import unittest


MODULE_PATH = (
    Path(__file__).parents[1] / "crawlers" / "youtube" / "download_video.py"
)
SPEC = importlib.util.spec_from_file_location("download_video", MODULE_PATH)
download_video = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(download_video)


class VideoIdTest(unittest.TestCase):
    def test_accepts_valid_youtube_id(self):
        self.assertEqual(download_video.video_id("-UkD8BG_9nU"), "-UkD8BG_9nU")

    def test_rejects_full_url(self):
        with self.assertRaises(argparse.ArgumentTypeError):
            download_video.video_id("https://youtu.be/-UkD8BG_9nU")

    def test_rejects_invalid_characters(self):
        with self.assertRaises(argparse.ArgumentTypeError):
            download_video.video_id("invalid$id!")

    def test_parser_accepts_id_starting_with_dash_after_separator(self):
        args = download_video.build_parser().parse_args(["--", "-UkD8BG_9nU"])
        self.assertEqual(args.video_id, "-UkD8BG_9nU")

    def test_parser_accepts_supported_cookie_browser(self):
        args = download_video.build_parser().parse_args(
            ["--cookies-from-browser", "chrome", "--", "-UkD8BG_9nU"]
        )
        self.assertEqual(args.cookies_from_browser, "chrome")


if __name__ == "__main__":
    unittest.main()
