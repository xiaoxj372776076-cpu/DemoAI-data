import importlib.util
from pathlib import Path
import struct
import sys
import tempfile
import unittest


MODULE_PATH = (
    Path(__file__).parents[1]
    / "skill"
    / "blender-scene"
    / "scripts"
    / "run_scene.py"
)
SPEC = importlib.util.spec_from_file_location("run_blender_scene", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def minimal_png(width: int, height: int) -> bytes:
    return b"\x89PNG\r\n\x1a\n" + b"\x00" * 8 + struct.pack(">II", width, height)


class BlenderSceneRunnerTests(unittest.TestCase):
    def test_reads_png_dimensions(self):
        with tempfile.TemporaryDirectory() as directory:
            image = Path(directory) / "render.png"
            image.write_bytes(minimal_png(960, 720))

            self.assertEqual(MODULE.png_dimensions(image), (960, 720))

    def test_validates_complete_artifact_bundle(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            (output / "prompt.txt").write_text("Create a table\n", encoding="utf-8")
            (output / "scene.py").write_text("import bpy\n", encoding="utf-8")
            (output / "scene.blend").write_bytes(b"BLENDER")
            (output / "render.png").write_bytes(minimal_png(640, 480))

            result = MODULE.validate_artifacts(output)

            self.assertEqual(result["renders"][0]["width"], 640)
            self.assertEqual(result["renders"][0]["height"], 480)
            self.assertEqual(len(result["blend_files"]), 1)

    def test_rejects_incomplete_bundle(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            (output / "prompt.txt").write_text("Create a table\n", encoding="utf-8")

            with self.assertRaisesRegex(RuntimeError, "incomplete Blender output"):
                MODULE.validate_artifacts(output)


if __name__ == "__main__":
    unittest.main()
