"""Focused regression checks for the offscreen camera readback patch."""
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prepare_action_step_project import patch_frame_grabber


class FrameGrabberPatchTests(unittest.TestCase):
    def test_renders_before_readback_and_is_idempotent(self):
        original = (
            "using UnityEngine;\n"
            "public static class FrameGrabber {\n"
            "    public static byte[] CaptureFrame(Camera camera) {\n"
            "        RenderTexture targetTexture = camera.targetTexture;\n"
            "        RenderTexture.active = targetTexture;\n"
            "        texture.ReadPixels(rect, 0, 0);\n"
            "    }\n"
            "}\n"
        )
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "FrameGrabber.cs"
            source.write_text(original, encoding="utf-8")

            patch_frame_grabber(source)
            first_result = source.read_text(encoding="utf-8")
            patch_frame_grabber(source)
            second_result = source.read_text(encoding="utf-8")

        render = "camera.Render(); // Force camera update before reading its RenderTexture"
        self.assertEqual(first_result, second_result)
        self.assertEqual(first_result.count(render), 1)
        self.assertLess(first_result.index(render), first_result.index("RenderTexture.active"))
        self.assertLess(first_result.index(render), first_result.index("ReadPixels"))
        self.assertIn("targetTexture == null && Application.isBatchMode", first_result)
        self.assertIn("targetTexture.Create();", first_result)
        self.assertLess(first_result.index("camera.targetTexture = targetTexture;"), first_result.index(render))


if __name__ == "__main__":
    unittest.main()
